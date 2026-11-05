"""
Tone reading — Christman AI

Reads tone cues from a person's words (a word list, negation-aware), or from
a tone profile the caller already holds. It reports what it read. It sets no
delivery: pacing, warmth, structure and how a reply is laid out are the
being's decisions (item 6 below).

WHAT CHANGED AND WHY
--------------------

1. MISSING TONE FIELDS SILENTLY BECAME ZERO.

       if getattr(tone_profile, "distress_score", 0.0) >= 0.7:
           self.emotion_state = "serious_support"

   Five `getattr(..., 0.0)` calls. If `ToneEngine` returned a profile without
   `distress_score` — a version change, a different engine, a partial object —
   the default was 0.0 and the distress branch could never fire. No error, no
   log. The support path just quietly stopped existing.

   Missing fields are now detected and reported. A profile without
   `distress_score` produces `state = "unknown"`, not "not distressed".

2. NEGATION-BLIND SUBSTRING MATCHING IN THE FALLBACK.

       if any(word in text_lower for word in ["good","great","awesome",...]):
           label = "positive"

   "not good" contains "good". The fallback analyzer read negated distress as
   positive affect — the same defect `emotion_quantifier` had, and it is fixed
   the same way: word boundaries plus negation scope, shared from
   `structural_affect` rather than reimplemented.

3. THE FALLBACK WAS INDISTINGUISHABLE FROM THE REAL ENGINE.

   `analyze_user_input` used `ToneEngine` when importable and a keyword
   list when not, returning the same shape either way. A caller could not
   tell which had run. `source` is now on every result.

4. CANNED EMPATHY WAS INJECTED INTO RESPONSES.

       intro_parts.append("I love the energy you're bringing.")

   `format_response` prepended fixed sentences to the reply. In an AAC surface
   the reply is spoken as the being's voice, so this put words there that no
   model produced. Phrasing then came from a caller-supplied table, empty by
   default. Since 2026-11-04 (item 6) the table and `format_response` are
   gone: nothing here touches the being's reply.

5. `reset()` called `self.__init__()`, which re-runs construction on a live
   object and silently drops any constructor arguments a subclass added.

6. WHAT CHANGED AND WHY (2026-11-04): IT READS; THE BEING DECIDES.
   Everett's ruling, in his words: "we can have this knowledge in the sound
   package, but it can't be making those decisions. It's up to the being to
   make those decisions, not something that just creates a voice."

   Before this, ToneManager decided the being's delivery. Its docstring said
   "It decides pacing, warmth, structure and validation." A distress score
   of 0.7 or more set the state "serious_support", a speech rate of 145, a
   "gentle" warmth and a "guided" structure; other cues set speech rates
   (160, 180, a 0.85 slow-down), warmth and validation levels; the being's
   own response mode was turned into a state name ("supportive",
   "corrective"); the keyword reader named its result for a posture
   ("compassionate", "supportive"); and `format_response` re-laid the
   being's reply out as bullet points and could put configured phrases in
   front of it. All of that is removed: `DEFAULT_PROFILE`, the delivery
   profile, `get_speech_controls`, `format_response`, the intro phrases and
   `extract_speech_controls`. `apply_profile` no longer takes the being's
   mode.

   What stays is the reading of the words: the cues, the markers matched
   (word-boundary and negation-aware), the profile's scores exactly as read
   (`scores`, None where absent) and the fields it was missing.
   `emotion_state` now names what was read: the strongest group of words
   found ("distress", "hearing_support", "confusion", "positive_affect", in
   that order of weight, the same weights as before), or, from a profile,
   "high_distress" / "distress_below_0.7" / "unknown".

   Checked 2026-11-04: nothing in this repository calls ToneManager.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# One negation set for the whole package. 2026-10-29: this read a top-level
# `structural_affect` module that exists nowhere, so the ImportError branch
# always ran and a private copy of NEGATORS (missing "nobody", "neither",
# "nor", "hardly", "barely", "ain't" in places) silently diverged from the
# reader's own.
# Everett, 2026-10-29 (ruling AE): the shared reading words, with every
# correction ruled today, are the starting point; a client's learned words
# extend or override them through ReadingWords.for_client.
from ..audio.structural_affect import SHARED_READING_WORDS, ReadingWords


_WORD = re.compile(r"[a-z']+")
CLAUSE_BOUNDARY = "\x00"
_CLAUSE_PUNCT = re.compile(r"[,.;:!?]+")
NEGATION_WINDOW = 4

#: Tone-profile fields this manager reads. A profile missing any of them
#: cannot be interpreted, and that is reported rather than defaulted to zero.
REQUIRED_PROFILE_FIELDS: Tuple[str, ...] = (
    "needs_validation", "wants_action", "emotional_intensity",
    "humor_score", "sarcasm_score", "distress_score",
)

#: The cue thresholds on a caller's tone profile. Carried from before
#: 2026-11-04; no reason was ever written beside them, so each is listed for
#: Everett to give its reason or change it. They label cues; nothing acts on
#: a cue here.
CUE_THRESHOLDS: Dict[str, float] = {
    "needs_validation": 0.4,      # cue "validation_needed"
    "wants_action": 0.5,          # cue "action_needed"
    "emotional_intensity": 0.5,   # cue "heightened_intensity"
    "humor_score": 0.2,           # cue "humor_present"
    "sarcasm_score": 0.3,         # cue "sarcasm_present"
    "distress_score": 0.7,        # cue "high_distress"
}


def _tokens(text: str) -> List[str]:
    """Words with clause boundaries preserved, so negation stops at a comma."""
    out: List[str] = []
    for chunk in _CLAUSE_PUNCT.split((text or "").lower()):
        words = _WORD.findall(chunk)
        if not words:
            continue
        if out:
            out.append(CLAUSE_BOUNDARY)
        out.extend(words)
    return out


def _negated_at(tokens: Sequence[str], index: int,
                words: ReadingWords = SHARED_READING_WORDS) -> bool:
    for back in range(1, NEGATION_WINDOW + 1):
        j = index - back
        if j < 0:
            return False
        tok = tokens[j]
        if tok in words.negators:
            return True
        if tok == CLAUSE_BOUNDARY or tok in words.scope_breakers:
            return False
    return False


def _live_matches(tokens: Sequence[str], markers: Set[str],
                  words: ReadingWords = SHARED_READING_WORDS) -> List[str]:
    """Markers present and NOT inside a negation. Word-boundary matched."""
    found: List[str] = []
    joined = " ".join(tokens)
    for marker in markers:
        parts = marker.split()
        if len(parts) == 1:
            for i, tok in enumerate(tokens):
                if tok == marker and not _negated_at(tokens, i, words):
                    found.append(marker)
                    break
        elif f" {marker} " in f" {joined} ":
            idx = next(
                (i for i in range(len(tokens) - len(parts) + 1)
                 if list(tokens[i:i + len(parts)]) == parts),
                None,
            )
            if idx is not None and not _negated_at(tokens, idx, words):
                found.append(marker)
    return found


@dataclass
class ToneReading:
    """One interpretation, with its provenance attached."""

    emotion_state: str = "unknown"
    cues: List[str] = field(default_factory=list)
    source: str = "none"                 # caller_profile | keyword_fallback | none
    missing_fields: List[str] = field(default_factory=list)
    matched_markers: Dict[str, List[str]] = field(default_factory=dict)
    #: A caller's tone profile, exactly as read: None where a field was
    #: absent or not a number. Empty for the keyword reading.
    scores: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion_state": self.emotion_state,
            "cues": list(self.cues),
            "source": self.source,
            "is_fallback": self.source == "keyword_fallback",
            "missing_fields": list(self.missing_fields),
            "matched_markers": dict(self.matched_markers),
            "scores": dict(self.scores),
        }


class ToneManager:
    """
    Reads tone cues from words, or from a tone profile the caller holds.

    Does not replace ToneScore or ToneEngine. It reports what it read; it
    sets no pacing, warmth, structure or validation, and it does not touch
    the being's reply. Those are the being's decisions (2026-11-04).
    """

    def __init__(self, reading_words: Optional[ReadingWords] = None) -> None:
        # Ruling AE: the shared reading words, or this client's learned layer
        # over them (ReadingWords.for_client).
        self.words = reading_words or SHARED_READING_WORDS
        self.reading = ToneReading()

    # -- Interpretation -------------------------------------------------------

    def analyze_user_input(self, text: str) -> ToneReading:
        """
        Read the words for tone cues.

        Returns a ToneReading rather than a bare string, so the caller can see
        which analyzer ran and whether any expected field was missing.

        2026-10-29: this tried a top-level `tone_engine` module first. No such
        module exists in this package or anywhere it ships, so the keyword
        reader always ran. That is now said plainly instead of tried. A caller
        holding a real tone profile passes it to `apply_profile`.
        """
        reading = self._from_keywords(text)
        self.reading = reading
        return reading

    def apply_profile(self, profile: Any) -> ToneReading:
        """
        Read a tone profile the caller already has.

        Missing fields are reported, never read as 0.0. The being's response
        mode is not taken: how to respond is the being's (2026-11-04).
        """
        missing = [f for f in REQUIRED_PROFILE_FIELDS if not hasattr(profile, f)]
        if missing:
            # The predecessor defaulted these to 0.0, so a profile without
            # distress_score could never reach the distress reading.
            logger.error(
                "ToneProfile is missing %s. These are NOT treated as zero.",
                missing,
            )
        reading = self._apply_profile(profile, missing)
        self.reading = reading
        return reading

    def _apply_profile(self, profile: Any, missing: List[str]) -> ToneReading:
        def score(field_name: str) -> Optional[float]:
            """None when absent. Never 0.0-by-default."""
            if not hasattr(profile, field_name):
                return None
            try:
                return float(getattr(profile, field_name))
            except (TypeError, ValueError):
                return None

        scores = {name: score(name) for name in REQUIRED_PROFILE_FIELDS}
        cues: List[str] = []
        for name, cue in (("needs_validation", "validation_needed"),
                          ("wants_action", "action_needed"),
                          ("emotional_intensity", "heightened_intensity"),
                          ("humor_score", "humor_present"),
                          ("sarcasm_score", "sarcasm_present")):
            value = scores[name]
            if value is not None and value >= CUE_THRESHOLDS[name]:
                cues.append(cue)

        distress = scores["distress_score"]
        if distress is None:
            # Cannot rule distress in OR out. Say so.
            state = "unknown"
            cues.append("distress_unmeasured")
        elif distress >= CUE_THRESHOLDS["distress_score"]:
            state = "high_distress"
            cues.append("high_distress")
        else:
            state = "distress_below_0.7"

        return ToneReading(
            emotion_state=state, cues=cues, source="caller_profile",
            missing_fields=missing, scores=scores,
        )

    def _from_keywords(self, text: str) -> ToneReading:
        """
        Keyword fallback. Word-boundary and negation-aware.

        The predecessor used `word in text_lower`, so "not good" matched "good"
        and read as positive affect.
        """
        tokens = _tokens(text)
        if not tokens:
            return ToneReading(source="keyword_fallback")

        # Each group: its words, and its weight when more than one is found
        # (the same weights as before 2026-11-04). Until then each group also
        # carried a posture label ("compassionate", "supportive") that named
        # how to respond; the reading is now named for what was read.
        groups: Dict[str, Tuple[Set[str], int]] = {
            "distress": ({"sad", "upset", "hurt", "pain", "difficult",
                          "struggling"}, 3),
            "hearing_support": ({"can't hear", "cannot hear", "hard to hear",
                                 "slow down"}, 2),
            "confusion": ({"confused", "lost", "not sure",
                           "don't understand"}, 2),
            "positive_affect": ({"good", "great", "awesome", "excited",
                                 "happy", "love"}, 1),
        }

        cues: List[str] = []
        matched: Dict[str, List[str]] = {}
        best: Optional[Tuple[int, str]] = None

        for name, (markers, weight) in groups.items():
            hits = _live_matches(tokens, markers, self.words)
            if hits:
                matched[name] = hits
                cues.append(name)
                if best is None or weight > best[0]:
                    best = (weight, name)

        return ToneReading(
            emotion_state=best[1] if best else "unknown",
            cues=cues, source="keyword_fallback", matched_markers=matched,
        )

    # -- The reading ----------------------------------------------------------

    def get_emotional_context(self) -> str:
        return self.reading.emotion_state

    def reset(self) -> None:
        """Clear the last reading without re-running the constructor."""
        self.reading = ToneReading()


__all__ = [
    "ToneManager", "ToneReading", "REQUIRED_PROFILE_FIELDS", "CUE_THRESHOLDS",
]

# ==============================================================================
# Patent Pending
# © 2026 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# Nothing Vital Lives Below Root.
# ==============================================================================
