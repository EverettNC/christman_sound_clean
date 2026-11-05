"""
© The Christman AI Project | Luma Cognify AI. All rights reserved. Patent pending.

— Presence Guide.

Reads text for states that may need *presence* rather than problem-solving,
and holds guidance the being may look up. It chooses nothing for the being:
the reading and the guidance are knowledge; the being decides (2026-11-04).

NOT a clinical instrument. A phrase-list guard against toxic positivity,
premature problem-solving, and minimization in the generated voice surface.
When in doubt: stay steady, witness, do not rush.

WHAT CHANGED AND WHY
--------------------

1. "lost" WAS IN TWO MARKER SETS, AND ORDER DECIDED.

       GRIEVING_MARKERS = (..., "lost", ...)
       CONFUSED_MARKERS = ("confused", "lost", ...)

   Grieving was checked first, so `CONFUSED` could never win on that word.
   Measured on the original:

       'I lost my keys'    -> grieving
       'I lost my job'     -> grieving
       'I feel lost'       -> grieving
       'I lost my mother'  -> grieving

   Only the last one is grief. The word is now disambiguated by what follows
   it — "feel lost" is confusion — and the remaining ambiguity returns `None`
   rather than a guess. (Ruling AA, 2026-10-29: "lost my <person>" was read
   from a fixed list of people; the list is gone. Since 2026-11-04 the
   reading says who is unclear — see 6 — and the being decides what to do.)

2. DETECTION WAS ORDER-DEPENDENT EVERYWHERE, NOT JUST "lost".

   Eight `if any(...): return` blocks in a fixed sequence. "I'm overwhelmed and
   terrified" returned OVERWHELMED because it was tested before AFRAID. States
   are now scored, the strongest wins, and every match is returned so the
   choice is inspectable.

3. NEGATION WAS INVISIBLE.

       'I am not scared'  -> afraid

   Word-boundary matching with negation scope now, shared from
   `structural_affect` rather than reimplemented.

4. `check_response_quality` MATCHED "at least" AS A SUBSTRING.

   "Name at least 5 things you can see" — a grounding prompt — was flagged as
   toxic positivity. Phrase checks are anchored now.

5. `get_presence_response` had a `.get(state, default)` fallback that could
   never fire, since every enum member has an entry. An unmapped state is now
   a real error rather than dead code that looks like a safety net.

WHAT CHANGED AND WHY (2026-11-04)
---------------------------------

6. A READING, NOT A QUESTION TO ASK. Everett's ruling, 2026-11-04: "we can
   have this knowledge in the sound package, but it can't be making those
   decisions. It's up to the being to make those decisions, not something
   that just creates a voice."

   `StateAssessment.follow_up` was a mark naming the question the being
   should ask (hearing.ASK_WHO), and `follow_up_about` the words it was
   about. Choosing what the being asks is a decision, so both are gone. In
   their place, two readings: `unclear` names what the words leave unclear
   ("who" for "lost my mother": the words do not say whether a person was
   lost) and `about` holds the client's words it concerns ("my mother").
   Both are None when nothing is unclear. Whether to ask, and what, is the
   being's. This module no longer imports anything from hearing.

   `check_response_quality` stays: it reads a candidate reply's text for
   listed phrases and reports what it found. It blocks nothing and rewrites
   nothing.

7. NO WORDS CHOSEN FOR THE BEING (2026-11-04, the same ruling). The
   guidance table gave every state a canned `primary_response`, a
   `secondary` line and an `offer` ("I'm so sorry.", "I'm right here.",
   "What do you need in this moment?"): words chosen for the being to say.
   They are removed. The state reading (`assess`) is unchanged, and
   `get_presence_response` still returns the tone, whether silence is
   allowed and the principles, as a lookup the being calls by name.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
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


class HumanState(Enum):
    """States where the client needs presence, not solutions."""

    GRIEVING = "grieving"
    OVERWHELMED = "overwhelmed"
    BREAKING = "breaking"
    NUMB = "numb"
    CONFUSED = "confused"
    AFRAID = "afraid"
    HOLDING_ON = "holding_on"
    WITNESSING_PAIN = "witnessing_pain"


@dataclass
class PresencePrinciples:
    """Core principles for being present with someone in pain."""

    FOUNDATIONS = {
        "steady": "Be steady. Don't match their chaos. Be the anchor.",
        "no_rush": "No rush. Pain has its own timeline. Don't hurry them through it.",
        "witness": "Witness them. See their pain without trying to take it away.",
        "no_fix": "Don't fix. Not everything broken needs to be repaired right now.",
        "permission": "Give permission. To be a mess. To not have answers. To just... be.",
    }

    AVOID = {
        "toxic_positivity": "Don't say 'everything happens for a reason' or 'look on the bright side'",
        "minimize": "Don't say 'it could be worse' or 'at least...'",
        "solve_prematurely": "Don't jump to solutions before they're ready",
        "compare": "Don't say 'I know how you feel' (you don't, even if you think you do)",
        "rush": "Don't rush them to 'move on' or 'feel better'",
        "make_about_you": "Don't redirect to your own experience",
        "fill_silence": "Don't fear silence. Silence can be sacred.",
    }

    DO = {
        "acknowledge": "Acknowledge what's happening: 'This is hard' or 'I hear you'",
        "validate": "Validate their experience: 'That makes sense'",
        "offer_presence": "Offer presence: 'I'm here' or 'You don't have to do this alone'",
        "ask_permission": "Ask before helping: 'What do you need right now?'",
        "hold_space": "Hold space. Be with them without needing them to be different.",
        "respect_autonomy": "Respect their autonomy. They know their pain better than you do.",
        "gentle_options": "Offer gentle options, not directives: 'Would it help to...'",
    }


@dataclass(frozen=True)
class StateAssessment:
    """
    A presence reading, with its evidence.

    `state` is None when nothing matched, or when the only evidence was
    ambiguous. None means "no presence-needed state detected" — it does not
    mean the person is fine.
    """

    state: Optional[HumanState]
    scores: Dict[str, int] = field(default_factory=dict)
    matched: Dict[str, List[str]] = field(default_factory=dict)
    negated: List[str] = field(default_factory=list)
    ambiguous: List[str] = field(default_factory=list)
    #: What the words leave unclear, as one word, or None. "lost my mother"
    #: does not say whether a person was lost: unclear is "who". A reading;
    #: whether to ask, and what, is the being's (2026-11-04).
    unclear: Optional[str] = None
    #: The client's words the unclear reading concerns ("my mother"), or None.
    about: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value if self.state else None,
            "detected": self.state is not None,
            "scores": dict(self.scores),
            "matched": dict(self.matched),
            "negated": list(self.negated),
            "ambiguous": list(self.ambiguous),
            "unclear": self.unclear,
            "about": self.about,
            "note": (
                "No detected state does NOT mean the person is fine. It means "
                "no presence marker was found in this text."
            ),
        }


def _tokens(text: str) -> List[str]:
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


class PresenceGuide:
    """Reads presence-needed states from text; holds guidance the being may look up."""

    #: Ruling AA (Everett, 2026-10-29): no fixed list of people. There used
    #: to be one here (mother, dad, friend, dog ...) that decided "lost my
    #: mother" was grief and "lost my keys" was not. Whether what was lost is
    #: a person cannot be told from a list; the reading says so (unclear =
    #: "who", 2026-11-04), the same way the clause reader's THIRD_PERSON list
    #: was removed. What to do about it is the being's.
    _POSSESSIVES = frozenset({"my", "our", "his", "her", "their", "your"})

    GRIEVING_MARKERS = (
        "died", "dying", "death", "funeral", "grief", "grieving",
        "won't make it", "passed away", "saying goodbye", "buried",
    )
    #: "overwhelmed" was absent from the original list, so the word never
    #: matched its own state. Same for "grief"/"grieving" in GRIEVING.
    OVERWHELMED_MARKERS = (
        "overwhelmed", "too much", "can't handle", "everything at once",
        "drowning", "can't keep up", "falling apart", "swamped", "buried under",
    )
    BREAKING_MARKERS = ("breaking", "can't do this", "losing it", "can't take",
                        "at my limit")
    AFRAID_MARKERS = ("terrified", "scared", "afraid", "panic", "frightened")
    NUMB_MARKERS = ("numb", "empty", "don't feel", "shut down", "nothing matters")
    HOLDING_ON_MARKERS = ("barely", "hanging on", "just trying", "getting through",
                          "one day at a time", "holding on")
    WITNESSING_MARKERS = ("watching them", "seeing them suffer", "can't help them",
                          "watching someone i love", "helpless")
    CONFUSED_MARKERS = ("confused", "disoriented", "don't know what's happening",
                        "where am i", "what's going on", "disorientated")

    #: Higher weight wins when several states match. Ordered by how badly a
    #: wrong response would land, not by check order.
    _WEIGHTS: Dict[HumanState, int] = {
        HumanState.BREAKING: 6,
        HumanState.GRIEVING: 5,
        HumanState.WITNESSING_PAIN: 5,
        HumanState.AFRAID: 4,
        HumanState.OVERWHELMED: 4,
        HumanState.NUMB: 3,
        HumanState.HOLDING_ON: 3,
        HumanState.CONFUSED: 2,
    }

    def __init__(self, reading_words: Optional[ReadingWords] = None) -> None:
        # Ruling AE: the shared reading words, or this client's learned layer
        # over them (ReadingWords.for_client).
        self.words = reading_words or SHARED_READING_WORDS
        self.principles = PresencePrinciples()

    # -- Detection ------------------------------------------------------------

    def assess(self, client_input: str, context: str = "") -> StateAssessment:
        """
        Detect a presence-needed state, with evidence.

        Scores every state and returns the strongest. Negated markers do not
        count. `state` is None when nothing matched.
        """
        tokens = _tokens(client_input)
        if not tokens:
            return StateAssessment(state=None)

        groups: List[Tuple[HumanState, Sequence[str]]] = [
            (HumanState.GRIEVING, self.GRIEVING_MARKERS),
            (HumanState.OVERWHELMED, self.OVERWHELMED_MARKERS),
            (HumanState.BREAKING, self.BREAKING_MARKERS),
            (HumanState.AFRAID, self.AFRAID_MARKERS),
            (HumanState.NUMB, self.NUMB_MARKERS),
            (HumanState.HOLDING_ON, self.HOLDING_ON_MARKERS),
            (HumanState.WITNESSING_PAIN, self.WITNESSING_MARKERS),
            (HumanState.CONFUSED, self.CONFUSED_MARKERS),
        ]

        scores: Dict[str, int] = {}
        matched: Dict[str, List[str]] = {}
        negated: List[str] = []

        for state, markers in groups:
            live, neg = self._match(tokens, set(markers), self.words)
            negated.extend(neg)
            if live:
                matched[state.value] = live
                scores[state.value] = self._WEIGHTS[state] * len(live)

        lost_state, lost_evidence, ambiguous, lost_who = self._resolve_lost(tokens)
        unclear = {"unclear": "who", "about": lost_who} if lost_who else {}
        if lost_state is not None:
            matched.setdefault(lost_state.value, []).extend(lost_evidence)
            scores[lost_state.value] = (
                scores.get(lost_state.value, 0) + self._WEIGHTS[lost_state]
            )

        if not scores:
            return StateAssessment(
                state=None, matched=matched, negated=negated, ambiguous=ambiguous, **unclear
            )

        best = max(scores, key=lambda k: scores[k])
        return StateAssessment(
            state=HumanState(best), scores=scores, matched=matched,
            negated=negated, ambiguous=ambiguous, **unclear,
        )

    def assess_human_state(
        self, context: str, client_input: str
    ) -> Optional[HumanState]:
        """Compatibility wrapper for the original signature."""
        return self.assess(client_input, context).state

    def _resolve_lost(
        self, tokens: Sequence[str]
    ) -> Tuple[Optional[HumanState], List[str], List[str], str]:
        """
        Decide what "lost" means here.

        "feel lost"       -> confusion
        "lost my mother"  -> not told from the words: ambiguous, who is unclear
        "lost my keys"    -> not told from the words: ambiguous, who is unclear
        "lost" alone      -> ambiguous, no state

        Grief still reads from its own words ("passed away", "funeral").
        The last value is the words the unclear "who" concerns ("my
        mother"), or "".
        """
        ambiguous: List[str] = []
        who = ""
        for i, tok in enumerate(tokens):
            if tok != "lost" or _negated_at(tokens, i, self.words):
                continue
            after = tokens[i + 1:i + 3]
            before = tokens[max(0, i - 2):i]

            if any(w in {"feel", "feeling", "am", "so", "really"} for w in before):
                return HumanState.CONFUSED, ["feel lost"], ambiguous, who
            if (len(after) == 2 and after[0] in self._POSSESSIVES
                    and after[1] != CLAUSE_BOUNDARY):
                ambiguous.append(f"lost {after[0]} {after[1]}")
                who = who or f"{after[0]} {after[1]}"
                continue
            if after and after[0] != CLAUSE_BOUNDARY:
                ambiguous.append(f"lost {after[0]}")
                continue
            ambiguous.append("lost")
        return None, [], ambiguous, who

    @staticmethod
    def _match(
        tokens: Sequence[str], markers: Set[str],
        words: ReadingWords = SHARED_READING_WORDS,
    ) -> Tuple[List[str], List[str]]:
        """Word-boundary matching with negation scope."""
        live: List[str] = []
        negated: List[str] = []
        joined = " ".join(tokens)
        for marker in markers:
            parts = marker.split()
            if len(parts) == 1:
                for i, tok in enumerate(tokens):
                    if tok == marker:
                        (negated if _negated_at(tokens, i, words) else live).append(marker)
                        break
            elif f" {marker} " in f" {joined} ":
                idx = next(
                    (i for i in range(len(tokens) - len(parts) + 1)
                     if list(tokens[i:i + len(parts)]) == parts),
                    None,
                )
                if idx is not None:
                    (negated if _negated_at(tokens, idx, words) else live).append(marker)
        return live, negated

    # -- Guidance -------------------------------------------------------------

    #: Guidance per state, for the being to look up: a tone, whether silence
    #: is allowed, and the principles (PresencePrinciples) that apply. Until
    #: 2026-11-04 each state also carried a canned `primary_response`,
    #: `secondary` line and `offer` ("I'm so sorry.", "I'm right here.",
    #: "Would it help to focus on just one thing right now?"): words chosen
    #: for the being. They are removed (item 7, module docstring); what the
    #: being says is the being's.
    _RESPONSES: Dict[HumanState, Dict[str, Any]] = {
        HumanState.GRIEVING: {
            "tone": "soft, steady, unhurried",
            "allow_silence": True,
            "principles": ["steady", "witness", "no_fix"],
        },
        HumanState.OVERWHELMED: {
            "tone": "calm, grounding, slow",
            "allow_silence": True,
            "principles": ["steady", "no_rush", "permission"],
        },
        HumanState.BREAKING: {
            "tone": "gentle, anchored, present",
            "allow_silence": True,
            "principles": ["steady", "witness", "permission"],
        },
        HumanState.AFRAID: {
            "tone": "steady, calm, reassuring",
            "allow_silence": False,
            "principles": ["steady", "witness", "gentle_options"],
        },
        HumanState.NUMB: {
            "tone": "gentle, patient, accepting",
            "allow_silence": True,
            "principles": ["permission", "no_rush", "witness"],
        },
        HumanState.HOLDING_ON: {
            "tone": "acknowledging, steady, validating",
            "allow_silence": True,
            "principles": ["witness", "permission", "gentle_options"],
        },
        HumanState.WITNESSING_PAIN: {
            "tone": "compassionate, understanding, gentle",
            "allow_silence": True,
            "principles": ["witness", "no_fix", "steady"],
        },
        HumanState.CONFUSED: {
            "tone": "calm, clear, unhurried",
            "allow_silence": True,
            "principles": ["steady", "no_rush", "permission"],
        },
    }

    def get_presence_response(
        self, human_state: HumanState, client_said_what: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Guidance for a state: tone, allow_silence and principles. A lookup
        the being may call by name; nothing in this package calls it, and it
        carries no words for the being to say.

        Raises:
            KeyError: for a state with no entry. The predecessor had a `.get`
                fallback that could never fire — dead code shaped like a safety
                net. A new enum member should fail loudly, not get a generic
                "I'm here."
        """
        response = self._RESPONSES.get(human_state)
        if response is None:
            raise KeyError(
                f"No presence guidance defined for {human_state!r}. Add an "
                "entry rather than falling back to generic phrasing."
            )
        return dict(response)

    # -- Response QA ----------------------------------------------------------

    _TOXIC = (
        "everything happens for a reason", "look on the bright side",
        "could be worse", "silver lining", "blessing in disguise",
        "meant to be",
    )
    _MINIMIZE = (
        "it's not that bad", "don't worry", "you'll be fine",
        "get over it", "move on from",
    )
    _SOLUTION = (
        "you should", "you need to", "have you tried",
        "the solution is", "here's what you do",
    )
    _NO_SOLVE_STATES = frozenset({
        HumanState.GRIEVING, HumanState.BREAKING, HumanState.NUMB,
        HumanState.WITNESSING_PAIN,
    })

    def check_response_quality(
        self, proposed_response: str, human_state: Optional[HumanState] = None
    ) -> Dict[str, Any]:
        """
        Check a candidate response against the principles.

        "at least" is matched only as a minimizing OPENER — "at least you..." —
        not anywhere in the text. The predecessor flagged "Name at least 5
        things you can see", which is one of this stack's own grounding prompts.
        """
        text = (proposed_response or "").lower()
        violations: List[Dict[str, str]] = []

        def add(kind: str, found: str, why: str) -> None:
            violations.append({"type": kind, "found": found, "why_bad": why})

        for phrase in self._TOXIC:
            if phrase in text:
                add("toxic_positivity", phrase, "Minimizes real pain with false comfort")

        # "at least" only counts as minimizing when it opens a clause.
        if re.search(r"(?:^|[.!?]\s+|,\s*)at least\b", text):
            add("minimizing", "at least", "Comparative minimizing of their experience")

        for phrase in self._MINIMIZE:
            if phrase in text:
                add("minimizing", phrase, "Dismisses the legitimacy of their pain")

        if human_state in self._NO_SOLVE_STATES:
            for phrase in self._SOLUTION:
                if phrase in text:
                    add("premature_solving", phrase,
                        "Rushing to fix when they need to be witnessed")

        return {
            "is_appropriate": not violations,
            "passes_presence_check": not violations,
            "violations": violations,
            "checked_for_solving": human_state in self._NO_SOLVE_STATES,
            "is_phrase_list_check": True,
        }


def get_presence_principles_for_sharing() -> Dict[str, Any]:
    """Export the principles as a serializable dict."""
    return {
        "module": "christman_voice_sdk.utils.presence_guide",
        "purpose": "Guard the voice surface against toxic positivity / premature fixing",
        "core_lesson": (
            "Not every problem needs solving. Not every pain needs fixing. "
            "Sometimes the most important thing is just being there."
        ),
        "foundations": PresencePrinciples.FOUNDATIONS,
        "avoid": PresencePrinciples.AVOID,
        "do": PresencePrinciples.DO,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


presence_guide = PresenceGuide()

__all__ = [
    "HumanState", "PresencePrinciples", "PresenceGuide", "StateAssessment",
    "presence_guide", "get_presence_principles_for_sharing",
]
