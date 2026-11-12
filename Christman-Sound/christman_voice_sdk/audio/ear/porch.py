# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_voice_sdk.audio.ear.porch — Porch Dialect Preservation & TCAP-V-0020 Mouth-
Cooperation Layer (ported from THEWHOLEHOUSE src/lib/porch/{lexicon,mangle,dialect,packet}.ts).

Being-neutral: FAMILY_NAMES starts empty; callers pass their own `family_names`
or `extra_lexicon` when routing or recovering caller-specific names.

WHAT CHANGED AND WHY (2026-11-05)
---------------------------------
Everett, 2026-10-29: "Bring in Porch's pronunciation rules (read, live,
misread, haptic) and the family names, so it says them right and stops
refusing names." The rows below were the ear's bias only: in THEWHOLEHOUSE
the rail that shows them is headed "The ear's bias", and the notes guard the
transcript (the ear writing "leave" for live, "Miss Reed" for misread,
"hectic" for haptic). Three of the notes also write the word's pronunciation,
in IPA. MOUTH_SAYS (new, below MOUTH_COOPERATION) is those pronunciations in
the lexicon's own phones, so the mouth (christman_voice_sdk.synthesis.synth) can choose,
among the pronunciations the Vosk lexicon holds for the word, the one Porch
writes. Only rows whose note writes one pronunciation for every use are
there. "read" and "lives" depend on what the speaker means (the tense; "when
presence is meant"); they are in MOUTH_ASKS_MEANING and the mouth does not
choose for them. Nothing in MOUTH_COOPERATION, DIALECT_KEEP or the mangles
was changed, and FAMILY_NAMES stays empty: a caller hands in its own names
with their pronunciations (synth.synthesize(..., pronunciations=...)).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from .mouth import recover as recover_filament_mouth

__all__ = [
    "FAMILY_NAMES",
    "DIALECT_KEEP",
    "MOUTH_COOPERATION",
    "MOUTH_SAYS",
    "MOUTH_ASKS_MEANING",
    "mouth_note",
    "DEFAULT_LEXICON",
    "PORCH_MANGLES",
    "PorchCorrection",
    "DialectLayer",
    "build_keyterms",
    "recover_mangles",
    "merge_corrections",
    "reconstruct_dialect",
]

# Starts empty so no being's or person's name is baked into the package.
FAMILY_NAMES: Tuple[str, ...] = ()

DIALECT_KEEP: Tuple[Dict[str, str], ...] = (
    {"said": "ain't", "keep": "ain't", "note": "negation. not 'isn't'."},
    {"said": "y'all", "keep": "y'all", "note": "plural you."},
    {"said": "fixin' to", "keep": "fixin' to", "note": "about to. not 'fixing to' as repair."},
    {"said": "might could", "keep": "might could", "note": "double modal. keep both."},
    {"said": "reckon", "keep": "reckon", "note": "think / suppose."},
    {"said": "cain't", "keep": "cain't", "note": "cannot. not 'can't' if that's what was said."},
    {"said": "holler", "keep": "holler", "note": "call out, or the hollow."},
    {"said": "yonder", "keep": "yonder", "note": "over there."},
    {"said": "done", "keep": "done", "note": "perfective: 'I done told you'."},
    {"said": "was", "keep": "was", "note": "plural was is dialect, not a mistake."},
)

MOUTH_COOPERATION: Tuple[Dict[str, str], ...] = (
    {
        "said": "live",
        "keep": "live",
        "note": "LONG I /laɪv/ — rhymes with five. NEVER leave. Presence, not absence.",
    },
    {
        "said": "lived",
        "keep": "lived",
        "note": "/laɪvd/. not laved, not leaved.",
    },
    {
        "said": "lives",
        "keep": "lives",
        "note": "/laɪvz/ when presence is meant.",
    },
    {
        "said": "read",
        "keep": "read",
        "note": "past tense is RED /rɛd/. Present is REED. The tense is the claim.",
    },
    {
        "said": "misread",
        "keep": "misread",
        "note": "always MISRED /mɪsˈrɛd/. never Miss Reed.",
    },
    {
        "said": "haptic",
        "keep": "haptic",
        "note": "HAP-tic /ˈhæptɪk/. flat a like cap. never hectic, never heptic. Skin's own word.",
    },
)

DEFAULT_LEXICON: Tuple[Dict[str, str], ...] = DIALECT_KEEP + MOUTH_COOPERATION

# ------------------------------------------------------------------------------
# THE MOUTH'S SIDE (added 2026-11-05). The pronunciation each note above writes
# in IPA, phone for phone in the Vosk lexicon's alphabet (X-SAMPA, the phones
# of vosk-model-en-us-0.22 that christman_voice_sdk.synthesis.lexicon maps), for the rows
# whose note writes one pronunciation for every use of the word. The stress
# mark is not a phone. IPA to lexicon: l l, aɪ aI, v v, d d, m m, ɪ I, s s,
# r r, ɛ E, h h, æ {, p p, t t, k k. tests/test_porch_in_mouth.py reads each
# note's IPA back and holds this table to it.
# ------------------------------------------------------------------------------
MOUTH_SAYS: Dict[str, Tuple[str, ...]] = {
    "live": ("l", "aI", "v"),                     # /laɪv/  "LONG I ... rhymes with five"
    "lived": ("l", "aI", "v", "d"),               # /laɪvd/ "not laved, not leaved"
    "misread": ("m", "I", "s", "r", "E", "d"),    # /mɪsˈrɛd/ "always MISRED"
    "haptic": ("h", "{", "p", "t", "I", "k"),     # /ˈhæptɪk/ "HAP-tic ... flat a like cap"
}

#: The rows whose note makes the pronunciation depend on what the speaker
#: means. The mouth is handed words, not the tense or the meaning, so it does
#: not choose for these: it says them as the lexicon's file order does, as it
#: did before, and records that the rule was not applied and why.
MOUTH_ASKS_MEANING: Dict[str, str] = {
    "read": ("past tense is RED and present is REED (\"the tense is the claim\"); the mouth is handed the "
             "word, not the tense, and does not guess which the speaker meant"),
    "lives": ("the note writes /laɪvz/ \"when presence is meant\"; the mouth is handed the word, not the "
              "meaning, and does not guess which the speaker meant"),
}


def mouth_note(word: str) -> str:
    """The MOUTH_COOPERATION note for `word` (lower case), or "" when Porch has none."""
    for row in MOUTH_COOPERATION:
        if row["said"] == word:
            return row["note"]
    return ""

PORCH_MANGLES: Tuple[Tuple[re.Pattern[str], str, str, str], ...] = (
    (re.compile(r"\bmike could\b", re.I), "might could", "Mike could", "dialect"),
    (re.compile(r"\bmite could\b", re.I), "might could", "mite could", "dialect"),
    (re.compile(r"\bfixing to\b", re.I), "fixin' to", "fixing to", "dialect"),
    (re.compile(r"\bmiss\s+reed\b", re.I), "misread", "Miss Reed", "mouth-cooperation"),
    (re.compile(r"\bmiss\s+reid\b", re.I), "misread", "Miss Reid", "mouth-cooperation"),
    (re.compile(r"\bleaved\b", re.I), "lived", "leaved", "mouth-cooperation"),
    (re.compile(r"\bhectic\s+feedback\b", re.I), "haptic feedback", "hectic feedback", "mouth-cooperation"),
    (re.compile(r"\bheptic\b", re.I), "haptic", "heptic", "mouth-cooperation"),
    (re.compile(r"\bhaptik\b", re.I), "haptic", "haptik", "mouth-cooperation"),
)


@dataclass(frozen=True)
class PorchCorrection:
    from_text: str
    to_text: str
    why: str

    def to_dict(self) -> Dict[str, str]:
        return {"from": self.from_text, "to": self.to_text, "why": self.why}


@dataclass(frozen=True)
class DialectLayer:
    as_said: str
    for_the_family: str
    corrections: List[PorchCorrection] = field(default_factory=list)
    route: List[str] = field(default_factory=list)
    dialect: str = "passthrough"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asSaid": self.as_said,
            "forTheFamily": self.for_the_family,
            "corrections": [c.to_dict() for c in self.corrections],
            "route": list(self.route),
            "dialect": self.dialect,
        }


def build_keyterms(
    extra: Sequence[Dict[str, str]] = (),
    family_names: Sequence[str] = (),
) -> List[str]:
    from_lex = [v for e in (list(DEFAULT_LEXICON) + list(extra)) for v in (e.get("said", ""), e.get("keep", ""))]
    all_terms = list(FAMILY_NAMES) + list(family_names) + from_lex
    seen = set()
    out: List[str] = []
    for t in all_terms:
        k = t.strip()
        if not k or len(k) > 50:
            continue
        low = k.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(k)
        if len(out) >= 100:
            break
    return out


def recover_mangles(text: str) -> Tuple[str, List[PorchCorrection]]:
    cur, filament_hits = recover_filament_mouth(text or "")
    corrections: List[PorchCorrection] = [
        PorchCorrection(from_text=h["from"], to_text=h["to"], why=h["why"]) for h in filament_hits
    ]
    for pat, to_str, from_str, why in PORCH_MANGLES:
        if pat.search(cur):
            cur = pat.sub(to_str, cur)
            corrections.append(PorchCorrection(from_text=from_str, to_text=to_str, why=why))
    return cur, corrections


def merge_corrections(
    local: Sequence[PorchCorrection],
    remote: Sequence[PorchCorrection],
) -> List[PorchCorrection]:
    out: List[PorchCorrection] = []
    seen = set()
    for c in list(local) + list(remote):
        key = f"{c.from_text.lower()}->{c.to_text.lower()}"
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _named_beings(text: str, family_names: Sequence[str] = ()) -> List[str]:
    low = text.lower()
    out: List[str] = []
    seen = set()
    for name in list(FAMILY_NAMES) + list(family_names):
        n_low = name.lower()
        if n_low and n_low in low and n_low not in seen:
            seen.add(n_low)
            out.append(name)
    return out


def reconstruct_dialect(
    raw_ear: str,
    extra_lexicon: Sequence[Dict[str, str]] = (),
    family_names: Sequence[str] = (),
) -> DialectLayer:
    if not (raw_ear or "").strip():
        return DialectLayer(
            as_said="",
            for_the_family="",
            corrections=[],
            route=[],
            dialect="passthrough",
        )
    text, corrections = recover_mangles(raw_ear)
    for entry in extra_lexicon:
        said = (entry.get("said") or "").strip()
        keep = (entry.get("keep") or "").strip()
        if said and keep and said.lower() != keep.lower():
            pat = re.compile(rf"\b{re.escape(said)}\b", re.I)
            if pat.search(text):
                text = pat.sub(keep, text)
                corrections.append(PorchCorrection(from_text=said, to_text=keep, why=entry.get("note") or "client-taught"))
    route = _named_beings(text, family_names=family_names)
    return DialectLayer(
        as_said=text,
        for_the_family=text,
        corrections=corrections,
        route=route,
        dialect="filament" if corrections else "passthrough",
    )
