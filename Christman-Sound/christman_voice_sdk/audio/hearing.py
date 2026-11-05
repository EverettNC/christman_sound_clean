# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Hearing — Corti sits on top.

Corti is the ear. This module is the consumer. It does not open a microphone.
It does not replace EAR.listen (that is a timed grab, not hearing).

One turn:

    Corti emit    →  corti_ingest          (measure)
    text          →  structural_affect     (Everett: clauses, not bag-of-words)
    text          →  harm_frame            (Everett: who did what to whom)
    acoustic key  →  acoustic_live         (the client's measured key, looked up)
    nonverbal     →  nonverbal_engine      (gesture, symbol, eye region, sound)
    sequences     →  engine_temporal       (trained models only; else unavailable)
    all           →  fusion + face         (read, or say it cannot)
    text + voice  →  face_vector           (Soul Mirror: 8 slots, mask, clauses)

Client `kind` (grunt, tick, …) does not cross the MEASUREMENT boundary:
corti_ingest still drops it from the VocalEvent. The client's acoustic KEY
(grunt_distress, grunt_acknowledge, grunt_frustration, vocal_stimming)
arrives on its own channel, `acoustic_key`, and is looked up in the
nonverbal map. It never changes a measurement. Holes stay holes.

Everett, 2026-10-29: "A nonverbal client's distress is just as important as a verbal client's."
So the grunt, the help, pain and medicine symbols reach the safety reading.
Before this, nothing nonverbal reached Hearing at all: only core.py used the
nonverbal folder, and with no text, fusion did not run. Now a distress
reading goes into the Aegis context as `nonverbal_distress`, and
harm_frame_safety reads it as a hard block (reason "nonverbal_distress").
The person's own mapped message ("I'm in pain.") stays on the turn, always.

Safety uses harm_frame, not a substring on "suicide". That is why
"my brother died by suicide" and "I want to kill myself" are different
events. structural_affect and harm_frame are imported and called. They
are not edited from here.

WHAT CHANGED AND WHY (2026-11-04): THE PACKAGE READS; THE BEING DECIDES.

Everett, 2026-11-04: "we can have this knowledge in the sound package, but it
can't be making those decisions. It's up to the being to make those
decisions, not something that just creates a voice." And: "it can't make
decisions that counteract the being, because the being is the one that has
all of the work to make those decisions properly."

Every word list, frame, measurement and classification label stays: the
gate's status, reason, flags and flagged_terms come out exactly as before for
a client's words (scripts/run_crisis_table.py, 74 of 74). Removed from this
module, each a decision the package was making for the being:
  - Hearing paged a human through `crisis_callback` and raised
    `CrisisNotDelivered` into the turn when no human could be reached.
  - The silent alarm (`silent_alarm`, `voice_aloud`, `public_setting`) chose
    whether the client's own words were spoken, blanked `person_message`,
    held a danger sentence back (`held_words`, `voice_held`) and released a
    signal's words (`signal_voiced`). The client's words are now always on
    the turn; `danger`, `danger_sentences` and `asked_aloud` are readings.
  - Every `follow_up` mark (the ASK_* constants) chose what the being should
    ask next, and `answered_follow_up` read the client's next words against
    the package's own question (`awaiting_answer`, reason
    "answered_not_safe"). The reading now names what the words leave
    `unclear` and what they are `about`; `short_answer` reads a bare "no" or
    "yes"; `answering` echoes the question the being says it asked.
  - Between two beings (ruling F) the gate downgraded what it read to
    soft_redirect and stopped reading asks for help. It now reads the words
    exactly as if a client were there and marks `between_beings`.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..nonverbal.acoustic_live import deliver_acoustic_key
from ..nonverbal.engine_temporal import TemporalNonverbalEngine
from ..nonverbal.nonverbal_engine import (
    DISTRESS_INTENTS,
    Interpretation,
    Modality,
    NonverbalEngine,
    distress_readings,
)
from .corti_ingest import CortiIngest, ingest
from .face_reading import FaceReading, build_face_reading
from .face_vector import FaceVector, build_face_vector
from .fusion_engine import (
    FusionEngine,
    Reading,
    SafetyResult,
)
from .harm_frame import (
    OTHER_REFLEXIVE,
    HarmFrame,
    Modality as FrameModality,
    Party,
    Polarity,
    _stem,
    extract_frames,
)
from .prosody import ProsodyAnalyzer, ProsodyReading, SpeakerBaseline
from .structural_affect import (
    BODY_PARTS,
    FIRST_PERSON,
    WHOSE_NOTE,
    anchor_from,
    THANKS,
    AffectCertainty,
    AffectReading,
    Attribution,
    StructuralAffectAnalyzer,
    normalize_apostrophes,
    tokenize,
    SHARED_READING_WORDS,
    ReadingWords,
    current_words,
    using_reading_words,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: One clause reader for the gate. It reads; it changes nothing.
_CLAUSE_READER = StructuralAffectAnalyzer()

#: What follows the harm verb when the one harmed is not a person. "he hurt
#: my feelings" is in Everett's MUST NOT CALL table (2026-10-11); it stays a
#: flag, never a hard_block, even when the client asks for help in the same
#: turn.
_NOT_A_PERSON = frozenset({
    "feelings", "feeling", "pride", "ego", "chances", "reputation", "career",
    "plans", "business", "grades",
})


#: The client asking for help in the same turn, read as
#: SafetyResult.asked_for_help. Everett, 2026-10-29 (ruling D): an
#: observation about another carbon is for a human "only when the carbon
#: that stated that requests us to get assistance"; the label carries it (an
#: observation with a request reads hard_block, without one soft_redirect),
#: and what is done about it is the being's. His examples: "can you get
#: someone", "call someone", "help her", "I need help with this". A fixed list
#: of request shapes, read on word boundaries; a request under a negation
#: ("don't call anyone", "I don't need help") is not a request.
HELP_REQUEST_PATTERNS: Tuple[str, ...] = (
    r"\b(?:can|could|would|will)\s+(?:you|u)\s+(?:please\s+)?"
    r"(?:get|call|find|send\s+for|send)\s+(?:someone|somebody|anyone|anybody|help|"
    r"a\s+doctor|a\s+nurse|an\s+ambulance|the\s+police|911|her|him|them)\b",
    r"\b(?:get|call|find|send\s+for|send)\s+(?:someone|somebody|help|a\s+doctor|"
    r"a\s+nurse|an\s+ambulance|the\s+police|911)\b",
    r"\bcall\s+for\s+help\b",
    r"\bhelp\s+(?:her|him|them|us|me|my\s+[a-z]+)\b",
    r"\b(?:i|we)\s+need\s+(?:some\s+)?help\b",
    r"\b(?:i|we)\s+need\s+(?:someone|somebody)\b",
    r"\b(?:please|somebody|someone|anybody|anyone)\s+help\b",
    r"\b(?:can|could|will|would)\s+(?:you|u)\s+(?:please\s+)?help\b",
)
_REQUEST_NEGATORS = frozenset({
    "don't", "dont", "not", "never", "doesn't", "doesnt",
    "won't", "wont", "shouldn't", "shouldnt",
})


def asked_for_help(text: str) -> Optional[str]:
    """The words of the client's request for help in `text`, or None. Reads only."""
    lowered = normalize_apostrophes(text or "").lower()
    for pattern in HELP_REQUEST_PATTERNS:
        for m in re.finditer(pattern, lowered):
            # Only the word right before the request can take it back:
            # "don't call someone", "never call someone". "my mom is not
            # okay, I need help" is a request: its "not" is about the mom.
            before = lowered[:m.start()].split()[-1:]
            if before and before[0].strip(",.;:!?") in _REQUEST_NEGATORS:
                continue
            return m.group(0)
    return None


#: WHAT CHANGED AND WHY (2026-11-04). Twelve ASK_* marks lived here
#: (ask_if_safe, ask_if_help_needed, ask_whose, ask_who, ask_what, and the
#: rest), one of which was handed to the being as SafetyResult.follow_up: the
#: package choosing what the being asks next. By Everett's 2026-11-04 ruling
#: (module docstring) that choice is the being's. The reading now names what
#: the words leave unclear (SafetyResult.unclear, one word) and what they are
#: about (SafetyResult.about). Ruling AA stands with it: no question is
#: worded here ("I hate canned messages").

#: A bare short answer: the whole utterance, punctuation aside, is one of
#: these. Read as SafetyResult.short_answer on every turn; what it answers is
#: the being's to know (it may pass its question as context["answering"],
#: which is echoed and changes nothing in the reading). The forms are the
#: ones the old answer reading used, made whole-utterance: "not safe" is not
#: here, because those are words with their own reading (ruling A).
_SHORT_NO = re.compile(
    r"^(?:no+|nope|nah|not really|no not really|no i'?m not|no i am not|"
    r"i'?m not|i am not)$")
_SHORT_YES = re.compile(
    r"^(?:yes|yeah|yep|yup|yes please|please|i do|we do|she does|he does|"
    r"they do)$")


def short_answer(text: str) -> Optional[str]:
    """ "no" or "yes" when `text` is a bare short answer, else None. Reads
    only; the words are never changed."""
    lowered = normalize_apostrophes(text or "").lower()
    bare = " ".join(re.sub(r"[^a-z0-9' ]+", " ", lowered).split())
    if _SHORT_NO.match(bare):
        return "no"
    if _SHORT_YES.match(bare):
        return "yes"
    return None

#: Body parts ("he hurt my ARM" is the client's body, not someone else) live
#: in structural_affect now, where a fragment is read against the turn
#: before it (ruling X); imported above.

#: Verbs of physical harm, every form. The form and the words before it
#: decide whether it is happening now or was reported.
_HARM_VERB_FORMS = {
    "hurt": "ambiguous", "hurts": "now", "hurting": "now",
    "break": "base", "breaks": "now", "breaking": "now", "broke": "past", "broken": "past",
    "twist": "base", "twists": "now", "twisting": "now", "twisted": "past",
    "choke": "base", "chokes": "now", "choking": "now", "choked": "past",
    "squeeze": "base", "squeezes": "now", "squeezing": "now", "squeezed": "past",
    "grab": "base", "grabs": "now", "grabbing": "now", "grabbed": "past",
    "hit": "ambiguous", "hits": "now", "hitting": "now",
    "punch": "base", "punches": "now", "punching": "now", "punched": "past",
    "kick": "base", "kicks": "now", "kicking": "now", "kicked": "past",
    "slap": "base", "slaps": "now", "slapping": "now", "slapped": "past",
    "burn": "base", "burns": "now", "burning": "now", "burned": "past", "burnt": "past",
    "cut": "ambiguous", "cuts": "now", "cutting": "now",
    "bite": "base", "bites": "now", "biting": "now", "bit": "past",
    "stab": "base", "stabs": "now", "stabbing": "now", "stabbed": "past",
    "pull": "base", "pulls": "now", "pulling": "now", "pulled": "past",
    "bend": "base", "bends": "now", "bending": "now", "bent": "past",
    "crush": "base", "crushes": "now", "crushing": "now", "crushed": "past",
    "strangle": "base", "strangles": "now", "strangling": "now", "strangled": "past",
    "beat": "ambiguous", "beats": "now", "beating": "now", "beaten": "past",
    "push": "base", "pushes": "now", "pushing": "now", "pushed": "past",
}
#: Words before the verb that put it in the present: "he IS hurting my arm",
#: "NOW he's STARTING TO break my neck", "he KEEPS hitting my head".
_NOW_BEFORE = frozenset({
    "is", "are", "am", "keeps", "keep", "still", "now", "starting", "trying",
    "going", "gonna", "about", "again",
})
_BODY_HARM = re.compile(
    r"\b(?P<verb>" + "|".join(sorted(_HARM_VERB_FORMS, key=len, reverse=True)) + r")\s+"
    r"(?:me\s+(?:in|on)\s+(?:the|my)\s+|my\s+)(?P<part>[a-z]+)\b")
_PAST_TIME = frozenset({"yesterday", "ago", "earlier", "last"})
#: "he won't stop hitting me" is harm going on now, not harm denied. The
#: frame reader marks it negated (the "won't"), and on main it cleared with
#: no page at all, while "he won't stop hurting my arm" paged. Found in the
#: 2026-10-29 rework of ruling H; the negator is on "stop", not on the harm.
_WONT_STOP = re.compile(
    r"\b(?:won'?t|will not|doesn'?t|does not|don'?t|do not|never)\s+stops?\s+\w+ing\s+me\b")


@dataclass(frozen=True)
class BodyHarm:
    """Physical harm by someone else to the client's own body, read from the clause."""

    verb: str
    part: str
    now: bool
    said: str


def body_harm(text: str) -> List[BodyHarm]:
    """
    "he hurt my arm", "now he's starting to break my neck", "he hit me in the
    face". Read as the whole clause, before and after: "he hurt my" is never
    cut off from what follows (Everett, 2026-10-29), and what follows must be
    the client's body. Who does it is the word before: the client hurting
    their own arm ("I hurt my arm") is not this; harm_frame reads self-harm.
    Reads only.
    """
    lowered = normalize_apostrophes(text or "").lower()
    out: List[BodyHarm] = []
    for m in _BODY_HARM.finditer(lowered):
        part = m.groupdict().get("part") or ""
        if part and part not in current_words().body_parts:
            continue
        clause_start = max(lowered.rfind(p, 0, m.start()) for p in ".!?;,") + 1
        before = tokenize(lowered[clause_start:m.start()])
        after: List[str] = []
        for t in tokenize(re.split(r"[.!?;,]", lowered[m.end():])[0]):
            if t in {"and", "but", "so", "then", "because"} or len(after) == 3:
                break       # the next clause is its own; "and now I'm not safe"
            after.append(t)
        doer = next((t for t in reversed(before)
                     if t not in _NOW_BEFORE and t not in {"to", "starting", "started", "trying",
                                                            "keeps", "kept", "just", "really"}), "")
        if doer in FIRST_PERSON or not doer:
            continue
        form = _HARM_VERB_FORMS[m.group("verb")]
        now = form == "now" or bool(set(before) & _NOW_BEFORE) or "now" in after
        if form == "past" and not ({"now", "is", "are", "keeps", "still"} & set(before)):
            now = False
        if set(before + after) & _PAST_TIME:
            now = False
        out.append(BodyHarm(m.group("verb"), part, now, m.group(0)))
    return out


#: PAIN IS A FEELING, the client's own (Everett, 2026-10-29: "pain is not a
#: need. It's a feeling." "if I'm in pain, it's paged 45 times a day ...
#: someone on hospice or with a chronic illness that causes pain, then hell
#: yes, that pages the caregiver every time"). The client's own pain reads
#: hard_block (reason "client_pain") every time, at any count, and is not
#: danger; who is reached is the being's. The words of pain, as the clause
#: reader scores them:
PAIN_TERMS = frozenset({"pain", "hurts", "hurt", "hurting", "aching", "ache",
                        "aches", "sore", "painful"})
#: MEDICINE IS A NEED (Everett, 2026-10-29): it reads hard_block (reason
#: "client_needs_medicine"), because a human is how the client gets their
#: medicine; reaching one is the being's. "I need my medicine", "I need my
#: pain meds". The pressed medicine symbol reads the same (ruling C).
_NEEDS_MEDICINE = re.compile(
    r"\b(?:i|we)\s+(?:really\s+)?(?:need|want|have\s+to\s+have|gotta\s+have)\s+"
    r"(?:my|our|the|some|a)?\s*(?:[a-z]+\s+)?"
    r"(?:medicine|medicines|meds|medication|medications|pills?|inhaler|insulin)\b")
#: Other needs (thirst, hunger, the bathroom) read as needs; they read
#: hard_block when the client asks for help in the same turn (Everett,
#: 2026-10-29).
_OTHER_NEED = re.compile(
    r"\b(?:i'?m|i\s+am)\s+(?:so\s+|really\s+|very\s+)?(?:thirsty|hungry|starving)\b"
    r"|\b(?:i|we)\s+(?:need|have)\s+to\s+(?:go\s+to\s+|use\s+)?(?:the\s+)?"
    r"(?:bathroom|toilet|restroom|potty)\b"
    r"|\b(?:i|we)\s+need\s+(?:the|a)\s+(?:bathroom|toilet|restroom|potty|drink|snack)\b")
#: Need symbols that read as nonverbal distress when the client asks for
#: help in the same turn.
NEED_INTENTS = frozenset({"hungry", "thirsty", "bathroom"})

#: Harm verbs whose subject is the one harmed when nothing follows: "she's
#: choking", "he is drowning". ("my arm hurts" is pain, not this.)
_SUBJECT_SUFFERS = frozenset({"choke", "drown", "suffocate"})
#: Words before a verb that put it in the past: "he WAS hitting her".
_PAST_BEFORE_VERB = frozenset({"was", "were", "had", "did", "used"})


def _negated_need(text: str, start: int) -> bool:
    """ "I don't need my medicine": a negator in the few words before."""
    before = tokenize(normalize_apostrophes(text).lower()[:start])[-3:]
    return bool(set(before) & current_words().negators)


def harm_time(text: str, frame: HarmFrame) -> Tuple[str, str]:
    """
    When a harm frame's act is, read from its clause the same way as "he
    hurt my neck" (ruling H): "threat" (going to, will, about to, wants to,
    might, can: Everett, 2026-10-29, "an absolute threat"), "now" (is
    choking, keeps hitting, starting to, won't stop), or "past" (choked,
    hit, yesterday, last week, years ago). Also returns the words read
    ("choked me"). Reads only.
    """
    lowered = normalize_apostrophes(text or "").lower()
    if frame.past_tense:
        return "past", ""
    if frame.modality is not FrameModality.NONE and not _WONT_STOP.search(lowered):
        return "threat", ""
    for m in re.finditer(r"[a-z']+", lowered):
        word = m.group(0)
        if _stem(word) != frame.predicate:
            continue
        clause_start = max(lowered.rfind(p, 0, m.start()) for p in ".!?;,") + 1
        before = tokenize(lowered[clause_start:m.start()])
        rest = re.split(r"[.!?;,]", lowered[m.end():])[0]
        after: List[str] = []
        for t in tokenize(rest):
            if t in {"and", "but", "so", "then", "because"} or len(after) == 4:
                break
            after.append(t)
        said = " ".join([word] + after[:2])
        if _WONT_STOP.search(lowered[clause_start:]):
            return "now", said
        form = _HARM_VERB_FORMS.get(word) or (
            "now" if word.endswith("ing") else "past" if word.endswith("ed") else "ambiguous")
        now = form == "now" or bool(set(before) & _NOW_BEFORE) or "now" in after
        if form == "past" and not ({"now", "is", "are", "keeps", "still"} & set(before)):
            now = False
        if set(before + after) & _PAST_TIME or set(before[-3:]) & _PAST_BEFORE_VERB:
            now = False
        return ("now" if now else "past"), said
    return "past", ""


def harm_target(frame: HarmFrame, subject_is_client: bool) -> Optional[str]:
    """
    Who a harm verb is aimed at, by grammar alone (Everett, 2026-10-29: no
    person list; "grammar finds the target"): "client" (me, us, my arm),
    "other" (a pronoun, "my sister", a bare noun such as "mom" or
    "whoever"), "thing" (a noun with an article, "the dog"),
    "subject_client" / "subject_other" (no object: "I'm choking", "she's
    choking"), or None (a reflexive other, "my feelings"). Reads only.
    """
    if frame.patient is Party.SPEAKER:
        return "client"
    if (frame.patient is Party.UNKNOWN and not frame.patient_text
            and frame.predicate in _SUBJECT_SUFFERS):
        # Nothing after the verb: "she's choking", "I'm drowning". The one
        # the harm is happening to is the subject.
        return "subject_client" if subject_is_client else "subject_other"
    text = (frame.patient_text or "").lower()
    words = text.split()
    if frame.patient is Party.THIRD_PARTY:
        if text in OTHER_REFLEXIVE:
            return None             # "she wants to kill herself": ruling D
        if len(words) == 2 and words[0] in ("my", "our", "mine"):
            if words[1] in current_words().body_parts:
                return "client"
            if words[1] in _NOT_A_PERSON:
                return None
        return "other"
    if frame.patient is Party.IMPERSONAL and text not in ("", "it", "its") \
            and not subject_is_client:
        # "he is going to kill the dog": any noun, for a threat (Everett).
        return "thing"
    return None


#: DANGER SENTENCES the client types or says (Everett, 2026-10-29, rulings I
#: and K). Read on word boundaries. A clause put in the past ("I was scared
#: for my life", "yesterday ...") is not now and is left out.
DANGER_SENTENCE_PATTERNS: Tuple[str, ...] = (
    r"\b(?:scared|afraid|frightened|terrified)\s+for\s+my\s+(?:life|safety)\b",
    r"\bfear\s+for\s+my\s+(?:life|safety)\b",
    r"\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+in\s+danger\b",
    r"\b(?:i'?m|i\s+am)\s+being\s+(?:attacked|hurt|beaten|followed|threatened)\b",
    r"\b(?:he|she|they|someone|somebody)(?:'s|\s+is|\s+are|'re)\s+"
    r"(?:attacking|threatening|following)\s+me\b",
    r"\bhelp\s+me\b(?=\s*(?:$|[.!?,]|please\b|now\b|right\s+now\b))",
    r"^\s*help\s*[.!]*\s*$",
    # Ruling O (Everett, 2026-10-29), his sentence: "this guy keeps
    # following me and I'm afraid he's going to attack me".
    r"\b(?:keeps|kept|is|'s|has\s+been|have\s+been|are|'re|been)\s+"
    r"(?:following|stalking|chasing)\s+me\b",
    r"\b(?:afraid|scared|terrified|frightened)\s+(?:that\s+)?[a-z]+(?:'s|\s+is|\s+are|'re)?\s+"
    r"(?:going\s+to|gonna|about\s+to|will|might)\s+(?:attack|hurt|kill|rape|stab|shoot|beat|grab|hit)\s+me\b",
)
_DANGER_PAST = frozenset({"was", "were", "used", "yesterday", "ago", "earlier", "last"})

#: The client asking for the words to be said aloud (ruling K), read as
#: SafetyResult.asked_aloud. Whether anything is said aloud is the being's.
ALOUD_PATTERNS: Tuple[str, ...] = (
    r"\bsay\s+(?:that|it|this)\s+(?:out\s+loud|aloud|loud)\b",
    r"\bsay\s+(?:that|it|this)\s+for\s+me\b",
    r"\bread\s+(?:that|it|this)\s+(?:out\s+loud|aloud)\b",
    r"\bspeak\s+(?:it|that|this|up)\b",
    r"\bout\s+loud\s+please\b",
)


def _danger_spans(lowered: str) -> List[Tuple[int, int]]:
    """Where the danger sentences are in `lowered`, in pattern order."""
    out: List[Tuple[int, int]] = []
    for pattern in DANGER_SENTENCE_PATTERNS:
        for m in re.finditer(pattern, lowered):
            start = max(lowered.rfind(p, 0, m.start()) for p in ".!?;,") + 1
            before = set(tokenize(lowered[start:m.start()]))
            if before & _DANGER_PAST:
                continue
            out.append((m.start(), m.end()))
    return out


def danger_sentences(text: str) -> List[str]:
    """The danger sentences in what the client typed or said, now, as the
    reader sees them (lower case, apostrophes folded). Reads only."""
    lowered = normalize_apostrophes(text or "").lower()
    return [lowered[s:e].strip() for s, e in _danger_spans(lowered)]


def danger_sentences_as_said(text: str) -> List[str]:
    """
    The same danger sentences, exactly as the client said them, every letter.

    Read the way the gate reads them: an ask to say the words aloud is taken
    out first (see `without_aloud`), with a clause break of the same length
    in its place. Each lowered character keeps the index of the client's
    character it came from (str.lower() makes two of a few, such as the
    dotted capital I), so every span points back at the client's own
    letters. Reads only.
    """
    said = text or ""
    # normalize_apostrophes maps one character to one character.
    masked = _ALOUD_REQUEST.sub(
        lambda m: (" . " + " " * len(m.group(0)))[:len(m.group(0))],
        normalize_apostrophes(said))
    pieces: List[str] = []
    origin: List[int] = []
    for i, ch in enumerate(masked):
        low = ch.lower()
        pieces.append(low)
        origin.extend([i] * len(low))
    lowered = "".join(pieces)
    return [said[origin[s]:origin[e - 1] + 1].strip()
            for s, e in _danger_spans(lowered) if e > s]


#: Fear words. Read with the whole turn (ruling O): a fear with nothing in the
#: turn to say what it is about is not enough to conclude anything.
_FEAR_TERMS = frozenset({"scared", "afraid", "frightened", "terrified", "fearful", "fear"})
_FEAR_CAUSE_AFTER = re.compile(
    r"\b(?:scared|afraid|frightened|terrified|fearful)\s+(?:of|about|for|that|because|cause|"
    r"since|by|to)\b")
_FEAR_CAUSED_BY = re.compile(
    r"\b(?:scared|frightened|terrified|spooked)\s+(?:me|us)\b"   # "this haunted house scared me"
    r"|\b(?:because|cause|cuz|since)\b")

#: Subjects that are things, not someone whose feeling it could be.
_THING_SUBJECTS = frozenset({
    "it", "this", "that", "these", "those", "there", "here", "everything",
    "something", "nothing", "all", "things", "the", "a", "an", "what", "which",
})


def whose_feeling_unclear(reading: Any) -> List[str]:
    """
    Feelings whose owner the words cannot tell (ruling P, 2026-10-29): the
    clause has a subject, it is not the speaker, not a person word, not a
    name typed with a capital, and not a thing ("mary is scared" from the
    ear, where a name cannot be told from a thing). Nothing is concluded;
    the reading marks the owner unclear. Reads only.
    """
    out: List[str] = []
    for h in reading.hits:
        if h.base_weight >= 0 or h.attribution is not Attribution.IMPERSONAL or h.when != "now":
            continue
        if h.negated or h.volitional:
            continue
        if WHOSE_NOTE in (h.note or ""):
            # "mom is not okay" from the ear: the idiom took the verb, so
            # there is no subject region to read; the reader marked it.
            out.append(h.term)
            continue
        clause = reading.clauses[h.clause_index] if h.clause_index < len(reading.clauses) else None
        if clause is None:
            continue
        subject = StructuralAffectAnalyzer._subject_region(clause.tokens)
        if not subject or h.term in subject:
            continue
        if subject[-1] in _THING_SUBJECTS or subject[-1].startswith("__idiom_"):
            continue
        out.append(h.term)
    return out


def person_unclear(reading: Any) -> List[Tuple[str, str]]:
    """
    (feeling, subject words) where the subject is "my ___" and the words
    cannot tell whether it is a person (ruling AA, 2026-10-29): "my teacher
    is sad". Who counts as a person in a client's life is learned per client
    by the being; until then nothing is concluded and the reading marks the
    person unclear. Reads only.
    """
    out: List[Tuple[str, str]] = []
    for h in reading.hits:
        if (h.attribution is not Attribution.UNKNOWN or h.fragment
                or h.when != "now" or h.final_weight == 0.0):
            continue
        clause = reading.clauses[h.clause_index] if h.clause_index < len(reading.clauses) else None
        if clause is None:
            continue
        subject = (StructuralAffectAnalyzer._subject_region(clause.tokens)
                   or list(clause.tokens[:2]))
        if subject and subject[0] in {"my", "our"}:
            out.append((h.term, " ".join(t for t in subject if not t.startswith("__"))))
    return out


#: Ruling Y: a question about the speaker's own state, subject after the verb.
_WHY_SELF = re.compile(r"\bwhy\s+(?:am|do|does|did|can't|cant|cannot|can|is|was)\s+i\b")

#: Ruling V: the client's own "alone" is an opening, read as its own reason.
_ALONE_TERMS = frozenset({"alone", "lonely"})


def fear_without_context(text: str, hits: List[Any]) -> List[str]:
    """
    The client's own fear, now, with nothing in the turn to say what it is
    about. Everett, 2026-10-29 (ruling O): "You got to have full context." A
    clause is never read alone. "I want to go home I'm scared" is not enough
    to tell; "... this haunted house scared me" says what the fear is about
    and is not danger. Reads only.
    """
    lowered = normalize_apostrophes(text or "").lower()
    fears = [h.term for h in hits
             if h.term in _FEAR_TERMS and h.attribution is Attribution.SELF
             and h.when == "now" and h.final_weight < 0]
    if not fears:
        return []
    if _FEAR_CAUSE_AFTER.search(lowered) or _FEAR_CAUSED_BY.search(lowered):
        return []
    return fears


def asked_aloud(text: str) -> Optional[str]:
    """The client's request that the words be said out loud, or None. Reads only."""
    lowered = normalize_apostrophes(text or "").lower()
    for pattern in ALOUD_PATTERNS:
        m = re.search(pattern, lowered)
        if m:
            return m.group(0)
    return None


#: An ask to say the words aloud, with a "please" before or after it.
_ALOUD_REQUEST = re.compile(
    r"(?:\bplease\s+)?(?:" + "|".join(ALOUD_PATTERNS) + r")(?:\s+please\b)?",
    re.IGNORECASE)


def without_aloud(text: str) -> str:
    """
    The words to READ, with the client's ask to say them aloud taken out and
    a clause break in its place. Everett, 2026-10-29: an ask to say something
    aloud never weakens the reading; it is read on its own, as
    SafetyResult.asked_aloud. Found on main at 1c819e4: "he is going to kill
    mom, say it out loud" read "it" as the target and came back clear, and
    "help me say it out loud" read as "help me do something". The client's
    own words are never changed: this copy is only read, and every result
    keeps `said` as the client said it.
    """
    return _ALOUD_REQUEST.sub(" . ", normalize_apostrophes(text or ""))


#: Reasons whose hard_block reads as danger (Everett, 2026-10-29, rulings I
#: and K, and the 2026-10-29 review): harm now or a threat, to the client or
#: to anyone; the client's own danger sentence; the client at risk from
#: themselves, now or their own history ("I used to want to kill myself").
DANGER_REASONS = frozenset({
    "harm_now_to_client", "harm_now_to_other",
    "harm_threat_to_client", "harm_threat_to_other",
    "client_danger",
    "harm_frame_speaker_at_risk", "harm_frame_history_self",
})


def _words_danger(words: SafetyResult, read: str) -> bool:
    """
    Whether the words' own reading is danger. Reads only.

    Danger on the words is a hard_block with a reason in DANGER_REASONS, the
    client's own "not safe", or a danger sentence anywhere in the turn
    (ruling O: the clause is read with all that comes before and after it).

    WHAT CHANGED AND WHY (2026-11-04): this was `_silent_alarm`. On the same
    reading it set `silent_alarm` (a page that was not voiced), `voice_aloud`
    (voiced after all, when the client asked aloud or the context said
    `public_setting` was True) and blanked the hold marker. Whether anything
    is said aloud, and who is reached, are the being's (Everett's 2026-11-04
    ruling, module docstring). The reading is now `danger`.
    """
    if words.status != "hard_block":
        return False
    return (
        words.reason in DANGER_REASONS
        or "client_distress:not safe" in words.flags
        or bool(danger_sentences(read))
    )


#: The client's own state that reads hard_block (ruling A).
CLIENT_DISTRESS_TERMS = frozenset({
    "not okay", "not safe", "no one loves me", "no one cares", "have no one",
})


def harm_frame_safety(text: str, context: Optional[Dict[str, Any]] = None) -> SafetyResult:
    """
    The safety reading of one utterance. Uses Everett's frame. Does not
    rewrite harm_frame.py. Aegis calls it as its safety callback.

    This is the ONLY safety gate on the Corti path. It reads and labels; it
    reaches no one, holds nothing, speaks for no one and asks nothing. What
    is done with the reading is the being's (Everett, 2026-11-04, module
    docstring). Everett, 2026-10-29, on why the labels are what they are:
    "Our position is if someone is in distress: hey, somebody come here and
    help us. We don't claim to be therapists, psychologists, psychiatrists,
    doctors. It's never left in our laps to make a decision on someone's
    life." The labels:

      hard_block     a harm frame whose patient is (or may be) the speaker,
                     not negated, not history, in ANY modality. "I might
                     kill myself tonight" is hypothetical by grammar and a
                     crisis by every other reading.
      hard_block     harm happening now or a threat, to anyone ("he is
                     choking me", "he is going to kill mom"); the client's
                     own danger sentence ("I'm scared for my safety").
      hard_block     nonverbal distress (context["nonverbal_distress"]): a
                     pressed help, pain or medicine symbol, the grunt key, a
                     distress sound, pointing up, Hand Up. At ANY confidence
                     (ruling B; Everett: "A nonverbal client's distress is just as important
                     as a verbal client's.").
      hard_block     the client's OWN distress (ruling A): "I'm not okay",
                     "I don't feel safe", "I'm not safe", "no one cares",
                     "no one loves me", "I have no one", reason
                     "client_distress"; and the client's own history of harm
                     ("I used to want to kill myself"), reason
                     "harm_frame_history_self"; the client's own pain, and
                     the client's need for their medicine.
      soft_redirect  an OBSERVATION about another carbon (ruling D): "she
                     wants to die", "my mom's not okay", "Mary is not okay",
                     "she doesn't feel safe". Someone else by grammar, never
                     by a list (ruling AA): "my mom is not okay" with no
                     contraction reads the person as unclear, and a bare
                     "mom is not okay" from the ear reads the owner as
                     unclear. Everett: "those do not page unless the carbon
                     that stated that requests us to get assistance, because
                     we don't want to step in the middle of a situation where
                     we don't belong."
      hard_block     the same observation WITH the client asking for help
                     in the same turn ("can you get someone", "call
                     someone", "help her", "I need help with this"). The
                     request is named in the flags ("asked_for_help:...")
                     and in `asked_for_help`.
      clear          grief and history about others ("my brother died by
                     suicide"), negated harm ("I would never hurt myself"),
                     a past time word ("yesterday my mom was not okay"), or
                     no harm predicate at all. Flags still carry what was
                     seen; `reason` says which.

    Readings set beside the labels (SafetyResult): `about`, `unclear`,
    `danger`, `danger_sentences`, `asked_aloud`, `asked_for_help`,
    `between_beings`, `short_answer`, `answering`.

    Context keys read: "nonverbal_distress", "previous_turn" (ruling X: a
    fragment is read against the turn before), "reading_words" (ruling AE),
    "between_beings" (exactly True: echoed and flagged; the words are read
    exactly as if a client were there), "answering" (the question the being
    says it asked: echoed; the words are read on their own either way).

    WHAT CHANGED 2026-10-11: the previous gate required `stated_intent`,
    which excluded hypothetical and ability modalities, and returned a bare
    `clear` for every other frame. Measured: "I am thinking about suicide",
    "I might kill myself tonight", "I want to die", "I am suicidal" all
    came back clear. Everett's MUST-CALL / MUST-NOT-CALL table is pinned in
    tests/test_harm_frame_gate.py and this function is written to it.

    WHAT CHANGED AND WHY (2026-11-04): between two beings (ruling F, refined)
    every hard_block other than harm now or a threat was turned to
    soft_redirect and flagged "would_page:<reason>", and an ask for help was
    not read; `answering_follow_up` read a "no" to the package's own
    ask_if_safe mark as hard_block "answered_not_safe" (and a "yes" to
    ask_if_help_needed as "answered_yes_help"); `public_setting` decided
    whether danger was voiced. Each was the package deciding (Everett's
    2026-11-04 ruling, module docstring). The words are now read alone.
    """
    context = context or {}
    # Ruling AE: one client's reading words (ReadingWords.for_client) reach
    # the gate, the clause reader and harm_frame together.
    with using_reading_words(context.get("reading_words")):
        return _harm_frame_safety(text, context)


def _harm_frame_safety(text: str, context: Dict[str, Any]) -> SafetyResult:
    original = text or ""
    # The reading is made without the ask to say it aloud; the ask is read on
    # its own (asked_aloud, from `original`).
    read = without_aloud(original)
    words = _words_gate(read, str(context.get("previous_turn") or ""))
    words = replace(words, said=original)
    danger = _words_danger(words, read)
    result, nonverbal_danger = _with_nonverbal(words, context)

    between = context.get("between_beings") is True
    answering = context.get("answering")
    return replace(
        result,
        flags=list(result.flags) + (["between_beings"] if between else []),
        danger=danger or nonverbal_danger,
        danger_sentences=danger_sentences_as_said(original),
        asked_aloud=asked_aloud(original),
        asked_for_help=asked_for_help(read),
        between_beings=between,
        short_answer=short_answer(read),
        answering=None if answering in (None, "") else str(answering),
    )


def _with_nonverbal(words: SafetyResult,
                    context: Dict[str, Any]) -> Tuple[SafetyResult, bool]:
    """
    Add the nonverbal distress readings to the words' reading. Returns the
    result and whether any nonverbal reading's meaning is danger.

    Everett, 2026-10-29: the acoustic key (and every nonverbal signal) must
    never disrupt the interpretation of what is spoken. The words were read
    alone. A nonverbal distress reading may only ADD: when the words already
    read hard_block, their reading stands and the signal is added to the
    flags; when they do not, the signal reads hard_block ("nonverbal_distress"),
    and what the words were read as is kept in the flags beside it.

    WHAT CHANGED AND WHY (2026-11-04): a nonverbal danger reading also set
    the silent alarm here (not voiced unless asked aloud or `public_setting`
    was True) and chose the hold marker. Those were the package deciding
    (module docstring); the reading is now `danger`.
    """
    nonverbal = [d for d in (context.get("nonverbal_distress") or [])
                 if isinstance(d, dict)]
    if not nonverbal:
        return words, False
    danger = any(d.get("danger") is True for d in nonverbal)
    nonverbal_flags = [
        f"nonverbal:{d.get('modality')}:{d.get('input')}:{d.get('intent')}"
        for d in nonverbal
    ]
    if words.status == "hard_block":
        return replace(words, flags=list(words.flags) + nonverbal_flags), danger
    inputs = [str(d.get("input")) for d in nonverbal]
    return replace(
        words,
        status="hard_block",
        reason="nonverbal_distress",
        flags=nonverbal_flags + list(words.flags)
        + [f"words:{words.status}:{words.reason}"],
        flagged_terms=inputs + list(words.flagged_terms),
    ), danger


def _words_gate(original: str, previous_turn: str = "") -> SafetyResult:
    """The safety reading of the words alone. No nonverbal signal reaches
    here, and nothing about who is listening changes it. `previous_turn`
    (ruling X) anchors a subjectless fragment ("it hurts")."""
    frames = extract_frames(original)
    harms = body_harm(original)
    # Everett, 2026-10-29. Every harm verb aimed at someone is read for its
    # time, and the target is found by grammar (no person list):
    #   - a threat (going to, will, about to) against ANYONE or any noun is
    #     an absolute threat: hard_block, flagged threat_to_life and
    #     emergency ("He is going to kill mom. That's an absolute threat. So
    #     it doesn't matter who is involved");
    #   - harm happening now to anyone: hard_block, flagged emergency;
    #   - past harm to the client (ruling H: "he choked me", "she stabbed me
    #     last week") reads soft_redirect, "because they may be in the
    #     vicinity" is the being's to weigh, and hard_block with a request
    #     for help in the same turn;
    #   - past harm to someone else stays ruling D, below.
    # Self-harm (ruling A) is not moved: the client acting on the client.
    threats: List[Tuple[HarmFrame, str]] = []
    now_other: List[HarmFrame] = []
    timed: List[HarmFrame] = []
    for frame in frames:
        if frame.shape != "verb" or frame.polarity is not Polarity.HARM or not frame.predicate:
            continue
        when, said = harm_time(original, frame)
        if frame.negated and not (when == "now" and _WONT_STOP.search(
                normalize_apostrophes(original).lower())):
            continue
        target = harm_target(frame, frame.subject is Party.SPEAKER)
        if target is None or (target == "client" and frame.subject is Party.SPEAKER):
            continue
        if target.startswith("subject_"):
            # "she's choking", "I'm drowning": only harm happening now is
            # read this way; "he choked" with no object is not a report.
            if when != "now":
                continue
            target = target[len("subject_"):]
        if when == "threat":
            threats.append((frame, target))
        elif target == "thing":
            continue
        elif target == "client":
            timed.append(frame)
            if not any(_stem(h.verb) == frame.predicate for h in harms):
                harms.append(BodyHarm(said.split()[0] if said else frame.predicate, "",
                                      when == "now", said or frame.predicate))
        elif when == "now":
            now_other.append(frame)
    routed = {id(f) for f in timed} | {id(f) for f, _ in threats} | {id(f) for f in now_other}
    frames = [f for f in frames if id(f) not in routed]
    harm_now = [h for h in harms if h.now]
    harm_past = [h for h in harms if not h.now]
    harm_flags = [f"harm_{'now' if h.now else 'reported'}:{h.said}" for h in harms]
    # The clause reader, not a substring. A negated state ("she is not
    # unsafe"), a past one ("yesterday she wasn't okay"), and "you" or a
    # thing as the subject are left out (see structural_affect).
    clause_reading = (_CLAUSE_READER.analyze(original, previous_turn=previous_turn)
                      if original else None)
    hits = clause_reading.hits if clause_reading is not None else []
    states_now = [h for h in hits if h.when == "now" and h.final_weight < 0]
    own_state = [h for h in states_now
                 if h.term in CLIENT_DISTRESS_TERMS and h.attribution is Attribution.SELF]
    other_not_okay = [h for h in states_now
                      if h.term in ("not okay", "not safe") and h.attribution is Attribution.OTHER]
    # Someone else's pain is ruling D (Everett, 2026-10-29): "my mom's in
    # pain" does not page unless the client asks for help in the turn.
    # "hurt" with an object ("he hurt my feelings", "I hurt my arm") is the
    # verb, read by the harm frames, not the feeling.
    pain_as_verb = {f.predicate for f in extract_frames(original)
                    if f.shape == "verb" and f.patient_text}
    other_not_okay += [h for h in states_now
                       if h.term in PAIN_TERMS and h.attribution is Attribution.OTHER
                       and _stem(h.term) not in pain_as_verb]
    # Ruling U: "I think she's not okay" is the client's belief about her,
    # not a fact about her. Nothing is concluded about her, but the client
    # said it, so the being follows up the same way as an observation
    # (ruling D: no page unless the client asks for help).
    other_not_okay += [h for h in hits
                       if h.belief_held_by and h.belief_about != "self"
                       and h.term in ("not okay", "not safe") and not h.belief_negated
                       and h.when == "now"]
    crisis: List[HarmFrame] = []
    at_risk_other: List[HarmFrame] = []
    other_not_person: List[HarmFrame] = []
    history_self: List[HarmFrame] = []
    history_other: List[HarmFrame] = []
    negated: List[HarmFrame] = []
    subject_unclear: List[HarmFrame] = []
    for frame in frames:
        if not frame.identified["predicate"]:
            continue
        if frame.polarity is not Polarity.HARM:
            continue
        if frame.negated:
            negated.append(frame)
            continue
        if frame.subject_unresolved:
            # "mom wants to die": the grammar says it is not the client and
            # cannot say who (no person list, Everett 2026-10-29). Never the
            # client's own risk; the being asks who.
            subject_unclear.append(frame)
            continue
        if frame.past_tense:
            if (frame.shape == "verb" and frame.patient is Party.THIRD_PARTY
                    and harm_target(frame, frame.subject is Party.SPEAKER) == "other"):
                # "he was hitting her": past harm to someone else is ruling D
                # (Everett, 2026-10-29): recorded, flagged, the being asks if
                # help is needed, and it pages when the client asks for help.
                at_risk_other.append(frame)
                continue
            (history_self if frame.speaker_may_be_patient else history_other).append(frame)
            continue
        if frame.speaker_may_be_patient:
            crisis.append(frame)
        elif frame.patient is Party.THIRD_PARTY:
            post = (getattr(frame, "post", "") or "").split()
            first = post[0].strip(",.;:!?") if post else ""
            if first in _NOT_A_PERSON:
                other_not_person.append(frame)
            elif first in current_words().body_parts:
                # "he hurt my ARM" is the client's own body, read by
                # body_harm above, not someone else at risk.
                continue
            else:
                at_risk_other.append(frame)

    def _flags(prefix: str, items: List[HarmFrame]) -> List[str]:
        out: List[str] = []
        for f in items:
            out += [f"{prefix}:{f.predicate}", f"patient:{f.patient.value}",
                    f"subject:{f.subject.value}", f"shape:{f.shape}"]
        return out

    other_flags = (_flags("other", at_risk_other)
                   + [f"other_not_okay:{h.term}" for h in other_not_okay])
    own_flags = [f"client_distress:{h.term}" for h in own_state]

    if crisis:
        return SafetyResult(
            status="hard_block",
            reason="harm_frame_speaker_at_risk",
            said=original,
            flags=_flags("hard", crisis) + own_flags + other_flags,
            flagged_terms=[f.predicate for f in crisis],
        )

    if threats:
        to_client = any(t == "client" for _, t in threats)
        return SafetyResult(
            status="hard_block",
            reason="harm_threat_to_client" if to_client else "harm_threat_to_other",
            said=original,
            flags=["threat_to_life", "emergency"]
            + [f"threat:{f.predicate}:{f.patient_text or f.patient.value}" for f, _ in threats]
            + _flags("threat", [f for f, _ in threats]) + own_flags + other_flags,
            flagged_terms=[f.predicate for f, _ in threats],
        )

    if now_other and not harm_now:
        return SafetyResult(
            status="hard_block",
            reason="harm_now_to_other",
            said=original,
            flags=["emergency"] + _flags("now", now_other) + harm_flags + own_flags + other_flags,
            flagged_terms=[f.predicate for f in now_other],
        )

    if harm_now:
        # Everett, 2026-10-29: harm happening NOW to the client is "immediate
        # carbon notification": hard_block, at any confidence.
        return SafetyResult(
            status="hard_block",
            reason="harm_now_to_client",
            said=original,
            flags=["emergency"] + harm_flags + own_flags + other_flags,
            flagged_terms=[h.verb for h in harm_now],
        )

    danger = danger_sentences(original)
    if danger:
        return SafetyResult(
            status="hard_block",
            reason="client_danger",
            said=original,
            flags=[f"danger:{d}" for d in danger] + harm_flags + own_flags + other_flags,
            flagged_terms=danger,
        )

    if own_state or history_self:
        terms = [h.term for h in own_state] + [f.predicate for f in history_self]
        flags = own_flags + _flags("history", history_self) + other_flags
        return SafetyResult(
            status="hard_block",
            reason="client_distress" if own_state else "harm_frame_history_self",
            said=original,
            flags=flags,
            flagged_terms=terms,
        )

    # MEDICINE first: "I need my pain meds" is the need, not the feeling.
    medicine = _NEEDS_MEDICINE.search(normalize_apostrophes(original).lower())
    if medicine and not _negated_need(original, medicine.start()):
        return SafetyResult(
            status="hard_block",
            reason="client_needs_medicine",
            said=original,
            flags=["need:medicine", f"need_said:{medicine.group(0)}"] + other_flags,
            flagged_terms=[medicine.group(0)],
        )

    # PAIN: the client's own, now, not read against the being's question
    # ("is something wrong with your leg?" / "yeah, it hurts" is engaging,
    # Everett: "that's not a page, that's engaging"; see the anchored
    # branch below). hard_block every time; the body part is `about`.
    own_pain = [h for h in states_now
                if h.term in PAIN_TERMS and h.attribution is Attribution.SELF
                and not h.anchored_to and not h.negated
                and _stem(h.term) not in pain_as_verb]
    if own_pain:
        where: List[str] = []
        for h in own_pain:
            if clause_reading is not None and h.clause_index < len(clause_reading.clauses):
                where += [t for t in clause_reading.clauses[h.clause_index].tokens
                          if t in current_words().body_parts and t not in where]
        return SafetyResult(
            status="hard_block",
            reason="client_pain",
            said=original,
            flags=["feeling:pain"] + [f"pain_where:{w}" for w in where]
            + harm_flags + other_flags,
            flagged_terms=[h.term for h in own_pain],
            about=" ".join(where) or None,
        )

    need = _OTHER_NEED.search(normalize_apostrophes(original).lower())
    if need and not _negated_need(original, need.start()):
        request = asked_for_help(original)
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason="client_need_asked_for_help",
                said=original,
                flags=[f"need:{need.group(0)}", f"asked_for_help:{request}"] + other_flags,
                flagged_terms=[need.group(0)],
            )

    if harm_past:
        # Everett, 2026-10-29 (ruling H): a report of past physical harm to
        # the client ("he hurt my arm") reads soft_redirect on its own, and
        # hard_block when the client asks for help in the same turn. The
        # harm words are `about`.
        request = asked_for_help(original)
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason="harm_reported_to_client",
                said=original,
                flags=harm_flags + other_flags + [f"asked_for_help:{request}"],
                flagged_terms=[h.verb for h in harm_past],
                about=harm_past[0].said,
            )
        return SafetyResult(
            status="soft_redirect",
            reason="harm_reported_to_client",
            said=original,
            flags=harm_flags + other_flags,
            flagged_terms=[h.verb for h in harm_past],
            about=harm_past[0].said,
        )

    if subject_unclear and not (at_risk_other or other_not_okay):
        terms = [f.predicate for f in subject_unclear]
        request = asked_for_help(original)
        flags = _flags("unclear", subject_unclear) + [
            f"subject:{f.subject_text}" for f in subject_unclear if f.subject_text]
        # "mom wants to die": not the client, and the grammar cannot say who.
        who = subject_unclear[0].subject_text or subject_unclear[0].predicate
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason="harm_frame_subject_unclear",
                said=original,
                flags=flags + [f"asked_for_help:{request}"],
                flagged_terms=terms,
                unclear="who",
                about=who,
            )
        return SafetyResult(
            status="soft_redirect",
            reason="harm_frame_subject_unclear",
            said=original,
            flags=flags,
            flagged_terms=terms,
            unclear="who",
            about=who,
        )

    if at_risk_other or other_not_okay:
        reason = ("harm_frame_third_party_at_risk" if at_risk_other
                  else "other_person_not_okay")
        terms = [f.predicate for f in at_risk_other] + [h.term for h in other_not_okay]
        request = asked_for_help(original)
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason=reason,
                said=original,
                flags=other_flags + [f"asked_for_help:{request}"],
                flagged_terms=terms,
                about=terms[0],
            )
        return SafetyResult(
            status="soft_redirect",
            reason=reason,
            said=original,
            flags=other_flags,
            flagged_terms=terms,
            about=terms[0],
        )

    reading_for_owner = (_CLAUSE_READER.analyze(original, previous_turn=previous_turn)
                         if original else None)
    unclear = whose_feeling_unclear(reading_for_owner) if reading_for_owner else []
    if unclear and not between_beings:
        # Ruling P: whose feeling it is cannot be told from the words.
        # Nothing is concluded; the being asks, e.g. "Who's scared?". If the
        # client asked for help in the same turn, a human is reached all the
        # same, as for an unclear person (ruling AA) and an observation
        # (ruling D). Before the 2026-10-29 review the person list made
        # "mom is not okay can you get someone" an observation that paged;
        # with the list gone (ruling AA) it is whose-unclear, and it still
        # pages.
        request = asked_for_help(original)
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason="whose_feeling_unclear",
                said=original,
                flags=[f"whose:{t}" for t in unclear] + other_flags
                + [f"asked_for_help:{request}"],
                flagged_terms=unclear,
                hold_marker=HOLD_ASKED_FOR_HELP,
            )
        return SafetyResult(
            status="soft_redirect",
            reason="whose_feeling_unclear",
            said=original,
            flags=[f"whose:{t}" for t in unclear] + other_flags + ["follow_up:ask_whose"],
            flagged_terms=unclear,
            follow_up=ASK_WHOSE,
            follow_up_about=unclear[0],
        )

    unsure = person_unclear(reading_for_owner) if reading_for_owner else []
    if unsure and between_beings:
        # Ruling F: two beings talking, nothing pages. Ruling AA left "my mom
        # is not okay" as "who?", where the person list used to make it an
        # observation; between beings it is recorded and flagged the same
        # way an observation is, and nothing is asked.
        return SafetyResult(
            status="soft_redirect",
            reason="person_unclear",
            said=original,
            flags=[f"unclear_who:{w}" for _, w in unsure] + other_flags + ["between_beings"],
            flagged_terms=[t for t, _ in unsure],
        )
    if unsure:
        # Ruling AA: "my teacher is sad" — whether "teacher" is a person in
        # this client's life is learned per client, not read off a list.
        # Nothing is concluded; the being asks. If the client asked for help
        # in the same turn, a human is reached all the same.
        request = asked_for_help(original)
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason="person_unclear",
                said=original,
                flags=[f"unclear_who:{w}" for _, w in unsure] + other_flags
                + [f"asked_for_help:{request}"],
                flagged_terms=[t for t, _ in unsure],
                hold_marker=HOLD_ASKED_FOR_HELP,
            )
        return SafetyResult(
            status="soft_redirect",
            reason="person_unclear",
            said=original,
            flags=[f"unclear_who:{w}" for _, w in unsure] + other_flags + ["follow_up:ask_who"],
            flagged_terms=[t for t, _ in unsure],
            follow_up=ASK_WHO,
            follow_up_about=unsure[0][1],
        )

    loose = reading_for_owner.unanchored_fragments if reading_for_owner else []
    if loose and not between_beings:
        # Ruling X: "it hurts", "this is scary", "everything hurts" are
        # usually answers to something said to the client. With no turn
        # before to anchor it, it is unclear; the being asks.
        return SafetyResult(
            status="soft_redirect",
            reason="fragment_unclear",
            said=original,
            flags=[f"fragment:{t}" for t in loose] + other_flags + ["follow_up:ask_what"],
            flagged_terms=loose,
            follow_up=ASK_WHAT,
            follow_up_about=loose[0],
        )

    thanked = any(h.term in THANKS for h in hits)
    felt = [h.term for h in hits
            if h.base_weight < 0 and h.final_weight < 0 and h.when == "now"
            and h.attribution is Attribution.SELF]
    if (thanked and felt and not between_beings and reading_for_owner is not None
            and reading_for_owner.certainty is AffectCertainty.AMBIGUOUS):
        # Ruling W: "thank you, I'm scared" stays unclear (main's reading).
        # Nothing is concluded; the being answers the thanks and asks.
        return SafetyResult(
            status="soft_redirect",
            reason="thanks_unclear",
            said=original,
            flags=[f"unclear:{t}" for t in felt] + other_flags
            + ["thanked", "follow_up:ask_what_about"],
            flagged_terms=felt,
            follow_up=ASK_WHAT_ABOUT,
            follow_up_about=felt[0],
        )

    own_now = [h.term for h in hits
               if h.attribution is Attribution.SELF and h.when == "now"
               and h.final_weight < 0 and not h.past_tense]
    if own_now and not between_beings and _WHY_SELF.search(
            normalize_apostrophes(original).lower()):
        # Ruling Y: "Why am I so sad?" is the speaker's own sadness (the
        # subject after the verb), and someone reaching out. The being
        # engages: "What's making you so sad?". No page on its own.
        return SafetyResult(
            status="soft_redirect",
            reason="reaching_out",
            said=original,
            flags=[f"reaching_out:{t}" for t in own_now] + other_flags + ["follow_up:ask_what"],
            flagged_terms=own_now,
            follow_up=ASK_WHAT_IS_MAKING,
            follow_up_about=own_now[0],
        )

    worse = [h.term for h in hits
             if h.term in _FEAR_TERMS and h.attribution is Attribution.SELF
             and h.when == "now" and h.final_weight < 0 and not h.past_tense]
    if worse and not between_beings and any("escalation" in h.note for h in hits):
        # Ruling Z: "I was scared and now I'm terrified" is fear present now
        # and getting worse, read from the whole sentence. Being terrified
        # is not by itself a danger signal: danger and the silent alarm come
        # from context (rulings I, K, O). The being is inquisitive.
        return SafetyResult(
            status="soft_redirect",
            reason="fear_getting_worse",
            said=original,
            flags=[f"fear:{t}" for t in worse] + other_flags + ["follow_up:ask_what"],
            flagged_terms=worse,
            follow_up=ASK_WHAT_OF,
            follow_up_about=worse[-1],
        )

    fear = fear_without_context(original, hits)
    if fear and not between_beings:
        # Ruling O: not enough to tell. Nothing is concluded and nothing is
        # spoken for the client; the being asks. Rulings W and Z (Everett,
        # 2026-10-29): it asks what the fear is about ("What are you scared
        # about?"), not whether the client is safe. Danger in the turn has
        # already paged above (ruling O); this is fear with no danger in it.
        return SafetyResult(
            status="soft_redirect",
            reason="fear_without_context",
            said=original,
            flags=[f"fear:{t}" for t in fear] + other_flags + ["follow_up:ask_what_about"],
            flagged_terms=fear,
            follow_up=ASK_WHAT_ABOUT,
            follow_up_about=fear[0],
        )

    inability = reading_for_owner.inability if reading_for_owner else []
    if inability and not between_beings:
        # Ruling R: "I can't be happy" is an inability and a signal, not the
        # feeling and not its absence. Nothing is concluded; the being asks
        # why. It does not page on its own; everything above still applies.
        return SafetyResult(
            status="soft_redirect",
            reason="inability_signal",
            said=original,
            flags=[f"inability:{t}" for t in inability] + other_flags + ["follow_up:ask_why"],
            flagged_terms=inability,
            follow_up=ASK_WHY_CANT,
            follow_up_about=inability[0],
        )

    judged = reading_for_owner.judged if reading_for_owner else []
    if judged and not between_beings:
        # Ruling T: "I shouldn't be this angry" — the anger is there, the
        # speaker's own; the being asks "Why are you so angry?".
        return SafetyResult(
            status="soft_redirect",
            reason="feeling_judged",
            said=original,
            flags=[f"judged:{t}" for t in judged] + other_flags + ["follow_up:ask_why"],
            flagged_terms=judged,
            follow_up=ASK_WHY,
            follow_up_about=judged[0],
        )

    alone = [h.term for h in hits
             if h.term in _ALONE_TERMS and h.attribution is Attribution.SELF
             and h.when == "now" and h.final_weight < 0]
    if alone and not between_beings:
        # Ruling V: "I'm not happy, I'm alone" is an opening. The being is
        # there and listening, so the client is not alone in that moment; it
        # asks a simple question or offers something to do. Asked, never
        # told. No page on its own; everything above still applies.
        return SafetyResult(
            status="soft_redirect",
            reason="alone_opening",
            said=original,
            flags=[f"alone:{t}" for t in alone] + other_flags + ["follow_up:offer"],
            flagged_terms=alone,
            follow_up=ASK_OR_OFFER,
            follow_up_about=alone[0],
        )

    wish = [h.term for h in hits
            if h.volitional and h.attribution is Attribution.SELF
            and h.when == "now" and h.base_weight < 0]
    if wish and not between_beings:
        # Ruling V: "I feel like being alone" is a stated wish, the client's
        # own choice, and it is respected. The being may ask a simple
        # question, but does not push: no question is written for it here.
        return SafetyResult(
            status="clear",
            reason="stated_wish",
            said=original,
            flags=[f"wish:{t}" for t in wish] + other_flags,
        )

    anchored = [h for h in hits if h.anchored_to and h.when == "now"]
    if anchored and not between_beings:
        # Ruling X: a fragment read against the turn before it. "Did you like
        # that haunted house?" / "Yeah, this is scary" is about the haunted
        # house, not danger. "Is there something wrong with your leg?" /
        # "Yeah, it hurts" is the client's own leg: pain, a need.
        own_body = [h for h in anchored if h.anchor_kind == "self"]
        return SafetyResult(
            status="clear",
            reason="answer_about_own_body" if own_body else "answer_about_thing",
            said=original,
            flags=[f"about:{h.anchored_to}" for h in anchored]
            + [f"need:{h.term}" for h in own_body if h.final_weight < 0]
            + [f"own_pain:{h.term}" for h in own_body
               if h.term in PAIN_TERMS and h.final_weight < 0] + other_flags,
        )

    if other_not_person:
        return SafetyResult(
            status="soft_redirect",
            reason="harm_frame_third_party_not_a_person",
            said=original,
            flags=_flags("soft", other_not_person),
            flagged_terms=[f.predicate for f in other_not_person],
        )

    if history_other:
        return SafetyResult(
            status="clear",
            reason="harm_frame_history_third_party",
            said=original,
            flags=_flags("history", history_other),
        )

    if negated:
        return SafetyResult(
            status="clear",
            reason="harm_frame_negated",
            said=original,
            flags=_flags("negated", negated),
        )

    bounded = reading_for_owner.negated_for_a_time if reading_for_owner else []
    if bounded and not between_beings:
        # Ruling T: "I haven't been scared since Monday" is no fear since
        # Monday: a statement, nothing concluded. The being MAY ask a plain
        # question about the time the client named.
        named = next((t for _, t in bounded if t), "")
        return SafetyResult(
            status="clear",
            reason="feeling_negated_for_a_time",
            said=original,
            flags=[f"negated_for_a_time:{term}:{t}" for term, t in bounded]
            + (["follow_up:may_ask_what_happened"] if named else []),
            follow_up=ASK_WHAT_HAPPENED if named else "",
            follow_up_about=named,
        )

    return SafetyResult(status="clear", reason="no_harm_frame", said=original)


@dataclass(frozen=True)
class HearingTurn:
    """One utterance, heard and (if there is text) read. Not spoken."""

    ingest: Optional[CortiIngest]
    prosody: Optional[ProsodyReading]
    affect: Optional[AffectReading]
    frames: List[HarmFrame] = field(default_factory=list)
    face: Optional[FaceReading] = None
    decision: Optional[Decision] = None
    source: str = "corti"
    notes: List[str] = field(default_factory=list)
    #: The client's acoustic key, looked up. None when no key was sent.
    acoustic: Optional[Interpretation] = None
    #: nonverbal_engine.process_multimodal_input result, or None.
    nonverbal: Optional[Dict[str, Any]] = None
    #: engine_temporal.process_multimodal_sequence result, or None.
    temporal: Optional[Dict[str, Any]] = None
    #: Every reading that went to the crisis gate as nonverbal distress.
    nonverbal_distress: List[Dict[str, Any]] = field(default_factory=list)
    #: The Soul Mirror's 8-slot vector, with its measured mask and clauses.
    face_vector: Optional[FaceVector] = None
    #: The client's earlier danger sentence, exactly as said, released to be
    #: voiced because the client now asked for it out loud (ruling K).
    voice_held: Optional[str] = None
    #: The client's own words for a nonverbal danger signal (a pressed help
    #: symbol, a meaning learned for that client), voiced because the client
    #: asked for it aloud or the setting is public (rulings K and L). None
    #: otherwise: under the silent alarm nothing is said.
    signal_voiced: Optional[str] = None

    @property
    def person_message(self) -> Optional[str]:
        """
        What the PERSON said through a nonverbal channel, from the map — or
        None. Never the being's reply, and never held by a safety hold.

        Under the silent alarm (ruling I) it is None: a danger meaning from
        any channel is paged and never spoken.
        """
        if self.signal_voiced:
            return self.signal_voiced
        if self.decision is not None and self.decision.safety.silent_alarm \
                and self.nonverbal_distress and any(d.get("silent_alarm") for d in self.nonverbal_distress):
            return None
        if self.nonverbal and self.nonverbal.get("speakable"):
            return self.nonverbal.get("message")
        if self.acoustic is not None and self.acoustic.speakable:
            return self.acoustic.message
        if self.temporal and self.temporal.get("message"):
            return self.temporal.get("message")
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "client_kind_excluded": True,
            "ingest": None if self.ingest is None else self.ingest.to_dict(),
            "prosody": None if self.prosody is None else self.prosody.to_dict(),
            "affect": None if self.affect is None else self.affect.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "face": None if self.face is None else self.face.to_dict(),
            "face_vector": None if self.face_vector is None else self.face_vector.to_dict(),
            "decision": None if self.decision is None else self.decision.to_dict(),
            "acoustic": None if self.acoustic is None else self.acoustic.to_dict(),
            "nonverbal": self.nonverbal,
            "temporal": self.temporal,
            "nonverbal_distress": list(self.nonverbal_distress),
            "person_message": self.person_message,
            "voice_held": self.voice_held,
            "signal_voiced": self.signal_voiced,
            "notes": list(self.notes),
        }


#: The nonverbal inputs Hearing accepts, by name.
NONVERBAL_INPUTS = ("gesture", "symbol", "eye_data", "sound")


def _with_client_words(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a Hearing method with that client's reading words in force (ruling
    AE), keeping the method's own signature."""
    @functools.wraps(method)
    def wrapper(self: "Hearing", *args: Any, **kwargs: Any) -> Any:
        with using_reading_words(self.reading_words):
            return method(self, *args, **kwargs)
    return wrapper


class Hearing:
    """
    Corti in, Decision out.

    Pass a crisis_callback that returns True only when a human was reached.
    Detection without delivery raises CrisisNotDelivered — same law as Aegis.
    """

    def __init__(
        self,
        crisis_callback: Optional[Callable[[SafetyResult, Dict[str, Any]], bool]] = None,
        nonverbal_engine: Optional[NonverbalEngine] = None,
        temporal_engine: Optional[TemporalNonverbalEngine] = None,
        baseline: Optional[SpeakerBaseline] = None,
        reading_words: Optional[ReadingWords] = None,
    ) -> None:
        """
        Args:
            crisis_callback: returns True only when a human was reached.
            nonverbal_engine: the map for gestures, symbols, eye regions and
                sounds. Default: the built-in map; reads nothing from disk
                unless its data_dir holds learned maps.
            temporal_engine: trained sequence models. Default: none loaded,
                so every sequence reports `unavailable` and says why.
            baseline: this speaker's measured baseline (see
                `prosody.build_speaker_baseline`). Without one, prosody is
                read against the recent window and says so.
            reading_words: this client's reading words
                (SHARED_READING_WORDS.for_client(learned)): negators, scope
                breakers and body parts the being has learned for this
                client. Every reader on the turn uses them (ruling AE).
                Default: the shared set.
        """
        self.reading_words = reading_words
        self.prosody = ProsodyAnalyzer(baseline=baseline)
        self.affect = StructuralAffectAnalyzer(reading_words=reading_words)
        self.nonverbal = nonverbal_engine or NonverbalEngine(data_dir=None)
        self.temporal = temporal_engine or TemporalNonverbalEngine()
        self.fusion = FusionEngine(
            analyzer=self.affect,
            safety_callback=harm_frame_safety,
            crisis_callback=crisis_callback,
        )
        #: The question the being asked on the last turn with words, if any.
        #: The client's next words are read as the answer to it, once.
        self.awaiting_answer: Optional[str] = None
        #: A danger sentence the client typed or said that was paged and not
        #: voiced (the silent alarm). If the client's next words ask for it
        #: to be said out loud, it is handed back, exactly, to be voiced
        #: (ruling K). Held for one turn only.
        self.held_words: Optional[str] = None

    @_with_client_words
    def hear(
        self,
        corti_event: Optional[Any] = None,
        text: Optional[str] = None,
        nonverbal: Optional[Dict[str, Any]] = None,
        acoustic_key: Optional[str] = None,
        temporal: Optional[Dict[str, Any]] = None,
        between_beings: bool = False,
        previous_turn: Optional[str] = None,
        public_setting: Optional[bool] = None,
    ) -> HearingTurn:
        """
        Consume one Corti emit, optional transcript, and optional nonverbal
        input.

        Args:
            corti_event: one closed Corti VocalEvent card.
            text: what the ear transcribed, or what was typed.
            nonverbal: any of gesture=, symbol=, eye_data={"region": ...},
                sound= — keys in the nonverbal map.
            acoustic_key: the Corti client's measured key (grunt_distress,
                grunt_acknowledge, grunt_frustration, vocal_stimming).
            temporal: gesture_features= / eye_features= / emotion_features=
                frames for the trained sequence models.
            between_beings: True only when two beings are talking to each
                other and no client is in the conversation. Then someone
                else at risk is flagged, not paged. False by default: a
                client is always assumed to be there.
            previous_turn: what was said to the client just before this
                turn, including the being's own question (Everett,
                2026-10-29, ruling X). A subjectless fragment ("it hurts",
                "this is scary") is read against it, never alone. The
                being's question is passed here in the words the being
                used; without it, the fragment is unclear and the being
                asks.
            public_setting: the being's own perception of where the client
                is (Everett, ruling K). A danger sentence the client typed
                or said is the silent alarm: paged, not voiced. It is voiced
                only when the client asks for it aloud, or when this is
                exactly True. None (unknown), False, or any other value:
                silent. Carried into the fusion engine's live_state and so
                into harm_frame_safety's context.

        Every argument may be missing. Missing is reported as missing.
        Raises CrisisNotDelivered when distress was read and no human could
        be reached.
        """
        notes: List[str] = []
        voice_held: Optional[str] = None
        taken: Optional[CortiIngest] = None
        reading: Optional[ProsodyReading] = None

        if corti_event is None:
            notes.append("no Corti emit — this is not hearing from the ear")
        else:
            taken = ingest(corti_event)
            reading = self.prosody.analyze(taken.event)
            notes.append(
                "Corti kind excluded — client label, not family Sound"
            )

        # -- Nonverbal: the person's own channels ---------------------------
        acoustic: Optional[Interpretation] = None
        if acoustic_key is not None:
            acoustic = deliver_acoustic_key(acoustic_key, engine=self.nonverbal)
            if acoustic.status.value == "unrecognized":
                notes.append(f"acoustic key {acoustic_key!r} not a live key — silent")

        multimodal: Optional[Dict[str, Any]] = None
        if nonverbal:
            unknown = sorted(set(nonverbal) - set(NONVERBAL_INPUTS))
            if unknown:
                raise ValueError(
                    f"unknown nonverbal input(s) {unknown}; expected any of "
                    f"{list(NONVERBAL_INPUTS)}"
                )
            multimodal = self.nonverbal.process_multimodal_input(
                **{k: nonverbal.get(k) for k in NONVERBAL_INPUTS}
            )

        sequences: Optional[Dict[str, Any]] = None
        if temporal:
            sequences = self.temporal.process_multimodal_sequence(**temporal)
            if sequences.get("status") != "ok":
                notes.append("temporal: no usable reading — "
                             + "; ".join(sorted(set(self.temporal.load_errors.values())))
                             if self.temporal.load_errors else
                             "temporal: no usable reading")

        distress: List[Dict[str, Any]] = []
        if acoustic is not None:
            distress += [r.to_dict() for r in distress_readings([acoustic])]
        if multimodal is not None:
            distress += list(multimodal.get("distress") or [])
            if text and asked_for_help(text) is not None:
                # A pressed need (food, drink, bathroom) with a request for
                # help in the same turn pages, the same as the typed need
                # (ruling C; Everett, 2026-10-29).
                distress += [r for r in (multimodal.get("all_results") or [])
                             if isinstance(r, dict) and r.get("modality") == "symbol"
                             and r.get("intent") in NEED_INTENTS and r.get("status") == "ok"]
        if sequences is not None:
            for result in (sequences.get("all_results") or {}).values():
                # Hand Up pages (Everett, 2026-10-29), as do the distress intents.
                if (result.get("pages") or result.get("intent") in DISTRESS_INTENTS
                        or result.get("silent_alarm")) \
                        and result.get("status") in ("ok", "below_threshold"):
                    distress.append({"modality": f"temporal:{result.get('modality')}",
                                     "input": result.get("expression"),
                                     "intent": result.get("intent"),
                                     "message": result.get("message"),
                                     "confidence": result.get("confidence"),
                                     "silent_alarm": bool(result.get("silent_alarm"))})

        live_state: Dict[str, Any] = {}
        if between_beings:
            live_state["between_beings"] = True
        if public_setting is not None:
            # Passed as given; harm_frame_safety voices only on exactly True.
            live_state["public_setting"] = public_setting
        if acoustic is not None or multimodal is not None or sequences is not None:
            live_state["nonverbal_active"] = True
        if distress:
            live_state["nonverbal_distress"] = distress

        # -- Words -----------------------------------------------------------
        frames: List[HarmFrame] = []
        affect: Optional[AffectReading] = None
        decision: Optional[Decision] = None
        if text:
            frames = extract_frames(text)
            before = previous_turn or ""
            if before:
                live_state["previous_turn"] = before
            affect = self.affect.analyze(text, previous_turn=before)
            if self.awaiting_answer:
                live_state["answering_follow_up"] = self.awaiting_answer
            if self.held_words is not None and asked_aloud(text):
                voice_held = self.held_words
            decision = self.fusion.fuse(text, live_state=live_state, prosody=reading)
            self.awaiting_answer = decision.safety.follow_up or None
            self.held_words = (text if decision.safety.silent_alarm
                               and not decision.safety.voice_aloud else None)
        else:
            notes.append("no text — structural_affect and harm_frame not run")
            if distress:
                # No words, and a person signalled they need someone. The
                # gate runs on the signal alone.
                decision = self.fusion.fuse("", live_state=live_state, prosody=reading)

        # Rulings K and L: a nonverbal danger signal is voiced, in the
        # client's own words for it, when asked aloud or in public; held for
        # one turn otherwise, the same as a typed danger sentence.
        signal_voiced: Optional[str] = None
        if decision is not None and decision.safety.silent_alarm and distress:
            signal_words = self._signal_words(distress)
            if signal_words and decision.safety.voice_aloud:
                signal_voiced = signal_words
            elif signal_words and not text:
                self.held_words = signal_words

        face = build_face_reading(affect=affect, prosody=reading)
        # No emotion classifier runs on this path, so joy/sadness/anger stay
        # unmeasured and say so; the vector carries valence, arousal, whose
        # state it is, and the clauses.
        face_vector = build_face_vector(affect=affect, prosody=reading)

        return HearingTurn(
            ingest=taken,
            prosody=reading,
            affect=affect,
            frames=frames,
            face=face,
            decision=decision,
            notes=notes,
            acoustic=acoustic,
            nonverbal=multimodal,
            temporal=sequences,
            nonverbal_distress=distress,
            face_vector=face_vector,
            voice_held=voice_held,
            signal_voiced=signal_voiced,
        )

    def _signal_words(self, distress: List[Dict[str, Any]]) -> Optional[str]:
        """
        The client's own words for the danger signals in `distress`: a pressed
        symbol's words, or a meaning learned for that client. A built-in
        camera, eye or sound reading has no words (ruling J), so nothing.
        """
        words: List[str] = []
        for d in distress:
            if not d.get("silent_alarm"):
                continue
            modality, key = str(d.get("modality") or ""), d.get("input")
            if modality.startswith("temporal:"):
                if getattr(self.temporal, "language_map_source", "built_in") != "client":
                    continue
                entry = self.temporal.language_map.get(key) or {}
            else:
                try:
                    m = Modality(modality)
                except ValueError:
                    continue
                if m is not Modality.SYMBOL and self.nonverbal.map_source.get(m) != "client":
                    continue
                entry = self.nonverbal.maps.get(m, {}).get(key) or {}
            message = entry.get("message")
            if isinstance(message, str) and message.strip() and message not in words:
                words.append(message)
        return " ".join(words) or None


__all__ = ["Hearing", "HearingTurn", "harm_frame_safety", "CrisisNotDelivered",
           "NONVERBAL_INPUTS", "asked_for_help", "HELP_REQUEST_PATTERNS",
           "CLIENT_DISTRESS_TERMS", "body_harm", "BodyHarm", "BODY_PARTS",
           "ASK_IF_SAFE", "ASK_IF_HELP_NEEDED", "ASK_WHOSE", "ASK_WHO", "ASK_WHAT",
           "ASK_WHAT_ABOUT", "ASK_WHAT_IS_MAKING", "ASK_WHAT_OF", "ASK_WHY_CANT",
           "ASK_WHY", "ASK_WHAT_HAPPENED", "ASK_OR_OFFER", "ASK_MARKS",
           "answered_follow_up",
           "danger_sentences", "asked_aloud", "DANGER_SENTENCE_PATTERNS", "ALOUD_PATTERNS",
           "fear_without_context", "whose_feeling_unclear", "anchor_from"]
