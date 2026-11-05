# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
#
# Truth. Dignity. Protection. Transparency. No Erasure.
# Contact: contact@thechristmanaiproject.com
# ==============================================================================

"""
Nonverbal intent engine — gestures, symbols, eye regions, vocalizations.

For a nonverbal client this module IS their voice. Everything below follows from
that: it may report what it recognized, or report that it recognized nothing.
It may not invent a sentence.

WHAT CHANGED AND WHY
--------------------

1. IT ADDED RANDOM NOISE TO EVERY CONFIDENCE. Three sites, :354 :401 :442.

       confidence_variation = random.uniform(-0.05, 0.05)
       result["confidence"] = min(1.0, max(0.1, result["confidence"] + ...))

   Comment said "simulate real-world variation." Measured, one gesture 'nod'
   classified five times:

       0.8937   0.9205   0.8513   0.9173   0.9280

   The stored confidence is a fixed number in a table. Jittering it does not
   make it a measurement; it makes a constant look like one. Deleted. The
   table value is reported as-is, and it is labelled `prior`, because that is
   what it is.

2. IT SPOKE FOR AN INPUT IT DID NOT RECOGNIZE.

       result = {"intent": "unknown", "confidence": 0.3, ...}
       result["message"] = "I'm trying to communicate something."

   Measured on an unrecognized gesture: confidence 0.2998, message
   "I'm trying to communicate something." That sentence was placed in a
   nonverbal person's mouth for an input the system could not read.
   `process_sound` did the same at 0.4; `process_multimodal_input` with no
   inputs at all returned 0.1 and "I'm not sure what you're trying to
   communicate."

   An unrecognized input now returns status="unrecognized", confidence None,
   message None. Absence of a reading is reported as absence.

3. THE CONFIDENCE THRESHOLD WAS SET AND NEVER READ.

       self.confidence_threshold = 0.6

   One occurrence in 938 lines — the assignment. Nothing was ever gated on it
   and there was no way for the client to reject or undo a reading. It is now
   read in one place, `_gate()`, which every public entry point passes
   through, and `reject_last()` gives the client an undo that also records the
   correction. (Since 2026-11-04, item 9, the bar labels a reading; it no
   longer takes the client's words away.)

4. THE LEARNING SYSTEM WAS INERT, NOT DECAYING.

   The prior note recorded a "confidence only decays" bug. Measured, it does
   not decay — it never moves at all:

       before=0.9  after=0.9  changed=False
       gate requires delta > 0.05 ; actual delta = 0.045

   `new = current*(1-lr) + rate*lr` with lr=0.05 yields a maximum delta of
   current*0.05, which cannot exceed 0.05 for any confidence <= 1.0, so
   `if abs(new - current) > 0.05` never opened. Nothing it computed ever
   reached a model file.

   Underneath that, `success` never incremented: `classify_gesture` called
   `record_interaction` without an outcome, so `count` rose and `success`
   stayed 0 — 26 interactions, success 0, rate 0.0.

   Both are fixed by removing the guesswork rather than tuning it. Confidence
   now updates ONLY from interactions whose outcome a human confirmed, needs
   MIN_CONFIRMED_SAMPLES of them, and moves by a stated step. Unconfirmed
   interactions are counted separately and never train anything.

5. MULTIMODAL INTERACTIONS WERE NEVER RECORDED.

       self.record_interaction("multimodal", input_key, result)

   `section_map` had no "multimodal" key, so this logged "Unknown input type"
   and returned. Verified: usage_stats['multimodal'] was {} before and after.
   The section existed in the stats file and was never written.

6. AN UNEXPECTED emotion_tier CRASHED THE ENGINE.

       emotion_tier_votes[tier] += weight     ->  KeyError 'critical'

   Tiers come from JSON model files on disk, so a hand-edited or learned file
   took the engine down mid-classification. Tiers are now validated at load;
   an unknown tier is rejected there, loudly, with the entry skipped.

7. IMPORTING IT WROTE TO DISK.

       os.makedirs(self.data_dir, exist_ok=True)   # in __init__

   Verified: constructing the engine created `data/learning` under whatever
   the cwd happened to be. Directories are now created only inside an explicit
   save, and `_save_*` writes atomically via a temp file and os.replace, so a
   crash mid-write cannot leave a truncated JSON that the loader then swallows.

8. `logging.basicConfig(level=INFO)` AT MODULE SCOPE reconfigured root logging
   for every application that imported this file. Removed; a module logger
   with a NullHandler, per the SDK convention.

9. WHAT CHANGED AND WHY (2026-11-04): THE ENGINE READS; THE BEING DECIDES.
   Everett's ruling, in his words:

       "we can have this knowledge in the sound package, but it can't be
       making those decisions. It's up to the being to make those
       decisions, not something that just creates a voice."
       "it can't make decisions that counteract the being, because the
       being is the one that has all of the work to make those decisions
       properly."

   Before this, the engine took the client's mapped words away in two
   places. A reading under the confidence bar came back with message None.
   A danger reading (help, scared, attacked, any map entry whose words
   speak of fear or attack) came back with message None and `silent_alarm`
   True, and the rest of the package read that as an order to reach a
   human and say nothing. `Interpretation.speakable` told the caller
   whether to voice a reading. In a multimodal input a distress intent
   overrode the measured vote, and one danger reading anywhere blanked the
   message for the whole input.

   Now the engine reports what it read, and the being decides:
     - The client's mapped words are always on the reading. Under the bar
       the reading is labelled BELOW_THRESHOLD (`below_threshold` True)
       and carries its confidence, the threshold and its words.
     - A danger meaning is reported, not acted on: `danger` True and
       `danger_term`, the intent or the word in the message that matched.
     - Every recognized reading carries `category`: "danger", "need" or
       "other" (None when nothing was recognized). The intent sets stay as
       knowledge, named for what they are: DANGER_CATEGORY_INTENTS (was
       DANGER_INTENTS), NEED_CATEGORY_INTENTS (was the need list named for
       paging, same members), DISTRESS_INTENTS (both).
     - `silent_alarm` and `speakable` are removed from Interpretation.
     - Multimodal reports the tallies, the measured winner (or the tie, when
       the top tallies are equal), `distress_present` and `danger_present`.
       Nothing overrides the vote and nothing blanks the message.
   `is_danger`, `danger_term` and `distress_readings` are queries. The
   package does nothing with their answers.

KEPT DELIBERATELY
-----------------
`get_emotional_indicators` maps `stimming` and `vocal_stimming` to
self-regulation 0.9, above anxiety and overwhelm. It reads stimming as a
person managing themselves rather than as a symptom to escalate. That is a
dignity judgement encoded in data and it survives this rewrite unchanged.
Grunt keys (`grunt_distress`, `grunt_acknowledge`, `grunt_frustration`) are
accepted only as keys from the Corti client router. This engine does not
compute them. Family Sound does not carry `kind`.
"""

from __future__ import annotations

import json
import re
import logging
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Modality(str, Enum):
    GESTURE = "gesture"
    SYMBOL = "symbol"
    EYE = "eye"
    SOUND = "sound"
    MULTIMODAL = "multimodal"


#: Stats section per modality. Every member of Modality has one — the absence
#: of "multimodal" here is what silently dropped every multimodal interaction.
SECTION: Dict[Modality, str] = {
    Modality.GESTURE: "gestures",
    Modality.SYMBOL: "symbols",
    Modality.EYE: "eye_regions",
    Modality.SOUND: "sound_patterns",
    Modality.MULTIMODAL: "multimodal",
}

#: The only tiers the vote table can hold. Validated at load, not at vote time.
VALID_TIERS = ("mild", "moderate", "strong", "urgent")

#: Interactions with a CONFIRMED outcome required before confidence moves.
#: 10: a chosen starting value, held until client data replaces it. Reason:
#: one bad day of 3 or 4 misreads must not retrain a signal.
MIN_CONFIRMED_SAMPLES = 10

#: How far confidence may move in one update. Stated, not emergent.
#: 0.05: a chosen starting value, held until client data replaces it. Reason:
#: a 0.9 symbol needs 6 confirmed-wrong updates to fall below the 0.6 floor,
#: so one update can never drop a signal under the bar.
CONFIDENCE_STEP = 0.05

#: Floor and ceiling on a learned confidence. A prior may not learn its way to
#: certainty, and it may not decay to unusable either.
#: 0.20 and 0.95: chosen starting values, held until client data replaces
#: them. Reason: 0.20 keeps a decayed signal on record to relearn instead of
#: deleting it; 0.95 leaves room for "maybe I misread" on every signal.
CONFIDENCE_FLOOR = 0.20
CONFIDENCE_CEILING = 0.95

#: CATEGORY LABELS (2026-11-04). Every recognized reading carries one:
#: "danger", "need" or "other". The label is knowledge about the reading.
#: What happens next is the being's decision, not this package's.
CATEGORY_DANGER = "danger"
CATEGORY_NEED = "need"
CATEGORY_OTHER = "other"
CATEGORIES = (CATEGORY_DANGER, CATEGORY_NEED, CATEGORY_OTHER)

#: Intents read as category "danger": help and danger. Everett, 2026-10-29
#: (ruling I) grouped them: a danger signal is help, scared, being attacked,
#: "I'm scared for my safety", from any channel. His reason for treating it
#: apart: saying it out loud can get the client hurt worse and the device
#: taken. That reason is the being's to act on; this set only names the
#: intents. The grunt key, the "distress" sound and the point_up gesture map
#: to "help"; the help symbol maps to "need_help". Any entry whose words
#: speak of fear, danger or attack (DANGER_WORDS) is read as danger too.
#: Everett, 2026-10-29 (ruling AF): "A nonverbal client's distress is just
#: as important as a verbal client's."
DANGER_CATEGORY_INTENTS = frozenset({
    "help", "need_help", "danger", "scared", "afraid", "fear", "unsafe",
    "attacked", "attack", "threat", "emergency",
})

#: Intents read as category "need". The same members the need list held
#: before 2026-11-04: pain and need_medicine. Everett, 2026-10-29: pain is
#: not a need, "It's a feeling", the client's own; medicine is a need. The
#: label "need" on pain is carried from the old list, and what to call it
#: is Everett's call. Thirst, hunger and the
#: bathroom are not in this set (hearing.NEED_INTENTS lists them apart), so
#: they read "other" here.
NEED_CATEGORY_INTENTS = frozenset({"pain", "need_medicine"})

#: Both categories together: the intents read as distress.
DISTRESS_INTENTS = DANGER_CATEGORY_INTENTS | NEED_CATEGORY_INTENTS

#: Words in a map entry's message that make it a danger meaning, whatever its
#: intent is called. Matched as words.
DANGER_WORDS = re.compile(
    r"\b(?:scared|afraid|frightened|terrified|danger|dangerous|unsafe|attack|"
    r"attacked|attacking|safety)\b", re.I)


def danger_term(intent: Optional[str], message: Optional[str] = None) -> Optional[str]:
    """
    What makes a meaning danger, or None.

    The intent itself when it is in DANGER_CATEGORY_INTENTS; otherwise the
    first DANGER_WORDS match in the message, exactly as written there. A
    reading only: nothing is done with it here.
    """
    if (intent or "") in DANGER_CATEGORY_INTENTS:
        return intent
    found = DANGER_WORDS.search(message or "")
    return found.group(0) if found else None


def is_danger(intent: Optional[str], message: Optional[str] = None) -> bool:
    """True when the meaning reads as danger (see danger_term). A query only."""
    return danger_term(intent, message) is not None


def category_of(intent: Optional[str], message: Optional[str] = None) -> Optional[str]:
    """
    The category label for a meaning: "danger", "need" or "other".

    None when there is no meaning at all (no intent and no words): nothing
    was recognized, so nothing is labelled.
    """
    if not intent and not message:
        return None
    if is_danger(intent, message):
        return CATEGORY_DANGER
    if intent in NEED_CATEGORY_INTENTS:
        return CATEGORY_NEED
    return CATEGORY_OTHER


#: WHERE THE WORDS COME FROM (Everett, 2026-10-29, rulings C, J and L). A
#: map entry carries its meaning, not a speak/silent flag. The engine
#: returns whatever words the map holds for a reading, and reports every
#: reading whether or not it has words:
#:   - a pressed symbol is the client's own word (ruling C);
#:   - a meaning in the client's OWN map, read from data_dir, is a meaning
#:     learned for that client (ruling J);
#:   - THE CAMERA'S BUILT-IN MAPS CARRY NO WORDING (ruling J). The built-in
#:     gesture and eye-region maps hold message None: a wave, a nod, a thumbs
#:     up, a held centre gaze is reported as a reading (key, intent,
#:     confidence) with no words. Until 2026-10-29's review, a built-in wave
#:     said "Hello there!", a nod "Yes, I agree.", a thumbs up "I like
#:     this." and a centre gaze "I select this option." for the client.
#:     Those sentences are removed from the data;
#:   - RULING J COVERS SOUNDS TOO (Everett, 2026-10-29): the built-in sound
#:     map carries no wording; a sound, tic or click has words only once
#:     they are learned for that client, in the client's own map. (Before, a
#:     built-in click said "I choose this option.") GENERIC_FEELING_INTENTS
#:     is kept for callers that read it.
#: An older map with no such fields reads exactly as before.
GENERIC_FEELING_INTENTS = frozenset({
    "self_regulate", "overwhelmed", "thinking", "unsure", "excited",
    "frustrated", "focus", "searching",
})
#: A camera, eye or sound reading whose map entry has no words (ruling J).
NOT_LEARNED_REASON = ("no meaning learned for this client: the reading is reported, "
                      "not spoken")


def distress_readings(readings: "List[Interpretation]") -> "List[Interpretation]":
    """
    The recognized readings whose category is "danger" or "need".

    A query. Everett, 2026-10-29: "A nonverbal client's distress is just as
    important as a verbal client's." So a recognized distress reading is
    listed at ANY confidence, under the bar or over it: the bar never
    decides whether a reading is distress. Unrecognized input never counts;
    nothing is invented. The package does nothing with the answer; the being
    decides.
    """
    return [
        r for r in readings
        if r.status in (Status.OK, Status.BELOW_THRESHOLD)
        and (r.intent in DISTRESS_INTENTS or r.danger)
    ]


class Status(str, Enum):
    OK = "ok"                          # recognized, at or above threshold
    BELOW_THRESHOLD = "below_threshold"  # recognized, not confident enough
    UNRECOGNIZED = "unrecognized"      # input not in the map
    NO_INPUT = "no_input"              # nothing was supplied


@dataclass(frozen=True)
class Interpretation:
    """
    One reading.

    `confidence` is a PRIOR from the map, optionally adjusted by confirmed
    outcomes. It is not a measurement of this particular input, and
    `confidence_is_prior` says so on every result. It is None whenever there
    is nothing to be confident about.

    `message` is the client's words for this reading, exactly as the map
    holds them. It is None when the map entry carries no words (the built-in
    camera and microphone maps, ruling J) or when nothing was recognized. It
    is never taken away because of the bar or because the reading is danger:
    whether to voice it is the being's decision (2026-11-04).
    """

    modality: Modality
    status: Status
    input_data: Optional[str] = None
    intent: Optional[str] = None
    expression: Optional[str] = None
    emotion_tier: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    reason: Optional[str] = None
    threshold: Optional[float] = None
    confirmed_samples: int = 0
    interaction_id: Optional[str] = None
    #: "danger", "need" or "other" for a recognized reading; None otherwise.
    category: Optional[str] = None
    #: The meaning reads as danger (help or danger). Reported, never acted on.
    danger: bool = False
    #: What matched: the intent, or the word in the message. None if not danger.
    danger_term: Optional[str] = None

    @property
    def below_threshold(self) -> bool:
        """The reading is under the confidence bar. Its words are still on it."""
        return self.status is Status.BELOW_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "status": self.status.value,
            "input": self.input_data,
            "intent": self.intent,
            "expression": self.expression,
            "emotion_tier": self.emotion_tier,
            "confidence": self.confidence,
            "confidence_known": self.confidence is not None,
            "confidence_is_prior": True,
            "message": self.message,
            "reason": self.reason,
            "threshold": self.threshold,
            "confirmed_samples": self.confirmed_samples,
            "interaction_id": self.interaction_id,
            "below_threshold": self.below_threshold,
            "category": self.category,
            "danger": self.danger,
            "danger_term": self.danger_term,
        }


class MapValidationError(ValueError):
    """A loaded map entry is malformed. Raised at load, never at vote time."""


# -----------------------------------------------------------------------------
# Default maps. Confidence values are PRIORS — how much this system trusts the
# mapping itself, not a measurement of any particular input.
#
# EVERY NUMBER BELOW IS A CHOSEN STARTING VALUE, NOT A MEASUREMENT. Each is
# held until client data replaces it. They were in these tables before
# 2026-10-29 with no reason written; the reasons are written now (Everett's
# ruling, 2026-10-29: every fixed number gets its reason beside it):
#
#   0.9  a deliberate signal with one meaning: a pressed symbol ("pain",
#        "help"), a nod, a shake, a thumbs up/down, the distress sound keys.
#        Highest, so a call for help clears the 0.6 floor with room to spare.
#   0.8  a deliberate signal with more than one everyday meaning: a wave
#        (hello or goodbye), pointing up, an open palm, a held centre gaze,
#        a loud sound (excited or upset).
#   0.7  a signal that is often made without meaning to: stimming, rapid
#        blinking, corner glances, a click, a short vowel.
#   0.6  a sound that often carries no message at all: a hum, a soft sound.
#        Set AT the floor: it is not labelled below the bar (the test is
#        "below 0.6"), and it loses any vote against a stronger reading.
#
# Checked entry by entry (2026-10-29). These do not fit the reasons above,
# and no reason of their own was ever written, so they are listed here as
# they stand for Everett to give the reason or change the value:
#   symbols at 0.8: question, tired, play, music, book, outside (each a
#     pressed symbol with one meaning, which the 0.9 reason would cover).
#   sounds at 0.8: repeated_sound, grunt_acknowledge, grunt_frustration.
#   eye region at 0.8: long_stare. At 0.7: rapid_scan.
#   vocal_stimming at 0.9, while the stimming gesture is 0.7.
#
# THE SYMBOLS' NUMBERS ARE NOT UNDER THE BAR (Everett's ruling C,
# 2026-10-29: a symbol or key the client pressed counts like a spoken
# word). The bar
# (confidence_threshold) is for readings this package could get wrong: a
# camera's gesture, an eye region, a sound the microphone classified. A
# pressed symbol is not a guess, so it is never labelled below the bar (see
# NonverbalEngine._gate). The number stays on the reading and still weighs
# in a multimodal vote. And no number, on any channel, decides whether a
# reading is labelled distress: category and danger are read at any
# confidence.
# -----------------------------------------------------------------------------

# THE CAMERA'S BUILT-IN MAPS CARRY NO WORDING (Everett, ruling J): the
# camera has words only for meanings learned for that client. A gesture or
# eye region here is a reading (intent, confidence, tier) with message None;
# the client's own map in data_dir is where a meaning's words come from.
DEFAULT_GESTURES: Dict[str, Dict[str, Any]] = {
    "nod":         {"intent": "affirm",        "confidence": 0.9, "expression": "positive", "emotion_tier": "moderate", "message": None},
    "shake":       {"intent": "deny",          "confidence": 0.9, "expression": "negative", "emotion_tier": "moderate", "message": None},
    "point_up":    {"intent": "help",          "confidence": 0.8, "expression": "urgent",   "emotion_tier": "moderate", "message": None},
    "wave":        {"intent": "greet",         "confidence": 0.8, "expression": "positive", "emotion_tier": "mild",     "message": None},
    "thumbs_up":   {"intent": "like",          "confidence": 0.9, "expression": "positive", "emotion_tier": "strong",   "message": None},
    "thumbs_down": {"intent": "dislike",       "confidence": 0.9, "expression": "negative", "emotion_tier": "strong",   "message": None},
    "open_palm":   {"intent": "stop",          "confidence": 0.8, "expression": "negative", "emotion_tier": "strong",   "message": None},
    "stimming":    {"intent": "self_regulate", "confidence": 0.7, "expression": "neutral",  "emotion_tier": "mild",     "message": None},
    "rapid_blink": {"intent": "overwhelmed",   "confidence": 0.7, "expression": "negative", "emotion_tier": "urgent",   "message": None},
}

DEFAULT_SYMBOLS: Dict[str, Dict[str, Any]] = {
    "food":     {"intent": "hungry",         "confidence": 0.9, "expression": "urgent",      "emotion_tier": "moderate", "message": "I'm hungry."},
    "drink":    {"intent": "thirsty",        "confidence": 0.9, "expression": "urgent",      "emotion_tier": "moderate", "message": "I'm thirsty."},
    "bathroom": {"intent": "bathroom",       "confidence": 0.9, "expression": "urgent",      "emotion_tier": "strong",   "message": "I need the bathroom."},
    "pain":     {"intent": "pain",           "confidence": 0.9, "expression": "negative",    "emotion_tier": "strong",   "message": "I'm in pain."},
    "happy":    {"intent": "express_joy",    "confidence": 0.9, "expression": "positive",    "emotion_tier": "moderate", "message": "I'm happy."},
    "sad":      {"intent": "express_sadness","confidence": 0.9, "expression": "negative",    "emotion_tier": "moderate", "message": "I'm sad."},
    "help":     {"intent": "need_help",      "confidence": 0.9, "expression": "urgent",      "emotion_tier": "strong",   "message": "I need help."},
    "question": {"intent": "ask_question",   "confidence": 0.8, "expression": "inquisitive", "emotion_tier": "mild",     "message": "I have a question."},
    "tired":    {"intent": "tired",          "confidence": 0.8, "expression": "negative",    "emotion_tier": "moderate", "message": "I'm tired."},
    "medicine": {"intent": "need_medicine",  "confidence": 0.9, "expression": "urgent",      "emotion_tier": "strong",   "message": "I need my medicine."},
    "yes":      {"intent": "affirm",         "confidence": 0.9, "expression": "positive",    "emotion_tier": "moderate", "message": "Yes."},
    "no":       {"intent": "deny",           "confidence": 0.9, "expression": "negative",    "emotion_tier": "moderate", "message": "No."},
    "play":     {"intent": "want_play",      "confidence": 0.8, "expression": "positive",    "emotion_tier": "mild",     "message": "I want to play."},
    "music":    {"intent": "want_music",     "confidence": 0.8, "expression": "positive",    "emotion_tier": "mild",     "message": "I want music."},
    "book":     {"intent": "want_book",      "confidence": 0.8, "expression": "positive",    "emotion_tier": "mild",     "message": "I want a book."},
    "outside":  {"intent": "want_outside",   "confidence": 0.8, "expression": "positive",    "emotion_tier": "moderate", "message": "I want to go outside."},
}

DEFAULT_EYE_REGIONS: Dict[str, Dict[str, Any]] = {
    "top_left":     {"intent": "previous",  "confidence": 0.7, "expression": "neutral",     "emotion_tier": "mild",     "message": None},
    "top_right":    {"intent": "next",      "confidence": 0.7, "expression": "neutral",     "emotion_tier": "mild",     "message": None},
    "bottom_left":  {"intent": "cancel",    "confidence": 0.7, "expression": "negative",    "emotion_tier": "moderate", "message": None},
    "bottom_right": {"intent": "confirm",   "confidence": 0.7, "expression": "positive",    "emotion_tier": "moderate", "message": None},
    "center":       {"intent": "select",    "confidence": 0.8, "expression": "attentive",   "emotion_tier": "mild",     "message": None},
    "long_stare":   {"intent": "focus",     "confidence": 0.8, "expression": "attentive",   "emotion_tier": "strong",   "message": None},
    "rapid_scan":   {"intent": "searching", "confidence": 0.7, "expression": "inquisitive", "emotion_tier": "moderate", "message": None},
}

# THE MICROPHONE'S BUILT-IN SOUND MAP CARRIES NO WORDING EITHER (Everett,
# 2026-10-29: ruling J covers sounds too). A hum, a click, a grunt is a
# reading (intent, confidence, tier) with no words; a sound meaning has
# words only once it is learned for that client, in the client's own map.
# Until this review a built-in click said "I choose this option.", a short
# vowel "I acknowledge that." and a repeated sound "Please pay attention to
# this." for the client. Those sentences are removed.
DEFAULT_SOUNDS: Dict[str, Dict[str, Any]] = {
    "hum":            {"intent": "thinking",    "confidence": 0.6, "expression": "neutral",  "emotion_tier": "mild",   "message": None},
    "click":          {"intent": "select",      "confidence": 0.7, "expression": "neutral",  "emotion_tier": "mild",   "message": None},
    "distress":       {"intent": "help",        "confidence": 0.9, "expression": "negative", "emotion_tier": "urgent", "message": None},
    "soft":           {"intent": "unsure",      "confidence": 0.6, "expression": "neutral",  "emotion_tier": "mild",   "message": None},
    "loud":           {"intent": "excited",     "confidence": 0.8, "expression": "positive", "emotion_tier": "strong", "message": None},
    "short_vowel":    {"intent": "acknowledge", "confidence": 0.7, "expression": "neutral",  "emotion_tier": "mild",   "message": None},
    "repeated_sound":     {"intent": "insistent",     "confidence": 0.8, "expression": "urgent",   "emotion_tier": "strong", "message": None},
    "grunt_distress":     {"intent": "help",          "confidence": 0.9, "expression": "negative", "emotion_tier": "urgent", "message": None},
    "grunt_acknowledge":  {"intent": "acknowledge",   "confidence": 0.8, "expression": "neutral",  "emotion_tier": "mild",   "message": None},
    "grunt_frustration":  {"intent": "frustrated",    "confidence": 0.8, "expression": "negative", "emotion_tier": "strong", "message": None},
    "vocal_stimming":     {"intent": "self_regulate", "confidence": 0.9, "expression": "neutral",  "emotion_tier": "mild",   "message": None},
}

#: Emotional indicators per gesture. KEPT AS WRITTEN — see the header note.
#: `stimming` reads as self-regulation above anxiety and overwhelm on purpose.
#: The values are a rank order, not measurements: 0.9 is the first reading of
#: that signal, 0.8 the second, 0.7/0.6/0.5 the less likely ones. Chosen
#: starting values, held until client data replaces them.
EMOTIONAL_INDICATORS: Dict[str, Dict[str, float]] = {
    "nod":         {"agreement": 0.9, "acceptance": 0.8, "interest": 0.6},
    "shake":       {"disagreement": 0.9, "rejection": 0.8, "frustration": 0.5},
    "point_up":    {"urgency": 0.8, "attention": 0.9, "importance": 0.7},
    "wave":        {"greeting": 0.9, "friendliness": 0.8, "openness": 0.7},
    "thumbs_up":   {"approval": 0.9, "satisfaction": 0.8, "happiness": 0.7},
    "thumbs_down": {"disapproval": 0.9, "dissatisfaction": 0.8, "disappointment": 0.7},
    "open_palm":   {"stopping": 0.9, "boundary": 0.8, "caution": 0.7},
    "stimming":        {"anxiety": 0.8, "overwhelm": 0.7, "self-regulation": 0.9},
    "rapid_blink":     {"distress": 0.7, "anxiety": 0.6, "overwhelm": 0.8},
    "vocal_stimming":  {"anxiety": 0.8, "overwhelm": 0.7, "self-regulation": 0.9},
    "grunt_distress":  {"urgency": 0.9, "distress": 0.9, "fear": 0.8},
}

_REQUIRED_KEYS = ("intent", "confidence", "expression", "emotion_tier", "message")


def validate_map(name: str, table: Dict[str, Any], strict: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Validate a map at LOAD time.

    The predecessor validated nothing and discovered malformed entries during
    a vote, as `KeyError 'critical'` mid-classification. Every rejection here
    is logged with the entry that caused it and the entry is dropped, so a
    hand-edited file degrades the map instead of taking the engine down.

    Args:
        strict: raise MapValidationError instead of dropping bad entries.
    """
    clean: Dict[str, Dict[str, Any]] = {}
    for key, entry in (table or {}).items():
        try:
            if not isinstance(entry, dict):
                raise MapValidationError(f"{name}[{key!r}] is {type(entry).__name__}, expected object")
            missing = [k for k in _REQUIRED_KEYS if k not in entry]
            if missing:
                raise MapValidationError(f"{name}[{key!r}] missing {missing}")
            tier = entry["emotion_tier"]
            if tier not in VALID_TIERS:
                raise MapValidationError(
                    f"{name}[{key!r}] emotion_tier={tier!r} not in {VALID_TIERS}"
                )
            conf = float(entry["confidence"])
            if not 0.0 <= conf <= 1.0:
                raise MapValidationError(f"{name}[{key!r}] confidence {conf} outside [0,1]")
            # None is a reading with no wording (ruling J: the camera's
            # built-in maps). A message that is there must not be blank.
            if entry["message"] is not None and (
                    not isinstance(entry["message"], str) or not entry["message"].strip()):
                raise MapValidationError(f"{name}[{key!r}] message is empty")
        except (MapValidationError, TypeError, ValueError) as exc:
            if strict:
                raise MapValidationError(str(exc)) from exc
            logger.error("Rejected map entry: %s", exc)
            continue
        item = dict(entry)
        item["confidence"] = conf
        clean[key] = item
    return clean


#: Two tallies closer than this are equal. 1e-9: a chosen value. Reason:
#: weighted sums of the same priors can differ in the last float bits
#: (0.8 x 0.9 against 0.72), and the smallest real difference the maps can
#: make is far larger (priors move in steps of 0.05, weighted 0.8 at least).
TIE_TOLERANCE = 1e-9


def _leaders(tallies: Dict[str, float]) -> List[str]:
    """Every key that shares the top tally, sorted. More than one is a tie."""
    if not tallies:
        return []
    top = max(tallies.values())
    return sorted(k for k, v in tallies.items() if abs(v - top) <= TIE_TOLERANCE)


def _atomic_write_json(path: str, payload: Any) -> None:
    """
    Write JSON atomically.

    The predecessor opened the target and wrote in place. A crash mid-write
    left a truncated file, and the loader caught only JSONDecodeError — so on
    the next start the client's learned map silently reverted to defaults with a
    warning nobody was watching.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class NonverbalEngine:
    """
    Classifies nonverbal input into intent.

    Every public entry point returns an `Interpretation`. There is no path that
    returns a message for an input the engine did not recognize, and no path
    that returns a confidence it did not get from a validated map.
    """

    def __init__(
        self,
        data_dir: Optional[str] = "data/learning",
        # 0.6: a chosen starting value, not a measurement. Held until client
        # data replaces it. Reason: above the 0.5 coin-flip line, and below
        # every deliberate signal in the maps (0.7-0.9), so the floor labels
        # guesses without ever labelling a pressed symbol.
        confidence_threshold: float = 0.6,
        strict_maps: bool = False,
    ) -> None:
        """
        Args:
            data_dir: where learned maps and stats live. NOT created here —
                only inside an explicit save. Importing this module and
                constructing the engine touch no disk. None means no disk at
                all: built-in maps, nothing read, `save` refuses.
            confidence_threshold: readings below this are labelled
                BELOW_THRESHOLD, with their confidence, this threshold and
                their words. The being decides what to do with them.
            strict_maps: raise on a malformed map entry instead of dropping it.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold {confidence_threshold} outside [0,1]")

        self.data_dir = data_dir
        self.confidence_threshold = float(confidence_threshold)
        self.strict_maps = bool(strict_maps)

        # One lock for every mutable structure below. The predecessor mutated
        # maps from a background learning thread while classification read
        # them, with nothing between the two.
        self._lock = threading.RLock()

        #: Where each map came from: "client" (the client's own map, read
        #: from data_dir: meanings learned for that client) or "built_in".
        self.map_source: Dict[Modality, str] = {}
        self.maps: Dict[Modality, Dict[str, Dict[str, Any]]] = {}
        for modality, name, default in (
            (Modality.GESTURE, "gestures", DEFAULT_GESTURES),
            (Modality.SYMBOL, "symbols", DEFAULT_SYMBOLS),
            (Modality.EYE, "eye_regions", DEFAULT_EYE_REGIONS),
            (Modality.SOUND, "sound_patterns", DEFAULT_SOUNDS),
        ):
            loaded = self._load(name)
            self.map_source[modality] = "client" if loaded else "built_in"
            self.maps[modality] = validate_map(name, loaded or default, self.strict_maps)

        self.usage_stats: Dict[str, Any] = self._load_stats()
        self.history: Deque[Interpretation] = deque(maxlen=100)

        self._learning = threading.Event()
        self._learning_thread: Optional[threading.Thread] = None
        self._interaction_seq = 0

        empty = [m.value for m, t in self.maps.items() if not t]
        if empty:
            logger.error("Maps loaded EMPTY for %s — those modalities will "
                         "report unrecognized for every input.", empty)
        logger.info("NonverbalEngine ready. threshold=%.2f maps=%s",
                    self.confidence_threshold,
                    {m.value: len(t) for m, t in self.maps.items()})

    # -- Persistence ----------------------------------------------------------

    def _path(self, name: str) -> str:
        return os.path.join(self.data_dir, f"{name}.json")

    def _load(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a map. Never writes. Returns None when there is nothing usable."""
        if self.data_dir is None:
            return None
        path = self._path(f"{name}_model")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"expected object, got {type(data).__name__}")
            logger.info("Loaded %s map: %d entries", name, len(data))
            return data
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            # Loud, and it says what is being used instead. The predecessor
            # logged a warning and silently continued on defaults.
            logger.error(
                "%s is unusable (%s). FALLING BACK TO BUILT-IN DEFAULTS — any "
                "learning stored in that file is not in effect.", path, exc,
            )
            return None

    def _load_stats(self) -> Dict[str, Any]:
        blank = {s: {} for s in SECTION.values()}
        blank["last_updated"] = None
        if self.data_dir is None:
            return blank
        path = self._path("usage_stats")
        if not os.path.exists(path):
            return blank
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("expected object")
            for section in SECTION.values():
                data.setdefault(section, {})
            return data
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.error("%s unusable (%s); starting fresh stats.", path, exc)
            return blank

    def save(self) -> List[str]:
        """
        Persist maps and stats. EXPLICIT — nothing here runs from __init__.

        Returns the paths written.
        """
        if self.data_dir is None:
            raise ValueError("this engine was built with data_dir=None; it has no disk")
        written: List[str] = []
        with self._lock:
            for modality, table in self.maps.items():
                name = {"gesture": "gestures", "symbol": "symbols",
                        "eye": "eye_regions", "sound": "sound_patterns"}[modality.value]
                path = self._path(f"{name}_model")
                _atomic_write_json(path, table)
                written.append(path)
            self.usage_stats["last_updated"] = datetime.now(timezone.utc).isoformat()
            path = self._path("usage_stats")
            _atomic_write_json(path, self.usage_stats)
            written.append(path)
        logger.info("Saved %d file(s) under %s", len(written), self.data_dir)
        return written


    # -- Classification -------------------------------------------------------

    def _gate(self, interp: Interpretation) -> Interpretation:
        """
        The single place the threshold is read.

        A reading under the bar is labelled BELOW_THRESHOLD and keeps its
        words, its confidence, its category and its danger reading; the
        threshold goes with it. Nothing is taken away. The predecessor had a
        threshold attribute and no gate at all; until 2026-11-04 this gate
        blanked the words of a reading under the bar.
        """
        if interp.status is not Status.OK:
            return interp
        # A PRESSED SYMBOL IS NOT UNDER THE BAR (Everett's ruling C, 2026-10-29:
        # a symbol or key the client pressed counts like a spoken word). The bar
        # exists for readings this package could get wrong: a camera's
        # gesture, an eye region, a sound the microphone classified. A symbol
        # is not a reading; the client chose it. So the typed confidence in
        # the symbol map never labels a pressed symbol below the bar. It is
        # still carried on the reading, and still weighs in a multimodal vote.
        if interp.modality is Modality.SYMBOL:
            return interp
        if interp.confidence is None or interp.confidence < self.confidence_threshold:
            return replace(
                interp,
                status=Status.BELOW_THRESHOLD,
                reason=(
                    f"confidence {interp.confidence} is below the "
                    f"{self.confidence_threshold} threshold"
                ),
                threshold=self.confidence_threshold,
            )
        return interp

    def _classify(self, modality: Modality, key: Optional[str]) -> Interpretation:
        """Look `key` up in `modality`'s map. Never invents an entry."""
        if not key:
            return Interpretation(
                modality=modality, status=Status.NO_INPUT,
                reason="no input supplied", threshold=self.confidence_threshold,
            )

        with self._lock:
            entry = self.maps.get(modality, {}).get(key)
            stats = self.usage_stats.get(SECTION[modality], {}).get(key, {})
            confirmed = int(stats.get("confirmed", 0))

        if entry is None:
            # The predecessor returned confidence 0.3 and the sentence
            # "I'm trying to communicate something." for exactly this case.
            return Interpretation(
                modality=modality, status=Status.UNRECOGNIZED, input_data=key,
                reason=f"{key!r} is not in the {modality.value} map",
                threshold=self.confidence_threshold,
            )

        # The client's words, exactly as the map holds them. Never taken
        # away: not for danger, not for the bar (2026-11-04). The built-in
        # camera and microphone maps hold no words (ruling J), so a built-in
        # gesture, eye or sound reading has message None and says why.
        message: Optional[str] = entry["message"]
        reason: Optional[str] = None
        if modality in (Modality.GESTURE, Modality.EYE, Modality.SOUND) and not message:
            reason = NOT_LEARNED_REASON
        term = danger_term(entry["intent"], message)
        interp = Interpretation(
            modality=modality,
            status=Status.OK,
            input_data=key,
            intent=entry["intent"],
            expression=entry["expression"],
            emotion_tier=entry["emotion_tier"],
            confidence=entry["confidence"],     # the prior, unjittered
            message=message,
            reason=reason,
            threshold=self.confidence_threshold,
            confirmed_samples=confirmed,
            interaction_id=self._next_id(),
            category=category_of(entry["intent"], message),
            danger=term is not None,
            danger_term=term,
        )
        interp = self._gate(interp)
        self._record(interp)
        return interp

    def classify_gesture(self, gesture_name: Optional[str]) -> Interpretation:
        return self._classify(Modality.GESTURE, gesture_name)

    def classify_symbol(self, symbol: Optional[str]) -> Interpretation:
        return self._classify(Modality.SYMBOL, symbol)

    def process_eye_movement(self, eye_data: Optional[Dict[str, Any]]) -> Interpretation:
        region = (eye_data or {}).get("region")
        return self._classify(Modality.EYE, region)

    def process_sound(self, sound_pattern: Optional[str]) -> Interpretation:
        return self._classify(Modality.SOUND, sound_pattern)

    # -- Multimodal -----------------------------------------------------------

    def process_multimodal_input(
        self,
        gesture: Optional[str] = None,
        eye_data: Optional[Dict[str, Any]] = None,
        sound: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Combine modalities and report what the vote measured.

        Only readings at or over the bar vote. A below-threshold or
        unrecognized reading adds nothing to the tallies — the predecessor
        let a fabricated 0.3 "unknown" vote alongside a real 0.9 gesture.
        Every reading, under the bar or over it, is in `all_results` with its
        words.

        Nothing overrides the vote (2026-11-04). When two or more intents
        share the top tally the status is "tie" and `tied_intents` names
        them; the engine does not pick one. Distress anywhere in the input is
        reported (`distress`, `distress_present`, `danger_present`), never
        acted on.

        Returns a dict rather than an Interpretation because it carries the
        per-modality breakdown alongside the combined reading.
        """
        parts: List[Interpretation] = []
        for value, fn in (
            (gesture, self.classify_gesture),
            (symbol, self.classify_symbol),
            (eye_data, self.process_eye_movement),
            (sound, self.process_sound),
        ):
            if value:
                parts.append(fn(value))  # type: ignore[operator]

        usable = [p for p in parts if p.status is Status.OK and p.confidence is not None]
        distress = distress_readings(parts)
        readings = {
            "all_results": [p.to_dict() for p in parts],
            # Recognized danger or need readings, under the bar or over it.
            "distress": [p.to_dict() for p in distress],
            "distress_present": bool(distress),
            "danger_present": any(p.danger for p in parts),
            "threshold": self.confidence_threshold,
        }

        if not usable:
            return {
                "status": "no_usable_reading",
                "primary": None,
                "intent": None,
                "tied_intents": [],
                "message": None,
                "tallies": {},
                "tier_tallies": {},
                **readings,
                "note": (
                    "No modality produced a reading at or above the confidence "
                    "threshold. This is NOT a neutral or calm state. A reading "
                    "under the bar is in all_results, labelled below_threshold, "
                    "with its words."
                ),
            }

        # Eye input is weighted below the others; that weighting is a stated
        # choice, not a measurement.
        # 0.8 for eye: a chosen starting value, held until client data
        # replaces it. Reason: a glance is more often unintended than a
        # pressed symbol, a gesture or a sound, so an eye reading must not tie
        # one of those at equal confidence. The weight applies to the tally
        # only; the bar is read on each reading's own confidence.
        # 1.0 for the others: no reason yet to rank them against each other.
        WEIGHT = {Modality.GESTURE: 1.0, Modality.SYMBOL: 1.0,
                  Modality.SOUND: 1.0, Modality.EYE: 0.8}

        tallies: Dict[str, float] = {}
        tier_tallies: Dict[str, float] = {t: 0.0 for t in VALID_TIERS}
        for p in usable:
            w = (p.confidence or 0.0) * WEIGHT.get(p.modality, 1.0)
            tallies[p.intent or ""] = tallies.get(p.intent or "", 0.0) + w
            # Tier is guaranteed valid by validate_map — no KeyError path.
            tier_tallies[p.emotion_tier or "mild"] += w

        # The measured winner, and nothing over it. Until 2026-11-04 any
        # distress intent here overrode the vote (added 2026-10-29, when
        # gesture "nod" + sound "grunt_distress" said "Yes, I agree." and
        # nothing else). Measured on 2ad3ac0, before this change: nod +
        # grunt_distress tie at 0.9 each; the override reported "help", and
        # the tier vote, also tied, reported "moderate" by table order. Now a
        # tie is reported as a tie, and the distress is reported beside it.
        leaders = _leaders(tallies)
        tier_leaders = _leaders(tier_tallies)
        strongest = max(usable, key=lambda p: p.confidence or 0.0)

        key = "+".join(sorted(p.input_data or "" for p in usable))
        self._record_raw(Modality.MULTIMODAL, key)

        measured = {
            "tallies": tallies,
            "tier_tallies": tier_tallies,
            "emotion_tier": tier_leaders[0] if len(tier_leaders) == 1 else None,
            "tied_tiers": tier_leaders if len(tier_leaders) > 1 else [],
            "strongest_overall": strongest.to_dict(),
            "confidence_is_prior": True,
            "modalities": [p.modality.value for p in usable],
            **readings,
        }

        if len(leaders) > 1:
            return {
                "status": "tie",
                "primary": None,
                "intent": None,
                "tied_intents": leaders,
                "confidence": None,
                "message": None,
                **measured,
                "note": (
                    "The top tallies are equal; tied_intents names them. Each "
                    "reading, with its words, is in all_results. The engine "
                    "does not pick one."
                ),
            }

        best_intent = leaders[0]
        # 2026-10-29: the message came from the single strongest reading
        # overall, and was dropped whenever that reading had lost the vote.
        # Measured before: gesture "wave" + eye "center" + sound "click" —
        # "select" won the vote, "wave" was strongest, message None. The
        # person was understood and their words lost. The message now comes
        # from the strongest reading OF the winning intent.
        agreeing = [p for p in usable if p.intent == best_intent]
        # Ruling J left the camera's built-in readings with no wording. Of the
        # readings that agree, one that carries words outranks one that does
        # not (as engine_temporal does), so a clicked choice keeps its words
        # when a wordless eye reading agreed with it.
        worded = [p for p in agreeing if p.message]
        winner = max(worded or agreeing, key=lambda p: p.confidence or 0.0)
        combined_confidence = sum(p.confidence or 0.0 for p in agreeing) / len(agreeing)

        return {
            "status": "ok",
            "primary": winner.to_dict(),
            "intent": best_intent,
            "tied_intents": [],
            "confidence": combined_confidence,
            # The winning reading's words, exactly as mapped, or None when
            # that reading carries none. Never taken away.
            "message": winner.message,
            **measured,
        }


    # -- Interaction record + the client's undo -------------------------------

    def _next_id(self) -> str:
        with self._lock:
            self._interaction_seq += 1
            return f"nv-{int(time.time())}-{self._interaction_seq}"

    def _record(self, interp: Interpretation) -> None:
        """Record an interaction with its outcome UNKNOWN. Outcome is not assumed."""
        self.history.append(interp)
        if interp.input_data:
            self._record_raw(interp.modality, interp.input_data)

    def _record_raw(self, modality: Modality, key: str) -> None:
        section = SECTION[modality]          # every Modality has one now
        with self._lock:
            bucket = self.usage_stats.setdefault(section, {}).setdefault(
                key, {"count": 0, "confirmed": 0, "rejected": 0, "last_used": None}
            )
            bucket["count"] += 1
            bucket["last_used"] = datetime.now(timezone.utc).isoformat()

    def confirm_last(self, interaction_id: Optional[str] = None) -> bool:
        """
        A human confirms the engine read the person correctly.

        This is the ONLY thing that trains confidence upward. The predecessor
        passed successful=True for every interaction with nothing having
        confirmed anything.
        """
        return self._resolve(interaction_id, correct=True)

    def reject_last(self, interaction_id: Optional[str] = None) -> bool:
        """
        The client's undo. Marks the reading wrong and trains confidence down.

        The predecessor had no override or undo path of any kind — a wrong
        reading was simply spoken and there was no way to say so.
        """
        return self._resolve(interaction_id, correct=False)

    def _resolve(self, interaction_id: Optional[str], correct: bool) -> bool:
        with self._lock:
            target: Optional[Interpretation] = None
            if interaction_id is None:
                target = self.history[-1] if self.history else None
            else:
                for item in reversed(self.history):
                    if item.interaction_id == interaction_id:
                        target = item
                        break
            if target is None or not target.input_data:
                logger.warning("No interaction to resolve (id=%r).", interaction_id)
                return False

            bucket = self.usage_stats.setdefault(SECTION[target.modality], {}).setdefault(
                target.input_data,
                {"count": 0, "confirmed": 0, "rejected": 0, "last_used": None},
            )
            bucket["confirmed" if correct else "rejected"] += 1

            self._maybe_update_confidence(target.modality, target.input_data, bucket)
        logger.info("Interaction %s marked %s", target.interaction_id,
                    "correct" if correct else "WRONG")
        return True

    def _maybe_update_confidence(
        self, modality: Modality, key: str, bucket: Dict[str, Any]
    ) -> None:
        """
        Move a prior, using ONLY confirmed and rejected outcomes.

        Caller holds the lock.

        The predecessor computed `success/count` where success never
        incremented, then gated the write behind `abs(delta) > 0.05` when the
        maximum possible delta was 0.05 — so it computed a decay it could
        never apply. This uses a stated step and a stated sample floor, and it
        does not touch a prior until a human has actually judged it.
        """
        resolved = int(bucket["confirmed"]) + int(bucket["rejected"])
        if resolved < MIN_CONFIRMED_SAMPLES:
            return

        entry = self.maps.get(modality, {}).get(key)
        if entry is None:
            return

        rate = int(bucket["confirmed"]) / resolved
        current = float(entry["confidence"])
        direction = 1.0 if rate >= 0.5 else -1.0
        # Step scales with how lopsided the evidence is: 50/50 moves nothing.
        magnitude = abs(rate - 0.5) * 2.0
        updated = current + direction * CONFIDENCE_STEP * magnitude
        updated = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, updated))

        if abs(updated - current) < 1e-9:
            return

        entry["confidence"] = updated
        logger.info(
            "Confidence for %s[%r]: %.3f -> %.3f (%d confirmed / %d resolved)",
            modality.value, key, current, updated, bucket["confirmed"], resolved,
        )

    # -- Background learning --------------------------------------------------

    def start_learning(self, interval_seconds: float = 300.0) -> bool:
        """Persist learned priors periodically. Off by default."""
        if self.data_dir is None:
            logger.error("start_learning refused: data_dir=None, nothing to persist to")
            return False
        if self._learning.is_set():
            return False
        self._learning.set()
        self._learning_thread = threading.Thread(
            target=self._learning_loop, args=(float(interval_seconds),),
            name="nonverbal-learning", daemon=True,
        )
        self._learning_thread.start()
        logger.info("Learning persistence started (every %.0fs)", interval_seconds)
        return True

    def stop_learning(self, timeout: float = 5.0) -> bool:
        """
        Stop and JOIN.

        The predecessor joined with a 5s timeout against a loop that slept 60s,
        returned True, and left the thread running. The wait is now on an
        Event, so clearing it wakes the loop immediately.
        """
        if not self._learning.is_set():
            return False
        self._learning.clear()
        thread = self._learning_thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.error("Learning thread did not stop within %.1fs.", timeout)
                return False
        self._learning_thread = None
        logger.info("Learning persistence stopped.")
        return True

    def _learning_loop(self, interval: float) -> None:
        while self._learning.is_set():
            # wait() returns as soon as the flag is cleared — no dead sleep.
            if self._learning.wait(timeout=interval):
                if not self._learning.is_set():
                    break
            try:
                self.save()
            except OSError as exc:
                logger.error("Periodic save failed: %s", exc)

    # -- Introspection --------------------------------------------------------

    def get_emotional_indicators(self, gesture_name: str) -> Dict[str, float]:
        """
        Emotional indicators for a gesture, or {} when there are none.

        `stimming` reads as self-regulation 0.9 — above anxiety and overwhelm —
        deliberately. See the module header.
        """
        return dict(EMOTIONAL_INDICATORS.get(gesture_name, {}))

    def set_confidence_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold {value} outside [0,1]")
        with self._lock:
            self.confidence_threshold = float(value)
        logger.info("Confidence threshold set to %.2f", value)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "threshold": self.confidence_threshold,
                "map_sizes": {m.value: len(t) for m, t in self.maps.items()},
                "learning_active": self._learning.is_set(),
                "history_depth": len(self.history),
                "data_dir": self.data_dir,
                "adds_random_noise": False,
                "speaks_for_unrecognized_input": False,
                # The bar labels readings; it never takes words away.
                "threshold_withholds_words": False,
            }


_engine: Optional[NonverbalEngine] = None
_engine_lock = threading.Lock()


def get_nonverbal_engine(**kwargs: Any) -> NonverbalEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = NonverbalEngine(**kwargs)
        return _engine


__all__ = [
    "NonverbalEngine", "Interpretation", "Modality", "Status",
    "MapValidationError", "validate_map", "get_nonverbal_engine",
    "DEFAULT_GESTURES", "DEFAULT_SYMBOLS", "DEFAULT_EYE_REGIONS",
    "DEFAULT_SOUNDS", "EMOTIONAL_INDICATORS", "VALID_TIERS",
    "MIN_CONFIRMED_SAMPLES", "CONFIDENCE_STEP", "TIE_TOLERANCE",
    "DISTRESS_INTENTS", "distress_readings",
    "DANGER_CATEGORY_INTENTS", "NEED_CATEGORY_INTENTS",
    "CATEGORY_DANGER", "CATEGORY_NEED", "CATEGORY_OTHER", "CATEGORIES",
    "is_danger", "danger_term", "category_of", "DANGER_WORDS",
    "GENERIC_FEELING_INTENTS", "NOT_LEARNED_REASON",
]

# ==============================================================================
# Patent Pending — The Christman AI Project / Luma Cognify AI
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
