# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training
# of this material is prohibited.
#
# Truth. Dignity. Protection. Transparency. No Erasure.
# Contact: contact@thechristmanaiproject.com
# https://thechristmanaiproject.com
# ==============================================================================

"""
Sound pattern recognition for nonverbal communicators.

WHAT CHANGED AND WHY
--------------------
The previous implementation invented sound patterns:

    sound_pattern = random.choice(["hum", "click", "distress", "soft", "loud"])
    confidence = random.uniform(0.6, 0.95)

One of those five is `distress`, which `classify_sound_intent` mapped to the
intent `help` at 0.9. So the service could manufacture a help request from a
person who made no sound at all, and hand it downstream carrying a number that
looked measured. `classify_sound_intent` then added `random.uniform(-0.1, 0.1)`
of jitter, which made a constant look like a reading.

Worse, the branch structure meant this ran even with a real engine attached:

    if self.using_real_recognition and self.last_detected_sound:
        ...
    else:
        # random generator lived here

Whenever no real detection was pending — the common case, most of the time —
control fell to `else` and the service fabricated one.

For a nonverbal client, an invented distress signal is not a glitch in a demo.
Either someone is dispatched to a person who never called, or the alerts get
noised out and a real one is ignored later. Both outcomes are produced by the
same three lines.

THE RULE THIS FILE NOW FOLLOWS
------------------------------
No detector, no detection. When no backend is attached, `detect_sound_pattern`
returns None forever and says so once at startup. It does not degrade into
simulation, because a fabricated signal is worse than no signal — no signal is
visibly absent, and a fabricated one is invisibly wrong.

WHAT CHANGED AND WHY (2026-11-04) — THE BEING DECIDES
-----------------------------------------------------
Everett, 2026-11-04: "we can have this knowledge in the sound package, but it
can't be making those decisions. It's up to the being to make those
decisions, not something that just creates a voice."

Before this, the `help` intent was an ESCALATING_INTENT: classify_sound_intent
handed it to an `escalation_callback` that had to report a human reached, and
raised EscalationNotDelivered when none was configured or it did not return
True. The service was deciding who gets paged. Gone: escalation_callback,
ESCALATING_INTENTS, _escalate, EscalationNotDelivered, and the
requires_escalation / escalated fields. Every intent now carries a category
label (INTENT_CATEGORY: `help` is "danger", the rest "other"), and the being
decides what a category calls for. Recognized speech also keeps the client's
transcript and the words that matched (`transcript`, `matched_tokens` in the
detection's metadata), so the being can read why a pattern was given.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .recognition_result import RecognitionResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------

#: Patterns this service can report. A backend returning anything outside this
#: set is rejected rather than passed through — an unknown label reaching intent
#: classification would fall to the "unknown" default and be treated as a real,
#: if uncertain, reading.
KNOWN_PATTERNS = frozenset({"hum", "click", "distress", "soft", "loud", "speech"})

#: Pattern -> intent. Deliberately carries NO confidence number.
#:
#: The old version attached one (distress -> help at 0.9) which conflated two
#: separate things: how sure the detector is that the sound occurred, and how
#: reliable this mapping is. The first is measurable and comes from the backend.
#: The second is a design decision made by people, not a measurement, and giving
#: it a decimal made it look like the former.
INTENT_MAP: Dict[str, str] = {
    "hum": "thinking",
    "click": "select",
    "distress": "help",
    "soft": "unsure",
    "loud": "excited",
    "speech": "communicate",
}

#: Intent -> category label: "danger" or "other". A label the being reads,
#: never an action taken here. `help` is "danger" because the nonverbal map
#: files the same intent there (nonverbal_engine's danger intents include
#: "help"); a word list written by people, not a measurement. Every intent in
#: INTENT_MAP has a category; a gap raises, it is never defaulted.
INTENT_CATEGORY: Dict[str, str] = {
    "thinking": "other",
    "select": "other",
    "help": "danger",
    "unsure": "other",
    "excited": "other",
    "communicate": "other",
}


# -----------------------------------------------------------------------------
# Backend protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class SoundDetectorBackend(Protocol):
    """
    What a real sound-pattern detector must provide.

    Implementations analyze audio. This service does not — it coordinates
    and classifies intent. Keeping the two apart is what makes it
    impossible for the coordination layer to quietly become the detector, which
    is precisely what happened before.
    """

    def analyze(self, audio_data: Any) -> Optional["SoundDetection"]:
        """
        Analyze audio and return a detection, or None if nothing was detected.

        Must return None rather than a low-confidence guess when no pattern is
        present. Must never fabricate.
        """
        ...


@dataclass(frozen=True)
class SoundDetection:
    """
    One detected sound pattern.

    Attributes:
        pattern: A member of KNOWN_PATTERNS.
        confidence: Measured confidence in [0.0, 1.0], or None when the backend
            reports no measurement. Never a constant chosen to look plausible.
        timestamp: Detection time (epoch seconds).
        source: Backend identity, for audit.
        metadata: Backend diagnostics.
    """

    pattern: str
    confidence: Optional[float]
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pattern not in KNOWN_PATTERNS:
            raise ValueError(
                f"Unknown sound pattern {self.pattern!r}. Known: "
                f"{sorted(KNOWN_PATTERNS)}. An unrecognized label would be "
                "classified as an uncertain-but-real intent downstream."
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            # Positive-form bound check: NaN fails it, where `< 0 or > 1` would
            # let NaN through.
            raise ValueError(
                f"confidence must be within [0.0, 1.0] or None, got "
                f"{self.confidence!r}."
            )


@dataclass(frozen=True)
class IntentClassification:
    """
    An intent derived from a detected pattern.

    `detection_confidence` is the backend's measurement of the *sound*, carried
    through unchanged. There is no separate "intent confidence", because no
    model measures one — INTENT_MAP is a fixed table written by people. Putting
    a number on it would be inventing a measurement.

    `category` is the intent's label from INTENT_CATEGORY ("danger" or
    "other"). It is a reading for the being; nothing here acts on it.
    """

    intent: str
    pattern: str
    detection_confidence: Optional[float]
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "pattern": self.pattern,
            "category": self.category,
            "detection_confidence": self.detection_confidence,
            "detection_confidence_known": self.detection_confidence is not None,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Service
# -----------------------------------------------------------------------------


class SoundRecognitionService:
    """
    Coordinates sound-pattern detection for nonverbal communicators.

    Detects nothing itself. Requires a backend implementing
    SoundDetectorBackend. With no backend it reports unavailable and returns
    None from every detection call — it never substitutes generated data.

    Thread safety: detection state is guarded by a lock. The previous version
    mutated `last_detected_sound` from a speech callback thread while
    `detect_sound_pattern` read it from another, which could drop or duplicate
    a detection.
    """

    def __init__(
        self,
        backend: Optional[SoundDetectorBackend] = None,
        speech_engine: Optional[Any] = None,
        detection_ttl_seconds: float = 2.0,
    ) -> None:
        """
        Args:
            backend: Real detector. Without it, nothing is ever detected.
            speech_engine: Optional recognizer. When present, recognized speech
                is folded in as a pattern read from its words (see
                _pattern_from_text), with the transcript kept.
            detection_ttl_seconds: How long a detection stays fresh. Older ones
                are dropped rather than replayed as current, except distress
                (see _take_pending).
        """
        self._lock = threading.Lock()
        self.backend = backend
        self.speech_engine = speech_engine
        self.detection_ttl_seconds = float(detection_ttl_seconds)

        #: Queue, not a slot. 2026-10-11: a single `_pending` slot meant a
        #: later utterance overwrote an unpolled distress detection, and a
        #: distress detection older than the TTL was dropped at DEBUG level.
        #: Now: FIFO, capped; stale non-distress is dropped with a WARNING;
        #: distress is never dropped for age.
        self._pending: List[SoundDetection] = []
        self._pending_cap = 16
        self._is_listening = False

        #: Backend failures are counted and the last one kept, so a crashed
        #: detector is distinguishable from silence in get_status().
        self.backend_failures = 0
        self.last_backend_error: Optional[str] = None
        self.dropped_detections = 0

        self.available = backend is not None or speech_engine is not None

        if not self.available:
            logger.error(
                "SoundRecognitionService has no detector backend and no speech "
                "engine. No sound pattern will ever be reported. This service "
                "does NOT simulate detections."
            )
        else:
            logger.info(
                "SoundRecognitionService ready. backend=%s speech_engine=%s",
                type(backend).__name__ if backend else None,
                type(speech_engine).__name__ if speech_engine else None,
            )

    # -- Detection ------------------------------------------------------------

    def detect_sound_pattern(
        self, audio_data: Optional[Any] = None
    ) -> Optional[SoundDetection]:
        """
        Return a detected sound pattern, or None.

        Returns None — and keeps returning None — when no backend is attached.
        There is no simulation branch.

        Args:
            audio_data: Audio to analyze. When omitted, only a pending
                detection queued by the speech callback can be returned.

        Returns:
            SoundDetection, or None when nothing was detected.
        """
        pending = self._take_pending()
        if pending is not None:
            return pending

        if self.backend is None or audio_data is None:
            return None

        try:
            detection = self.backend.analyze(audio_data)
        except Exception as exc:
            # A crashed backend is not a detection. Log, count, return
            # nothing — never substitute a guess for a failed analysis. The
            # count is what makes this distinguishable from "no sound".
            with self._lock:
                self.backend_failures += 1
                self.last_backend_error = f"{type(exc).__name__}: {exc}"
            logger.error("Sound detector backend failed: %s", exc, exc_info=True)
            return None

        if detection is None:
            return None

        if not isinstance(detection, SoundDetection):
            with self._lock:
                self.backend_failures += 1
                self.last_backend_error = (
                    f"contract violation: backend returned {type(detection).__name__}"
                )
            logger.error(
                "Backend returned %s, expected SoundDetection. Discarding.",
                type(detection).__name__,
            )
            return None

        logger.info(
            "Detected sound pattern: %s (confidence: %s)",
            detection.pattern,
            "unmeasured" if detection.confidence is None
            else f"{detection.confidence:.2f}",
        )
        return detection

    def _take_pending(self) -> Optional[SoundDetection]:
        """
        Pop the oldest fresh pending detection.

        Stale non-distress detections are dropped — with a WARNING and a
        count, not at DEBUG. A `distress` detection is never dropped for
        age: a help request three seconds old is still a help request.
        """
        while True:
            with self._lock:
                if not self._pending:
                    return None
                pending = self._pending.pop(0)

            age = time.time() - pending.timestamp
            if age > self.detection_ttl_seconds and pending.pattern != "distress":
                with self._lock:
                    self.dropped_detections += 1
                logger.warning(
                    "Dropping stale %s detection (%.2fs old, ttl %.2fs).",
                    pending.pattern, age, self.detection_ttl_seconds,
                )
                continue
            if age > self.detection_ttl_seconds:
                logger.warning(
                    "Delivering a %.2fs-old distress detection past the %.2fs "
                    "ttl — a help request is not dropped for age.",
                    age, self.detection_ttl_seconds,
                )
            return pending

    def _queue_pending(self, detection: SoundDetection) -> None:
        """
        Enqueue. When full, the oldest NON-distress entry is evicted; a
        distress detection is never evicted to make room.
        """
        with self._lock:
            self._pending.append(detection)
            if len(self._pending) > self._pending_cap:
                for i, item in enumerate(self._pending):
                    if item.pattern != "distress":
                        del self._pending[i]
                        self.dropped_detections += 1
                        logger.warning(
                            "Pending queue full (%d); evicted oldest %s detection.",
                            self._pending_cap, item.pattern,
                        )
                        break
                else:
                    logger.error(
                        "Pending queue holds %d undelivered distress detections. "
                        "Nothing is polling detect_sound_pattern().",
                        len(self._pending),
                    )

    # -- Intent ---------------------------------------------------------------

    def classify_sound_intent(
        self, detection: Optional[SoundDetection]
    ) -> Optional[IntentClassification]:
        """
        Map a detection to an intent and its category label. Reads only:
        nothing is sent anywhere, and nothing here acts on the category. The
        being decides what a "danger" intent calls for.

        No randomness anywhere. The detection's measured confidence is carried
        through unchanged; nothing is added, jittered, or defaulted.

        Args:
            detection: A SoundDetection, or None.

        Returns:
            IntentClassification, or None when given None.

        Raises:
            ValueError: when KNOWN_PATTERNS, INTENT_MAP and INTENT_CATEGORY
                have drifted apart. A gap is never defaulted.
        """
        if detection is None:
            return None

        intent = INTENT_MAP.get(detection.pattern)
        if intent is None:
            # Unreachable while SoundDetection validates its pattern, but if
            # KNOWN_PATTERNS and INTENT_MAP ever drift apart, this must fail
            # loudly rather than default to a plausible-looking "unknown".
            raise ValueError(
                f"Pattern {detection.pattern!r} is in KNOWN_PATTERNS but absent "
                "from INTENT_MAP. The two have drifted apart."
            )
        category = INTENT_CATEGORY.get(intent)
        if category is None:
            raise ValueError(
                f"Intent {intent!r} is in INTENT_MAP but absent from "
                "INTENT_CATEGORY. The two have drifted apart."
            )

        classification = IntentClassification(
            intent=intent,
            pattern=detection.pattern,
            detection_confidence=detection.confidence,
            category=category,
            metadata={
                "source": detection.source,
                "timestamp": detection.timestamp,
                **detection.metadata,
            },
        )
        logger.debug(
            "Classified %s as %s (category %s).", detection.pattern, intent, category
        )
        return classification

    # -- Lifecycle ------------------------------------------------------------

    def start_listening(self) -> bool:
        """
        Start the attached speech engine, if any.

        Returns:
            True if something was actually started. False when there is nothing
            to start — the old version logged "Started simulated sound
            recognition" and returned as though it had.
        """
        if self._is_listening:
            logger.warning("Already listening.")
            return False

        if self.speech_engine is None:
            logger.error(
                "start_listening() called with no speech engine. Nothing "
                "started. No simulation will run."
            )
            return False

        # Rule 6 (2026-10-29 review): an engine that failed to start was
        # logged and reported as False, the same answer as "no engine
        # attached". Now the failure is raised, with the reason.
        try:
            self.speech_engine.start_listening(callback=self._on_speech)
        except Exception as exc:
            raise RuntimeError(
                f"the speech engine failed to start ({type(exc).__name__}: {exc})") from exc

        self._is_listening = True
        logger.info("Speech recognition started.")
        return True

    def stop_listening(self) -> bool:
        """Stop the attached speech engine. False if it was not running."""
        if not self._is_listening:
            logger.warning("Not listening.")
            return False

        self._is_listening = False
        with self._lock:
            undelivered = [d for d in self._pending if d.pattern == "distress"]
            self._pending.clear()
        if undelivered:
            logger.error(
                "stop_listening() discarded %d undelivered distress detection(s).",
                len(undelivered),
            )

        if self.speech_engine is not None:
            try:
                self.speech_engine.stop_listening()
            except Exception as exc:
                logger.error("Error stopping speech engine: %s", exc, exc_info=True)
                return False

        logger.info("Speech recognition stopped.")
        return True

    # -- Speech bridge --------------------------------------------------------

    def _on_speech(self, result: RecognitionResult) -> None:
        """
        Fold recognized speech in as a `speech` pattern.

        Accepts a RecognitionResult rather than the old (text, confidence,
        metadata) triple, so a failure can no longer arrive looking like an
        utterance. Anything that is not real client speech is ignored.
        """
        if not isinstance(result, RecognitionResult):
            logger.error(
                "Speech callback received %s, expected RecognitionResult. "
                "Ignoring.",
                type(result).__name__,
            )
            return

        if not result.is_user_speech:
            logger.debug("Ignoring non-speech result: %s", result.status.value)
            return

        pattern, matched = self._pattern_from_text(result.text)

        detection = SoundDetection(
            pattern=pattern,
            confidence=result.confidence,   # measured or None, carried as-is
            timestamp=time.time(),
            source=f"speech:{result.source}",
            # The client's words and the words that set the pattern, so the
            # being reads why it was given (2026-11-04).
            metadata={
                "matched_from_text": True,
                "transcript": result.text,
                "matched_tokens": matched,
            },
        )

        self._queue_pending(detection)

    @staticmethod
    def _pattern_from_text(text: str) -> Tuple[str, List[str]]:
        """
        Map recognized words to a sound pattern.

        Returns (pattern, matched_tokens): the pattern, and the client's words
        that set it, in the order said, each once. matched_tokens is empty
        for plain `speech`, which no word list set.

        Two defects in the original are fixed here.

        1. Substring matching. It used `word in text.lower()`, so "helpful"
           contained "help" and promoted an ordinary sentence to a distress
           signal — the same class of error as matching "kill" inside "skill".
           Matching is now on whitespace tokens with punctuation stripped.

        2. Demonstratives in the select set. The original matched
           {"select", "this", "that", "click"}. "this" and "that" are among the
           most common words in English, so nearly any sentence classified as a
           select action. They are removed; only unambiguous action words remain.

        The two sets are tuned in OPPOSITE directions on purpose, because their
        errors do not cost the same thing:

        - SELECT is tuned to avoid false positives. A wrongly-fired select
          performs an action the client did not ask for, and there is no undo
          path in this stack. Silence is the safer error.
        - DISTRESS is tuned to avoid false negatives. A missed help request from
          someone who cannot repeat it is the worst outcome available here, so
          "need" stays in the set even though it will catch "I need a blanket."
          That labels a blanket request distress. The label is a reading, not
          an action: the detection carries the transcript and the word that
          matched, and the being reads them and decides what the words call
          for (2026-11-04).

        That asymmetry is a judgment about harm, not a measurement. It is
        written here so it can be argued with rather than discovered later.
        """
        said: List[str] = []
        for token in text.split():
            word = token.strip(".,!?;:\"'").lower()
            if word and word not in said:
                said.append(word)

        def matching(words: frozenset) -> List[str]:
            return [w for w in said if w in words]

        # Recall-weighted: better to label a blanket request distress, with
        # the word that matched, than to miss a real one.
        found = matching(frozenset({"help", "need", "urgent", "emergency", "hurt", "pain", "stop"}))
        if found:
            return "distress", found
        # Precision-weighted: unambiguous action verbs only. No demonstratives.
        found = matching(frozenset({"select", "click", "choose", "pick"}))
        if found:
            return "click", found
        found = matching(frozenset({"hmm", "hm", "um", "uh", "thinking"}))
        if found:
            return "hum", found
        return "speech", []

    # -- Status ---------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Current service state, for health checks and audit."""
        with self._lock:
            pending_count = len(self._pending)
            pending_distress = sum(1 for d in self._pending if d.pattern == "distress")
            failures = self.backend_failures
            last_error = self.last_backend_error
            dropped = self.dropped_detections

        return {
            "available": self.available,
            "is_listening": self._is_listening,
            "has_backend": self.backend is not None,
            "has_speech_engine": self.speech_engine is not None,
            "has_pending_detection": pending_count > 0,
            "pending_detections": pending_count,
            "pending_distress": pending_distress,
            "backend_failures": failures,
            "last_backend_error": last_error,
            "dropped_detections": dropped,
            "simulation_supported": False,
        }


__all__ = [
    "SoundRecognitionService",
    "SoundDetectorBackend",
    "SoundDetection",
    "IntentClassification",
    "KNOWN_PATTERNS",
    "INTENT_MAP",
    "INTENT_CATEGORY",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
