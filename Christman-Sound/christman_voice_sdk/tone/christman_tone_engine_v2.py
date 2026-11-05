"""
CHRISTMAN EMOTION & TONESCORE ENGINE v2
"Nothing Vital Lives Below Root"

Raw audio -> physiological intensity + classifier emotion. Readings only:
what to do with them is the being's decision (see 6 below).

Sovereign implementation: scipy + numpy, no librosa.

WHAT CHANGED AND WHY
--------------------

1. THE DISTRESS PATH WAS UNREACHABLE.

   `EMOTION_LABELS` (:31) declared ELEVEN names. The model,
   `superb/wav2vec2-base-superb-er`, emits FOUR:

       id2label: {'0':'neu','1':'hap','2':'ang','3':'sad'}

   `enumerate(probabilities)` yields four items, so indices 4-10 were never
   assigned. Those seven include `tremble` and `last_breath` — and

       if dominant_emotion in ["tremble", "last_breath"] ...:
           action_state = "HOLD_SPACE"                          # :89

   was the only emotion route to HOLD_SPACE. **It could not fire.** The single
   branch in this file meant to catch someone in crisis had no path to execute.

   Of the four that did get assigned, two were wrong: `ang -> "proud"` and
   `sad -> "teasing"`. Anger read as pride. Sadness read as teasing.

   Labels now come from `model.config.id2label`. The engine reports at startup
   which requested states the model cannot produce, instead of silently never
   producing them.

2. STEREO WAS DESTROYED (:53).

   `np.mean(y, axis=0)` on a `(frames, channels)` array averages over FRAMES —
   it collapses the file to one number per channel. axis=1. REMEDIATION
   Phase 2 lists this.

3. `intensity_norm = np.clip(rms_energy * 400, 0, 1)` (:65) — the 400 is
   uncalibrated and unexplained, and it was the ONLY surviving trigger for
   HOLD_SPACE. It is named an uncalibrated chosen value and exposed in the
   result. Since 2026-11-04 it triggers nothing: it scales the `intensity`
   reading and that is all (see 6).

4. `except Exception: return None` (:104) swallowed every failure into a bare
   None. A caller could not tell a missing file from a corrupt model. Failures
   are now typed results.

5. The cadence fingerprint hashed raw audio bytes — a stable identifier derived
   from a person's voice. Retained, but off unless asked for, and documented as
   biometric-adjacent.

WHAT CHANGED AND WHY (2026-11-04)
---------------------------------

6. THE ENGINE NO LONGER ROUTES. Everett's ruling, 2026-11-04: "we can have
   this knowledge in the sound package, but it can't be making those
   decisions. It's up to the being to make those decisions, not something
   that just creates a voice."

   `action_state` (NORMAL / HOLD_SPACE / UNKNOWN), `_route`,
   `HOLD_SPACE_STATES` and `INTENSITY_HOLD_THRESHOLD` are gone, and so is the
   startup error that existed only to say no emotion could reach HOLD_SPACE.
   The result carries the readings: `intensity` (was `physical_intensity`),
   `dominant_state` and `raw_scores`. "We could not read the emotion" is still
   distinguishable from "read and found neutral": `raw_scores` is None and
   `emotions_available` is False, with the reason in `notes`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal

from .emotion_labels import (
    LabelResolutionError,
    LabelSet,
    dominant,
    label_probabilities,
    resolve_labels,
    unavailable_classes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: States this engine would report if a model could produce them.
DESIRED_STATES: List[str] = [
    "neutral", "happy", "proud", "teasing", "annoyed", "sarcastic",
    "sweetheart", "laugh", "tremble", "emphasis", "last_breath",
]

#: RMS -> 0-1 intensity. A chosen starting value (reason: carried over from the
#: original engine; it is uncalibrated and there is no measurement behind it).
#: Held until client data replaces it.
INTENSITY_SCALE = 400.0


@dataclass(frozen=True)
class ToneEngineResult:
    """
    One analysis: readings only. Optional fields are None when unmeasured.
    What to do with a reading is the being's decision (Everett, 2026-11-04).
    """

    status: str                       # ok | unavailable | error
    modality: str = "audio"
    dominant_state: Optional[str] = None
    intensity: Optional[float] = None  # RMS * INTENSITY_SCALE, clipped 0-1; uncalibrated
    raw_scores: Optional[Dict[str, float]] = None
    cadence_fingerprint: Optional[str] = None
    label_set: Optional[Dict[str, Any]] = None
    unavailable_states: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "modality": self.modality,
            "dominant_state": self.dominant_state,
            "intensity": self.intensity,
            "intensity_is_uncalibrated": True,
            "intensity_scale": INTENSITY_SCALE,
            "raw_scores": self.raw_scores,
            "emotions_available": self.raw_scores is not None,
            "cadence_fingerprint": self.cadence_fingerprint,
            "label_set": self.label_set,
            "unavailable_states": list(self.unavailable_states),
            "notes": list(self.notes),
        }


class ChristmanToneEngine:
    """
    Physiological intensity plus paralinguistic emotion, reported as readings.

    The engine routes nothing and decides nothing (Everett, 2026-11-04): the
    being reads `intensity`, `dominant_state` and `raw_scores` and decides.
    "We could not read the emotion" is `raw_scores` None, never a neutral.
    """

    def __init__(
        self,
        model_name: str = "superb/wav2vec2-base-superb-er",
        require_model: bool = False,
        emit_fingerprint: bool = False,
    ) -> None:
        """
        Args:
            require_model: raise if the classifier will not load.
            emit_fingerprint: include a SHA1 of the waveform. OFF by default —
                a stable hash of someone's voice is biometric-adjacent, and it
                was previously emitted on every call.
        """
        self.model_name = model_name
        self.emit_fingerprint = emit_fingerprint
        self.processor = None
        self.model = None
        self.device = None
        self.labels: Optional[LabelSet] = None
        self.unavailable_reason: Optional[str] = None
        self.unavailable_states: List[str] = list(DESIRED_STATES)

        try:
            import torch
            from transformers import (
                Wav2Vec2FeatureExtractor,
                Wav2Vec2ForSequenceClassification,
            )
        except ImportError as exc:
            self.unavailable_reason = f"torch/transformers not installed: {exc}"
            logger.error(self.unavailable_reason)
            if require_model:
                raise
            return

        try:
            # Offline only (Everett's ruling): read from this machine, never fetched.
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, local_files_only=True)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, local_files_only=True)
            self.model.eval()
            self.device = torch.device(
                "mps" if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available() else "cpu"
            )
            self.model.to(self.device)
            self.labels = resolve_labels(self.model, model_name)

            # Say out loud what this model CANNOT produce, at startup.
            self.unavailable_states = unavailable_classes(self.labels, DESIRED_STATES)
            if self.unavailable_states:
                logger.warning(
                    "%s cannot produce %d of %d desired states: %s. They will "
                    "never appear in dominant_state or raw_scores.",
                    model_name, len(self.unavailable_states), len(DESIRED_STATES),
                    self.unavailable_states,
                )
        except (Exception, LabelResolutionError) as exc:
            self.processor = self.model = self.labels = None
            self.unavailable_reason = f"failed to load {model_name}: {exc}"
            logger.error(self.unavailable_reason)
            if require_model:
                raise

    @property
    def emotions_available(self) -> bool:
        return self.model is not None and self.labels is not None

    def analyze_audio(self, wav_path: str) -> ToneEngineResult:
        """Analyze a wav file. Returns a typed result; never a bare None."""
        notes: List[str] = []

        try:
            sr, raw = wavfile.read(wav_path)
        except FileNotFoundError:
            return ToneEngineResult(status="error", notes=[f"file not found: {wav_path}"])
        except Exception as exc:
            return ToneEngineResult(status="error", notes=[f"failed to read audio: {exc}"])

        y = raw.astype(np.float32) / 32768.0 if raw.dtype == np.int16 \
            else raw.astype(np.float32)

        if y.ndim > 1:
            # axis=1. axis=0 averages over frames and destroys the audio.
            y = np.mean(y, axis=1)
        if y.size == 0:
            return ToneEngineResult(status="error", notes=["file contains no samples"])

        if sr != 16000:
            # Polyphase resampling. The original used np.interp on a linspace,
            # which is linear interpolation with no anti-aliasing — it folds
            # high frequencies down into the band the classifier reads.
            target = int(round(len(y) * 16000 / float(sr)))
            if target <= 0:
                return ToneEngineResult(status="error", notes=["resample yields no samples"])
            y = scipy_signal.resample(y, target).astype(np.float32)
            notes.append(f"resampled {sr}Hz -> 16000Hz")
            sr = 16000

        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        intensity = float(np.clip(rms * INTENSITY_SCALE, 0.0, 1.0))

        scores, emo_note = self._classify(y, sr)
        if emo_note:
            notes.append(emo_note)

        dom = dominant(scores) if scores else None

        if scores is None:
            notes.append(
                "emotion unavailable — only intensity was measured"
            )

        return ToneEngineResult(
            status="ok",
            dominant_state=dom,
            intensity=intensity,
            raw_scores=scores,
            cadence_fingerprint=(
                hashlib.sha1(y.tobytes()).hexdigest()[:16]
                if self.emit_fingerprint else None
            ),
            label_set=self.labels.to_dict() if self.labels else None,
            unavailable_states=list(self.unavailable_states),
            notes=notes,
        )

    def _classify(self, y: np.ndarray, sr: int):
        """(scores, None) or (None, reason)."""
        if not self.emotions_available:
            return None, self.unavailable_reason or "emotion model not loaded"
        try:
            import torch

            inputs = self.processor(
                y, sampling_rate=sr, return_tensors="pt", padding=True
            ).input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            return label_probabilities(self.labels, probs), None
        except LabelResolutionError as exc:
            logger.error("Label mismatch: %s", exc)
            return None, f"label mismatch: {exc}"
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            return None, f"classification failed: {exc}"


__all__ = [
    "ChristmanToneEngine", "ToneEngineResult", "DESIRED_STATES",
    "INTENSITY_SCALE",
]
