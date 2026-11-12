"""
Shorty Voice Engine V3 - ULTRA Tier Implementation

WHAT CHANGED AND WHY (2026-10-17) — OUTSIDE VOICE CUT
-----------------------------------------------------
Everett's ruling: the rebuild cuts every outside speech engine. This
class drove Coqui XTTS to render; that render call is the only thing that
came out. Removed (quoted from main 99a6cce):

    from ..engines.synthesis_result import SOVEREIGN_NOT_WIRED, SynthesisUnavailable
    ...
    logger.info("Loading XTTS voice synthesis engine...")
    self.xtts = XTTSEngine(device=device)
    ...
    self.xtts.load_voice(reference_audio)
    ...
    result = self.xtts.synthesize(
        text=text,
        emotion_params=quant["voice_params"],
        **kwargs
    )
    return result

Kept whole: Everett's emotion quantification — SHORTY_EMOTIONS,
`quantify_emotion`, `_get_emotion_score`, `_emotion_to_voice_params` — the
emotion detector and the measured baseline from the reference clip.

`synthesize` still quantifies the emotion first (so a bad emotion or a
missing baseline fails exactly as before), then raises
`SynthesisUnavailable(SOVEREIGN_NOT_WIRED)` (this path is retired; the voice is voice_synthesis) carrying the
quantified voice parameters on the exception as `voice_params`. It never
returns a result that pretends to be speech.

WHAT CHANGED AND WHY (2026-11-12) — EVERETT'S OWN SYNTHESIZER, WHERE XTTS WAS
------------------------------------------------------------------------------
Everett: "why would I need to use someone else's XTTS when we have our own?"
The render call that came out on 2026-10-17 was never replaced; `synthesize`
quantified the emotion and then raised. It now renders through the SDK's own
mouth (voice_synthesis -> synth -> christman_synth.c / christman_vocoder.c)
in the voice the caller passes (`voice=` to the constructor or to
`synthesize`). There is no default voice: with none, it raises
SynthesisUnavailable and says so, still carrying `voice_params`.

How Everett's voice parameters reach the sound, each one applied or refused,
never dropped:
  tempo_factor  -> the mouth's `rate` control (it multiplies the voice's
                   measured speaking rate; the mouth refuses a value outside
                   its range and says which).
  pitch_shift   -> the mouth's `f0` control: the voice's own measured median
                   pitch moved by that many semitones.
  energy_boost  -> the samples scaled by that factor; if that would pass full
                   scale, the whole utterance is scaled to full scale instead
                   and `energy_boost_applied` in the result says what it was.
These need a frame voice. A measured formant profile has no input for pitch
or rate, so a non-neutral parameter with one is refused with the reason.

EARLIER HISTORY OF THIS FILE
----------------------------
1. FABRICATED BASELINES ERADICATED.
   The `_get_default_baseline()` method invented highly specific decimal scores 
   (e.g., 'happy': 0.9123) if the audio analysis failed. An analysis that fails 
   must not invent data. If there is no baseline, it stays None.

2. UNKNOWN EMOTIONS NO LONGER DEFAULT TO 0.5000.
   If an emotion was requested but unknown, `quantify_emotion` forced a 0.5 score.
   It now explicitly raises a ValueError.

3. SECURITY (Rule 12): SECURED TORCH.LOAD.
   Added `weights_only=True` to `torch.load` to prevent arbitrary code execution 
   from pickled tensors.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Any, Optional, Dict, List
import time

from ..engines.synthesis_result import SynthesisUnavailable
from ..timbre.shorty_emotion import ShortyEmotionDetector
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ShortyVoiceEngineV3:
    SHORTY_EMOTIONS = [
        "neutral", "happy", "proud", "teasing", "annoyed",
        "sarcastic", "sweetheart", "laugh", "tremble",
        "emphasis", "last_breath"
    ]

    def __init__(
        self,
        reference_audio: Optional[Path] = None,
        pca_model_path: str = "models/shorty_emotion_pca.pt",
        scaler_path: str = "models/shorty_emotion_scaler.pt",
        device: str = "auto",
        voice: Any = None,
    ):
        self.device = device
        #: The caller's own voice: a frame voice folder, a voice profile file,
        #: or one already loaded. None until the caller gives one.
        self.voice = voice
        self._mouth = None
        logger.info("Loading Shorty's emotion detector...")
        self.emotion_detector = ShortyEmotionDetector(
            pca_model_path=pca_model_path,
            scaler_path=scaler_path
        )

        self.reference_audio = None
        self.emotion_baseline = None

        if reference_audio:
            self.load_voice(reference_audio)
        else:
            logger.info("Shorty voice engine initialized (no reference loaded)")

    def load_voice(self, reference_audio: Path):
        logger.info(f"Loading Shorty's voice from {reference_audio.name}")

        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        self.reference_audio = reference_audio

        logger.info("Analyzing reference audio for emotional baseline...")
        try:
            emotion_result = self.emotion_detector.detect_emotion_from_audio(
                str(reference_audio)
            )
            self.emotion_baseline = emotion_result['scores']
            logger.info(f"Baseline emotion: {emotion_result['dominant_emotion']} "
                       f"({emotion_result['confidence']:.2%})")
            
        except Exception as e:
            # FIX: Fail honestly. Do not generate a fake baseline.
            logger.error(f"Could not analyze emotion baseline: {e}. Baseline is unavailable.")
            self.emotion_baseline = None

    def synthesize(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        *,
        voice: Any = None,
        **kwargs
    ) -> Any:
        """
        Quantify the emotion, then speak `text` through the SDK's own mouth in
        the caller's voice, with Everett's voice parameters applied (see the
        note at the top of this file). Returns the mouth's SynthesisResult,
        with `voice_params`, `applied` and `energy_boost_applied` on it.
        `kwargs` are the mouth's own named controls, passed through.
        """
        emotion = "neutral"
        exaggeration = 0.0

        if emotion_params:
            emotion = emotion_params.get("emotion", "neutral")
            exaggeration = emotion_params.get("exaggeration", 0.0)

        quant = self.quantify_emotion(text, emotion, exaggeration)
        params = quant["voice_params"]
        where = "ShortyVoiceEngineV3.synthesize"

        def refuse(reason: str, cause: Optional[BaseException] = None):
            err = SynthesisUnavailable(reason, where=where)
            err.voice_params = params
            logger.error("Shorty did not speak %r: %s", (text or "")[:40], err)
            if cause is not None:
                raise err from cause
            raise err

        chosen = voice if voice is not None else getattr(self, "voice", None)
        if chosen is None:
            refuse("no voice given: pass the caller's own voice (voice=<frame voice folder or voice profile file>, "
                   "or a loaded FrameVoice or VoiceProfile); there is no default voice")

        from .voice_synthesis import SpeechSynthesisEngine, resolve_voice
        from .stitch import FrameVoice
        try:
            loaded = resolve_voice(chosen)
        except SynthesisUnavailable as exc:
            exc.voice_params = params
            raise

        controls = dict(kwargs)
        tempo = float(params["tempo_factor"])
        shift = float(params["pitch_shift"])
        boost = float(params["energy_boost"])
        if isinstance(loaded, FrameVoice):
            if tempo != 1.0:
                controls.setdefault("rate", tempo)
            if shift != 0.0:
                controls.setdefault("f0", float(loaded.measures["f0_median_hz"]) * 2 ** (shift / 12.0))
        elif tempo != 1.0 or shift != 0.0:
            refuse(f"the voice {loaded.name} is a measured formant profile, which has no input for pitch or rate; "
                   f"pitch_shift {shift:+.2f} st and tempo_factor {tempo:.2f} need a frame voice")

        if getattr(self, "_mouth", None) is None:
            self._mouth = SpeechSynthesisEngine()
        result = self._mouth._render(text, None, "en", voice=loaded, controls=controls)

        applied_boost = boost
        if boost != 1.0 and result.audio is not None and len(result.audio) > 0:
            peak = max(abs(s) for s in result.audio)
            if peak > 0 and peak * boost > 1.0:
                applied_boost = 1.0 / peak
            for i in range(len(result.audio)):
                result.audio[i] = result.audio[i] * applied_boost
        result.voice_params = params
        result.energy_boost_applied = applied_boost
        result.applied = list(self._mouth.last_applied or [])
        return result

    def quantify_emotion(
        self,
        text: str,
        emotion: str = "neutral",
        exaggeration: float = 0.0,
        analyze_audio: Optional[Path] = None
    ) -> Dict:
        if analyze_audio and analyze_audio.exists():
            try:
                result = self.emotion_detector.detect_emotion_from_audio(str(analyze_audio))
                emotion = result['dominant_emotion']
                base_score = result['confidence']
            except Exception as e:
                logger.error(f"Emotion analysis failed: {e}")
                raise RuntimeError(f"Failed to extract emotion from audio: {e}")
        else:
            base_score = self._get_emotion_score(emotion)

        if emotion not in self.SHORTY_EMOTIONS:
            raise ValueError(f"Unknown emotion '{emotion}'. Cannot quantify.")

        exaggeration = max(-1.0, min(1.0, exaggeration))

        if exaggeration >= 0:
            adjusted_score = base_score + (1.0 - base_score) * exaggeration * 0.5
        else:
            adjusted_score = base_score + (base_score - 0.5) * exaggeration

        adjusted_score = max(0.0, min(1.0, adjusted_score))
        voice_params = self._emotion_to_voice_params(emotion, adjusted_score, exaggeration)

        return {
            "emotion": emotion,
            "base_score": round(base_score, 4),
            "adjusted_score": round(adjusted_score, 4),
            "exaggeration": round(exaggeration, 4),
            "voice_params": voice_params
        }

    def _get_emotion_score(self, emotion: str) -> float:
        """Returns the actual measured baseline, or explicitly fails."""
        if self.emotion_baseline and emotion in self.emotion_baseline:
            return self.emotion_baseline[emotion]
        raise ValueError(f"No baseline established for emotion: {emotion}. Call load_voice() with a valid sample first.")

    def _emotion_to_voice_params(self, emotion: str, score: float, exaggeration: float) -> Dict:
        params = {"pitch_shift": 0.0, "tempo_factor": 1.0, "energy_boost": 1.0}
        intensity = score * (1.0 + exaggeration * 0.5)

        if emotion == "happy":
            params["pitch_shift"] = 1.0 * intensity + (exaggeration * 1.5)
            params["tempo_factor"] = 1.0 + (0.05 * intensity) + (exaggeration * 0.15)
            params["energy_boost"] = 1.0 + (0.1 * intensity) + (exaggeration * 0.3)
        elif emotion == "last_breath":
            params["pitch_shift"] = -3.0 - (exaggeration * 1.0)
            params["tempo_factor"] = 0.6 - (exaggeration * 0.2)
            params["energy_boost"] = 0.4 - (exaggeration * 0.3)
        else:
            params["pitch_shift"] = 0.5 * intensity
            params["energy_boost"] = 1.0 + (0.1 * intensity)

        params["pitch_shift"] = max(-12.0, min(12.0, params["pitch_shift"]))
        params["tempo_factor"] = max(0.5, min(2.0, params["tempo_factor"]))
        params["energy_boost"] = max(0.1, min(2.0, params["energy_boost"]))

        return params
