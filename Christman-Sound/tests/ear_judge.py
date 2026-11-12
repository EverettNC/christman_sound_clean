# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The ear the voice tests are judged by.

The voice tests are heard by christman_voice_sdk.audio.ear, Vosk's recognizer (2026-11-05,
Everett: Vosk is the ear), in this process, on the model folder given. From
2026-10-29 to 2026-11-05 they were heard by a decoder of the package's own,
now held in held/claude-ear-2026-10-28. Nothing is played.

The ear hears 16-bit mono PCM at 16 kHz. A rendering at another rate is
resampled to 16 kHz with scipy's polyphase filter, and half a second of
digital silence is added at the end, as the old judge did, so the last word
is closed before the final answer is read.
"""

from __future__ import annotations

import io
import wave
from math import gcd
from pathlib import Path
from typing import Any, Iterable, List, Union

EAR_SR = 16000


def load(model_dir: Union[str, Path]) -> Any:
    """The model folder, read once. Raises with the ear's own reason."""
    from christman_voice_sdk.audio.ear import load_model

    return load_model(Path(model_dir).expanduser())


def _pcm16k(wav_bytes: bytes) -> bytes:
    import numpy as np
    from scipy.signal import resample_poly

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        if (w.getsampwidth(), w.getnchannels()) != (2, 1):
            raise ValueError(f"the ear takes 16-bit mono PCM (is {8 * w.getsampwidth()}-bit, "
                             f"{w.getnchannels()} channel(s))")
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if sr != EAR_SR:
        g = gcd(sr, EAR_SR)
        pcm = np.clip(np.round(resample_poly(pcm.astype(np.float64), EAR_SR // g, sr // g)),
                      -32768, 32767).astype("<i2")
    return pcm.tobytes() + b"\x00\x00" * (EAR_SR // 2)


def hear_bytes(model: Any, wav_bytes: bytes) -> str:
    """What the ear heard in one WAV, every utterance joined with one space."""
    from christman_voice_sdk.audio.ear import Ear

    ear = Ear(model, sample_rate=EAR_SR, words=True)
    return " ".join(h.text for h in ear.hear(_pcm16k(wav_bytes)) if h.text).strip().lower()


def hear_files(model: Any, paths: Iterable[Union[str, Path]]) -> List[str]:
    return [hear_bytes(model, Path(p).read_bytes()) for p in paths]
