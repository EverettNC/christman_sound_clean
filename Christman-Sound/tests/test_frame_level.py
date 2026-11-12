# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
A frame voice speaks at the level of its recordings.

christman_vocoder does not normalise: "its level is the recordings' level,
which is part of the voice" (christman_vocoder.c). That holds only when the
frame analyzer stores each envelope in the vocoder's units, power per sample.
Until 2026-11-12 it stored them 24 to 28 dB low (see _envelope_at in
christman_voice_sdk/synthesis/frames.py), and a voice built from a recording
at -15 dB spoke at -41 dB. These tests analyse a signal of known level and
hold the analyzer and the vocoder to it.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.synthesis.frames import FRAME_SR, HOP, analyze_frames, env_grid_hz  # noqa: E402


def _db(x) -> float:
    return 10.0 * math.log10(sum(v * v for v in x) / len(x) + 1e-18)


def _implied_power_db(env_db, grid_hz, sr) -> float:
    """The power per sample an envelope holds: its mean over linear frequency."""
    total, bins = 0.0, 512
    j = 0
    for k in range(bins + 1):
        hz = k * (sr / 2) / bins
        while j + 1 < len(grid_hz) - 1 and grid_hz[j + 1] < hz:
            j += 1
        span = grid_hz[j + 1] - grid_hz[j]
        frac = min(1.0, max(0.0, (hz - grid_hz[j]) / span)) if span > 0 else 0.0
        total += 10.0 ** ((env_db[j] * (1 - frac) + env_db[j + 1] * frac) / 10.0)
    return 10.0 * math.log10(total / (bins + 1))


def _signals():
    rng = random.Random(20261112)
    n = FRAME_SR  # one second
    noise = [rng.gauss(0.0, 0.1) for _ in range(n)]
    # A buzz at 110 Hz: 30 equal harmonics, the kind of sound a voiced frame is.
    buzz = [0.02 * sum(math.sin(2 * math.pi * 110.0 * h * i / FRAME_SR) for h in range(1, 31)) for i in range(n)]
    return {"noise": noise, "buzz": buzz}


@pytest.mark.parametrize("kind", ["noise", "buzz"])
def test_the_envelope_holds_the_signals_own_power(kind):
    x = _signals()[kind]
    track = analyze_frames(x, sr=FRAME_SR, hop=HOP)
    grid = env_grid_hz(FRAME_SR)
    middle = range(track.n // 4, 3 * track.n // 4, 7)
    implied = sorted(_implied_power_db(track.env_db[i], grid, FRAME_SR) for i in middle)
    median = implied[len(implied) // 2]
    assert abs(median - _db(x)) < 1.5, (
        f"{kind}: the envelopes hold {median:.1f} dB, the signal is {_db(x):.1f} dB")


@pytest.mark.needs_dsp
@pytest.mark.parametrize("kind", ["noise", "buzz"])
def test_frames_resynthesized_come_out_at_the_recordings_level(kind):
    from christman_voice_sdk.engines.dsp import get_dsp
    from christman_voice_sdk.synthesis.frames import AP_EDGES

    x = _signals()[kind]
    track = analyze_frames(x, sr=FRAME_SR, hop=HOP)
    env = [v for row in track.env_db for v in row]
    ap = [v for row in track.ap for v in row]
    y = get_dsp().vocoder_render(track.f0, env, env_grid_hz(FRAME_SR), ap, list(AP_EDGES), HOP, FRAME_SR, 0.0, 0.0, 1)
    lo, hi = len(y) // 4, 3 * len(y) // 4
    assert abs(_db(y[lo:hi]) - _db(x[lo:hi])) < 2.0, (
        f"{kind}: resynthesized at {_db(y[lo:hi]):.1f} dB, the recording is {_db(x[lo:hi]):.1f} dB")
