# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
measure_vowels_from_frame_bank (2026-11-05): a voice's vowels measured inside
the phone units its frame bank holds. The recordings here are made in the test
with known formants (the same source-filter generator test_formants.py uses), a
bank.json names where each vowel sits, and the measurement must find the
formants that were put there, within test_formants.py's tolerances.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import wave
from array import array
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.engines.dsp import get_dsp  # noqa: E402

if not get_dsp().available:
    pytest.skip(f"not run: christman_dsp is not built here ({get_dsp().error}); "
                "build it with python3 scripts/build_dsp.py", allow_module_level=True)

from christman_voice_sdk.synthesis.stitch import FrameVoiceUnavailable  # noqa: E402
from voice_creation_center.booth import (  # noqa: E402
    BANK_VOWEL_MIN_UNITS,
    measure_vowels_from_frame_bank,
)
from christman_voice_sdk.synthesis.voice_profile import load_voice_profile  # noqa: E402

SR, HOP, F0 = 16000, 80, 120.0
BW = (80.0, 100.0, 160.0)


def _voiced(formant_track, seconds):
    """A voiced stretch whose formants follow formant_track(t in 0..1) -> (F1, F2, F3)."""
    n = int(SR * seconds)
    tp, tn = 0.6 * 0.66, 0.6 * 0.34
    phase, flow = 0.0, []
    for _ in range(n):
        phase = (phase + F0 / SR) % 1.0
        flow.append(0.5 * (1 - math.cos(math.pi * phase / tp)) if phase < tp
                    else math.cos(math.pi * (phase - tp) / (2 * tn)) if phase < tp + tn else 0.0)
    y = [flow[0]] + [flow[i] - flow[i - 1] for i in range(1, n)]
    for j in range(3):
        out, y1, y2 = [0.0] * n, 0.0, 0.0
        for i in range(n):
            r = math.exp(-math.pi * BW[j] / SR)
            c1, c2 = 2 * r * math.cos(2 * math.pi * formant_track(i / n)[j] / SR), -r * r
            v = (1 - c1 - c2) * y[i] + c1 * y1 + c2 * y2
            out[i], y2, y1 = v, y1, v
        y = out
    peak = max(abs(v) for v in y) or 1.0
    return [0.5 * v / peak for v in y]


def _write(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(array("h", (int(max(-1, min(1, v)) * 32767) for v in samples)).tobytes())


AA = (730.0, 1090.0, 2440.0)
IY = (270.0, 2290.0, 3010.0)
AY_ON, AY_OFF = (730.0, 1090.0, 2440.0), (300.0, 2200.0, 2900.0)


def _ay(t):
    # first third at its onset, last third at its offset, a straight move between
    if t < 1 / 3:
        return AY_ON
    if t > 2 / 3:
        return AY_OFF
    u = (t - 1 / 3) * 3
    return tuple(a + u * (b - a) for a, b in zip(AY_ON, AY_OFF))


def _bank(folder: Path, plan):
    """plan: [(phone, track, seconds)] spoken in order with 0.1 s of silence between."""
    folder.mkdir(parents=True, exist_ok=True)
    gap = [0.0] * int(0.1 * SR)
    samples, units = list(gap), []
    for phone, track, seconds in plan:
        start = len(samples) // HOP
        samples += _voiced(track, seconds)
        units.append({"rec": 0, "phone": phone, "start": start, "end": len(samples) // HOP})
        samples += gap
    _write(folder / "rec.wav", samples)
    (folder / "bank.json").write_text(json.dumps({
        "sr": SR, "hop": HOP, "recordings": [{"path": str(folder / "rec.wav"), "n": len(samples) // HOP}],
        "units": units}), encoding="utf-8")


def _close(measured, true, f1=False):
    return abs(measured - true) <= (max(0.05 * true, F0 / 2) if f1 else 0.05 * true)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    folder = tmp_path_factory.mktemp("bank")
    plan = [("AA", lambda t: AA, 0.2)] * 6 + [("IY", lambda t: IY, 0.2)] * 6 + [("AY", _ay, 0.3)] * 6
    plan += [("UW", lambda t: (300.0, 870.0, 2240.0), 0.2)] * (BANK_VOWEL_MIN_UNITS - 2)  # too few units
    plan += [("EH", lambda t: (530.0, 1840.0, 2480.0), 0.03)] * 6                        # every unit too short
    _bank(folder, plan)
    return folder, measure_vowels_from_frame_bank(folder)


def test_steady_vowels_are_found_where_they_were_put(bank):
    _, m = bank
    for ph, (t1, t2, t3) in (("AA", AA), ("IY", IY)):
        v = m["vowels"][ph]
        assert v["n_units"] == 6, v
        assert _close(v["f1_hz"], t1, f1=True), (ph, v)
        assert _close(v["f2_hz"], t2) and _close(v["f3_hz"], t3), (ph, v)


def test_a_gliding_vowel_carries_its_own_onset_and_offset(bank):
    _, m = bank
    g = m["glides"]["AY"]
    assert _close(g["onset_hz"][1], AY_ON[1]), g      # starts near "ah", not where it ends
    assert _close(g["offset_hz"][1], AY_OFF[1]), g
    assert _close(g["onset_hz"][0], AY_ON[0], f1=True) and _close(g["offset_hz"][0], AY_OFF[0], f1=True), g


def test_too_few_units_and_too_short_units_give_no_value(bank):
    _, m = bank
    assert "UW" not in m["vowels"] and m["too_few_units"]["UW"] == BANK_VOWEL_MIN_UNITS - 2
    assert "EH" not in m["vowels"] and "EH" not in m["too_few_units"]   # 30 ms units are never measured


def test_no_bank_is_refused(tmp_path):
    with pytest.raises(FrameVoiceUnavailable):
        measure_vowels_from_frame_bank(tmp_path)


def test_the_script_rewrites_only_vowels_and_glides(bank, tmp_path):
    folder, m = bank
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from voice_fixture import write_test_voice
    src = write_test_voice(tmp_path / "in")
    dst = tmp_path / "out" / "voice.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/measure_vowels_from_bank.py"), str(folder), str(src), str(dst)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    before, after = json.loads(src.read_text()), json.loads(dst.read_text())
    assert after["vowels"].keys() == m["vowels"].keys()
    assert after["glides"].keys() == m["glides"].keys()
    assert "UW" in after["vowels_not_measured_from_bank"]
    for k in ("voice", "f0", "pitch_contour", "rate"):
        assert after[k] == before[k], k
    v = load_voice_profile(dst)
    assert v.vowel_formants["AA"][1] == m["vowels"]["AA"]["f2_hz"]
    assert v.glides["AY"][0][1] == m["glides"]["AY"]["onset_hz"][1]
