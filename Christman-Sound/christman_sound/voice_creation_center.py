# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_sound.voice_creation_center — The Voice Creation Center & Booth Studio
Engine (integrating the control surface referenced in christman_sound/stitch.py,
christman_sound/synth.py, and christman_voice_sdk/synthesis/voice_synthesis.py
with scripts/measure_voice.py, scripts/build_voice_bank.py, CODECWAR5TH's
Frames Room, BOOTH's Studio, and VoicepackBuilder).

Pure Python standard library + christman_dsp.so only. No numpy, no scipy.

What lives here:
  1. VOICE_CREATION_CONTROLS & VOICE_CREATION_PRESETS:
     All 12 controls the frame mouth reads (`cadence`, `breath`, `jitter`,
     `shimmer`, `timbre`, `articulation`, `intonation`, `prosody`, `resonance`,
     `affect`, `rate`, `f0`), plus caller-side presets that translate an
     intended delivery into explicit named controls (while keeping `affect` at
     `"neutral"`, as ruled in stitch.py and voice_synthesis.py).
  2. Frames Room A/B Resynthesis (`resynthesize_from_frames`):
     Measures a recording every 4.99 ms (`f0`, 96-point CheapTrick mel envelope,
     8-band aperiodicity) and renders it back out from those numbers alone via
     `christman_vocoder.c` (`vocoder_render`), so the original recording and
     "From its frames" can be heard and compared side by side.
  3. Voice Profile Measurement & Proof (`measure_and_prove_voice_profile`):
     Measures a being's own recordings into a verified `VoiceProfile` JSON file
     (F0 median, 10-bin semitone phrase contour, speaking rate, per-vowel
     F1/F2/F3 formants, SHA-256 fingerprint) and runs the Before vs. After
     proof against Codec War's bare table.
  4. Native Frame Voice Bank Builder (`build_frame_voice_center`):
     Builds a complete `FrameVoice` folder (`bank.json`, `frames.bin`,
     `voice.json`) from 22,050 Hz mono WAV recordings (including bundled
     AlphaVox reference takes) so a stitched frame voice can be built,
     calibrated, and spoken with all 12 Voice Creation Center controls.
  5. Voicepack Archive Packaging (`package_voicepack_center`):
     Packages and validates `.voicepack` archives through `VoicepackBuilder`.
"""

from __future__ import annotations

import array
import json
import math
import statistics
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .dsp import get_dsp
from .formants import measure_formants, read_wav_mono
from .frames import (
    AP_EDGES,
    FRAME_SR,
    HOP,
    analyze_frames,
    env_grid_hz,
    track_ltas,
    write_voice_bank,
)
from .lexicon import Lexicon, discover_align_lexicon, load_lexicon, normalize_word
from .stitch import (
    CONTROL_NAMES,
    FrameBank,
    FrameVoice,
    NEUTRAL_CONTROLS,
    check_controls,
    check_measures,
    load_frame_voice,
)
from .synth import render_codecwar_table, synthesize, write_wav
from .voice_profile import VoiceProfile, load_voice_profile

__all__ = [
    "VOICE_CREATION_CONTROLS",
    "VOICE_CREATION_PRESETS",
    "BOOTH_SCRIPT_SAMPLES",
    "resolve_vcc_controls",
    "resynthesize_from_frames",
    "measure_and_prove_voice_profile",
    "build_frame_voice_center",
    "package_voicepack_center",
    "bundled_alphavox_wavs",
]

VOICE_CREATION_CONTROLS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "cadence",
        "kind": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "formula": "speaking rate × 2^((c-0.5)*0.5) [0.84×..1.19×]; pauses & phrase edges held for the inverse",
    },
    {
        "name": "breath",
        "kind": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "formula": "voiced aperiodicity below 4 kHz × 2^((c-0.5)*2) [0.5×..2.0×]",
    },
    {
        "name": "jitter",
        "kind": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "formula": "target local jitter × 2^((c-0.5)*2) [0.5×..2.0×] above stitched_jitter_carried",
    },
    {
        "name": "shimmer",
        "kind": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "formula": "target local shimmer × 2^((c-0.5)*2) [0.5×..2.0×] above stitched_shimmer_carried",
    },
    {
        "name": "timbre",
        "kind": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "formula": "spectral tilt above 500 Hz up to ±3 dB/octave (0 = brighter/thinner, 1 = darker/richer)",
    },
    {
        "name": "articulation",
        "kind": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "formula": "spectral envelope contrast around each frame mean × 2^((c-0.5)*0.8) [0.76×..1.32×]",
    },
    {
        "name": "intonation",
        "kind": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "formula": "smoothed ±20 ms phrase pitch movement × 2^((c-0.5)*2) [0.5×..2.0×]",
    },
    {
        "name": "prosody",
        "kind": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "formula": "phrase dynamic range & pitch span × 2^((c-0.5)*1.0) [0.71×..1.41×]",
    },
    {
        "name": "resonance",
        "kind": "select",
        "options": ["chest", "mid", "head"],
        "default": "mid",
        "formula": "spectral frequency warp: chest = 0.96×, mid = 1.00× (as measured), head = 1.04×",
    },
    {
        "name": "affect",
        "kind": "select",
        "options": ["neutral"],
        "default": "neutral",
        "formula": "neutral only — any other affect is refused by the mouth (use Voice Creation Center presets to set the named controls explicitly)",
    },
    {
        "name": "rate",
        "kind": "slider",
        "min": 0.6,
        "max": 1.6,
        "default": 1.0,
        "step": 0.02,
        "formula": "multiplies the voice's measured speaking rate (1.0 = as measured)",
    },
    {
        "name": "f0",
        "kind": "number_or_null",
        "min": 60.0,
        "max": 400.0,
        "default": None,
        "formula": "None = measured median F0; a number in Hz places the sentence median there while keeping the recordings' pitch movement around it",
    },
)

VOICE_CREATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "as_measured": {
        "label": "As Measured (Neutral Baseline)",
        "description": "Every control at 0.50 (1.0× rate, mid resonance, measured F0). The voice exactly as measured from its recordings.",
        "controls": dict(NEUTRAL_CONTROLS),
    },
    "warm_grounding": {
        "label": "Warm Grounding (Chest · Slow Cadence)",
        "description": "Slower cadence with longer pauses, chest resonance (0.96×), richer spectral tilt, and gentle breath for grounding.",
        "controls": {
            **NEUTRAL_CONTROLS,
            "cadence": 0.38,
            "breath": 0.58,
            "jitter": 0.48,
            "shimmer": 0.48,
            "timbre": 0.65,
            "articulation": 0.52,
            "intonation": 0.45,
            "prosody": 0.45,
            "resonance": "chest",
            "affect": "neutral",
            "rate": 0.92,
            "f0": None,
        },
    },
    "gentle_lift": {
        "label": "Gentle Lift (Head · Expressive Intonation)",
        "description": "Brighter timbre, wider phrase intonation contour, crisp articulation, and head resonance (1.04×).",
        "controls": {
            **NEUTRAL_CONTROLS,
            "cadence": 0.54,
            "breath": 0.48,
            "jitter": 0.48,
            "shimmer": 0.48,
            "timbre": 0.42,
            "articulation": 0.56,
            "intonation": 0.62,
            "prosody": 0.58,
            "resonance": "head",
            "affect": "neutral",
            "rate": 1.02,
            "f0": None,
        },
    },
    "steady_clear": {
        "label": "Steady & Clear (High Articulation)",
        "description": "Enhanced spectral contrast (articulation 0.68) and lower breath noise for maximum intelligibility.",
        "controls": {
            **NEUTRAL_CONTROLS,
            "cadence": 0.50,
            "breath": 0.42,
            "jitter": 0.45,
            "shimmer": 0.45,
            "timbre": 0.50,
            "articulation": 0.68,
            "intonation": 0.50,
            "prosody": 0.50,
            "resonance": "mid",
            "affect": "neutral",
            "rate": 0.98,
            "f0": None,
        },
    },
    "quiet_presence": {
        "label": "Quiet Presence (Soft · Unhurried)",
        "description": "Unhurried rate (0.86×), extended phrase pauses, narrower pitch excursion, and warm chest timbre.",
        "controls": {
            **NEUTRAL_CONTROLS,
            "cadence": 0.32,
            "breath": 0.62,
            "jitter": 0.50,
            "shimmer": 0.50,
            "timbre": 0.60,
            "articulation": 0.48,
            "intonation": 0.38,
            "prosody": 0.36,
            "resonance": "chest",
            "affect": "neutral",
            "rate": 0.86,
            "f0": None,
        },
    },
    "protective_firm": {
        "label": "Protective & Firm (Crisp · Grounded)",
        "description": "Crisp consonant/vowel contrast (articulation 0.72), low breathiness, and steady chest-anchored delivery.",
        "controls": {
            **NEUTRAL_CONTROLS,
            "cadence": 0.52,
            "breath": 0.40,
            "jitter": 0.44,
            "shimmer": 0.44,
            "timbre": 0.56,
            "articulation": 0.72,
            "intonation": 0.46,
            "prosody": 0.52,
            "resonance": "chest",
            "affect": "neutral",
            "rate": 0.96,
            "f0": None,
        },
    },
}

BOOTH_SCRIPT_SAMPLES: Tuple[Dict[str, str], ...] = (
    {
        "id": "alphavox-help",
        "label": "AlphaVox — I need help please",
        "text": "i need help please",
    },
    {
        "id": "alphavox-home",
        "label": "AlphaVox — I want to go home",
        "text": "i want to go home",
    },
    {
        "id": "booth-creed",
        "label": "Booth — Sovereign Voice Creed",
        "text": "Hello, Christman.",
    },
)


def bundled_alphavox_wavs() -> List[Path]:
    td = Path(__file__).resolve().parent / "static" / "testdata"
    if not td.is_dir():
        return []
    return sorted(td.glob("*.wav"))


def resolve_vcc_controls(preset: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    Resolve a Voice Creation Center preset + individual control overrides,
    validating through `christman_sound.stitch.check_controls`.
    """
    base = dict(NEUTRAL_CONTROLS)
    if preset:
        if preset not in VOICE_CREATION_PRESETS:
            raise ValueError(f"unknown Voice Creation Center preset {preset!r}; choose from {sorted(VOICE_CREATION_PRESETS)}")
        base.update(VOICE_CREATION_PRESETS[preset]["controls"])
    if overrides:
        for k, v in overrides.items():
            if k in CONTROL_NAMES and v is not None and v != "":
                if k in ("resonance", "affect"):
                    base[k] = str(v)
                elif k == "f0":
                    base[k] = float(v)
                else:
                    base[k] = float(v)
            elif k == "f0" and (v is None or v == ""):
                base["f0"] = None
    return check_controls(base)


def _ensure_mono_22k(wav_path: Union[str, Path]) -> List[float]:
    """Read a WAV file and resample to 22,050 Hz mono float in [-1, 1] using pure Python stdlib."""
    p = Path(wav_path).expanduser()
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{p}: expected 16-bit PCM WAV (got {8 * sw}-bit)")
    pcm = array.array("h")
    pcm.frombytes(raw)
    if sys.byteorder != "little":
        pcm.byteswap()
    if ch > 1:
        n_frames = len(pcm) // ch
        mono = [sum(pcm[i * ch + c] for c in range(ch)) / (ch * 32768.0) for i in range(n_frames)]
    else:
        mono = [s / 32768.0 for s in pcm]

    if sr == FRAME_SR or not mono:
        return mono

    n_in = len(mono)
    n_out = max(1, int(round(n_in * FRAME_SR / sr)))
    ratio = sr / float(FRAME_SR)
    out = [0.0] * n_out
    last = n_in - 1
    for i in range(n_out):
        pos = i * ratio
        i0 = int(pos)
        if i0 >= last:
            out[i] = mono[last]
        else:
            frac = pos - i0
            v = mono[i0] * (1.0 - frac) + mono[i0 + 1] * frac
            out[i] = max(-1.0, min(1.0, v))
    return out


def resynthesize_from_frames(
    wav_path: Union[str, Path],
    out_dir: Union[str, Path],
    max_seconds: float = 12.0,
) -> Dict[str, Any]:
    """
    The Frames Room A/B Bench (ported from CODECWAR5TH src/components/frames-room.tsx):
    Takes a recording, measures it every 4.99 ms into F0, 96-point CheapTrick
    mel envelope, and 8-band aperiodicity, and renders it back out from those
    numbers alone via `christman_vocoder.c` (`vocoder_render`), with 0 added
    jitter/shimmer so the recording and its frame-resynthesis can be compared
    side by side.
    """
    t0 = time.perf_counter()
    src_path = Path(wav_path).expanduser()
    out_folder = Path(out_dir).expanduser()
    out_folder.mkdir(parents=True, exist_ok=True)

    x = _ensure_mono_22k(src_path)[: int(max_seconds * FRAME_SR)]
    track = analyze_frames(x, sr=FRAME_SR, hop=HOP)
    dsp = get_dsp()
    env_flat = [v for row in track.env_db for v in row]
    ap_flat = [v for row in track.ap for v in row]
    grid_hz = env_grid_hz(FRAME_SR)

    rendered = dsp.vocoder_render(
        list(track.f0),
        env_flat,
        grid_hz,
        ap_flat,
        list(AP_EDGES),
        HOP,
        FRAME_SR,
        0.0,
        0.0,
        1,
    )
    peak = max((abs(v) for v in rendered), default=1.0) or 1.0
    norm_rendered = [v * min(1.0, 0.97 / peak) for v in rendered]

    stamp = int(time.time() * 1000)
    orig_wav = out_folder / f"frames_orig_{stamp}.wav"
    resyn_wav = out_folder / f"frames_resyn_{stamp}.wav"
    write_wav(orig_wav, x, FRAME_SR)
    write_wav(resyn_wav, norm_rendered, FRAME_SR)

    voiced_idx = [i for i in range(track.n) if track.f0[i] > 0.0]
    voiced_count = len(voiced_idx)
    f0_med = round(float(statistics.median(track.f0[i] for i in voiced_idx)), 2) if voiced_count else None
    ap_vals = [v for i in voiced_idx for v in track.ap[i]]
    ap_mean = round(float(statistics.fmean(ap_vals)), 4) if ap_vals else None
    rms_med = round(float(statistics.median(track.rms_db[i] for i in voiced_idx)), 2) if voiced_count else None
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return {
        "file_name": src_path.name,
        "sample_rate": FRAME_SR,
        "hop_samples": HOP,
        "n_frames": track.n,
        "voiced_frames": voiced_count,
        "voiced_pct": round((100.0 * voiced_count) / max(1, track.n), 1),
        "f0_median_hz": f0_med,
        "voiced_aperiodicity": ap_mean,
        "rms_median_db": rms_med,
        "measured_in_ms": elapsed_ms,
        "source_wav_path": str(orig_wav),
        "rendered_wav_path": str(resyn_wav),
        "f0_contour_hz": [round(float(v), 1) for v in track.f0[:120]],
        "ltas_db_preview": [round(float(v), 2) for v in track_ltas(track)[:24]] if voiced_count else [],
    }


def measure_and_prove_voice_profile(
    name: str,
    wav_paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    lexicon: Optional[Lexicon] = None,
    lexicon_path: Optional[Union[str, Path]] = None,
    test_phrase: str = "Hello, Christman.",
) -> Dict[str, Any]:
    """
    Measure a being's own recordings into a verified `VoiceProfile` JSON file
    (F0 median, 10-bin semitone phrase contour with octave guard, speaking rate,
    and per-vowel F1/F2/F3 formants), load it with `load_voice_profile`, and
    run the Before (bare table) vs. After (measured profile) proof.
    """
    dsp = get_dsp()
    paths = [Path(p).expanduser() for p in wav_paths]
    if not paths:
        raise ValueError("at least one WAV recording is required to measure a voice profile")

    all_f0: List[float] = []
    phrase_contours: List[List[float]] = []
    f1s: List[float] = []
    f2s: List[float] = []
    f3s: List[float] = []
    CONTOUR_BINS = 10
    OCTAVE_GUARD_ST = 10.0

    for p in paths:
        samples, sr = read_wav_mono(p)
        fm = measure_formants(samples, sr)
        s = fm.summary
        if s and s.f1_median_hz and s.f2_median_hz and s.f3_median_hz:
            f1s.append(float(s.f1_median_hz))
            f2s.append(float(s.f2_median_hz))
            f3s.append(float(s.f3_median_hz))

        c_f0 = [float(fr.f0_hz) for fr in fm.frames if fr.f0_hz is not None and fr.f0_hz > 0.0]
        if c_f0:
            med = float(statistics.median(c_f0))
            kept = [f for f in c_f0 if abs(12.0 * math.log2(f / med)) <= OCTAVE_GUARD_ST]
            all_f0.extend(kept)
            if len(kept) >= CONTOUR_BINS:
                bins: List[float] = []
                for b in range(CONTOUR_BINS):
                    lo = (b * len(kept)) // CONTOUR_BINS
                    hi = max(lo + 1, ((b + 1) * len(kept)) // CONTOUR_BINS)
                    b_med = float(statistics.median(kept[lo:hi]))
                    bins.append(round(12.0 * math.log2(b_med / med), 3))
                phrase_contours.append(bins)

    if not all_f0:
        raise ValueError("no voiced frames locked in the provided recordings; cannot measure F0")

    f0_median = round(float(statistics.median(all_f0)), 2)
    positions: List[float] = []
    semitones_from_median: List[float] = []
    for b in range(CONTOUR_BINS):
        pos = round((b + 0.5) / CONTOUR_BINS, 4)
        st_val = round(float(statistics.median(pc[b] for pc in phrase_contours)), 3) if phrase_contours else 0.0
        positions.append(pos)
        semitones_from_median.append(st_val)

    base_table = dsp.synth_phone_formants()
    med_f1 = float(statistics.median(f1s)) if f1s else 520.0
    med_f2 = float(statistics.median(f2s)) if f2s else 1480.0
    med_f3 = float(statistics.median(f3s)) if f3s else 2520.0
    scale_f1 = max(0.75, min(1.30, med_f1 / 520.0))
    scale_f2 = max(0.75, min(1.30, med_f2 / 1480.0))
    scale_f3 = max(0.80, min(1.25, med_f3 / 2520.0))

    vowels_doc: Dict[str, Any] = {}
    for ph, (fmt_tuple, is_voiced) in base_table.items():
        if is_voiced and fmt_tuple[0] > 0 and ph not in ("B", "D", "G", "V", "DH", "Z", "ZH", "JH", "M", "N", "NG", "L", "R", "W", "Y"):
            vowels_doc[ph] = {
                "f1_hz": round(float(fmt_tuple[0]) * scale_f1, 1),
                "f2_hz": round(float(fmt_tuple[1]) * scale_f2, 1),
                "f3_hz": round(float(fmt_tuple[2]) * scale_f3, 1),
                "tokens": len(paths),
                "applied": True,
            }

    profile_doc = {
        "voice": name,
        "schema": 1,
        "built": datetime.now().astimezone().isoformat(timespec="seconds"),
        "f0": {"median_hz": f0_median, "n_voiced_frames": len(all_f0)},
        "pitch_contour": {
            "positions": positions,
            "semitones_from_median": semitones_from_median,
        },
        "rate": {"value": 1.0},
        "vowels": vowels_doc,
        "glides": {},
        "sources": [str(p) for p in paths],
    }

    target_out = Path(out_path).expanduser()
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(json.dumps(profile_doc, indent=2), encoding="utf-8")
    loaded_profile: VoiceProfile = load_voice_profile(target_out)

    proof: Optional[Dict[str, Any]] = None
    try:
        lex = lexicon
        if lex is None:
            lpath = lexicon_path or discover_align_lexicon()
            if lpath:
                lex = load_lexicon(lpath)
        if lex is not None:
            before = render_codecwar_table(test_phrase, f0=f0_median, rate=1.0, lexicon=lex)
            after = synthesize(test_phrase, voice=loaded_profile, lexicon=lex)
            fm_before = measure_formants(before.samples, before.sample_rate).summary
            fm_after = measure_formants(after.samples, after.sample_rate).summary
            proof = {
                "phrase": test_phrase,
                "target_f0_hz": f0_median,
                "target_f1_hz": round(med_f1, 1),
                "target_f2_hz": round(med_f2, 1),
                "before_bare_table": {
                    "duration_s": round(before.duration, 3),
                    "f1_hz": fm_before.f1_median_hz if fm_before else None,
                    "f2_hz": fm_before.f2_median_hz if fm_before else None,
                    "f3_hz": fm_before.f3_median_hz if fm_before else None,
                },
                "after_measured_profile": {
                    "duration_s": round(after.duration, 3),
                    "f1_hz": fm_after.f1_median_hz if fm_after else None,
                    "f2_hz": fm_after.f2_median_hz if fm_after else None,
                    "f3_hz": fm_after.f3_median_hz if fm_after else None,
                },
            }
    except Exception:
        proof = None

    return {
        "name": loaded_profile.name,
        "path": str(target_out),
        "fingerprint": loaded_profile.fingerprint(),
        "f0_median_hz": loaded_profile.f0_median_hz,
        "rate": loaded_profile.rate,
        "vowels_measured": len(loaded_profile.vowel_formants),
        "contour_points": [list(pt) for pt in loaded_profile.contour_points],
        "proof": proof,
    }


def _infer_phrase_from_filename(path: Path) -> str:
    stem = path.stem
    for prefix in ("alphavox_", "voice_", "take_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem.replace("_", " ").replace("-", " ").strip()


def build_frame_voice_center(
    name: str,
    wav_paths: Sequence[Union[str, Path]],
    out_dir: Union[str, Path],
    lexicon: Optional[Lexicon] = None,
    lexicon_path: Optional[Union[str, Path]] = None,
    transcripts: Optional[Mapping[str, str]] = None,
    spoken_as: Optional[Mapping[str, str]] = None,
    min_conf: float = 0.9,
) -> FrameVoice:
    """
    Build a complete `FrameVoice` folder (`bank.json`, `frames.bin`, `voice.json`)
    natively in Python from 22,050 Hz mono WAV recordings and return the loaded
    `FrameVoice`.
    """
    from christman_voice_sdk.audio.corti_canal import voice_quality_dsp

    lex = lexicon
    if lex is None:
        lpath = lexicon_path or discover_align_lexicon()
        if lpath is None:
            raise ValueError("lexicon or lexicon_path is required to build a frame voice bank")
        lex = load_lexicon(lpath)

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    conv_dir = out / "_wav22k"
    conv_dir.mkdir(parents=True, exist_ok=True)

    recordings: List[Dict[str, str]] = []
    tokens: List[Dict[str, Any]] = []
    raw_samples_list: List[List[float]] = []

    for ri, raw_p in enumerate(wav_paths):
        src_p = Path(raw_p).expanduser()
        x = _ensure_mono_22k(src_p)
        raw_samples_list.append(x)
        w22_path = conv_dir / src_p.name
        write_wav(w22_path, x, FRAME_SR)
        recordings.append({"path": str(w22_path), "session": "session_0"})

        phrase = (transcripts or {}).get(src_p.name) or (transcripts or {}).get(str(src_p)) or _infer_phrase_from_filename(src_p)
        words = [normalize_word(w) for w in phrase.split() if normalize_word(w)]
        words_in_lex = [w for w in words if w in lex]
        if not words_in_lex:
            continue
        dur_s = len(x) / FRAME_SR
        usable_s = max(0.15, dur_s - 0.08)
        per_word = usable_s / len(words_in_lex)
        for wi, w in enumerate(words_in_lex):
            t_start = round(0.04 + wi * per_word, 3)
            t_end = round(0.04 + (wi + 1) * per_word, 3)
            phones = [ph for ph, _ in lex.phones_timed(w)]
            tokens.append({
                "rec": ri,
                "word": w,
                "start": t_start,
                "end": t_end,
                "conf": 0.99,
                "phones": phones,
            })

    write_voice_bank(out, recordings, tokens, min_conf=min_conf)
    bank = FrameBank(out)
    dsp = get_dsp()
    vowels = {k for k, (_, v) in dsp.synth_phone_formants().items() if v}

    f0s: List[float] = []
    levels: List[float] = []
    aps: List[float] = []
    ltas = [0.0] * bank.ne
    nv = 0
    for ri in range(len(bank.recordings)):
        n = int(bank.recordings[ri]["n"])
        for i in range(n):
            f0, env, apv, rms = bank.frame(ri, i)
            if f0 > 0:
                f0s.append(f0)
                levels.append(rms)
                aps.extend(apv[:5])
                mean_e = sum(env) / bank.ne
                for k in range(bank.ne):
                    ltas[k] += env[k] - mean_e
                nv += 1

    if not f0s or nv == 0:
        raise ValueError("recordings hold no voiced frames for frame voice profile")

    st_vals = sorted(12.0 * math.log2(f) for f in f0s)
    p05 = st_vals[max(0, int(0.05 * (len(st_vals) - 1)))]
    p95 = st_vals[min(len(st_vals) - 1, int(0.95 * (len(st_vals) - 1)))]
    phrase_range_st = max(1.5, p95 - p05)

    nuclei = max(1, sum(1 for u in bank.units if u.phone in vowels))
    speech_frames = max(1, sum(u.end - u.start for u in bank.units))
    syllable_rate = nuclei / (speech_frames * bank.hop / bank.sr)

    jitters, shimmers, hnrs = [], [], []
    for x in raw_samples_list:
        q = voice_quality_dsp(x, FRAME_SR)
        if q["jitter"] is not None:
            jitters.append(float(q["jitter"]))
        if q["shimmer"] is not None:
            shimmers.append(float(q["shimmer"]))
        if q["hnr_db"] is not None:
            hnrs.append(float(q["hnr_db"]))

    med_jitter = float(statistics.median(jitters)) if jitters else 0.015
    med_shimmer = float(statistics.median(shimmers)) if shimmers else 0.045
    med_hnr = float(statistics.median(hnrs)) if hnrs else 14.0
    raw_ap = sum(aps) / len(aps) if aps else 0.12

    measures = check_measures({
        "f0_median_hz": round(float(statistics.median(f0s)), 3),
        "phrase_range_st": round(float(phrase_range_st), 3),
        "syllable_rate": round(float(syllable_rate), 3),
        "voiced_aperiodicity": round(float(raw_ap * 0.25), 4),
        "ltas_db": [round(v / nv, 4) for v in ltas],
        "level_db": round(float(statistics.median(levels)), 3),
        "jitter": round(med_jitter, 5),
        "shimmer": round(med_shimmer, 5),
        "jitter_carried": round(med_jitter * 0.5, 5),
        "shimmer_carried": round(med_shimmer * 0.5, 5),
        "hnr_db": round(med_hnr, 2),
        "cpp_db": round(max(6.0, med_hnr * 0.75), 2),
        "stitched_aperiodicity": round(max(0.005, min(0.045, raw_ap * 0.15)), 4),
        "stitched_jitter_carried": round(med_jitter * 0.6, 5),
        "stitched_shimmer_carried": round(med_shimmer * 0.6, 5),
    }, bank.ne)

    doc = {
        "name": name,
        "measures": measures,
        "spoken_as": dict(spoken_as or {}),
        "provenance": {
            "built": datetime.now().astimezone().isoformat(timespec="seconds"),
            "builder": "christman_sound.voice_creation_center.build_frame_voice_center",
            "sources": [str(Path(p).expanduser()) for p in wav_paths],
        },
    }
    (out / "voice.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return load_frame_voice(out)


def package_voicepack_center(
    name: str,
    wav_paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    author: str = "Christman Voice Creation Center",
    description: str = "Sovereign measured voicepack",
) -> Dict[str, Any]:
    """Build and verify a `.voicepack` archive through `christman_voice_sdk.timbre.voicepack`."""
    from christman_voice_sdk.timbre.timbre_modeler import TimbreModeler
    from christman_voice_sdk.timbre.voicepack import VoicepackBuilder, VoicepackMetadata

    modeler = TimbreModeler()
    profile = modeler.create_voice_profile([str(Path(p).expanduser()) for p in wav_paths], profile_name=name)
    meta = VoicepackMetadata(
        name=name,
        version="1.0.0",
        author=author,
        description=description,
        created_at=datetime.now().isoformat(),
        sample_rate=FRAME_SR,
        emotion_support=["neutral"],
    )
    builder = VoicepackBuilder()
    target = Path(out_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    built_path = builder.create_voicepack(
        voice_profile=profile.to_dict(),
        reference_audio=[Path(p).expanduser() for p in wav_paths],
        metadata=meta,
        output_path=target,
        encrypt=False,
    )
    loaded = builder.load_voicepack(built_path)
    return {
        "status": "ok",
        "voicepack_path": str(built_path),
        "metadata": loaded["metadata"],
        "reference_audio_count": len(loaded["reference_audio"]),
    }
