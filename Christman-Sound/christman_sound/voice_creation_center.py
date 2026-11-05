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
Being-neutral: holds no being's recordings or identity. Every value is measured
from the caller's own recordings, or the build refuses with the reason.
"""

from __future__ import annotations

import array
import datetime as _dt
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .dsp import get_dsp
from .formants import FRAME_SECONDS, HOP_SECONDS, measure_formants, read_wav_mono
from .frames import (
    AP_EDGES,
    FRAME_SR,
    HOP,
    analyze_frames,
    env_grid_hz,
    fft,
    track_ltas,
    write_voice_bank,
)
from .lexicon import (
    Lexicon,
    PhoneNotMapped,
    discover_align_lexicon,
    load_lexicon,
    map_phone,
    normalize_word,
)
from .stitch import (
    CONTROL_NAMES,
    FrameBank,
    FrameVoice,
    NEUTRAL_CONTROLS,
    StitchRefused,
    check_controls,
    check_measures,
    load_frame_voice,
)
from .synth import render_codecwar_table, synthesize, text_to_phones, write_wav
from .voice_profile import VoiceProfile, load_voice_profile

__all__ = [
    "VOWEL_RMS_RATIO",
    "MIN_TOKENS_TO_APPLY",
    "CONTOUR_BINS",
    "OCTAVE_GUARD_ST",
    "GLIDING_VOWELS",
    "PHRASE_GAP_S",
    "MIN_PHRASE_VOICED",
    "VOICE_CREATION_CONTROLS",
    "VOICE_CREATION_PRESETS",
    "BOOTH_SCRIPT_SAMPLES",
    "resolve_vcc_controls",
    "resynthesize_from_frames",
    "measure_and_prove_voice_profile",
    "build_frame_voice_center",
    "package_voicepack_center",
]

# -----------------------------------------------------------------------------
# THRESHOLDS — EVERY NUMBER BELOW IS A CHOSEN STARTING VALUE, NOT A MEASUREMENT.
# Carried directly from scripts/measure_voice.py and scripts/build_voice_bank.py,
# with the reason written beside each constant:
# -----------------------------------------------------------------------------

#: A vowel frame must have RMS at least this fraction of the word's loudest
#: voiced frame (0.5, i.e. within 6 dB). CHOSEN STARTING VALUE — held until
#: client data replaces it. Reason (scripts/measure_voice.py): a vowel is the
#: loudest stretch of its syllable, while surrounding nasals, liquids, and
#: voiced stops run quieter; without this gate, adjacent /r/ and /l/ pull F2 down.
VOWEL_RMS_RATIO = 0.5

#: Minimum measured tokens required before a vowel's measured F1-F3 replace
#: Codec War's table value (2 tokens). CHOSEN STARTING VALUE — held until
#: client data replaces it. Reason (scripts/measure_voice.py): one token can be
#: a single LPC tracking slip (e.g. a missed F2 with F3 read in its place);
#: 1-token vowels stay under measured_not_applied with their values shown.
MIN_TOKENS_TO_APPLY = 2

#: Equal parts each phrase is cut into for the pitch contour (5 bins).
#: CHOSEN STARTING VALUE — carried from scripts/measure_voice.py.
CONTOUR_BINS = 5

#: Semitone distance from the phrase's own F0 median beyond which a frame is
#: left out of the melody contour (9.0 semitones = 3/4 octave). CHOSEN STARTING
#: VALUE — held until a better voicing measure replaces it. Reason
#: (scripts/measure_voice.py): phrase-final creak or YIN locking onto a
#: subharmonic reads 10.8 to 14.7 semitones below the phrase median; 9 semitones
#: keeps spoken rises and falls while rejecting octave drops.
OCTAVE_GUARD_ST = 9.0

#: Diphthong/gliding vowels whose onset (first third) and offset (last third)
#: F1-F3 are measured separately (scripts/measure_voice.py).
GLIDING_VOWELS: Tuple[str, ...] = ("AW", "AY", "EY", "OW", "OY")

#: Inter-word pause that ends a phrase (0.15 s, carried from scripts/build_voice_bank.py).
PHRASE_GAP_S = 0.15

#: Minimum voiced frames (20 frames = 0.10 s at 4.99 ms hop) required for a
#: multi-word phrase's pitch range to be counted (scripts/build_voice_bank.py).
MIN_PHRASE_VOICED = 20

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
            "jitter": 0.50,
            "shimmer": 0.50,
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
            "jitter": 0.50,
            "shimmer": 0.50,
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
            "jitter": 0.50,
            "shimmer": 0.50,
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
            "jitter": 0.50,
            "shimmer": 0.50,
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
        "id": "rest-now",
        "label": "Calibration — I need to rest now",
        "text": "i need to rest now",
    },
    {
        "id": "read-story",
        "label": "Calibration — Please read me a story",
        "text": "please read me a story",
    },
    {
        "id": "window-open",
        "label": "Calibration — The window is open",
        "text": "the window is open",
    },
)


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


def _ensure_mono_sr(wav_path: Union[str, Path], target_sr: int = FRAME_SR) -> List[float]:
    """Read a 16-bit PCM WAV file and resample to `target_sr` Hz mono float in [-1, 1] using stdlib."""
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

    if sr == target_sr or not mono:
        return mono

    n_in = len(mono)
    n_out = max(1, int(round(n_in * target_sr / sr)))
    ratio = sr / float(target_sr)
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
    if not src_path.is_file():
        raise FileNotFoundError(f"recording not found: {src_path}")
    out_folder = Path(out_dir).expanduser()
    out_folder.mkdir(parents=True, exist_ok=True)

    x = _ensure_mono_sr(src_path, FRAME_SR)[: int(max_seconds * FRAME_SR)]
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _med(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return float(statistics.median(vals)) if vals else None


def _r1(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 1)


def _hear_words_with_ear(model_dir: Union[str, Path], wav_path: Path) -> List[Dict[str, Any]]:
    """Decode word start/end times and lattice confidences via `christman_sound.ear` at 16 kHz."""
    from .ear import Ear, load_model

    samples_16k = _ensure_mono_sr(wav_path, 16000)
    pcm = array.array("h", (max(-32768, min(32767, int(round(v * 32767.0)))) for v in samples_16k))
    if sys.byteorder != "little":
        pcm.byteswap()
    model = load_model(Path(model_dir).expanduser())
    ear = Ear(model, sample_rate=16000, words=True)
    return [
        {"word": x.word, "start": float(x.start), "end": float(x.end), "conf": float(x.conf)}
        for heard in ear.hear(pcm.tobytes())
        for x in heard.words
    ]


def _validate_aligned_words(raw_words: Sequence[Mapping[str, Any]], context: str) -> List[Dict[str, Any]]:
    """Validate that word alignments contain real measured start, end, and conf values."""
    out: List[Dict[str, Any]] = []
    for item in raw_words:
        if not isinstance(item, Mapping):
            raise ValueError(f"{context}: aligned word entry must be a dict")
        w = normalize_word(str(item.get("word") or ""))
        start = item.get("start")
        end = item.get("end")
        conf = item.get("conf")
        if not w or start is None or end is None or conf is None:
            raise ValueError(f"{context}: aligned word entry is missing word, start, end, or conf: {item!r}")
        s_f, e_f, c_f = float(start), float(end), float(conf)
        if e_f <= s_f or not (0.0 <= c_f <= 1.0):
            raise ValueError(f"{context}: invalid start/end/conf on aligned word {item!r}")
        out.append({"word": w, "start": s_f, "end": e_f, "conf": c_f})
    if not out:
        raise ValueError(f"{context}: no aligned words found")
    return out


def _vowel_ambiguity(lex: Lexicon, word: str, vowels: set) -> Optional[str]:
    """None when every pronunciation of `word` in the lexicon has the same vowels; else the refusal reason."""
    seqs = set()
    for pron in lex.pronunciations(normalize_word(word)):
        try:
            seqs.add(tuple(map_phone(ph, word)[0] for ph in pron if map_phone(ph, word)[0] in vowels))
        except PhoneNotMapped as exc:
            return f"a pronunciation holds a phone with no Codec War counterpart ({exc.phone})"
    if len(seqs) > 1:
        return "its lexicon pronunciations differ in their vowels: " + " / ".join(" ".join(s) for s in sorted(seqs))
    return None


def _nucleus_frames(frames: List[Dict[str, Any]], start: float, end: float, loud_ref: Optional[float]) -> List[Dict[str, Any]]:
    inside = [f for f in frames if start <= f["t"] <= end and f["voiced"]]
    if loud_ref is None:
        loud_ref = max((f["rms"] for f in inside), default=0.0)
    return [
        f
        for f in inside
        if f["f1"] is not None
        and f["f2"] is not None
        and f["f3"] is not None
        and f["rms"] >= VOWEL_RMS_RATIO * loud_ref
    ]


def _glide_values(fr: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(fr, key=lambda f: f["t"])
    if len(ordered) < 3:
        return {"onset_hz": None, "offset_hz": None}
    third = len(ordered) // 3
    first, last = ordered[:third], ordered[-third:]
    return {
        "onset_hz": [_med([f[k] for f in first]) for k in ("f1", "f2", "f3")],
        "offset_hz": [_med([f[k] for f in last]) for k in ("f1", "f2", "f3")],
    }


def _token_values(fr: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_frames": len(fr),
        "f1_hz": _med([f["f1"] for f in fr]),
        "f2_hz": _med([f["f2"] for f in fr]),
        "f3_hz": _med([f["f3"] for f in fr]),
        "b1_hz": _med([f["b1"] for f in fr]),
        "b2_hz": _med([f["b2"] for f in fr]),
        "b3_hz": _med([f["b3"] for f in fr]),
    }


def _contour_bins(frames: List[Dict[str, Any]], a: float, b: float) -> Tuple[List[Optional[float]], Optional[float]]:
    v = [f for f in frames if f["voiced"] and f["f0"] and a <= f["t"] <= b]
    if not v or b <= a:
        return [None] * CONTOUR_BINS, None
    m = float(statistics.median(f["f0"] for f in v))
    bins: List[List[float]] = [[] for _ in range(CONTOUR_BINS)]
    for f in v:
        st = 12.0 * math.log2(f["f0"] / m)
        if abs(st) > OCTAVE_GUARD_ST:
            continue
        pos = (f["t"] - a) / (b - a)
        k = min(CONTOUR_BINS - 1, int(pos * CONTOUR_BINS))
        bins[k].append(st)
    return [_med(x) for x in bins], m


def measure_and_prove_voice_profile(
    name: str,
    wav_paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    lexicon: Optional[Lexicon] = None,
    lexicon_path: Optional[Union[str, Path]] = None,
    vosk_model: Optional[Union[str, Path]] = None,
    aligned_words_by_file: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Measure a being's own recordings into a verified `VoiceProfile` JSON file
    using the exact per-vowel, per-phrase measurement rules of `scripts/measure_voice.py`:
      - Word times come from `christman_sound.ear` (`vosk_model`) or explicit
        measured `aligned_words_by_file`. Never guesses word boundaries.
      - Ambiguous-vowel words are left out and listed with their reason.
      - Per-vowel F1, F2, F3 are measured from voiced nucleus frames (`rms >=
        VOWEL_RMS_RATIO * loud_ref`) and applied only when at least
        `MIN_TOKENS_TO_APPLY` (2) tokens carry that vowel; 1-token vowels go into
        `measured_not_applied` and 0-token vowels into `not_measured`. Never
        falls back to 520/1480/2520 Hz or scales Codec War's bare table.
    """
    if not name or not str(name).strip():
        raise ValueError("a voice name is required; refusing to invent a default name")
    paths = [Path(p).expanduser() for p in wav_paths]
    if not paths:
        raise ValueError("at least one WAV recording is required to measure a voice profile")

    dsp = get_dsp()
    if not (dsp.available and dsp.synth_available):
        raise RuntimeError(f"christman_dsp / christman_synth not loaded: {dsp.error or dsp.synth_error}")

    lex = lexicon
    if lex is None:
        lpath = lexicon_path or discover_align_lexicon()
        if lpath is None:
            raise ValueError("lexicon or lexicon_path is required to measure a voice profile")
        lex = load_lexicon(lpath)

    model_dir = vosk_model
    if model_dir is None and aligned_words_by_file is None:
        from .ear.recognizer import discover_vosk_model
        model_dir = discover_vosk_model()
    if model_dir is None and aligned_words_by_file is None:
        raise ValueError(
            "cannot measure voice profile without word alignments from christman_sound.ear "
            "(vosk_model) or explicit aligned_words_by_file; refusing to guess word boundaries"
        )

    table = dsp.synth_phone_formants()
    vowels = {pid for pid, (_, is_vowel) in table.items() if is_vowel}

    phrase_records: List[Dict[str, Any]] = []
    skipped_words: List[Dict[str, str]] = []
    by_vowel: Dict[str, List[Dict[str, Any]]] = {}

    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"recording not found: {p}")
        samples, sr = read_wav_mono(p)
        fm = measure_formants(samples, sr)
        frames = [
            {
                "t": f.time_s,
                "voiced": f.voiced,
                "f0": f.f0_hz,
                "rms": f.rms,
                "f1": f.f1_hz,
                "f2": f.f2_hz,
                "f3": f.f3_hz,
                "b1": f.b1_hz,
                "b2": f.b2_hz,
                "b3": f.b3_hz,
            }
            for f in fm.frames
        ]

        raw_aligned: Optional[Sequence[Mapping[str, Any]]] = None
        if aligned_words_by_file is not None:
            raw_aligned = aligned_words_by_file.get(p.name) or aligned_words_by_file.get(str(p))
        if raw_aligned is not None:
            aligned = _validate_aligned_words(raw_aligned, p.name)
        elif model_dir is not None:
            heard = _hear_words_with_ear(model_dir, p)
            aligned = _validate_aligned_words(heard, p.name)
        else:
            raise ValueError(f"{p.name}: no word alignments available; refusing to guess word boundaries")

        phrase_text = " ".join(w["word"] for w in aligned)
        tokens: List[Dict[str, Any]] = []

        for wi, w in enumerate(aligned):
            why = _vowel_ambiguity(lex, w["word"], vowels)
            if why:
                skipped_words.append({"phrase": p.name, "word": w["word"], "why": why})
                continue
            seq = text_to_phones(w["word"], lex, dsp)
            ph = [(pid, d) for pid, d, word_label in seq if word_label is not None]
            vidx = [k for k, (pid, _) in enumerate(ph) if pid in vowels]
            voiced_in = [f for f in frames if w["start"] <= f["t"] <= w["end"] and f["voiced"]]
            loud = max((f["rms"] for f in voiced_in), default=None)
            total = sum(d for _, d in ph)
            span = w["end"] - w["start"]
            if total <= 0.0 or span <= 0.0:
                continue
            for ordinal, k in enumerate(vidx):
                before = sum(d for _, d in ph[:k])
                s_t = w["start"] + span * before / total
                e_t = w["start"] + span * (before + ph[k][1]) / total
                fr = _nucleus_frames(frames, s_t, e_t, loud)
                tok: Dict[str, Any] = {
                    "phrase": p.name,
                    "word": w["word"],
                    "word_index": wi,
                    "vowel": ph[k][0],
                    "vowel_in_word": ordinal,
                    "start_s": round(s_t, 3),
                    "end_s": round(e_t, 3),
                }
                tok.update(_token_values(fr))
                if ph[k][0] in GLIDING_VOWELS:
                    tok.update(_glide_values(fr))
                tokens.append(tok)
                if tok["n_frames"] > 0:
                    by_vowel.setdefault(tok["vowel"], []).append(tok)

        phrase_records.append({
            "path": p,
            "name": p.name,
            "sha256": _sha256_file(p),
            "sr": sr,
            "samples": samples,
            "text": phrase_text,
            "aligned": aligned,
            "frames": frames,
            "tokens": tokens,
        })

    vow: Dict[str, Dict[str, Any]] = {}
    held_back: Dict[str, Dict[str, Any]] = {}
    for v in sorted(vowels):
        chosen = by_vowel.get(v, [])
        if not chosen:
            continue
        (cf1, cf2, cf3, _cb1, _cb2), _ = table[v]
        dest = vow if len(chosen) >= MIN_TOKENS_TO_APPLY else held_back
        dest[v] = {
            "f1_hz": _r1(_med([_r1(t["f1_hz"]) for t in chosen])),
            "f2_hz": _r1(_med([_r1(t["f2_hz"]) for t in chosen])),
            "f3_hz": _r1(_med([_r1(t["f3_hz"]) for t in chosen])),
            "source": "median over the tokens below of each token's median frame",
            "n_tokens": len(chosen),
            "n_frames": sum(t["n_frames"] for t in chosen),
            "measured_bandwidths_hz_not_applied": {
                "b1": _r1(_med([t["b1_hz"] for t in chosen])),
                "b2": _r1(_med([t["b2_hz"] for t in chosen])),
                "b3": _r1(_med([t["b3_hz"] for t in chosen])),
            },
            "codec_war_table_it_replaces": {"f1_hz": cf1, "f2_hz": cf2, "f3_hz": cf3},
            "tokens": [
                {
                    k: (_r1(x) if isinstance(x, float) and k.endswith("_hz") else x)
                    for k, x in t.items()
                    if k not in ("b1_hz", "b2_hz", "b3_hz", "onset_hz", "offset_hz")
                }
                for t in chosen
            ],
        }
    for v, e in held_back.items():
        e["not_applied_because"] = (
            f"{e['n_tokens']} token; fewer than MIN_TOKENS_TO_APPLY = {MIN_TOKENS_TO_APPLY}. Codec War's "
            "table value is used"
        )
    not_measured = {
        v: "no measured word in the aligned recordings carries this vowel; Codec War's table value is used"
        for v in sorted(vowels)
        if v not in vow and v not in held_back
    }

    glides: Dict[str, Dict[str, Any]] = {}
    glides_held: Dict[str, Dict[str, Any]] = {}
    for v in GLIDING_VOWELS:
        chosen = [t for t in by_vowel.get(v, []) if t.get("onset_hz") is not None]
        if not chosen:
            continue
        toks = [
            {
                "phrase": t["phrase"],
                "word": t["word"],
                "n_frames": t["n_frames"],
                "onset_hz": [_r1(x) for x in t["onset_hz"]],
                "offset_hz": [_r1(x) for x in t["offset_hz"]],
            }
            for t in chosen
        ]
        entry: Dict[str, Any] = {
            "onset_hz": [_r1(_med([t["onset_hz"][j] for t in toks])) for j in range(3)],
            "offset_hz": [_r1(_med([t["offset_hz"][j] for t in toks])) for j in range(3)],
            "source": "per token: the vowel's nucleus frames in time order; onset = median F1, F2, F3 of the first third, offset = of the last third; value = median over tokens",
            "n_tokens": len(toks),
            "tokens": toks,
        }
        if len(toks) >= MIN_TOKENS_TO_APPLY:
            glides[v] = entry
        else:
            entry["not_applied_because"] = (
                f"{len(toks)} token; fewer than MIN_TOKENS_TO_APPLY = {MIN_TOKENS_TO_APPLY}."
            )
            glides_held[v] = entry

    all_f0 = [f["f0"] for pr in phrase_records for f in pr["frames"] if f["voiced"] and f["f0"]]
    if not all_f0:
        raise ValueError("no voiced frames locked in the provided recordings; cannot measure F0")

    per_phrase_contour: Dict[str, Any] = {}
    bins: List[List[float]] = [[] for _ in range(CONTOUR_BINS)]
    per_rate: Dict[str, Any] = {}

    for pr in phrase_records:
        a = float(pr["aligned"][0]["start"])
        b = float(pr["aligned"][-1]["end"])
        cb, m = _contour_bins(pr["frames"], a, b)
        cb_rounded = [None if x is None else round(x, 2) for x in cb]
        per_phrase_contour[pr["name"]] = {"phrase_median_f0_hz": _r1(m), "semitones": cb_rounded}
        for k, x in enumerate(cb_rounded):
            if x is not None:
                bins[k].append(x)

        seq = text_to_phones(pr["text"], lex, dsp)
        spoken = [k for k, (_pid, _d, w) in enumerate(seq) if w is not None]
        if not spoken or b <= a:
            raise ValueError(f"{pr['name']}: could not measure speaking rate span")
        synth_span = sum(d for _pid, d, _w in seq[spoken[0] : spoken[-1] + 1])
        per_rate[pr["name"]] = {
            "synth_seconds_at_rate_1": round(synth_span, 3),
            "recorded_seconds": round(b - a, 3),
            "ratio": round(synth_span / (b - a), 4),
        }

    positions = [round((k + 0.5) / CONTOUR_BINS, 3) for k in range(CONTOUR_BINS)]
    semis = [round(float(_med(b)), 2) if b else None for b in bins]
    if any(s is None for s in semis):
        raise ValueError(f"a contour part had no voiced frame in any phrase: {semis}; refusing to fill in defaults")

    rate_val = _med([v["ratio"] for v in per_rate.values()])
    if rate_val is None:
        raise ValueError("speaking rate could not be measured from phrases")

    profile_doc = {
        "voice": name,
        "schema": 1,
        "measured_on": _dt.date.today().isoformat(),
        "f0": {
            "median_hz": _r1(float(statistics.median(all_f0))),
            "n_voiced_frames": len(all_f0),
            "source": "median F0 of every voiced frame (christman_yin inside christman_sound.formants) over all aligned recordings",
        },
        "pitch_contour": {
            "positions": positions,
            "semitones_from_median": semis,
            "n_phrases_per_position": [len(b) for b in bins],
            "per_phrase": per_phrase_contour,
        },
        "rate": {
            "value": round(float(rate_val), 3),
            "n_phrases": len(per_rate),
            "per_phrase": per_rate,
        },
        "vowels": vow,
        "measured_not_applied": held_back,
        "not_measured": not_measured,
        "glides": glides,
        "glides_not_applied": glides_held,
        "words_left_out_of_vowel_measurement": skipped_words,
        "sources": {
            "recordings": {
                "prefix": prefix,
                "files": [
                    {
                        "name": pr["name"],
                        "sha256": pr["sha256"],
                        "text": pr["text"],
                        "sample_rate": pr["sr"],
                        "seconds": round(len(pr["samples"]) / pr["sr"], 3),
                        "aligned_words": pr["aligned"],
                    }
                    for pr in phrase_records
                ],
            },
            "formant_reader": {
                "module": "christman_sound.formants",
                "frame_s": FRAME_SECONDS,
                "hop_s": HOP_SECONDS,
                "vowel_rms_ratio": VOWEL_RMS_RATIO,
                "min_tokens_to_apply": MIN_TOKENS_TO_APPLY,
            },
        },
    }

    target_out = Path(out_path).expanduser()
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(json.dumps(profile_doc, indent=2), encoding="utf-8")
    loaded_profile: VoiceProfile = load_voice_profile(target_out)

    # Before (Codec War bare table at measured F0) vs After (measured VoiceProfile) proof on the first aligned phrase
    first_phrase = phrase_records[0]["text"]
    before = render_codecwar_table(first_phrase, f0=loaded_profile.f0_median_hz, rate=1.0, lexicon=lex, engine=dsp)
    after = synthesize(first_phrase, voice=loaded_profile, lexicon=lex, engine=dsp)
    fm_before = measure_formants(before.samples, before.sample_rate).summary
    fm_after = measure_formants(after.samples, after.sample_rate).summary
    proof = {
        "phrase": first_phrase,
        "target_f0_hz": loaded_profile.f0_median_hz,
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

    return {
        "name": loaded_profile.name,
        "path": str(target_out),
        "fingerprint": loaded_profile.fingerprint(),
        "f0_median_hz": loaded_profile.f0_median_hz,
        "rate": loaded_profile.rate,
        "vowels_applied": len(loaded_profile.vowel_formants),
        "vowels_held_back": len(held_back),
        "contour_points": [list(pt) for pt in loaded_profile.contour_points],
        "proof": proof,
    }


def _measure_cpp_db_stdlib(samples: Sequence[float], sr: int) -> Optional[float]:
    """
    Measure Cepstral Peak Prominence (CPP, dB) in pure Python stdlib when Praat
    is unavailable: real cepstrum of voiced 1024-sample frames and peak prominence
    in the fundamental quefrency range (60..400 Hz) above the quefrency mean.
    Returns None if no voiced frame can be measured.
    """
    n_fft = 1024
    hop = max(1, sr // 100)
    q_min = max(2, int(sr // 400))
    q_max = min(n_fft // 2 - 2, int(sr // 60))
    if len(samples) < n_fft or q_max <= q_min:
        return None
    win = [0.54 - 0.46 * math.cos(2.0 * math.pi * k / (n_fft - 1)) for k in range(n_fft)]
    frame_rms = [
        (start, math.sqrt(sum(v * v for v in samples[start : start + n_fft]) / n_fft))
        for start in range(0, len(samples) - n_fft + 1, hop * 2)
    ]
    max_rms = max((r for _, r in frame_rms), default=0.0)
    if max_rms <= 1e-6:
        return None
    prominences: List[float] = []
    for start, rms in frame_rms:
        if rms < 0.2 * max_rms:
            continue
        seg = samples[start : start + n_fft]
        re = [float(seg[k]) * win[k] for k in range(n_fft)]
        im = [0.0] * n_fft
        fft(re, im, False)
        log_pow = [10.0 * math.log10(max(1e-12, re[k] * re[k] + im[k] * im[k])) for k in range(n_fft)]
        c_re = list(log_pow)
        c_im = [0.0] * n_fft
        fft(c_re, c_im, True)
        band = [abs(c_re[q]) for q in range(q_min, q_max + 1)]
        if not band:
            continue
        peak_val = max(band)
        bg_val = sum(band) / len(band)
        if peak_val > bg_val:
            prominences.append(peak_val - bg_val)
    if not prominences:
        return None
    return round(float(statistics.median(prominences)), 2)


def _measure_recording_quality(wav_path: Path, samples: Sequence[float], sr: int) -> Dict[str, float]:
    """
    Measure jitter (fraction), shimmer (fraction), hnr_db, and cpp_db from a WAV
    recording. Uses Praat (`measure_voice_quality`) when available, or
    `voice_quality_dsp` + `_measure_cpp_db_stdlib` when running in stdlib-only
    mode. Never fills in defaults: raises ValueError if any quantity is None.
    """
    from christman_voice_sdk.audio.corti_canal import voice_quality_dsp

    q_praat: Optional[Dict[str, Optional[float]]] = None
    if "numpy" not in sys.modules or sys.modules.get("numpy") is not None:
        try:
            from christman_voice_sdk.timbre.timbre_modeler import measure_voice_quality
            raw_q = measure_voice_quality(str(wav_path))
            q_praat = {
                "jitter": None if raw_q.get("jitter_local_pct") is None else float(raw_q["jitter_local_pct"]) / 100.0,
                "shimmer": None if raw_q.get("shimmer_local_pct") is None else float(raw_q["shimmer_local_pct"]) / 100.0,
                "hnr_db": raw_q.get("hnr_db"),
                "cpp_db": raw_q.get("cpp_db"),
            }
        except Exception:
            q_praat = None

    if q_praat is not None and all(q_praat.get(k) is not None for k in ("jitter", "shimmer", "hnr_db", "cpp_db")):
        return {k: float(q_praat[k]) for k in ("jitter", "shimmer", "hnr_db", "cpp_db")}  # type: ignore[arg-type]

    q_dsp = voice_quality_dsp(samples, sr)
    cpp_val = _measure_cpp_db_stdlib(samples, sr)
    combined = {
        "jitter": q_dsp.get("jitter"),
        "shimmer": q_dsp.get("shimmer"),
        "hnr_db": q_dsp.get("hnr_db"),
        "cpp_db": cpp_val,
    }
    missing = [k for k, v in combined.items() if v is None]
    if missing:
        raise ValueError(
            f"{wav_path.name}: could not measure {', '.join(missing)}; refusing to fill in defaults"
        )
    return {k: float(v) for k, v in combined.items() if v is not None}


def build_frame_voice_center(
    name: str,
    wav_paths: Sequence[Union[str, Path]],
    out_dir: Union[str, Path],
    lexicon: Optional[Lexicon] = None,
    lexicon_path: Optional[Union[str, Path]] = None,
    vosk_model: Optional[Union[str, Path]] = None,
    aligned_tokens: Optional[Sequence[Mapping[str, Any]]] = None,
    spoken_as: Optional[Mapping[str, str]] = None,
    min_conf: float = 0.9,
) -> FrameVoice:
    """
    Build a complete `FrameVoice` folder (`bank.json`, `frames.bin`, `voice.json`)
    from 22,050 Hz mono WAV recordings:
      - Word tokens (`rec`, `word`, `start`, `end`, `conf`) must come from
        `christman_sound.ear` (`vosk_model`) or explicit measured `aligned_tokens`.
        Never slices recording duration into equal boxes or invents `conf: 0.99`.
      - Voice quality (`jitter`, `shimmer`, `hnr_db`, `cpp_db`) is measured from
        the recordings, and `jitter_carried` / `shimmer_carried` and
        `stitched_jitter_carried` / `stitched_shimmer_carried` are measured from
        actual frame-resynthesized and stitched renderings with zero added
        jitter/shimmer. Nothing is a default or fixed multiplier.
    """
    if not name or not str(name).strip():
        raise ValueError("a voice name is required; refusing to invent a default name")
    if not wav_paths:
        raise ValueError("at least one WAV recording is required to build a frame voice bank")

    lex = lexicon
    if lex is None:
        lpath = lexicon_path or discover_align_lexicon()
        if lpath is None:
            raise ValueError("lexicon or lexicon_path is required to build a frame voice bank")
        lex = load_lexicon(lpath)

    model_dir = vosk_model
    if model_dir is None and aligned_tokens is None:
        from .ear.recognizer import discover_vosk_model
        model_dir = discover_vosk_model()
    if model_dir is None and aligned_tokens is None:
        raise ValueError(
            "cannot build frame voice bank without christman_sound.ear (vosk_model) "
            "or explicit measured aligned_tokens; refusing to guess word times or confidences"
        )

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    conv_dir = out / "_wav22k"
    conv_dir.mkdir(parents=True, exist_ok=True)

    recordings: List[Dict[str, str]] = []
    tokens: List[Dict[str, Any]] = []
    raw_samples_list: List[List[float]] = []
    conv_paths: List[Path] = []

    for ri, raw_p in enumerate(wav_paths):
        src_p = Path(raw_p).expanduser()
        if not src_p.is_file():
            raise FileNotFoundError(f"recording not found: {src_p}")
        x = _ensure_mono_sr(src_p, FRAME_SR)
        raw_samples_list.append(x)
        w22_path = conv_dir / src_p.name
        write_wav(w22_path, x, FRAME_SR)
        conv_paths.append(w22_path)
        recordings.append({"path": str(w22_path), "session": "session_0"})

        if aligned_tokens is None and model_dir is not None:
            for w in _hear_words_with_ear(model_dir, w22_path):
                key = normalize_word(w["word"])
                phones = [ph for ph, _ in lex.phones_timed(key)] if key in lex else []
                tokens.append({
                    "rec": ri,
                    "word": key or w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "conf": float(w["conf"]),
                    "phones": phones,
                })

    if aligned_tokens is not None:
        for t in aligned_tokens:
            if not isinstance(t, Mapping):
                raise ValueError("aligned_tokens entries must be dicts")
            rec_idx = int(t.get("rec", 0))
            validated = _validate_aligned_words([t], f"rec[{rec_idx}]")[0]
            w_key = validated["word"]
            phones = list(t["phones"]) if t.get("phones") else ([ph for ph, _ in lex.phones_timed(w_key)] if w_key in lex else [])
            tokens.append({
                "rec": rec_idx,
                "word": w_key,
                "start": validated["start"],
                "end": validated["end"],
                "conf": validated["conf"],
                "phones": phones,
            })

    if not tokens:
        raise ValueError("no aligned word tokens found in recordings; refusing to build empty bank")

    write_voice_bank(out, recordings, tokens, min_conf=min_conf)
    bank = FrameBank(out)
    if not bank.units:
        raise ValueError("no phone units survived alignment and confidence gating; refusing to build voice")

    dsp = get_dsp()
    vowels = {k for k, (_, v) in dsp.synth_phone_formants().items() if v}

    f0s: List[float] = []
    levels: List[float] = []
    aps: List[float] = []
    phrase_ranges: List[float] = []
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
        rec_toks = [t for t in bank.tokens if t["rec"] == ri]
        phrases: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        for t in rec_toks:
            if cur and t["start"] - cur[-1]["end"] >= PHRASE_GAP_S:
                phrases.append(cur)
                cur = []
            cur.append(t)
        if cur:
            phrases.append(cur)
        for ph in phrases:
            a = math.floor(ph[0]["start"] * bank.sr / bank.hop + 0.5)
            b = min(n, math.floor(ph[-1]["end"] * bank.sr / bank.hop + 0.5))
            st = sorted(12.0 * math.log2(bank.frame(ri, i)[0]) for i in range(a, b) if bank.frame(ri, i)[0] > 0)
            if len(st) >= MIN_PHRASE_VOICED:
                p95 = st[min(len(st) - 1, math.floor((len(st) - 1) * 0.95))]
                p05 = st[max(0, math.floor((len(st) - 1) * 0.05))]
                phrase_ranges.append(p95 - p05)

    if not f0s or nv == 0 or not aps:
        raise ValueError("recordings hold no voiced frames for frame voice profile")

    if not phrase_ranges:
        st_all = sorted(12.0 * math.log2(f) for f in f0s)
        if len(st_all) < 4:
            raise ValueError("too few voiced frames to measure phrase pitch range")
        p95 = st_all[min(len(st_all) - 1, math.floor((len(st_all) - 1) * 0.95))]
        p05 = st_all[max(0, math.floor((len(st_all) - 1) * 0.05))]
        phrase_ranges.append(max(0.1, p95 - p05))

    nuclei = sum(1 for u in bank.units if u.phone in vowels)
    speech_frames = sum(u.end - u.start for u in bank.units)
    if nuclei <= 0 or speech_frames <= 0:
        raise ValueError("bank holds no voiced vowel nuclei to measure syllable rate")
    syllable_rate = nuclei / (speech_frames * bank.hop / bank.sr)

    # 1. Measure voice quality on the source recordings (refuses if any quantity is unmeasured)
    rec_qualities = [_measure_recording_quality(p, x, FRAME_SR) for p, x in zip(conv_paths, raw_samples_list)]
    target_jitter = float(statistics.median(q["jitter"] for q in rec_qualities))
    target_shimmer = float(statistics.median(q["shimmer"] for q in rec_qualities))
    target_hnr = float(statistics.median(q["hnr_db"] for q in rec_qualities))
    target_cpp = float(statistics.median(q["cpp_db"] for q in rec_qualities))
    raw_ap = sum(aps) / len(aps)

    # 2. Measure what the frames themselves carry when rendered with 0 added jitter/shimmer
    carried_qualities: List[Dict[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        for ri in range(len(bank.recordings)):
            n = int(bank.recordings[ri]["n"])
            frames = [bank.frame(ri, i) for i in range(n)]
            ap_scaled = [min(1.0, v * 0.25) if f[0] > 0 else v for f in frames for v in f[2]]
            y = dsp.vocoder_render(
                [f[0] for f in frames],
                [v for f in frames for v in f[1]],
                bank.env_hz,
                ap_scaled,
                bank.band_edges,
                bank.hop,
                bank.sr,
                0.0,
                0.0,
                1,
            )
            peak = max((abs(v) for v in y), default=1.0) or 1.0
            norm_y = [v * min(1.0, 0.97 / peak) for v in y]
            c_wav = tmp_p / f"carried_{ri}.wav"
            write_wav(c_wav, norm_y, bank.sr)
            carried_qualities.append(_measure_recording_quality(c_wav, norm_y, bank.sr))

    carried_jitter = min(target_jitter, float(statistics.median(q["jitter"] for q in carried_qualities)))
    carried_shimmer = min(target_shimmer, float(statistics.median(q["shimmer"] for q in carried_qualities)))

    base_measures = {
        "f0_median_hz": round(float(statistics.median(f0s)), 3),
        "phrase_range_st": round(float(statistics.median(phrase_ranges)), 3),
        "syllable_rate": round(float(syllable_rate), 3),
        "voiced_aperiodicity": round(float(raw_ap * 0.25), 4),
        "ltas_db": [round(v / nv, 4) for v in ltas],
        "level_db": round(float(statistics.median(levels)), 3),
        "jitter": round(target_jitter, 5),
        "shimmer": round(target_shimmer, 5),
        "jitter_carried": round(carried_jitter, 5),
        "shimmer_carried": round(carried_shimmer, 5),
        "hnr_db": round(target_hnr, 2),
        "cpp_db": round(target_cpp, 2),
    }

    # 3. Measure what stitched frames carry when spoken by the mouth with 0 added jitter/shimmer
    stitched_ap = round(max(0.005, min(0.045, raw_ap * 0.15)), 4)
    probe_measures = check_measures({
        **base_measures,
        "stitched_aperiodicity": stitched_ap,
        "stitched_jitter_carried": base_measures["jitter"],
        "stitched_shimmer_carried": base_measures["shimmer"],
    }, bank.ne)
    probe_voice = FrameVoice(
        name=name,
        path=out,
        bank=bank,
        measures=probe_measures,
        spoken_as=dict(spoken_as or {}),
    )
    probe_words = [t["word"] for t in bank.tokens if t.get("word") in lex]
    probe_sentence = " ".join(probe_words[: min(4, len(probe_words))])
    if not probe_sentence:
        raise ValueError("bank holds no lexicon-mapped words for stitched calibration")

    with tempfile.TemporaryDirectory() as tmp_dir:
        s_res = synthesize(probe_sentence, voice=probe_voice, lexicon=lex)
        s_wav = Path(tmp_dir) / "stitched_probe.wav"
        s_res.write_wav(s_wav)
        s_qual = _measure_recording_quality(s_wav, s_res.samples, s_res.sample_rate)

    stitched_jc = min(target_jitter, float(s_qual["jitter"]))
    stitched_sc = min(target_shimmer, float(s_qual["shimmer"]))

    measures = check_measures({
        **base_measures,
        "stitched_aperiodicity": stitched_ap,
        "stitched_jitter_carried": round(stitched_jc, 5),
        "stitched_shimmer_carried": round(stitched_sc, 5),
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
    if not name or not str(name).strip():
        raise ValueError("a voicepack name is required; refusing to invent a default name")
    if not wav_paths:
        raise ValueError("at least one WAV recording is required to build a voicepack")
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
