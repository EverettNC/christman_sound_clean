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
import os
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
from .build_voice_bank import VoiceBuildRefused, build_voice_bank
from .lexicon import LEXICON_ENV, Lexicon, LexiconUnavailable, discover_align_lexicon, load_lexicon, normalize_word
from .measure_voice import measure_voice_profile
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


def _ear_and_lexicon(
    lexicon_path: Optional[Union[str, Path]] = None,
    vosk_model: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """
    The lexicon file and the ear's model folder a measurement runs on.
    Nothing is assumed: with no lexicon found it refuses (LexiconUnavailable),
    with no model folder it refuses (VoiceBuildRefused). The model folder is
    the one passed, else $CHRISTMAN_VOSK_MODEL or $VOSK_MODEL_PATH, else the
    model the lexicon came from (<model>/graph/phones/align_lexicon.txt).
    """
    lpath = Path(lexicon_path).expanduser() if lexicon_path else discover_align_lexicon()
    if lpath is None or not Path(lpath).is_file():
        raise LexiconUnavailable(
            f"no lexicon: pass lexicon_path or set ${LEXICON_ENV} to "
            "vosk-model-en-us-0.22/graph/phones/align_lexicon.txt"
        )
    model = vosk_model or os.environ.get("CHRISTMAN_VOSK_MODEL") or os.environ.get("VOSK_MODEL_PATH")
    model_dir = Path(model).expanduser() if model else Path(lpath).expanduser().resolve().parents[2]
    if not model_dir.is_dir():
        raise VoiceBuildRefused(
            f"no ear model folder at {model_dir}: pass vosk_model or set $CHRISTMAN_VOSK_MODEL"
        )
    return Path(lpath), model_dir


def measure_and_prove_voice_profile(
    name: str,
    wav_paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    lexicon_path: Optional[Union[str, Path]] = None,
    vosk_model: Optional[Union[str, Path]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Measure a being's own recordings into a voice profile, and prove it.

    WHAT CHANGED AND WHY (2026-10-30, Everett's ruling: the voice matches the
    recordings by the numbers): this runs christman_sound.measure_voice, the
    same measurement scripts/measure_voice.py runs. Each vowel's F1, F2, F3
    come from that vowel's own voiced frames, the words' times from
    christman_sound.ear, the pitch contour and the rate from the phrases, and
    the before/after proof is the script's own (before, after, and after with
    each phrase held out). It used to take one average over all the audio and
    stretch Codec War's whole vowel table by it, fall back to 520, 1480 and
    2520 Hz when nothing was measured, mark every vowel "applied", and set the
    rate to 1.0 without measuring it. Now a vowel is applied only when it was
    measured from at least two tokens; the rest are listed under not_measured
    and measured_not_applied. A value that cannot be measured refuses
    (VoiceBuildRefused). Each recording's words come from its file name after
    `prefix`; a recording the ear hears as other words is left out and named.
    When the proof cannot be made the profile is still written and the proof
    says why, in place of an empty value.
    """
    paths = [Path(p).expanduser() for p in wav_paths]
    if not paths:
        raise VoiceBuildRefused("no recordings given: a voice profile is measured from the being's own recordings")
    lpath, model_dir = _ear_and_lexicon(lexicon_path, vosk_model)
    target_out = Path(out_path).expanduser()
    target_out.parent.mkdir(parents=True, exist_ok=True)
    proof_path = target_out.with_name(target_out.stem + ".proof.json")
    result = measure_voice_profile(
        paths=paths,
        out=target_out,
        name=name,
        vosk_model=model_dir,
        lexicon=lpath,
        prefix=prefix,
        prove=proof_path,
        script="christman_sound.voice_creation_center.measure_and_prove_voice_profile",
    )
    profile = result["profile"]
    loaded: VoiceProfile = load_voice_profile(target_out)
    files = profile["sources"]["recordings"]["files"]
    return {
        "name": loaded.name,
        "path": str(target_out),
        "fingerprint": loaded.fingerprint(),
        "f0_median_hz": loaded.f0_median_hz,
        "rate": loaded.rate,
        "vowels_measured": len(profile.get("vowels") or {}),
        "measured_not_applied": profile.get("measured_not_applied"),
        "not_measured": profile.get("not_measured"),
        "recordings_left_out": [{"name": f["name"], "why": f["excluded"]} for f in files if f["excluded"]],
        "contour_points": [list(pt) for pt in loaded.contour_points],
        "proof": result["proof"],
        "proof_path": str(proof_path) if proof_path.is_file() else None,
    }


def build_frame_voice_center(
    name: str,
    wav_paths: Sequence[Union[str, Path]],
    out_dir: Union[str, Path],
    lexicon_path: Optional[Union[str, Path]] = None,
    vosk_model: Optional[Union[str, Path]] = None,
    spoken_as: Optional[Mapping[str, str]] = None,
    min_conf: float = 0.9,
) -> FrameVoice:
    """
    Build a frame voice folder (bank.json, frames.bin, voice.json) from a
    being's own recordings and return it loaded.

    WHAT CHANGED AND WHY (2026-10-30, Everett's ruling: the voice matches the
    recordings by the numbers, jitter carried exactly): this runs
    christman_sound.build_voice_bank, the same build scripts/build_voice_bank.py
    runs. Every word's start, end and confidence come from christman_sound.ear,
    and a word under `min_conf` is not cut into units. Jitter, shimmer, HNR and
    CPPS are measured by Praat, and what the renderer's own frames carry is
    measured from renderings, calibrated on whole recordings and on stitched
    sentences. It used to split each recording into equal slices, one per
    word, each stamped with confidence 0.99; fill in jitter 0.015, shimmer
    0.045, HNR 14 dB and aperiodicity 0.12 for anything not measured; and
    make up the carried values by multiplying (jitter x 0.5 and x 0.6, CPPS as
    0.75 of HNR). Now a value that cannot be measured stops the build with its
    name (VoiceBuildRefused, or PraatUnavailable when Praat is not installed).
    Recordings must be 22050 Hz, 16-bit, mono WAV, as the build has always
    required: nothing converts them here.
    """
    paths = [Path(p).expanduser() for p in wav_paths]
    if not paths:
        raise VoiceBuildRefused("no recordings given: a frame voice is built from the being's own recordings")
    lpath, model_dir = _ear_and_lexicon(lexicon_path, vosk_model)
    out = Path(out_dir).expanduser()
    build_voice_bank(
        name=name,
        out=out,
        recordings=paths,
        profile_from=paths,
        vosk_model=model_dir,
        lexicon=lpath,
        min_conf=min_conf,
        spoken_as=dict(spoken_as or {}),
    )
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
