# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_sound.dispersion — Multi-Canal Dispersion, Corti War, and Refused-Mix
Guard (ported from CODECWAR5TH src/lib/audio/{corti,scenes,war,mic-tape,noise}.ts
and src/lib/discern.ts).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dsp import get_dsp
from .synth import CODECWAR_SAMPLE_RATE, render_codecwar_table

__all__ = [
    "MAX_CANALS",
    "SCENE_DURATION",
    "CORTI_ONSET_RMS",
    "CORTI_OFFSET_RMS",
    "CORTI_TAPE_CAP",
    "RefusedMixError",
    "CanalSource",
    "SceneSpec",
    "SCENES",
    "kettle_pcm",
    "hvac_pcm",
    "render_canal_source",
    "card_from_frames",
    "tape_from_frames",
    "discern_canals",
    "fuse_pcm",
    "fused_claim",
    "sources_kept",
    "sources_lost",
    "mix_analysis_label",
    "refuse_mixed_canal",
    "run_war_scene",
]

MAX_CANALS = 4
SCENE_DURATION = 2.6
CORTI_ONSET_RMS = 0.016
CORTI_OFFSET_RMS = 0.0065
CORTI_MIN_VOICED_PROB = 0.62
CORTI_GRUNT_F0_MAX = 230.0
CORTI_TAPE_CAP = 360


class RefusedMixError(ValueError):
    """Raised when a caller attempts to read a single speaker's card from a fused multi-source mix."""


@dataclass(frozen=True)
class CanalSource:
    id: str
    label: str
    kind: str
    gain: float
    onset_s: float = 0.0
    text: str = ""
    phones: Tuple[Tuple[str, float], ...] = ()
    f0: float = 140.0
    rate: float = 1.0


@dataclass(frozen=True)
class SceneSpec:
    id: str
    title: str
    subtitle: str
    sources: Tuple[CanalSource, ...]


SCENES: Dict[str, SceneSpec] = {
    "kitchen": SceneSpec(
        id="kitchen",
        title="Kitchen — Speaker + Kettle + HVAC",
        subtitle="A person asking for water while the kettle whistles at 2.3 kHz and the HVAC hums at 120 Hz.",
        sources=(
            CanalSource(
                id="speaker-a",
                label="Person in kitchen",
                kind="voice",
                gain=0.85,
                onset_s=0.15,
                text="please bring me water",
                phones=(
                    ("P", 1.0), ("L", 1.0), ("IY", 1.0), ("Z", 1.0),
                    ("B", 1.0), ("R", 1.0), ("IH", 1.0), ("NG", 1.0),
                    ("M", 1.0), ("IY", 1.0),
                    ("W", 1.0), ("AO", 1.0), ("D", 0.5), ("ER", 1.0),
                ),
                f0=132.0,
                rate=1.0,
            ),
            CanalSource(
                id="kettle",
                label="Whistling kettle (2.3 kHz)",
                kind="kettle",
                gain=0.45,
                onset_s=0.0,
            ),
            CanalSource(
                id="hvac",
                label="HVAC blower (120 Hz + air)",
                kind="hvac",
                gain=0.25,
                onset_s=0.0,
            ),
        ),
    ),
    "two-booths": SceneSpec(
        id="two-booths",
        title="Two Booths — Overlapping Voices",
        subtitle="Two speakers talking over one another at 118 Hz and 215 Hz in the same room.",
        sources=(
            CanalSource(
                id="booth-low",
                label="Speaker 1 (118 Hz)",
                kind="voice",
                gain=0.80,
                onset_s=0.10,
                text="the filament is open",
                phones=(
                    ("DH", 1.0), ("AH", 1.0),
                    ("F", 1.0), ("IH", 1.0), ("L", 1.0), ("AH", 1.0), ("M", 1.0), ("AH", 1.0), ("N", 1.0), ("T", 1.0),
                    ("IH", 1.0), ("Z", 1.0),
                    ("OW", 1.0), ("P", 1.0), ("AH", 1.0), ("N", 1.0),
                ),
                f0=118.0,
                rate=1.0,
            ),
            CanalSource(
                id="booth-high",
                label="Speaker 2 (215 Hz)",
                kind="voice",
                gain=0.75,
                onset_s=0.25,
                text="read the red book",
                phones=(
                    ("R", 1.0), ("EH", 1.0), ("D", 1.0),
                    ("DH", 1.0), ("AH", 1.0),
                    ("R", 1.0), ("EH", 1.0), ("D", 1.0),
                    ("B", 1.0), ("UH", 1.0), ("K", 1.0),
                ),
                f0=215.0,
                rate=1.05,
            ),
        ),
    ),
    "crisis": SceneSpec(
        id="crisis",
        title="Crisis — Distressed Voice Under Room Noise",
        subtitle="A sharp-onset distressed vocalization masked by heavy appliance noise.",
        sources=(
            CanalSource(
                id="distressed",
                label="Distressed caller (265 Hz)",
                kind="voice",
                gain=0.90,
                onset_s=0.08,
                text="help me now",
                phones=(
                    ("HH", 1.0), ("EH", 1.0), ("L", 1.0), ("P", 1.0),
                    ("M", 1.0), ("IY", 1.0),
                    ("N", 1.0), ("AW", 1.0),
                ),
                f0=265.0,
                rate=1.15,
            ),
            CanalSource(
                id="kettle-heavy",
                label="Appliance hiss + whistle",
                kind="kettle",
                gain=0.55,
                onset_s=0.0,
            ),
            CanalSource(
                id="hvac-heavy",
                label="Ventilation rumble",
                kind="hvac",
                gain=0.35,
                onset_s=0.0,
            ),
        ),
    ),
}


def kettle_pcm(duration_s: float = SCENE_DURATION, sr: int = CODECWAR_SAMPLE_RATE, gain: float = 0.45) -> np.ndarray:
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    fm = 12.0 * np.sin(2.0 * np.pi * 5.5 * t)
    whistle = 0.65 * np.sin(2.0 * np.pi * (2320.0 + fm) * t) + 0.20 * np.sin(2.0 * np.pi * (4640.0 + 2.0 * fm) * t)
    rng = np.random.default_rng(2320)
    steam = rng.standard_normal(n) * 0.15
    env = np.minimum(1.0, t / 0.12) * np.minimum(1.0, (duration_s - t) / 0.12)
    return (whistle + steam) * env * float(gain)


def hvac_pcm(duration_s: float = SCENE_DURATION, sr: int = CODECWAR_SAMPLE_RATE, gain: float = 0.25) -> np.ndarray:
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    hum = (
        0.55 * np.sin(2.0 * np.pi * 60.0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 120.0 * t + 0.4)
        + 0.15 * np.sin(2.0 * np.pi * 180.0 * t + 1.1)
    )
    rng = np.random.default_rng(120)
    raw = rng.standard_normal(n)
    air = np.zeros(n, dtype=np.float64)
    alpha = 0.08
    acc = 0.0
    for i in range(n):
        acc = (1.0 - alpha) * acc + alpha * float(raw[i])
        air[i] = acc
    return (0.65 * hum + 0.35 * air) * float(gain)


def render_canal_source(src: CanalSource, duration_s: float = SCENE_DURATION, sr: int = CODECWAR_SAMPLE_RATE) -> np.ndarray:
    n = int(round(duration_s * sr))
    out = np.zeros(n, dtype=np.float64)
    offset = max(0, min(n, int(round(src.onset_s * sr))))
    if src.kind == "kettle":
        seg = kettle_pcm(duration_s - src.onset_s, sr=sr, gain=src.gain)
    elif src.kind == "hvac":
        seg = hvac_pcm(duration_s - src.onset_s, sr=sr, gain=src.gain)
    else:
        rendered = render_codecwar_table(
            None,
            phones=list(src.phones),
            f0=src.f0,
            rate=src.rate,
            sample_rate=sr,
        )
        seg = np.asarray(rendered.samples, dtype=np.float64) * float(src.gain)
    take = min(n - offset, len(seg))
    if take > 0:
        out[offset : offset + take] = seg[:take]
    return out


def tape_from_frames(samples: Sequence[float], sr: int = CODECWAR_SAMPLE_RATE, frame_ms: float = 16.0) -> List[Dict[str, Any]]:
    dsp = get_dsp()
    x = np.asarray(samples, dtype=np.float64)
    hop = max(1, int(round((frame_ms / 1000.0) * sr)))
    win = min(len(x), max(hop * 2, 1024))
    tape: List[Dict[str, Any]] = []
    for idx, start in enumerate(range(0, max(0, len(x) - win + 1), hop)):
        if len(tape) >= CORTI_TAPE_CAP:
            break
        chunk = x[start : start + win]
        t_ms = round(idx * frame_ms, 2)
        if dsp.available:
            rms_v = float(dsp.rms(chunk))
            zcr_v = float(dsp.zcr(chunk))
            f0_raw = float(dsp.yin(chunk, sr))
        else:
            rms_v = float(np.sqrt(np.mean(chunk * chunk)))
            signs = np.signbit(chunk)
            zcr_v = float(np.sum(signs[1:] != signs[:-1])) / max(1, len(chunk) - 1)
            f0_raw = 0.0
        f0_v: Optional[float] = round(f0_raw, 2) if (rms_v >= CORTI_OFFSET_RMS and 50.0 <= f0_raw <= 500.0) else None
        tape.append({
            "t": t_ms,
            "rms": round(rms_v, 5),
            "zcr": round(zcr_v, 5),
            "f0": f0_v,
        })
    return tape


def card_from_frames(samples: Sequence[float], sr: int = CODECWAR_SAMPLE_RATE, label: str = "canal") -> Dict[str, Any]:
    tape = tape_from_frames(samples, sr=sr)
    active = [f for f in tape if float(f["rms"]) >= CORTI_OFFSET_RMS]
    if not active:
        return {
            "label": label,
            "active": False,
            "durationMs": 0.0,
            "attackMs": None,
            "decayMs": None,
            "peakRms": 0.0,
            "f0": None,
            "f1": None,
            "f2": None,
            "zcr": 0.0,
            "voiced": False,
            "tape": tape,
        }
    t0 = float(active[0]["t"])
    t1 = float(active[-1]["t"])
    dur_ms = max(16.0, t1 - t0 + 16.0)
    peak_frame = max(active, key=lambda f: float(f["rms"]))
    peak_rms = float(peak_frame["rms"])
    peak_t = float(peak_frame["t"])
    attack_ms = max(0.0, peak_t - t0)
    decay_ms = max(0.0, t1 - peak_t)
    f0s = [float(f["f0"]) for f in active if f["f0"] is not None]
    med_f0 = round(float(statistics.median(f0s)), 2) if f0s else None
    mean_zcr = round(float(statistics.fmean(float(f["zcr"]) for f in active)), 5)
    voiced = len(f0s) >= max(2, len(active) // 4)

    dsp = get_dsp()
    f1_val: Optional[float] = None
    f2_val: Optional[float] = None
    if dsp.available and voiced:
        x = np.asarray(samples, dtype=np.float64)
        mid = len(x) // 2
        seg = x[max(0, mid - 1024) : min(len(x), mid + 1024)]
        if len(seg) >= 256:
            try:
                from .formants import measure_formants
                fm = measure_formants(seg, sr).summary
                f1_val = round(float(fm.f1_median_hz), 1) if (fm and fm.f1_median_hz) else None
                f2_val = round(float(fm.f2_median_hz), 1) if (fm and fm.f2_median_hz) else None
            except Exception:
                f1_val = None
                f2_val = None

    return {
        "label": label,
        "active": True,
        "durationMs": round(dur_ms, 1),
        "attackMs": round(attack_ms, 1),
        "decayMs": round(decay_ms, 1),
        "peakRms": round(peak_rms, 5),
        "f0": med_f0,
        "f1": f1_val,
        "f2": f2_val,
        "zcr": mean_zcr,
        "voiced": voiced,
        "tape": tape,
    }


def fuse_pcm(canals: Sequence[np.ndarray]) -> np.ndarray:
    if not canals:
        return np.zeros(0, dtype=np.float64)
    length = max(len(c) for c in canals)
    acc = np.zeros(length, dtype=np.float64)
    for c in canals:
        acc[: len(c)] += c
    peak = float(np.max(np.abs(acc))) if length else 0.0
    if peak > 0.99:
        acc = acc * (0.99 / peak)
    return acc


def sources_kept(canals: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(c["id"]) for c in canals[:MAX_CANALS]]


def sources_lost(canals: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(c["id"]) for c in canals[MAX_CANALS:]]


def mix_analysis_label(active_sources: int) -> str:
    if active_sources <= 0:
        return "silent — no active source"
    if active_sources == 1:
        return "single source — clean canal"
    return f"refused mix ({active_sources} concurrent sources on one wire; separate into canals first)"


def refuse_mixed_canal(active_sources: int) -> None:
    if active_sources > 1:
        raise RefusedMixError(mix_analysis_label(active_sources))


def fused_claim(fused_card: Dict[str, Any], separated_cards: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    voiced_canals = [c for c in separated_cards if c.get("voiced") and c.get("f0") is not None]
    true_f0s = [c["f0"] for c in voiced_canals]
    fused_f0 = fused_card.get("f0")
    corrupted = False
    if len(true_f0s) >= 2:
        corrupted = True
    elif len(true_f0s) == 1 and fused_f0 is not None:
        corrupted = abs(float(fused_f0) - float(true_f0s[0])) > 12.0
    elif len(true_f0s) == 1 and fused_f0 is None:
        corrupted = True
    return {
        "fused_f0_hz": fused_f0,
        "true_canal_f0s_hz": true_f0s,
        "pitch_corrupted_by_mix": corrupted,
        "fused_zcr": fused_card.get("zcr"),
        "verdict": mix_analysis_label(len(separated_cards)),
    }


def discern_canals(scene_id: str = "kitchen", sr: int = CODECWAR_SAMPLE_RATE) -> Dict[str, Any]:
    if scene_id not in SCENES:
        raise KeyError(f"unknown scene {scene_id!r}; choose from {sorted(SCENES)}")
    scene = SCENES[scene_id]
    kept_specs = scene.sources[:MAX_CANALS]
    pcm_list = [render_canal_source(s, duration_s=SCENE_DURATION, sr=sr) for s in kept_specs]
    canal_cards = [
        {
            "id": s.id,
            "label": s.label,
            "kind": s.kind,
            "text": s.text,
            "target_f0": s.f0 if s.kind == "voice" else None,
            "card": card_from_frames(pcm, sr=sr, label=s.label),
        }
        for s, pcm in zip(kept_specs, pcm_list)
    ]
    fused_audio = fuse_pcm(pcm_list)
    fused_card = card_from_frames(fused_audio, sr=sr, label="fused-mono-wire")
    comparison = fused_claim(fused_card, [c["card"] for c in canal_cards])
    return {
        "scene": {
            "id": scene.id,
            "title": scene.title,
            "subtitle": scene.subtitle,
            "max_canals": MAX_CANALS,
        },
        "sources_kept": [s.id for s in kept_specs],
        "sources_lost": [s.id for s in scene.sources[MAX_CANALS:]],
        "canals": canal_cards,
        "fused_wire": fused_card,
        "comparison": comparison,
    }


def run_war_scene(scene_id: str = "kitchen") -> Dict[str, Any]:
    return discern_canals(scene_id=scene_id)
