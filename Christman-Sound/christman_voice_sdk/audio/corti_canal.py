# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_voice_sdk.audio.corti_canal — Greenwood (1961) Cochlea, Cycle-by-Cycle
DSP Voice Quality, SpeakerSeat, AlphaVox Grunt Physics & Stimming Router, and
Continuous 20 Hz Canal Recorder (ported from everettnc/corti and
everettnc/thewholehouse src/lib/audio/* & src/lib/sound/*).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from christman_sound.dsp import get_dsp

__all__ = [
    "GREENWOOD_A",
    "GREENWOOD_ALPHA",
    "GREENWOOD_K",
    "ONSET_RMS",
    "OFFSET_RMS",
    "VAD_RMS_FLOOR",
    "MIN_VOICED_PROB",
    "TAPE_CAP",
    "CANAL_SAMPLE_MS",
    "CANAL_MAX_FRAMES",
    "ELEMENTS_CANAL_DIR",
    "freq_to_place",
    "place_to_freq",
    "hz_to_note",
    "vad_from_dsp",
    "voice_quality_dsp",
    "SpeakerSeat",
    "VocalTracker",
    "Corpus",
    "AlphaVoxRoute",
    "analyze_grunt_physics",
    "detect_vocal_stimming",
    "CanalFrame",
    "CanalRecorder",
    "elements_mounted",
    "write_canal_frames",
    "read_canal_range",
]

GREENWOOD_A = 165.4
GREENWOOD_ALPHA = 2.1
GREENWOOD_K = 0.88

ONSET_RMS = 0.016
OFFSET_RMS = 0.0065
VAD_RMS_FLOOR = 0.008
MIN_VOICED_PROB = 0.62
TAPE_CAP = 360

CANAL_SAMPLE_MS = 50
CANAL_MAX_FRAMES = 144_000
ELEMENTS_CANAL_DIR = Path("/Volumes/ELEMENTS/THEWHOLEHOUSE/canal")

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def freq_to_place(hz: float) -> float:
    if hz is None or hz <= 0 or not math.isfinite(hz):
        return 0.0
    pos = math.log10(float(hz) / GREENWOOD_A + GREENWOOD_K) / GREENWOOD_ALPHA
    return max(0.0, min(1.0, pos))


def place_to_freq(place: float) -> float:
    x = max(0.0, min(1.0, float(place)))
    return GREENWOOD_A * (10.0 ** (GREENWOOD_ALPHA * x) - GREENWOOD_K)


def hz_to_note(hz: Optional[float]) -> Optional[Dict[str, Any]]:
    if hz is None or hz <= 0 or not math.isfinite(hz):
        return None
    midi_exact = 69.0 + 12.0 * math.log2(float(hz) / 440.0)
    midi = int(round(midi_exact))
    cents = round((midi_exact - midi) * 100.0, 1)
    name = _NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    return {
        "note": f"{name}{octave}",
        "midi": midi,
        "cents": cents,
        "hz": round(float(hz), 2),
    }


def vad_from_dsp(rms: float, zcr: float, f0: Optional[float], f0_prob: float = 1.0) -> bool:
    if rms < VAD_RMS_FLOOR:
        return False
    if f0 is not None and f0 >= 55.0 and f0 <= 500.0 and f0_prob > MIN_VOICED_PROB:
        return True
    return rms >= ONSET_RMS and 0.01 <= zcr <= 0.38


def voice_quality_dsp(
    samples: Sequence[float],
    sr: int,
    f0: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < 256 or sr <= 0:
        return {"jitter": None, "shimmer": None, "hnr_db": None}
    dsp = get_dsp()
    if f0 is None:
        if dsp.available:
            raw_f0 = float(dsp.yin(x, sr))
            f0 = raw_f0 if 55.0 <= raw_f0 <= 500.0 else None
    if f0 is None or f0 <= 0.0:
        return {"jitter": None, "shimmer": None, "hnr_db": None}

    t0 = sr / float(f0)
    min_lag = max(4, int(round(t0 * 0.75)))
    max_lag = int(round(t0 * 1.25))
    peaks: List[int] = []
    pos = 0
    limit = len(x) - max_lag - 1
    while pos < limit:
        win_end = min(len(x), pos + max_lag)
        seg = x[pos + min_lag : win_end]
        if len(seg) == 0:
            break
        rel = int(np.argmax(seg))
        pk = pos + min_lag + rel
        peaks.append(pk)
        pos = pk

    if len(peaks) < 5:
        return {"jitter": None, "shimmer": None, "hnr_db": None}

    periods: List[float] = []
    amps: List[float] = []
    cycles: List[np.ndarray] = []
    target_len = max(8, int(round(t0)))
    grid = np.linspace(0.0, 1.0, target_len, endpoint=False)

    for a, b in zip(peaks[:-1], peaks[1:]):
        p_samples = b - a
        if not (min_lag <= p_samples <= max_lag):
            continue
        cyc = x[a:b]
        periods.append(float(p_samples) / sr)
        amps.append(float(np.max(np.abs(cyc))))
        src_grid = np.linspace(0.0, 1.0, len(cyc), endpoint=False)
        cycles.append(np.interp(grid, src_grid, cyc))

    if len(periods) < 4:
        return {"jitter": None, "shimmer": None, "hnr_db": None}

    mean_p = statistics.fmean(periods)
    mean_a = statistics.fmean(amps)
    jitter = (
        statistics.fmean(abs(periods[i] - periods[i - 1]) for i in range(1, len(periods))) / mean_p
        if mean_p > 1e-9
        else None
    )
    shimmer = (
        statistics.fmean(abs(amps[i] - amps[i - 1]) for i in range(1, len(amps))) / mean_a
        if mean_a > 1e-9
        else None
    )

    mat = np.vstack(cycles)
    template = np.mean(mat, axis=0)
    sig_pow = float(np.mean(template * template))
    resid = mat - template[None, :]
    noise_pow = max(1e-12, float(np.mean(resid * resid)))
    hnr_db: Optional[float] = None
    if sig_pow > 1e-12:
        hnr_db = round(min(60.0, 10.0 * math.log10(sig_pow / noise_pow)), 2)

    return {
        "jitter": round(float(jitter), 5) if jitter is not None else None,
        "shimmer": round(float(shimmer), 5) if shimmer is not None else None,
        "hnr_db": hnr_db,
    }


@dataclass
class SpeakerSeat:
    MIN_LOCKS: int = 3
    f0_locks: List[float] = field(default_factory=list)
    rms_locks: List[float] = field(default_factory=list)
    f1_locks: List[float] = field(default_factory=list)
    f2_locks: List[float] = field(default_factory=list)

    def observe(
        self,
        median_f0: Optional[float],
        peak_rms: Optional[float],
        f1: Optional[float] = None,
        f2: Optional[float] = None,
    ) -> None:
        if median_f0 is not None and 55.0 <= median_f0 <= 500.0:
            self.f0_locks.append(float(median_f0))
            if peak_rms is not None and peak_rms > 0:
                self.rms_locks.append(float(peak_rms))
            if f1 is not None and f1 > 0:
                self.f1_locks.append(float(f1))
            if f2 is not None and f2 > 0:
                self.f2_locks.append(float(f2))

    @property
    def seated(self) -> bool:
        return len(self.f0_locks) >= self.MIN_LOCKS

    def baseline(self) -> Optional[Dict[str, Any]]:
        if not self.seated:
            return None
        return {
            "seated": True,
            "locks": len(self.f0_locks),
            "resting_f0_hz": round(float(statistics.median(self.f0_locks)), 2),
            "resting_rms": round(float(statistics.median(self.rms_locks)), 5) if self.rms_locks else None,
            "resting_f1_hz": round(float(statistics.median(self.f1_locks)), 1) if self.f1_locks else None,
            "resting_f2_hz": round(float(statistics.median(self.f2_locks)), 1) if self.f2_locks else None,
        }


class VocalTracker:
    def __init__(self, ring_capacity: int = 8) -> None:
        self.ring: Deque[Dict[str, Any]] = deque(maxlen=ring_capacity)
        self.seat = SpeakerSeat()
        self._active = False
        self._onset_ms: float = 0.0
        self._peak_rms: float = 0.0
        self._peak_ms: float = 0.0
        self._last_f1: Optional[float] = None
        self._last_f2: Optional[float] = None
        self._tape: List[Dict[str, Any]] = []

    def step(
        self,
        t_ms: float,
        rms: float,
        zcr: float,
        f0: Optional[float] = None,
        f0_prob: float = 1.0,
        f1: Optional[float] = None,
        f2: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        locked_f0 = float(f0) if (f0 is not None and f0 > 0 and f0_prob > MIN_VOICED_PROB) else None
        if not self._active:
            if rms >= ONSET_RMS:
                self._active = True
                self._onset_ms = float(t_ms)
                self._peak_rms = float(rms)
                self._peak_ms = 0.0
                self._last_f1 = f1 if (f1 and f1 > 0) else None
                self._last_f2 = f2 if (f2 and f2 > 0) else None
                self._tape = [{"t": 0.0, "rms": round(float(rms), 5), "zcr": round(float(zcr), 5), "f0": locked_f0}]
            return None

        rel_t = max(0.0, float(t_ms) - self._onset_ms)
        if len(self._tape) < TAPE_CAP:
            self._tape.append({"t": round(rel_t, 2), "rms": round(float(rms), 5), "zcr": round(float(zcr), 5), "f0": locked_f0})
        if rms > self._peak_rms:
            self._peak_rms = float(rms)
            self._peak_ms = rel_t
        if f1 and f1 > 0:
            self._last_f1 = float(f1)
        if f2 and f2 > 0:
            self._last_f2 = float(f2)

        if rms < OFFSET_RMS:
            return self.close(t_ms)
        return None

    def close(self, t_ms: float) -> Optional[Dict[str, Any]]:
        if not self._active or not self._tape:
            self._active = False
            return None
        dur_ms = max(16.0, float(t_ms) - self._onset_ms)
        attack_ms = self._peak_ms
        decay_ms = max(0.0, dur_ms - self._peak_ms)
        f0s = [float(f["f0"]) for f in self._tape if f["f0"] is not None]
        med_f0 = round(float(statistics.median(f0s)), 2) if f0s else None
        zcrs = [float(f["zcr"]) for f in self._tape]
        mean_zcr = round(float(statistics.fmean(zcrs)), 5) if zcrs else 0.0
        voiced = len(f0s) > 0

        card = {
            "startedAt": round(self._onset_ms, 1),
            "durationMs": round(dur_ms, 1),
            "attackMs": round(attack_ms, 1),
            "decayMs": round(decay_ms, 1),
            "peakRms": round(self._peak_rms, 5),
            "f0": med_f0,
            "f1": round(self._last_f1, 1) if self._last_f1 else None,
            "f2": round(self._last_f2, 1) if self._last_f2 else None,
            "zcr": mean_zcr,
            "voiced": voiced,
            "tape": list(self._tape),
        }
        self.ring.append(card)
        if voiced and med_f0 is not None:
            self.seat.observe(med_f0, self._peak_rms, self._last_f1, self._last_f2)
        self._active = False
        self._tape = []
        return card


class Corpus:
    FREQUENCIES_HZ: Tuple[float, ...] = (110.0, 146.83, 196.0, 220.0, 261.63, 329.63, 440.0)

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def check_calibration_lock(
        self,
        target_hz: float,
        measured_hz: Optional[float],
        measured_prob: float = 1.0,
    ) -> Dict[str, Any]:
        if measured_hz is None or measured_hz <= 0 or measured_prob <= MIN_VOICED_PROB:
            rec = {
                "target_hz": target_hz,
                "measured_hz": None,
                "error_cents": None,
                "locked": False,
            }
        else:
            cents = 1200.0 * math.log2(float(measured_hz) / float(target_hz))
            rec = {
                "target_hz": target_hz,
                "measured_hz": round(float(measured_hz), 2),
                "error_cents": round(cents, 2),
                "locked": abs(cents) <= 25.0,
            }
        self.records.append(rec)
        return rec


@dataclass(frozen=True)
class AlphaVoxRoute:
    key: str
    intent: str
    regulatory: bool
    urgent: bool
    confidence: float
    physics_note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "intent": self.intent,
            "regulatory": self.regulatory,
            "urgent": self.urgent,
            "confidence": round(self.confidence, 3),
            "physics_note": self.physics_note,
        }


def detect_vocal_stimming(events: Sequence[Dict[str, Any]]) -> bool:
    if len(events) < 3:
        return False
    recent = list(events)[-4:]
    f0s = [float(e["f0"]) for e in recent if e.get("f0") is not None]
    durs = [float(e.get("durationMs") or e.get("duration", 0.0) * 1000.0) for e in recent]
    if len(f0s) < 3 or any(d <= 0 for d in durs):
        return False
    f0_cv = (statistics.pstdev(f0s) / statistics.fmean(f0s)) if statistics.fmean(f0s) > 0 else 1.0
    dur_cv = (statistics.pstdev(durs) / statistics.fmean(durs)) if statistics.fmean(durs) > 0 else 1.0
    return f0_cv <= 0.08 and dur_cv <= 0.22


def analyze_grunt_physics(
    event: Dict[str, Any],
    seat: Optional[SpeakerSeat] = None,
    recent_events: Sequence[Dict[str, Any]] = (),
) -> AlphaVoxRoute:
    if recent_events and detect_vocal_stimming(recent_events):
        return AlphaVoxRoute(
            key="vocal_stimming",
            intent="Self-regulation / rhythmic vocal stimming (do not interrupt or treat as distress)",
            regulatory=True,
            urgent=False,
            confidence=0.88,
            physics_note="steady F0 and duration cadence across >= 3 vocalizations",
        )

    dur_ms = float(event.get("durationMs") or (float(event.get("duration") or 0.2) * 1000.0))
    attack_ms = float(event.get("attackMs") if event.get("attackMs") is not None else (float(event.get("attack") or 0.08) * 1000.0))
    peak_rms = float(event.get("peakRms") or event.get("peak_rms") or 0.02)
    f0 = event.get("f0") or event.get("median_f0")

    base = seat.baseline() if (seat and seat.seated) else None
    f0_ratio = (float(f0) / float(base["resting_f0_hz"])) if (f0 and base and base.get("resting_f0_hz")) else None

    if attack_ms <= 85.0 and (peak_rms >= 0.12 or (f0_ratio is not None and f0_ratio >= 1.35) or (f0 is not None and float(f0) >= 260.0 and peak_rms >= 0.08)):
        return AlphaVoxRoute(
            key="grunt_distress",
            intent="Immediate discomfort, pain, or acute distress",
            regulatory=False,
            urgent=True,
            confidence=0.86,
            physics_note=f"sharp onset ({attack_ms:.0f} ms), peak RMS {peak_rms:.3f}, F0 {f0} Hz",
        )

    if dur_ms >= 450.0 and peak_rms >= 0.05:
        return AlphaVoxRoute(
            key="grunt_frustration",
            intent="Effortful groan / frustration / need for assistance",
            regulatory=False,
            urgent=False,
            confidence=0.79,
            physics_note=f"sustained vocalization ({dur_ms:.0f} ms) with peak RMS {peak_rms:.3f}",
        )

    return AlphaVoxRoute(
        key="grunt_acknowledge",
        intent="Calm vocal check-in / acknowledgment / presence",
        regulatory=False,
        urgent=False,
        confidence=0.75,
        physics_note=f"moderate onset ({attack_ms:.0f} ms), duration {dur_ms:.0f} ms",
    )


@dataclass(frozen=True)
class CanalFrame:
    t_ms: float
    rms: float
    zcr: float
    f0: Optional[float]
    f1: Optional[float] = None
    f2: Optional[float] = None
    place: Optional[float] = None
    vad: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t_ms, 1),
            "rms": round(self.rms, 5),
            "zcr": round(self.zcr, 5),
            "f0": None if self.f0 is None else round(self.f0, 2),
            "f1": None if self.f1 is None else round(self.f1, 1),
            "f2": None if self.f2 is None else round(self.f2, 1),
            "place": None if self.place is None else round(self.place, 4),
            "vad": self.vad,
        }


class CanalRecorder:
    def __init__(self, max_frames: int = CANAL_MAX_FRAMES) -> None:
        self.max_frames = max_frames
        self.frames: Deque[CanalFrame] = deque(maxlen=max_frames)

    def record(
        self,
        t_ms: float,
        rms: float,
        zcr: float,
        f0: Optional[float] = None,
        f1: Optional[float] = None,
        f2: Optional[float] = None,
    ) -> CanalFrame:
        clean_f0 = float(f0) if (f0 is not None and f0 > 0) else None
        place = freq_to_place(clean_f0) if clean_f0 is not None else None
        vad = vad_from_dsp(float(rms), float(zcr), clean_f0)
        frame = CanalFrame(
            t_ms=float(t_ms),
            rms=float(rms),
            zcr=float(zcr),
            f0=clean_f0,
            f1=float(f1) if (f1 and f1 > 0) else None,
            f2=float(f2) if (f2 and f2 > 0) else None,
            place=place,
            vad=vad,
        )
        self.frames.append(frame)
        return frame

    def slice_range(self, from_ms: float, to_ms: float) -> List[CanalFrame]:
        lo, hi = float(from_ms), float(to_ms)
        return [f for f in self.frames if lo <= f.t_ms <= hi]

    def recover_truncated_tape(self, onset_ms: float, duration_ms: float) -> List[Dict[str, Any]]:
        sliced = self.slice_range(onset_ms, onset_ms + duration_ms)
        return [
            {
                "t": round(f.t_ms - onset_ms, 2),
                "rms": round(f.rms, 5),
                "zcr": round(f.zcr, 5),
                "f0": None if f.f0 is None else round(f.f0, 2),
            }
            for f in sliced
        ]


def elements_mounted(mount_point: Path = Path("/Volumes/ELEMENTS")) -> bool:
    try:
        if not mount_point.is_dir():
            return False
        return mount_point.stat().st_dev != Path("/").stat().st_dev
    except OSError:
        return False


def _hour_filename(t_ms: float) -> str:
    dt = datetime.fromtimestamp(float(t_ms) / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d-%H.jsonl")


def write_canal_frames(
    frames: Sequence[CanalFrame],
    canal_dir: Optional[Path] = None,
    allow_local_dir: bool = False,
) -> Dict[str, Any]:
    target_dir = Path(canal_dir) if canal_dir is not None else ELEMENTS_CANAL_DIR
    if not allow_local_dir and str(target_dir).startswith("/Volumes/ELEMENTS"):
        if not elements_mounted(Path("/Volumes/ELEMENTS")):
            return {
                "written": 0,
                "mounted": False,
                "reason": "/Volumes/ELEMENTS is not mounted (device matches root filesystem)",
            }
    target_dir.mkdir(parents=True, exist_ok=True)
    by_hour: Dict[str, List[str]] = {}
    for f in frames:
        fname = _hour_filename(f.t_ms)
        by_hour.setdefault(fname, []).append(json.dumps(f.to_dict()))
    for fname, lines in by_hour.items():
        with (target_dir / fname).open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return {
        "written": len(frames),
        "mounted": True,
        "files": sorted(by_hour.keys()),
        "dir": str(target_dir),
    }


def read_canal_range(
    from_ms: float,
    to_ms: float,
    canal_dir: Optional[Path] = None,
) -> List[CanalFrame]:
    target_dir = Path(canal_dir) if canal_dir is not None else ELEMENTS_CANAL_DIR
    if not target_dir.is_dir():
        return []
    out: List[CanalFrame] = []
    for p in sorted(target_dir.glob("*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                t = float(d["t"])
                if from_ms <= t <= to_ms:
                    out.append(
                        CanalFrame(
                            t_ms=t,
                            rms=float(d["rms"]),
                            zcr=float(d["zcr"]),
                            f0=None if d.get("f0") is None else float(d["f0"]),
                            f1=None if d.get("f1") is None else float(d["f1"]),
                            f2=None if d.get("f2") is None else float(d["f2"]),
                            place=None if d.get("place") is None else float(d["place"]),
                            vad=bool(d.get("vad", False)),
                        )
                    )
        except (OSError, ValueError, KeyError):
            continue
    return sorted(out, key=lambda f: f.t_ms)
