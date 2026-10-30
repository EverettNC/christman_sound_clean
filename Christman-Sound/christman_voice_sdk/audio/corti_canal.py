# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_voice_sdk.audio.corti_canal — Greenwood (1961) Cochlea, Cycle-by-Cycle
DSP Voice Quality, SpeakerSeat, Nonverbal Grunt Physics & Stimming Router, and
Continuous 20 Hz Canal Recorder (ported from everettnc/corti and
everettnc/thewholehouse src/lib/audio/* & src/lib/sound/*).

Pure Python standard library + christman_dsp.so only. No numpy, no scipy.
An unmeasured value stays None; nothing here fills in a missing measurement.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

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
    "DISTRESS_MAX_ATTACK_MS",
    "DISTRESS_MIN_PEAK_RMS",
    "DISTRESS_MIN_F0_RATIO",
    "DISTRESS_UNSEATED_MIN_F0_HZ",
    "DISTRESS_UNSEATED_MIN_PEAK_RMS",
    "FRUSTRATION_MIN_DURATION_MS",
    "FRUSTRATION_MIN_PEAK_RMS",
    "STIMMING_MIN_EVENTS",
    "STIMMING_MAX_F0_CV",
    "STIMMING_MAX_DURATION_CV",
    "freq_to_place",
    "place_to_freq",
    "hz_to_note",
    "vad_from_dsp",
    "voice_quality_dsp",
    "SpeakerSeat",
    "VocalTracker",
    "Corpus",
    "GruntPhysicsRoute",
    "analyze_grunt_physics",
    "detect_vocal_stimming",
    "CanalFrame",
    "CanalRecorder",
    "elements_mounted",
    "write_canal_frames",
    "read_canal_range",
]

# Greenwood (1961) human cochlear frequency-position function constants:
# f(x) = A * (10^(alpha * x) - k), where x in [0, 1] is normalized distance from apex.
GREENWOOD_A = 165.4
GREENWOOD_ALPHA = 2.1
GREENWOOD_K = 0.88

# -----------------------------------------------------------------------------
# THRESHOLDS — EVERY NUMBER BELOW IS A CHOSEN STARTING VALUE, NOT A MEASUREMENT.
# Each is held until client data replaces it, with its reason written beside it:
# -----------------------------------------------------------------------------

#: Utterance onset gate (0.016 RMS, ~-36 dBFS). CHOSEN STARTING VALUE, held
#: until client data replaces it. Reason: carried from Corti vocalization.ts;
#: set at 2x the 0.008 silence floor (VAD_RMS_FLOOR) so room air does not open
#: an utterance.
ONSET_RMS = 0.016

#: Utterance offset hysteresis floor (0.0065 RMS). CHOSEN STARTING VALUE, held
#: until client data replaces it. Reason: carried from Corti vocalization.ts;
#: set below VAD_RMS_FLOOR (0.008) so a trailing vowel decay does not chatter
#: open and closed.
OFFSET_RMS = 0.0065

#: Voice-activity RMS silence floor (0.008 RMS, ~-42 dBFS). CHOSEN STARTING
#: VALUE, held until client data replaces it. Reason: carried from Corti
#: dsp.ts and house.ts; below 0.008 RMS the room is treated as silent.
VAD_RMS_FLOOR = 0.008

#: YIN periodicity confidence floor (0.62). CHOSEN STARTING VALUE, held until
#: client data replaces it. Reason: carried from Corti vocalization.ts; below
#: 0.62 (YIN cumulative mean normalized difference >= 0.38) the frame is
#: aperiodic and f0 stays None rather than guessing a pitch.
MIN_VOICED_PROB = 0.62

#: Per-utterance frame cap (360 frames, ~6 s at 60 Hz). CHOSEN STARTING VALUE,
#: held until client data replaces it. Reason: carried from Corti
#: vocalization.ts; bounds per-event memory while keeping the onset head intact.
TAPE_CAP = 360

#: Canal tick interval (50 ms = 20 Hz) and 2-hour ring cap (144,000 frames).
CANAL_SAMPLE_MS = 50
CANAL_MAX_FRAMES = 144_000
ELEMENTS_CANAL_DIR = Path("/Volumes/ELEMENTS/THEWHOLEHOUSE/canal")

#: Distress maximum attack time (85.0 ms). CHOSEN STARTING VALUE, NOT A
#: MEASUREMENT — held until client data replaces it. Reason: an involuntary
#: startle, pain cry, or acute distress reflex reaches peak amplitude within
#: ~50-85 ms, faster than a deliberate conversational vowel onset (~100-180 ms).
DISTRESS_MAX_ATTACK_MS = 85.0

#: Distress minimum peak loudness when pitch is not relied on (0.12 RMS).
#: CHOSEN STARTING VALUE, NOT A MEASUREMENT — held until client data replaces
#: it. Reason: 7.5x the 0.016 onset gate (ONSET_RMS), separating an acute
#: high-energy cry or call from a quiet breath or murmur.
DISTRESS_MIN_PEAK_RMS = 0.12

#: Distress minimum pitch jump relative to the speaker's seated resting F0
#: (1.35x resting pitch). CHOSEN STARTING VALUE, NOT A MEASUREMENT — held until
#: client data replaces it. Reason: 1.35x is ~5.2 semitones (a musical fourth)
#: above the speaker's seated median F0 in SpeakerSeat, marking a sharp upward
#: pitch excursion relative to that speaker's own voice.
DISTRESS_MIN_F0_RATIO = 1.35

#: Distress unseated pitch floor (260.0 Hz) and companion loudness floor
#: (0.08 RMS). CHOSEN STARTING VALUES, NOT MEASUREMENTS — held until client
#: data replaces them. Reason: 260 Hz sits just above C4 (261.63 Hz) and above
#: Corti's 230 Hz low-grunt bin (vocalization.ts) when SpeakerSeat is cold
#: (fewer than 3 voiced locks), paired with 0.08 RMS (5x ONSET_RMS) so faint
#: background sound cannot trip an unseated pitch gate.
DISTRESS_UNSEATED_MIN_F0_HZ = 260.0
DISTRESS_UNSEATED_MIN_PEAK_RMS = 0.08

#: Frustration minimum duration (450.0 ms) and companion loudness floor
#: (0.05 RMS). CHOSEN STARTING VALUES, NOT MEASUREMENTS — held until client
#: data replaces them. Reason: 450 ms matches Corti's groan duration boundary
#: (ms >= 430 in vocalization.ts, rounded to nine 50 ms canal frames = 450 ms)
#: for a sustained effortful groan, paired with 0.05 RMS (~3x ONSET_RMS) so a
#: faint sigh or mic hum is not read as frustration.
FRUSTRATION_MIN_DURATION_MS = 450.0
FRUSTRATION_MIN_PEAK_RMS = 0.05

#: Vocal stimming cadence gates: minimum 3 events, F0 coefficient of variation
#: <= 0.08 (~1.3 semitones), duration coefficient of variation <= 0.22.
#: CHOSEN STARTING VALUES, NOT MEASUREMENTS — held until client data replaces
#: them. Reason: at least 3 repetitions are needed to measure a recurring
#: rhythm (pstdev / fmean); 0.08 F0 CV keeps pitch near-constant across
#: repetitions and 0.22 duration CV keeps syllable length rhythmic.
STIMMING_MIN_EVENTS = 3
STIMMING_MAX_F0_CV = 0.08
STIMMING_MAX_DURATION_CV = 0.22

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


def _resample_cycle(cyc: Sequence[float], target_len: int) -> List[float]:
    n = len(cyc)
    if n == 0:
        return [0.0] * target_len
    if n == 1:
        return [float(cyc[0])] * target_len
    out = [0.0] * target_len
    scale = float(n) / float(target_len)
    last = n - 1
    for k in range(target_len):
        pos = k * scale
        i0 = int(pos)
        if i0 >= last:
            out[k] = float(cyc[last])
        else:
            frac = pos - i0
            out[k] = float(cyc[i0]) * (1.0 - frac) + float(cyc[i0 + 1]) * frac
    return out


def voice_quality_dsp(
    samples: Sequence[float],
    sr: int,
    f0: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    x = [float(v) for v in samples]
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
        win_start = pos + min_lag
        win_end = min(len(x), pos + max_lag)
        if win_start >= win_end:
            break
        best_idx = win_start
        best_val = x[win_start]
        for idx in range(win_start + 1, win_end):
            if x[idx] > best_val:
                best_val = x[idx]
                best_idx = idx
        peaks.append(best_idx)
        pos = best_idx

    if len(peaks) < 5:
        return {"jitter": None, "shimmer": None, "hnr_db": None}

    periods: List[float] = []
    amps: List[float] = []
    cycles: List[List[float]] = []
    target_len = max(8, int(round(t0)))

    for a, b in zip(peaks[:-1], peaks[1:]):
        p_samples = b - a
        if not (min_lag <= p_samples <= max_lag):
            continue
        cyc = x[a:b]
        periods.append(float(p_samples) / sr)
        amps.append(max(abs(v) for v in cyc))
        cycles.append(_resample_cycle(cyc, target_len))

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

    n_cyc = len(cycles)
    template = [sum(cycles[c][k] for c in range(n_cyc)) / n_cyc for k in range(target_len)]
    sig_pow = sum(v * v for v in template) / target_len
    resid_sq_sum = 0.0
    for cyc in cycles:
        for k in range(target_len):
            d = cyc[k] - template[k]
            resid_sq_sum += d * d
    noise_pow = max(1e-12, resid_sq_sum / (n_cyc * target_len))
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
    #: Minimum voiced utterances required before SpeakerSeat reports a resting
    #: baseline (3 locks). CHOSEN STARTING VALUE, held until client data
    #: replaces it. Reason: carried from Corti speaker.ts (MIN_VOICED_TO_SEAT = 3),
    #: the smallest sample that has a true middle value (median) rather than a
    #: single grunt.
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
class GruntPhysicsRoute:
    key: str
    intent: Optional[str]
    regulatory: bool
    urgent: bool
    confidence: Optional[float]
    physics_note: str
    duration_ms: Optional[float] = None
    attack_ms: Optional[float] = None
    peak_rms: Optional[float] = None
    f0_hz: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "intent": self.intent,
            "regulatory": self.regulatory,
            "urgent": self.urgent,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "duration_ms": None if self.duration_ms is None else round(self.duration_ms, 2),
            "attack_ms": None if self.attack_ms is None else round(self.attack_ms, 2),
            "peak_rms": None if self.peak_rms is None else round(self.peak_rms, 5),
            "f0_hz": None if self.f0_hz is None else round(self.f0_hz, 2),
            "physics_note": self.physics_note,
        }


def _extract_ms(event: Mapping[str, Any], *ms_keys: str, sec_key: Optional[str] = None) -> Optional[float]:
    """Read a millisecond measurement, or convert seconds to ms if present.
    Never fills in a default when the measurement is absent."""
    for k in ms_keys:
        if k in event and event[k] is not None:
            try:
                val = float(event[k])
                if math.isfinite(val):
                    return val
            except (TypeError, ValueError):
                return None
    if sec_key and sec_key in event and event[sec_key] is not None:
        try:
            val = float(event[sec_key]) * 1000.0
            if math.isfinite(val):
                return val
        except (TypeError, ValueError):
            return None
    return None


def _extract_float(event: Mapping[str, Any], *keys: str) -> Optional[float]:
    """Read a float measurement without inventing a fallback when missing."""
    for k in keys:
        if k in event and event[k] is not None:
            try:
                val = float(event[k])
                if math.isfinite(val):
                    return val
            except (TypeError, ValueError):
                return None
    return None


def _measure_stimming_cadence(events: Sequence[Mapping[str, Any]]) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    """Measure F0 and duration coefficient of variation across recent vocalizations.
    If any required measurement is missing on the recent events, returns (False, None, None, None)."""
    if len(events) < STIMMING_MIN_EVENTS:
        return False, None, None, None
    recent = list(events)[-4:]
    f0s: List[float] = []
    durs: List[float] = []
    for e in recent:
        if not isinstance(e, Mapping):
            return False, None, None, None
        f0_val = _extract_float(e, "f0", "median_f0")
        dur_val = _extract_ms(e, "durationMs", "duration_ms", sec_key="duration")
        if f0_val is None or f0_val <= 0.0 or dur_val is None or dur_val <= 0.0:
            return False, None, None, None
        f0s.append(f0_val)
        durs.append(dur_val)
    if len(f0s) < STIMMING_MIN_EVENTS:
        return False, None, None, None
    mean_f0 = statistics.fmean(f0s)
    mean_dur = statistics.fmean(durs)
    if mean_f0 <= 0.0 or mean_dur <= 0.0:
        return False, None, None, None
    f0_cv = statistics.pstdev(f0s) / mean_f0
    dur_cv = statistics.pstdev(durs) / mean_dur
    if f0_cv <= STIMMING_MAX_F0_CV and dur_cv <= STIMMING_MAX_DURATION_CV:
        f0_stability = max(0.0, min(1.0, 1.0 - (f0_cv / STIMMING_MAX_F0_CV)))
        dur_stability = max(0.0, min(1.0, 1.0 - (dur_cv / STIMMING_MAX_DURATION_CV)))
        measured_conf = round(max(0.01, min(1.0, 0.5 * (f0_stability + dur_stability))), 4)
        return True, measured_conf, f0_cv, dur_cv
    return False, None, f0_cv, dur_cv


def detect_vocal_stimming(events: Sequence[Mapping[str, Any]]) -> bool:
    matched, _conf, _f0_cv, _dur_cv = _measure_stimming_cadence(events)
    return matched


def analyze_grunt_physics(
    event: Optional[Mapping[str, Any]],
    seat: Optional[SpeakerSeat] = None,
    recent_events: Sequence[Mapping[str, Any]] = (),
) -> GruntPhysicsRoute:
    """
    Route a nonverbal vocalization from its measured envelope (duration, attack,
    peak RMS, and F0).

    - Never fills in missing measurements: if duration, attack, or peak_rms is
      missing (or <= 0 for duration/peak_rms, or < 0 for attack), returns
      key="unknown" with confidence=None and names the missing measurements in
      physics_note.
    - Confidence is computed from the measured quantities of the sound itself,
      never a fixed constant.
    """
    if recent_events:
        stim_ok, stim_conf, f0_cv, dur_cv = _measure_stimming_cadence(recent_events)
        if stim_ok and stim_conf is not None and f0_cv is not None and dur_cv is not None:
            return GruntPhysicsRoute(
                key="vocal_stimming",
                intent="Self-regulation / rhythmic vocal stimming (do not interrupt or treat as distress)",
                regulatory=True,
                urgent=False,
                confidence=stim_conf,
                physics_note=f"measured rhythmic cadence across {len(list(recent_events)[-4:])} vocalizations (F0 CV {f0_cv:.4f} <= {STIMMING_MAX_F0_CV}, duration CV {dur_cv:.4f} <= {STIMMING_MAX_DURATION_CV})",
            )

    if not event or not isinstance(event, Mapping):
        return GruntPhysicsRoute(
            key="unknown",
            intent=None,
            regulatory=False,
            urgent=False,
            confidence=None,
            physics_note="could not measure: no vocal event provided",
        )

    raw_dur = _extract_ms(event, "durationMs", "duration_ms", sec_key="duration")
    raw_attack = _extract_ms(event, "attackMs", "attack_ms", sec_key="attack")
    raw_rms = _extract_float(event, "peakRms", "peak_rms")
    raw_f0 = _extract_float(event, "f0", "median_f0")

    dur_ms = raw_dur if (raw_dur is not None and raw_dur > 0.0) else None
    attack_ms = raw_attack if (raw_attack is not None and raw_attack >= 0.0) else None
    peak_rms = raw_rms if (raw_rms is not None and raw_rms > 0.0) else None
    f0 = raw_f0 if (raw_f0 is not None and raw_f0 > 0.0) else None

    missing: List[str] = []
    if dur_ms is None:
        missing.append("duration_ms")
    if attack_ms is None:
        missing.append("attack_ms")
    if peak_rms is None:
        missing.append("peak_rms")

    if missing:
        return GruntPhysicsRoute(
            key="unknown",
            intent=None,
            regulatory=False,
            urgent=False,
            confidence=None,
            duration_ms=dur_ms,
            attack_ms=attack_ms,
            peak_rms=peak_rms,
            f0_hz=f0,
            physics_note=f"could not measure: missing {', '.join(missing)}",
        )

    assert dur_ms is not None and attack_ms is not None and peak_rms is not None

    base = seat.baseline() if (seat and seat.seated) else None
    f0_ratio = (
        (float(f0) / float(base["resting_f0_hz"]))
        if (f0 is not None and base and base.get("resting_f0_hz"))
        else None
    )

    loud_enough = peak_rms >= DISTRESS_MIN_PEAK_RMS
    seated_pitch_jump = f0_ratio is not None and f0_ratio >= DISTRESS_MIN_F0_RATIO
    unseated_high_pitch = (
        f0_ratio is None
        and f0 is not None
        and f0 >= DISTRESS_UNSEATED_MIN_F0_HZ
        and peak_rms >= DISTRESS_UNSEATED_MIN_PEAK_RMS
    )

    if attack_ms <= DISTRESS_MAX_ATTACK_MS and (loud_enough or seated_pitch_jump or unseated_high_pitch):
        attack_sharpness = max(0.0, min(1.0, (DISTRESS_MAX_ATTACK_MS - attack_ms) / DISTRESS_MAX_ATTACK_MS))
        rms_strength = max(0.0, min(1.0, peak_rms / (2.0 * DISTRESS_MIN_PEAK_RMS)))
        if f0_ratio is not None:
            pitch_strength = max(0.0, min(1.0, (f0_ratio - 1.0) / (2.0 * (DISTRESS_MIN_F0_RATIO - 1.0))))
        elif f0 is not None:
            pitch_strength = max(0.0, min(1.0, (f0 - DISTRESS_UNSEATED_MIN_F0_HZ) / DISTRESS_UNSEATED_MIN_F0_HZ))
        else:
            pitch_strength = rms_strength
        measured_conf = round(max(0.01, min(1.0, (attack_sharpness + rms_strength + pitch_strength) / 3.0)), 4)
        return GruntPhysicsRoute(
            key="grunt_distress",
            intent="Immediate discomfort, pain, or acute distress",
            regulatory=False,
            urgent=True,
            confidence=measured_conf,
            duration_ms=dur_ms,
            attack_ms=attack_ms,
            peak_rms=peak_rms,
            f0_hz=f0,
            physics_note=f"sharp onset ({attack_ms:.1f} ms <= {DISTRESS_MAX_ATTACK_MS:.0f} ms), peak RMS {peak_rms:.3f}, F0 {f0} Hz",
        )

    if dur_ms >= FRUSTRATION_MIN_DURATION_MS and peak_rms >= FRUSTRATION_MIN_PEAK_RMS:
        dur_excess = max(0.0, min(1.0, (dur_ms - FRUSTRATION_MIN_DURATION_MS) / FRUSTRATION_MIN_DURATION_MS))
        rms_excess = max(0.0, min(1.0, (peak_rms - FRUSTRATION_MIN_PEAK_RMS) / (3.0 * FRUSTRATION_MIN_PEAK_RMS)))
        measured_conf = round(max(0.01, min(1.0, 0.5 * (dur_excess + rms_excess))), 4)
        return GruntPhysicsRoute(
            key="grunt_frustration",
            intent="Effortful groan / frustration / need for assistance",
            regulatory=False,
            urgent=False,
            confidence=measured_conf,
            duration_ms=dur_ms,
            attack_ms=attack_ms,
            peak_rms=peak_rms,
            f0_hz=f0,
            physics_note=f"sustained vocalization ({dur_ms:.1f} ms >= {FRUSTRATION_MIN_DURATION_MS:.0f} ms) with peak RMS {peak_rms:.3f}",
        )

    if peak_rms < OFFSET_RMS:
        return GruntPhysicsRoute(
            key="unknown",
            intent=None,
            regulatory=False,
            urgent=False,
            confidence=None,
            duration_ms=dur_ms,
            attack_ms=attack_ms,
            peak_rms=peak_rms,
            f0_hz=f0,
            physics_note=f"could not measure: peak_rms {peak_rms:.5f} below offset floor {OFFSET_RMS}",
        )

    onset_calm = max(0.0, min(1.0, attack_ms / max(DISTRESS_MAX_ATTACK_MS, dur_ms)))
    brevity = max(0.0, min(1.0, (FRUSTRATION_MIN_DURATION_MS - dur_ms) / FRUSTRATION_MIN_DURATION_MS))
    audibility = max(0.0, min(1.0, (peak_rms - OFFSET_RMS) / max(1e-6, DISTRESS_MIN_PEAK_RMS - OFFSET_RMS)))
    measured_conf = round(max(0.01, min(1.0, (onset_calm + brevity + audibility) / 3.0)), 4)

    return GruntPhysicsRoute(
        key="grunt_acknowledge",
        intent="Calm vocal check-in / acknowledgment / presence",
        regulatory=False,
        urgent=False,
        confidence=measured_conf,
        duration_ms=dur_ms,
        attack_ms=attack_ms,
        peak_rms=peak_rms,
        f0_hz=f0,
        physics_note=f"moderate onset ({attack_ms:.1f} ms), duration {dur_ms:.1f} ms, peak RMS {peak_rms:.3f}",
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
