# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_voice_sdk.audio.lucent_reader — Gullstrand Two-Surface Cornea Optics,
4x4 Head Pose, 9-Cell Gaze, Blink Log, 5-Part Rhythm Tracker, and
Body-Signal Producer Bridge (ported from everettnc/lucent
src/lib/cornea/{optics,house-record}.ts and src/lib/reader/{geometry,rhythm,time,reading}.ts).

Pure Python standard library only. Unmeasured inputs stay None.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from .body_signals import BodySignalFrame

__all__ = [
    "N_AIR",
    "N_CORNEA",
    "N_AQUEOUS",
    "EYE_CLOSED_RATIO",
    "GAZE_CELL_THRESHOLD",
    "BLINK_THRESHOLD",
    "CORNEA_LAYERS",
    "CORNEA_PRESETS",
    "BODY_PARTS",
    "CorneaParams",
    "CorneaEvaluation",
    "evaluate_optics",
    "trace_meridian_rays",
    "pose_from_matrix",
    "gaze_from_blendshapes",
    "iris_from_landmarks",
    "mouth_from_landmarks",
    "BlinkLog",
    "RhythmTracker",
    "find_repetition",
    "local_stamp",
    "reading_to_body_signal_frame",
]

# Gullstrand schematic eye refractive indices:
N_AIR = 1.000
N_CORNEA = 1.376
N_AQUEOUS = 1.336

#: Palpebral fissure aspect ratio (height / width < 0.20) below which the lid
#: is closed and iris vertical position is not measurable. CHOSEN STARTING
#: VALUE, NOT A MEASUREMENT — held until client data replaces it. Reason:
#: carried from everettnc/lucent src/lib/reader/geometry.ts (EAR < 0.20).
EYE_CLOSED_RATIO = 0.20

#: Horizontal/vertical gaze deflection threshold (0.18) separating center/middle
#: from left/right/up/down in the 3x3 gaze grid. CHOSEN STARTING VALUE, NOT A
#: MEASUREMENT — held until client data replaces it. Reason: carried from
#: everettnc/lucent src/lib/reader/geometry.ts (~10 deg blendshape deflection).
GAZE_CELL_THRESHOLD = 0.18

#: Mean bilateral eyeBlink blendshape threshold (0.55) marking a blink event.
#: CHOSEN STARTING VALUE, NOT A MEASUREMENT — held until client data replaces
#: it. Reason: carried from everettnc/lucent src/lib/reader/geometry.ts (lid
#: closure past half-aperture).
BLINK_THRESHOLD = 0.55

CORNEA_LAYERS: Tuple[Dict[str, Any], ...] = (
    {"id": "epithelium", "name": "Epithelium", "thickness_um": 50.0, "role": "Barrier & tear-film optical interface"},
    {"id": "bowman", "name": "Bowman's Layer", "thickness_um": 12.0, "role": "Acellular collagen weave stabilizing anterior curvature"},
    {"id": "stroma", "name": "Stroma", "thickness_um": 470.0, "role": "90% of thickness; 200+ orthogonal lamellae for transparency"},
    {"id": "descemet", "name": "Descemet's Membrane", "thickness_um": 10.0, "role": "Elastic basement membrane supporting endothelium"},
    {"id": "endothelium", "name": "Endothelium", "thickness_um": 5.0, "role": "Na+/K+-ATPase pump maintaining stromal deturgescence (78% water)"},
)


@dataclass(frozen=True)
class CorneaParams:
    r_anterior_mm: float = 7.70
    r_posterior_mm: float = 6.80
    cct_um: float = 547.0
    asphericity_q: float = -0.26
    astigmatism_d: float = 0.0
    axis_deg: float = 0.0
    axial_length_mm: float = 24.0
    lens_power_d: float = 19.11
    endothelial_density: float = 2600.0


CORNEA_PRESETS: Dict[str, CorneaParams] = {
    "emmetropic": CorneaParams(7.70, 6.80, 547.0, -0.26, 0.0, 0.0, 24.0, 19.11, 2600.0),
    "myopia": CorneaParams(7.50, 6.70, 540.0, -0.26, 0.25, 180.0, 25.4, 19.11, 2600.0),
    "hyperopia": CorneaParams(7.95, 6.90, 555.0, -0.22, 0.25, 90.0, 22.8, 19.11, 2600.0),
    "astigmatism": CorneaParams(7.65, 6.80, 545.0, -0.26, 2.50, 90.0, 24.0, 19.11, 2600.0),
    "keratoconus": CorneaParams(6.40, 5.50, 430.0, -0.85, 4.75, 115.0, 24.1, 19.11, 2400.0),
    "post_lasik": CorneaParams(8.45, 6.80, 465.0, 0.35, 0.50, 180.0, 25.1, 19.11, 2550.0),
    "fuchs_edema": CorneaParams(7.80, 6.60, 650.0, -0.20, 1.00, 90.0, 24.0, 19.11, 620.0),
}


@dataclass(frozen=True)
class CorneaEvaluation:
    d_anterior: float
    d_posterior: float
    d_cornea_total: float
    d_eye_total: float
    focal_length_mm: float
    defocus_d: float
    stromal_scatter: float
    spherical_aberration_um: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "d_anterior": round(self.d_anterior, 3),
            "d_posterior": round(self.d_posterior, 3),
            "d_cornea_total": round(self.d_cornea_total, 3),
            "d_eye_total": round(self.d_eye_total, 3),
            "focal_length_mm": round(self.focal_length_mm, 3),
            "defocus_d": round(self.defocus_d, 3),
            "stromal_scatter": round(self.stromal_scatter, 4),
            "spherical_aberration_um": round(self.spherical_aberration_um, 4),
        }


def evaluate_optics(params: CorneaParams = CORNEA_PRESETS["emmetropic"]) -> CorneaEvaluation:
    r1_m = params.r_anterior_mm / 1000.0
    r2_m = params.r_posterior_mm / 1000.0
    t_m = params.cct_um * 1e-6
    d1 = (N_CORNEA - N_AIR) / r1_m
    d2 = (N_AQUEOUS - N_CORNEA) / r2_m
    d_cornea = d1 + d2 - (t_m / N_CORNEA) * d1 * d2
    d_eye = d_cornea + params.lens_power_d
    focal_mm = (N_AQUEOUS / d_eye) * 1000.0
    vitreous_demand_d = N_AQUEOUS / (params.axial_length_mm / 1000.0)
    defocus_d = d_eye - vitreous_demand_d
    edema_factor = max(0.0, (params.cct_um - 560.0) / 200.0) + max(0.0, (1000.0 - params.endothelial_density) / 1000.0)
    scatter = min(1.0, 0.04 + 0.55 * edema_factor)
    sph_aberr = 0.27 * (1.0 + params.asphericity_q / 0.52)
    return CorneaEvaluation(
        d_anterior=d1,
        d_posterior=d2,
        d_cornea_total=d_cornea,
        d_eye_total=d_eye,
        focal_length_mm=focal_mm,
        defocus_d=defocus_d,
        stromal_scatter=scatter,
        spherical_aberration_um=sph_aberr,
    )


def trace_meridian_rays(params: CorneaParams = CORNEA_PRESETS["emmetropic"], n_rays: int = 9, pupil_mm: float = 4.0) -> List[Dict[str, float]]:
    ev = evaluate_optics(params)
    half = pupil_mm * 0.5
    rays: List[Dict[str, float]] = []
    for i in range(max(3, n_rays)):
        y0 = -half + (2.0 * half * i) / (max(3, n_rays) - 1)
        r = params.r_anterior_mm
        q = params.asphericity_q
        disc = max(0.01, r * r - (1.0 + q) * y0 * y0)
        z_sag = (y0 * y0) / (r + math.sqrt(disc))
        slope = (y0 * (1.0 + q)) / math.sqrt(disc)
        inc_rad = math.atan(slope)
        sin_refr = max(-0.99, min(0.99, (N_AIR / N_AQUEOUS) * math.sin(inc_rad)))
        refr_rad = math.asin(sin_refr)
        ray_ang = inc_rad - refr_rad
        y_retina = y0 - math.tan(ray_ang) * (params.axial_length_mm - z_sag)
        rays.append({
            "y_pupil_mm": round(y0, 3),
            "z_sag_mm": round(z_sag, 4),
            "ray_angle_deg": round(math.degrees(ray_ang), 3),
            "y_retina_mm": round(y_retina, 4),
            "focal_mm": round(ev.focal_length_mm, 3),
        })
    return rays


def pose_from_matrix(m: Sequence[float]) -> Optional[Dict[str, float]]:
    if len(m) != 16:
        return None
    col_major = abs(float(m[3])) < 1e-4 and abs(float(m[7])) < 1e-4 and abs(float(m[11])) < 1e-4
    if col_major:
        r00, r10, r20 = float(m[0]), float(m[1]), float(m[2])
        r21, r22 = float(m[6]), float(m[10])
    else:
        r00, r10, r20 = float(m[0]), float(m[4]), float(m[8])
        r21, r22 = float(m[9]), float(m[10])
    pitch = math.degrees(math.atan2(-r20, math.hypot(r21, r22)))
    yaw = math.degrees(math.atan2(r10, r00))
    roll = math.degrees(math.atan2(r21, r22))
    return {
        "pitch_deg": round(pitch, 2),
        "yaw_deg": round(yaw, 2),
        "roll_deg": round(roll, 2),
        "layout": "column_major" if col_major else "row_major",
    }


def gaze_from_blendshapes(shapes: Optional[Mapping[str, float]]) -> Dict[str, Any]:
    """Compute 9-cell gaze and blink state from measured eye blendshapes.
    If no blendshapes are provided, returns None fields rather than inventing center gaze."""
    if not shapes or not isinstance(shapes, Mapping):
        return {
            "x": None,
            "y": None,
            "cell": None,
            "blink": None,
            "reason": "could not measure: no eye blendshapes provided",
        }
    left_in = float(shapes.get("eyeLookInLeft", 0.0))
    left_out = float(shapes.get("eyeLookOutLeft", 0.0))
    right_in = float(shapes.get("eyeLookInRight", 0.0))
    right_out = float(shapes.get("eyeLookOutRight", 0.0))
    up = 0.5 * (float(shapes.get("eyeLookUpLeft", 0.0)) + float(shapes.get("eyeLookUpRight", 0.0)))
    down = 0.5 * (float(shapes.get("eyeLookDownLeft", 0.0)) + float(shapes.get("eyeLookDownRight", 0.0)))
    blink_l = float(shapes.get("eyeBlinkLeft", 0.0))
    blink_r = float(shapes.get("eyeBlinkRight", 0.0))

    gx = 0.5 * ((right_out - right_in) + (left_in - left_out))
    gy = up - down
    col = "left" if gx < -GAZE_CELL_THRESHOLD else ("right" if gx > GAZE_CELL_THRESHOLD else "center")
    row = "up" if gy > GAZE_CELL_THRESHOLD else ("down" if gy < -GAZE_CELL_THRESHOLD else "middle")
    blink = 0.5 * (blink_l + blink_r) > BLINK_THRESHOLD
    return {
        "x": round(gx, 4),
        "y": round(gy, 4),
        "cell": f"{row}-{col}",
        "blink": blink,
    }


def iris_from_landmarks(
    outer_corner: Tuple[float, float],
    inner_corner: Tuple[float, float],
    upper_lid: Tuple[float, float],
    lower_lid: Tuple[float, float],
    iris_center: Tuple[float, float],
) -> Dict[str, Any]:
    width = math.hypot(inner_corner[0] - outer_corner[0], inner_corner[1] - outer_corner[1])
    height = math.hypot(upper_lid[0] - lower_lid[0], upper_lid[1] - lower_lid[1])
    if width <= 1e-6:
        return {"open_ratio": None, "closed": None, "x_norm": None, "y_norm": None}
    ratio = height / width
    closed = ratio < EYE_CLOSED_RATIO
    x_norm = (iris_center[0] - outer_corner[0]) / (inner_corner[0] - outer_corner[0] or 1e-6)
    y_norm = (iris_center[1] - lower_lid[1]) / (upper_lid[1] - lower_lid[1] or 1e-6) if not closed else None
    return {
        "open_ratio": round(ratio, 4),
        "closed": closed,
        "x_norm": round(x_norm, 4),
        "y_norm": None if y_norm is None else round(y_norm, 4),
    }


def mouth_from_landmarks(
    left_corner: Tuple[float, float],
    right_corner: Tuple[float, float],
    upper_lip: Tuple[float, float],
    lower_lip: Tuple[float, float],
) -> Dict[str, Optional[float]]:
    width = math.hypot(right_corner[0] - left_corner[0], right_corner[1] - left_corner[1])
    aperture = math.hypot(lower_lip[0] - upper_lip[0], lower_lip[1] - upper_lip[1])
    if width <= 1e-6:
        return {"aperture_ratio": None, "width": None}
    return {
        "aperture_ratio": round(aperture / width, 4),
        "width": round(width, 4),
    }


class BlinkLog:
    def __init__(self, window_ms: float = 60_000.0) -> None:
        self.window_ms = window_ms
        self._blinks: Deque[float] = deque()
        self._was_closed = False

    def observe(self, t_ms: float, closed: bool) -> bool:
        t = float(t_ms)
        fired = False
        if closed and not self._was_closed:
            self._blinks.append(t)
            fired = True
        self._was_closed = bool(closed)
        cutoff = t - self.window_ms
        while self._blinks and self._blinks[0] < cutoff:
            self._blinks.popleft()
        return fired

    def per_minute(self, now_ms: Optional[float] = None) -> Optional[float]:
        if not self._blinks:
            return 0.0 if now_ms is not None else None
        ref = float(now_ms) if now_ms is not None else self._blinks[-1]
        cutoff = ref - self.window_ms
        active = [b for b in self._blinks if b >= cutoff]
        return round(float(len(active)), 2)


BODY_PARTS: Tuple[str, ...] = ("head", "jaw", "hand_image_left", "hand_image_right", "shoulders")


def find_repetition(samples: Sequence[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    if len(samples) < 8:
        return None
    ordered = sorted(samples, key=lambda p: p[0])
    dur_s = (ordered[-1][0] - ordered[0][0]) / 1000.0
    if dur_s <= 0.25:
        return None
    vals = [float(v) for _, v in ordered]
    mean_v = sum(vals) / len(vals)
    centered = [v - mean_v for v in vals]
    amp = 0.5 * (max(vals) - min(vals))
    if amp < 1e-4:
        return None
    crossings = 0
    for a, b in zip(centered[:-1], centered[1:]):
        if (a <= 0.0 < b) or (a >= 0.0 > b):
            crossings += 1
    cycles = crossings / 2.0
    if cycles < 1.5:
        return None
    hz = cycles / dur_s
    return {
        "hz": round(hz, 3),
        "amplitude": round(amp, 4),
        "cycles": round(cycles, 2),
    }


class RhythmTracker:
    def __init__(self, window_ms: float = 5000.0) -> None:
        self.window_ms = window_ms
        self._series: Dict[str, Deque[Tuple[float, float]]] = {p: deque() for p in BODY_PARTS}

    def push(self, part: str, t_ms: float, value: float) -> None:
        if part not in self._series:
            raise KeyError(f"unknown body part {part!r}; expected one of {BODY_PARTS}")
        q = self._series[part]
        q.append((float(t_ms), float(value)))
        cutoff = float(t_ms) - self.window_ms
        while q and q[0][0] < cutoff:
            q.popleft()

    def summary(self) -> Dict[str, Optional[Dict[str, float]]]:
        return {part: find_repetition(list(q)) for part, q in self._series.items()}


def local_stamp(now: Optional[datetime] = None) -> Dict[str, Any]:
    dt = now or datetime.now().astimezone()
    hour = dt.hour
    tod = "night" if (hour < 6 or hour >= 21) else ("morning" if hour < 12 else ("afternoon" if hour < 17 else "evening"))
    return {
        "iso": dt.isoformat(timespec="seconds"),
        "hour": hour,
        "time_of_day": tod,
        "weekday": dt.strftime("%A"),
    }


def reading_to_body_signal_frame(
    t_ms: float,
    *,
    breath_freq: Optional[float] = None,
    breath_amp: Optional[float] = None,
    eye_jitter: Optional[float] = None,
    stare_lock: Optional[float] = None,
    tremor: Optional[float] = None,
    source: str = "lucent_reader",
) -> BodySignalFrame:
    return BodySignalFrame(
        t_ms=float(t_ms),
        breath_freq=breath_freq,
        breath_amp=breath_amp,
        eye_jitter=eye_jitter,
        stare_lock=stare_lock,
        tremor=tremor,
        source=source,
    )
