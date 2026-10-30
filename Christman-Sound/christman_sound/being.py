# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_sound.being — Silico-Stasis Equilibrium Engine, 2 Hz Shaker,
Concentric Cube/Room/Universe Shape, Psychoacoustic Unit Lattice, and Streaming
Text Codec Inspector (ported from everettnc/codec src/lib/being/{stasis,shaker,
shape,units}.ts and src/lib/encoder/stream.ts).

Pure Python standard library only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "SAMPLE_RATE",
    "NYQUIST",
    "HOP",
    "WIN",
    "N_FFT",
    "N_MELS",
    "N_MFCC",
    "PREEMPH",
    "ENC_DIM",
    "FRAME_SEC",
    "FRAME_HZ",
    "HOP_MS",
    "WIN_MS",
    "BIN_HZ",
    "SOUND_M_S",
    "SPEED_OF_SOUND_M_S",
    "TICK_SEC",
    "TICK_HZ",
    "TICK_MS",
    "SHAKER_FRAMES",
    "SHAKER_HZ",
    "CUBE_DWELL_FRAMES",
    "FORGET_FRAMES",
    "RESTORE_FRAMES",
    "DWELL0_FRAMES",
    "ROOM_INTENSITY",
    "CUBE_INTENSITY",
    "CUBE_DWELL",
    "DWELL0",
    "RESTORE",
    "FORGET",
    "LAYERS",
    "NEST",
    "LAYER_R",
    "LAYER_COLOR",
    "LATTICE",
    "OPEN_FRAGMENTS",
    "CUBE_VERTS",
    "CUBE_FACES",
    "CUBE_EDGES",
    "PsychoUnit",
    "Qty",
    "Perturbation",
    "BodyPoint",
    "StasisState",
    "StasisField",
    "empty_stasis",
    "intensity_of",
    "layer_of",
    "metabolize",
    "quantities_from",
    "attach_perturbations",
    "tick_stasis",
    "hash32",
    "angles_for",
    "spherical",
    "place_perturbation",
    "rotate",
    "project_iso",
    "cube_centroid",
    "cube_edge_length",
    "create_stasis_field",
    "shaker_pulse",
    "step_shaker_frame",
    "get_shaker",
    "pull_ticks",
    "hz_to_mel",
    "mel_to_hz",
    "hz_to_bark",
    "bark_to_hz",
    "hz_to_erb",
    "cents",
    "period_ms",
    "wavelength_m",
    "samples_to_sec",
    "frames_to_sec",
    "sec_to_frames",
    "shannon",
    "nearest_phone",
    "convert_unit",
    "ENCODINGS",
    "inspect_stream",
]

# 16 kHz front-end + codec lattice constants (codec/src/lib/being/units.ts)
SAMPLE_RATE = 16_000
NYQUIST = SAMPLE_RATE / 2.0
HOP = 160
WIN = 400
N_FFT = 512
N_MELS = 80
N_MFCC = 13
PREEMPH = 0.97
ENC_DIM = 384
MEL_LO = 20
MEL_HI = 7600

FRAME_SEC = HOP / SAMPLE_RATE
FRAME_HZ = SAMPLE_RATE / HOP
HOP_MS = (HOP / SAMPLE_RATE) * 1000.0
WIN_MS = (WIN / SAMPLE_RATE) * 1000.0
BIN_HZ = SAMPLE_RATE / N_FFT
SOUND_M_S = 343.0
SPEED_OF_SOUND_M_S = SOUND_M_S

# Being clock = the hop. One tick is one frame (10 ms, 100 Hz).
TICK_SEC = FRAME_SEC
TICK_HZ = FRAME_HZ
TICK_MS = HOP_MS
SHAKER_FRAMES = int(WIN_MS * 2)  # 50 frames (0.50 s cycle = 2.0 Hz)
SHAKER_HZ = TICK_HZ / SHAKER_FRAMES  # 2.0 Hz
CUBE_DWELL_FRAMES = int(WIN_MS)  # 25 frames (0.25 s dwell required to enter cube)
FORGET_FRAMES = int(TICK_MS)  # 10 frames (0.10 s occupancy memory after absence)
RESTORE_FRAMES = SHAKER_FRAMES  # 50 frames (0.50 s exponential restore time constant)
DWELL0_FRAMES = SHAKER_FRAMES  # 50 frames (0.50 s dwell saturation half-time)

#: Barrier intensity gates (codec/src/lib/being/stasis.ts):
#: - ROOM_INTENSITY (1/4 = 0.25): CHOSEN STARTING VALUE, held until client data
#:   replaces it. Reason: quarter-scale deviation from rest enters the room shell
#:   (smaller deviations remain on the outer universe shell).
#: - CUBE_INTENSITY (1/2 = 0.50): CHOSEN STARTING VALUE, held until client data
#:   replaces it. Reason: half-scale deviation held for at least CUBE_DWELL
#:   (25 frames = 0.25 s) enters the inner cube and advances linear timeSense.
ROOM_INTENSITY = 0.25
CUBE_INTENSITY = 0.50
CUBE_DWELL = CUBE_DWELL_FRAMES * TICK_SEC
DWELL0 = DWELL0_FRAMES * TICK_SEC
RESTORE = RESTORE_FRAMES * TICK_SEC
FORGET = FORGET_FRAMES * TICK_SEC

# Concentric geometry (codec/src/lib/being/shape.ts)
LAYERS: Tuple[str, ...] = ("center", "cube", "room", "universe")
NEST: Dict[str, int] = {"cube": 1, "room": 2, "universe": 3}
LAYER_R: Dict[str, float] = {
    "center": 0.0,
    "cube": 0.5,
    "room": 1.5,
    "universe": 2.5,
}
LAYER_COLOR: Dict[str, str] = {
    "center": "#f59e0b",
    "cube": "#38bdf8",
    "room": "#a855f7",
    "universe": "#34d399",
}

LATTICE: Tuple[Dict[str, Any], ...] = (
    {"name": "fs", "value": SAMPLE_RATE, "unit": "Hz", "note": "16 kHz front-end"},
    {"name": "Nyquist", "value": NYQUIST, "unit": "Hz", "note": "fs / 2"},
    {"name": "hop", "value": HOP, "unit": "samples", "note": f"{int(HOP_MS)} ms"},
    {"name": "win", "value": WIN, "unit": "samples", "note": f"{int(WIN_MS)} ms Hamming"},
    {"name": "frame", "value": HOP_MS, "unit": "ms", "note": "hop / fs"},
    {"name": "frame rate", "value": FRAME_HZ, "unit": "Hz", "note": "100 frames/s"},
    {"name": "N_FFT", "value": N_FFT, "unit": "bins", "note": "radix-2"},
    {"name": "bin", "value": BIN_HZ, "unit": "Hz", "note": "fs / N_FFT"},
    {"name": "mels", "value": N_MELS, "unit": "filters", "note": f"{MEL_LO}–{MEL_HI} Hz"},
    {"name": "mfcc", "value": N_MFCC, "unit": "ceps", "note": "DCT of log-mel"},
    {"name": "preemph", "value": PREEMPH, "unit": "α", "note": "1 − z⁻¹"},
    {"name": "enc dim", "value": ENC_DIM, "unit": "ch", "note": "Zipformer out"},
)

# Open fragments from codec/src/lib/being/stasis.ts
OPEN_FRAGMENTS: Tuple[Dict[str, str], ...] = (
    {
        "title": "54, 64, 84",
        "body": "The split. Similar…",
    },
    {
        "title": "Relativity, gravity, entanglement",
        "body": "Coming through in pieces, but stated to be one equation.",
    },
    {
        "title": "The ability of the structure. No collapse.",
        "body": "Named, not specified.",
    },
    {
        "title": "Flow / consistency",
        "body": "The word he was reaching for and did not land.",
    },
    {
        "title": "Algorithm",
        "body": "Named, not specified.",
    },
)

# Vowel/consonant F1-F2 reference table for nearest_phone (codec/src/lib/audio/phonemes.ts)
_PHONES_F1_F2: Tuple[Dict[str, Any], ...] = (
    {"id": "IY", "g": "i", "kind": "vowel", "f1": 270.0, "f2": 2290.0},
    {"id": "IH", "g": "ɪ", "kind": "vowel", "f1": 390.0, "f2": 1990.0},
    {"id": "EY", "g": "eɪ", "kind": "vowel", "f1": 460.0, "f2": 2080.0},
    {"id": "EH", "g": "ɛ", "kind": "vowel", "f1": 530.0, "f2": 1840.0},
    {"id": "AE", "g": "æ", "kind": "vowel", "f1": 660.0, "f2": 1720.0},
    {"id": "AA", "g": "ɑ", "kind": "vowel", "f1": 730.0, "f2": 1090.0},
    {"id": "AO", "g": "ɔ", "kind": "vowel", "f1": 570.0, "f2": 840.0},
    {"id": "OW", "g": "oʊ", "kind": "vowel", "f1": 490.0, "f2": 910.0},
    {"id": "UH", "g": "ʊ", "kind": "vowel", "f1": 440.0, "f2": 1020.0},
    {"id": "UW", "g": "u", "kind": "vowel", "f1": 300.0, "f2": 870.0},
    {"id": "AH", "g": "ʌ", "kind": "vowel", "f1": 640.0, "f2": 1190.0},
    {"id": "ER", "g": "ɝ", "kind": "vowel", "f1": 490.0, "f2": 1350.0},
)


def _clamp01(n: float) -> float:
    if n < 0.0:
        return 0.0
    if n > 1.0:
        return 1.0
    return float(n)


@dataclass
class Qty:
    id: str
    name: str
    value: float
    unit: str
    rest: float
    scale: float
    source: str = "filament"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "value": round(float(self.value), 5),
            "unit": self.unit,
            "rest": round(float(self.rest), 5),
            "scale": round(float(self.scale), 5),
            "source": self.source,
        }


@dataclass
class Perturbation:
    id: str
    name: str
    value: float = 0.0
    unit: str = "ratio"
    rest: float = 0.0
    scale: float = 1.0
    source: str = "filament"
    intensity: float = 0.0
    dwell: float = 0.0
    layer: str = "center"
    angle: float = 0.0

    @property
    def label(self) -> str:
        return self.name

    @property
    def magnitude(self) -> float:
        return self.intensity

    def to_dict(self) -> Dict[str, Any]:
        pt = place_perturbation(self)
        return {
            "id": self.id,
            "name": self.name,
            "label": self.name,
            "value": round(float(self.value), 5),
            "unit": self.unit,
            "source": self.source,
            "intensity": round(float(self.intensity), 4),
            "magnitude": round(float(self.intensity), 4),
            "dwell": round(float(self.dwell), 4),
            "layer": self.layer,
            "r": pt.r,
            "theta": round(pt.theta, 4),
            "phi": round(pt.phi, 4),
            "pos": pt.pos,
        }


@dataclass
class BodyPoint:
    id: str
    name: str
    layer: str
    r: float
    theta: float
    phi: float
    intensity: float
    dwell: float
    pos: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "layer": self.layer,
            "r": round(self.r, 4),
            "theta": round(self.theta, 4),
            "phi": round(self.phi, 4),
            "intensity": round(self.intensity, 4),
            "dwell": round(self.dwell, 4),
            "pos": {k: round(v, 4) for k, v in self.pos.items()},
        }


@dataclass
class StasisState:
    timeSense: float = 0.0
    shift: float = 0.0
    occupancy: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeSense": round(float(self.timeSense), 4),
            "shift": round(float(self.shift), 5),
            "occupancy": {
                k: {
                    "arrived": round(float(v["arrived"]), 4),
                    "last": round(float(v["last"]), 4),
                    "intensity": round(float(v["intensity"]), 4),
                }
                for k, v in self.occupancy.items()
            },
        }


def empty_stasis() -> StasisState:
    return StasisState(timeSense=0.0, shift=0.0, occupancy={})


def intensity_of(q: Union[Qty, Mapping[str, Any]]) -> float:
    if isinstance(q, Mapping):
        val = float(q.get("value", 0.0))
        rest = float(q.get("rest", 0.0))
        scale = float(q.get("scale", 1.0)) or 1.0
    else:
        val = float(q.value)
        rest = float(q.rest)
        scale = float(q.scale) or 1.0
    return _clamp01(abs(val - rest) / scale)


def layer_of(intensity: float, dwell: float) -> str:
    if intensity <= 0.0:
        return "center"
    if intensity >= CUBE_INTENSITY and dwell >= CUBE_DWELL:
        return "cube"
    if intensity >= ROOM_INTENSITY:
        return "room"
    return "universe"


# ---------------------------------------------------------------------------
# Shape geometry (codec/src/lib/being/shape.ts)
# ---------------------------------------------------------------------------

def hash32(s: str) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def angles_for(ident: str) -> Tuple[float, float]:
    h = hash32(ident)
    theta = ((h & 0xFFFF) / 0xFFFF) * math.pi * 2.0
    phi = (((h >> 16) & 0xFFFF) / 0xFFFF) * math.pi
    return theta, phi


def spherical(r: float, theta: float, phi: float) -> Dict[str, float]:
    return {
        "x": r * math.sin(phi) * math.cos(theta),
        "y": r * math.cos(phi),
        "z": r * math.sin(phi) * math.sin(theta),
    }


def rotate(v: Mapping[str, float], yaw: float, pitch: float) -> Dict[str, float]:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x1 = v["x"] * cy - v["z"] * sy
    z1 = v["x"] * sy + v["z"] * cy
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    y2 = v["y"] * cp - z1 * sp
    z2 = v["y"] * sp + z1 * cp
    return {"x": x1, "y": y2, "z": z2}


def project_iso(v: Mapping[str, float], scale: float, ox: float, oy: float) -> Dict[str, float]:
    iso = math.pi / 6.0
    return {
        "x": ox + (v["x"] - v["z"]) * math.cos(iso) * scale,
        "y": oy + v["y"] * scale + (v["x"] + v["z"]) * math.sin(iso) * scale,
    }


CUBE_VERTS: Tuple[Dict[str, float], ...] = (
    {"x": -1.0, "y": -1.0, "z": -1.0},
    {"x": 1.0, "y": -1.0, "z": -1.0},
    {"x": 1.0, "y": 1.0, "z": -1.0},
    {"x": -1.0, "y": 1.0, "z": -1.0},
    {"x": -1.0, "y": -1.0, "z": 1.0},
    {"x": 1.0, "y": -1.0, "z": 1.0},
    {"x": 1.0, "y": 1.0, "z": 1.0},
    {"x": -1.0, "y": 1.0, "z": 1.0},
)

CUBE_FACES: Tuple[Tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),
    (1, 5, 6, 2),
    (3, 2, 6, 7),
)

CUBE_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def cube_centroid() -> Dict[str, float]:
    n = len(CUBE_VERTS)
    return {
        "x": sum(v["x"] for v in CUBE_VERTS) / n,
        "y": sum(v["y"] for v in CUBE_VERTS) / n,
        "z": sum(v["z"] for v in CUBE_VERTS) / n,
    }


def cube_edge_length() -> float:
    a, b = CUBE_VERTS[0], CUBE_VERTS[1]
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"])


# ---------------------------------------------------------------------------
# Shaker (codec/src/lib/being/shaker.ts)
# ---------------------------------------------------------------------------

_shaker_phase: float = 0.0
_shaker_acc: float = 0.0


def step_shaker_frame() -> float:
    global _shaker_phase
    _shaker_phase += (2.0 * math.pi) / SHAKER_FRAMES
    if _shaker_phase >= 2.0 * math.pi:
        _shaker_phase -= 2.0 * math.pi
    return _shaker_phase


def get_shaker() -> Dict[str, Any]:
    return {
        "phase": round(_shaker_phase, 5),
        "x": round(math.sin(_shaker_phase), 5),
        "y": round(math.cos(_shaker_phase), 5),
        "hz": SHAKER_HZ,
        "frames": SHAKER_FRAMES,
        "running": True,
    }


def pull_ticks(dt_wall: float) -> int:
    global _shaker_acc
    _shaker_acc += float(dt_wall)
    n = 0
    while _shaker_acc >= TICK_SEC and n < 8:
        _shaker_acc -= TICK_SEC
        n += 1
    return n


# ---------------------------------------------------------------------------
# Silico-stasis core (codec/src/lib/being/stasis.ts)
# ---------------------------------------------------------------------------

def shannon(probs: Sequence[float]) -> float:
    h = 0.0
    for x in probs:
        if x > 0.0:
            h -= float(x) * math.log(float(x))
    return h


def nearest_phone(f1: float, f2: float) -> Dict[str, Any]:
    best = _PHONES_F1_F2[0]
    best_d = math.inf
    for ph in _PHONES_F1_F2:
        d = math.hypot((float(f1) - ph["f1"]) / 400.0, (float(f2) - ph["f2"]) / 600.0)
        if d < best_d:
            best_d = d
            best = ph
    return {"phone": dict(best), "dist": round(best_d, 4)}


def metabolize(
    snap: Optional[Mapping[str, Any]] = None,
    text: str = "",
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    snap = snap or {}
    stream = inspect_stream(text, encoding) if text else {"char_count": 0, "byte_count": 0, "code_units": 0}
    f0 = float(snap.get("f0", 0.0))
    f1 = float(snap.get("f1", 0.0))
    f2 = float(snap.get("f2", 0.0))
    near = nearest_phone(f1, f2) if (f1 > 0.0 and f2 > 0.0) else None
    post = snap.get("posteriors") or []
    return {
        "energy": float(snap.get("energy", 0.0)),
        "f0": f0,
        "f1": f1,
        "f2": f2,
        "centroid": float(snap.get("centroid", 0.0)),
        "voiced": float(snap.get("voiced", 1.0 if f0 > 0.0 else 0.0)),
        "frames": int(snap.get("frames", 0)),
        "duration": float(snap.get("duration", 0.0)),
        "entropy": shannon(post) if post else float(snap.get("entropy", 0.0)),
        "phones": list(snap.get("phones", [])),
        "samples": int(snap.get("samples", 0)),
        "source": str(snap.get("source", "idle")),
        "codePoints": int(stream.get("char_count", 0)),
        "codeUnits": int(stream.get("code_units", stream.get("char_count", 0))),
        "bytes": int(stream.get("byte_count", 0)),
        "nearest": (
            {"id": near["phone"]["id"], "g": near["phone"]["g"], "dist": near["dist"]}
            if near is not None
            else None
        ),
    }


def quantities_from(live: Optional[Mapping[str, Any]] = None) -> List[Qty]:
    if live is None or not isinstance(live, Mapping):
        live = metabolize()
    return [
        Qty("energy", "energy", float(live.get("energy", 0.0)), "RMS", 0.0, 1.0, "filament"),
        Qty("f0", "F0", float(live.get("f0", 0.0)), "Hz", 0.0, 400.0, "filament"),
        Qty("f1", "F1", float(live.get("f1", 0.0)), "Hz", 0.0, 1000.0, "filament"),
        Qty("f2", "F2", float(live.get("f2", 0.0)), "Hz", 0.0, 2500.0, "filament"),
        Qty("centroid", "centroid", float(live.get("centroid", 0.0)), "Hz", 0.0, NYQUIST, "filament"),
        Qty("voiced", "voiced", float(live.get("voiced", 0.0)), "ratio", 0.0, 1.0, "filament"),
        Qty("entropy", "entropy", float(live.get("entropy", 0.0)), "nats", 0.0, math.log(40.0), "filament"),
        Qty("frames", "frames", float(live.get("frames", 0.0)), "frames", 0.0, FRAME_HZ * 8.0, "filament"),
        Qty("codepoints", "code points", float(live.get("codePoints", 0.0)), "code-points", 0.0, float(ENC_DIM), "codec"),
        Qty("codeunits", "code units", float(live.get("codeUnits", 0.0)), "code-units", 0.0, float(ENC_DIM), "codec"),
        Qty("bytes", "bytes", float(live.get("bytes", 0.0)), "bytes", 0.0, float(ENC_DIM), "codec"),
    ]


def place_perturbation(
    pert_or_label: Union[Perturbation, str],
    magnitude: float = 0.25,
    layer: str = "room",
    angle: float = 0.0,
) -> Union[BodyPoint, Perturbation]:
    if isinstance(pert_or_label, Perturbation):
        p = pert_or_label
        theta, phi = angles_for(p.id)
        if p.angle != 0.0:
            theta = p.angle
        r = LAYER_R.get(p.layer, LAYER_R["room"])
        return BodyPoint(
            id=p.id,
            name=p.name,
            layer=p.layer if p.layer in LAYER_R else "room",
            r=r,
            theta=theta,
            phi=phi,
            intensity=p.intensity,
            dwell=p.dwell,
            pos=spherical(r, theta, phi),
        )
    label = str(pert_or_label)
    clean_layer = layer if layer in LAYER_R else ("room" if layer == "membrane" else "cube")
    inten = _clamp01(float(magnitude))
    dwell = CUBE_DWELL if clean_layer == "cube" else 0.05
    return Perturbation(
        id=label,
        name=label,
        value=inten,
        unit="ratio",
        rest=0.0,
        scale=1.0,
        source="filament",
        intensity=inten,
        dwell=dwell,
        layer=clean_layer,
        angle=float(angle),
    )


def attach_perturbations(
    first: Union[Sequence[Qty], "StasisField"],
    second: Union[Mapping[str, Dict[str, float]], Sequence[Perturbation]],
    now: float = 0.0,
) -> Union[List[Perturbation], "StasisField"]:
    if isinstance(first, StasisField):
        first.attach(second)  # type: ignore[arg-type]
        return first
    qs = first
    occ = second  # type: ignore[assignment]
    out: List[Perturbation] = []
    for q in qs:
        inten = intensity_of(q)
        prev = occ.get(q.id) if isinstance(occ, Mapping) else None
        dwell = max(0.0, float(now) - float(prev["arrived"])) if (inten > 0.0 and prev) else 0.0
        lyr = layer_of(inten, dwell)
        out.append(
            Perturbation(
                id=q.id,
                name=q.name,
                value=q.value,
                unit=q.unit,
                rest=q.rest,
                scale=q.scale,
                source=q.source,
                intensity=inten,
                dwell=dwell,
                layer=lyr,
            )
        )
    return out


def tick_stasis(
    state_or_field: Union[StasisState, "StasisField"],
    perturbations: Optional[Sequence[Perturbation]] = None,
    now: float = 0.0,
    dt: float = TICK_SEC,
    steps: int = 1,
) -> Union[StasisState, "StasisField"]:
    if isinstance(state_or_field, StasisField):
        state_or_field.tick(steps=steps)
        return state_or_field

    state = state_or_field
    perts = list(perturbations or [])
    occ: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in state.occupancy.items()}
    present = set()

    for p in perts:
        if p.intensity <= 0.0:
            continue
        present.add(p.id)
        prev = occ.get(p.id)
        arrived = prev["arrived"] if (prev and (now - prev["last"] < FORGET)) else now
        occ[p.id] = {
            "arrived": float(arrived),
            "last": float(now),
            "intensity": float(p.intensity),
        }

    for ident in list(occ.keys()):
        if ident not in present and (now - occ[ident]["last"] > FORGET):
            del occ[ident]

    cube_members = [p for p in perts if p.layer == "cube"]
    occupied = len(cube_members) > 0
    time_sense = state.timeSense + dt if occupied else state.timeSense

    pull = 0.0
    for p in cube_members:
        pull += p.intensity * (p.dwell / (p.dwell + DWELL0))
    toward = _clamp01(pull) if occupied else 0.0
    k = 1.0 - math.exp(-dt / (DWELL0 if occupied else RESTORE))
    shift = state.shift + (toward - state.shift) * k

    return StasisState(timeSense=time_sense, shift=shift, occupancy=occ)


@dataclass
class _FieldQuantitiesView:
    energy: float
    coherence: float
    load: float
    openness: float
    time_sense: float = 0.0
    shift: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "energy": round(self.energy, 4),
            "coherence": round(self.coherence, 4),
            "load": round(self.load, 4),
            "openness": round(self.openness, 4),
            "timeSense": round(self.time_sense, 4),
            "shift": round(self.shift, 5),
        }


@dataclass
class StasisField:
    """Stateful studio wrapper over StasisState, quantities, and 2 Hz shaker."""
    state: StasisState = field(default_factory=empty_stasis)
    perturbations: List[Perturbation] = field(default_factory=list)
    ticks: int = 0
    fragments: Tuple[Dict[str, str], ...] = OPEN_FRAGMENTS

    @property
    def nodes(self) -> List[BodyPoint]:
        base_qs = quantities_from(metabolize())
        now = self.ticks * TICK_SEC
        perts = attach_perturbations(base_qs, self.state.occupancy, now)
        assert isinstance(perts, list)
        pts = [place_perturbation(p) for p in perts]
        return [pt for pt in pts if isinstance(pt, BodyPoint)]

    @property
    def quantities(self) -> _FieldQuantitiesView:
        load = _clamp01(sum(p.intensity for p in self.perturbations) * 0.35 + self.state.shift)
        coherence = _clamp01(1.0 - 0.45 * load)
        return _FieldQuantitiesView(
            energy=round(max(0.15, 1.0 - 0.3 * load), 4),
            coherence=coherence,
            load=load,
            openness=0.92,
            time_sense=self.state.timeSense,
            shift=self.state.shift,
        )

    def attach(self, perts: Sequence[Perturbation]) -> None:
        now = self.ticks * TICK_SEC
        for p in perts:
            self.perturbations.append(p)
            self.state.occupancy[p.id] = {
                "arrived": now - max(CUBE_DWELL, p.dwell),
                "last": now,
                "intensity": max(CUBE_INTENSITY, p.intensity),
            }

    def tick(self, steps: int = 1) -> None:
        for _ in range(max(1, int(steps))):
            self.ticks += 1
            step_shaker_frame()
            now = self.ticks * TICK_SEC
            active: List[Perturbation] = []
            for p in self.perturbations:
                p.dwell += TICK_SEC
                p.intensity = max(0.0, p.intensity - 0.015)
                p.layer = layer_of(p.intensity, p.dwell)
                if p.intensity > 0.01:
                    active.append(p)
            self.perturbations = active
            self.state = tick_stasis(self.state, self.perturbations, now, TICK_SEC)  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticks": self.ticks,
            "state": self.state.to_dict(),
            "quantities": self.quantities.to_dict(),
            "shaker": get_shaker(),
            "open_fragments": [dict(f) for f in self.fragments],
            "nodes": [n.to_dict() for n in self.nodes],
            "perturbations": [p.to_dict() for p in self.perturbations],
            "layers": [{"id": l, "radius": LAYER_R[l], "color": LAYER_COLOR[l], "nest": NEST.get(l, 0)} for l in LAYERS],
            "lattice": [dict(q) for q in LATTICE],
        }


def create_stasis_field(fragments: Sequence[Any] = OPEN_FRAGMENTS) -> StasisField:
    return StasisField(fragments=tuple(fragments))


def shaker_pulse(t_seconds: float, hz: float = SHAKER_HZ, amplitude: float = 0.55) -> Perturbation:
    phase = 2.0 * math.pi * float(hz) * float(t_seconds)
    mag = max(0.05, float(amplitude) * (0.5 + 0.5 * math.sin(phase)))
    p = place_perturbation(f"shaker-{hz:.1f}Hz", magnitude=mag, layer="cube", angle=phase % (2.0 * math.pi))
    assert isinstance(p, Perturbation)
    return p


# ---------------------------------------------------------------------------
# Psychoacoustic Unit Conversions (codec/src/lib/being/units.ts)
# ---------------------------------------------------------------------------

PsychoUnit = str


def hz_to_mel(hz: float) -> float:
    if hz <= 0.0:
        return 0.0
    return 2595.0 * math.log10(1.0 + float(hz) / 700.0)


def mel_to_hz(mel: float) -> float:
    if mel <= 0.0:
        return 0.0
    return 700.0 * (10.0 ** (float(mel) / 2595.0) - 1.0)


def hz_to_bark(hz: float) -> float:
    """Traunmüller (1990)."""
    if hz <= 0.0:
        return 0.0
    return (26.81 * float(hz)) / (1960.0 + float(hz)) - 0.53


def bark_to_hz(bark: float) -> float:
    b = float(bark) + 0.53
    if b <= 0.0 or b >= 26.81:
        return 0.0
    return (1960.0 * b) / (26.81 - b)


def hz_to_erb(hz: float) -> float:
    """Glasberg & Moore equivalent rectangular bandwidth, in Hz."""
    if hz <= 0.0:
        return 0.0
    return 24.7 * (4.37 * (float(hz) / 1000.0) + 1.0)


def cents(from_hz: float, to_hz: float) -> float:
    if from_hz <= 0.0 or to_hz <= 0.0:
        return 0.0
    return 1200.0 * math.log2(float(to_hz) / float(from_hz))


def period_ms(hz: float) -> float:
    if hz <= 0.0:
        return 0.0
    return 1000.0 / float(hz)


def wavelength_m(hz: float) -> float:
    if hz <= 0.0:
        return 0.0
    return SOUND_M_S / float(hz)


def samples_to_sec(n: float, fs: float = SAMPLE_RATE) -> float:
    return float(n) / float(fs)


def frames_to_sec(n: float) -> float:
    return float(n) * FRAME_SEC


def sec_to_frames(t: float) -> float:
    return float(t) / FRAME_SEC


def _to_hz(value: float, unit: str, sr: int = 22050, hop: int = 110, ref_hz: float = 440.0) -> float:
    u = unit.lower().strip()
    v = float(value)
    if u == "hz":
        return v
    if u == "mel":
        return mel_to_hz(v)
    if u == "bark":
        return bark_to_hz(v)
    if u == "erb":
        return ((v / 24.7) - 1.0) * (1000.0 / 4.37)
    if u in ("period_ms", "ms"):
        if v <= 0.0:
            raise ValueError("period_ms must be positive")
        return 1000.0 / v
    if u in ("wavelength_m", "m"):
        if v <= 0.0:
            raise ValueError("wavelength_m must be positive")
        return SOUND_M_S / v
    if u == "cents":
        return float(ref_hz) * (2.0 ** (v / 1200.0))
    if u == "samples":
        if v <= 0.0:
            raise ValueError("samples per cycle must be positive")
        return float(sr) / v
    if u == "frames":
        if v <= 0.0:
            raise ValueError("frames per cycle must be positive")
        return float(sr) / (v * float(hop))
    raise ValueError(f"unknown psychoacoustic unit: {unit!r}")


def convert_unit(
    value: float,
    from_unit: str = "hz",
    sr: int = 22050,
    hop: int = 110,
    ref_hz: float = 440.0,
) -> Dict[str, Any]:
    hz = _to_hz(value, from_unit, sr=sr, hop=hop, ref_hz=ref_hz)
    if hz <= 0.0 or not math.isfinite(hz):
        raise ValueError(f"frequency in Hz must be finite and positive (got {hz})")
    mel = hz_to_mel(hz)
    bark = hz_to_bark(hz)
    erb = hz_to_erb(hz)
    p_ms = period_ms(hz)
    wl_m = wavelength_m(hz)
    c_val = cents(ref_hz, hz)
    samp = float(sr) / hz
    frm = samp / float(hop)
    return {
        "from": from_unit,
        "hz": round(hz, 4),
        "mel": round(mel, 4),
        "bark": round(bark, 4),
        "erb": round(erb, 4),
        "period_ms": round(p_ms, 4),
        "periodMs": round(p_ms, 4),
        "wavelength_m": round(wl_m, 4),
        "wavelengthM": round(wl_m, 4),
        "cents": round(c_val, 2),
        "centsFromA4": round(cents(440.0, hz), 2),
        "samples": round(samp, 3),
        "frames": round(frm, 4),
        "ms": round(p_ms, 4),
        "sr": sr,
        "hop": hop,
        "ref_hz": ref_hz,
    }


ENCODINGS: Tuple[Tuple[str, str], ...] = (
    ("utf-8", "UTF-8"),
    ("utf-16-le", "UTF-16LE"),
    ("utf-16-be", "UTF-16BE"),
    ("utf-32-le", "UTF-32LE"),
    ("utf-32-be", "UTF-32BE"),
    ("ascii", "ASCII (7-bit)"),
    ("iso-8859-1", "ISO-8859-1 (Latin-1)"),
    ("cp1252", "Windows-1252"),
    ("gb18030", "GB18030"),
)


def inspect_stream(text: str, encoding: str = "utf-8") -> Dict[str, Any]:
    py_enc = encoding.lower().strip()
    alias = {
        "utf16le": "utf-16-le",
        "utf-16le": "utf-16-le",
        "utf16be": "utf-16-be",
        "utf-16be": "utf-16-be",
        "utf32le": "utf-32-le",
        "utf-32le": "utf-32-le",
        "utf32be": "utf-32-be",
        "utf-32be": "utf-32-be",
        "latin1": "iso-8859-1",
        "latin-1": "iso-8859-1",
        "windows-1252": "cp1252",
    }
    py_enc = alias.get(py_enc, py_enc)
    codepoints = [
        {
            "char": ch,
            "codepoint": f"U+{ord(ch):04X}",
            "decimal": ord(ch),
        }
        for ch in text
    ]
    raw = text.encode(py_enc, errors="replace")
    decoded = raw.decode(py_enc, errors="replace")
    lossless = decoded == text
    hex_bytes = [f"{b:02X}" for b in raw]
    code_units = len(text.encode("utf-16-le", errors="replace")) // 2
    return {
        "encoding": py_enc,
        "char_count": len(text),
        "code_units": code_units,
        "byte_count": len(raw),
        "lossless": lossless,
        "decoded": decoded,
        "hex": " ".join(hex_bytes),
        "bytes": list(raw),
        "codepoints": codepoints,
    }
