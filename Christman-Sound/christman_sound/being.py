# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_sound.being — Silico-Stasis Equilibrium Engine, 2 Hz Shaker,
Psychoacoustic Unit Converter, and Streaming Text Codec Inspector (ported from
everettnc/codec src/lib/being/{stasis,shaker,shape,units}.ts and
src/lib/encoder/stream.ts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

__all__ = [
    "LAYERS",
    "LAYER_R",
    "LAYER_COLOR",
    "OPEN_FRAGMENTS",
    "SHAKER_HZ",
    "SPEED_OF_SOUND_M_S",
    "PsychoUnit",
    "Fragment",
    "Quantities",
    "Perturbation",
    "Node",
    "StasisField",
    "metabolize",
    "quantities_from",
    "attach_perturbations",
    "place_perturbation",
    "tick_stasis",
    "create_stasis_field",
    "shaker_pulse",
    "convert_unit",
    "ENCODINGS",
    "inspect_stream",
]

LAYERS: Tuple[str, ...] = ("core", "mantle", "cortex", "membrane")
LAYER_R: Dict[str, float] = {
    "core": 0.18,
    "mantle": 0.42,
    "cortex": 0.68,
    "membrane": 0.92,
}
LAYER_COLOR: Dict[str, str] = {
    "core": "#f59e0b",
    "mantle": "#38bdf8",
    "cortex": "#a855f7",
    "membrane": "#34d399",
}

SHAKER_HZ = 2.0
SPEED_OF_SOUND_M_S = 343.0


@dataclass(frozen=True)
class Fragment:
    id: str
    layer: str
    label: str
    mass: float
    tension: float
    openness: float


OPEN_FRAGMENTS: Tuple[Fragment, ...] = (
    Fragment("sovereign-voice", "core", "Own Voice Only (No Substitute)", 0.95, 0.08, 0.92),
    Fragment("rule-13-honesty", "core", "Cardinal Rule 13 — Null Is Not Zero", 1.00, 0.05, 0.96),
    Fragment("cochlea-corti", "mantle", "Greenwood Cochlea + 360-Frame Tape", 0.82, 0.14, 0.88),
    Fragment("vosk-ear-022", "mantle", "Sovereign C/NumPy Kaldi Ear (0.22)", 0.88, 0.12, 0.90),
    Fragment("porch-mouth-coop", "cortex", "TCAP-V-0020 Mouth-Cooperation & Dialect", 0.76, 0.18, 0.86),
    Fragment("memory-chamber", "cortex", "TCAP-V-0044 Five-Element Memory Chamber", 0.84, 0.15, 0.89),
    Fragment("lucent-cornea", "membrane", "Gullstrand Cornea & 5-Part Rhythm Reader", 0.72, 0.16, 0.85),
    Fragment("skin-smell-house", "membrane", "OpenSmell 21-Profile & Haptic Skin Stream", 0.70, 0.19, 0.84),
)


@dataclass
class Quantities:
    energy: float
    coherence: float
    load: float
    openness: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "energy": round(self.energy, 4),
            "coherence": round(self.coherence, 4),
            "load": round(self.load, 4),
            "openness": round(self.openness, 4),
        }


@dataclass
class Perturbation:
    id: str
    label: str
    layer: str
    angle: float
    magnitude: float
    age: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "layer": self.layer,
            "angle": round(self.angle, 4),
            "magnitude": round(self.magnitude, 4),
            "age": self.age,
        }


@dataclass
class Node:
    id: str
    layer: str
    label: str
    r: float
    target_r: float
    theta: float
    omega: float
    mass: float
    tension: float
    openness: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "label": self.label,
            "r": round(self.r, 4),
            "target_r": round(self.target_r, 4),
            "theta": round(self.theta, 4),
            "omega": round(self.omega, 5),
            "x": round(self.r * math.cos(self.theta), 4),
            "y": round(self.r * math.sin(self.theta), 4),
            "mass": round(self.mass, 4),
            "tension": round(self.tension, 4),
            "openness": round(self.openness, 4),
        }


@dataclass
class StasisField:
    nodes: List[Node]
    quantities: Quantities
    perturbations: List[Perturbation] = field(default_factory=list)
    ticks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticks": self.ticks,
            "quantities": self.quantities.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
            "perturbations": [p.to_dict() for p in self.perturbations],
            "layers": [{"id": l, "radius": LAYER_R[l], "color": LAYER_COLOR[l]} for l in LAYERS],
        }


def quantities_from(fragments: Sequence[Fragment] = OPEN_FRAGMENTS) -> Quantities:
    if not fragments:
        return Quantities(energy=0.0, coherence=0.0, load=0.0, openness=0.0)
    n = len(fragments)
    energy = sum(f.mass * (1.0 - 0.3 * f.tension) for f in fragments) / n
    load = sum(f.tension * f.mass for f in fragments) / n
    openness = sum(f.openness for f in fragments) / n
    coherence = max(0.0, min(1.0, (energy * 0.55 + openness * 0.45) * (1.0 - 0.5 * load)))
    return Quantities(
        energy=max(0.0, min(1.0, energy)),
        coherence=coherence,
        load=max(0.0, min(1.0, load)),
        openness=max(0.0, min(1.0, openness)),
    )


def metabolize(fragments: Sequence[Fragment] = OPEN_FRAGMENTS) -> List[Node]:
    by_layer: Dict[str, List[Fragment]] = {l: [] for l in LAYERS}
    for f in fragments:
        by_layer.setdefault(f.layer, []).append(f)
    nodes: List[Node] = []
    for layer in LAYERS:
        items = by_layer.get(layer, [])
        base_r = LAYER_R.get(layer, 0.5)
        for idx, f in enumerate(items):
            theta = (2.0 * math.pi * idx) / max(1, len(items))
            omega = (0.012 / (base_r + 0.1)) * (1.0 if LAYERS.index(layer) % 2 == 0 else -1.0)
            nodes.append(
                Node(
                    id=f.id,
                    layer=layer,
                    label=f.label,
                    r=base_r,
                    target_r=base_r,
                    theta=theta,
                    omega=omega,
                    mass=f.mass,
                    tension=f.tension,
                    openness=f.openness,
                )
            )
    return nodes


def place_perturbation(label: str, magnitude: float = 0.25, layer: str = "membrane", angle: float = 0.0) -> Perturbation:
    clean_layer = layer if layer in LAYER_R else "membrane"
    return Perturbation(
        id=f"p-{abs(hash((label, magnitude, clean_layer, angle))) % 100000}",
        label=label,
        layer=clean_layer,
        angle=float(angle),
        magnitude=max(0.0, min(1.0, float(magnitude))),
        age=0,
    )


def attach_perturbations(field_state: StasisField, perturbations: Sequence[Perturbation]) -> StasisField:
    field_state.perturbations.extend(perturbations)
    return field_state


def tick_stasis(field_state: StasisField, steps: int = 1) -> StasisField:
    for _ in range(max(1, int(steps))):
        field_state.ticks += 1
        coh = field_state.quantities.coherence
        absorb = 0.004 * (0.4 + coh)
        remaining: List[Perturbation] = []
        for p in field_state.perturbations:
            p.age += 1
            p.magnitude = max(0.0, p.magnitude - absorb)
            if p.magnitude > 0.005:
                remaining.append(p)
        field_state.perturbations = remaining

        p_load = sum(p.magnitude for p in field_state.perturbations)
        for node in field_state.nodes:
            node.theta = (node.theta + node.omega) % (2.0 * math.pi)
            push = 0.0
            for p in field_state.perturbations:
                if p.layer == node.layer:
                    d_ang = math.cos(node.theta - p.angle)
                    push += p.magnitude * 0.06 * d_ang
            spring = (node.target_r - node.r) * 0.18
            node.r = max(0.05, min(1.15, node.r + spring + push))

        base_q = quantities_from(OPEN_FRAGMENTS)
        load = max(0.0, min(1.0, base_q.load + 0.25 * p_load))
        coherence = max(0.0, min(1.0, base_q.coherence * (1.0 - 0.22 * p_load)))
        field_state.quantities = Quantities(
            energy=base_q.energy,
            coherence=coherence,
            load=load,
            openness=base_q.openness,
        )
    return field_state


def create_stasis_field(fragments: Sequence[Fragment] = OPEN_FRAGMENTS) -> StasisField:
    return StasisField(
        nodes=metabolize(fragments),
        quantities=quantities_from(fragments),
        perturbations=[],
        ticks=0,
    )


def shaker_pulse(t_seconds: float, hz: float = SHAKER_HZ, amplitude: float = 0.18) -> Perturbation:
    phase = 2.0 * math.pi * hz * float(t_seconds)
    mag = max(0.01, float(amplitude) * (0.5 + 0.5 * math.sin(phase)))
    return place_perturbation(label=f"shaker-{hz:.1f}Hz", magnitude=mag, layer="membrane", angle=phase % (2.0 * math.pi))


PsychoUnit = str


def _to_hz(value: float, unit: str, sr: int = 22050, hop: int = 110, ref_hz: float = 440.0) -> float:
    u = unit.lower().strip()
    v = float(value)
    if u == "hz":
        return v
    if u == "mel":
        return 700.0 * (10.0 ** (v / 2595.0) - 1.0)
    if u == "bark":
        return 600.0 * math.sinh(v / 6.0)
    if u == "erb":
        return (10.0 ** (v / 21.4) - 1.0) / 0.00437
    if u in ("period_ms", "ms"):
        if v <= 0:
            raise ValueError("period_ms must be positive")
        return 1000.0 / v
    if u in ("wavelength_m", "m"):
        if v <= 0:
            raise ValueError("wavelength_m must be positive")
        return SPEED_OF_SOUND_M_S / v
    if u == "cents":
        return float(ref_hz) * (2.0 ** (v / 1200.0))
    if u == "samples":
        if v <= 0:
            raise ValueError("samples per cycle must be positive")
        return float(sr) / v
    if u == "frames":
        if v <= 0:
            raise ValueError("frames per cycle must be positive")
        return float(sr) / (v * float(hop))
    raise ValueError(f"unknown psychoacoustic unit: {unit!r}")


def convert_unit(
    value: float,
    from_unit: str = "hz",
    sr: int = 22050,
    hop: int = 110,
    ref_hz: float = 440.0,
) -> Dict[str, float]:
    hz = _to_hz(value, from_unit, sr=sr, hop=hop, ref_hz=ref_hz)
    if hz <= 0 or not math.isfinite(hz):
        raise ValueError(f"frequency in Hz must be finite and positive (got {hz})")
    mel = 2595.0 * math.log10(1.0 + hz / 700.0)
    bark = 6.0 * math.asinh(hz / 600.0)
    erb = 21.4 * math.log10(1.0 + 0.00437 * hz)
    period_ms = 1000.0 / hz
    wavelength_m = SPEED_OF_SOUND_M_S / hz
    cents = 1200.0 * math.log2(hz / float(ref_hz))
    samples = float(sr) / hz
    frames = samples / float(hop)
    return {
        "hz": round(hz, 4),
        "mel": round(mel, 4),
        "bark": round(bark, 4),
        "erb": round(erb, 4),
        "period_ms": round(period_ms, 4),
        "wavelength_m": round(wavelength_m, 4),
        "cents": round(cents, 2),
        "samples": round(samples, 3),
        "frames": round(frames, 4),
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
    return {
        "encoding": py_enc,
        "char_count": len(text),
        "byte_count": len(raw),
        "lossless": lossless,
        "decoded": decoded,
        "hex": " ".join(hex_bytes),
        "bytes": list(raw),
        "codepoints": codepoints,
    }
