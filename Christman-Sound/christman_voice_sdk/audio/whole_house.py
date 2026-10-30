# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_voice_sdk.audio.whole_house — OpenSmell 21-Profile Classifier, Skin
Haptic Posture Classifier, House Fifth Stream, and TCAP-V-0044 Five-Element
Memory Chamber (ported from everettnc/thewholehouse
src/lib/smell/*, src/lib/skin/*, and src/modules/memory/{model,house}.ts).

No sample content or personal scenes ship in the Memory Chamber; every
MemoryChamber instance starts empty. Unmeasured sensor inputs stay None and
report that they could not measure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "HOUSE_CHANNEL",
    "SMELL_PROFILES",
    "SENSOR_CHANNELS",
    "SMELL_HONESTY_RULE",
    "parse_channels",
    "classify",
    "SKIN_POSTURES",
    "ELEMENTS",
    "RESONANCE_GATE",
    "SCALAR_HAZARD_LOAD",
    "PROFILE_SIMILARITY_FLOOR",
    "HAZARD_CONFIDENCE_FLOOR",
    "RECOIL_MIN_IMU_G",
    "RECOIL_MIN_PAD",
    "TREMOR_MIN_IMU_G",
    "TREMOR_MIN_MEAN_PAD",
    "STROKE_MIN_PAD_VAR",
    "PRESS_MIN_PAD",
    "HOLD_MIN_MEAN_PAD",
    "DEFAULT_SCENES",
    "parse_smell_payload",
    "classify_smell",
    "posture_of",
    "frame_from_sense",
    "merge_house",
    "whole_environment",
    "describe_house",
    "Scene",
    "ShadowHit",
    "CompletionResult",
    "MemoryChamber",
]

HOUSE_CHANNEL = "thewholehouse"

# -----------------------------------------------------------------------------
# THRESHOLDS — EVERY NUMBER BELOW IS A CHOSEN STARTING VALUE, NOT A MEASUREMENT.
# Each is held until client data replaces it, with its reason written beside it:
# -----------------------------------------------------------------------------

#: Skin haptic posture thresholds. CHOSEN STARTING VALUES, NOT MEASUREMENTS —
#: held until client data replaces them. Reasons:
#: - RECOIL_MIN_IMU_G (1.2 g) & RECOIL_MIN_PAD (0.55): sharp jolt above 1 g with
#:   firm contact (>55% pad range) marks sudden withdrawal/impact.
#: - TREMOR_MIN_IMU_G (0.45 g) & TREMOR_MIN_MEAN_PAD (0.12): moderate IMU
#:   oscillation while touching (>12% mean pad pressure) marks shaking/tremor.
#: - STROKE_MIN_PAD_VAR (0.08): spatial variance across the 4 pads while mean
#:   pressure is moderate (0.10..0.65) marks a moving sweep across pads.
#: - PRESS_MIN_PAD (0.72): any pad in the top ~28% of full-scale pressure marks
#:   a deliberate press.
#: - HOLD_MIN_MEAN_PAD (0.15): steady contact above 15% mean pad pressure marks
#:   a resting hand hold; below 0.15 with active pads is light rest.
RECOIL_MIN_IMU_G = 1.2
RECOIL_MIN_PAD = 0.55
TREMOR_MIN_IMU_G = 0.45
TREMOR_MIN_MEAN_PAD = 0.12
STROKE_MIN_PAD_VAR = 0.08
PRESS_MIN_PAD = 0.72
HOLD_MIN_MEAN_PAD = 0.15

#: Memory Chamber completion gate (1.15). CHOSEN STARTING VALUE, NOT A
#: MEASUREMENT — held until client data replaces it. Reason: carried from
#: thewholehouse src/modules/memory/model.ts (RESONANCE_GATE = 1.15); requires
#: at least two sensory doors (2 * 0.65 = 1.30 >= 1.15) to open before a scene
#: completes.
RESONANCE_GATE = 1.15

# -----------------------------------------------------------------------------
# OpenSmell, the nose. WHAT CHANGED AND WHY (2026-10-30): this section held 21
# household smells (coffee, citrus, pine, bleach, garlic...) each with four
# numbers that appear nowhere in OpenSmell or THEWHOLEHOUSE, matched by
# distance after padding missing channels with zeros, and a single MQ-135
# reading could raise a hazard at 0.75. None of that was OpenSmell. It is now a
# true port of THEWHOLEHOUSE src/lib/smell/{types,profiles,parse,classify}.ts:
# OpenSmell's 21 live profiles (source: EverettNC/OpenSmell
# opensmell_engine_truth.json), each read from named sensor channels, scored
# coverage x mean intensity x specificity, alert at 0.7, and a single scalar
# reading never becomes a profile.
# -----------------------------------------------------------------------------

#: types.ts SMELL_HONESTY_RULE, word for word.
SMELL_HONESTY_RULE = (
    "OpenSmell is one MQ-135 voltage, or a labeled file. Channel names are proxy labels, not GC-MS. "
    "Not a diagnosis. Sample air is not the house."
)

#: types.ts SENSOR_CHANNELS, in its order.
SENSOR_CHANNELS: Tuple[str, ...] = (
    "acetone", "isoprene", "ammonia", "benzene", "alkanes", "aldehydes", "hydrocarbons",
    "dimethyl_sulfide", "sulfur", "aliphatic_acids", "skatole", "ketones", "sebum_vocs",
    "lipid_oxidation", "ethanol_trace", "toluene", "ethane", "propanol", "butane", "methane_trace",
)

#: types.ts, carried as written there.
DETECT_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2
ALERT_THRESHOLD = 0.7
CLASSIFY_TOP_N = 3

#: profiles.ts LIVE_PROFILES: 21 profiles. The catalog of 26 is not used here.
SMELL_PROFILES: Tuple[Dict[str, Any], ...] = (
    {"key": "alzheimers", "condition": "Alzheimer's Disease", "category": "neurological", "severity": "high", "channels": ("lipid_oxidation",)},
    {"key": "bladder_cancer", "condition": "Bladder Cancer", "category": "cancer", "severity": "critical", "channels": ("alkanes", "benzene")},
    {"key": "breast_cancer", "condition": "Breast Cancer", "category": "cancer", "severity": "critical", "channels": ("aliphatic_acids", "hydrocarbons")},
    {"key": "c_diff", "condition": "C. difficile", "category": "infectious", "severity": "high", "channels": ("aliphatic_acids", "propanol", "skatole")},
    {"key": "colorectal_cancer", "condition": "Colorectal Cancer", "category": "cancer", "severity": "critical", "channels": ("ammonia", "sulfur", "skatole")},
    {"key": "covid19", "condition": "COVID-19", "category": "infectious", "severity": "high", "channels": ("isoprene", "aldehydes")},
    {"key": "depressive_spiral", "condition": "Depressive Spiral", "category": "psychiatric", "severity": "high", "channels": ("dimethyl_sulfide", "acetone")},
    {"key": "diabetes_type1", "condition": "Diabetes (Type 1)", "category": "metabolic", "severity": "high", "channels": ("acetone",)},
    {"key": "diabetes_type2", "condition": "Diabetes (Type 2)", "category": "metabolic", "severity": "high", "channels": ("acetone",)},
    {"key": "fight_or_flight", "condition": "Fight-or-Flight Escalation", "category": "psychiatric", "severity": "critical", "channels": ("isoprene", "ammonia")},
    {"key": "ketoacidosis", "condition": "Ketoacidosis", "category": "metabolic", "severity": "critical", "channels": ("acetone", "propanol")},
    {"key": "liver_disease", "condition": "Liver Disease", "category": "metabolic", "severity": "high", "channels": ("dimethyl_sulfide",)},
    {"key": "lung_cancer", "condition": "Lung Cancer", "category": "cancer", "severity": "critical", "channels": ("alkanes", "benzene", "aldehydes")},
    {"key": "ovarian_cancer", "condition": "Ovarian Cancer", "category": "cancer", "severity": "critical", "channels": ("aldehydes", "hydrocarbons")},
    {"key": "parkinsons", "condition": "Parkinson's Disease", "category": "neurological", "severity": "high", "channels": ("sebum_vocs", "aldehydes")},
    {"key": "pre_seizure", "condition": "Pre-Seizure / Fit Warning", "category": "psychiatric", "severity": "critical", "channels": ("ammonia", "alkanes")},
    {"key": "prostate_cancer", "condition": "Prostate Cancer", "category": "cancer", "severity": "critical", "channels": ("aldehydes", "ketones")},
    {"key": "rage_cortisol", "condition": "Rage / Cortisol Spike", "category": "psychiatric", "severity": "high", "channels": ("acetone", "isoprene")},
    {"key": "renal_failure", "condition": "Renal Failure", "category": "metabolic", "severity": "critical", "channels": ("ammonia", "dimethyl_sulfide", "toluene")},
    {"key": "sepsis", "condition": "Sepsis", "category": "infectious", "severity": "critical", "channels": ("aliphatic_acids", "ammonia", "dimethyl_sulfide")},
    {"key": "tuberculosis", "condition": "Tuberculosis", "category": "infectious", "severity": "critical", "channels": ("alkanes",)},
)
LIVE_PROFILE_COUNT = 21
assert len(SMELL_PROFILES) == LIVE_PROFILE_COUNT

_MAX_SIG = math.sqrt(max(1, *(len(p["channels"]) for p in SMELL_PROFILES)))


def _num(value: Any) -> Optional[float]:
    """parse.ts num(): a finite number, or a string that is one; else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            n = float(value)
        except ValueError:
            return None
        return n if math.isfinite(n) else None
    return None


def _clamp01(n: float) -> float:
    return max(0.0, min(1.0, n))


def _analog_to_intensity(n: float) -> float:
    """parse.ts analogToIntensity(): a 10-bit or 12-bit ADC count, or 0-1."""
    if 1 < n <= 1023:
        return _clamp01(n / 1023)
    if 1 < n <= 4095:
        return _clamp01(n / 4095)
    return _clamp01(n)


def parse_channels(raw: Any) -> Optional[Dict[str, float]]:
    """parse.ts parseChannels(): only the named channels, each as intensity; None when there are none."""
    if not isinstance(raw, Mapping):
        return None
    out: Dict[str, float] = {}
    for key in SENSOR_CHANNELS:
        v = _num(raw.get(key))
        if v is None:
            continue
        out[key] = _analog_to_intensity(v)
    return out or None


def parse_smell_payload(raw: Any, fallback_kind: str = "serial") -> Optional[Dict[str, Any]]:
    """parse.ts parseSmellPayload(): a reading, or None when there is nothing to read."""
    if not isinstance(raw, Mapping):
        n = _num(raw) if not isinstance(raw, str) else None
        if n is None:
            return None
        return {"kind": fallback_kind, "claim": "record" if fallback_kind == "sample" else "living",
                "intensity": _analog_to_intensity(n), "baseline": None, "channels": None, "at": time.time()}
    kind = raw.get("kind") if raw.get("kind") in ("serial", "file", "sample") else fallback_kind
    claim = raw.get("claim") if raw.get("claim") in ("simulation", "record", "living") else (
        "record" if kind in ("sample", "file") else "living")
    channels = parse_channels(raw.get("channels") if raw.get("channels") is not None else raw)
    intensity_raw = _num(raw.get("intensity"))
    if intensity_raw is None:
        intensity_raw = _num(raw.get("aout"))
    if intensity_raw is None:
        intensity_raw = _num(raw.get("value"))
    from_channels = (sum(channels.values()) / max(1, len(channels))) if channels else None
    if intensity_raw is None and not channels:
        return None
    base = intensity_raw if intensity_raw is not None else from_channels
    return {
        "kind": kind,
        "claim": claim,
        "intensity": _analog_to_intensity(base if base is not None else 0.0),
        "baseline": _num(raw.get("baseline")),
        "channels": channels,
        "at": time.time(),
        "note": raw.get("note") if isinstance(raw.get("note"), str) else None,
    }


def classify(
    channels: Optional[Mapping[str, float]],
    detect_threshold: float = DETECT_THRESHOLD,
    min_confidence: float = MIN_CONFIDENCE,
    top_n: int = CLASSIFY_TOP_N,
) -> List[Dict[str, Any]]:
    """classify.ts classify(): coverage x mean intensity x specificity. A single scalar does not become twenty species."""
    if not channels:
        return []
    detected: Dict[str, float] = {}
    for key in SENSOR_CHANNELS:
        raw = channels.get(key)
        if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            continue
        v = _clamp01(float(raw))
        if v >= detect_threshold:
            detected[key] = v
    if not detected:
        return []
    hits: List[Dict[str, Any]] = []
    for profile in SMELL_PROFILES:
        matched = [ch for ch in profile["channels"] if ch in detected]
        if not matched:
            continue
        coverage = len(matched) / len(profile["channels"])
        mean_intensity = sum(detected[ch] for ch in matched) / len(matched)
        specificity = math.sqrt(len(matched)) / _MAX_SIG
        confidence = min(1.0, coverage * mean_intensity * specificity)
        if confidence < min_confidence:
            continue
        hits.append({
            "profileKey": profile["key"],
            "condition": profile["condition"],
            "category": profile["category"],
            "severity": profile["severity"],
            "matchedChannels": matched,
            "coverage": coverage,
            "meanIntensity": mean_intensity,
            "specificity": specificity,
            "confidence": confidence,
            "alert": confidence >= ALERT_THRESHOLD,
        })
    hits.sort(key=lambda h: h["confidence"], reverse=True)
    return hits[:top_n]


def classify_smell(raw: Any, fallback_kind: str = "serial") -> Dict[str, Any]:
    """
    A take (types.ts SmellTake): the reading, its hits, whether it was a
    scalar only, and the honesty block. With nothing to read it says so.
    """
    reading = parse_smell_payload(raw, fallback_kind)
    if reading is None:
        return {"available": False, "reading": None, "hits": [], "scalarOnly": None,
                "reason": "could not measure: no smell reading received"}
    scalar_only = not reading["channels"]
    return {
        "available": True,
        "reading": reading,
        "hits": classify(reading["channels"]),
        "scalarOnly": scalar_only,
        "honesty": {
            "nose": reading["kind"],
            "claim": reading["claim"],
            "channels": "scalar" if scalar_only else "proxy-labels",
            "clinical": False,
            "rule": SMELL_HONESTY_RULE,
        },
    }


SKIN_POSTURES: Tuple[str, ...] = ("rest", "hold", "press", "tremor", "recoil", "stroke")


def posture_of(
    pads: Optional[Sequence[float]],
    temp_c: Optional[float] = None,
    imu_delta_g: Optional[float] = None,
    pad_variance: Optional[float] = None,
) -> Optional[str]:
    """Classify skin contact posture from measured pad pressures and IMU delta.
    If no pad measurements are provided, returns None rather than inventing 'rest'."""
    if not pads:
        return None
    vals = [max(0.0, min(1.0, float(p))) for p in pads]
    mean_p = sum(vals) / len(vals)
    max_p = max(vals)
    imu = float(imu_delta_g) if imu_delta_g is not None else 0.0
    var_p = float(pad_variance) if pad_variance is not None else (
        sum((p - mean_p) ** 2 for p in vals) / len(vals)
    )
    if imu >= RECOIL_MIN_IMU_G and max_p >= RECOIL_MIN_PAD:
        return "recoil"
    if imu >= TREMOR_MIN_IMU_G and mean_p >= TREMOR_MIN_MEAN_PAD:
        return "tremor"
    if var_p >= STROKE_MIN_PAD_VAR and 0.10 <= mean_p <= 0.65:
        return "stroke"
    if max_p >= PRESS_MIN_PAD:
        return "press"
    if mean_p >= HOLD_MIN_MEAN_PAD:
        return "hold"
    return "rest"


def frame_from_sense(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build a skin frame from measured sensor values.
    Never fills in [0, 0, 0, 0] when pads are absent; unmeasured fields stay None."""
    if not raw or not isinstance(raw, Mapping) or not raw.get("pads"):
        return {
            "available": False,
            "pads": None,
            "mean_pressure": None,
            "temp_c": None,
            "imu_delta_g": None,
            "posture": None,
            "honesty_note": "could not measure: no skin sensor pads provided",
        }
    pads = [float(v) for v in raw["pads"]]
    temp_c = float(raw["temp_c"]) if raw.get("temp_c") is not None else None
    imu = float(raw["imu_delta_g"]) if raw.get("imu_delta_g") is not None else None
    mean_p = sum(pads) / len(pads)
    var_p = sum((p - mean_p) ** 2 for p in pads) / len(pads)
    posture = posture_of(pads, temp_c=temp_c, imu_delta_g=imu, pad_variance=var_p)
    return {
        "available": True,
        "pads": [round(p, 4) for p in pads],
        "mean_pressure": round(mean_p, 4),
        "temp_c": temp_c,
        "imu_delta_g": round(imu, 4) if imu is not None else None,
        "posture": posture,
        "honesty_note": "measured skin pad frame",
    }


def merge_house(
    ear: Optional[Dict[str, Any]] = None,
    cornea: Optional[Dict[str, Any]] = None,
    smell: Optional[Dict[str, Any]] = None,
    skin: Optional[Dict[str, Any]] = None,
    porch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    active_organs = [
        name
        for name, val in (("ear", ear), ("cornea", cornea), ("smell", smell), ("skin", skin), ("porch", porch))
        if val is not None
    ]
    return {
        "channel": HOUSE_CHANNEL,
        "active_organs": active_organs,
        "organ_count": len(active_organs),
        "ear": ear,
        "cornea": cornea,
        "smell": smell,
        "skin": skin,
        "porch": porch,
        "timestamp": time.time(),
    }


def whole_environment(house_state: Mapping[str, Any]) -> Dict[str, str]:
    porch = house_state.get("porch") or {}
    ear = house_state.get("ear") or {}
    smell = house_state.get("smell") or {}
    skin = house_state.get("skin") or {}
    cornea = house_state.get("cornea") or {}

    route = porch.get("route") or []
    who = ", ".join(route) if route else str(ear.get("speaker") or "")
    what = str(porch.get("asSaid") or ear.get("text") or skin.get("posture") or "")
    where_parts = []
    if cornea.get("cell"):
        where_parts.append(f"gaze {cornea['cell']}")
    where = " · ".join(p for p in where_parts if p)
    when = str((cornea.get("stamp") or {}).get("time_of_day") or "")
    feeling = str(skin.get("posture") or ear.get("mode") or "")
    return {
        "who": who,
        "what": what,
        "where": where,
        "when": when,
        "feeling": feeling,
    }


def describe_house(house_state: Mapping[str, Any]) -> str:
    env = whole_environment(house_state)
    parts = [p for p in (env["when"], env["who"], env["what"], env["where"], env["feeling"]) if p]
    return " · ".join(parts)


ELEMENTS: Tuple[str, ...] = ("who", "what", "where", "when", "feeling")


def _tokens(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split() if len(t) >= 2]


@dataclass
class Scene:
    id: str
    title: str
    who: str = ""
    what: str = ""
    where: str = ""
    when: str = ""
    feeling: str = ""
    weight: float = 1.0
    shadow: bool = False
    reclaimed: bool = False
    dream: bool = False

    def facets(self) -> Dict[str, str]:
        return {
            "who": self.who,
            "what": self.what,
            "where": self.where,
            "when": self.when,
            "feeling": self.feeling,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "who": self.who,
            "what": self.what,
            "where": self.where,
            "when": self.when,
            "feeling": self.feeling,
            "weight": round(self.weight, 4),
            "shadow": self.shadow,
            "reclaimed": self.reclaimed,
            "dream": self.dream,
        }


@dataclass(frozen=True)
class ShadowHit:
    scene_id: str
    feeling: str
    resonance: float
    matched_doors: Tuple[str, ...]
    note: str = "shadow somatic marker only — full scene withheld until explicitly reclaimed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "feeling": self.feeling,
            "resonance": round(self.resonance, 4),
            "matched_doors": list(self.matched_doors),
            "note": self.note,
        }


@dataclass(frozen=True)
class CompletionResult:
    completed: Optional[Scene]
    resonance: float
    matched_doors: Tuple[str, ...]
    matched_tokens: Tuple[str, ...]
    shadows: Tuple[ShadowHit, ...] = ()
    gate_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed": self.completed.to_dict() if self.completed else None,
            "resonance": round(self.resonance, 4),
            "matched_doors": list(self.matched_doors),
            "matched_tokens": list(self.matched_tokens),
            "shadows": [s.to_dict() for s in self.shadows],
            "gate_reason": self.gate_reason,
        }


# Every MemoryChamber starts empty — no sample content or personal scenes.
DEFAULT_SCENES: Tuple[Scene, ...] = ()


class MemoryChamber:
    def __init__(self, scenes: Optional[Sequence[Scene]] = None) -> None:
        self.scenes: List[Scene] = [
            Scene(**vars(s)) for s in (scenes if scenes is not None else DEFAULT_SCENES)
        ]
        self.accumulators: Dict[str, float] = {s.id: 0.0 for s in self.scenes}

    def add_scene(self, scene: Scene) -> Scene:
        self.scenes.append(scene)
        self.accumulators.setdefault(scene.id, 0.0)
        return scene

    def _token_scene_freq(self) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for s in self.scenes:
            seen = set()
            for val in s.facets().values():
                for tok in _tokens(val):
                    seen.add(tok)
            for tok in seen:
                freq[tok] = freq.get(tok, 0) + 1
        return freq

    def complete(self, query: Mapping[str, str]) -> CompletionResult:
        active_query = {k: str(query.get(k) or "").strip() for k in ELEMENTS if str(query.get(k) or "").strip()}
        if not active_query:
            return CompletionResult(None, 0.0, (), (), (), "no sensory doors opened")

        tok_freq = self._token_scene_freq()
        query_tokens = {tok for val in active_query.values() for tok in _tokens(val)}
        has_unique_token = any(tok_freq.get(t, 0) == 1 for t in query_tokens)

        if len(active_query) < 2 and not has_unique_token:
            return CompletionResult(
                None,
                0.0,
                tuple(active_query.keys()),
                tuple(sorted(query_tokens)),
                (),
                "gateway refused: fewer than 2 doors opened and no unique anchor token",
            )

        best_scene: Optional[Scene] = None
        best_res = 0.0
        best_doors: Tuple[str, ...] = ()
        best_toks: Tuple[str, ...] = ()
        shadows: List[ShadowHit] = []

        for scene in self.scenes:
            facets = scene.facets()
            matched_doors: List[str] = []
            matched_tokens: List[str] = []
            raw_score = 0.0

            for elem, f_val in facets.items():
                f_toks = set(_tokens(f_val))
                overlap = f_toks & query_tokens
                if elem in active_query:
                    q_toks = set(_tokens(active_query[elem]))
                    direct = f_toks & q_toks
                    if direct:
                        matched_doors.append(elem)
                        matched_tokens.extend(direct)
                        raw_score += 0.65 + 0.25 * len(direct)
                        continue
                if overlap:
                    matched_doors.append(elem)
                    matched_tokens.extend(overlap)
                    raw_score += 0.45 + 0.15 * len(overlap)

            resonance = raw_score * scene.weight + 0.20 * self.accumulators.get(scene.id, 0.0)
            if len(matched_doors) >= 2 and resonance >= RESONANCE_GATE:
                if scene.shadow and not scene.reclaimed:
                    shadows.append(
                        ShadowHit(
                            scene_id=scene.id,
                            feeling=scene.feeling,
                            resonance=resonance,
                            matched_doors=tuple(matched_doors),
                        )
                    )
                    continue
                if resonance > best_res:
                    best_res = resonance
                    best_scene = scene
                    best_doors = tuple(matched_doors)
                    best_toks = tuple(sorted(set(matched_tokens)))

        if best_scene is None:
            why = "shadow marker surfaced (scene withheld)" if shadows else f"no scene reached resonance gate {RESONANCE_GATE}"
            return CompletionResult(None, best_res, best_doors, best_toks, tuple(shadows), why)

        return CompletionResult(best_scene, best_res, best_doors, best_toks, tuple(shadows), "completed")

    def reclaim_scene(self, scene_id: str) -> Optional[Scene]:
        for s in self.scenes:
            if s.id == scene_id:
                s.reclaimed = True
                return s
        return None

    def pulse_accumulators(self, query: Mapping[str, str], amount: float = 0.35) -> Dict[str, float]:
        q_toks = {tok for val in query.values() for tok in _tokens(str(val))}
        for s in self.scenes:
            s_toks = {tok for val in s.facets().values() for tok in _tokens(val)}
            if q_toks & s_toks:
                self.accumulators[s.id] = min(2.0, self.accumulators.get(s.id, 0.0) + float(amount))
        return dict(self.accumulators)

    def decay_accumulators(self, factor: float = 0.85) -> Dict[str, float]:
        for k in list(self.accumulators):
            self.accumulators[k] = round(self.accumulators[k] * float(factor), 4)
        return dict(self.accumulators)

    def apply_night(self, level: int = 1) -> Dict[str, Any]:
        lvl = max(1, min(3, int(level)))
        before = len(self.scenes)
        for s in self.scenes:
            acc = self.accumulators.get(s.id, 0.0)
            if lvl == 1:
                s.weight = s.weight * 0.94 + 0.15 * acc
            elif lvl == 2:
                s.weight = s.weight * 0.88 + 0.22 * acc
            else:
                s.weight = max(0.50, min(1.60, s.weight * 0.90 + 0.18 * acc))
            self.accumulators[s.id] = 0.0
        if lvl >= 2:
            self.scenes = [s for s in self.scenes if s.weight >= 0.30 or s.shadow]
        return {
            "level": lvl,
            "scenes_before": before,
            "scenes_after": len(self.scenes),
            "weights": {s.id: round(s.weight, 4) for s in self.scenes},
        }

    def apply_rem(self) -> Optional[Scene]:
        eligible = sorted((s for s in self.scenes if not s.shadow or s.reclaimed), key=lambda s: s.weight, reverse=True)
        if len(eligible) < 2:
            return None
        a, b = eligible[0], eligible[1]
        dream_id = f"rem-{a.id[:10]}-{b.id[:10]}"
        for existing in self.scenes:
            if existing.id == dream_id:
                return existing
        dream = Scene(
            id=dream_id,
            title=f"REM Bridge: {a.title} × {b.title}",
            who=f"{a.who}, {b.who}",
            what=f"{a.what} woven with {b.what}",
            where=a.where,
            when=b.when,
            feeling=f"{a.feeling} / {b.feeling}",
            weight=round(0.5 * (a.weight + b.weight), 3),
            dream=True,
        )
        self.add_scene(dream)
        return dream
