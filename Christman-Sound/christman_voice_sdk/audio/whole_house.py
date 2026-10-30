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
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "HOUSE_CHANNEL",
    "SMELL_PROFILES",
    "SKIN_POSTURES",
    "ELEMENTS",
    "RESONANCE_GATE",
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

SMELL_PROFILES: Tuple[Dict[str, Any], ...] = (
    {"id": "clean_air", "label": "Clean Baseline Air", "category": "baseline", "vector": (0.05, 0.05, 0.05, 0.05)},
    {"id": "coffee_roast", "label": "Fresh Coffee (Furan / Pyrazine)", "category": "kitchen", "vector": (0.62, 0.48, 0.18, 0.35)},
    {"id": "citrus_terpene", "label": "Citrus Peel (d-Limonene)", "category": "botanical", "vector": (0.74, 0.22, 0.15, 0.28)},
    {"id": "pine_pinene", "label": "Pine Resin (alpha-Pinene)", "category": "botanical", "vector": (0.68, 0.19, 0.22, 0.31)},
    {"id": "lavender_linalool", "label": "Lavender (Linalool)", "category": "botanical", "vector": (0.58, 0.25, 0.12, 0.24)},
    {"id": "ethanol_vapor", "label": "Ethanol / Sanitizer", "category": "solvent", "vector": (0.88, 0.30, 0.20, 0.15)},
    {"id": "isopropanol", "label": "Isopropyl Alcohol", "category": "solvent", "vector": (0.85, 0.35, 0.25, 0.18)},
    {"id": "acetone_ketone", "label": "Acetone", "category": "solvent", "vector": (0.90, 0.42, 0.28, 0.12)},
    {"id": "ammonia_nh3", "label": "Ammonia (NH3)", "category": "hazard", "vector": (0.45, 0.88, 0.32, 0.20)},
    {"id": "hydrogen_sulfide", "label": "Hydrogen Sulfide (H2S)", "category": "hazard", "vector": (0.38, 0.82, 0.76, 0.25)},
    {"id": "carbon_monoxide", "label": "Carbon Monoxide / Incomplete Combustion", "category": "hazard", "vector": (0.30, 0.40, 0.92, 0.45)},
    {"id": "wood_smoke", "label": "Wood Smoke (Guaiacol / Syringol)", "category": "combustion", "vector": (0.72, 0.55, 0.84, 0.68)},
    {"id": "toast_maillard", "label": "Toasted Bread (Maillard)", "category": "kitchen", "vector": (0.52, 0.44, 0.38, 0.42)},
    {"id": "burnt_plastic", "label": "Overheated Polymer / Insulation", "category": "hazard", "vector": (0.78, 0.66, 0.88, 0.74)},
    {"id": "natural_gas_mercaptan", "label": "Natural Gas + t-Butyl Mercaptan", "category": "hazard", "vector": (0.82, 0.75, 0.65, 0.22)},
    {"id": "petrichor_geosmin", "label": "Rain on Soil (Geosmin)", "category": "ambient", "vector": (0.35, 0.18, 0.14, 0.48)},
    {"id": "ozone_o3", "label": "Ozone (O3)", "category": "ambient", "vector": (0.22, 0.64, 0.48, 0.16)},
    {"id": "vinegar_acetic", "label": "Acetic Acid / Vinegar", "category": "kitchen", "vector": (0.64, 0.52, 0.20, 0.30)},
    {"id": "bleach_hypochlorite", "label": "Sodium Hypochlorite (Bleach)", "category": "cleaning", "vector": (0.28, 0.78, 0.44, 0.18)},
    {"id": "garlic_allicin", "label": "Cooking Garlic (Allicin)", "category": "kitchen", "vector": (0.60, 0.68, 0.30, 0.40)},
    {"id": "damp_mildew", "label": "Damp Mold / Microbial MVOC", "category": "ambient", "vector": (0.42, 0.38, 0.28, 0.72)},
)


def parse_smell_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, (int, float)):
        return {"channels": [float(raw)], "temp_c": None, "humidity_pct": None}
    if isinstance(raw, Mapping):
        if "channels" in raw and isinstance(raw["channels"], Sequence):
            ch = [float(v) for v in raw["channels"]]
        elif "mq135" in raw:
            ch = [float(raw["mq135"])]
        else:
            ch = [float(raw[k]) for k in ("voc", "nh3", "co", "moisture") if k in raw and raw[k] is not None]
        return {
            "channels": ch,
            "temp_c": float(raw["temp_c"]) if raw.get("temp_c") is not None else None,
            "humidity_pct": float(raw["humidity_pct"]) if raw.get("humidity_pct") is not None else None,
        }
    return {"channels": [], "temp_c": None, "humidity_pct": None}


def classify_smell(raw: Any) -> Dict[str, Any]:
    parsed = parse_smell_payload(raw)
    ch = parsed["channels"]
    if not ch:
        return {
            "available": False,
            "channel_count": 0,
            "air_load": None,
            "species": [],
            "hazard": False,
            "honesty_note": "no sensor channels received",
        }
    if len(ch) == 1:
        load = max(0.0, min(1.0, float(ch[0])))
        return {
            "available": True,
            "channel_count": 1,
            "air_load": round(load, 4),
            "species": [],
            "hazard": load >= 0.75,
            "honesty_note": "1 scalar channel (MQ-135) measures nonspecific gas load only; cannot separate chemical species.",
        }

    vec = tuple(float(ch[i]) if i < len(ch) else 0.0 for i in range(4))
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for prof in SMELL_PROFILES:
        pv = prof["vector"]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, pv)) / 4.0)
        sim = max(0.0, 1.0 - dist)
        scored.append((sim, prof))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = [
        {
            "id": p["id"],
            "label": p["label"],
            "category": p["category"],
            "confidence": round(sim, 3),
        }
        for sim, p in scored[:3]
        if sim >= 0.55
    ]
    load = max(0.0, min(1.0, sum(vec) / len(vec)))
    hazard = any(item["category"] == "hazard" and item["confidence"] >= 0.70 for item in top)
    return {
        "available": True,
        "channel_count": len(ch),
        "air_load": round(load, 4),
        "species": top,
        "hazard": hazard,
        "temp_c": parsed["temp_c"],
        "humidity_pct": parsed["humidity_pct"],
        "honesty_note": "multi-channel MOS profile match",
    }


SKIN_POSTURES: Tuple[str, ...] = ("rest", "hold", "press", "tremor", "recoil", "stroke")


def posture_of(
    pads: Sequence[float],
    temp_c: Optional[float] = None,
    imu_delta_g: float = 0.0,
    pad_variance: float = 0.0,
) -> str:
    vals = [max(0.0, min(1.0, float(p))) for p in pads] if pads else [0.0, 0.0, 0.0, 0.0]
    mean_p = sum(vals) / len(vals)
    max_p = max(vals)
    if imu_delta_g >= 1.2 and max_p >= 0.55:
        return "recoil"
    if imu_delta_g >= 0.45 and mean_p >= 0.12:
        return "tremor"
    if pad_variance >= 0.08 and 0.10 <= mean_p <= 0.65:
        return "stroke"
    if max_p >= 0.72:
        return "press"
    if mean_p >= 0.15:
        return "hold"
    return "rest"


def frame_from_sense(raw: Mapping[str, Any]) -> Dict[str, Any]:
    pads = [float(v) for v in (raw.get("pads") or [0.0, 0.0, 0.0, 0.0])]
    temp_c = float(raw["temp_c"]) if raw.get("temp_c") is not None else None
    imu = float(raw.get("imu_delta_g") or 0.0)
    mean_p = sum(pads) / len(pads) if pads else 0.0
    var_p = sum((p - mean_p) ** 2 for p in pads) / len(pads) if pads else 0.0
    posture = posture_of(pads, temp_c=temp_c, imu_delta_g=imu, pad_variance=var_p)
    return {
        "pads": [round(p, 4) for p in pads],
        "mean_pressure": round(mean_p, 4),
        "temp_c": temp_c,
        "imu_delta_g": round(imu, 4),
        "posture": posture,
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
    who = ", ".join(route) if route else str(ear.get("speaker") or "present caller")
    what = str(porch.get("asSaid") or ear.get("text") or skin.get("posture") or "quiet presence")
    species = smell.get("species") or []
    where_parts = []
    if species:
        where_parts.append(str(species[0].get("label", "")))
    if cornea.get("cell"):
        where_parts.append(f"gaze {cornea['cell']}")
    where = " · ".join(p for p in where_parts if p) or "the house"
    when = str((cornea.get("stamp") or {}).get("time_of_day") or "now")
    feeling = str(skin.get("posture") or ear.get("mode") or "steady")
    return {
        "who": who,
        "what": what,
        "where": where,
        "when": when,
        "feeling": feeling,
    }


def describe_house(house_state: Mapping[str, Any]) -> str:
    env = whole_environment(house_state)
    return f"[{env['when']}] {env['who']} — {env['what']} ({env['where']}; feeling: {env['feeling']})"


ELEMENTS: Tuple[str, ...] = ("who", "what", "where", "when", "feeling")
RESONANCE_GATE = 1.15


def _tokens(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split() if len(t) >= 2]


@dataclass
class Scene:
    id: str
    title: str
    who: str
    what: str
    where: str
    when: str
    feeling: str
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


DEFAULT_SCENES: Tuple[Scene, ...] = (
    Scene(
        id="porch-evening-coffee",
        title="Evening Coffee on the Porch",
        who="Everett, Misty",
        what="drinking fresh coffee and talking through the voice architecture",
        where="front porch, fresh coffee roast in the air",
        when="evening",
        feeling="warm, grounded, steady",
        weight=1.35,
    ),
    Scene(
        id="alphavox-first-lock",
        title="AlphaVox First Sovereign Frame Lock",
        who="Everett, AlphaVox, Brockston",
        what="hearing the stitched voice speak from its own measured frames without a robotic table",
        where="studio bench",
        when="late night",
        feeling="electric, proud, clear",
        weight=1.45,
    ),
    Scene(
        id="hillsboro-fraud-dispute",
        title="Disputed Credit One / LVNV Collection Call",
        who="Everett, debt collector",
        what="demanding written FDCPA validation for the disputed 2022 Hillsboro account",
        where="phone line, Columbus / Gahanna",
        when="afternoon",
        feeling="guarded, firm, protective",
        weight=1.25,
        shadow=True,
    ),
)


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
