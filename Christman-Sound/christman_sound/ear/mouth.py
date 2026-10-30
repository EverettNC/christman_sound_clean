# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_sound.ear.mouth — Filament Mouth Map + Carbon Pile (ported from
THEFILAMENT ear/mouth.py).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "CONF_FLOOR",
    "CARBON_MAX",
    "MANGLES",
    "carbon_path",
    "recover",
    "word_conf",
    "shape_words",
    "load_carbon",
    "save_carbon",
    "aside_for_carbon",
    "dress",
]

CONF_FLOOR = 0.55
CARBON_MAX = 80

MANGLES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bchristmas\b", re.I), "Christman"),
    (re.compile(r"\bchrist\s*man\b", re.I), "Christman"),
    (re.compile(r"\bchris\s*man\b", re.I), "Christman"),
    (re.compile(r"\bchrisman\b", re.I), "Christman"),
    (re.compile(r"\bchristmann\b", re.I), "Christman"),
    (re.compile(r"\bkristman\b", re.I), "Christman"),
    (re.compile(r"\bkristmann\b", re.I), "Christman"),
    (re.compile(r"\bchris\s*men\b", re.I), "Christman"),
    (re.compile(r"\bson\s+of\s+a\s+bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsonofa\s*bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsonofabitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsome\s+bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsum\s+bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bson\s+a\s+bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bson\s+bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsombitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsumbich\b", re.I), "sumbitch"),
    (re.compile(r"\bsun\s+bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsomebeach\b", re.I), "sumbitch"),
    (re.compile(r"\baverett\b", re.I), "Everett"),
    (re.compile(r"\beverette\b", re.I), "Everett"),
    (re.compile(r"\beveritt\b", re.I), "Everett"),
)


def carbon_path() -> Path:
    root = Path(os.environ.get("FILAMENT_HOME") or (Path.home() / ".filament")).expanduser()
    return root / "carbon.json"


def recover(text: str) -> Tuple[str, List[Dict[str, str]]]:
    next_text = text or ""
    hits: List[Dict[str, str]] = []
    for pat, keep in MANGLES:
        if pat.search(next_text):
            next_text = pat.sub(keep, next_text)
            hits.append({"from": pat.pattern, "to": keep, "why": "mouth map"})
    return next_text, hits


def word_conf(raw: Dict[str, Any]) -> Optional[float]:
    for key in ("conf", "confidence"):
        val = raw.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def shape_words(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for w in payload.get("result") or payload.get("partial_result") or payload.get("words") or []:
        if not isinstance(w, dict):
            continue
        tok = str(w.get("word") or w.get("text") or "").strip()
        if not tok:
            continue
        kept, _hits = recover(tok)
        conf = word_conf(w)
        unsure = conf is not None and conf < CONF_FLOOR
        out.append({
            "text": kept,
            "raw": tok,
            "start": float(w.get("start") or 0.0),
            "end": float(w.get("end") or 0.0),
            "conf": conf,
            "unsure": unsure,
        })
    return out


def load_carbon(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    target = path or carbon_path()
    try:
        if target.is_file():
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_carbon(rows: List[Dict[str, Any]], path: Optional[Path] = None) -> None:
    target = path or carbon_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows[:CARBON_MAX], indent=2), encoding="utf-8")


def aside_for_carbon(words: List[Dict[str, Any]], heard: str, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    pile = load_carbon(path)
    now = time.time()
    added: List[Dict[str, Any]] = []
    for w in words:
        if not w.get("unsure"):
            continue
        row = {
            "word": w.get("raw") or w.get("text"),
            "kept": w.get("text"),
            "conf": w.get("conf"),
            "heard": heard,
            "at": now,
            "for": "Carbon",
        }
        pile.insert(0, row)
        added.append(row)
        try:
            from .board import add_residual
            add_residual(str(row.get("word") or ""), "", "unsure")
        except Exception:
            pass
    if added:
        save_carbon(pile, path)
    return added


def dress(result: Dict[str, Any], payload: Optional[Dict[str, Any]] = None, path: Optional[Path] = None) -> Dict[str, Any]:
    raw_text = str(result.get("text") or result.get("partial") or "").strip()
    words = shape_words(payload if payload is not None else result)
    kept, hits = recover(raw_text)
    for w in words:
        w_kept, _ = recover(str(w.get("text") or ""))
        w["text"] = w_kept
    result["text"] = kept
    if "transcript" in result:
        result["transcript"] = kept
    if result.get("partial") and not result.get("final"):
        result["partial"] = kept
    result["words"] = words
    result["mouth"] = hits
    unsure = [w for w in words if w.get("unsure")]
    result["unsure"] = unsure
    if result.get("final") is True or (result.get("ok") and result.get("final") is None):
        if unsure:
            aside_for_carbon(unsure, kept, path=path)
            result["carbon"] = True
        else:
            result["carbon"] = False
    return result
