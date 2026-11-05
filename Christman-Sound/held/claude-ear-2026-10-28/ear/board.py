# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_sound.ear.board — Family Board, Residual Board, and 15-Region Dialect
Tour Map (ported from THEFILAMENT ear/board.py and ear/dialect-map.js).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "CAP",
    "DIALECT_MAP",
    "DIALECT_TOUR",
    "filament_root",
    "add_family",
    "add_residual",
    "snapshot",
    "region_for_token",
]

CAP = 200

DIALECT_MAP: Tuple[Dict[str, Any], ...] = (
    {
        "key": "kentucky",
        "label": "Kentucky",
        "lang": "en-US",
        "tokens": ["holler", "y'uns", "nigh", "reckon", "poke", "yonder", "ain't", "mash", "cain't"],
    },
    {
        "key": "tennessee",
        "label": "Tennessee",
        "lang": "en-US",
        "tokens": ["holler", "reckon", "yonder", "ain't", "mash", "nigh", "y'uns", "cain't", "bless"],
    },
    {
        "key": "texas",
        "label": "Texas drawl",
        "lang": "en-US",
        "tokens": ["howdy", "y'all", "fixin'", "over yonder", "darlin'", "pardner", "cain't", "might could", "all hat", "ma'am"],
    },
    {
        "key": "louisiana",
        "label": "Louisiana Creole · bayou",
        "lang": "en-US",
        "tokens": ["cher", "sha", "lagniappe", "couyon", "envie", "boudin", "who dat", "yeah you right", "fais-dodo", "pirogue", "bayou", "gumbo", "zydeco", "ova", "dere", "dat", "dis"],
    },
    {
        "key": "florida",
        "label": "Florida cracker · hep · redneck",
        "lang": "en-US",
        "tokens": ["y'all", "ain't", "cain't", "fixin'", "yonder", "reckon", "might could", "palmetto", "gator", "hammock", "swamp", "hep", "cracker", "tain't", "sho nuff", "done told", "boy"],
    },
    {
        "key": "washington",
        "label": "Washington hippies",
        "lang": "en-US",
        "tokens": ["the mountain", "drizzle", "the pass", "spendy", "you bet", "copacetic", "far out", "rainier", "moss", "the island"],
    },
    {
        "key": "montana",
        "label": "Montana",
        "lang": "en-US",
        "tokens": ["big sky", "huckleberry", "the divide", "you bet", "range", "the pass", "butte", "howdy"],
    },
    {
        "key": "utah",
        "label": "Utah",
        "lang": "en-US",
        "tokens": ["the trail", "slickrock", "wash", "the lake", "ward", "pioneer", "mesa", "hoodoo", "desert"],
    },
    {
        "key": "vegas",
        "label": "Vegas · oreo · Chinese · Mandarin",
        "lang": "en-US",
        "tokens": [
            "the strip", "comped", "whale", "high roller", "tap out",
            "bruh", "on god", "no cap", "you feel me", "straight up", "fam",
            "dim sum", "siu mai", "cha siu", "lai see", "m'goi",
            "ni-hao", "xie-xie", "hao", "ma", "dui", "mei-you",
        ],
    },
    {
        "key": "minnesota",
        "label": "Minnesota",
        "lang": "en-US",
        "tokens": ["ope", "you betcha", "dontcha know", "uff-da", "hotdish", "yeah no", "no yeah"],
    },
    {
        "key": "brooklyn",
        "label": "Brooklyn",
        "lang": "en-US",
        "tokens": ["madonn'", "gabagool", "fuhgeddabou", "oobatz", "bodega", "deadass", "yerr", "brick", "son"],
    },
    {
        "key": "bronx",
        "label": "the Bronx",
        "lang": "en-US",
        "tokens": ["deadass", "bodega", "wallahi", "ock", "nahmean", "brick", "son", "fam", "good looks", "yerr"],
    },
    {
        "key": "caribbean",
        "label": "Caribbean US",
        "lang": "en-US",
        "tokens": ["irie", "wagwan", "likkle", "yute", "mon", "ting", "bruh"],
    },
    {
        "key": "hawaiian",
        "label": "Hawaiian Pidgin",
        "lang": "en-US",
        "tokens": ["da kine", "howzit", "pau", "mahalo", "bumbai"],
    },
    {
        "key": "thai",
        "label": "Thai",
        "lang": "th-TH",
        "tokens": ["sawasdee", "khrap", "ka", "nam prik", "pad krapow", "sabai"],
    },
)

DIALECT_TOUR: Tuple[str, ...] = (
    "kentucky", "tennessee", "texas", "louisiana", "florida", "washington", "montana", "utah", "vegas",
)


def filament_root() -> Path:
    return Path(os.environ.get("FILAMENT_HOME") or (Path.home() / ".filament")).expanduser()


def _load(path: Path) -> List[Dict[str, Any]]:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _save(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows[:CAP], indent=2), encoding="utf-8")


def _add(path: Path, token: str, region: str, why: str) -> Optional[Dict[str, Any]]:
    tok = str(token or "").strip()
    if not tok:
        return None
    rows = _load(path)
    key = tok.lower()
    for row in rows:
        if str(row.get("said") or "").lower() == key:
            row["hits"] = int(row.get("hits") or 1) + 1
            row["at"] = time.time()
            row["region"] = region or row.get("region") or region_for_token(tok)
            _save(path, rows)
            return row
    row = {
        "said": tok,
        "keep": tok,
        "region": region or region_for_token(tok),
        "why": why,
        "hits": 1,
        "at": time.time(),
        "organ": "porch",
    }
    rows.insert(0, row)
    _save(path, rows)
    return row


def region_for_token(token: str) -> str:
    low = str(token or "").strip().lower()
    for entry in DIALECT_MAP:
        if low in [t.lower() for t in entry["tokens"]]:
            return str(entry["key"])
    return ""


def add_family(token: str, region: str = "", root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    base = root or filament_root()
    return _add(base / "family.json", token, region, "caught on the fibers. family board.")


def add_residual(token: str, region: str = "", why: str = "slipped", root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    base = root or filament_root()
    return _add(base / "residual.json", token, region, why)


def snapshot(root: Optional[Path] = None) -> Dict[str, Any]:
    base = root or filament_root()
    return {
        "ok": True,
        "organ": "porch",
        "family": _load(base / "family.json"),
        "residual": _load(base / "residual.json"),
        "dialect_regions": list(DIALECT_MAP),
    }
