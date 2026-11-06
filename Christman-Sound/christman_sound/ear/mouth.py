# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_sound.ear.mouth — Filament Mouth Map + Carbon Pile (ported from
THEFILAMENT ear/mouth.py).

Being-neutral: holds no person's or being's name by default. Callers may pass
`extra_mangles` when recovering custom vocabulary.

WHAT CHANGED AND WHY (2026-11-05)
Two failures were swallowed. A Carbon pile that could not be read, or was not
a JSON list, read as an empty pile, and the next word set aside wrote over it;
now a missing pile is empty (nothing set aside yet), and an unreadable one
raises, naming the file, and is left as it is. And a residual-board write that
failed for an unsure word was passed over with `except Exception: pass`; now
it raises. The Filament ear server (christman_sound/filament) runs through
this file and still returns the words heard when the pile fails, with the
failure named in its reply.

WHAT CHANGED AND WHY (2026-11-05, the owner's own words)
On 2026-10-30 the rows that recovered a person's name from the ways the ear
writes it came out of MANGLES, under the ruling that the package names no
one. Nothing was put in their place, so a Filament run from the package wrote
its owner's name the way the ear mishears it. The rows now have a home that is
not the package: a file the owner keeps beside the Carbon pile,
$FILAMENT_HOME/mouth.json (~/.filament/mouth.json by default), a JSON list of
rows, each {"heard": "what the ear writes", "keep": "what was said"}. "heard"
is plain words, not a pattern: it is matched whole, in any letter case, with
any run of spaces between its words. load_own_mangles() reads the file and
the Filament ear server hands the rows to dress(). No file is no rows (the
owner has taught none); a file that cannot be read, or holds anything but
such rows, raises, naming the file and the row, and is left as it is.
MANGLES itself is unchanged and still holds no name.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .board import add_residual

__all__ = [
    "CONF_FLOOR",
    "CARBON_MAX",
    "MANGLES",
    "carbon_path",
    "own_words_path",
    "load_own_mangles",
    "recover",
    "word_conf",
    "shape_words",
    "load_carbon",
    "save_carbon",
    "aside_for_carbon",
    "dress",
]

#: Word lattice confidence floor (0.55) below which a heard word is flagged as
#: unsure and set aside on the Carbon pile rather than trusted silently.
#: CHOSEN STARTING VALUE, NOT A MEASUREMENT — held until client data replaces
#: it. Reason: carried from THEFILAMENT ear/mouth.py (CONF_FLOOR = 0.55).
CONF_FLOOR = 0.55
CARBON_MAX = 80

# 2026-11-04 (Everett): "sumbitch" and "son of a bitch" are two different
# things. The five rows that turned "son of a bitch" (and its run-together
# spellings) into "sumbitch" came out; the rows left recover "sumbitch" from
# the ways the ear mishears it.
MANGLES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsome\s+bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsum\s+bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsombitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsumbich\b", re.I), "sumbitch"),
    (re.compile(r"\bsun\s+bitch\b", re.I), "sumbitch"),
    (re.compile(r"\bsomebeach\b", re.I), "sumbitch"),
)


def carbon_path() -> Path:
    root = Path(os.environ.get("FILAMENT_HOME") or (Path.home() / ".filament")).expanduser()
    return root / "carbon.json"


def own_words_path() -> Path:
    """The owner's own words: mouth.json, beside the Carbon pile."""
    return carbon_path().parent / "mouth.json"


#: One read of the owner's file per change to it: the path, the file's
#: modification time and size, and the rows compiled from it. A live stream
#: posts several times a second and the file changes when its owner edits it.
_OWN_READ: Optional[Tuple[str, int, int, Tuple[Tuple[re.Pattern[str], str], ...]]] = None


def load_own_mangles(path: Optional[Path] = None) -> Tuple[Tuple[re.Pattern[str], str], ...]:
    """The owner's own rows, compiled as MANGLES is, for recover()'s extra_mangles.

    No file is no rows. A file that is not readable JSON, is not a list, or
    has a row that is not {"heard": words, "keep": words} raises ValueError
    naming the file (and the row); the file is left as it is.
    """
    global _OWN_READ
    target = path or own_words_path()
    try:
        stat = target.stat()
    except FileNotFoundError:
        return ()
    key = (str(target), stat.st_mtime_ns, stat.st_size)
    if _OWN_READ is not None and _OWN_READ[:3] == key:
        return _OWN_READ[3]
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"The owner's words at {target} are not readable JSON ({exc}); the file was left as it is.") from exc
    if not isinstance(data, list):
        raise ValueError(f"The owner's words at {target} hold {type(data).__name__}, not a list of rows; "
                         "the file was left as it is.")
    rows: List[Tuple[re.Pattern[str], str]] = []
    for n, row in enumerate(data, 1):
        heard = row.get("heard") if isinstance(row, dict) else None
        keep = row.get("keep") if isinstance(row, dict) else None
        if not (isinstance(heard, str) and heard.split() and isinstance(keep, str) and keep.strip()):
            raise ValueError(f"Row {n} of the owner's words at {target} is {row!r}; a row is "
                             '{"heard": "what the ear writes", "keep": "what was said"}. The file was left as it is.')
        # Whole words only, any case, any run of spaces between them. The
        # look-arounds stand where \b would, and also hold for a word that
        # ends in an apostrophe (fixin'), where \b does not.
        words = r"\s+".join(re.escape(w) for w in heard.split())
        rows.append((re.compile(rf"(?<!\w){words}(?!\w)", re.I), keep.strip()))
    compiled = tuple(rows)
    _OWN_READ = (*key, compiled)
    return compiled


def recover(
    text: str,
    extra_mangles: Sequence[Tuple[re.Pattern[str], str]] = (),
) -> Tuple[str, List[Dict[str, str]]]:
    next_text = text or ""
    hits: List[Dict[str, str]] = []
    for pat, keep in list(MANGLES) + list(extra_mangles):
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


def shape_words(
    payload: Dict[str, Any],
    extra_mangles: Sequence[Tuple[re.Pattern[str], str]] = (),
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for w in payload.get("result") or payload.get("partial_result") or payload.get("words") or []:
        if not isinstance(w, dict):
            continue
        tok = str(w.get("word") or w.get("text") or "").strip()
        if not tok:
            continue
        kept, _hits = recover(tok, extra_mangles=extra_mangles)
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
    """The words set aside for Carbon. No pile yet is empty; a pile that cannot
    be read or is not a JSON list raises, and is left as it is."""
    target = path or carbon_path()
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"The Carbon pile at {target} is not readable JSON ({exc}); it was left as it is.") from exc
    if not isinstance(data, list):
        raise ValueError(f"The Carbon pile at {target} holds {type(data).__name__}, not a list; it was left as it is.")
    return data


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
        add_residual(str(row.get("word") or ""), "", "unsure")
    if added:
        save_carbon(pile, path)
    return added


def dress(
    result: Dict[str, Any],
    payload: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
    extra_mangles: Sequence[Tuple[re.Pattern[str], str]] = (),
) -> Dict[str, Any]:
    raw_text = str(result.get("text") or result.get("partial") or "").strip()
    words = shape_words(payload if payload is not None else result, extra_mangles=extra_mangles)
    kept, hits = recover(raw_text, extra_mangles=extra_mangles)
    for w in words:
        w_kept, _ = recover(str(w.get("text") or ""), extra_mangles=extra_mangles)
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
