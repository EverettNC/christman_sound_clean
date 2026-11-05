# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""christman_sound.ear: the package's speech to text, fully offline.

The words come from Vosk's recognizer on a model folder on disk, the
full vosk-model-en-us-0.22 by default: the recognizer the Filament runs. Corti
sits on top of the hearing (christman_voice_sdk/audio/hearing.py); this is the
word ear under it. With the recognizer come THEFILAMENT's mouth map, Carbon
pile, Family & Residual boards, and THEWHOLEHOUSE's Porch dialect +
TCAP-V-0020 mouth-cooperation layer:

    hear        Ear, Heard, Word, load_model, EarUnavailable: Vosk's answers as Python objects
    recognizer  discover_vosk_model, find_graph; Model and Recognizer are Vosk's own
    mouth       Filament mouth map + Carbon pile (CONF_FLOOR = 0.55)
    board       Family & Residual boards + 15-region Dialect Map
    porch       TCAP-V-0020 mouth-cooperation + 33 family names + dialect preservation

WHAT CHANGED AND WHY (2026-11-05)
Everett: Vosk is the ear, and Corti sits on top. From 2026-10-29 to 2026-11-05
this package decoded words with its own port of the ear-core a Claude Code
session built in CODECWAR5TH (model file reader, MFCC and i-vectors, the
acoustic network, a C graph search), after that session took the vosk library
out (EverettNC/Christman-Sound 7b6e1ce). That ear is kept whole, not installed,
in held/claude-ear-2026-10-28, with its tests, so it can be looked at.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "Ear": "hear",
    "Heard": "hear",
    "Word": "hear",
    "EarUnavailable": "hear",
    "load_model": "hear",
    "Model": "recognizer",
    "Recognizer": "recognizer",
    "find_graph": "recognizer",
    "discover_vosk_model": "recognizer",
    "CONF_FLOOR": "mouth",
    "recover": "mouth",
    "dress": "mouth",
    "load_carbon": "mouth",
    "save_carbon": "mouth",
    "add_family": "board",
    "add_residual": "board",
    "snapshot": "board",
    "DIALECT_MAP": "board",
    "FAMILY_NAMES": "porch",
    "DIALECT_KEEP": "porch",
    "MOUTH_COOPERATION": "porch",
    "recover_mangles": "porch",
    "reconstruct_dialect": "porch",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        mod = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
