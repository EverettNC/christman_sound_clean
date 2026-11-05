# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""christman_sound.ear: the package's own speech to text, fully offline.

Ported from ear-core (CODECWAR5TH/ear-core) with the math unchanged, plus
THEFILAMENT's mouth map, Carbon pile, Family & Residual boards, and
THEWHOLEHOUSE's Porch dialect + TCAP-V-0020 mouth-cooperation layer:

    modelfile   the binary model file format (final.mdl, final.ie, final.dubm, final.mat)
    features    MFCC from conf/mfcc.conf, with dither; online i-vectors from ivector/
    acoustic    the transition model and the network in am/final.mdl, run in chunks
    decoder     the decoding graph HCLG.fst, memory-mapped and searched in C;
                word lattices and MBR confidences
    recognizer  Model, Recognizer, find_graph, discover_vosk_model
    hear        Ear, Heard, Word: the same answers as Python objects
    mouth       Filament mouth map + Carbon pile (CONF_FLOOR = 0.55)
    board       Family & Residual boards + 15-region Dialect Map
    porch       TCAP-V-0020 mouth-cooperation + 33 family names + dialect preservation
    wer         word error rate
    build       compiles csrc/ into libear.dylib / libear.so

No recognizer library is imported; numpy and the C library built from csrc/
are the whole engine. Build the C library once per machine:

    python -m christman_sound.ear.build --check
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "Ear": "hear",
    "Heard": "hear",
    "Word": "hear",
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
