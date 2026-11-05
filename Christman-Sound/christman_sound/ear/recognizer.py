"""Where the ear's model is, and the recognizer it runs: Vosk.

    discover_vosk_model()   the model folder on disk ($VOSK_MODEL_PATH, then
                            the full vosk-model-en-us-0.22 in ~/Downloads or
                            ~/models, then the lgraph one)
    find_graph(model_dir)   the folder holding the static HCLG.fst graph
    Model, Recognizer       Vosk's model and recognizer classes, read when asked
                            for, so finding a model needs no recognizer library

WHAT CHANGED AND WHY (2026-11-05)
From 2026-10-29 to 2026-11-05 this file was a recognizer of its own (MFCC,
i-vectors, an acoustic network and a graph search in this package's code),
written by a Claude Code session in place of the vosk library it had removed.
Everett: Vosk is the ear. That recognizer is kept, not installed, in
held/claude-ear-2026-10-28/ear/recognizer.py; the model-finding functions below
are unchanged from it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def discover_vosk_model() -> Optional[Path]:
    """
    Locate the local vosk-model-en-us-0.22 (preferred, as it holds both HCLG.fst
    and align_lexicon.txt) or vosk-model-en-us-0.22-lgraph on disk under
    $VOSK_MODEL_PATH, ~/Downloads, or ~/models.
    """
    env = (os.environ.get("VOSK_MODEL_PATH") or "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    home = Path.home()
    candidates = [
        home / "Downloads" / "vosk-model-en-us-0.22",
        home / "models" / "vosk-model-en-us-0.22",
        home / "Downloads" / "vosk-model-en-us-0.22-lgraph",
        home / "models" / "vosk-model-en-us-0.22-lgraph",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def find_graph(model_dir: Path) -> Path:
    """The static graph: EAR_GRAPH, the model's own graph/ if it holds HCLG.fst,
    or, for a folder named <name>-lgraph (which keeps a dynamic graph, HCLr.fst
    and Gr.fst), the graph/ of the folder <name> beside it or in ~/Downloads or
    ~/models, which holds the same language model compiled as HCLG.fst."""
    env = (os.environ.get("EAR_GRAPH") or "").strip()
    candidates = [Path(env).expanduser()] if env else []
    candidates.append(model_dir / "graph")
    name = model_dir.name
    if name.endswith("-lgraph"):
        base_name = name[: -len("-lgraph")]
        candidates.append(model_dir.parent / base_name / "graph")
        home = Path.home()
        candidates.append(home / "Downloads" / base_name / "graph")
        candidates.append(home / "models" / base_name / "graph")
    for c in candidates:
        if (c / "HCLG.fst").is_file():
            return c
    raise FileNotFoundError(f"no HCLG.fst for {model_dir} (looked in {', '.join(map(str, candidates))})")


def __getattr__(name: str) -> Any:
    """Model and Recognizer are Vosk's own, imported only when asked for."""
    if name in ("Model", "Recognizer"):
        from .hear import _vosk

        vosk = _vosk()
        return vosk.Model if name == "Model" else vosk.KaldiRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
