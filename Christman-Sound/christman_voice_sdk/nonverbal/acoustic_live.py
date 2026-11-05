# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Live acoustic key from the Corti client into the caller.

This module does NOT measure F0, attack, or porosity. That measurement
already happened on the metal in the Corti client. What arrives
here is a KEY. This file looks the key up. It does not re-guess it.

grunt_distress is not guessed here. The client measured elevation, attack,
and porosity. This module reports the mapped reading only when that key is
in the map; an unknown key is reported unrecognized, with no words.

WHAT CHANGED AND WHY (2026-11-04). Everett's ruling, in his words:

    "we can have this knowledge in the sound package, but it can't be
    making those decisions. It's up to the being to make those decisions,
    not something that just creates a voice."
    "it can't make decisions that counteract the being, because the being
    is the one that has all of the work to make those decisions properly."

The reading returned here is nonverbal_engine's Interpretation, which no
longer carries a flag telling the caller to page anyone or to stay silent.
It carries the client's mapped words, `category`, `danger` and
`danger_term`; the being decides what to do with them. This docstring said
the module "speaks" a sentence; it never did, and now says what it does.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .nonverbal_engine import (
    Interpretation,
    Modality,
    NonverbalEngine,
    Status,
    get_nonverbal_engine,
)

LIVE_KEYS = frozenset(
    {
        "grunt_distress",
        "grunt_acknowledge",
        "grunt_frustration",
        "vocal_stimming",
    }
)


def deliver_acoustic_key(
    key: str,
    engine: Optional[NonverbalEngine] = None,
) -> Interpretation:
    """
    Look up one live key from Corti and return the reading. An unknown,
    empty or foreign key is reported UNRECOGNIZED with no words and no
    category; nothing is guessed for it.
    """
    name = (key or "").strip()
    if name not in LIVE_KEYS:
        return Interpretation(
            modality=Modality.SOUND,
            status=Status.UNRECOGNIZED,
            input_data=name or None,
            reason="no live acoustic key — Corti did not measure a named event",
            message=None,
        )

    nv = engine or get_nonverbal_engine()
    return nv.process_sound(name)


def deliver_acoustic_key_dict(key: str) -> Dict[str, Any]:
    return deliver_acoustic_key(key).to_dict()
