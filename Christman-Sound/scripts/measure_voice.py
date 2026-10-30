#!/usr/bin/env python3
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The command line for christman_sound.measure_voice.

2026-10-30: the code moved into the package (christman_sound/measure_voice.py) so the
Voice Creation Center and this command run the same measurement, one code
path. Run this exactly as before; see that module for what it does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from christman_sound import measure_voice as _impl  # noqa: E402

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_impl.main())
