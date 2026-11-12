# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""python3 -m christman_voice_sdk.audio.filament: the Filament ear server.

WHAT CHANGED AND WHY (2026-11-05)
New, with the server's move into the package. The guard matters: the worker is
started with multiprocessing's spawn, which imports this file again in the
worker under another name, and the worker must not open a second door.
"""

from christman_voice_sdk.audio.filament.server import main

if __name__ == "__main__":
    main()
