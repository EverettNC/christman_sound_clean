# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""christman_voice_sdk.audio.filament: the Filament ear server, offline Vosk on 127.0.0.1.

    christman-filament                  # the console script (pyproject.toml)
    python3 -m christman_voice_sdk.audio.filament

    FILAMENT_EAR_PORT   the door, 4850 by default
    VOSK_MODEL_PATH     the model folder (else the full vosk-model-en-us-0.22
                        in ~/Downloads or ~/models)
    FILAMENT_HOME       the Carbon pile and the boards (~/.filament by default)

    server.py   the HTTP door and the worker the model sits in
    pages/      the Filament's pages and dialect map, shipped as package data;
                pages/filament.html is frozen (pages/FROZEN.md)

The words come through christman_voice_sdk.audio.ear (Vosk's recognizer); the mouth map,
Carbon pile and boards are christman_voice_sdk.audio.ear.mouth and .board. Corti sits on
top of the hearing (christman_voice_sdk/audio/hearing.py); this is the word
ear under it.

WHAT CHANGED AND WHY (2026-11-05)
New. Everett's ruling of 2026-10-29: Christman-Sound holds everything he built
for sound, and the Filament ear server (THEFILAMENT ear/vosk_ear.py) moves into
the package. The folder is named for the Filament, the organ it is; the console
script follows the package's christman-* scripts (christman-corti,
christman-word-ear). Nothing is imported here, so `import christman_voice_sdk.audio.filament`
starts nothing and loads no recognizer.
"""
