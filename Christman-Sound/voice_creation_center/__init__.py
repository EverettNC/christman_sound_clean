# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The Christman Voice Creation Center, inside Christman-Sound.

Everett, 2026-11-05: "I'm still not seeing the voice creation center in
there either." It came in from EverettNC/Voice_Creation_Center at 5940b21,
with its history (git subtree). Its parts:

    voice_creation_center.Voice_registry            the voice pack inventory
    voice_creation_center.voice_express              the express lane: pre-rendered phrases
    voice_creation_center.voice_ingestor             the front door: audio in, phrases out
    voice_creation_center.vcc_paths                  the one base directory for all of it
    voice_creation_center.engines.voice_engine       speaks through christman_sound.synth
    voice_creation_center.engines.voice_validator    the content check before any sound
    voice_creation_center.engines.voice_path_registrar   builds the express index from folders
    voice_creation_center.engines.voice_loader / voice_logger / voice_cache_manager
    voice_creation_center.studio.live_studio         record, listen, validate, approve, export
    voice_creation_center.studio.recorder            the microphone (needs sounddevice)
    voice_creation_center.studio.take_listener       listens to a take

It names no being and no person: each being brings its own voice, phrases
and folder. Installed with the package (pip install -e .). The ingestor and
the path registrar run as

    python3 -m voice_creation_center.voice_ingestor --help
    python3 -m voice_creation_center.engines.voice_path_registrar --help

Where it writes: packs/, inventory/, express/, incoming/, logs/ and the
studio's raw takes go under one base directory (vcc_paths): the base_dir a
caller passes, else $VCC_HOME, else this folder. Installed editable, this
folder is the checkout, as it was in the VCC's own repo.

Christman-Sound also holds christman_sound/voice_creation_center.py, which is
a different thing with the same name: the measuring and voice-building engine
(measure a voice profile, build a frame voice bank, package a voicepack).
"""
