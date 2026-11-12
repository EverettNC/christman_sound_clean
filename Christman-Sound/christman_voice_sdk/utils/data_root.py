"""
Where the SDK keeps what it writes for one person: one rule, in one place.

    $CHRISTMAN_AI_HOME, else ~/.christman_ai

WHAT CHANGED AND WHY (2026-11-05)
---------------------------------
The rule was written in christman_voice_sdk.audio.config (sdk_data_root) and
followed by the voicepacks, the music memory and the Voice Creation Center.
Four places did not follow it and wrote into ~/.christman_ai whatever the
setting said: the log files (utils/logger.py), the voice profiles
(integration/voice_capture_client.py), the voice cache
(synthesis/voice_synthesis.py, voice_creation_center/app.py) and the audio cache
(audio/enhanced_speech_recognition.py). On Everett's Mac a test run that had
named another folder still added lines to his own logs. The rule now lives
here, with no dependency of its own, so the logger (which every module
imports) can follow it too; audio.config still offers sdk_data_root under the
same name. The folder is read when it is asked for, not when a module is
imported, so a setting made before the first write is the one used.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The setting that names the SDK's data folder.
DATA_ROOT_ENV = "CHRISTMAN_AI_HOME"


def sdk_data_root() -> Path:
    """The SDK's data folder: $CHRISTMAN_AI_HOME, else ~/.christman_ai."""
    raw = os.environ.get(DATA_ROOT_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".christman_ai"


__all__ = ["DATA_ROOT_ENV", "sdk_data_root"]
