"""
speech_response.py — speak a being's reply.

WHAT CHANGED AND WHY (2026-10-17) — OUTSIDE VOICE CUT
-----------------------------------------------------
Everett's ruling: the rebuild cuts every outside speech engine. This
file was nothing but a wrapper around pyttsx3, the operating system's speech
engine. The original is kept whole at
quarantine/2026-10_outside_voices/christman_voice_sdk/synthesis/speech_response.py.
Removed (quoted from main 99a6cce):

    import pyttsx3
    ...
    eng = pyttsx3.init(driverName="nsss") if sys.platform == "darwin" else pyttsx3.init()
    eng.setProperty("rate", 180)
    eng.setProperty("volume", 1.0)
    ...
    engine.say(text)
    engine.runAndWait()

`speak()` keeps its name so callers still import, and raises
`SynthesisUnavailable(SOVEREIGN_NOT_WIRED)` (this path is retired; the voice is voice_synthesis). It never
speaks in a voice that is not the being's own, and it never returns as if it
had spoken.

WHAT CHANGED AND WHY (2026-11-04) — THE BEING DECIDES
-----------------------------------------------------
Everett, 2026-11-04: "we can have this knowledge in the sound package, but it
can't be making those decisions. It's up to the being to make those
decisions, not something that just creates a voice."

`speak()` ran the reply guard first: a reply carrying "you're not okay" or
"calm down", or (with silent_alarm=True) a reply announcing help, raised
ForbiddenReply instead of SynthesisUnavailable. The guard call and the
silent_alarm parameter are gone; a caller that still passes silent_alarm
gets a TypeError. The lists are still here to read:
reply_guard.forbidden_lines(reply, danger=...) names the lines, and the
being decides what to say.

WHAT CHANGED AND WHY (2026-11-12) — IT SPEAKS, THROUGH THE SDK'S OWN MOUTH
---------------------------------------------------------------------------
Everett: every module in the SDK is used. `speak()` raised and voiced
nothing since pyttsx3 came out. It now speaks the reply through
voice_synthesis (Everett's own synthesizer, in C) in the voice the caller
passes. There is no default voice: with none it raises SynthesisUnavailable
and says so. `tone_profile` reaches the mouth as its named controls; a key
that is not one is refused by the mouth with the reason, never dropped.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..engines.synthesis_result import SynthesisUnavailable

logger = logging.getLogger(__name__)


def speak(text: str, tone_profile: Optional[dict] = None, *, voice: Any = None,
          lexicon_path: Optional[str] = None, play_audio: bool = True,
          output_path: Optional[str] = None) -> Optional[str]:
    """
    Speak `text` in the caller's `voice` (a frame voice folder, a voice
    profile file, or one already loaded) and return the WAV's path. With no
    voice, raises `SynthesisUnavailable` with the reason; nothing is voiced.
    """
    if voice is None:
        err = SynthesisUnavailable(
            "no voice given: pass the caller's own voice (voice=<frame voice folder or voice profile file>, or a "
            "loaded FrameVoice or VoiceProfile); there is no default voice", where="speech_response.speak")
        logger.error("Did not speak %r: %s", (text or "")[:40], err)
        raise err
    from .voice_synthesis import SpeechSynthesisEngine
    mouth = SpeechSynthesisEngine(lexicon_path=lexicon_path)
    return mouth.generate_speech_audio(text, emotion_params=tone_profile or None, voice=voice,
                                       play_audio=play_audio, output_path=output_path)


__all__ = ["speak"]
