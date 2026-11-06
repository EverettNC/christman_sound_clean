# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""One recording gives the same words at the Filament's door and the word-ear's.

Before 2026-11-05 the two heard a whole tape differently: the Filament's /stt
gave it to one recognizer as one piece with half a second of silence after;
the SDK's engine, which the word-ear receiver sits on, closed the utterance
at every pause and added no silence. On one 234-second recording, on the same
model, they differed on 13 words of about 685. Both now hear a tape through
christman_sound.ear.Ear.hear_tape.

The first checks need no model: they hold Ear.hear_tape to what it does with
the recognizer, and hold both callers to it. The last two hear real speech
through both doors and compare word for word; they need the model folder and
recordings (16 kHz 16-bit mono .s16) named by EAR_GOLDEN, and are NOT RUN
without them.

WHAT CHANGED AND WHY (2026-11-05)
New, with Ear.hear_tape.
"""

from __future__ import annotations

import io
import os
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_sound.ear import hear as hear_mod  # noqa: E402
from christman_sound.ear.hear import Ear, Heard, Word  # noqa: E402

MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22")
TAPES = Path(os.environ.get("EAR_GOLDEN") or Path.home() / "claude-work" / "lane8" / "golden" / "lgraph") / "pcm"
#: The longest tape is heard too when this is 1 (about a minute of hearing).
LONG_ENV = "EAR_LINE_UP_LONG"


class _Recognizer:
    """Stands where Vosk's recognizer does and writes down what it is given."""

    def __init__(self, endpoint_at=()):
        self.bites, self.results, self.finals = [], 0, 0
        self._endpoint_at = set(endpoint_at)

    def AcceptWaveform(self, data):  # noqa: N802 - Vosk's name
        self.bites.append(bytes(data))
        return len(self.bites) in self._endpoint_at

    def Result(self):  # noqa: N802
        self.results += 1
        return '{"text": "closed early"}'

    def FinalResult(self):  # noqa: N802
        self.finals += 1
        return '{"text": "all of it", "result": [{"word": "all", "start": 0.1, "end": 0.3, "conf": 0.9}]}'


def _ear_on(rec, rate=16000) -> Ear:
    ear = Ear.__new__(Ear)   # no model here: only what hear_tape does with the recognizer is checked
    ear.sample_rate = rate
    ear._rec = rec
    return ear


def test_hear_tape_gives_every_bite_to_one_recognizer_and_closes_nothing_at_a_pause():
    rec = _Recognizer(endpoint_at={2, 3})        # the recognizer says "endpoint" twice on the way
    pcm = bytes(range(256)) * 50                 # 12800 bytes
    heard = _ear_on(rec).hear_tape(pcm)
    assert heard == Heard("all of it", (Word("all", 0.1, 0.3, 0.9),))
    assert rec.results == 0 and rec.finals == 1, "an utterance was closed at a pause"
    tail = b"\x00" * 16000                       # half a second at 16 kHz, 16-bit
    assert b"".join(rec.bites) == pcm + tail
    assert [len(b) for b in rec.bites] == [4000] * 7 + [800]


def test_the_silence_after_a_tape_is_half_a_second_at_the_ear_rate():
    assert hear_mod.TAPE_TAIL_S == 0.5 and hear_mod.HEAR_BITE == 4000
    rec = _Recognizer()
    _ear_on(rec, rate=8000).hear_tape(b"\x01\x02" * 10)
    assert b"".join(rec.bites) == b"\x01\x02" * 10 + b"\x00" * 8000


def test_hear_still_closes_each_utterance_for_callers_that_want_them_apart():
    rec = _Recognizer(endpoint_at={1})
    out = _ear_on(rec).hear(b"\x00" * 8000)
    assert [h.text for h in out] == ["closed early", "all of it"] and rec.results == 1
    assert b"".join(rec.bites) == b"\x00" * 8000, "hear() adds no silence"


class _TapeEar:
    """Stands where Ear does in the two callers and counts how each one hears."""

    calls: list = []

    def __init__(self, model, sample_rate=16000, words=True):
        self.model = model

    def hear_tape(self, pcm, bite=4000):
        type(self).calls.append(("hear_tape", bytes(pcm), bite))
        return Heard("same words", (Word("same", 0.0, 0.2, 1.0), Word("words", 0.2, 0.5, 0.5)))

    def hear(self, pcm, bite=4000):
        raise AssertionError("a caller closed utterances at pauses (Ear.hear) instead of hearing a tape")

    def accept(self, pcm):
        raise AssertionError("a caller fed the recognizer itself instead of hearing a tape")


def test_the_filament_stt_and_the_sdk_engine_both_hear_through_hear_tape(monkeypatch):
    import christman_sound.ear as ear_pkg
    from christman_sound.filament import server
    from christman_voice_sdk.audio import speech_recognition_engine as sre

    pcm = b"\x10\x00" * 3000
    _TapeEar.calls = []
    monkeypatch.setattr(server, "Ear", _TapeEar)
    job = server._tape_job(object(), pcm)
    assert job["final"] is True and job["tape"] is True
    assert job["payload"]["text"] == "same words"
    assert [w["word"] for w in job["payload"]["result"]] == ["same", "words"]

    monkeypatch.setattr(ear_pkg, "Ear", _TapeEar, raising=False)
    monkeypatch.setattr(sre.EarBackend, "_ensure_model", lambda self: object())
    backend = sre.EarBackend.__new__(sre.EarBackend)
    backend.model_path = "unused"
    out = backend.transcribe(pcm, 16000)
    assert out.text == "same words" and out.confidence == pytest.approx(0.75)
    assert out.detail["word_count"] == 2 and "hear_tape" in out.detail["heard_as"]

    assert _TapeEar.calls == [("hear_tape", pcm, 4000), ("hear_tape", pcm, 4000)]


def test_the_word_ear_no_longer_says_the_filament_posts_to_it():
    from christman_voice_sdk.audio import word_ear_receiver

    opening = word_ear_receiver.__doc__.split("WHAT CHANGED AND WHY")[0]
    assert "Filament and Porch post" not in opening
    assert "A caller posts one catch" in opening


# ------------------------------------------------------------ real speech


def _needs_speech(*names: str) -> list[Path]:
    tapes = [TAPES / n for n in names]
    missing = [str(t) for t in tapes if not t.is_file()]
    if not (MODEL / "graph" / "HCLG.fst").is_file() or missing:
        pytest.skip(f"not run: needs the model folder at {MODEL} and the recordings {', '.join(names)} "
                    f"in {TAPES} (EAR_MODEL, EAR_GOLDEN); runs on Everett's Mac")
    try:
        import vosk  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"not run: the vosk library is not installed here ({exc})")
    return tapes


def _wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


def _both_doors(tapes: list[Path]):
    """Each tape's words at the Filament's /stt job and at the word-ear receiver, on one model."""
    from christman_sound.ear import load_model
    from christman_sound.ear.hear import _vosk
    from christman_sound.filament import server
    from christman_voice_sdk.audio.speech_recognition_engine import EarBackend, SpeechRecognitionEngine
    from christman_voice_sdk.audio.word_ear_receiver import WordEarReceiver

    _vosk().SetLogLevel(-1)
    model = load_model(MODEL)
    receiver = WordEarReceiver(engine=SpeechRecognitionEngine(backend=EarBackend(str(MODEL))))
    for tape in tapes:
        pcm = tape.read_bytes()
        at_filament = server._tape_job(model, pcm)["payload"]
        at_word_ear = receiver.take_wav(_wav(pcm))
        yield tape.name, at_filament, at_word_ear


def test_short_phrases_give_the_same_words_at_both_doors():
    tapes = _needs_speech("alphavox_i_need_help_please.s16", "alphavox_i_want_to_go_home.s16",
                          "alphavox_i_am_thirsty_i_need_a_drink.s16", "alphavox_i_want_to_listen_to_music.s16")
    said = {}
    for name, at_filament, at_word_ear in _both_doors(tapes):
        assert at_word_ear.status.value == "ok", (name, at_word_ear.to_dict())
        assert at_word_ear.text == at_filament["text"] != "", name
        said[name] = at_word_ear.text
    assert said["alphavox_i_need_help_please.s16"] == "i need help please"
    assert said["alphavox_i_want_to_go_home.s16"] == "i want to go home"


def test_a_four_minute_recording_gives_the_same_words_at_both_doors():
    if os.environ.get(LONG_ENV) != "1":
        pytest.skip(f"not run: hears a 234-second recording through both doors (about a minute); set {LONG_ENV}=1")
    (tape,) = _needs_speech("AlphaVoxFullVoice.s16")
    ((_name, at_filament, at_word_ear),) = list(_both_doors([tape]))
    assert at_word_ear.status.value == "ok"
    words = at_filament["text"].split()
    assert len(words) > 600, len(words)
    assert at_word_ear.text.split() == words
