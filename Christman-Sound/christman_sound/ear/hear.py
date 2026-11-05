"""The ear's plain surface: a model folder in, words with times and confidences out.

    from christman_sound.ear import Ear

    ear = Ear("~/models/vosk-model-en-us-0.22")       # a model folder on disk
    for piece in pcm_pieces:                           # 16-bit mono PCM, 16 kHz
        if ear.accept(piece):                          # True at an endpoint
            heard = ear.result()                       # Heard(text, words)
    heard = ear.final()                                # flush the rest
    for w in heard.words:
        print(w.word, w.start, w.end, w.conf)          # seconds, seconds, 0..1

Ear.hear(pcm) does the whole loop for one recording and returns every
utterance it heard, endpoints included. One model can serve many Ears:
Ear(model) with a model from load_model() skips reading the folder again.

WHAT CHANGED AND WHY (2026-11-05)
Everett: Vosk is the ear's recognizer, on the full model he downloaded
(vosk-model-en-us-0.22). On 2026-10-29 a Claude Code session took the vosk
library out of the package (commit 7b6e1ce in EverettNC/Christman-Sound,
"Remove the old recognizer") and put its own decoder here in its place, a port
of the ear-core it had built in CODECWAR5TH the day before. That decoder is
kept, not installed, in held/claude-ear-2026-10-28 so it can be looked at.
This file is the same surface it had (Ear, Heard, Word, load_model), so every
caller is unchanged; the words now come from Vosk's recognizer, the one
the Filament runs. Corti sits on top of the hearing (see
christman_voice_sdk/audio/hearing.py); this is the word ear under it.

Everything here is a thin reading of Vosk's own JSON answers, so what Ear says
is exactly what Vosk says. Without the vosk library, load_model raises
EarUnavailable with the reason; nothing hears in its place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Bytes per AcceptWaveform call in Ear.hear: 2000 samples, an eighth of a second.
HEAR_BITE = 4000


class EarUnavailable(RuntimeError):
    """The ear cannot hear here: the vosk library or the model folder is missing. The reason is in the message."""


def _vosk() -> Any:
    try:
        import vosk  # the recognizer the Filament runs
    except ImportError as exc:
        raise EarUnavailable(
            f"the vosk library is not installed here ({exc}). Install Christman-Sound whole "
            "(pip install -e .) or pip install vosk; there is no other ear and no cloud fallback"
        ) from exc
    return vosk


@dataclass(frozen=True)
class Word:
    word: str
    start: float  # seconds from the start of the stream
    end: float  # seconds; the word ends here
    conf: float  # its posterior in the word lattice, 0 to 1


@dataclass(frozen=True)
class Heard:
    text: str
    words: tuple[Word, ...] = ()

    @classmethod
    def from_json(cls, answer: str) -> "Heard":
        d = json.loads(answer)
        words = tuple(Word(w["word"], float(w["start"]), float(w["end"]), float(w["conf"]))
                      for w in d.get("result", ()))
        return cls(d.get("text", ""), words)


def load_model(model_dir: str | Path, graph_dir: str | Path | None = None) -> Any:
    """Read a model folder once (am/, conf/, graph/, ivector/, and rescore/ when the folder holds it)."""
    if graph_dir is not None:
        raise ValueError("graph_dir is not taken: Vosk reads the graph inside the model folder")
    folder = Path(model_dir).expanduser()
    if not folder.is_dir():
        raise EarUnavailable(f"model folder not found: {folder}")
    return _vosk().Model(str(folder))


def _pcm_bytes(pcm: Any) -> bytes:
    """16-bit little-endian PCM as bytes: bytes as they are, or an int16 array."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        return bytes(pcm)
    dtype = getattr(pcm, "dtype", None)
    if dtype is not None:
        if str(dtype) not in ("int16", "<i2"):
            raise TypeError("PCM arrays must be int16")
        return pcm.astype("<i2", copy=False).tobytes()
    if getattr(pcm, "typecode", None) == "h":  # array.array('h')
        return pcm.tobytes()
    raise TypeError(f"PCM must be bytes or an int16 array, not {type(pcm).__name__}")


class Ear:
    """Streaming speech to text over one model folder, fully offline, through Vosk."""

    def __init__(self, model: str | Path | Any, sample_rate: int = 16000, words: bool = True,
                 graph_dir: str | Path | None = None) -> None:
        vosk = _vosk()
        self.model = load_model(model, graph_dir) if isinstance(model, (str, Path)) else model
        if not isinstance(self.model, vosk.Model):
            raise TypeError(f"model must be a model folder or a model from load_model(), not {type(model).__name__}")
        self._rec = vosk.KaldiRecognizer(self.model, float(sample_rate))
        self._rec.SetWords(words)

    def accept(self, pcm: Any) -> bool:
        """Take 16-bit little-endian PCM (bytes, or an int16 array). True at an endpoint."""
        return bool(self._rec.AcceptWaveform(_pcm_bytes(pcm)))

    def partial(self) -> str:
        """The best words so far in the utterance under way."""
        return json.loads(self._rec.PartialResult())["partial"]

    def result(self) -> Heard:
        """Close the utterance at an endpoint; the stream carries on after it."""
        return Heard.from_json(self._rec.Result())

    def final(self) -> Heard:
        """Flush everything taken so far; the next accept starts fresh."""
        return Heard.from_json(self._rec.FinalResult())

    def hear(self, pcm: Any, bite: int = HEAR_BITE) -> list[Heard]:
        """One whole recording: every utterance closed at an endpoint, then the final one."""
        pcm = _pcm_bytes(pcm)
        out = []
        for i in range(0, len(pcm), bite):
            if self.accept(pcm[i : i + bite]):
                out.append(self.result())
        out.append(self.final())
        return out
