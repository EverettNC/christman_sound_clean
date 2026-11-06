# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""The Filament ear server: offline Vosk behind one HTTP door on 127.0.0.1.

Door: $FILAMENT_EAR_PORT, 4850 by default. Not Whole House. Not the Bench.
Whisper is not in this body. Empty ear stays empty. No invented speech.

    christman-filament                 # the console script
    python3 -m christman_sound.filament

Doors (the same as THEFILAMENT ear/vosk_ear.py):

    GET  /health /stt /live    is the model seated; where it is; the door's port
    POST /stt /transcribe /    a whole tape (raw, a WAV, or multipart `file`) -> words
    POST /live                 live 16 kHz 16-bit mono PCM, one stream per
                               caller (?session= or X-Filament-Session);
                               ?reset=1 starts the stream, ?end=1 flushes it
    GET  /carbon               the words set aside for Carbon; POST /carbon clears it
    GET  /board /porch         the Family and Residual boards; POST adds a row
    GET  /  /ui /door          filament-door.html
    GET  /housing              filament-housing.html
    GET  /live.html /live-ui   filament-live.html
    GET  /bare /frozen /field  filament.html (frozen: see pages/FROZEN.md)
    GET  /simple               ui.html
    GET  /dialect-map.js       the pages' dialect map

The words come from Vosk's recognizer through christman_sound.ear (Ear, on a
model from load_model): the same recognizer vosk_ear.py ran directly. The
model sits once, in a worker process of its own; this process never imports
the recognizer, so if the recognizer aborts, the door stays up. The model
folder is found by christman_sound.ear.recognizer.discover_vosk_model()
($VOSK_MODEL_PATH, then the full vosk-model-en-us-0.22 in ~/Downloads or
~/models). Words are dressed by christman_sound.ear.mouth and the boards are
christman_sound.ear.board; both keep their files under $FILAMENT_HOME
(~/.filament by default), where the Filament kept them.

WHAT CHANGED AND WHY (2026-11-05)
Everett's ruling of 2026-10-29: Christman-Sound holds everything he built for
sound, and the Filament ear server moves into the package. This file is
THEFILAMENT ear/vosk_ear.py (EverettNC/THEFILAMENT 568d3cf), brought in with
its live fixes (no final lost when the worker is slow; one stream per
caller). Every door, every JSON field and every answer on the success path is
the same. What is different, and why:

- It hears through christman_sound.ear's Ear and load_model in place of
  making Vosk's recognizer itself. Same recognizer, same 4000-byte bites,
  same half second of silence after a tape or at the end of a stream.
- The model is found by discover_vosk_model() in place of a list holding one
  machine's home folder.
- Fail loud, where vosk_ear.py went quiet (Everett's rule: an ear that cannot
  hear says why; it never answers empty as if nothing was said):
    * A live post the worker could not hear answered ok:true with empty text.
      Now it answers ok:false (503) with the reason, and the stream starts fresh.
    * A worker that died answered ok:true with empty text. Now the post says
      the worker stopped; when a new worker starts, every stream the dead one
      held hears, once, that what it held was not heard.
    * A stream let go because 32 others were open was dropped without a word.
      Now that caller hears why on its next post.
    * A live reply that is late (the worker is still hearing) still answers
      at once with no words, as before, and the words still come on a later
      reply; the quiet answer now carries "pending": true and a note, so it
      cannot be read as silence.
    * The two places that swallowed an error around the recognizer's silence
      tail and the worker's kill no longer catch anything.
    * A page missing from the install is a 500 naming it, not "UI missing"
      with 200 or a different page in its place.
    * A body over 32 MiB is a 413 saying so, not a 400 saying no tape came.
    * A /board post that is not a JSON object is a 400 with the reason, not
      an empty row.
    * If the Carbon pile or the residual board cannot be read or written, the
      words heard are still returned, with "carbon": false and
      "carbon_error" naming the failure.
    * Any other failure inside a request is a 500 with the reason (and the
      traceback on stderr), not a dropped connection.
- The Carbon pile and the boards are read and written under one lock: the
  door answers on many threads, and two writers at once could leave a file
  half written.
- /health's "github" names the package's public home in place of the private
  THEFILAMENT repository (public links point only at The-ChristmanAI-Project).
- The words are dressed by the package's mouth, not THEFILAMENT's copy: the
  package's mouth holds no person's name (Everett's ruling, 2026-10-30) and
  no longer turns "son of a bitch" into "sumbitch" (2026-11-04).
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from christman_sound.ear.board import add_family, add_residual, snapshot
from christman_sound.ear.hear import Ear, EarUnavailable, Heard, load_model
from christman_sound.ear.mouth import CONF_FLOOR, dress, load_carbon, save_carbon, shape_words
from christman_sound.ear.recognizer import discover_vosk_model

DOOR = int(os.environ.get("FILAMENT_EAR_PORT", "4850"))
#: Standing rule: the ear listens on this machine only. Never the wildcard.
HOST = "127.0.0.1"
#: Where the server's source is, as the public sees it.
SOURCE = "https://github.com/The-ChristmanAI-Project/Christman-Sound"
RATE = 16000


def resolve_model_path() -> Optional[str]:
    """The model folder on disk, or None when there is none to seat."""
    found = discover_vosk_model()
    return str(found) if found is not None else None


MODEL_PATH = resolve_model_path()

_model_error: Optional[str] = None
#: How long the door waits for the worker to seat the model.
SEAT_WAIT_S = float(os.environ.get("FILAMENT_SEAT_WAIT_S", "120"))
#: How long a whole tape on /stt may take in the worker.
TAPE_WAIT_S = float(os.environ.get("FILAMENT_TAPE_WAIT_S", "600"))

#: The pages, shipped as package data in christman_sound/filament/pages.
PAGES = {
    "door": "filament-door.html",
    "housing": "filament-housing.html",
    "bare": "filament.html",
    "live": "filament-live.html",
    "simple": "ui.html",
    "map": "dialect-map.js",
}


def page_bytes(name: str) -> bytes:
    """One shipped page, as it is on disk. A page missing from the install raises."""
    page = resources.files(__package__).joinpath("pages", name)
    if not page.is_file():
        raise FileNotFoundError(f"The page {name} is not in this install of christman_sound.filament (looked at {page}).")
    return page.read_bytes()


def seat_model():
    """The model sits once, in the worker. The door only asks whether it sat."""
    global _model_error
    if MODEL_PATH is None or not Path(MODEL_PATH).is_dir():
        where = MODEL_PATH or "$VOSK_MODEL_PATH, ~/Downloads/vosk-model-en-us-0.22 or ~/models/vosk-model-en-us-0.22"
        _model_error = f"Vosk model is not seated: no model folder at {where}. Empty ear stays empty."
        return None
    hub = _live_hub()
    ok, err = hub.seated(SEAT_WAIT_S)
    _model_error = None if ok else (err or "Vosk did not sit in time. Empty ear stays empty.")
    return hub if ok else None


def ffmpeg_bin() -> str:
    for candidate in ("/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg", "ffmpeg"):
        if candidate == "ffmpeg" or Path(candidate).is_file():
            return candidate
    return "ffmpeg"


def to_pcm16_mono_16k(raw: bytes) -> tuple[bytes, float]:
    """16 kHz 16-bit mono PCM from a tape. A WAV already in that form is read as
    it is; anything else (another rate, stereo, float, webm, mp3) is decoded by ffmpeg."""
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(raw), "rb") as wf:
                shape = (wf.getnchannels(), wf.getframerate(), wf.getsampwidth())
                n = wf.getnframes()
                pcm = wf.readframes(n)
        except wave.Error:
            # A WAV the wave module does not read (float samples, a compressed
            # codec). It is not dropped: ffmpeg decodes it below, and says so if it cannot.
            shape = None
        if shape == (1, RATE, 2):
            return pcm, n / RATE

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as src:
        src.write(raw)
        src_path = src.name
    dst = src_path + ".s16"
    try:
        proc = subprocess.run(
            [ffmpeg_bin(), "-y", "-i", src_path, "-ac", "1", "-ar", str(RATE), "-f", "s16le", dst],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not Path(dst).is_file():
            said = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            why = said[-1] if said else f"exit code {proc.returncode}"
            raise RuntimeError(f"ffmpeg could not pull PCM from that tape ({why}).")
        pcm = Path(dst).read_bytes()
        return pcm, len(pcm) / (RATE * 2)
    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(dst).unlink(missing_ok=True)


SILENCE_TAIL = b"\x00" * (RATE * 2 // 2)  # 0.5s. Last word needs room to align.

#: The door answers callers on many threads at once. The Carbon pile and the
#: boards are files read, changed and written whole; one writer at a time.
PILE_LOCK = threading.Lock()


def _dress(body: dict, payload: dict) -> dict:
    """The mouth map and the Carbon pile, through christman_sound.ear.mouth.

    dress() fills `body` in place (text, words, mouth, unsure) before it sets
    unsure words aside on the Carbon pile. If the pile or the residual board
    cannot be read or written, the words that were heard are still returned,
    and the failure is named in the reply and on stderr.
    """
    try:
        with PILE_LOCK:
            return dress(body, payload)
    except (OSError, ValueError) as exc:
        why = f"The unsure words could not be set aside for Carbon: {exc}"
        print(f"Filament: {why}", file=sys.stderr, flush=True)
        body["carbon"] = False
        body["carbon_error"] = why
        return body


def hear(pcm: bytes) -> dict:
    """A whole tape for /stt, heard in the worker."""
    hub = seat_model()
    if hub is None:
        return {"ok": False, "error": _model_error, "seated": False}
    _jid, reply = hub.call("", "tape", pcm, TAPE_WAIT_S)
    if reply is None or reply.get("error"):
        why = (reply or {}).get("error") or "The ear did not finish that tape. Empty ear stays empty."
        return {"ok": False, "error": why, "seated": bool((reply or {}).get("seated", True))}
    final = reply.get("payload") or {}
    text = (final.get("text") or "").strip()
    words = shape_words(final)
    return _dress(
        {
            "ok": True,
            "text": text,
            "transcript": text,
            "words": words,
            "duration": len(pcm) / (RATE * 2),
            "language": "en",
            "engine": "vosk",
            "organ": "filament",
            "cloud": False,
            "seated": True,
        },
        final,
    )


def words_from(payload: dict) -> list[dict]:
    return shape_words(payload)


def _live_packet(
    final: bool,
    text: str = "",
    partial: str = "",
    payload: dict | None = None,
    job: int = 0,
) -> dict:
    body = {
        "ok": True,
        "final": final,
        "text": text,
        "partial": partial,
        "words": words_from(payload or {}),
        "engine": "vosk",
        "organ": "filament",
        "cloud": False,
        "seated": True,
        "job": job,
    }
    return _dress(body, payload or {})


#: How long one /live post waits for the worker before it answers without it.
#: A reply that lands after this is not thrown away: a final is held for the
#: next reply on the same stream, so no final is ever lost.
LIVE_WAIT_S = float(os.environ.get("FILAMENT_LIVE_WAIT_S", "2.5"))
#: An end post waits for its final to land, up to this long, while the worker lives.
END_WAIT_S = float(os.environ.get("FILAMENT_END_WAIT_S", "120"))
#: Fast talk fills one post. Feed the recognizer in 4000-byte bites so it can endpoint.
#: Four seconds without an endpoint balloons the lattice and the ear goes dumb.
HOLD_MAX = RATE * 2 * 4
CHUNK = 4000


def payload_of(heard: Heard) -> dict:
    """Vosk's own answer, rebuilt from what Ear read out of it: the text, and each
    word with its start, end and confidence. Nothing is added."""
    return {
        "text": heard.text,
        "result": [{"conf": w.conf, "end": w.end, "start": w.start, "word": w.word} for w in heard.words],
    }


def _join_finals(finals: list[dict]) -> dict:
    texts = [str(p.get("text") or "").strip() for p in finals]
    text = " ".join(t for t in texts if t)
    result: list = []
    for p in finals:
        result.extend(p.get("result") or [])
    payload = {"text": text, "result": result}
    return {"final": True, "text": text, "partial": "", "payload": payload}


def _fresh(model) -> Ear:
    return Ear(model, sample_rate=RATE, words=True)


def _live_job(model, state: dict, op: str, pcm: bytes) -> dict:
    """One /live post, in the worker. Every bite of audio is heard; no final is dropped."""
    if op == "reset" or state.get("ear") is None:
        state["ear"] = _fresh(model)
        state["held"] = 0
        if op == "reset" and not pcm:
            return {"final": False, "text": "", "partial": "", "payload": {}}
    ear = state["ear"]
    finals: list[dict] = []
    for i in range(0, len(pcm), CHUNK):
        piece = pcm[i : i + CHUNK]
        if ear.accept(piece):
            finals.append(payload_of(ear.result()))
            ear = state["ear"] = _fresh(model)
            state["held"] = 0
            continue
        state["held"] += len(piece)
        if state["held"] >= HOLD_MAX:
            ear.accept(SILENCE_TAIL)
            finals.append(payload_of(ear.final()))
            ear = state["ear"] = _fresh(model)
            state["held"] = 0
    if op == "end":
        ear.accept(SILENCE_TAIL)
        finals.append(payload_of(ear.final()))
        state["ear"] = None
        state["held"] = 0
    if finals:
        return _join_finals(finals)
    said = ear.partial()
    return {"final": False, "text": "", "partial": said.strip(), "payload": {"partial": said}}


#: Streams kept at once. The one idle longest is let go past this many, and
#: its caller hears why on its next post.
MAX_STREAMS = 32
_SESSION_OK = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def session_of(value: str | None) -> str:
    """A caller's stream name. Empty means the shared stream old callers use."""
    value = (value or "").strip()
    return value if _SESSION_OK.match(value) else ""


def _tape_job(model, pcm: bytes) -> dict:
    """A whole tape for /stt: its own recognizer, 0.5 s of silence after it.

    2026-11-05: the loop that was here is Ear.hear_tape now (the same bites,
    the same silence, one final), so the SDK's engine and the word-ear
    receiver hear a tape exactly as this door does.
    """
    return {"final": True, "tape": True, "payload": payload_of(_fresh(model).hear_tape(pcm, bite=CHUNK))}


def _run_tape(model, jid: int, sid: str, pcm: bytes, out_q) -> None:
    try:
        reply = _tape_job(model, pcm)
    except Exception as exc:  # noqa: BLE001 - every failure goes back to the caller, named
        reply = {"tape": True, "seated": True, "error": f"The ear could not hear that tape: {type(exc).__name__}: {exc}."}
    out_q.put(("reply", jid, sid, reply))


def _ear_worker(model_path: str, in_q, out_q) -> None:
    """Own process, and the only place the model sits. If the recognizer
    aborts, the HTTP door stays up."""
    try:
        from christman_sound.ear.hear import _vosk

        _vosk().SetLogLevel(-1)
        model = load_model(model_path)
    except EarUnavailable as exc:
        why = f"{exc}. Empty ear stays empty."
        model = None
    except Exception as exc:  # noqa: BLE001 - the reason goes to every caller, below
        why = f"Vosk would not sit: {type(exc).__name__}: {exc}. Empty ear stays empty."
        model = None
    if model is None:
        out_q.put(("seat", False, why))
        while True:
            job = in_q.get()
            if job is None:
                return
            out_q.put(("reply", job[0], job[1], {"error": why, "seated": False}))
    out_q.put(("seat", True, None))
    streams: dict[str, dict] = {}
    while True:
        job = in_q.get()
        if job is None:
            return
        jid, sid, op, pcm = job
        if op == "tape":
            # A tape runs beside the live streams, so a long tape never holds up live words.
            threading.Thread(target=_run_tape, args=(model, jid, sid, pcm or b"", out_q), daemon=True).start()
            continue
        state = streams.pop(sid, None) or {"ear": None, "held": 0}
        streams[sid] = state  # most recently used last
        while len(streams) > MAX_STREAMS:
            gone = next(iter(streams))
            if streams.pop(gone).get("held", 0) > 0:  # it held audio not yet heard to an end
                out_q.put(("let_go", gone))
        try:
            reply = _live_job(model, state, op, pcm or b"")
        except Exception as exc:  # noqa: BLE001 - the reason goes back to the caller
            state["ear"] = None
            state["held"] = 0
            reply = {
                "seated": True,
                "error": (f"The ear could not hear that: {type(exc).__name__}: {exc}. "
                          "This stream starts fresh; what it held was not heard."),
            }
        out_q.put(("reply", jid, sid, reply))


class LiveHub:
    """The parent side of the live worker.

    Each caller has its own stream, named by the session it sends; callers
    that send none share one stream, as before. One reader thread takes every
    reply off the worker's queue and hands it to the post waiting for it. A
    post that gave up waiting leaves its job behind; when that late reply lands
    with words, it is held for that same stream and delivered on its next
    reply. Nothing the worker heard is thrown away. What the worker could not
    hear (a failed job, a stream let go, a worker that stopped) is held the
    same way, as a reason, and given on that stream's next reply.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._cv = threading.Condition()
        self._proc = None
        self._in = None
        self._gen = 0
        self._next = 0
        self._waiting: set[int] = set()
        self._replies: dict[int, dict] = {}
        self._late: dict[str, list[dict]] = {}
        self._lost: dict[str, list[str]] = {}
        self._open: set[str] = set()
        self._seat: tuple[bool, str | None] | None = None

    def _ensure(self) -> None:
        with self._cv:
            if self._proc is not None and self._proc.is_alive():
                return
            if self._proc is not None:
                why = (f"The ear's worker stopped (exit code {self._proc.exitcode}) and was started again; "
                       "what this stream held was not heard.")
                for sid in self._open:
                    self._lost.setdefault(sid, []).append(why)
                self._open.clear()
            ctx = mp.get_context("spawn")
            in_q = ctx.Queue()
            out_q = ctx.Queue()
            proc = ctx.Process(target=_ear_worker, args=(self.model_path, in_q, out_q), daemon=True)
            proc.start()
            self._gen += 1
            self._seat = None
            self._proc, self._in = proc, in_q
            threading.Thread(target=self._read, args=(out_q, self._gen), daemon=True).start()

    def _read(self, out_q, gen: int) -> None:
        while True:
            try:
                msg = out_q.get(timeout=1.0)
            except queue.Empty:
                if gen != self._gen:
                    return
                continue
            except (EOFError, OSError) as exc:
                # The worker's end of the queue is gone. A caller waiting on it
                # sees the worker stopped (call() checks), and hears so.
                print(f"Filament: the worker's reply queue closed ({exc}).", file=sys.stderr, flush=True)
                return
            if msg[0] == "seat":
                with self._cv:
                    if gen == self._gen:
                        self._seat = (bool(msg[1]), msg[2])
                    self._cv.notify_all()
                continue
            if msg[0] == "let_go":
                with self._cv:
                    self._open.discard(msg[1])
                    self._lost.setdefault(msg[1], []).append(
                        f"This stream sat idle while {MAX_STREAMS} others were open and was let go; "
                        "what it held was not heard.")
                continue
            _kind, jid, sid, reply = msg
            with self._cv:
                if jid in self._waiting:
                    self._replies[jid] = reply
                    self._cv.notify_all()
                elif reply.get("tape"):
                    continue  # its /stt caller gave up after TAPE_WAIT_S and heard the tape did not finish
                elif reply.get("error"):
                    self._lost.setdefault(sid, []).append(reply["error"])
                elif reply.get("final") and str(reply.get("text") or "").strip():
                    self._late.setdefault(sid, []).append(reply)

    def call(self, sid: str, op: str, pcm: bytes, wait_s: float) -> tuple[int, dict | None]:
        """Send one job and wait up to wait_s. None means the worker is alive and
        still hearing it; a worker that stopped answers with an error."""
        self._ensure()
        with self._cv:
            self._next += 1
            jid = self._next
            self._waiting.add(jid)
            proc, in_q = self._proc, self._in
            if op != "tape":
                self._open.add(sid)
        in_q.put((jid, sid, op, pcm))
        deadline = time.monotonic() + wait_s
        with self._cv:
            while jid not in self._replies:
                if not proc.is_alive():
                    self._waiting.discard(jid)
                    self._open.discard(sid)  # it hears it here, so not again when the worker restarts
                    return jid, {
                        "seated": False,
                        "error": (f"The ear's worker stopped (exit code {proc.exitcode}) before it answered; "
                                  "that audio, and what this stream held, was not heard. "
                                  "It starts again on the next call."),
                    }
                left = deadline - time.monotonic()
                if left <= 0:
                    self._waiting.discard(jid)
                    return jid, None
                self._cv.wait(min(left, 0.5))
            self._waiting.discard(jid)
            reply = self._replies.pop(jid)
            if op == "end" and not reply.get("error"):
                self._open.discard(sid)
            return jid, reply

    def seated(self, wait_s: float) -> tuple[bool, str | None]:
        """Start the worker if it is not running, and wait for it to seat the model."""
        self._ensure()
        deadline = time.monotonic() + wait_s
        with self._cv:
            while self._seat is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False, "Vosk did not sit in time. Empty ear stays empty."
                if self._proc is None or not self._proc.is_alive():
                    return False, "The ear's worker stopped before the model sat. Empty ear stays empty."
                self._cv.wait(min(left, 0.5))
            return self._seat

    def restart(self, informed: str | None = None) -> None:
        """Stop the worker; the next call starts a new one. `informed` is a
        stream that already heard it lost its audio, so it does not hear it twice."""
        with self._cv:
            proc = self._proc
            if informed is not None:
                self._open.discard(informed)
        if proc is not None:
            proc.kill()  # a process that already ended is not an error here
            proc.join(timeout=5)

    def take_late(self, sid: str) -> list[dict]:
        with self._cv:
            return self._late.pop(sid, [])

    def take_lost(self, sid: str) -> list[str]:
        with self._cv:
            return self._lost.pop(sid, [])

    def hold(self, sid: str, finals: list[dict]) -> None:
        """Put finals back for this stream's next reply."""
        if finals:
            with self._cv:
                self._late[sid] = finals + self._late.get(sid, [])


_hub: LiveHub | None = None


def _live_hub() -> LiveHub:
    global _hub
    if _hub is None:
        _hub = LiveHub(MODEL_PATH)
    return _hub


def _pending_packet() -> dict:
    """The worker is alive and still hearing this post. Its words come later."""
    return {
        "ok": True,
        "final": False,
        "text": "",
        "partial": "",
        "words": [],
        "engine": "vosk",
        "seated": True,
        "pending": True,
        "note": "The ear is still hearing this. Its words come on a later reply on this stream.",
    }


def _unheard(why: str, seated: bool) -> dict:
    return {"ok": False, "seated": seated, "error": why, "engine": "vosk"}


def live_hear(pcm: bytes, reset: bool = False, end: bool = False, session: str = "") -> dict:
    """Live PCM into Vosk. Not a tape. Not a clip. One stream per session."""
    if seat_model() is None:
        return _unheard(_model_error, False)
    sid = session_of(session)
    op = "reset" if reset else "end" if end else "pcm"
    hub = _live_hub()
    try:
        jid, reply = hub.call(sid, op, pcm or b"", END_WAIT_S if end else LIVE_WAIT_S)
    except (OSError, ValueError, AssertionError) as exc:
        # The queue to the worker broke under this post. The worker is started
        # again on the next call, and every stream it held hears why.
        hub.restart(informed=sid)
        return _unheard(f"The ear's worker could not be reached ({type(exc).__name__}: {exc}); "
                        "that audio, and what this stream held, was not heard. "
                        "It starts again on the next call.", False)
    lost = hub.take_lost(sid)
    if reply is not None and reply.get("error"):
        return _unheard(" ".join(lost + [reply["error"]]), bool(reply.get("seated")))
    finals = hub.take_late(sid)
    if reply is not None and reply.get("final"):
        finals.append(reply)
    if lost:
        # Said once, on this reply. Any words that were heard wait for the next.
        hub.hold(sid, finals)
        return _unheard(" ".join(lost), True)
    if finals:
        joined = _join_finals([f.get("payload") or {"text": f.get("text")} for f in finals])
        return _live_packet(True, text=joined["text"], payload=joined["payload"], job=jid)
    if reply is None:
        return _pending_packet()
    return _live_packet(False, partial=reply.get("partial") or "", payload=reply.get("payload") or {}, job=jid)


def extract_file(headers: dict[str, str], body: bytes) -> bytes:
    ctype = headers.get("Content-Type", "")
    if "multipart/form-data" not in ctype:
        return body
    bound = ""
    for part in ctype.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            bound = part.split("=", 1)[1].strip().strip('"')
    if not bound:
        return body
    marker = b"--" + bound.encode()
    chunks = body.split(marker)
    for chunk in chunks:
        if b"filename=" not in chunk and b"name=\"file\"" not in chunk:
            continue
        split = chunk.split(b"\r\n\r\n", 1)
        if len(split) != 2:
            continue
        data = split[1]
        if data.endswith(b"\r\n"):
            data = data[:-2]
        if data.endswith(b"--"):
            data = data[:-2]
        return data
    return body


#: The largest tape the door reads.
MAX_BODY = 32 * 1024 * 1024

_ROUTES = {
    "/": PAGES["door"], "/ui": PAGES["door"], "/door": PAGES["door"],
    "/housing": PAGES["housing"],
    "/live.html": PAGES["live"], "/live-ui": PAGES["live"],
    "/bare": PAGES["bare"], "/frozen": PAGES["bare"], "/field": PAGES["bare"],
    "/simple": PAGES["simple"],
}


class Door(BaseHTTPRequestHandler):
    """The Filament's HTTP door (vosk_ear.py's Ear handler)."""

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError) as exc:
            print(f"Filament: the caller left before its answer was sent ({exc}).", file=sys.stderr, flush=True)

    def _page(self, name: str, ctype: str) -> None:
        raw = page_bytes(name)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _failed(self, exc: Exception) -> None:
        traceback.print_exc(file=sys.stderr)
        self._json(500, {"ok": False, "error": f"The ear's door failed: {type(exc).__name__}: {exc}"})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type,authorization,x-filament-session")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._get()
        except Exception as exc:  # noqa: BLE001 - answered as a 500 with the reason
            self._failed(exc)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._post()
        except Exception as exc:  # noqa: BLE001 - answered as a 500 with the reason
            self._failed(exc)

    def _get(self) -> None:
        path = urlparse(self.path).path
        if path in _ROUTES:
            self._page(_ROUTES[path], "text/html; charset=utf-8")
            return
        if path == "/dialect-map.js":
            self._page(PAGES["map"], "application/javascript; charset=utf-8")
            return
        if path in ("/health", "/stt", "/live"):
            seated = seat_model() is not None
            self._json(
                200,
                {
                    "ok": seated,
                    "seated": seated,
                    "engine": "vosk",
                    "organ": "filament",
                    "github": SOURCE,
                    "door": DOOR,
                    "model": MODEL_PATH if seated else None,
                    "error": None if seated else _model_error,
                    "filaments": 7200,
                    "housing": "v2",
                    "wholeHouse": False,
                    "live": True,
                    "carbonFloor": CONF_FLOOR,
                },
            )
            return
        if path == "/carbon":
            with PILE_LOCK:
                pile = load_carbon()
            self._json(200, {"ok": True, "for": "Carbon", "count": len(pile), "aside": pile})
            return
        if path in ("/board", "/porch"):
            with PILE_LOCK:
                board = snapshot()
            self._json(200, board)
            return
        self._json(404, {"ok": False, "error": "Not this nerve."})

    def _post(self) -> None:
        path = urlparse(self.path).path
        query = urlparse(self.path).query
        params = dict(part.split("=", 1) if "=" in part else (part, "1") for part in query.split("&") if part)
        if path not in ("/", "/stt", "/transcribe", "/live", "/carbon", "/board", "/porch"):
            self._json(404, {"ok": False, "error": "Not this nerve."})
            return
        if path == "/carbon":
            with PILE_LOCK:
                save_carbon([])
            self._json(200, {"ok": True, "for": "Carbon", "count": 0, "aside": []})
            return
        if path in ("/board", "/porch"):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0) or 0) or b"{}"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._json(400, {"ok": False, "error": f"The board takes a JSON object; that was not JSON ({exc})."})
                return
            if not isinstance(payload, dict):
                self._json(400, {"ok": False, "error": f"The board takes a JSON object, not {type(payload).__name__}."})
                return
            token = str(payload.get("token") or payload.get("said") or "").strip()
            region = str(payload.get("region") or "")
            bin_name = str(payload.get("bin") or payload.get("board") or "family").lower()
            with PILE_LOCK:
                if bin_name == "residual":
                    row = add_residual(token, region, str(payload.get("why") or "slipped"))
                else:
                    row = add_family(token, region)
                board = snapshot()
            self._json(200, {"ok": True, "row": row, **board})
            return
        if seat_model() is None:
            self._json(503, _unheard(_model_error, False))
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._json(413, {"ok": False, "error": f"That tape is {length} bytes; the ear takes up to {MAX_BODY}."})
            return
        body = self.rfile.read(length) if length > 0 else b""

        if path == "/live":
            reset = params.get("reset") in ("1", "true")
            end = params.get("end") in ("1", "true")
            ctype = self.headers.get("Content-Type", "")
            pcm = body
            if "multipart/form-data" in ctype or body[:4] == b"RIFF":
                try:
                    audio = extract_file({k: v for k, v in self.headers.items()}, body)
                    pcm, _dur = to_pcm16_mono_16k(audio)
                except (RuntimeError, OSError, EOFError, wave.Error) as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                    return
            session = unquote(params.get("session") or "") or self.headers.get("X-Filament-Session") or ""
            result = live_hear(pcm, reset=reset, end=end, session=session)
            self._json(200 if result.get("ok") else 503, result)
            return

        if length <= 0:
            self._json(400, {"ok": False, "error": "No tape reached the ear."})
            return
        audio = extract_file({k: v for k, v in self.headers.items()}, body)
        if len(audio) < 64:
            self._json(400, {"ok": False, "error": "The fibers did not catch speech."})
            return
        try:
            pcm, _dur = to_pcm16_mono_16k(audio)
        except (RuntimeError, OSError, EOFError, wave.Error) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        result = hear(pcm)
        if not result.get("ok"):
            self._json(503, result)
            return
        if not (result.get("text") or "").strip():
            self._json(
                200,
                {
                    **result,
                    "text": "",
                    "error": None,
                    "note": "Empty ear stays empty. No invented speech.",
                },
            )
            return
        self._json(200, result)


def serve(host: str = HOST, port: int = DOOR) -> ThreadingHTTPServer:
    """The door, bound and ready; the caller runs serve_forever(). Only 127.0.0.1 is taken."""
    if host != HOST:
        raise ValueError(f"The Filament listens on {HOST} only, not {host}.")
    return ThreadingHTTPServer((host, port), Door)


def main() -> None:
    seated = seat_model() is not None
    print(f"Filament ear on {HOST}:{DOOR}  vosk={'seated' if seated else 'UNSEATED'}", flush=True)
    if not seated:
        print(_model_error, flush=True)
    serve(HOST, DOOR).serve_forever()


if __name__ == "__main__":
    main()
