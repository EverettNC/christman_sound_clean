# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""The Filament ear server hears: real runs on the real recognizer and model.

Brought from THEFILAMENT ear/tests (test_live.py, test_one_model.py,
test_sessions.py), now importing the package. Every check runs Vosk's
recognizer on the full vosk-model-en-us-0.22 through christman_voice_sdk.audio.ear, on
spoken phrases from the folder named by $FILAMENT_TEST_AUDIO (see
filament_helpers.py). Without the model or the recordings, each test is NOT
RUN and says which is missing. The HTTP checks use a port the system hands
out on 127.0.0.1, never 4850, and stop what they start.

WHAT CHANGED AND WHY (2026-11-05)
New in the package. Beside THEFILAMENT's own checks: the late answers carry
"pending", /stt answers over HTTP as hear() does, a body over 32 MiB is a 413,
and the door's own process never imports the recognizer (checked in a fresh
process, since this test run imports Vosk elsewhere).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import threading
import urllib.parse

import pytest

from filament_helpers import (LIVE_CHUNK, ROOT, SILENCE_1S5, ask, pcm, running_door, said,
                              seated_server, three, wav_of)


@pytest.fixture
def ear(tmp_path, monkeypatch):
    monkeypatch.setenv("FILAMENT_HOME", str(tmp_path / "filament"))
    server = seated_server()
    wait = server.LIVE_WAIT_S
    yield server
    server.LIVE_WAIT_S = wait


def stream(ear, tape: bytes, sid: str = "") -> list[dict]:
    out = [ear.live_hear(b"", reset=True, session=sid)]
    for i in range(0, len(tape), LIVE_CHUNK):
        out.append(ear.live_hear(tape[i : i + LIVE_CHUNK], session=sid))
    out.append(ear.live_hear(b"", end=True, session=sid))
    return out


def assert_heard_all_three(text: str) -> None:
    for piece in ("help please", "go home", "thirsty", "drink"):
        assert piece in text, text


# ------------------------------------------------- /live never loses a final

def test_slow_worker_every_final_still_arrives(ear):
    on_time = said(stream(ear, three()))
    assert_heard_all_three(on_time)
    # Now the post never waits: every reply is late. The old door threw those away.
    ear.LIVE_WAIT_S = 0.0
    packets = stream(ear, three())
    quiet = [p for p in packets[1:-1] if "job" not in p]
    assert quiet, "the worker was never late, so this run proved nothing"
    # A late answer says it is late; it is never read as silence.
    assert all(p["pending"] is True and p["ok"] is True and p["text"] == "" for p in quiet)
    assert said(packets) == on_time


def test_audio_on_reset_and_on_end_is_heard(ear):
    # Codec War posts its first chunk with reset, and a caller may post its last with end.
    tape = pcm("music")
    half = (len(tape) // 4) * 2  # on a sample boundary
    first = ear.live_hear(tape[:half], reset=True)
    last = ear.live_hear(tape[half:], end=True)
    assert last["final"] is True
    assert said([first, last]) == "i want to listen to music"


def test_one_long_post_keeps_every_phrase(ear):
    # One post holding three phrases: an endpoint inside it must not cut the rest off.
    ear.live_hear(b"", reset=True)
    one = ear.live_hear(three())
    last = ear.live_hear(b"", end=True)
    assert_heard_all_three(said([one, last]))


def test_live_packets_carry_the_filament_fields(ear):
    packets = stream(ear, pcm("help") + SILENCE_1S5, sid="fields")
    finals = [p for p in packets if p.get("final") and p.get("text")]
    assert finals
    for p in finals:
        assert p["ok"] is True and p["engine"] == "vosk" and p["organ"] == "filament"
        assert p["cloud"] is False and p["seated"] is True and isinstance(p["job"], int)
        assert p["partial"] == "" and isinstance(p["mouth"], list) and isinstance(p["unsure"], list)
        for w in p["words"]:
            assert set(w) == {"text", "raw", "start", "end", "conf", "unsure"}
            assert 0.0 <= w["conf"] <= 1.0 and w["start"] <= w["end"]


# ------------------------------------------------- the model sits once

def test_door_process_never_loads_the_model(ear, tmp_path):
    tape = tmp_path / "help.s16"
    tape.write_bytes(pcm("help"))
    code = (
        "import json, multiprocessing as mp, sys\n"
        "from christman_voice_sdk.audio.filament import server\n"
        "out = server.hear(open(sys.argv[1], 'rb').read())\n"
        "print(json.dumps({'ok': out['ok'], 'text': out.get('text'),"
        " 'raw': [w['raw'] for w in out.get('words', [])],"
        " 'vosk': 'vosk' in sys.modules,"
        " 'workers': len([p for p in mp.active_children() if p.is_alive()])}))\n"
    )
    env = dict(os.environ, VOSK_MODEL_PATH=ear.MODEL_PATH, FILAMENT_HOME=str(tmp_path / "filament"),
               PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    run = subprocess.run([sys.executable, "-c", code, str(tape)], capture_output=True, text=True,
                         env=env, timeout=600, check=False)
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout.strip().splitlines()[-1])
    assert got == {"ok": True, "text": "i need help please", "raw": ["i", "need", "help", "please"],
                   "vosk": False, "workers": 1}


def test_tape_and_live_at_once(ear):
    tape = (pcm("drink") + SILENCE_1S5) * 6
    got: dict = {}
    t = threading.Thread(target=lambda: got.setdefault("tape", ear.hear(tape)))
    t.start()
    live = stream(ear, pcm("music"), sid="beside-a-tape")
    t.join(timeout=300)
    assert said(live) == "i want to listen to music"
    assert got["tape"]["text"].count("thirsty") == 6


def test_health_says_seated(ear):
    with running_door(ear) as port:
        code, _h, body = ask(port, "GET", "/health")
    h = json.loads(body)
    assert code == 200 and h["ok"] is True and h["seated"] is True
    assert h["model"] == ear.MODEL_PATH and h["error"] is None
    assert ear._model_error is None
    workers = [p for p in mp.active_children() if p.is_alive()]
    assert len(workers) == 1, workers


# ------------------------------------------------- one stream per caller

def chunks(tape: bytes) -> list[bytes]:
    return [tape[i : i + LIVE_CHUNK] for i in range(0, len(tape), LIVE_CHUNK)]


def interleave(a_tape: bytes, b_tape: bytes, post) -> tuple[str, str]:
    a, b = chunks(a_tape + SILENCE_1S5), chunks(b_tape + SILENCE_1S5)
    got_a = [post(b"", "alpha", reset=True)]
    got_b = [post(b"", "bravo", reset=True)]
    for i in range(max(len(a), len(b))):
        if i < len(a):
            got_a.append(post(a[i], "alpha"))
        if i < len(b):
            got_b.append(post(b[i], "bravo"))
    got_a.append(post(b"", "alpha", end=True))
    got_b.append(post(b"", "bravo", end=True))
    return said(got_a), said(got_b)


def test_two_callers_interleaved_do_not_mix(ear):
    a, b = interleave(pcm("help"), pcm("music"), lambda c, sid, **kw: ear.live_hear(c, session=sid, **kw))
    assert a == "i need help please"
    assert b == "i want to listen to music"


def test_caller_with_no_session_keeps_the_shared_stream(ear):
    ear.live_hear(b"", reset=True)
    for i in range(0, len(pcm("home")), LIVE_CHUNK):
        ear.live_hear(pcm("home")[i : i + LIVE_CHUNK])
    assert said([ear.live_hear(b"", end=True)]) == "i want to go home"


def test_over_http_query_and_header(ear):
    with running_door(ear) as port:
        def post(chunk, sid, reset=False, end=False):
            q = "&".join(x for x in ("reset=1" if reset else "", "end=1" if end else "") if x)
            url = "/live" + (f"?{q}" if q else "")
            headers = {"Content-Type": "application/octet-stream"}
            if sid == "alpha":
                # percent-encoded, as a browser's URL tools may send it
                url += ("&" if q else "?") + "session=" + urllib.parse.quote("codec-war:alpha", safe="")
            else:
                headers["X-Filament-Session"] = sid
            code, _h, body = ask(port, "POST", url, body=chunk, headers=headers, timeout=180)
            assert code == 200, body
            return json.loads(body)

        a, b = interleave(pcm("drink"), pcm("home"), post)
    assert a == "i am thirsty i need a drink"
    assert b == "i want to go home"


# ------------------------------------------------- /stt over HTTP

def test_stt_over_http_answers_as_hear_does(ear):
    direct = ear.hear(pcm("help"))
    with running_door(ear) as port:
        code, _h, raw_body = ask(port, "POST", "/stt", body=wav_of(pcm("help")),
                                 headers={"Content-Type": "audio/wav"})
        tape = wav_of(pcm("help"))
        form = (b"--Fil\r\nContent-Disposition: form-data; name=\"file\"; filename=\"catch.wav\"\r\n"
                b"Content-Type: audio/wav\r\n\r\n" + tape + b"\r\n--Fil--\r\n")
        code2, _h2, form_body = ask(port, "POST", "/transcribe", body=form,
                                    headers={"Content-Type": "multipart/form-data; boundary=Fil"})
    assert code == 200 and code2 == 200
    for got in (json.loads(raw_body), json.loads(form_body)):
        assert got == direct
    assert direct["text"] == direct["transcript"] == "i need help please"
    assert direct["engine"] == "vosk" and direct["organ"] == "filament" and direct["cloud"] is False
    assert direct["language"] == "en" and abs(direct["duration"] - len(pcm("help")) / 32000) < 1e-9


def test_a_tape_over_32_mib_is_refused_by_size(ear):
    with running_door(ear) as port:
        with socket.create_connection(("127.0.0.1", port), timeout=120) as s:
            s.sendall(b"POST /stt HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: audio/wav\r\n"
                      b"Content-Length: 33554433\r\n\r\n")
            reply = b""
            while b"\r\n\r\n" not in reply or not reply.rstrip().endswith(b"}"):
                piece = s.recv(65536)
                if not piece:
                    break
                reply += piece
    head, _sep, body = reply.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.0 413")
    assert json.loads(body)["error"] == "That tape is 33554433 bytes; the ear takes up to 33554432."
