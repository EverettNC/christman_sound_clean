# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""What the Filament ear server's tests share: the model, the recordings, a door.

The recordings are not in the package (it holds no being's voice). The live
tests hear four spoken phrases from a folder named by $FILAMENT_TEST_AUDIO,
16 kHz 16-bit mono, as .wav or raw .s16 files whose names end in the phrase:

    i_need_help_please, i_want_to_go_home, i_am_thirsty_i_need_a_drink,
    i_want_to_listen_to_music

(THEFILAMENT ear/tests/audio holds such a set.) Without that folder, or
without the full vosk-model-en-us-0.22, a test that needs them is NOT RUN
and says which is missing.

WHAT CHANGED AND WHY (2026-11-05)
New, with the Filament ear server's move into the package. It carries what
THEFILAMENT ear/tests/test_live.py shared with the other tests there.
"""

from __future__ import annotations

import http.client
import os
import sys
import threading
import wave
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vosk_models import FULL_MODEL, is_full_model  # noqa: E402

AUDIO_ENV = "FILAMENT_TEST_AUDIO"
PHRASES = {
    "help": "i_need_help_please",
    "home": "i_want_to_go_home",
    "drink": "i_am_thirsty_i_need_a_drink",
    "music": "i_want_to_listen_to_music",
}
SILENCE_1S5 = b"\x00" * (16000 * 2 * 3 // 2)
LIVE_CHUNK = 3200  # what THEFILAMENT src/lib/filament/live.ts posts


def recording(key: str) -> Path:
    raw = os.environ.get(AUDIO_ENV, "").strip()
    if not raw:
        pytest.skip(f"not run: needs a folder of the four spoken phrases ({', '.join(PHRASES.values())}) "
                    f"as 16 kHz mono .wav or .s16, named by {AUDIO_ENV}; none is named here")
    folder = Path(raw).expanduser()
    stem = PHRASES[key]
    hits = sorted(folder.glob(f"*{stem}.wav")) or sorted(folder.glob(f"*{stem}.s16"))
    if not hits:
        pytest.skip(f"not run: no recording ending in {stem}.wav or {stem}.s16 in {folder} ({AUDIO_ENV})")
    return hits[0]


def pcm(key: str) -> bytes:
    """16 kHz 16-bit mono PCM of one phrase. A .wav in any other form fails, by name."""
    path = recording(key)
    if path.suffix == ".s16":
        return path.read_bytes()
    with wave.open(str(path), "rb") as wf:
        shape = (wf.getframerate(), wf.getnchannels(), wf.getsampwidth())
        assert shape == (16000, 1, 2), f"{path} is {shape}, not 16 kHz mono 16-bit"
        return wf.readframes(wf.getnframes())


def wav_of(pcm_bytes: bytes) -> bytes:
    """A 16 kHz mono 16-bit WAV around PCM, as a caller would post it."""
    import io

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_bytes)
    return out.getvalue()


def said(packets: list[dict]) -> str:
    return " ".join(p["text"] for p in packets if p.get("final") and p.get("text")).lower().strip()


def three() -> bytes:
    return pcm("help") + SILENCE_1S5 + pcm("home") + SILENCE_1S5 + pcm("drink") + SILENCE_1S5


def seated_server():
    """The server module, with the full model found. Otherwise NOT RUN, with the reason."""
    from christman_voice_sdk.audio.filament import server

    path = server.MODEL_PATH
    if path is None:
        pytest.skip("not run: no Vosk model folder here ($VOSK_MODEL_PATH, ~/Downloads or ~/models); "
                    f"the live tests run where the full {FULL_MODEL} is")
    if not is_full_model(path):
        pytest.skip(f"not run: needs the full {FULL_MODEL} model; the server found {path}")
    return server


@contextmanager
def running_door(server):
    """The real door on a port the system hands out, on 127.0.0.1. Stopped after."""
    httpd = server.serve(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def ask(port: int, method: str, path: str, body: bytes | None = None,
        headers: dict | None = None, timeout: float = 300) -> tuple[int, dict, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        conn.close()
