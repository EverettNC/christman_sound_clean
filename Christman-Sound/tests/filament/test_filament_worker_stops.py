# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""When the ear's worker stops, every caller it was hearing for hears why.

A real run: the real worker process, with the full model, is stopped in the
middle of two callers' streams (the server's own worker, started by this
test; never the Filament on 4850). vosk_ear.py answered such a post with
ok:true and no words, as if nothing had been said. The server says the worker
stopped, gives the other open stream the same reason on its next post, starts
a new worker, and hears again.

WHAT CHANGED AND WHY (2026-11-05)
New, with the Filament ear server's move into the package. This file runs
after the other live tests (it is last by name), because it stops the worker
they share.
"""

from __future__ import annotations

import pytest

from filament_helpers import LIVE_CHUNK, SILENCE_1S5, pcm, said, seated_server


@pytest.fixture
def ear(tmp_path, monkeypatch):
    monkeypatch.setenv("FILAMENT_HOME", str(tmp_path / "filament"))
    return seated_server()


def test_a_stopped_worker_is_told_to_every_open_stream(ear):
    music = pcm("music")
    half = (len(music) // 4) * 2
    assert ear.live_hear(b"", reset=True, session="speaking")["ok"] is True
    assert ear.live_hear(music[:half], session="speaking")["ok"] is True
    assert ear.live_hear(b"", reset=True, session="bystander")["ok"] is True
    assert ear.live_hear(pcm("home")[:LIVE_CHUNK], session="bystander")["ok"] is True

    hub = ear._live_hub()
    worker = hub._proc
    worker.kill()
    worker.join(timeout=30)
    assert not worker.is_alive()

    reason = ear.live_hear(music[half:], session="speaking")
    assert reason["ok"] is False and reason["seated"] is True, reason
    assert "worker stopped" in reason["error"] and "not heard" in reason["error"]
    assert "text" not in reason
    assert hub._proc is not worker and hub._proc.is_alive()

    bystander = ear.live_hear(b"", end=True, session="bystander")
    assert bystander["ok"] is False and "worker stopped" in bystander["error"], bystander

    # Said once. The new worker heard the rest of the post the reason came on;
    # the stream's end brings that out, with no second notice.
    rest = ear.live_hear(b"", end=True, session="speaking")
    assert rest["ok"] is True and rest["final"] is True, rest

    # And the new worker hears both callers again from here.
    tape = pcm("help") + SILENCE_1S5
    again = [ear.live_hear(b"", reset=True, session="speaking")]
    for i in range(0, len(tape), LIVE_CHUNK):
        again.append(ear.live_hear(tape[i : i + LIVE_CHUNK], session="speaking"))
    again.append(ear.live_hear(b"", end=True, session="speaking"))
    assert all(p["ok"] for p in again), again
    assert said(again) == "i need help please"
    later = ear.live_hear(b"", end=True, session="bystander")
    assert later["ok"] is True and later["final"] is True
