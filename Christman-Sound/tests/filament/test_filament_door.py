# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""The Filament ear server's door, run for real with no model needed.

Each check starts the real door on a port the system hands out on 127.0.0.1
and stops it after; nothing here touches 4850. The Carbon pile and the boards
live in a folder made for the test ($FILAMENT_HOME). The live hearing is in
test_filament_live.py, which needs the model.

WHAT CHANGED AND WHY (2026-11-05)
New, with the server's move into the package from THEFILAMENT ear/vosk_ear.py.
Besides the doors, these check the places vosk_ear.py went quiet on a failure
and the server now names it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import wave
from pathlib import Path

import pytest

from filament_helpers import ROOT, ask, running_door, wav_of

PAGES_DIR = ROOT / "christman_voice_sdk" / "audio" / "filament" / "pages"
#: sha256 of ear/filament.html at EverettNC/THEFILAMENT 568d3cf. Frozen by
#: Everett (FROZEN.md): the package ships it byte for byte.
FROZEN_SHA256 = "581c60f16f5137a1f46abc172f17d8efaddc122a0c5582573d6299d2c622e7cb"

ROUTES = {
    "/": "filament-door.html", "/ui": "filament-door.html", "/door": "filament-door.html",
    "/housing": "filament-housing.html",
    "/live.html": "filament-live.html", "/live-ui": "filament-live.html",
    "/bare": "filament.html", "/frozen": "filament.html", "/field": "filament.html",
    "/simple": "ui.html",
}


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("FILAMENT_HOME", str(tmp_path / "filament"))
    from christman_voice_sdk.audio.filament import server as srv

    return srv


@pytest.fixture
def unseated(server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MODEL_PATH", str(tmp_path / "no-model-here"))
    return server


def test_the_frozen_page_ships_byte_for_byte():
    raw = (PAGES_DIR / "filament.html").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA256
    assert sorted(p.name for p in PAGES_DIR.iterdir()) == [
        "FROZEN.md", "dialect-map.js", "filament-door.html", "filament-housing.html",
        "filament-live.html", "filament.html", "ui.html"]


def test_every_page_route_serves_its_page(server):
    with running_door(server) as port:
        for route, name in ROUTES.items():
            code, head, body = ask(port, "GET", route)
            assert code == 200, route
            assert head["content-type"] == "text/html; charset=utf-8", route
            assert body == (PAGES_DIR / name).read_bytes(), route
            assert head["access-control-allow-origin"] == "*"
        code, head, body = ask(port, "GET", "/dialect-map.js")
        assert code == 200
        assert head["content-type"] == "application/javascript; charset=utf-8"
        assert body == (PAGES_DIR / "dialect-map.js").read_bytes()


def test_a_page_missing_from_the_install_is_named_not_replaced(server):
    with pytest.raises(FileNotFoundError, match=r"The page filament-garden\.html is not in this install"):
        server.page_bytes("filament-garden.html")
    assert server.page_bytes("filament.html") == (PAGES_DIR / "filament.html").read_bytes()


def test_unknown_nerves_and_options(server):
    with running_door(server) as port:
        for method in ("GET", "POST"):
            code, _h, body = ask(port, method, "/nowhere", body=b"" if method == "POST" else None)
            assert code == 404 and json.loads(body) == {"ok": False, "error": "Not this nerve."}
        code, head, _b = ask(port, "OPTIONS", "/live")
        assert code == 204
        assert head["access-control-allow-methods"] == "GET,POST,OPTIONS"
        assert head["access-control-allow-headers"] == "content-type,authorization,x-filament-session"


def test_unseated_ear_says_why_on_every_door(unseated, tmp_path):
    missing = str(tmp_path / "no-model-here")
    with running_door(unseated) as port:
        for path in ("/health", "/stt", "/live"):
            code, _h, body = ask(port, "GET", path)
            h = json.loads(body)
            assert code == 200
            assert h["ok"] is False and h["seated"] is False and h["model"] is None
            assert missing in h["error"] and "not seated" in h["error"]
            assert h["engine"] == "vosk" and h["organ"] == "filament" and h["carbonFloor"] == 0.55
            assert h["door"] == unseated.DOOR and h["live"] is True and h["wholeHouse"] is False
            assert h["github"].startswith("https://github.com/The-ChristmanAI-Project/")
        for path in ("/stt", "/transcribe", "/", "/live", "/live?reset=1"):
            code, _h, body = ask(port, "POST", path, body=wav_of(b"\x00\x01" * 1600),
                                 headers={"Content-Type": "audio/wav"})
            got = json.loads(body)
            assert code == 503, path
            assert got["ok"] is False and got["seated"] is False and missing in got["error"]
            assert "text" not in got


def test_no_model_folder_anywhere_names_where_it_looked(server, monkeypatch):
    monkeypatch.setattr(server, "MODEL_PATH", None)
    assert server.seat_model() is None
    assert "VOSK_MODEL_PATH" in server._model_error and "~/models" in server._model_error


def test_session_names_are_checked(server):
    assert server.session_of("codec-war:tab.1") == "codec-war:tab.1"
    assert server.session_of("") == ""
    assert server.session_of(None) == ""
    assert server.session_of("../etc") == ""
    assert server.session_of("x" * 65) == ""


def test_multipart_file_part_is_taken(server):
    tape = wav_of(b"\x10\x00" * 400)
    body = (b"--XyZ\r\nContent-Disposition: form-data; name=\"diarize\"\r\n\r\nfalse\r\n"
            b"--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"catch.wav\"\r\n"
            b"Content-Type: audio/wav\r\n\r\n" + tape + b"\r\n--XyZ--\r\n")
    got = server.extract_file({"Content-Type": "multipart/form-data; boundary=XyZ"}, body)
    assert got == tape
    assert server.extract_file({"Content-Type": "audio/wav"}, tape) == tape


def test_a_16k_mono_wav_is_read_as_it_is(server):
    samples = b"".join(int(1000 * ((i % 40) - 20)).to_bytes(2, "little", signed=True) for i in range(16000))
    pcm, seconds = server.to_pcm16_mono_16k(wav_of(samples))
    assert pcm == samples and seconds == 1.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="not run: ffmpeg is not on this machine")
def test_other_tapes_go_through_ffmpeg_and_a_bad_one_says_why(server, tmp_path):
    import io

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(b"\x00\x10" * 8000)  # one second at 8 kHz
    pcm, seconds = server.to_pcm16_mono_16k(out.getvalue())
    assert abs(seconds - 1.0) < 0.01 and len(pcm) % 2 == 0
    with pytest.raises(RuntimeError, match=r"ffmpeg could not pull PCM from that tape \(.+\)"):
        server.to_pcm16_mono_16k(b"this is not audio at all" * 8)
    assert not list(Path(server.tempfile.gettempdir()).glob("*.bin.s16"))


def test_boards_add_and_read(server, tmp_path):
    with running_door(server) as port:
        code, _h, body = ask(port, "POST", "/board", body=json.dumps({"token": "holler", "region": "kentucky"}).encode())
        got = json.loads(body)
        assert code == 200 and got["ok"] is True and got["row"]["said"] == "holler"
        assert got["row"]["region"] == "kentucky" and got["organ"] == "porch"
        code, _h, body = ask(port, "POST", "/porch", body=json.dumps({"said": "mumbled", "bin": "residual"}).encode())
        assert code == 200 and json.loads(body)["row"]["why"] == "slipped"
        code, _h, body = ask(port, "GET", "/board")
        snap = json.loads(body)
        assert [r["said"] for r in snap["family"]] == ["holler"]
        assert [r["said"] for r in snap["residual"]] == ["mumbled"]
        assert (tmp_path / "filament" / "family.json").is_file()


def test_a_board_post_that_is_not_a_json_object_is_refused(server):
    with running_door(server) as port:
        code, _h, body = ask(port, "POST", "/board", body=b"{not json")
        assert code == 400 and "not JSON" in json.loads(body)["error"]
        code, _h, body = ask(port, "POST", "/board", body=b'["holler"]')
        assert code == 400 and "not list" in json.loads(body)["error"]


def test_a_broken_board_is_named_and_left_as_it_is(server, tmp_path):
    home = tmp_path / "filament"
    home.mkdir()
    (home / "family.json").write_text("[{\"said\": \"holler\"", encoding="utf-8")
    with running_door(server) as port:
        code, _h, body = ask(port, "GET", "/board")
        got = json.loads(body)
        assert code == 500 and got["ok"] is False and "family.json" in got["error"]
        code, _h, body = ask(port, "POST", "/board", body=json.dumps({"token": "yonder"}).encode())
        assert code == 500 and "family.json" in json.loads(body)["error"]
    assert (home / "family.json").read_text(encoding="utf-8") == "[{\"said\": \"holler\""


def test_carbon_pile_read_and_cleared(server, tmp_path):
    home = tmp_path / "filament"
    home.mkdir()
    (home / "carbon.json").write_text(json.dumps([{"word": "mumbled", "for": "Carbon"}]), encoding="utf-8")
    with running_door(server) as port:
        code, _h, body = ask(port, "GET", "/carbon")
        assert code == 200 and json.loads(body) == {
            "ok": True, "for": "Carbon", "count": 1, "aside": [{"word": "mumbled", "for": "Carbon"}]}
        code, _h, body = ask(port, "POST", "/carbon", body=b"")
        assert code == 200 and json.loads(body)["count"] == 0
    assert json.loads((home / "carbon.json").read_text(encoding="utf-8")) == []


def test_a_broken_carbon_pile_is_named_and_left_as_it_is(server, tmp_path):
    home = tmp_path / "filament"
    home.mkdir()
    (home / "carbon.json").write_text("{\"word\":", encoding="utf-8")
    with running_door(server) as port:
        code, _h, body = ask(port, "GET", "/carbon")
    got = json.loads(body)
    assert code == 500 and got["ok"] is False and "carbon.json" in got["error"]
    assert (home / "carbon.json").read_text(encoding="utf-8") == "{\"word\":"


def _vosk_shaped(conf_second: float) -> dict:
    # The shape of Vosk's FinalResult with words on: the dressing is what is checked here.
    return {"text": "help me", "result": [
        {"conf": 1.0, "end": 0.42, "start": 0.12, "word": "help"},
        {"conf": conf_second, "end": 0.66, "start": 0.42, "word": "me"},
    ]}


def _tape_body(payload: dict) -> dict:
    text = payload["text"]
    return {"ok": True, "text": text, "transcript": text, "words": [], "duration": 0.7, "language": "en",
            "engine": "vosk", "organ": "filament", "cloud": False, "seated": True}


def test_unsure_words_go_to_carbon_and_the_residual_board(server, tmp_path):
    out = server._dress(_tape_body(_vosk_shaped(0.31)), _vosk_shaped(0.31))
    assert out["text"] == "help me" and [w["raw"] for w in out["words"]] == ["help", "me"]
    assert out["carbon"] is True and [w["raw"] for w in out["unsure"]] == ["me"]
    home = tmp_path / "filament"
    assert json.loads((home / "carbon.json").read_text(encoding="utf-8"))[0]["word"] == "me"
    assert json.loads((home / "residual.json").read_text(encoding="utf-8"))[0]["said"] == "me"


@pytest.mark.parametrize("broken", ["carbon.json", "residual.json"])
def test_words_heard_are_kept_when_the_pile_cannot_be_written(server, tmp_path, broken):
    home = tmp_path / "filament"
    home.mkdir()
    (home / broken).write_text("{oops", encoding="utf-8")
    out = server._dress(_tape_body(_vosk_shaped(0.31)), _vosk_shaped(0.31))
    assert out["ok"] is True and out["text"] == "help me" and out["transcript"] == "help me"
    assert [w["raw"] for w in out["words"]] == ["help", "me"] and len(out["unsure"]) == 1
    assert out["carbon"] is False and broken in out["carbon_error"]
    assert (home / broken).read_text(encoding="utf-8") == "{oops"


def test_the_door_listens_on_this_machine_only(server):
    with pytest.raises(ValueError, match="127.0.0.1 only"):
        server.serve(host="0.0.0.0", port=0)


def test_the_package_declares_the_console_script_and_the_pages():
    try:
        import tomllib
    except ModuleNotFoundError:
        pytest.skip("not run: tomllib needs Python 3.11 or later")
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["scripts"]["christman-filament"] == "christman_voice_sdk.audio.filament.server:main"
    assert meta["tool"]["setuptools"]["package-data"]["christman_voice_sdk.audio.filament"] == ["pages/*"]
