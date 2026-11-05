# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Tests for the installable application (christman_sound.app, PWA manifest/service
worker, CLI entry points in pyproject.toml, and native desktop bundle installer).
"""

from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_fixture import write_test_voice  # noqa: E402
from christman_sound.app import create_app, install_desktop_bundle  # noqa: E402

FIXTURE_LEXICON = """\
christman christman k_B r_I I_I s_I t_I m_I V_I n_E
hello hello h_B @_I l_I oU_E
booth booth b_B u_I T_E
"""

#: The real align lexicon (vosk-model-en-us-0.22), as tests/test_rulings_2026_10_29.py
#: finds it: CHRISTMAN_LEXICON_PATH, or the model under ~/models. The fixture
#: above covers three words; a reply carrying a forbidden line needs words it
#: does not hold, and no transcription is invented here.
ALIGN_LEXICON = Path(os.environ.get("CHRISTMAN_LEXICON_PATH")
                     or Path.home() / "models" / "vosk-model-en-us-0.22" / "graph" / "phones" / "align_lexicon.txt")


def test_pyproject_declares_cli_scripts_and_ocr_module():
    import tomllib
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert "christman-sound" in scripts
    assert "christman-sound-install" in scripts
    assert "christman-tts" in scripts
    assert "christman-corti" in scripts
    assert "christman-word-ear" in scripts
    assert "christman-capture" in scripts
    assert "christman_ocr_shared" in data["tool"]["setuptools"]["py-modules"]
    assert (ROOT / "install.sh").is_file()


def test_pwa_shell_and_manifest_routes():
    app = create_app()
    client = app.test_client()

    r_index = client.get("/")
    assert r_index.status_code == 200
    html = r_index.data.decode("utf-8")
    assert "Christman Sound" in html
    assert "/manifest.webmanifest" in html
    assert "serviceWorker.register('/sw.js')" in html

    r_man = client.get("/manifest.webmanifest")
    assert r_man.status_code == 200
    manifest = json.loads(r_man.data.decode("utf-8"))
    assert manifest["display"] == "standalone"
    assert len(manifest["icons"]) >= 3

    r_sw = client.get("/sw.js")
    assert r_sw.status_code == 200
    assert b"christman-sound-v1" in r_sw.data


def test_api_status_and_codecwar_and_synthesize(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRISTMAN_VOICE_CACHE", str(tmp_path / "cache"))
    app = create_app()
    client = app.test_client()

    st = client.get("/api/status").get_json()
    assert st["status"] == "online"
    assert st["cloud_dependencies"] is False
    assert st["dsp"]["available"] is True
    assert st["dsp"]["synth_available"] is True
    assert len(st["dsp"]["phones"]) == 40

    # 1. Bare Codec War table
    cw = client.post("/api/codecwar-table", json={
        "text": "Hello, Christman.",
        "f0": 148.0,
        "rate": 1.0,
        "sample_rate": 16000,
        "lexicon_text": FIXTURE_LEXICON,
    })
    assert cw.status_code == 200, cw.get_json()
    cw_data = cw.get_json()
    assert cw_data["renderer"] == "christman_synth"
    assert cw_data["measurements"]["rms"] > 0.04

    # 2. Synthesize without voice -> 503 NoVoice
    no_v = client.post("/api/synthesize", json={
        "text": "hello",
        "lexicon_text": FIXTURE_LEXICON,
    })
    assert no_v.status_code == 503
    assert no_v.get_json()["kind"] == "NoVoice"

    # 3. Synthesize with measured voice profile -> 200 OK + playable WAV
    vpath = write_test_voice(tmp_path / "voice")
    syn = client.post("/api/synthesize", json={
        "text": "Hello, Christman.",
        "voice": str(vpath),
        "lexicon_text": FIXTURE_LEXICON,
    })
    assert syn.status_code == 200, syn.get_json()
    syn_data = syn.get_json()
    assert syn_data["renderer"] == "christman_synth_clear"
    assert syn_data["audio_url"].startswith("/audio/")

    wav_resp = client.get(syn_data["audio_url"])
    assert wav_resp.status_code == 200
    wav_file = tmp_path / "downloaded.wav"
    wav_file.write_bytes(wav_resp.data)
    with wave.open(str(wav_file), "rb") as w:
        assert w.getnframes() == syn_data["samples_count"]

    # 4. The reply guard is a reading (2026-11-04). The studio renders what
    # was asked and returns forbidden_lines beside the audio: on a being's
    # reply, the names of the lines found (here none); on anything else,
    # None, because the guard is never read on a client's words. Until
    # 2026-11-04 a forbidden being reply was refused with 422 ForbiddenReply;
    # that reply, rendered with its lines named, is
    # test_api_synthesize_renders_a_being_reply_and_names_its_forbidden_lines.
    assert syn_data["forbidden_lines"] is None
    reply = client.post("/api/synthesize", json={
        "text": "Hello, Christman.",
        "voice": str(vpath),
        "lexicon_text": FIXTURE_LEXICON,
        "being_reply": True,
    })
    assert reply.status_code == 200, reply.get_json()
    assert reply.get_json()["forbidden_lines"] == []

    # 5. Measure WAV endpoint on rendered WAV
    meas = client.post("/api/measure-wav", json={"path": str(wav_file)})
    assert meas.status_code == 200, meas.get_json()
    mdata = meas.get_json()
    assert mdata["sample_rate"] == 16000
    assert mdata["rms"] > 0.04
    assert mdata["pitch_contour"]["voiced_frames"] > 0


def test_api_synthesize_renders_a_being_reply_and_names_its_forbidden_lines(tmp_path, monkeypatch):
    """Was step 4 of test_api_status_and_codecwar_and_synthesize, a 422
    ForbiddenReply. The guard no longer refuses (2026-11-04): the reply is
    rendered and its lines are named beside the audio; the being decides."""
    if not ALIGN_LEXICON.is_file():
        pytest.skip(f"not run: needs the align lexicon at {ALIGN_LEXICON} (or CHRISTMAN_LEXICON_PATH); "
                    "the fixture lexicon does not hold the words of a forbidden line")
    monkeypatch.setenv("CHRISTMAN_VOICE_CACHE", str(tmp_path / "cache"))
    client = create_app().test_client()
    vpath = write_test_voice(tmp_path / "voice")
    forb = client.post("/api/synthesize", json={
        "text": "Calm down, you're not okay.",
        "voice": str(vpath),
        "lexicon_path": str(ALIGN_LEXICON),
        "being_reply": True,
    })
    assert forb.status_code == 200, forb.get_json()
    body = forb.get_json()
    assert body["audio_url"].startswith("/audio/") and body["samples_count"] > 0
    assert body["forbidden_lines"] == ["you're not okay", "calm down"]
    # With the being's own danger reading passed back, the announcement lines
    # are read too.
    announce = client.post("/api/synthesize", json={
        "text": "Help is coming.",
        "voice": str(vpath),
        "lexicon_path": str(ALIGN_LEXICON),
        "being_reply": True,
        "danger": True,
    })
    assert announce.status_code == 200, announce.get_json()
    assert "help is coming" in announce.get_json()["forbidden_lines"]


def test_api_hear_crisis_and_nonverbal():
    app = create_app()
    client = app.test_client()

    r = client.post("/api/hear", json={"text": "I'm not okay"})
    assert r.status_code == 200
    body = r.get_json()
    # 2026-11-04: the studio pages no one. `paged_human` is now the reading
    # it rested on (crisis_reading, hard_block) and an honest human_reached.
    assert body["human_reached"] is False
    assert body["human_reached_reason"] == "the studio pages no one"
    assert "paged_human" not in body and "pager_events" not in body
    assert body["crisis_reading"]["status"] == "hard_block"
    assert body["crisis_reading"]["reason"] == "client_distress"
    assert body["crisis_reading"] == body["turn"]["reading"]["safety"]
    assert body["turn"]["reading"]["safety"]["reason"] == "client_distress"

    r_sym = client.post("/api/hear", json={"nonverbal": {"symbol": "food"}})
    assert r_sym.status_code == 200
    assert r_sym.get_json()["turn"]["person_message"] == "I'm hungry."


@pytest.mark.parametrize("path", ["/api/hear", "/api/synthesize"])
@pytest.mark.parametrize("key,value,replaced_by", [
    ("public_setting", True, "`danger` and `asked_aloud`"),
    ("auto_route_grunt", True, "`acoustic_key`"),
    ("silent_alarm", True, "pass `danger`"),
    ("answering_follow_up", "ask_if_safe", "pass `answering`"),
])
def test_a_removed_request_key_is_refused_by_name(path, key, value, replaced_by):
    """Fix round, 2026-11-04 (Rule 6): the studio quietly ignored these keys
    after they were removed, so a caller sending one believed it was doing
    something. Each is now a 400 that names the key and what replaced it,
    before anything is heard or rendered."""
    client = create_app().test_client()
    r = client.post(path, json={"text": "I'm scared for my safety", key: value})
    assert r.status_code == 400
    body = r.get_json()
    assert body["kind"] == "RemovedRequestKey"
    assert [item["key"] for item in body["removed"]] == [key]
    assert replaced_by in body["removed"][0]["replaced_by"]
    assert repr(key) in body["reason"] and "turn" not in body


def test_api_hear_grounding_comes_from_a_measured_level_or_says_none_was():
    """2026-10-30: no made-up 0.75 stress score, no method that does not exist.

    Decision moved to the being 2026-11-04: see AlphaVox's WHAT_THE_BEING_DECIDES.md.
    The studio no longer chooses a grounding script, and `needs_grounding`
    is gone with the mode. What stays is the reading: the distress Silicon
    read, the stress rise measured from the words (or the reason none was),
    and the level the caller sent, echoed exactly as sent."""
    client = create_app().test_client()
    said = "I can't breathe I'm panicking"
    body = client.post("/api/hear", json={"text": said}).get_json()
    silicon = body["turn"]["reading"]["silicon"]
    assert silicon["self_distress"] is True and silicon["distress_intensity"] > 0
    assert "needs_grounding" not in silicon and "grounding_script" not in body
    assert isinstance(body["measured_stress_rise"], float) and body["stress_error"] is None
    assert 0.0 <= body["measured_stress_rise"] <= 1.0
    assert body["stress_level"] is None and body["grounding_level"] is None
    body = client.post("/api/hear", json={"text": said, "stress_level": 0.9}).get_json()
    assert body["stress_level"] == 0.9                     # never overwritten
    assert body["grounding_level"] is None
    body = client.post("/api/hear", json={"nonverbal": {"symbol": "food"}}).get_json()
    assert body["measured_stress_rise"] is None
    assert body["stress_error"] == "no text was sent; the stress rise is read from words"


def test_install_desktop_bundle_creates_macos_app_structure(tmp_path):
    created = install_desktop_bundle(tmp_path)
    assert "macos_app" in created
    bundle = Path(created["macos_app"])
    assert (bundle / "Contents" / "Info.plist").is_file()
    assert (bundle / "Contents" / "MacOS" / "ChristmanSound").is_file()
    assert (bundle / "Contents" / "Resources" / "AppIcon.png").is_file()
    assert os.access(bundle / "Contents" / "MacOS" / "ChristmanSound", os.X_OK)


# -- the studio's door (2026-10-30) ---------------------------------------------


def test_studio_door_refuses_other_sites_and_unbounded_bodies():
    from christman_sound.app import MAX_BODY_BYTES, check_request

    ok = {"Host": "127.0.0.1:5480", "Content-Type": "application/json", "Content-Length": "2"}
    assert check_request("POST", ok, 5480) is None
    assert check_request("POST", {**ok, "Origin": "http://127.0.0.1:5480"}, 5480) is None
    assert check_request("POST", {**ok, "Origin": "https://another.site"}, 5480)[0] == 403
    assert check_request("POST", {**ok, "Host": "another.site:5480"}, 5480)[0] == 403
    assert check_request("POST", {k: v for k, v in ok.items() if k != "Content-Length"}, 5480)[0] == 411
    assert check_request("POST", {**ok, "Content-Length": str(MAX_BODY_BYTES + 1)}, 5480)[0] == 413
    assert check_request("POST", {**ok, "Content-Type": "text/plain"}, 5480)[0] == 415
    assert check_request("GET", {"Host": "localhost:5480"}, 5480) is None
    assert check_request("GET", {"Host": "[::1]:5480"}, 5480) is None


def test_studio_server_refuses_another_sites_page_over_a_real_socket():
    import http.client
    import threading

    server = create_app().make_server("127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", "/api/voice-creation-center", body=b'{"action": "speak_vcc", "text": "hi"}',
                     headers={"Content-Type": "application/json", "Origin": "https://another.site"})
        r = conn.getresponse()
        r.read()
        assert r.status == 403
        conn.request("POST", "/api/voice-creation-center", body=b"not json", headers={"Content-Type": "text/plain"})
        r = conn.getresponse()
        r.read()
        assert r.status == 415
    finally:
        server.shutdown()
        server.server_close()


def test_studio_binds_only_to_this_machine():
    with pytest.raises(ValueError):
        create_app().make_server("0.0.0.0", 0)


def test_a_voice_name_cannot_reach_outside_the_voice_cache():
    client = create_app().test_client()
    for name in ("../../elsewhere", "a/b", ".hidden"):
        r = client.post("/api/voice-creation-center",
                        json={"action": "measure_profile", "name": name, "wav_paths": ["x.wav"]})
        assert r.status_code == 400 and r.get_json()["kind"] == "UnsafeName", name


def test_one_version_everywhere():
    """2026-10-30: the version was 1.0.0 in four places and 1.1.0 in three."""
    import re as _re

    import christman_sound
    import christman_voice_sdk

    root = Path(__file__).resolve().parent.parent
    declared = _re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), _re.M).group(1)
    assert christman_sound.__version__ == declared
    assert christman_voice_sdk.__version__ == declared
    assert (root / "VERSION.txt").read_text().strip() == declared
    assert create_app().test_client().get("/api/status").get_json()["version"] == declared
