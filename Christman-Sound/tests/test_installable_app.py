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

    # 4. Forbidden being reply -> 422 ForbiddenReply
    forb = client.post("/api/synthesize", json={
        "text": "Calm down, you're not okay.",
        "voice": str(vpath),
        "lexicon_text": FIXTURE_LEXICON,
        "being_reply": True,
    })
    assert forb.status_code == 422
    assert forb.get_json()["kind"] == "ForbiddenReply"

    # 5. Measure WAV endpoint on rendered WAV
    meas = client.post("/api/measure-wav", json={"path": str(wav_file)})
    assert meas.status_code == 200, meas.get_json()
    mdata = meas.get_json()
    assert mdata["sample_rate"] == 16000
    assert mdata["rms"] > 0.04
    assert mdata["pitch_contour"]["voiced_frames"] > 0


def test_api_hear_crisis_and_nonverbal():
    app = create_app()
    client = app.test_client()

    r = client.post("/api/hear", json={"text": "I'm not okay"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["paged_human"] is True
    assert body["turn"]["decision"]["safety"]["reason"] == "client_distress"

    r_sym = client.post("/api/hear", json={"nonverbal": {"symbol": "food"}})
    assert r_sym.status_code == 200
    assert r_sym.get_json()["turn"]["person_message"] == "I'm hungry."


def test_install_desktop_bundle_creates_macos_app_structure(tmp_path):
    created = install_desktop_bundle(tmp_path)
    assert "macos_app" in created
    bundle = Path(created["macos_app"])
    assert (bundle / "Contents" / "Info.plist").is_file()
    assert (bundle / "Contents" / "MacOS" / "ChristmanSound").is_file()
    assert (bundle / "Contents" / "Resources" / "AppIcon.png").is_file()
    assert os.access(bundle / "Contents" / "MacOS" / "ChristmanSound", os.X_OK)
