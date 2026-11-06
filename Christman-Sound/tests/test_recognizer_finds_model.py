# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""The SDK's recognizer finds the model folder where the Filament finds it.

Before 2026-11-05 the engine read VOSK_MODEL_PATH and nothing else: on a Mac
with vosk-model-en-us-0.22 in ~/Downloads and nothing set, the Filament ear
server and the studio heard and this engine (and the word-ear receiver built
on it) reported no model. These checks run the real engine and the real
finder against folders made for the test. Which folder is chosen needs no
recognizer, so loading the model is the one step stood in for; hearing through
the model that is found is checked where the model is (test_rulings and the
Filament live tests).

WHAT CHANGED AND WHY (2026-11-05)
New, with the engine's change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.audio import speech_recognition_engine as sre  # noqa: E402

FULL = "vosk-model-en-us-0.22"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty home folder, no model setting, and no model read from disk."""
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("HOME", str(folder))
    monkeypatch.setenv("USERPROFILE", str(folder))  # Windows: Path.home() reads this
    monkeypatch.delenv(sre.VOSK_MODEL_PATH_ENV, raising=False)
    assert Path.home() == folder
    loaded = []
    monkeypatch.setattr(sre.EarBackend, "_ensure_model", lambda self: loaded.append(self.model_path))
    return folder


def _folder(path: Path) -> str:
    path.mkdir(parents=True)
    return str(path)


def test_nothing_set_and_no_folder_anywhere_says_where_it_looked(home):
    with pytest.raises(sre.LocalRecognitionUnavailable) as caught:
        sre.EarBackend()
    said = str(caught.value)
    for piece in ("No model folder", "model_path", "VOSK_MODEL_PATH", "~/Downloads", "~/models", FULL):
        assert piece in said, piece
    engine = sre.SpeechRecognitionEngine()
    assert engine.available is False and engine.unavailable_reason == said
    result = engine.recognize_from_bytes(b"\x00\x00" * 1600, sample_rate=16000)
    assert result.status.value == "unavailable" and result.text == ""


def test_nothing_set_finds_the_model_in_downloads(home):
    model = _folder(home / "Downloads" / FULL)
    backend = sre.EarBackend()
    assert backend.model_path == model and backend.model_found_by == "discover_vosk_model"
    engine = sre.SpeechRecognitionEngine()
    assert engine.available is True and engine.unavailable_reason is None
    assert engine.backend.model_path == model


def test_nothing_set_finds_the_model_in_models(home):
    model = _folder(home / "models" / FULL)
    assert sre.EarBackend().model_path == model


def test_the_full_model_is_taken_ahead_of_the_lgraph_one(home):
    _folder(home / "Downloads" / (FULL + "-lgraph"))
    full = _folder(home / "models" / FULL)
    assert sre.EarBackend().model_path == full


def test_the_setting_wins_over_a_folder_that_would_be_found(home, tmp_path, monkeypatch):
    _folder(home / "Downloads" / FULL)
    named = _folder(tmp_path / "named-model")
    monkeypatch.setenv(sre.VOSK_MODEL_PATH_ENV, named)
    backend = sre.EarBackend()
    assert backend.model_path == named and backend.model_found_by == "VOSK_MODEL_PATH"


def test_a_setting_that_points_at_nothing_is_an_error_not_passed_over(home, tmp_path, monkeypatch):
    _folder(home / "Downloads" / FULL)
    monkeypatch.setenv(sre.VOSK_MODEL_PATH_ENV, str(tmp_path / "not-there"))
    with pytest.raises(sre.LocalRecognitionUnavailable) as caught:
        sre.EarBackend()
    assert "not-there" in str(caught.value) and "VOSK_MODEL_PATH" in str(caught.value)


def test_a_path_passed_in_wins_over_both(home, tmp_path, monkeypatch):
    _folder(home / "Downloads" / FULL)
    monkeypatch.setenv(sre.VOSK_MODEL_PATH_ENV, _folder(tmp_path / "named-model"))
    passed = _folder(tmp_path / "passed-model")
    backend = sre.EarBackend(passed)
    assert backend.model_path == passed and backend.model_found_by == "model_path"


def test_the_engine_and_the_filament_find_the_same_folder(home):
    from christman_sound.ear import discover_vosk_model
    from christman_sound.filament import server

    model = _folder(home / "models" / FULL)
    assert str(discover_vosk_model()) == model == server.resolve_model_path() == sre.EarBackend().model_path


def test_word_ear_health_names_the_model_and_how_it_was_found(home):
    from christman_voice_sdk.audio.word_ear_receiver import WordEarReceiver

    health = WordEarReceiver().health()
    assert health["engine_available"] is False and health["model"] is None and health["model_found_by"] is None
    assert "~/Downloads" in health["unavailable_reason"]
    model = _folder(home / "Downloads" / FULL)
    health = WordEarReceiver().health()
    assert health["engine_available"] is True and health["unavailable_reason"] is None
    assert health["model"] == model and health["model_found_by"] == "discover_vosk_model"
