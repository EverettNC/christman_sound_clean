# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""A name a being hands in is spoken through every door, not only the mouth.

christman_sound.synth takes `pronunciations` (word -> phones in the lexicon's
alphabet): how a name the lexicon does not hold is said. Until 2026-11-05 the
doors above the mouth did not pass it on, so the same name was refused as
"not in the lexicon" through the app (/api/synthesize and the Voice Creation
Center's speak_vcc), the Voice Creation Center's engine and express lane, and
the SDK's speech synthesis engine. These checks speak a made-up word through
each door, in the test voice, on a three-word lexicon: refused without its
pronunciation, spoken with it, and the record says whose pronunciation it was.

Also here: a voice given to the app by name. The lookup imported a name the
capture client never had, so it failed every time.

WHAT CHANGED AND WHY (2026-11-05)
New, with the doors' change.
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

from voice_fixture import TEST_VOICE_NAME, write_test_voice  # noqa: E402

pytestmark = pytest.mark.needs_dsp

LEXICON = """\
hello hello h_B @_I l_I oU_E
booth booth b_B u_I T_E
this this D_B I_I s_E
"""
#: A word no lexicon holds, and its pronunciation in the lexicon's phones.
NAME = "hellobooth"
SAYS = {NAME: "h @ l oU b u T"}
TEXT = f"hello {NAME}"


@pytest.fixture
def voice(tmp_path) -> str:
    return str(write_test_voice(tmp_path / "voice"))


@pytest.fixture
def lexicon_file(tmp_path) -> str:
    path = tmp_path / "align_lexicon.txt"
    path.write_text(LEXICON, encoding="utf-8")
    return str(path)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from christman_sound.app import create_app

    monkeypatch.setenv("CHRISTMAN_VOICE_CACHE", str(tmp_path / "cache"))
    return create_app().test_client()


def _caller_record(applied) -> dict:
    (record,) = [a for a in applied if a.get("name") == "pronunciation" and a.get("word") == NAME]
    return record


# ------------------------------------------------------------------- the app


def test_app_refuses_the_name_without_its_pronunciation(client, voice):
    r = client.post("/api/synthesize", json={"text": TEXT, "voice": voice, "lexicon_text": LEXICON})
    body = r.get_json()
    assert r.status_code == 422 and body["kind"] == "WordNotInLexicon", body
    assert body["words"] == [NAME]


def test_app_speaks_the_name_with_its_pronunciation(client, voice, tmp_path):
    plain = client.post("/api/synthesize", json={"text": "hello", "voice": voice, "lexicon_text": LEXICON}).get_json()
    r = client.post("/api/synthesize", json={"text": TEXT, "voice": voice, "lexicon_text": LEXICON,
                                             "pronunciations": SAYS})
    body = r.get_json()
    assert r.status_code == 200 and body["status"] == "ok", body
    assert body["voice"] == TEST_VOICE_NAME
    assert body["duration"] > plain["duration"] + 0.2, "the name added no sound"
    assert [p["word"] for p in body["phones"] if p["word"]].count(NAME) >= 6
    record = _caller_record(body["applied"])
    assert "caller" in json.dumps(record) and "does not hold this word" in json.dumps(record)
    wav = tmp_path / "cache" / body["audio_url"].rsplit("/", 1)[1]
    with wave.open(str(wav), "rb") as w:
        assert w.getnframes() / w.getframerate() == pytest.approx(body["duration"], abs=0.01)


@pytest.mark.parametrize("handed_in, says", [
    (["hellobooth", "h @ l oU b u T"], "JSON object"),            # a list, not word -> phones
    ("h @ l oU b u T", "JSON object"),
    ({"hello booth": "h @ l oU b u T"}, "is not one word"),       # two words under one key
    ({NAME: "h @ l oU b u ZZ"}, "ZZ"),                            # a phone the lexicon's alphabet does not have
])
def test_app_refuses_pronunciations_it_cannot_use_by_name(client, voice, handed_in, says):
    r = client.post("/api/synthesize", json={"text": TEXT, "voice": voice, "lexicon_text": LEXICON,
                                             "pronunciations": handed_in})
    body = r.get_json()
    assert r.status_code == 422 and body["status"] == "refused", body
    assert says in body["reason"], body["reason"]


def test_app_refuses_pronunciations_with_phones(client, voice):
    r = client.post("/api/synthesize", json={"phones": [["HH", 0.08], ["OW", 0.2]], "voice": voice,
                                             "pronunciations": SAYS})
    body = r.get_json()
    assert r.status_code == 422 and "pronunciations are for text" in body["reason"], body


def test_app_finds_a_voice_given_by_name(client, tmp_path, monkeypatch):
    """The lookup imported PROFILES_DIR, which voice_capture_client never had:
    every voice given by name was answered 503 'the saved profiles could not
    be read'. It reads the client's profile_dir() now."""
    monkeypatch.setenv("CHRISTMAN_AI_HOME", str(tmp_path / "data"))
    from christman_voice_sdk.integration.voice_capture_client import profile_dir

    assert profile_dir() == tmp_path / "data" / "voice_profiles"
    write_test_voice(profile_dir()).rename(profile_dir() / "a_saved_voice.json")
    r = client.post("/api/synthesize", json={"text": "hello", "voice": "a_saved_voice", "lexicon_text": LEXICON})
    body = r.get_json()
    assert r.status_code == 200 and body["voice"] == TEST_VOICE_NAME, body
    # A name with no file under it still ends plainly, as no voice.
    r = client.post("/api/synthesize", json={"text": "hello", "voice": "nobody_saved_this", "lexicon_text": LEXICON})
    assert r.status_code in (422, 503) and r.get_json()["kind"] != "ProfilesUnavailable", r.get_json()
    source = (ROOT / "christman_sound" / "app.py").read_text(encoding="utf-8")
    assert "import PROFILES_DIR" not in source


def test_app_center_door_takes_the_name(client, lexicon_file):
    frame_voice = os.environ.get("VCC_TEST_FRAME_VOICE")
    if not frame_voice:
        pytest.skip("NOT RUN: no frame voice was handed in; set VCC_TEST_FRAME_VOICE to a frame voice folder "
                    "(the Center's speak_vcc applies controls, which only a frame voice takes)")
    ask = {"action": "speak_vcc", "voice": frame_voice, "text": TEXT, "lexicon_text": LEXICON}
    refused = client.post("/api/voice-creation-center", json=ask)
    assert refused.status_code != 200 and NAME in json.dumps(refused.get_json()), refused.get_json()
    r = client.post("/api/voice-creation-center", json={**ask, "pronunciations": SAYS})
    body = r.get_json()
    assert r.status_code == 200 and body["status"] == "ok", body
    _caller_record(body["applied"])
    bad = client.post("/api/voice-creation-center", json={**ask, "pronunciations": ["not", "an", "object"]})
    assert bad.status_code == 422 and "JSON object" in bad.get_json()["error"]


# ------------------------------------------- the Voice Creation Center's engine


@pytest.fixture
def center_engine(voice, lexicon_file):
    from voice_creation_center.engines.voice_engine import VoiceEngine

    engine = VoiceEngine(voice, lexicon_path=lexicon_file)
    assert engine.is_ready, engine.not_ready_reason
    return engine


def test_center_engine_refuses_the_name_without_its_pronunciation(center_engine):
    result = center_engine.synthesize(TEXT)
    assert result.success is False and "WordNotInLexicon" in result.error and NAME in result.error


def test_center_engine_speaks_the_name_with_its_pronunciation(center_engine, tmp_path):
    plain = center_engine.synthesize("hello")
    out = tmp_path / "said.wav"
    result = center_engine.synthesize(TEXT, output_path=str(out), pronunciations=SAYS)
    assert result.success is True, result.error
    assert result.voice == TEST_VOICE_NAME and out.is_file()
    assert result.duration_seconds > plain.duration_seconds + 0.2


def test_center_engine_names_a_pronunciation_it_cannot_use(center_engine):
    result = center_engine.synthesize(TEXT, pronunciations={NAME: "h @ l oU b u ZZ"})
    assert result.success is False and "ZZ" in result.error


def test_center_express_lane_carries_the_name(center_engine, tmp_path):
    from voice_creation_center.voice_express import Priority, VoiceExpress

    express = VoiceExpress(base_dir=str(tmp_path / "center"))
    assert express.load()
    being = "being under test"
    without = express.priority_synthesize(TEXT, being)
    express.process_next({being: center_engine})
    assert without.done and without.audio_path is None and NAME in (without.error or "")

    first = express.priority_synthesize(TEXT, being, priority=Priority.CRITICAL, pronunciations=SAYS)
    assert first.pronunciations == SAYS
    express.process_next({being: center_engine})
    assert first.done and first.error is None and Path(first.audio_path).is_file(), first.error

    # A request that hands in its own words is rendered with them, not answered from the cache.
    other = express.priority_synthesize(TEXT, being, pronunciations={NAME: "h @ l oU b u s"})
    calls = []
    real = center_engine.synthesize
    center_engine.synthesize = lambda *a, **k: calls.append(k) or real(*a, **k)
    try:
        express.process_next({being: center_engine})
    finally:
        center_engine.synthesize = real
    assert other.done and other.error is None
    assert [c.get("pronunciations") for c in calls] == [{NAME: "h @ l oU b u s"}]


# ------------------------------------------------- the SDK's synthesis engine


def test_sdk_engine_speaks_the_name_with_its_pronunciation(voice, lexicon_file, tmp_path):
    from christman_sound.lexicon import WordNotInLexicon
    from christman_voice_sdk.synthesis.voice_synthesis import SpeechSynthesisEngine

    engine = SpeechSynthesisEngine(cache_dir=str(tmp_path / "cache"), lexicon_path=lexicon_file, voice=voice)
    with pytest.raises(WordNotInLexicon) as caught:
        engine.generate_speech_audio(TEXT, play_audio=False)
    assert caught.value.words == [NAME]
    path = engine.generate_speech_audio(TEXT, play_audio=False, pronunciations=SAYS)
    assert path and Path(path).is_file()
    with wave.open(str(path), "rb") as w:
        assert w.getnframes() / w.getframerate() > 0.5
    assert engine.last_voice == TEST_VOICE_NAME
