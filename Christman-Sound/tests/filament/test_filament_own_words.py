# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""The owner's own words: a name the ear mishears comes back as it was said.

The package's mouth names no one (ruling of 2026-10-30). The rows that turn
the ear's spellings of a person's name back into the name live in a file the
owner keeps, $FILAMENT_HOME/mouth.json, and the Filament ear server applies
them to every tape and every live reply. These checks run the real mouth and
the real door, with no model needed, on a folder made for the test. The name
used here is made up; no person's name is in this file or in the package.

WHAT CHANGED AND WHY (2026-11-05)
New. Before this, a Filament run from the package had nowhere to learn its
owner's name, and wrote it the way the ear mishears it.
"""

from __future__ import annotations

import json

import pytest

from filament_helpers import ask, running_door

from christman_sound.ear import mouth

ROWS = [
    {"heard": "quill father", "keep": "Quillfeather"},
    {"heard": "quillfether", "keep": "Quillfeather"},
    {"heard": "fixin'", "keep": "fixin'"},
]


@pytest.fixture
def home(tmp_path, monkeypatch):
    folder = tmp_path / "filament"
    folder.mkdir()
    monkeypatch.setenv("FILAMENT_HOME", str(folder))
    return folder


@pytest.fixture
def server(home):
    from christman_sound.filament import server as srv

    return srv


def _teach(home, rows) -> None:
    (home / "mouth.json").write_text(json.dumps(rows), encoding="utf-8")


def _vosk_shaped(text: str) -> dict:
    words = [{"conf": 1.0, "start": 0.1 * n, "end": 0.1 * n + 0.09, "word": w} for n, w in enumerate(text.split())]
    return {"text": text, "result": words}


def _tape_body(payload: dict) -> dict:
    text = payload["text"]
    return {"ok": True, "text": text, "transcript": text, "words": [], "duration": 0.7, "language": "en",
            "engine": "vosk", "organ": "filament", "cloud": False, "seated": True}


def test_the_file_sits_beside_the_carbon_pile(home):
    assert mouth.own_words_path() == home / "mouth.json"
    assert mouth.own_words_path().parent == mouth.carbon_path().parent


def test_no_file_is_no_rows_and_nothing_is_written(home):
    assert mouth.load_own_mangles() == ()
    assert list(home.iterdir()) == []


def test_the_package_rows_still_name_no_one():
    # Every row the package itself holds keeps a lower-case word: none is a name.
    assert all(keep == keep.lower() for _pat, keep in mouth.MANGLES)


def test_a_row_is_matched_whole_in_any_case_across_any_spaces(home):
    _teach(home, ROWS)
    own = mouth.load_own_mangles()
    assert len(own) == 3
    said, hits = mouth.recover("ask Quill   FATHER and quillfether", extra_mangles=own)
    assert said == "ask Quillfeather and Quillfeather"
    assert [h["to"] for h in hits] == ["Quillfeather", "Quillfeather"]
    # Whole words only: a longer word that starts the same way is left alone.
    assert mouth.recover("the quill fathers met", extra_mangles=own)[0] == "the quill fathers met"
    assert mouth.recover("quillfethers", extra_mangles=own)[0] == "quillfethers"


def test_without_the_file_the_name_stays_as_the_ear_wrote_it(home):
    assert mouth.recover("ask quill father", extra_mangles=mouth.load_own_mangles())[0] == "ask quill father"


def test_an_edit_to_the_file_is_read_on_the_next_call(home):
    _teach(home, ROWS[:1])
    assert len(mouth.load_own_mangles()) == 1
    _teach(home, ROWS)
    assert len(mouth.load_own_mangles()) == 3
    (home / "mouth.json").unlink()
    assert mouth.load_own_mangles() == ()


@pytest.mark.parametrize("text, names", [
    ("{\"heard\":", ["mouth.json", "not readable JSON"]),
    ("{\"heard\": \"a\", \"keep\": \"b\"}", ["mouth.json", "dict, not a list"]),
    ("[{\"heard\": \"quill father\", \"keep\": \"Quillfeather\"}, {\"heard\": \"\", \"keep\": \"x\"}]",
     ["mouth.json", "Row 2"]),
    ("[{\"heard\": \"quill father\"}]", ["mouth.json", "Row 1"]),
    ("[\"quill father\"]", ["mouth.json", "Row 1"]),
    ("[{\"heard\": \"quill father\", \"keep\": 7}]", ["mouth.json", "Row 1"]),
])
def test_a_broken_file_raises_by_name_and_is_left_as_it_is(home, text, names):
    (home / "mouth.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        mouth.load_own_mangles()
    for piece in names:
        assert piece in str(caught.value)
    assert (home / "mouth.json").read_text(encoding="utf-8") == text


def test_the_server_writes_the_name_as_it_was_said(server, home):
    _teach(home, ROWS)
    heard = _vosk_shaped("this is quill father speaking")
    out = server._dress(_tape_body(heard), heard)
    assert out["ok"] is True
    assert out["text"] == "this is Quillfeather speaking" and out["transcript"] == out["text"]
    assert [h["to"] for h in out["mouth"]] == ["Quillfeather"]
    assert "mouth_error" not in out
    # One word the ear wrote as one token is recovered in the word list too.
    heard = _vosk_shaped("hello quillfether")
    out = server._dress(_tape_body(heard), heard)
    assert out["text"] == "hello Quillfeather"
    assert [(w["raw"], w["text"]) for w in out["words"]] == [("hello", "hello"), ("quillfether", "Quillfeather")]


def test_the_server_without_the_file_is_as_it_was(server, home):
    heard = _vosk_shaped("this is quill father speaking")
    out = server._dress(_tape_body(heard), heard)
    assert out["text"] == "this is quill father speaking" and out["mouth"] == []
    assert "mouth_error" not in out


def test_a_live_reply_carries_the_name_too(server, home):
    _teach(home, ROWS)
    heard = _vosk_shaped("quill father is here")
    packet = server._live_packet(True, text=heard["text"], payload=heard, job=3)
    assert packet["final"] is True and packet["text"] == "Quillfeather is here"
    partial = server._live_packet(False, partial="quill father", payload={"partial": "quill father"})
    assert partial["partial"] == "Quillfeather"


def test_a_broken_file_does_not_cost_the_words_heard(server, home, capsys):
    (home / "mouth.json").write_text("{oops", encoding="utf-8")
    heard = _vosk_shaped("this is quill father speaking")
    out = server._dress(_tape_body(heard), heard)
    assert out["ok"] is True and out["text"] == "this is quill father speaking"
    assert [w["raw"] for w in out["words"]] == heard["text"].split()
    assert "mouth.json" in out["mouth_error"] and "not applied" in out["mouth_error"]
    assert "mouth.json" in capsys.readouterr().err
    assert (home / "mouth.json").read_text(encoding="utf-8") == "{oops"


def test_health_says_how_many_rows_were_read_and_when_the_file_is_broken(server, home, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MODEL_PATH", str(tmp_path / "no-model-here"))
    with running_door(server) as port:
        code, _h, body = ask(port, "GET", "/health")
        assert code == 200 and json.loads(body)["ownWords"] == 0 and "ownWordsError" not in json.loads(body)
        _teach(home, ROWS)
        got = json.loads(ask(port, "GET", "/health")[2])
        assert got["ownWords"] == 3 and "ownWordsError" not in got
        (home / "mouth.json").write_text("[1]", encoding="utf-8")
        got = json.loads(ask(port, "GET", "/health")[2])
        assert got["ownWords"] == 0 and "mouth.json" in got["ownWordsError"]
