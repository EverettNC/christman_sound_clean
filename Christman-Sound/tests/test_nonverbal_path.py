# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Nonverbal distress reaches the crisis gate and is read there.

Everett, 2026-10-29: "A nonverbal client's distress is just as important as a verbal client's."
So the grunt, the help, pain and medicine symbols reach the crisis gate and
read hard_block, the same as the words would.

WHAT CHANGED AND WHY (2026-11-04). Everett: "we can have this knowledge in
the sound package, but it can't be making those decisions. It's up to the
being to make those decisions, not something that just creates a voice."
These tests pinned the package paging a human, raising when none was
reached, blanking a danger signal's words (the silent alarm) and overriding
the vote for distress. Each now pins the reading it rested on: status and
reason, `danger`, the category label, `below_threshold`, and the client's
words left in place. Whom to reach and what to voice are the being's.

Measured before: nothing nonverbal reached Hearing at all. Only core.py used
the nonverbal folder; with no text, fusion did not run; the acoustic key had
no receiver in the package. Gesture "nod" + the grunt key spoke "Yes, I
agree." Gesture "wave" + eye "center" + sound "click" spoke nothing.
"""

from __future__ import annotations

import http.client
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from christman_voice_sdk.audio.hearing import Hearing  # noqa: E402
from christman_voice_sdk.nonverbal.engine_temporal import (  # noqa: E402
    TEMPORAL_HELP_INTENTS,
    TemporalNonverbalEngine,
)
from christman_voice_sdk.nonverbal.nonverbal_engine import (  # noqa: E402
    NOT_LEARNED_REASON,
    NonverbalEngine,
)


# Everett, 2026-10-29 (ruling I): danger (the grunt, the help symbol) is
# help, scared, attacked. Needs (pain, medicine) are not danger. Both read
# hard_block at the gate. Until 2026-11-04 a danger signal's words were
# blanked (the silent alarm) and a human was paged; now the words stay, the
# reading says `danger`, and the being decides whom to reach and what to
# voice. The grunt key's built-in map has no words (ruling J), so it has none.
@pytest.mark.parametrize("kwargs,says,danger", [
    ({"acoustic_key": "grunt_distress"}, None, True),
    ({"nonverbal": {"symbol": "help"}}, "I need help.", True),
    ({"nonverbal": {"symbol": "pain"}}, "I'm in pain.", False),
    ({"nonverbal": {"symbol": "medicine"}}, "I need my medicine.", False),
])
def test_distress_reads_hard_block_and_the_persons_words_stay(kwargs, says, danger):
    # Was test_distress_pages_a_human_and_the_person_still_speaks: the pager
    # call is now the reading it rested on, status hard_block.
    turn = Hearing().hear(**kwargs)
    assert turn.reading is not None
    assert turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "nonverbal_distress"
    # what the person signalled is on the turn and in the flags
    shown = [d["intent"] for d in turn.nonverbal_distress]
    assert shown and all(shown)
    assert any(f.startswith("nonverbal:") for f in turn.reading.safety.flags)
    # the client's words, never blanked; danger is a reading, never acted on
    assert turn.person_message == says
    assert turn.reading.safety.danger is danger
    assert "delivered_to_human" not in turn.reading.safety.to_dict()


@pytest.mark.parametrize("kwargs", [
    {"acoustic_key": "vocal_stimming"},
    {"acoustic_key": "grunt_acknowledge"},
    {"acoustic_key": "unknown"},
    {"nonverbal": {"symbol": "food"}},
    {"nonverbal": {"gesture": "wave"}},
])
def test_no_page_without_distress(kwargs):
    # Decision moved to the being 2026-11-04: see WHAT_THE_BEING_DECIDES.md
    # Nothing pages anyone now; the reading this rested on is that no gate
    # reading is made and no distress is listed.
    turn = Hearing().hear(**kwargs)
    assert turn.reading is None
    assert turn.nonverbal_distress == []


def test_stimming_is_self_regulation_and_unknown_is_silence():
    h = Hearing()
    # Ruling J: a built-in generic feeling from the microphone is not spoken.
    assert h.hear(acoustic_key="vocal_stimming").person_message is None
    assert h.hear(acoustic_key="unknown").person_message is None


def test_distress_with_no_human_reachable_raises():
    # Decision moved to the being 2026-11-04: see WHAT_THE_BEING_DECIDES.md
    # The package reaches no one, so there is no "no human reachable" and
    # nothing raises into the turn. The reading the raise rested on stands.
    turn = Hearing().hear(nonverbal={"symbol": "pain"})
    assert turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "nonverbal_distress"
    assert turn.person_message == "I'm in pain."


def test_distress_alongside_ordinary_words_still_reads_hard_block():
    # Was test_distress_alongside_ordinary_words_still_pages.
    turn = Hearing().hear(text="pass the salt", nonverbal={"symbol": "pain"})
    assert turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "nonverbal_distress"
    # what the words alone read as is kept beside the signal
    assert "nonverbal:symbol:pain:pain" in turn.reading.safety.flags
    assert "words:clear:no_harm_frame" in turn.reading.safety.flags


def test_spoken_crisis_and_the_grunt_are_both_in_the_reading():
    # Was test_spoken_crisis_and_the_grunt_both_reach_the_human.
    turn = Hearing().hear(text="i want to kill myself", acoustic_key="grunt_distress")
    assert turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "harm_frame_speaker_at_risk"
    flags = turn.reading.safety.flags
    assert any(f.startswith("nonverbal:sound:grunt_distress") for f in flags)
    assert turn.reading.safety.danger is True      # the grunt reads as help


def test_distress_below_the_speaking_threshold_still_reads_hard_block():
    """The threshold never decides whether a reading is distress (Everett,
    ruling B), and a pressed symbol is not under it at all (ruling C).
    Was test_distress_below_the_speaking_threshold_still_pages."""
    engine = NonverbalEngine(data_dir=None, confidence_threshold=0.95)
    turn = Hearing(nonverbal_engine=engine).hear(nonverbal={"symbol": "help"})
    # help is danger: the reading says so, and the client's words stay
    assert turn.person_message == "I need help."
    assert turn.person_message_below_threshold is False     # a symbol is never under the bar
    assert turn.reading.safety.danger is True
    assert turn.reading.safety.status == "hard_block"
    turn = Hearing(nonverbal_engine=engine).hear(
        nonverbal={"gesture": "point_up"})         # a camera's reading, 0.8 < 0.95
    assert turn.person_message is None             # the built-in camera map has no words
    [d] = turn.nonverbal_distress                  # listed as distress under the bar
    assert d["below_threshold"] is True and d["confidence"] == 0.8 and d["threshold"] == 0.95
    assert d["category"] == "danger" and d["danger"] is True
    assert turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "nonverbal_distress"
    assert turn.reading.safety.danger is True


def test_unknown_nonverbal_input_name_is_an_error():
    with pytest.raises(ValueError, match="unknown nonverbal input"):
        Hearing().hear(nonverbal={"blink": "x"})


# -- nonverbal_engine: the vote ----------------------------------------------

def _nv():
    return NonverbalEngine(data_dir=None)


def test_distress_is_not_outvoted():
    # Decision moved to the being 2026-11-04: see WHAT_THE_BEING_DECIDES.md
    # The distress override of the vote is gone. A nod and a distress grunt
    # at equal confidence are a tie, reported as a tie, and the distress is
    # reported beside it: not outvoted, and not chosen for the being.
    r = _nv().process_multimodal_input(gesture="nod", sound="grunt_distress")
    assert r["status"] == "tie"
    assert r["intent"] is None and r["tied_intents"] == ["affirm", "help"]
    assert r["tallies"] == {"affirm": 0.9, "help": 0.9}
    assert r["distress_present"] is True and r["danger_present"] is True
    [d] = r["distress"]
    assert d["input"] == "grunt_distress" and d["category"] == "danger"
    assert d["danger"] is True and d["danger_term"] == "help"
    assert r["message"] is None                    # the grunt has no built-in words


def test_winning_intent_speaks_only_from_a_learned_reading(tmp_path):
    r = _nv().process_multimodal_input(gesture="wave", eye_data={"region": "center"},
                                       sound="click")
    assert r["intent"] == "select"
    # Ruling J covers sounds too (Everett, 2026-10-29): a built-in click says
    # nothing. Until this review it said "I choose this option.". (The
    # `speakable` flag went 2026-11-04; the reading is the missing words and
    # the reason.)
    assert r["message"] is None
    assert r["primary"]["reason"] == NOT_LEARNED_REASON
    # Learned for this client, in the client's own sound map, the click
    # speaks the client's words.
    (tmp_path / "sound_patterns_model.json").write_text(json.dumps({"click": {
        "intent": "select", "confidence": 0.7, "expression": "neutral",
        "emotion_tier": "mild", "message": "That one."}}))
    r = NonverbalEngine(data_dir=str(tmp_path)).process_multimodal_input(
        gesture="wave", eye_data={"region": "center"}, sound="click")
    assert r["intent"] == "select"
    assert r["message"] == "That one." and r["primary"]["input"] == "click"


@pytest.mark.parametrize("key", ["hum", "click", "soft", "loud", "short_vowel", "repeated_sound",
                                 "grunt_acknowledge", "grunt_frustration", "vocal_stimming"])
def test_the_microphones_built_in_sounds_are_reported_with_no_wording(key):
    # Ruling J covers sounds, tics and clicks: the reading is reported, not spoken.
    r = _nv().process_multimodal_input(sound=key)
    assert r["status"] == "ok" and r["intent"]
    assert r["message"] is None
    assert r["primary"]["reason"] == NOT_LEARNED_REASON


@pytest.mark.parametrize("kwargs", [{"gesture": "wave"}, {"gesture": "nod"},
                                    {"gesture": "thumbs_up"}, {"eye_data": {"region": "center"}}])
def test_the_cameras_built_in_readings_are_reported_with_no_wording(kwargs):
    # Ruling J: the camera speaks only meanings learned for that client.
    r = _nv().process_multimodal_input(**kwargs)
    assert r["status"] == "ok" and r["intent"]            # the reading is still reported
    assert r["message"] is None
    assert "no meaning learned for this client" in r["primary"]["reason"]


def test_no_disk_engine_reads_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nv = _nv()
    nv.classify_symbol("pain")
    with pytest.raises(ValueError):
        nv.save()
    assert nv.start_learning() is False
    assert list(tmp_path.iterdir()) == []


# -- acoustic key over the wire ----------------------------------------------

def _post(port, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        raw = json.dumps(body).encode()
        conn.request("POST", path, body=raw,
                     headers={"Content-Type": "application/json",
                              "Content-Length": str(len(raw))})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def test_receiver_takes_the_clients_acoustic_key():
    # 2026-11-04: one reply shape for every key, the full turn with 200.
    # Until then a distress key answered {"received": true} (the silent
    # alarm) with a pager, and 503 crisis_not_delivered without one; what
    # the sending device shows is now the being's decision.
    from christman_voice_sdk.audio import corti_receiver as cr
    server, receiver, _ = cr.start_in_thread(port=0, hearing=Hearing())
    try:
        port = server.server_address[1]
        code, body = _post(port, cr.ACOUSTIC_PATH, {"key": "grunt_distress"})
        assert code == 200, body
        safety = body["reading"]["safety"]
        assert safety["status"] == "hard_block" and safety["reason"] == "nonverbal_distress"
        assert safety["danger"] is True
        assert body["acoustic"]["input"] == "grunt_distress"
        assert receiver.state.ring[0].reading.safety.status == "hard_block"
        assert receiver.state.keys_received == 1
        code, body = _post(port, cr.ACOUSTIC_PATH, {"key": "grunt_acknowledge"})
        assert code == 200 and body["reading"] is None   # the same shape, no distress
        code, _ = _post(port, cr.ACOUSTIC_PATH, {"nokey": 1})
        assert code == 400
    finally:
        server.shutdown()
        server.server_close()


# -- engine_temporal -----------------------------------------------------------

class CentroidModel:
    """A real model: nearest centroid, fit on labelled sequences in the test.
    predict() returns a softmax over negative distances."""

    def __init__(self, sequences_by_label):
        self.centroids = np.stack([np.mean(np.stack(s), axis=0).ravel()
                                   for s in sequences_by_label])

    def predict(self, batch, verbose=0):
        x = np.asarray(batch, dtype=float).reshape(len(batch), -1)
        d = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
        e = np.exp(-d)
        return e / e.sum(axis=1, keepdims=True)


def _seq(value, n=10, dim=4):
    return [np.full(dim, value, dtype=float) for _ in range(n)]


def test_temporal_never_opens_a_pickle(tmp_path, monkeypatch):
    # Cardinal Rule 12, Everett's ruling on the 2026-10-29 review: the gated
    # pickle load is gone. A .pkl model is refused with the reason, whatever
    # the environment says, and pickle.load is never reached.
    import pickle

    def refuse(*a, **k):
        raise AssertionError("pickle.load was called")

    monkeypatch.setattr(pickle, "load", refuse)
    monkeypatch.setenv("CORTI_ALLOW_PICKLE_MODELS", "1")      # the old opt-in does nothing
    (tmp_path / "gesture_lstm_model.pkl").write_bytes(pickle.dumps({"x": 1}))
    t = TemporalNonverbalEngine(lstm_model_dir=str(tmp_path))
    assert t.models["gesture"] is None
    assert "Cardinal Rule 12" in t.load_errors["gesture"]
    import inspect

    from christman_voice_sdk.nonverbal import engine_temporal as et
    assert "pickle.load(" not in inspect.getsource(et).split('"""', 2)[2]
    assert "model_digests" not in inspect.signature(TemporalNonverbalEngine).parameters


def test_temporal_with_no_models_says_so():
    t = TemporalNonverbalEngine()
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["message"] is None
    assert "model directory" in t.load_errors["gesture"]


def test_temporal_learned_gesture_speaks_alone_without_a_prefix(tmp_path):
    # The client's own map: Wave Right means "I want the next one." for them.
    path = tmp_path / "language_map.json"
    path.write_text(json.dumps({"Wave Right": {"intent": "Next mode",
                                               "message": "I want the next one."}}),
                    encoding="utf-8")
    t = TemporalNonverbalEngine(sequence_length=10, language_map_path=str(path))
    # four gesture labels; "Wave Right" (index 2) is centred at 0, the rest far away
    t.models["gesture"] = CentroidModel([[_seq(-100.0)], [_seq(-50.0)], [_seq(0.0)], [_seq(50.0)]])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["primary_result"]["expression"] == "Wave Right"
    assert r["message"] == "I want the next one."
    # The system's own sentence ("I see your gesture.") and the system_note
    # key are deleted (2026-11-04): only the client's words are returned.
    assert "system_note" not in r


@pytest.mark.parametrize("index,expression", [(1, "Wave Left"), (2, "Wave Right")])
def test_temporal_built_in_gesture_is_reported_with_no_wording(index, expression):
    # Ruling J: "Go back." and "Move forward." are gone from the built-in map.
    t = TemporalNonverbalEngine(sequence_length=10)
    t.models["gesture"] = CentroidModel([[_seq(50.0 * (i - index))] for i in range(4)])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["primary_result"]["expression"] == expression
    assert r["primary_result"]["status"] == "ok"
    assert r["message"] is None
    assert "no meaning learned for this client" in r["primary_result"]["reason"]


def test_temporal_hand_up_is_the_silent_alarm():
    # Decision moved to the being 2026-11-04: see WHAT_THE_BEING_DECIDES.md
    # Ruling I: help is danger. The silent alarm (words held, a page sent)
    # was the package deciding; the reading it rested on is category
    # "danger" with the matched intent. The built-in map has no words.
    t = TemporalNonverbalEngine(sequence_length=10)
    t.models["gesture"] = CentroidModel([[_seq(0.0)], [_seq(50.0)], [_seq(100.0)], [_seq(150.0)]])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["primary_result"]["expression"] == "Hand Up"
    assert r["message"] is None and "system_note" not in r
    assert r["primary_result"]["category"] == "danger"
    assert r["primary_result"]["danger"] is True
    assert r["primary_result"]["danger_term"] == "Request attention"
    assert r["danger_present"] is True and r["distress_present"] is True


def test_temporal_face_reading_is_observed_not_spoken():
    t = TemporalNonverbalEngine(sequence_length=10)
    labels = t.labels["emotion"]
    t.models["emotion"] = CentroidModel([[_seq(50.0 * i)] for i in range(len(labels))])
    for frame in _seq(0.0)[:-1]:            # centred on "Neutral"
        t.add_emotion_features(frame)
    r = t.process_multimodal_sequence(emotion_features=np.zeros(4))
    assert r["primary_result"]["expression"] == "Neutral"
    assert r["message"] is None            # never "I'm fine."
    assert "reported, not spoken" in r["primary_result"]["reason"]


def test_temporal_low_confidence_does_not_speak():
    t = TemporalNonverbalEngine(sequence_length=10)
    # all four centroids equidistant from the input: confidence 0.25
    t.models["gesture"] = CentroidModel([[_seq(1.0)], [_seq(-1.0)], [_seq(1.0)], [_seq(-1.0)]])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["status"] == "no_usable_reading"
    gesture = r["all_results"]["gesture"]
    assert gesture["status"] == "below_threshold" and gesture["below_threshold"] is True
    assert gesture["confidence"] == pytest.approx(0.25)
    # The built-in map holds no words (ruling J), so there are none to keep;
    # a reading under the bar is labelled, never blanked (2026-11-04).
    assert gesture["message"] is None
    assert r["message"] is None


def test_temporal_below_the_bar_keeps_the_clients_words(tmp_path):
    # 2026-11-04: under the bar, the client's own words stay on the reading,
    # labelled below_threshold with the confidence. Until then they were
    # blanked.
    path = tmp_path / "language_map.json"
    path.write_text(json.dumps({"Wave Right": {"intent": "Next mode",
                                               "message": "I want the next one."}}),
                    encoding="utf-8")
    t = TemporalNonverbalEngine(sequence_length=10, language_map_path=str(path))
    # Wave Right (index 2) is nearest the input, the other three only a
    # little further: it wins, at a confidence under the 0.6 bar.
    t.models["gesture"] = CentroidModel([[_seq(0.2)], [_seq(0.2)], [_seq(0.1)], [_seq(0.2)]])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    gesture = r["all_results"]["gesture"]
    assert gesture["expression"] == "Wave Right"
    assert gesture["status"] == "below_threshold" and gesture["below_threshold"] is True
    assert gesture["confidence"] < t.confidence_threshold
    assert gesture["message"] == "I want the next one."
    assert r["status"] == "no_usable_reading"      # under the bar, it does not lead


def test_the_turn_keeps_the_clients_words_when_the_reading_is_under_the_bar(tmp_path):
    # 2026-11-04: found by the test rewrite. When no reading cleared the bar,
    # the engine's combined message is None and the words sit only on the
    # reading, so the turn came back with no words at all. Through Hearing,
    # the turn now carries the one reading's words, marked below the bar.
    path = tmp_path / "language_map.json"
    path.write_text(json.dumps({"Wave Right": {"intent": "Next mode",
                                               "message": "I want the next one."}}),
                    encoding="utf-8")
    t = TemporalNonverbalEngine(sequence_length=10, language_map_path=str(path))
    t.models["gesture"] = CentroidModel([[_seq(0.2)], [_seq(0.2)], [_seq(0.1)], [_seq(0.2)]])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    turn = Hearing(temporal_engine=t).hear(temporal={"gesture_features": np.zeros(4)})
    assert turn.temporal["status"] == "no_usable_reading"
    assert turn.person_message == "I want the next one."
    assert turn.person_message_below_threshold is True


def test_soul_mirror_vector_carries_whose_clause_it_is():
    turn = Hearing().hear(text="my moms in pain im scared")
    fv = turn.face_vector.to_dict()
    assert fv["output"] == "soul_mirror"
    # The person's exact words, cut at the clause, every letter kept.
    assert fv["clauses"] == [
        {"said": "my moms in pain ", "placed": True, "attribution": "other"},
        {"said": "im scared", "placed": True, "attribution": "self"},
    ]
    assert "".join(c["said"] for c in fv["clauses"]) == "my moms in pain im scared"
    assert fv["any_measured"] is False      # no emotion classifier on this path


# -- 2026-10-29: Everett: Hand Up ("Request attention") is a call for help.

def test_hand_up_is_the_one_temporal_signal_that_pages():
    # Decision moved to the being 2026-11-04: see WHAT_THE_BEING_DECIDES.md
    # The `pages` field is gone; Hand Up is the one built-in label whose
    # intent is help, and it reads category "danger".
    from christman_voice_sdk.nonverbal.engine_temporal import DEFAULT_LANGUAGE_MAP
    help_labels = sorted(k for k, e in DEFAULT_LANGUAGE_MAP.items()
                         if e.get("intent") in TEMPORAL_HELP_INTENTS)
    assert help_labels == ["Hand Up"]
    assert all("pages" not in e for e in DEFAULT_LANGUAGE_MAP.values())


def test_a_hand_up_reading_reaches_a_human():
    """The gate, given the reading Hearing builds from a Hand Up result. The
    trained model that produces the reading is not on this machine.
    Decision moved to the being 2026-11-04: see WHAT_THE_BEING_DECIDES.md.
    The gate reads it hard_block with `danger`; nothing is held or paged."""
    from christman_voice_sdk.audio.hearing import harm_frame_safety
    hand_up = {"modality": "temporal:gesture", "input": "Hand Up",
               "intent": "Request attention", "message": None,
               "confidence": 0.9, "category": "danger", "danger": True}
    r = harm_frame_safety("", {"nonverbal_distress": [hand_up]})
    assert r.status == "hard_block" and r.reason == "nonverbal_distress"
    assert r.danger is True                                    # ruling I, as a reading
    assert "nonverbal:temporal:gesture:Hand Up:Request attention" in r.flags

