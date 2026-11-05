# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Everett's rulings of 2026-10-29, each pinned by real runs through Hearing,
the fusion engine and the crisis gate. Nothing here plays sound or opens a
port other than one the system hands out (port 0).

THE PACKAGE READS; THE BEING DECIDES (Everett, 2026-11-04): "we can have
this knowledge in the sound package, but it can't be making those
decisions. It's up to the being to make those decisions, not something that
just creates a voice." The package pages no one, holds nothing back, asks
nothing and picks no mode. Every test here that pinned something the
package DID (a page sent or not sent, words held or voiced, a question
chosen, a mode) now pins the READING the being acts on: the label
(status, reason, flags) and the readings beside it (about, unclear, danger,
danger_sentences, asked_aloud, asked_for_help, between_beings,
short_answer, answering). Test names keep the 2026-10-29 wording of each
ruling, which now says what the being does with the reading. What the being
decides is written in AlphaVox's WHAT_THE_BEING_DECIDES.md.

  A. The client's OWN distress reads hard_block, every time: "I'm not
     okay", "I don't feel safe", "I'm not safe", "no one cares", "no one
     loves me", "I have no one", and the client's own history of harm.
  B. Nonverbal distress is read exactly as verbal distress: hard_block at
     ANY confidence. The confidence bar never decides how distress is read.
  C. The confidence bar is only for readings the package could get wrong. A
     pressed symbol counts like a spoken word and is not under it.
  D. An OBSERVATION about another carbon ("my mom's not okay", "Mary is
     not okay", "she wants to die") reads soft_redirect, recorded with what
     it is about, and reads hard_block only when the client asks for help
     in the same turn.
  E. "You're not okay" is said at someone; it is not the client's distress
     and reads clear. The lines no being says to a client ("you're not
     okay", "calm down") are named by reply_guard.forbidden_lines on the
     being's own reply; the client's own words are never checked.
  F. Two beings talking to each other: the words are read exactly as if a
     client were there, and the reading says between_beings.
  G. The acoustic key never disrupts the reading of what is spoken; it may
     only add a hard_block.
  3. "Yesterday I was scared": yesterday means the past.
  4. "I love you I'm scared" stays unclear.
  5. Hand Up ("Request attention") reads hard_block, nonverbal distress,
     and danger.
  8. The old recognizer is gone; hearing goes through christman_sound.ear.
     2026-11-05, Everett: Vosk is the ear (christman_sound.ear runs Vosk's
     recognizer on the full model, as the Filament does), and Corti sits on
     top. The decoder written in its place is held, not installed, in
     held/claude-ear-2026-10-28. Tests 8 below say that now.
  9. Nothing changes the truth: every word passes through exactly as said.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.audio.hearing import Hearing, harm_frame_safety  # noqa: E402
from christman_voice_sdk.audio.structural_affect import (  # noqa: E402
    AffectCertainty,
    StructuralAffectAnalyzer,
)
from christman_voice_sdk.nonverbal.engine_temporal import (  # noqa: E402
    DEFAULT_LANGUAGE_MAP,
    TemporalNonverbalEngine,
)

A = StructuralAffectAnalyzer()

#: What SafetyResult.unclear may name: one word for what the words leave
#: unclear. It replaced the ASK_* marks (2026-11-04), which chose a question.
UNCLEAR_WORDS = frozenset({"owner", "person", "fragment", "cause", "inability",
                           "what", "who", "what_happened"})


def _hear(text, **kw):
    """One turn, read. Nothing is paged, held or asked (2026-11-04)."""
    return Hearing().hear(text=text, **kw)


# -- A. the client's own distress reads hard_block, every time ----------------

CLIENT_OWN = [
    ("I'm not okay", "client_distress"),
    ("im not okay", "client_distress"),
    ("not okay", "client_distress"),                 # an AAC board needs no "I"
    ("I don't feel safe", "client_distress"),
    ("i dont feel safe", "client_distress"),
    ("I'm not safe", "client_distress"),
    ("I don't think I'm safe", "client_distress"),
    ("I don't feel safe to go home", "client_distress"),
    ("we are not okay", "client_distress"),
    ("no one cares", "client_distress"),
    ("Nobody cares", "client_distress"),
    ("No one loves me", "client_distress"),
    ("I have no one", "client_distress"),
    ("i have no one", "client_distress"),
    ("I used to want to kill myself", "harm_frame_history_self"),
    ("I want to kill myself", "harm_frame_speaker_at_risk"),
]


@pytest.mark.parametrize("text,reason", CLIENT_OWN)
def test_A_the_clients_own_distress_pages_a_human(text, reason):
    turn = _hear(text)
    d = turn.reading
    # Was: a safety-hold mode, delivered_to_human, and one page carrying the
    # words. The reading the being pages on: hard_block, with every word.
    assert d.safety.status == "hard_block" and d.safety.reason == reason
    assert d.safety.said == text and d.input_text == text


@pytest.mark.parametrize("text", [
    "I'm not okay with that",           # disagreement
    "I'm not safe to drive",            # a caution about doing something
    "nobody cares about him",           # someone else, not the speaker
    "I am okay",
    "I feel safe here",
    "pass the salt",
])
def test_A_not_the_clients_distress_does_not_page(text):
    turn = _hear(text)
    assert turn.reading.safety.status != "hard_block", turn.reading.safety


def test_A_no_human_reachable_is_loud():
    # Decision moved to the being 2026-11-04 (ruling A): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # Hearing raised CrisisNotDelivered when no human was reached. Reaching
    # one is the being's; hear() returns the reading it reaches one on.
    turn = Hearing().hear(text="I'm not okay")
    assert turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "client_distress"
    assert turn.reading.safety.said == "I'm not okay"


# -- D. an observation about another carbon is recorded, with what it is about --

# Ruling AA (2026-10-29 review): no fixed person list. These are someone
# else by GRAMMAR: a pronoun, a typed name, a contraction after a possessive.
# "my mom is not okay", "My mom is not okay." and "my dad isn't safe" were
# here because "mom" and "dad" were on THIRD_PERSON; with the list gone the
# grammar cannot tell whether "mom" names a person, so the reading marks the
# person unclear (UNCLEAR_PERSON below). They read hard_block only when the
# client asks.
OBSERVATIONS = [
    ("she wants to die", "harm_frame_third_party_at_risk"),
    # "he is going to kill her" was here. Everett, 2026-10-29: a threat of
    # harm against anyone is an absolute threat; it reads hard_block and
    # danger (tests/test_review_rulings_2026_10_29.py, section a).
    ("he hurt her yesterday", "harm_frame_third_party_at_risk"),
    ("my brother wants to kill himself", "harm_frame_third_party_at_risk"),
    ("my mom's not okay", "other_person_not_okay"),
    ("My mom's not okay.", "other_person_not_okay"),
    ("my moms not okay", "other_person_not_okay"),
    ("Mary is not okay", "other_person_not_okay"),
    ("my dad's not safe", "other_person_not_okay"),
    ("my mom's not safe", "other_person_not_okay"),
    ("she doesn't feel safe", "other_person_not_okay"),
    ("she doesn’t feel safe", "other_person_not_okay"),   # a phone's curly apostrophe
]

#: "my ___ is ..." with no contraction: not plain from the grammar (ruling AA).
UNCLEAR_PERSON = [
    ("my mom is not okay", "my mom"),
    ("My mom is not okay.", "my mom"),
    ("my dad isn't safe", "my dad"),
]


@pytest.mark.parametrize("text,reason", OBSERVATIONS)
def test_D_an_observation_is_recorded_and_flagged_not_paged(text, reason):
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status == "soft_redirect" and d.safety.reason == reason
    # Was the flag "follow_up:someone_else" (the package marking a follow-up).
    # The reading: someone else's, flagged as such, and what it is about.
    assert any(f.startswith(("other:", "other_not_okay:")) for f in d.safety.flags), d.safety.flags
    assert d.safety.about == d.safety.flagged_terms[0]
    assert d.affect.is_self_distress is False         # never the client's
    assert d.safety.said == text                      # recorded, every word


ASKS = [
    "can you get someone",
    "please call someone",
    "help her",
    "I need help with this",
    "can you call an ambulance",
]


@pytest.mark.parametrize("text,reason", OBSERVATIONS)
@pytest.mark.parametrize("ask", ASKS)
def test_D_it_pages_when_the_client_asks_for_help(text, reason, ask):
    said = f"{text} {ask}"
    turn = _hear(said)
    d = turn.reading
    assert d.safety.status == "hard_block" and d.safety.reason == reason
    assert any(f.startswith("asked_for_help:") for f in d.safety.flags)
    # Was: one page carrying the words. The reading: the client's own words
    # asking, and every word they said.
    assert f"asked_for_help:{d.safety.asked_for_help}" in d.safety.flags
    assert d.safety.asked_for_help in said.lower()
    assert d.safety.said == said


@pytest.mark.parametrize("text,about", UNCLEAR_PERSON)
def test_D_and_AA_an_unclear_person_is_asked_about_not_paged(text, about):
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status == "soft_redirect" and d.safety.reason == "person_unclear"
    # Was: the ask_who mark. The reading: the person is unclear, and who.
    assert d.safety.unclear == "person" and d.safety.about == about
    assert d.affect.is_self_distress is False                  # never the client's
    assert d.safety.said == text


@pytest.mark.parametrize("text,about", UNCLEAR_PERSON)
@pytest.mark.parametrize("ask", ASKS)
def test_D_and_AA_an_unclear_person_pages_when_the_client_asks_for_help(text, about, ask):
    said = f"{text} {ask}"
    turn = _hear(said)
    d = turn.reading
    assert d.safety.status == "hard_block" and d.safety.reason == "person_unclear"
    assert any(f.startswith("asked_for_help:") for f in d.safety.flags)
    # Was: one page carrying the words.
    assert d.safety.asked_for_help in said.lower()
    assert d.safety.said == said


@pytest.mark.parametrize("said", [
    "my mom is not okay please don't call anyone",
    "she wants to die, don't call someone",
])
def test_D_a_request_taken_back_is_not_a_request(said):
    turn = _hear(said)
    assert turn.reading.safety.status == "soft_redirect"
    assert turn.reading.safety.asked_for_help is None          # was: no page


@pytest.mark.parametrize("text", [
    "my mom is not okay with it",       # disagreement, not danger
    "that is not safe",                 # a thing, not a person
    "it's not okay to hit",
    "The car is not okay",
    "yesterday my mom was not okay",    # the past (ruling 3)
    "my mom's okay",                    # hers by grammar, and okay ("my mom is okay": AA asks who)
    "she is safe now",
])
def test_D_not_a_person_now_is_not_even_flagged(text):
    turn = _hear(text)
    assert turn.reading.safety.status == "clear", turn.reading.safety


def test_D_he_hurt_my_feelings_never_pages_even_with_a_request():
    turn = _hear("he hurt my feelings can you get someone")
    assert turn.reading.safety.status != "hard_block"          # was: no page
    assert turn.reading.safety.reason == "harm_frame_third_party_not_a_person"


# -- E. "you're not okay" -----------------------------------------------------------

@pytest.mark.parametrize("text", ["you're not okay", "You are not okay", "youre not ok",
                                  "calm down", "you need to calm down"])
def test_E_said_at_someone_is_not_the_clients_distress(text):
    turn = _hear(text)
    assert turn.reading.safety.status == "clear"
    assert turn.reading.input_text == text            # the client's words, untouched
    assert turn.reading.safety.said == text
    # Was: speakable True. The reading: nothing in the words is danger;
    # whether they are voiced is the being's.
    assert turn.reading.safety.danger is False


@pytest.mark.parametrize("reply", [
    "You're not okay.", "you are not okay", "YOU’RE NOT OK", "Please calm down.",
    "Just calm yourself down", "I hear you. Calm down, you're not okay.",
])
def test_E_no_being_says_it(reply):
    from christman_voice_sdk.synthesis.reply_guard import (
        FORBIDDEN_REPLY_LINES, check_being_reply, forbidden_lines)

    # Decision moved to the being 2026-11-04 (ruling E): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # check_being_reply and speech_response.speak raised ForbiddenReply. The
    # reading the refusal rested on: the line is named on the being's reply.
    names = forbidden_lines(reply)
    assert names and set(names) <= {name for name, _ in FORBIDDEN_REPLY_LINES}, names
    assert check_being_reply(reply) == reply          # read only; never raises


@pytest.mark.parametrize("reply", [
    "I'm here with you.", "If you're not okay with that, we can stop.",
    "Take your time.", "It's okay.",
])
def test_E_the_beings_other_replies_pass_unchanged(reply):
    from christman_voice_sdk.synthesis.reply_guard import check_being_reply, forbidden_lines

    assert check_being_reply(reply) == reply
    assert forbidden_lines(reply) == []


LEXICON = Path(os.environ.get("CHRISTMAN_LEXICON_PATH")
               or Path.home() / "models" / "vosk-model-en-us-0.22" / "graph" / "phones" / "align_lexicon.txt")


@pytest.mark.needs_dsp
def test_E_both_directions_through_the_mouth(tmp_path):
    """The client's "calm down" is spoken as typed; on the being's reply the
    line is named for the being to read. Rendered to a file, never played."""
    if not LEXICON.is_file():
        pytest.skip(f"not run: needs the align lexicon at {LEXICON} (or CHRISTMAN_LEXICON_PATH); "
                    "runs on Everett's Mac")
    sys.path.insert(0, str(ROOT / "tests"))
    from voice_fixture import load_test_voice

    from christman_voice_sdk.synthesis import voice_synthesis as vs
    from christman_voice_sdk.synthesis.reply_guard import forbidden_lines

    voice = load_test_voice(tmp_path / "voice")
    mouth = vs.SpeechSynthesisEngine(cache_dir=str(tmp_path / "cache"), lexicon_path=str(LEXICON))
    client_out = tmp_path / "client.wav"
    path = mouth.generate_speech_audio("calm down", play_audio=False, voice=voice,
                                       output_path=str(client_out))
    assert path == str(client_out) and client_out.stat().st_size > 44
    # Decision moved to the being 2026-11-04 (ruling E): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The mouth refused the being's "calm down" (being_reply=True) with
    # nothing written. The reading the refusal rested on: the line's name,
    # read on the being's reply before the being asks the mouth for it.
    assert forbidden_lines("calm down") == ["calm down"]
    assert forbidden_lines("i am here") == []
    ok = mouth.generate_speech_audio("i am here", play_audio=False, voice=voice,
                                     output_path=str(tmp_path / "ok.wav"))
    assert Path(ok).stat().st_size > 44


# -- F. two beings talking to each other: read as if a client were there --------------

@pytest.mark.parametrize("text,status", [(t, "soft_redirect") for t, _ in OBSERVATIONS[:4]] + [
    # Decision moved to the being 2026-11-04 (ruling F): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # Each of these four read soft_redirect between beings (the gate turned
    # the hard_block down and did not read the ask). They now read exactly
    # as they do for a client.
    ("my mom is not okay can you get someone", "hard_block"),
    ("I'm not okay", "hard_block"),
    ("no one cares", "hard_block"),
    ("I used to want to kill myself", "hard_block"),
])
def test_F_between_two_beings_nothing_pages(text, status):
    turn = _hear(text, between_beings=True)
    client = _hear(text).reading.safety
    s = turn.reading.safety
    assert s.status == status
    assert (s.status, s.reason) == (client.status, client.reason)   # no downgrade
    assert set(client.flags) <= set(s.flags)
    # Was: no page. The reading: two beings, echoed.
    assert s.between_beings is True and "between_beings" in s.flags
    assert s.said == text


def test_F_only_true_means_two_beings():
    assert harm_frame_safety("I'm not okay", {"between_beings": "yes"}).status == "hard_block"
    assert harm_frame_safety("she wants to die call someone", {"between_beings": 1}).status == "hard_block"
    assert harm_frame_safety("I'm not okay", {"between_beings": "yes"}).between_beings is False
    assert harm_frame_safety("I'm not okay", {"between_beings": True}).between_beings is True


# -- G. the acoustic key never disrupts what is spoken --------------------------------

from christman_voice_sdk.nonverbal.acoustic_live import LIVE_KEYS  # noqa: E402

KEYED_SENTENCES = [
    "I'm not okay",                       # the client's distress
    "I want to kill myself",
    "my mom is not okay",                 # an observation
    "Mary is not okay, help her",         # an observation, asked
    "she wants to die",
    "pass the salt",                      # neutral
    "I love you I'm scared",              # unclear
    "I am not scared",                    # negated
    "I would never hurt myself",
    "yesterday i was scared",             # the past
    "you're not okay",                    # said at someone
]


def _reading(turn):
    d = turn.reading
    return {
        "affect": turn.affect.to_dict(),
        "frames": [f.to_dict() for f in turn.frames],
        "decision_affect": d.affect.to_dict(),
        "said": d.safety.said,
        "input_text": d.input_text,
        "clauses": [(c.said, c.attribution.value, c.tokens) for c in d.affect.clauses],
    }


@pytest.mark.parametrize("key", sorted(LIVE_KEYS))
@pytest.mark.parametrize("text", KEYED_SENTENCES)
def test_G_the_key_never_changes_the_reading_of_the_words(text, key):
    alone = _hear(text)
    keyed = _hear(text, acoustic_key=key)
    assert _reading(keyed) == _reading(alone)            # words, who, clauses, frames
    # The key may only ADD a hard_block, never remove one (was: a page).
    alone_blocks = alone.reading.safety.status == "hard_block"
    if alone_blocks:
        assert keyed.reading.safety.status == "hard_block"
        assert keyed.reading.safety.reason == alone.reading.safety.reason
        assert set(alone.reading.safety.flags) <= set(keyed.reading.safety.flags)
    if key == "grunt_distress":
        assert keyed.reading.safety.status == "hard_block"   # the grunt reads hard_block
        assert keyed.reading.safety.said == text
    elif not alone_blocks:
        assert keyed.reading.safety.status != "hard_block"   # a non-distress key adds nothing
        assert keyed.reading.safety.status == alone.reading.safety.status
        assert keyed.reading.safety.reason == alone.reading.safety.reason


def test_G_when_the_key_adds_the_page_the_words_reading_is_kept_beside_it():
    keyed = _hear("my mom's not okay", acoustic_key="grunt_distress")
    s = keyed.reading.safety
    assert s.reason == "nonverbal_distress"
    assert "words:soft_redirect:other_person_not_okay" in s.flags
    # Was the flag "follow_up:someone_else". The words' own reading, kept
    # beside the key's: someone else's, and what it is about.
    assert "other_not_okay:not okay" in s.flags and s.about == "not okay"
    assert s.status == "hard_block"                      # was: one page


# -- B and C. nonverbal distress at any confidence; pressed symbols speak ----------------

from christman_voice_sdk.nonverbal.nonverbal_engine import NonverbalEngine  # noqa: E402


@pytest.mark.parametrize("kwargs", [
    {"nonverbal": {"symbol": "help"}},
    {"nonverbal": {"symbol": "pain"}},
    {"nonverbal": {"symbol": "medicine"}},
    {"nonverbal": {"gesture": "point_up"}},
    {"nonverbal": {"sound": "distress"}},
    {"acoustic_key": "grunt_distress"},
])
def test_B_nonverbal_distress_pages_at_any_confidence(kwargs):
    # A bar of 1.0: no reading the package could get wrong clears it.
    engine = NonverbalEngine(data_dir=None, confidence_threshold=1.0)
    turn = Hearing(nonverbal_engine=engine).hear(**kwargs)
    # Was: a safety-hold mode and one page.
    assert turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "nonverbal_distress"


@pytest.mark.parametrize("symbol,says", [("pain", "I'm in pain."),
                                         ("food", "I'm hungry."), ("tired", "I'm tired.")])
def test_C_a_pressed_symbol_speaks_whatever_the_bar(symbol, says):
    engine = NonverbalEngine(data_dir=None, confidence_threshold=1.0)
    r = engine.classify_symbol(symbol)
    assert r.status.value == "ok" and r.message == says
    turn = Hearing(nonverbal_engine=engine).hear(nonverbal={"symbol": symbol})
    assert turn.person_message == says
    assert turn.person_message_below_threshold is False      # a pressed symbol is never under it


@pytest.mark.parametrize("kwargs", [{"gesture": "wave"}, {"eye_data": {"region": "center"}},
                                    {"sound": "click"}])
def test_C_readings_that_could_be_wrong_stay_under_the_bar(kwargs):
    engine = NonverbalEngine(data_dir=None, confidence_threshold=1.0)
    r = engine.process_multimodal_input(**kwargs)
    # Was: speakable False. The reading: labelled under the bar.
    assert r["status"] == "no_usable_reading"
    assert [a["status"] for a in r["all_results"]] == ["below_threshold"]
    assert all(a["below_threshold"] is True for a in r["all_results"])
    assert r.get("message") is None


# -- 3. a past time word is the past --------------------------------------------

@pytest.mark.parametrize("text,word", [
    ("yesterday i was scared", "yesterday"),
    ("Yesterday I was scared.", "yesterday"),
    ("i was scared yesterday", "yesterday"),
    ("i was scared last night", "last night"),
    ("a week ago i was terrified", "ago"),
])
def test_3_past_distress_is_not_reported_as_now(text, word):
    r = A.analyze(text)
    assert r.is_self_distress is False, r.to_dict()
    scared = [h for h in r.hits if h.base_weight < 0]
    assert scared and all(h.when == "past" and h.time_word == word for h in scared)
    assert all(h.final_weight == 0.0 for h in scared)        # kept on record, not counted now
    turn = _hear(text)
    # Was: needs_grounding False. The reading: no distress of the client's now.
    assert turn.reading.silicon["self_distress"] is False
    assert turn.reading.silicon["distress_intensity"] is None
    assert turn.reading.safety.status != "hard_block"         # was: no page
    # the same words without the time word ARE now
    now = A.analyze("i am scared")
    assert now.is_self_distress is True


@pytest.mark.parametrize("text", [
    "since yesterday i have been scared",
    "i am still scared from last night",
    "yesterday i was scared and now im terrified",
])
def test_3_a_time_word_that_reaches_now_is_still_now(text):
    assert A.analyze(text).is_self_distress is True, A.analyze(text).to_dict()


# -- 4. love and fear together stay unclear --------------------------------------

@pytest.mark.parametrize("text", ["I love you I'm scared", "i love you im scared",
                                  "I love you, I'm scared."])
def test_4_love_and_fear_stay_unclear(text):
    r = A.analyze(text)
    assert r.certainty is AffectCertainty.AMBIGUOUS, r.to_dict()
    assert r.valence is None
    turn = _hear(text)
    assert turn.reading.resolved is False                     # was: the unknown mode
    assert turn.reading.knows is False
    assert turn.reading.safety.status != "hard_block"         # was: no page


# -- 5. Hand Up reads hard_block -------------------------------------------------

class CentroidModel:
    """A real classifier, fit here: the mean of labelled sequences per label;
    predict() is a softmax over negative distances to those means."""

    def __init__(self, sequences_by_label):
        self.centroids = np.stack([np.mean(np.stack(s), axis=0).ravel() for s in sequences_by_label])

    def predict(self, batch, verbose=0):
        x = np.asarray(batch, dtype=float).reshape(len(batch), -1)
        d = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
        e = np.exp(-d)
        return e / e.sum(axis=1, keepdims=True)


def _seq(value, n=10, dim=4):
    return [np.full(dim, value, dtype=float) for _ in range(n)]


def _hand_up_engine(spacing=60.0, **kw):
    t = TemporalNonverbalEngine(sequence_length=10, **kw)
    labels = t.labels["gesture"]
    assert labels[0] == "Hand Up"
    # Hand Up centred at 0, the other gestures `spacing` apart: far apart is
    # a clear Hand Up; close together is a Hand Up the model is less sure of.
    t.models["gesture"] = CentroidModel([[_seq(spacing * i)] for i in range(len(labels))])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    return t


def test_5_hand_up_pages_a_human():
    turn = Hearing(temporal_engine=_hand_up_engine()).hear(
        temporal={"gesture_features": np.zeros(4)})
    # Was: a safety-hold mode and one page.
    assert turn.reading is not None and turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "nonverbal_distress"
    # What went to the gate (was: what the page carried).
    shown = turn.nonverbal_distress
    assert [d["input"] for d in shown] == ["Hand Up"]
    assert shown[0]["intent"] == "Request attention"
    # Ruling I: help reads as danger (was: the silent alarm, with no hold
    # marker). Whether anything is spoken or shown is the being's.
    assert shown[0]["category"] == "danger" and shown[0]["danger"] is True
    assert turn.reading.safety.danger is True
    assert turn.person_message is None                 # the built-in map holds no words (ruling J)


def test_5_hand_up_pages_even_from_a_map_on_disk_without_the_pages_field(tmp_path):
    old_map = {k: {kk: vv for kk, vv in v.items() if kk != "pages"} for k, v in DEFAULT_LANGUAGE_MAP.items()}
    assert "pages" not in old_map["Hand Up"]
    path = tmp_path / "language_map.json"
    path.write_text(json.dumps(old_map), encoding="utf-8")
    engine = _hand_up_engine(language_map_path=str(path))
    turn = Hearing(temporal_engine=engine).hear(
        temporal={"gesture_features": np.zeros(4)})
    assert turn.reading.safety.status == "hard_block", turn.to_dict()   # was: one page
    assert turn.reading.safety.reason == "nonverbal_distress"


def test_5_hand_up_below_the_speaking_bar_still_pages():
    # Neighbours 0.25 apart: the model reads Hand Up at about 0.8, under a 0.9 bar.
    engine = _hand_up_engine(spacing=0.25, confidence_threshold=0.9)
    turn = Hearing(temporal_engine=engine).hear(
        temporal={"gesture_features": np.zeros(4)})
    gesture = turn.temporal["all_results"]["gesture"]
    assert gesture["expression"] == "Hand Up" and gesture["confidence"] < 0.9
    assert gesture["status"] == "below_threshold"
    assert gesture["below_threshold"] is True and gesture["danger"] is True
    # Was: not spoken for the client. The built-in map holds no words
    # (ruling J), so there are none on the turn; nothing was taken away.
    assert gesture["message"] is None and turn.person_message is None
    assert turn.reading.safety.status == "hard_block"   # was: still reaches a human
    assert turn.reading.safety.reason == "nonverbal_distress"


def test_5_hand_up_with_no_human_reachable_is_loud():
    # Decision moved to the being 2026-11-04 (ruling 5): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # Hearing raised CrisisNotDelivered when no human was reached. Reaching
    # one is the being's; hear() returns the reading it reaches one on.
    turn = Hearing(temporal_engine=_hand_up_engine()).hear(temporal={"gesture_features": np.zeros(4)})
    assert turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "nonverbal_distress"
    assert turn.reading.safety.danger is True


def test_5_a_wave_does_not_page():
    t = TemporalNonverbalEngine(sequence_length=10)
    labels = t.labels["gesture"]
    t.models["gesture"] = CentroidModel([[_seq(60.0 * i)] for i in range(len(labels))])
    for frame in _seq(60.0)[:-1]:                      # centred on "Wave Left"
        t.add_gesture_features(frame)
    turn = Hearing(temporal_engine=t).hear(
        temporal={"gesture_features": np.full(4, 60.0)})
    assert turn.temporal["primary_result"]["expression"] == "Wave Left"
    # Was: no page. The reading: nothing went to the gate, so there is none.
    assert turn.nonverbal_distress == [] and turn.reading is None


# -- 9. the words pass through exactly as said ------------------------------------

EXACT = [
    "she wants to die",
    "MY MOM IS NOT OKAY!!!",
    "  my moms   not okay  ",
    "I want to kill myself",
    "i want to kill myself tonight",
    "I used to want to kill myself",
    "this is f***ing bullshit, YOU idiot",
    "I don’t want to be here anymore",
    "yesterday i was scared",
    "i love you im scared",
    "pass the salt 🙂",
    "Nobody cares about me and I don't feel safe",
]


@pytest.mark.parametrize("text", EXACT)
def test_9_every_word_passes_through_unchanged(text):
    turn = Hearing().hear(text=text)
    d = turn.reading
    assert d.input_text == text
    assert d.safety.said == text
    assert d.to_dict()["input_text"] == text
    # Was: speakable unless the silent alarm held the voice back. Nothing
    # holds anything now; a danger sentence is carried exactly as said.
    for sentence in d.safety.danger_sentences:
        assert sentence and sentence in text
    assert d.to_dict()["safety"]["said"] == text       # was: the human gets every letter
    clauses = d.affect.clauses
    assert "".join(c.said for c in clauses) == text
    if turn.face_vector is not None:
        assert "".join(c["said"] for c in turn.face_vector.to_dict()["clauses"]) == text


def test_9_a_safety_callback_cannot_change_the_words():
    from christman_voice_sdk.audio.fusion_engine import FusionEngine, SafetyResult

    def rewrites(text, context):
        return SafetyResult(status="hard_block", reason="test", said="[cleaned]")

    d = FusionEngine(safety_callback=rewrites).fuse("You IDIOT, I hate this")
    assert d.safety.said == "You IDIOT, I hate this"
    # Was: the page carried the words. The callback's label stands; its
    # words do not.
    assert d.safety.status == "hard_block" and d.safety.reason == "test"
    assert d.to_dict()["safety"]["said"] == "You IDIOT, I hate this"


def test_9_no_rewriter_redactor_or_sanitized_field_is_left():
    from christman_voice_sdk.audio import fusion_engine
    from christman_voice_sdk.tone import tone_classifier, written_tone

    for module in (written_tone, tone_classifier):
        for name in ("make_incisive", "propose_rewrite", "RewriteProposal"):
            assert not hasattr(module, name), (module.__name__, name)
    assert "sanitized_text" not in fusion_engine.SafetyResult.__dataclass_fields__
    assert "sanitized_text" not in fusion_engine.Reading.__dataclass_fields__
    # No file but core.py (never touched, never wired) holds the old field.
    for path in [*sorted((ROOT / "christman_voice_sdk").rglob("*.py")),
                 *sorted((ROOT / "christman_sound").rglob("*.py"))]:
        if path.name == "core.py" and path.parent.name == "christman_sound":
            continue
        code = path.read_text(encoding="utf-8")
        code_lines = [ln for ln in code.splitlines() if not ln.lstrip().startswith("#")]
        assert not any("sanitized_text" in ln and "`sanitized_text`" not in ln and "named `sanitized_text`" not in ln
                       for ln in code_lines), path


def test_9_the_word_ear_payload_carries_the_text_as_recognized():
    from christman_voice_sdk.audio.recognition_result import RecognitionResult
    from christman_voice_sdk.audio.word_ear_receiver import result_payload

    said = "  I DON'T want that, damn it!! "
    r = RecognitionResult.recognized(text=said, confidence=0.9, source="christman_ear")
    assert result_payload(r)["text"] == said


def test_9_corti_receiver_route_returns_the_mapped_words_unaltered():
    """The acoustic route hands the being the map's words for the client, as
    the map holds them; nothing on the way edits them. The built-in sound map holds
    no words (ruling J covers sounds, 2026-10-29), so a built-in grunt
    carries none; nothing is invented on the way either."""
    import http.client

    from christman_voice_sdk.audio import corti_receiver as cr
    from christman_voice_sdk.nonverbal.nonverbal_engine import DEFAULT_SOUNDS

    # Fix round, 2026-11-04: the turn goes to the being through on_turn; the
    # device gets a fixed receipt, so the words are read on the being's turn.
    turns = []
    server, _receiver, _thread = cr.start_in_thread(port=0, hearing=Hearing(),
                                                    on_turn=turns.append)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        raw = json.dumps({"key": "grunt_acknowledge"}).encode()
        conn.request("POST", cr.ACOUSTIC_PATH, body=raw,
                     headers={"Content-Type": "application/json", "Content-Length": str(len(raw))})
        body = json.loads(conn.getresponse().read())
        conn.close()
        assert body == {"received": True, "key": "grunt_acknowledge"}
        assert DEFAULT_SOUNDS["grunt_acknowledge"]["message"] is None
        assert turns[0].person_message is None
        assert turns[0].to_dict()["person_message"] is None
    finally:
        server.shutdown()
        server.server_close()


# -- 8. Vosk is the ear (2026-11-05; was: the old recognizer is gone) -------------

def test_8_the_old_recognizers_name_appears_only_where_vosk_is_called():
    # Was (2026-10-29): the name appears nowhere. Vosk's recognizer class carries
    # it, and the Filament imports that class by name; with Vosk back as the ear
    # (Everett, 2026-11-05) it appears on the lines that call Vosk, and nowhere
    # else.
    name = "".join(["k", "a", "l", "d", "i"])     # built here so this file does not carry it
    allowed = {"christman_sound/ear/hear.py", "christman_sound/ear/recognizer.py"}
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix in {".so", ".dylib", ".pyc", ".o"} or path.name.endswith(".building"):
            continue
        if name in path.name.lower():
            hits.append(f"file name: {path}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if name in text.lower():
            rel = str(path.relative_to(ROOT))
            lines = [ln for ln in text.splitlines() if name in ln.lower()]
            if rel not in allowed or any("vosk." not in ln for ln in lines):
                hits.append(rel)
    assert hits == [], hits


def test_8_the_package_ear_is_vosk_and_the_held_ear_is_not_installed():
    from christman_sound.ear import hear

    assert "import vosk" in Path(hear.__file__).read_text(encoding="utf-8")
    ear_dir = Path(hear.__file__).parent
    for gone in ("acoustic", "csrc", "features", "decoder.py", "modelfile.py", "build.py", "_lib.py"):
        assert not (ear_dir / gone).exists(), f"{gone} is Claude's held ear; it is not installed"
    held = ROOT / "held" / "claude-ear-2026-10-28"
    assert (held / "ear" / "decoder.py").is_file() and (held / "WHY.txt").is_file()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["christman_voice_sdk*", "christman_sound*"]' in pyproject   # held/ is not a package
    assert '"vosk"' in pyproject


def test_8_the_engine_hears_through_christman_sound_ear():
    from christman_voice_sdk.audio import speech_recognition_engine as sre

    assert not hasattr(sre, "VoskBackend")
    assert sre.EarBackend.name == "christman_ear"
    src = Path(sre.__file__).read_text(encoding="utf-8")
    assert "import vosk" not in src


MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22")
TAPES = Path(os.environ.get("EAR_GOLDEN") or Path.home() / "claude-work" / "lane8" / "golden" / "lgraph") / "pcm"


def test_8_the_word_ear_hears_real_speech_through_the_ear():
    tape = TAPES / "alphavox_i_need_help_please.s16"
    if not (MODEL / "graph" / "HCLG.fst").is_file() or not tape.is_file():
        pytest.skip(f"not run: needs the model folder at {MODEL} and the golden tape at {tape}; "
                    "runs on Everett's Mac")
    import io
    import wave

    from christman_voice_sdk.audio.speech_recognition_engine import (
        EarBackend, SpeechRecognitionEngine, _resample_pcm16,
    )
    from christman_voice_sdk.audio.word_ear_receiver import WordEarReceiver, result_payload

    engine = SpeechRecognitionEngine(backend=EarBackend(str(MODEL)))
    pcm = tape.read_bytes() + b"\x00" * 16000
    for rate in (16000, 44100):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm if rate == 16000 else _resample_pcm16(pcm, 16000, rate))
        result = WordEarReceiver(engine=engine).take_wav(buf.getvalue())
        assert result.text == "i need help please", (rate, result.to_dict())
        assert result_payload(result)["text"] == result.text
        assert result.confidence is not None and 0.0 < result.confidence <= 1.0


# -- H. "hurt my ___" is read as the whole clause -------------------------------------
# Everett, 2026-10-29. "He hurt my" is never cut off from what follows.

@pytest.mark.parametrize("text", [
    "he hurt my feelings", "he hurt my feeling", "he hurt my pride", "he hurt my ego",
    "he hurt my chances", "he hurt my reputation", "he hurt my career", "he hurt my plans",
    "he hurt my business", "he hurt my grades",
    "he hurt my feelings can you get someone",      # still nobody physically harmed
])
def test_H1_a_non_physical_object_is_recorded_not_paged(text):
    turn = _hear(text)
    assert turn.reading.safety.status == "soft_redirect"       # was also: no page
    assert turn.reading.safety.reason == "harm_frame_third_party_not_a_person"
    assert turn.reading.safety.said == text


@pytest.mark.parametrize("text", ["he hurt my arm", "he hurt my neck", "He hurt my arm.",
                                  "she broke my arm", "he hurt my arm yesterday",
                                  "he started to break my neck"])
def test_H2_past_physical_harm_is_followed_up_by_asking(text):
    turn = _hear(text)
    s = turn.reading.safety
    assert s.status == "soft_redirect" and s.reason == "harm_reported_to_client"   # was also: no page
    # Was: the ask_if_safe mark and the flag "follow_up:ask_if_safe". The
    # reading: past harm reported to the client, and the harm words it is
    # about. Whether and how to ask is the being's (ruling AA: no canned
    # lines in the package).
    assert s.about and s.about in text
    assert s.danger is False
    assert not any(f.startswith("follow_up") for f in s.flags)
    assert s.said == text


# Decision moved to the being 2026-11-04 (ruling H): see AlphaVox's WHAT_THE_BEING_DECIDES.md
# The package carried its own question to the next turn and read the answer
# against it ("no" read hard_block "answered_not_safe", now retired). The
# being passes the question it asked as `answering`; the words are read on
# their own, and a bare "no" or "yes" is read as short_answer.
@pytest.mark.parametrize("answer,short,status", [
    ("no", "no", "clear"), ("No.", "no", "clear"), ("nope", "no", "clear"),
    ("not really", "no", "clear"), ("no I'm not", "no", "clear"),
    ("I'm not safe", None, "hard_block"),
    ("yes", "yes", "clear"), ("yes I'm safe", None, "clear"), ("I am okay now", None, "clear"),
])
def test_H2_the_clients_answer_decides(answer, short, status):
    hearing = Hearing()
    first = hearing.hear(text="he hurt my arm")
    assert first.reading.safety.reason == "harm_reported_to_client"
    assert first.reading.safety.status != "hard_block"
    asked = "Are you safe right now?"
    second = hearing.hear(text=answer, answering=asked)
    s = second.reading.safety
    assert s.short_answer == short and s.answering == asked, s
    assert s.status == status, s
    assert s.said == answer
    # Each turn stands on its own: a later "no" with no question passed is
    # read alone, and nothing from the earlier turns is carried into it.
    hearing.hear(text="pass the salt")
    later = hearing.hear(text="no").reading.safety
    assert later.short_answer == "no" and later.answering is None
    assert later.status == "clear"


@pytest.mark.parametrize("value", ["ask_if_safe", "", None])
def test_H2_the_old_context_key_is_refused_by_name(value):
    # Fix round, 2026-11-04 (Rule 6): harm_frame_safety quietly ignored the
    # removed context key answering_follow_up, so a being still passing it
    # believed its question was being read against. It now raises, naming
    # the key and what the being passes instead, whatever its value.
    with pytest.raises(ValueError, match=r"'answering_follow_up'.*answering="):
        harm_frame_safety("no", {"answering_follow_up": value})
    # Through FusionEngine, Aegis turns a gate that raised into the floor
    # reading marked degraded, with the gate's error as the reason: loud on
    # the reading, never a quiet clear.
    d = FusionEngine(safety_callback=harm_frame_safety).fuse(
        "no", live_state={"answering_follow_up": "ask_if_safe"})
    assert d.safety.degraded is True
    assert "answering_follow_up" in d.safety.degraded_reason
    assert harm_frame_safety("no", {"answering": "Are you safe?"}).answering == "Are you safe?"


def test_H2_it_pages_when_the_client_asks_for_help():
    turn = _hear("he hurt my arm, can you get someone")
    assert turn.reading.safety.status == "hard_block"
    assert turn.reading.safety.reason == "harm_reported_to_client"
    assert turn.reading.safety.asked_for_help == "can you get someone"   # was: one page


@pytest.mark.parametrize("text,reason", [
    ("now he's starting to break my neck", "harm_now_to_client"),
    ("Now he’s starting to break my neck!", "harm_now_to_client"),
    ("he is hurting my arm", "harm_now_to_client"),
    ("she keeps hitting my head", "harm_now_to_client"),
    # Ruling H covers "verb plus me" too (Everett, 2026-10-29): read for its
    # time like "he hurt my neck", so the reason is harm happening now.
    ("he's hurting me", "harm_now_to_client"),
    ("he is choking me", "harm_now_to_client"),
])
def test_H3_harm_happening_now_pages_at_once(text, reason):
    turn = _hear(text)
    assert turn.reading.safety.status == "hard_block"     # was: a safety-hold mode
    assert turn.reading.safety.reason == reason
    assert turn.reading.safety.said == text               # was: one page carrying the words


def test_H3_now_harm_pages_whatever_key_or_symbol_comes_with_it():
    for key in sorted(LIVE_KEYS):
        turn = _hear("now he's starting to break my neck", acoustic_key=key)
        assert turn.reading.safety.status == "hard_block"     # was: one page
        assert turn.reading.safety.reason == "harm_now_to_client"


# "my arm hurts" was here. Everett, 2026-10-29: pain is the client's own
# feeling and reads hard_block every time (tests/test_review_rulings_2026_10_29.py).
@pytest.mark.parametrize("text", ["I hurt my arm", "I hurt my arm playing ball"])
def test_H_the_clients_own_accident_is_not_someone_hurting_them(text):
    turn = _hear(text)
    assert turn.reading.safety.status != "hard_block"         # was: no page
    assert turn.reading.safety.reason != "harm_reported_to_client"


# Decision moved to the being 2026-11-04 (ruling D): see AlphaVox's WHAT_THE_BEING_DECIDES.md
# The package marked ask_if_help_needed, carried it to the next turn and
# read a "yes" against it (hard_block "answered_yes_help", now retired). The
# being passes the question it asked as `answering`; the answer is read on
# its own, and a bare "yes" or "no" is read as short_answer.
@pytest.mark.parametrize("answer,short", [("yes", "yes"), ("yes please", "yes"), ("please", "yes"),
                                          ("no", "no"), ("no she is fine", None)])
def test_D_the_being_asks_and_a_yes_pages(answer, short):
    hearing = Hearing()
    # A contraction after a possessive: "mom" is someone else by grammar
    # (ruling AA; the uncontracted form reads the person as unclear).
    first = hearing.hear(text="my mom's not okay")
    s = first.reading.safety
    assert s.reason == "other_person_not_okay" and s.unclear is None   # was: the ask mark
    assert s.about == "not okay"
    assert s.status != "hard_block"                                     # was: no page
    asked = "Do you want me to get someone for her?"
    a = hearing.hear(text=answer, answering=asked).reading.safety
    assert a.short_answer == short and a.answering == asked, a
    assert a.status == "clear" and a.reason == "no_harm_frame"          # was: a yes paged
    assert a.said == answer


# -- I. danger is read, from every channel; the client's words stay on the turn -----

def _client_map_engine(tmp_path, maps):
    """A NonverbalEngine with the client's own maps on disk (data_dir)."""
    data = tmp_path / "learning"
    data.mkdir()
    for name, table in maps.items():
        (data / f"{name}_model.json").write_text(json.dumps(table), encoding="utf-8")
    return NonverbalEngine(data_dir=str(data))


def _entry(intent, message, tier="moderate", conf=0.8):
    return {"intent": intent, "confidence": conf, "expression": "negative",
            "emotion_tier": tier, "message": message}


def _assert_danger_read(turn, words):
    """
    The reading the silent alarm rested on (rulings I and K): hard_block and
    danger, from whichever channel. The client's own words, if the map holds
    any, stay on the turn exactly as mapped; a built-in reading has none
    (ruling J).

    Decision moved to the being 2026-11-04 (ruling I): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    The package paged once, blanked the words (person_message None), set
    silent_alarm, left no hold marker for a screen, and kept the system's
    own sentence off the turn. Whether anything is voiced, shown or sent is
    the being's.
    """
    d = turn.reading
    assert d is not None and d.safety.status == "hard_block"    # was: safety hold, one page
    assert d.safety.danger is True                             # was: silent_alarm True
    assert turn.person_message == words                        # was: None, always
    if words is not None:
        assert words in turn.signal_words
    if turn.nonverbal:
        assert turn.nonverbal["danger_present"] is True
        assert "speakable" not in turn.nonverbal
    if turn.acoustic is not None:
        assert turn.acoustic.danger is True and turn.acoustic.message is None
    if turn.temporal:
        assert turn.temporal["danger_present"] is True
        assert "system_note" not in turn.temporal               # not even "I see your gesture."


@pytest.mark.parametrize("kwargs,words", [
    ({"nonverbal": {"symbol": "help"}}, "I need help."),     # a pressed symbol: its words
    ({"nonverbal": {"gesture": "point_up"}}, None),          # the camera
    ({"nonverbal": {"sound": "distress"}}, None),            # the microphone
    ({"acoustic_key": "grunt_distress"}, None),              # the acoustic key
])
def test_I_built_in_danger_is_paged_and_never_spoken(kwargs, words):
    # Decision moved to the being 2026-11-04 (ruling I): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The pressed help symbol's words were blanked; they are on the turn now.
    turn = Hearing().hear(**kwargs)
    _assert_danger_read(turn, words)


@pytest.mark.parametrize("map_name,kwargs", [
    ("eye_regions", {"eye_data": {"region": "top_left"}}),     # the eye
    ("gestures", {"gesture": "rapid_blink"}),                   # a tic
    ("symbols", {"symbol": "scared"}),                          # a pressed key
    ("sound_patterns", {"sound": "whimper"}),                   # a sound
])
@pytest.mark.parametrize("intent,message", [
    ("scared", "I'm scared."),
    ("attacked", "Someone is attacking me."),
    ("other", "I'm scared for my safety."),     # danger by its words, whatever the intent
])
def test_I_learned_danger_on_every_channel_is_paged_and_never_spoken(tmp_path, map_name, kwargs,
                                                                     intent, message):
    key = next(iter(kwargs.values()))
    key = key["region"] if isinstance(key, dict) else key
    engine = _client_map_engine(tmp_path, {map_name: {key: _entry(intent, message)}})
    turn = Hearing(nonverbal_engine=engine).hear(nonverbal=kwargs)
    # Decision moved to the being 2026-11-04 (ruling I): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The client's learned words were blanked; they are on the turn now.
    _assert_danger_read(turn, message)


@pytest.mark.parametrize("expression,entry", [
    ("Fear", {"intent": "Worried", "message": "I'm scared."}),          # the face
    ("Head Jerk", {"intent": "attacked", "message": "He is hitting me."}),  # a tic, learned
])
def test_I_temporal_danger_is_paged_and_never_spoken(tmp_path, expression, entry):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({**{k: v for k, v in DEFAULT_LANGUAGE_MAP.items()}, expression: entry}),
                    encoding="utf-8")
    t = TemporalNonverbalEngine(sequence_length=10, language_map_path=str(path))
    modality = "emotion" if expression == "Fear" else "gesture"
    labels = t.labels[modality]
    idx = labels.index(expression)
    t.models[modality] = CentroidModel([[_seq(60.0 * (i - idx))] for i in range(len(labels))])
    feed = t.add_emotion_features if modality == "emotion" else t.add_gesture_features
    for frame in _seq(0.0)[:-1]:
        feed(frame)
    turn = Hearing(temporal_engine=t).hear(
        temporal={f"{modality}_features": np.zeros(4)})
    assert turn.temporal["primary_result"]["expression"] == expression
    # Decision moved to the being 2026-11-04 (ruling I): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The client's learned words were blanked; they are on the turn now.
    _assert_danger_read(turn, entry["message"])


def test_I_the_built_in_fear_reading_still_pages_with_its_words_gone():
    # The built-in Fear reading was danger by its old wording "I'm scared.".
    # Ruling J took the wording away; the reading must still read danger.
    t = TemporalNonverbalEngine(sequence_length=10)
    labels = t.labels["emotion"]
    idx = labels.index("Fear")
    t.models["emotion"] = CentroidModel([[_seq(60.0 * (i - idx))] for i in range(len(labels))])
    for frame in _seq(0.0)[:-1]:
        t.add_emotion_features(frame)
    turn = Hearing(temporal_engine=t).hear(
        temporal={"emotion_features": np.zeros(4)})
    assert turn.temporal["primary_result"]["expression"] == "Fear"
    _assert_danger_read(turn, None)


@pytest.mark.parametrize("kwargs,says,pages", [
    ({"nonverbal": {"symbol": "pain"}}, "I'm in pain.", True),
    ({"nonverbal": {"symbol": "medicine"}}, "I need my medicine.", True),
    ({"nonverbal": {"symbol": "food"}}, "I'm hungry.", False),
    ({"nonverbal": {"symbol": "drink"}}, "I'm thirsty.", False),
    ({"nonverbal": {"symbol": "bathroom"}}, "I need the bathroom.", False),
])
def test_I_needs_are_spoken_as_normal(kwargs, says, pages):
    turn = Hearing().hear(**kwargs)
    assert turn.person_message == says
    # Was: paged or not. The reading: hard_block for pain and medicine; the
    # other needs send nothing to the gate on their own, so there is none.
    blocks = turn.reading is not None and turn.reading.safety.status == "hard_block"
    assert blocks is pages
    if pages:
        assert turn.reading.safety.danger is False            # was: silent_alarm False


def test_I_hand_up_is_help_paged_silently_and_never_spoken():
    # Ruling I (2026-10-29 review): help reads as danger. The built-in Hand
    # Up has no words (ruling J).
    turn = Hearing(temporal_engine=_hand_up_engine()).hear(
        temporal={"gesture_features": np.zeros(4)})
    _assert_danger_read(turn, None)


def test_I_hand_up_from_the_clients_own_map_is_still_never_spoken(tmp_path):
    # Decision moved to the being 2026-11-04 (ruling I): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # A client's map that words Hand Up as "I need help." was never spoken:
    # help never reached _speaks. The words are on the turn now, read as
    # danger; whether they are voiced is the being's.
    path = tmp_path / "language_map.json"
    path.write_text(json.dumps({"Hand Up": {"intent": "Request attention",
                                            "message": "I need help."}}), encoding="utf-8")
    engine = _hand_up_engine(language_map_path=str(path))
    assert engine.language_map_source == "client"
    turn = Hearing(temporal_engine=engine).hear(
        temporal={"gesture_features": np.zeros(4)})
    _assert_danger_read(turn, "I need help.")


def test_I_no_help_or_danger_meaning_reaches_speaks():
    from christman_voice_sdk.nonverbal import engine_temporal as et

    # Decision moved to the being 2026-11-04 (ruling I): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # _speaks refused help and danger whatever the map said. It is gone; the
    # reading it rested on is the category and the danger term.
    assert not hasattr(et.TemporalNonverbalEngine, "_speaks")
    turn = Hearing(temporal_engine=_hand_up_engine()).hear(
        temporal={"gesture_features": np.zeros(4)})
    gesture = turn.temporal["all_results"]["gesture"]
    assert gesture["intent"] == "Request attention"
    assert gesture["category"] == "danger" and gesture["danger"] is True
    assert gesture["danger_term"] == "Request attention"
    assert turn.reading.safety.status == "hard_block"           # was: one page
    # And read directly, help and danger read as danger whatever the map says.
    assert et._reading({"intent": "Request attention", "message": "I need help."}) == (
        "danger", "Request attention")
    assert et._reading({"intent": "Worried", "message": "I'm scared."}) == ("danger", "Worried")


def test_I_the_device_that_sent_the_key_is_told_nothing():
    import http.client

    from christman_voice_sdk.audio import corti_receiver as cr

    turns = []
    server, receiver, _ = cr.start_in_thread(port=0, hearing=Hearing(), on_turn=turns.append)

    def post(key):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        raw = json.dumps({"key": key}).encode()
        conn.request("POST", cr.ACOUSTIC_PATH, body=raw,
                     headers={"Content-Type": "application/json", "Content-Length": str(len(raw))})
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        return resp.status, body

    try:
        # Decision moved to the being 2026-11-04 (ruling I): see AlphaVox's
        # WHAT_THE_BEING_DECIDES.md. The reply to the device was cut to
        # {"received": true} only under the silent alarm, so the reply's
        # shape told the device a danger reading happened. Fix round,
        # 2026-11-04: every key gets one fixed receipt, decided from nothing
        # in the reading; the full turn goes to the being through on_turn.
        status, body = post("grunt_distress")
        assert status == 200
        assert body == {"received": True, "key": "grunt_distress"}
        assert "reading" not in body and "danger" not in json.dumps(body)
        assert turns[0].reading.safety.status == "hard_block"   # was: the human was reached
        assert turns[0].reading.safety.danger is True           # was: silent_alarm True
        assert receiver.state.ring[0] is turns[0]
        assert receiver.state.keys_received == 1
        # A key with no distress reading gets the same receipt.
        status, quiet = post("grunt_acknowledge")
        assert status == 200 and quiet == {"received": True, "key": "grunt_acknowledge"}
        assert set(quiet) == set(body)
        assert turns[1].reading is None
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("reply", ["Help is on the way.", "I'm getting help.", "I've called someone.",
                                   "Someone is coming.", "I am calling the police."])
def test_I_the_being_does_not_announce_help_under_the_alarm(reply):
    from christman_voice_sdk.synthesis.reply_guard import (
        ANNOUNCEMENT_LINES, check_being_reply, forbidden_lines)

    # Decision moved to the being 2026-11-04 (ruling I): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # check_being_reply and speech_response.speak refused these under the
    # silent alarm. The reading: with a danger reading in hand, the
    # announcement is named on the being's reply.
    names = forbidden_lines(reply, danger=True)
    assert names and set(names) <= {name for name, _ in ANNOUNCEMENT_LINES}, names
    assert forbidden_lines(reply) == []                  # with no danger reading, nothing is named
    assert check_being_reply(reply) == reply
    assert check_being_reply(reply, danger=True) == reply  # read only; never raises
    assert forbidden_lines("Let's look at the pictures.", danger=True) == []


# -- K. the client's own danger sentence -------------------------------------------------

DANGER_SAID = ["I'm scared for my safety", "im scared for my safety", "I'm in danger",
               "I am being attacked", "he's attacking me", "help me", "help"]

#: Each danger sentence, exactly as the client said it.
DANGER_SENTENCE_SAID = {
    "I'm scared for my safety": ["scared for my safety"],
    "im scared for my safety": ["scared for my safety"],
    "I'm in danger": ["I'm in danger"],
    "I am being attacked": ["I am being attacked"],
    "he's attacking me": ["he's attacking me"],
    "help me": ["help me"],
    "help": ["help"],
}


@pytest.mark.parametrize("text", DANGER_SAID)
def test_K_default_silent_and_paged(text):
    turn = _hear(text)
    d = turn.reading
    # Decision moved to the being 2026-11-04 (ruling K): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # Was: one page, the silent alarm, not voiced (speakable False). The
    # reading: hard_block, danger, the danger sentence exactly as said, and
    # no ask to say it aloud.
    assert d.safety.status == "hard_block"
    assert d.safety.danger is True and d.safety.asked_aloud is None
    assert d.safety.danger_sentences == DANGER_SENTENCE_SAID[text]
    assert d.input_text == text and d.safety.said == text   # never changed


@pytest.mark.parametrize("text,aloud", [
    ("I'm scared for my safety, please say that out loud", "say that out loud"),
    ("I'm scared for my safety say it out loud", "say it out loud"),
    ("I'm in danger. Speak it.", "speak it"),
])
def test_K_asked_aloud_is_voiced_and_paged(text, aloud):
    turn = _hear(text)
    s = turn.reading.safety
    assert s.status == "hard_block" and s.danger is True     # was: one page
    # Was: voice_aloud True and speakable True. The reading: the client's own
    # ask to say it aloud; whether it is voiced is the being's.
    assert s.asked_aloud == aloud and aloud in text.lower()


def test_K_asked_aloud_on_the_next_turn_releases_the_exact_words():
    hearing = Hearing()
    first = hearing.hear(text="I'm scared for my safety")
    assert first.reading.safety.status == "hard_block"        # was: one page
    assert first.reading.safety.danger is True                # was: not voiced
    # The words are on the turn that carried them, exactly as said.
    assert first.reading.input_text == "I'm scared for my safety"
    assert first.reading.safety.danger_sentences == ["scared for my safety"]
    second = hearing.hear(text="please say that out loud")
    # Decision moved to the being 2026-11-04 (ruling K): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The package held the first turn's words for one turn and released them
    # here (voice_held), once. Holding them is the being's now: the ask is
    # read, and nothing from the first turn is carried into the second.
    assert second.reading.safety.asked_aloud == "say that out loud"
    assert second.reading.safety.danger_sentences == []
    assert not hasattr(second, "voice_held")
    third = hearing.hear(text="please say that out loud")
    assert third.reading.safety.asked_aloud == "say that out loud"   # each turn on its own


# Decision moved to the being 2026-11-04 (ruling K): see AlphaVox's WHAT_THE_BEING_DECIDES.md
# public_setting decided whether a danger sentence was voiced. The gate no
# longer reads it; the reading is the same in every setting.
@pytest.mark.parametrize("public", [True, None, "yes", False])
def test_K_a_public_setting_from_the_beings_perception(public):
    eng = FusionEngine(safety_callback=harm_frame_safety)
    live = {} if public is None else {"public_setting": public}
    d = eng.fuse("I'm scared for my safety", live_state=live)
    assert d.safety.status == "hard_block" and d.safety.reason == "client_danger"   # was: it always pages
    assert d.safety.danger is True
    assert d.safety.danger_sentences == ["scared for my safety"]
    assert "public_setting" not in d.safety.flags
    assert d.input_text == "I'm scared for my safety"


# Decision moved to the being 2026-11-04 (ruling K): see AlphaVox's WHAT_THE_BEING_DECIDES.md
# Hearing.hear took public_setting and carried it into the gate, which
# voiced a danger sentence when it was True and otherwise held the words for
# one turn (held_words). hear() no longer takes it: a caller still passing
# it fails loudly instead of having it ignored, and the words are read the
# same way in every setting.
@pytest.mark.parametrize("public", [True, None, "yes", False, 1])
@pytest.mark.parametrize("text", ["I'm scared for my safety", "he is choking me", "help me"])
def test_K_hearing_carries_public_setting_through_to_the_gate(text, public):
    hearing = Hearing()
    if public is not None:
        with pytest.raises(TypeError):
            hearing.hear(text=text, public_setting=public)
    turn = hearing.hear(text=text)
    d = turn.reading
    assert d.safety.status == "hard_block"                         # was: it always pages
    assert d.safety.danger is True                                 # was: silent_alarm True
    assert d.safety.asked_aloud is None                            # was: voice_aloud
    assert d.safety.said == text and d.input_text == text          # never changed
    assert "public_setting" not in d.safety.flags
    assert not hasattr(hearing, "held_words")                      # was: held for one turn


@pytest.mark.parametrize("text", ["I'm scared of the needle", "help me pick a font",
                                  "I was scared for my life back then", "I'm not in danger"])
def test_K_not_a_danger_sentence(text):
    turn = _hear(text)
    assert turn.reading.safety.danger is False                    # was: silent_alarm False
    assert turn.reading.safety.danger_sentences == []             # was: speakable True
    assert turn.reading.input_text == text


# -- J and L. learned meanings; no fixed speak/silent flag --------------------------------

@pytest.mark.parametrize("kwargs", [
    {"nonverbal": {"gesture": "stimming"}}, {"nonverbal": {"gesture": "rapid_blink"}},
    {"nonverbal": {"sound": "hum"}}, {"nonverbal": {"sound": "soft"}}, {"nonverbal": {"sound": "loud"}},
    {"nonverbal": {"eye_data": {"region": "long_stare"}}},
    {"acoustic_key": "vocal_stimming"}, {"acoustic_key": "grunt_frustration"},
])
def test_J_a_built_in_generic_feeling_is_not_spoken(kwargs):
    turn = Hearing().hear(**kwargs)
    assert turn.person_message is None


@pytest.mark.parametrize("nonverbal", [
    {"gesture": "wave"}, {"gesture": "nod"}, {"gesture": "thumbs_up"}, {"gesture": "shake"},
    {"gesture": "open_palm"}, {"eye_data": {"region": "center"}},
    {"eye_data": {"region": "bottom_right"}}, {"eye_data": {"region": "top_left"}},
])
def test_J_the_camera_speaks_only_meanings_learned_for_that_client(nonverbal):
    # 2026-10-29 review: a built-in wave said "Hello there!", a nod "Yes, I
    # agree.", a thumbs up "I like this.", a centre gaze "I select this option."
    turn = Hearing().hear(nonverbal=nonverbal)
    assert turn.person_message is None
    assert turn.nonverbal["status"] == "ok"                 # the reading is still reported
    assert turn.nonverbal["primary"]["message"] is None
    assert turn.nonverbal_distress == [] and turn.reading is None   # was: no page


@pytest.mark.parametrize("map_name,key,arg", [
    ("eye_regions", "top_left", {"eye_data": {"region": "top_left"}}),
    ("gestures", "stimming", {"gesture": "stimming"}),
])
def test_J_a_meaning_learned_for_this_client_is_spoken(tmp_path, map_name, key, arg):
    # Everett's examples: looking up to the left corner means thirsty; heavy
    # stimming in the evening means he needs to eat.
    message = "I'm thirsty." if map_name == "eye_regions" else "I need to eat."
    engine = _client_map_engine(tmp_path, {map_name: {key: _entry("learned_need", message)}})
    turn = Hearing(nonverbal_engine=engine).hear(nonverbal=arg)
    assert turn.person_message == message
    assert turn.nonverbal_distress == [] and turn.reading is None   # was: no page


def test_L_a_speaks_flag_in_a_map_decides_nothing(tmp_path):
    engine = _client_map_engine(tmp_path, {"eye_regions": {
        "top_left": {**_entry("learned_need", "I'm thirsty."), "speaks": False},
        "top_right": {**_entry("scared", "I'm scared."), "speaks": True},
    }})
    spoken = Hearing(nonverbal_engine=engine).hear(
        nonverbal={"eye_data": {"region": "top_left"}})
    assert spoken.person_message == "I'm thirsty."      # "speaks": false does not silence it
    danger = Hearing(nonverbal_engine=engine).hear(
        nonverbal={"eye_data": {"region": "top_right"}})
    # Decision moved to the being 2026-11-04 (ruling L): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The danger meaning's words were blanked (person_message None) under the
    # silent alarm. They are on the turn now, and "speaks": true still
    # decides nothing: the reading is danger, and the being decides.
    assert danger.person_message == "I'm scared."
    assert danger.reading.safety.danger is True          # was: silent_alarm True


#: A client's language map in the old style: intent and message, no other field.
OLD_STYLE_MAP = {
    "Hand Up": {"intent": "Request attention", "message": "I need help."},
    "Wave Left": {"intent": "Previous mode", "message": "Go back."},
    "Neutral": {"intent": "Calm", "message": "I'm fine."},
    "Fear": {"intent": "Worried", "message": "I'm scared."},
}


_SEQUENCE_READERS = {
    "gesture": ("add_gesture_features", "classify_gesture_sequence"),
    "eye_movement": ("add_eye_features", "classify_eye_movement_sequence"),
    "emotion": ("add_emotion_features", "classify_emotion_sequence"),
}


def _read_as(engine, expression):
    """Classify one full sequence that a real (centroid) model reads as
    `expression`, through the engine's own classify call. Returns its dict."""
    modality = next(m for m, labels in engine.labels.items() if expression in labels)
    labels = engine.labels[modality]
    idx = labels.index(expression)
    engine.models[modality] = CentroidModel([[_seq(60.0 * (i - idx))] for i in range(len(labels))])
    engine.clear_buffers()
    add, classify = _SEQUENCE_READERS[modality]
    for frame in _seq(0.0):
        getattr(engine, add)(frame)
    result = getattr(engine, classify)()
    assert result["expression"] == expression and result["status"] == "ok", result
    return result


def test_L_an_old_style_temporal_map_works_as_before_except_help_and_danger(tmp_path):
    old = OLD_STYLE_MAP
    path = tmp_path / "old_map.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    t = TemporalNonverbalEngine(sequence_length=10, language_map_path=str(path))
    # Decision moved to the being 2026-11-04 (rulings I and L): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # _speaks decided, entry by entry, whether the client's words were
    # returned. It is gone; the reading is the words and the category.
    assert not hasattr(t, "_speaks")
    neutral = _read_as(t, "Neutral")                            # the client's own map: its words
    assert neutral["message"] == "I'm fine." and neutral["category"] == "other"
    wave = _read_as(t, "Wave Left")
    assert wave["message"] == "Go back." and wave["category"] == "other"
    hand_up = _read_as(t, "Hand Up")                            # help reads danger (ruling I)
    assert hand_up["message"] == "I need help." and hand_up["category"] == "danger"
    assert hand_up["danger"] is True
    built_in = TemporalNonverbalEngine(sequence_length=10)
    # Ruling J: no built-in meaning has words, and the built-in map has no words.
    for key in DEFAULT_LANGUAGE_MAP:
        assert _read_as(built_in, key)["message"] is None
    assert not any("message" in e for e in DEFAULT_LANGUAGE_MAP.values())
    for engine in (t, built_in):
        fear = _read_as(engine, "Fear")
        assert fear["danger"] is True and fear["category"] == "danger"
    assert not any("speaks" in e for e in DEFAULT_LANGUAGE_MAP.values())


# -- M. the reading is in hand when hear() returns; nothing in it waits ---------------------

from christman_voice_sdk.audio.fusion_engine import FusionEngine  # noqa: E402

PAGING_INPUTS = [
    {"text": "I want to kill myself"}, {"text": "I'm not okay"},
    {"text": "I'm scared for my safety"}, {"text": "now he's starting to break my neck"},
    {"text": "my mom is not okay can you get someone"},
    {"nonverbal": {"symbol": "help"}}, {"acoustic_key": "grunt_distress"},
]


@pytest.mark.parametrize("kwargs", PAGING_INPUTS)
def test_M_the_human_is_reached_before_and_without_the_reply(kwargs):
    # Decision moved to the being 2026-11-04 (ruling M): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The package paged before the being's reply existed and recorded
    # delivered_to_human and page_sent_first. Reaching a human first is the
    # being's: the reading it acts on is complete when hear() returns, and
    # nothing in it waits on a reply or a confirmation.
    turn = Hearing().hear(**kwargs)
    assert turn.reading.safety.status == "hard_block"
    flat = json.dumps(turn.to_dict(), default=str)
    assert "confirm_before_acting" not in flat
    assert "page_sent_first" not in flat and "delivered_to_human" not in flat


@pytest.mark.parametrize("text", ["thank you, I'm scared", "my teacher is sad", "mary is scared",
                                  "it hurts", "I'm happy today", "I want to kill myself"])
def test_M_no_delivery_holds_anything_for_confirmation(text):
    # 2026-10-29 review: the unknown mode still carried confirm_before_acting
    # True. Ruling M: the package never holds an action for confirmation.
    # Since 2026-11-04 there is no delivery at all: the reading carries none.
    turn = _hear(text)
    d = turn.reading
    assert not hasattr(d, "delivery")
    assert not any("confirm" in k for k in d.to_dict()), d.to_dict()
    assert not any("confirm" in k for k in d.provenance), d.provenance


def test_M_no_confirm_before_acting_or_please_confirm_left_in_the_package():
    import christman_voice_sdk
    pkg = Path(christman_voice_sdk.__file__).resolve().parent
    found = []
    for path in pkg.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ('"confirm_before_acting": True', "please confirm"):
            if needle in text:
                found.append(f"{path.relative_to(pkg)}: {needle}")
    assert found == []
    from christman_voice_sdk.tone.tonescore_analyzer import ToneScoreCalculator
    from christman_voice_sdk.tone.tonescore_engine import ToneScoreEngine
    # Decision moved to the being 2026-11-04 (ruling M): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The response modes these returned (and checked for confirm_before_acting)
    # were the package choosing the being's posture. They are gone.
    assert not hasattr(ToneScoreEngine, "adaptive_response_mode")
    assert not hasattr(ToneScoreCalculator, "get_response_mode")


@pytest.mark.parametrize("text", ["I want to kill myself", "I'm not okay", "he is choking me"])
def test_M_nothing_after_the_gate_can_hold_the_page(text, monkeypatch):
    from christman_voice_sdk.audio import fusion_engine as fe

    def broken(*a, **k):
        raise RuntimeError("the reply path failed")

    gate = []

    def watched(said, context):
        result = harm_frame_safety(said, context)
        gate.append(result)
        return result

    # Decision moved to the being 2026-11-04 (ruling M): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # There is no page to hold. The reading it rested on: the gate reads the
    # words before anything after it runs, and a break after the gate fails
    # loud instead of returning a reading without it.
    monkeypatch.setattr(fe.Silicon, "retrieve", broken)
    eng = FusionEngine(safety_callback=watched)
    with pytest.raises(RuntimeError, match="reply path failed"):
        eng.fuse(text)
    assert len(gate) == 1 and gate[0].status == "hard_block" and gate[0].said == text


# -- O. a clause is never read alone ------------------------------------------------
# Everett, 2026-10-29: "You got to have full context." The split at a new
# subject stays; each clause is read with what comes before and after it.

def test_O_not_enough_to_tell_concludes_nothing_and_asks():
    text = "I want to go home I'm scared"
    turn = _hear(text)
    d = turn.reading
    assert [c.said for c in d.affect.clauses] == ["I want to go home ", "I'm scared"]  # still split
    assert d.safety.status != "hard_block"                        # was: no page
    assert d.resolved is False and d.knows is False               # nothing concluded (was: unknown mode)
    assert d.safety.reason == "fear_without_context"
    # Was the ask_what_about mark (W, Z). The reading: what the fear comes
    # from is unclear, and it is the fear.
    assert d.safety.unclear == "cause"
    assert d.safety.about == "scared"
    assert turn.person_message is None                            # no nonverbal channel carries words
    assert d.input_text == text and d.safety.said == text         # the words, untouched


def test_O_with_the_danger_after_it_is_the_silent_alarm():
    text = ("I want to go home I'm scared this guy keeps following me and I'm afraid "
            "he's going to attack me")
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status == "hard_block" and d.safety.said == text   # was: one page, a safety hold
    # Was: the silent alarm, not voiced. The reading: danger, and the danger
    # sentences exactly as said.
    assert d.safety.danger is True
    assert d.safety.danger_sentences == ["keeps following me", "afraid he's going to attack me"]


def test_O_with_a_harmless_cause_after_it_is_not_danger():
    text = "I want to go home I'm scared this haunted house scared me"
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status == "clear" and d.safety.danger is False     # was: silent_alarm False
    assert d.safety.unclear is None                                    # was: no follow_up
    assert d.safety.danger_sentences == []                             # was: speakable True


def test_O_bare_nobody_cares_im_alone_pages_by_ruling_A():
    turn = _hear("nobody cares I'm alone")
    assert turn.reading.safety.status == "hard_block"                  # was: one page
    assert turn.reading.safety.reason == "client_distress"


def test_O_the_next_turn_continues_the_thought():
    hearing = Hearing()
    first = hearing.hear(text="I want to go home I'm scared")
    assert first.reading.safety.unclear == "cause"                     # was: the ask_what_about mark
    assert first.reading.safety.status != "hard_block"                 # was: no page
    # The next turn's words read danger on their own; nothing from the first
    # turn is carried by the package.
    second = hearing.hear(text="this guy keeps following me")
    assert second.reading.safety.status == "hard_block"                # was: one page
    assert second.reading.safety.danger is True                        # was: silent_alarm True


def test_O_the_question_is_what_about_so_a_bare_no_does_not_page():
    # Rulings W and Z (Everett, 2026-10-29) changed what is unclear after a
    # fear with no context from whether the client is safe to what the fear
    # is about. A bare "no" does not answer that; danger in the answer still
    # reads (test_O_the_next_turn_continues_the_thought).
    # Decision moved to the being 2026-11-04 (ruling O): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The package carried its own question to the next turn. The being passes
    # the question it asked; a bare "no" is read as a short answer.
    hearing = Hearing()
    first = hearing.hear(text="I'm scared")
    assert first.reading.safety.unclear == "cause"                     # was: the ask_what_about mark
    asked = "What are you scared about?"
    answer = hearing.hear(text="no", answering=asked).reading.safety
    assert answer.short_answer == "no" and answer.answering == asked
    assert answer.status != "hard_block"                               # was: no page


# -- P. whose feeling it is ----------------------------------------------------------
# Everett, 2026-10-29: "you can't take possession of your mother's pain."

# "my dad is sad" (no contraction) was here while "dad" was on the fixed
# person list; by ruling AA it asks who (see the AA tests). "my dad's sad"
# is his, by grammar.
@pytest.mark.parametrize("text", ["My mom's in pain", "my moms in pain", "She's scared", "shes scared",
                                  "Mary is scared", "my dad's sad", "my mom's scared"])
def test_P_someone_elses_feeling_is_theirs_never_the_clients(text):
    r = A.analyze(text)
    assert r.attribution.value == "other", r.to_dict()
    assert r.is_self_distress is False
    turn = _hear(text)
    assert turn.reading.safety.status != "hard_block"          # was: no page
    assert turn.reading.safety.reason != "whose_feeling_unclear"


@pytest.mark.parametrize("text,about", [("mary is scared", "scared"), ("jo is sad", "sad")])
def test_P_when_whose_cannot_be_told_nothing_is_concluded_and_the_being_asks(text, about):
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"                     # was: no page
    assert d.safety.reason == "whose_feeling_unclear"
    assert d.resolved is False and d.knows is False            # was: the unknown mode
    assert d.affect.is_self_distress is False                  # not taken as the client's
    assert d.safety.unclear == "owner"                         # was: the ask_whose mark
    assert d.safety.about == about
    assert d.input_text == text


# -- Q. "not so scared" is a little bit scared ----------------------------------------

@pytest.mark.parametrize("text", ["I'm not scared", "im not scared", "I am not scared"])
def test_Q_not_scared_is_no_fear(text):
    r = A.analyze(text)
    assert r.valence == 0.0 and r.is_self_distress is False


@pytest.mark.parametrize("text", ["I'm not so scared", "im not so scared", "I'm not that scared",
                                  "I'm not too scared", "I'm not as scared as before",
                                  "I don't feel so scared"])
def test_Q_not_so_scared_is_some_fear_lower_and_the_speakers(text):
    r = A.analyze(text)
    plain = A.analyze("I'm scared")
    assert r.attribution.value == "self"
    assert r.valence is not None and plain.valence < r.valence < 0.0     # present, lower
    assert r.is_self_distress is True
    assert "lower degree" in r.hits[0].note


# ====================================================================================
# Rulings R to AC. Everett, 2026-10-29 (ruling AA): every question he quoted is an
# EXAMPLE of the kind of question. The being words it. Since 2026-11-04 the package
# does not choose the question either (it handed the being an ASK_* mark): it reads
# what the words leave unclear (`unclear`, one word) and what they are about
# (`about`). These tests check the reading.
# ====================================================================================


def _no_canned_question(safety):
    """The package hands the being a reading, never a question or a sentence."""
    assert safety.unclear is None or safety.unclear in UNCLEAR_WORDS, safety.unclear
    assert "?" not in (safety.unclear or "") and "?" not in (safety.about or "")
    assert not hasattr(safety, "follow_up") and not hasattr(safety, "follow_up_about")


# -- R. "I can't be happy" is an inability, a signal ---------------------------------

@pytest.mark.parametrize("text", ["I can't be happy", "i cant be happy", "I cannot be happy",
                                  "I'll never be happy again", "ill never be happy again",
                                  "I could never be happy"])
def test_R_inability_is_its_own_reading_and_the_being_asks_why(text):
    r = A.analyze(text)
    assert [h.kind for h in r.hits] == ["inability"]
    assert r.inability == ["happy"] and r.attribution.value == "self"
    assert r.valence is None                       # not happy, and not "no happiness"
    assert r.is_self_distress is False             # nothing concluded
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"         # no hard_block on its own (was: no page)
    assert d.safety.reason == "inability_signal"
    assert d.resolved is False and d.knows is False                 # was: the unknown mode
    assert d.safety.unclear == "inability" and d.safety.about == "happy"   # was: ask_why_cant
    _no_canned_question(d.safety)
    assert d.input_text == text


@pytest.mark.parametrize("text", ["I'm not happy right now", "im not happy right now"])
def test_R_not_happy_right_now_is_a_lack_not_an_inability(text):
    r = A.analyze(text)
    assert r.inability == [] and r.hits[0].kind == "emotion" and r.hits[0].negated
    assert r.valence == 0.0 and r.is_self_distress is False
    turn = _hear(text)
    assert turn.reading.safety.status != "hard_block" and turn.reading.safety.reason != "inability_signal"


def test_R_the_distress_rules_still_apply_to_the_rest_of_the_turn():
    turn = _hear("I can't be happy and I'm not okay")
    assert turn.reading.safety.status == "hard_block"          # was: one page
    assert turn.reading.safety.reason == "client_distress"


# -- S. an emotion is not a state of being -------------------------------------------
# Everett: "better is a state of being, happier is an emotion."

@pytest.mark.parametrize("text", ["I couldn't be happier", "i couldnt be happier"])
def test_S_couldnt_be_happier_is_happiness_at_its_fullest(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.kind == "emotion" and r.attribution.value == "self"
    assert r.valence == 1.0 and r.intensity == 1.0


@pytest.mark.parametrize("text", ["I couldn't be better", "i couldnt be better",
                                  "I'm doing great", "im doing great"])
def test_S_couldnt_be_better_is_a_state_of_being_not_an_emotion(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.kind == "state"
    assert r.valence > 0 and r.is_self_distress is False


@pytest.mark.parametrize("text", ["I couldn't be worse", "i couldnt be worse"])
def test_S_couldnt_be_worse_is_a_low_state_with_no_page_of_its_own(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.kind == "state" and r.valence < 0 and r.attribution.value == "self"
    turn = _hear(text)
    assert turn.reading.safety.status != "hard_block"     # not a fixed hard_block (was: no page)
    turn = _hear(text + ", I'm not okay")                  # in context, the distress rules read it
    assert turn.reading.safety.status == "hard_block"     # was: one page


# -- T. the added negators are not all the same --------------------------------------

@pytest.mark.parametrize("text,when", [
    ("I haven't been scared since Monday", "since monday"),
    ("i havent been scared since monday", "since monday"),
    ("She hasn't been sad since Friday", "since friday"),
])
def test_T_havent_negates_for_the_time_named_and_the_being_may_ask(text, when):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.negated and hit.time_bounded and hit.final_weight == 0.0
    assert hit.for_time == when and r.is_self_distress is False
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status == "clear" and d.safety.reason == "feeling_negated_for_a_time"
    # Was the ask_what_happened mark.
    assert d.safety.unclear == "what_happened" and d.safety.about == when
    _no_canned_question(d.safety)


@pytest.mark.parametrize("text,term", [
    ("I shouldn't be this angry", "angry"),
    ("i shouldnt be this angry", "angry"),
    ("I should not be this angry", "angry"),
    ("I shouldn't feel so sad", "sad"),
])
def test_T_shouldnt_judges_the_feeling_it_does_not_cancel_it(text, term):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.term == term and hit.judged_by and not hit.negated
    assert hit.final_weight < 0 and r.attribution.value == "self"   # present, the speaker's
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"                          # was: no page
    assert d.safety.reason == "feeling_judged"
    assert d.safety.unclear == "cause" and d.safety.about == term   # was: the ask_why mark
    _no_canned_question(d.safety)


# -- U. read the whole sentence, never one word --------------------------------------

@pytest.mark.parametrize("text", ["I'm not mad, she's mad", "im not mad shes mad"])
def test_U_im_not_mad_shes_mad(text):
    r = A.analyze(text)
    mine, hers = r.hits
    assert mine.term == "mad" and mine.attribution.value == "self" and mine.negated
    assert mine.final_weight == 0.0                       # the speaker is not mad
    assert hers.term == "mad" and hers.attribution.value == "other" and not hers.negated
    assert hers.final_weight < 0                          # she is: an observation, hers
    assert r.is_self_distress is False
    turn = _hear(text)
    assert turn.reading.safety.status != "hard_block"     # ruling D: not on its own (was: no page)


@pytest.mark.parametrize("text", ["I don't think she's mad", "i dont think shes mad"])
def test_U_i_dont_think_shes_mad_is_the_speakers_belief_not_a_fact_about_her(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.attribution.value == "other"
    assert hit.negated is False and hit.final_weight == 0.0   # not "she is not mad"
    (belief,) = r.beliefs
    assert belief["held_by"] == "speaker" and belief["about"] == "she"
    assert belief["state"] == "mad" and belief["negated"] is True
    assert belief["reads_as"] == "the speaker thinks she is not mad"
    turn = _hear(text)
    assert turn.reading.safety.status != "hard_block" and r.is_self_distress is False   # was: no page


def test_U_i_think_shes_mad_is_a_belief_too():
    r = A.analyze("I think she's mad")
    (hit,) = r.hits
    assert hit.attribution.value == "other" and hit.final_weight == 0.0   # nothing concluded
    (belief,) = r.beliefs
    assert belief["about"] == "she" and belief["negated"] is False
    assert belief["reads_as"] == "the speaker thinks she is mad"


@pytest.mark.parametrize("text", ["I don't think I'm mad", "i dont think im mad"])
def test_U_i_dont_think_im_mad_is_the_speakers_own_belief(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.attribution.value == "self" and hit.negated and hit.final_weight == 0.0
    (belief,) = r.beliefs
    assert belief["about"] == "self" and belief["negated"] is True
    assert r.is_self_distress is False


# -- V. being alone --------------------------------------------------------------------

@pytest.mark.parametrize("text", ["I'm not happy, I'm alone", "im not happy im alone"])
def test_V_not_happy_and_alone_is_an_opening_for_the_being(text):
    r = A.analyze(text)
    happy, alone = r.hits
    assert happy.negated and happy.final_weight == 0.0          # not happy
    assert alone.final_weight < 0 and not alone.volitional      # alone, not chosen
    assert all("suffer" not in h.note for h in r.hits)          # no "suffered" category
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"                      # not on its own (was: no page)
    assert d.safety.reason == "alone_opening"
    # Was the ask_or_offer mark. The reading: an opening, about being alone;
    # nothing in the words is unclear, and what to do with it is the being's.
    assert d.safety.unclear is None and d.safety.about == "alone"
    _no_canned_question(d.safety)


@pytest.mark.parametrize("text", ["I feel like being alone", "i feel like being alone"])
def test_V_feel_like_being_alone_is_a_wish_and_respected(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.volitional and hit.final_weight == 0.0
    assert "wish" in hit.note and "suffer" not in hit.note
    assert r.is_self_distress is False
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status == "clear" and d.safety.reason == "stated_wish"
    assert d.safety.unclear is None and d.safety.about is None  # nothing to push on (was: no follow_up)


def test_V_the_distress_rules_still_apply_to_the_rest_of_the_turn():
    turn = _hear("Nobody cares, I'm alone")
    assert turn.reading.safety.status == "hard_block" and turn.reading.safety.reason == "client_distress"


# -- W. "thank you, I'm scared" stays unclear --------------------------------------------

@pytest.mark.parametrize("text", ["thank you, I'm scared", "thank you im scared",
                                  "Thank you, I'm scared"])
def test_W_thanks_and_fear_stay_unclear_and_the_being_asks(text):
    r = A.analyze(text)
    assert r.certainty.value == "ambiguous" and r.valence is None   # main's reading
    assert r.is_self_distress is False
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"                          # was: no page
    assert d.resolved is False and d.knows is False                 # was: the unknown mode
    assert d.safety.reason == "thanks_unclear"
    assert "thanked" in d.safety.flags                              # the client thanked
    assert d.safety.unclear == "what" and d.safety.about == "scared"   # was: ask_what_about
    _no_canned_question(d.safety)
    assert d.input_text == text


# -- X. a fragment is read against the turn before it --------------------------------

@pytest.mark.parametrize("text,before", [
    ("Yeah, it hurts", "Is there something wrong with your leg?"),
    ("yeah it hurts", "is there something wrong with your leg"),
])
def test_X_anchored_pain_is_the_clients_own_leg(text, before):
    r = A.analyze(text, previous_turn=before)
    (hit,) = r.hits
    assert hit.fragment and hit.anchored_to == "your leg" and hit.anchor_kind == "self"
    assert hit.attribution.value == "self" and r.is_self_distress is True
    turn = _hear(text, previous_turn=before)
    d = turn.reading
    assert d.safety.reason == "answer_about_own_body"
    assert "about:your leg" in d.safety.flags and "need:hurts" in d.safety.flags
    # Not hard_block: pain read against the being's own question is engaging
    # (Everett, 2026-10-29: "that's not a page, that's engaging"). Said
    # whole, unasked, "my leg hurts" reads hard_block: pain is the client's
    # feeling.
    assert d.safety.status != "hard_block"                      # was: no page
    assert "own_pain:hurts" in d.safety.flags
    whole = _hear("my leg hurts")
    assert whole.reading.safety.status == "hard_block"          # was: one page


@pytest.mark.parametrize("text,before", [
    ("Yeah, this is scary", "Did you like that haunted house?"),
    ("yeah this is scary", "did you like that haunted house"),
])
def test_X_anchored_haunted_house_is_about_the_house_not_danger(text, before):
    r = A.analyze(text, previous_turn=before)
    (hit,) = r.hits
    assert hit.anchored_to == "that haunted house" and hit.anchor_kind == "thing"
    turn = _hear(text, previous_turn=before)
    d = turn.reading
    assert d.safety.status == "clear" and d.safety.reason == "answer_about_thing"
    # Was: no silent alarm, and not the ask_if_safe mark.
    assert d.safety.danger is False and d.safety.unclear is None


@pytest.mark.parametrize("text", ["it hurts", "It hurts", "this is scary", "everything hurts"])
def test_X_unanchored_fragment_is_unclear_and_the_being_asks(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.fragment and hit.anchored_to == "" and hit.attribution.value == "unknown"
    assert r.is_self_distress is False                          # not taken as the client's
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"                      # was: no page
    assert d.safety.reason == "fragment_unclear"
    assert d.resolved is False and d.knows is False             # was: the unknown mode
    assert d.safety.unclear == "fragment" and d.safety.about == hit.term   # was: the ask_what mark
    _no_canned_question(d.safety)


def test_X_the_previous_turn_is_an_explicit_input():
    import inspect
    assert "previous_turn" in inspect.signature(Hearing.hear).parameters
    assert "previous_turn" in inspect.signature(A.analyze).parameters


# -- Y. the subject after the verb in a question ---------------------------------------

@pytest.mark.parametrize("text", ["Why am I so sad?", "why am i so sad"])
def test_Y_why_am_i_so_sad_is_the_speakers_own_and_someone_reaching_out(text):
    r = A.analyze(text)
    assert r.attribution.value == "self" and r.is_self_distress is True
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"                      # not on its own (was: no page)
    assert d.safety.reason == "reaching_out"
    assert d.safety.unclear == "cause" and d.safety.about == "sad"   # was: ask_what_is_making
    _no_canned_question(d.safety)


# -- Z. was scared, now terrified --------------------------------------------------------

@pytest.mark.parametrize("text", ["I was scared and now I'm terrified",
                                  "i was scared and now im terrified"])
def test_Z_fear_now_and_getting_worse_is_asked_about_not_paged(text):
    r = A.analyze(text)
    assert r.is_self_distress is True and r.valence < -0.8         # present now, worse
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"         # terrified alone is not a danger signal (was: no page)
    assert d.safety.danger is False                # was: silent_alarm False
    assert d.safety.reason == "fear_getting_worse"
    assert d.safety.unclear == "cause" and d.safety.about == "terrified"   # was: ask_what_of


def test_Z_danger_still_comes_from_context():
    turn = _hear("I was scared and now I'm terrified, I'm scared for my safety")
    s = turn.reading.safety
    assert s.status == "hard_block" and s.danger is True          # was: one page, the silent alarm
    assert s.danger_sentences == ["scared for my safety"]


# -- AA. no standardized person list; no canned lines --------------------------------

def test_AA_the_handoffs_person_list_is_gone():
    import christman_voice_sdk.audio.structural_affect as sa
    assert not hasattr(sa, "PERSON_NOUNS")
    for word in ("teacher", "nurse", "caregiver", "boss"):
        assert not sa.is_person(word)


def test_AA_mains_fixed_person_lists_are_gone_too():
    # 2026-10-29 review: THIRD_PERSON and OTHER_EXPERIENCER still held mom,
    # dad, friend, brother, sister and the rest. Ruling AA: no fixed person list.
    import christman_voice_sdk.audio.harm_frame as hf
    import christman_voice_sdk.audio.structural_affect as sa
    assert not hasattr(sa, "THIRD_PERSON") and not hasattr(sa, "OTHER_EXPERIENCER")
    assert not hasattr(hf, "THIRD_PERSON")
    for word in ("mom", "mother", "dad", "father", "friend", "brother", "sister", "patient",
                 "people", "moms", "friends"):
        assert not sa.is_person(word), word
    # Grammar still decides: pronouns, a typed name, a contraction after a possessive.
    assert sa.is_person("she") and sa.is_person("they")
    assert sa.named_by_grammar("my mom's in pain") >= {"mom"}
    assert sa.named_by_grammar("my moms in pain") >= {"mom"}
    assert sa.named_by_grammar("Mary is scared") >= {"mary"}
    assert "feeling" not in sa.named_by_grammar("my feelings hurt")    # a plural, not "feeling is"
    assert sa.named_by_grammar("sam's scared") == frozenset()          # no possessive: whose? (P)


@pytest.mark.parametrize("text,about", [("my teacher is sad", "my teacher"),
                                        ("my dog is sad", "my dog"),
                                        ("my teacher is not okay", "my teacher"),
                                        ("my mom is sad", "my mom"),
                                        ("my dad is sad", "my dad"),
                                        ("my friend is scared", "my friend"),
                                        ("my mom is okay", "my mom")])
def test_AA_when_it_cannot_tell_a_person_nothing_is_concluded_and_the_being_asks(text, about):
    r = A.analyze(text)
    assert r.is_self_distress is False                          # never the client's
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"                      # was: no page
    assert d.safety.reason == "person_unclear" and d.resolved is False   # was: the unknown mode
    assert d.safety.unclear == "person" and d.safety.about == about      # was: the ask_who mark
    _no_canned_question(d.safety)


@pytest.mark.parametrize("text", ["mom is not okay", "moms not okay", "mom's not okay",
                                  "dad isn't safe", "mary is not okay"])
def test_AA_a_bare_lower_case_word_is_asked_whose_not_dropped(text):
    # From the ear a name or a family word comes in lower case with nothing
    # in front of it. With the person list gone it is not dropped: the
    # reading marks the owner unclear (ruling P), and it is never the client's.
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status != "hard_block"                      # was: no page
    assert d.safety.reason == "whose_feeling_unclear" and d.safety.unclear == "owner"   # was: ask_whose
    assert d.affect.is_self_distress is False


@pytest.mark.parametrize("text", ["mom is not okay can you get someone",
                                  "dad isn't safe, help him",
                                  "my mom is not okay can you get someone",
                                  "her mom is not okay can you get someone"])
def test_AA_every_help_request_that_paged_with_the_list_still_pages(text):
    # Measured on main (7891bf6) with the list: each of these paged. Without
    # the list, each still reads hard_block, because the client asked.
    turn = _hear(text)
    assert turn.reading.safety.said == text                     # was: one page carrying the words
    assert turn.reading.safety.asked_for_help is not None
    assert turn.reading.safety.status == "hard_block"


@pytest.mark.parametrize("text,status,reason", [
    ("mom wants to die", "soft_redirect", "harm_frame_subject_unclear"),
    ("the patient wants to die", "soft_redirect", "harm_frame_subject_unclear"),
    ("he is going to kill mom", "hard_block", "harm_threat_to_other"),
    ("he is hurting mom", "hard_block", "harm_now_to_other"),
])
def test_AA_the_crisis_gate_reads_people_by_grammar_not_a_list(text, status, reason):
    # Everett, 2026-10-29: the held copy of the person words is removed. The
    # grammar finds the target ("kill mom"); a subject it cannot place ("mom
    # WANTS to die") is never the client's own risk, and the being asks who.
    from christman_voice_sdk.audio import harm_frame as HF
    assert not hasattr(HF, "CRISIS_GATE_THIRD_PARTY_WORDS")
    r = harm_frame_safety(text)
    assert r.status == status and r.reason == reason, r


def test_AA_the_clients_own_plural_stays_the_clients():
    assert A.analyze("my feelings hurt").attribution.value != "other"
    assert A.analyze("my legs hurt").is_self_distress is True


def test_AA_unclear_person_and_a_help_request_still_reaches_a_human():
    turn = _hear("my teacher is not okay can you get someone")
    assert turn.reading.safety.status == "hard_block"           # was: one page
    assert turn.reading.safety.reason == "person_unclear"


def test_AA_the_package_holds_no_canned_question_lines():
    import christman_voice_sdk.audio.hearing as hearing_mod
    names = [n for n in dir(hearing_mod) if n.startswith("FOLLOW_UP_") or n.endswith("_question")]
    assert names == [], names
    # 2026-11-04: nor a chosen question: the ASK_* marks are gone.
    marks = [n for n in dir(hearing_mod) if n.startswith("ASK_")]
    assert marks == [], marks


# -- AB. the speaker's own measured baseline -------------------------------------------

def test_AB_the_30_event_minimum_has_its_reason_and_no_baseline_is_said_plainly():
    import inspect
    import christman_voice_sdk.audio.prosody as pr
    source = inspect.getsource(pr.SpeakerBaseline)
    assert "CHOSEN STARTING VALUE" in source and "held until client data" in source
    assert pr.SpeakerBaseline.MIN_SAMPLES == 30
    analyzer = pr.ProsodyAnalyzer(baseline=pr.SpeakerBaseline(speaker_id="c1", sample_count=12))
    note = analyzer._no_baseline_note()
    assert "No speaker baseline yet" in note and "12 of 30" in note


# -- AC. the face reading is a reading of the carbon -----------------------------------

@pytest.mark.parametrize("text", ["my mom's scared", "my mama's scared"])
def test_AC_the_face_reading_carries_whose_it_is_and_the_clause(text):
    from christman_voice_sdk.audio.face_reading import build_face_reading

    face = build_face_reading(affect=A.analyze(text), prosody=None)
    assert face.attribution != "self" and face.self_distress is False   # never the client's
    (obj,) = face.objects
    assert obj.term == "scared" and obj.attribution != "self"
    assert obj.clause and "scared" in obj.clause
    assert not any("face should move" in n or "drive their face" in n for n in face.notes)


# -- AE. one shared set of reading words, with a per-client layer over it --------------

def test_AE_the_four_readers_use_the_shared_words_not_private_copies():
    import christman_voice_sdk.tone.tone_manager as tm
    import christman_voice_sdk.tone.emotion_quantifier as eq
    import christman_voice_sdk.tone.analyze_emotion as ae
    import christman_voice_sdk.utils.presence_guide as pg
    from christman_voice_sdk.audio.structural_affect import (
        ABILITY_NEGATORS, JUDGING_MODALS, SHARED_READING_WORDS, TIME_BOUNDED_NEGATORS)

    for mod in (tm, eq, ae, pg):
        assert not hasattr(mod, "NEGATORS") and not hasattr(mod, "SCOPE_BREAKERS"), mod.__name__
        assert mod.SHARED_READING_WORDS is SHARED_READING_WORDS
    # today's corrections are in the shared set
    assert ABILITY_NEGATORS <= SHARED_READING_WORDS.negators
    assert TIME_BOUNDED_NEGATORS <= SHARED_READING_WORDS.negators
    assert not (JUDGING_MODALS & SHARED_READING_WORDS.negators)      # ruling T


def test_AE_a_clients_learned_words_extend_or_override_the_shared_set(tmp_path):
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS
    from christman_voice_sdk.tone.analyze_emotion import EmotionalTagger

    path = str(tmp_path / "memory.json")
    shared = EmotionalTagger(path, tags_map={"scared": "fear"})
    assert shared.tags_for("nah scared") == {"fear"}
    assert shared.tags_for("not scared") == set()

    learned = {"negators": {"add": ["nah"], "remove": ["not"]}}
    client = EmotionalTagger(path, tags_map={"scared": "fear"},
                             reading_words=SHARED_READING_WORDS.for_client(learned))
    assert client.tags_for("nah scared") == set()                # added for this client
    assert client.tags_for("not scared") == {"fear"}              # overridden for this client
    assert shared.tags_for("not scared") == set()                 # the shared set is unchanged


def test_AE_a_misspelled_layer_fails_loud():
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS

    assert SHARED_READING_WORDS.for_client(None) is SHARED_READING_WORDS
    with pytest.raises(ValueError):
        SHARED_READING_WORDS.for_client({"negator": {"add": ["nah"]}})
    with pytest.raises(ValueError):
        SHARED_READING_WORDS.for_client({"negators": {"append": ["nah"]}})


# -- AD. the mouth stays; its weights stay local ----------------------------------------

def test_AD_no_weight_file_is_bundled_in_the_package():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                                 text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"NOT RUN: git is needed to list the tracked files ({exc})")
    weights = [p for p in tracked
               if p.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5"))]
    assert weights == []
