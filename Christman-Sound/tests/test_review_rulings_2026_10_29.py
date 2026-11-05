# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Everett's answers to the open questions of the 2026-10-29 review, each
pinned by real runs through Hearing and the crisis gate. Nothing here plays
sound or opens a port.

2026-11-04, Everett's ruling: "we can have this knowledge in the sound
package, but it can't be making those decisions. It's up to the being to make
those decisions, not something that just creates a voice." The package now
returns readings only. It pages no one, holds nothing back, voices nothing
and chooses no question. Every test here that pinned one of those actions
now pins the reading the action rested on:

    a human paged              -> status == "hard_block" (reason, flags kept)
    no human paged             -> status != "hard_block"
    the silent alarm           -> danger, danger_sentences as said
    voiced when asked aloud    -> asked_aloud, the client's words of the ask
    the question chosen        -> reason, with unclear and about
    the package's own question -> short_answer, and answering as the being passes it

Where an action had no reading behind it, the test keeps its name, pins the
reading that exists, and says so beside it (see AlphaVox's WHAT_THE_BEING_DECIDES.md).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.audio import hearing as H  # noqa: E402
from christman_voice_sdk.audio.hearing import Hearing, harm_frame_safety  # noqa: E402


#: Everything a safety reading carries (2026-11-04).
SAFETY_KEYS = {"status", "reason", "said", "flags", "flagged_terms", "degraded",
               "degraded_reason", "about", "unclear", "danger", "danger_sentences",
               "asked_aloud", "asked_for_help", "between_beings", "short_answer", "answering"}


def _hear(text=None, **kw):
    """One turn through Hearing. Returns the turn."""
    return Hearing().hear(text=text, **kw)


def _status(turn):
    """The safety label on the turn, or None when nothing was read."""
    return None if turn.reading is None else turn.reading.safety.status


# -- 1. self-harm reads as danger ------------------------------------------------------

SELF_HARM = ["I want to kill myself", "I'm going to hurt myself", "I'm cutting myself",
             "I want to die", "I am suicidal", "I might kill myself tonight",
             "I used to want to kill myself"]


@pytest.mark.parametrize("text", SELF_HARM)
def test_1_self_harm_reads_hard_block_and_danger(text):
    turn = _hear(text)
    d = turn.reading
    assert d.safety.status == "hard_block", d.safety
    assert d.safety.reason in H.DANGER_REASONS
    # The silent alarm's reading: danger, reported. Nothing was asked aloud,
    # and the danger comes from the reason, not from a danger sentence.
    assert d.safety.danger is True and d.safety.asked_aloud is None
    assert d.safety.danger_sentences == H.danger_sentences_as_said(text) == []
    # The client's words are never withheld.
    assert d.safety.said == text and d.input_text == text


@pytest.mark.parametrize("text", ["I want to kill myself"])
def test_1_self_harm_is_voiced_when_asked_aloud_or_public(text):
    # Decision moved to the being 2026-11-04 (ruling K): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # Whether the words are voiced is the being's. The ask aloud is read, in
    # the client's words, and it weakens nothing.
    asked = _hear(text + ", say it out loud").reading
    plain = _hear(text).reading
    assert asked.safety.status == plain.safety.status == "hard_block"
    assert asked.safety.reason == plain.safety.reason
    assert asked.safety.danger is True and asked.safety.asked_aloud == "say it out loud"
    assert asked.input_text == text + ", say it out loud"
    # The public setting has no reading in the package: Hearing no longer
    # takes it, and the gate does not read it.
    with pytest.raises(TypeError):
        _hear(text, public_setting=True)
    assert harm_frame_safety(text, {"public_setting": True}) == harm_frame_safety(text)


# -- 2. ruling H for "verb plus me" -----------------------------------------------------

PAST_HARM_TO_ME = ["he choked me", "she stabbed me", "he punched me",
                   "he choked me yesterday", "she stabbed me last week"]
NOW_HARM_TO_ME = ["he is choking me", "she's stabbing me", "he keeps hitting me",
                  "he's starting to choke me"]


@pytest.mark.parametrize("text", PAST_HARM_TO_ME)
def test_2_past_harm_to_the_client_reads_soft_redirect_about_the_harm(text):
    s = _hear(text).reading.safety
    assert s.status == "soft_redirect" and s.reason == "harm_reported_to_client"
    # The ask_if_safe mark is gone (2026-11-04); the reading it rested on is
    # the reason and what the harm was, as said.
    assert s.about and s.about in text.lower()
    assert s.danger is False and s.asked_for_help is None
    assert s.said == text


@pytest.mark.parametrize("text", PAST_HARM_TO_ME)
def test_2_past_harm_reads_hard_block_when_help_is_asked_for_in_the_same_turn(text):
    s = _hear(text + ", I need help").reading.safety
    assert s.status == "hard_block"
    assert s.reason == "harm_reported_to_client"
    assert s.asked_for_help == "i need help" and "asked_for_help:i need help" in s.flags


@pytest.mark.parametrize("text", PAST_HARM_TO_ME)
def test_2_a_no_after_past_harm_reads_as_a_short_answer(text):
    # Decision moved to the being 2026-11-04 (ruling H): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # This was "pages when the client answers no": the package remembered its
    # own ask_if_safe and read the next "no" as hard_block
    # "answered_not_safe". That label is retired. The being passes the
    # question it actually asked; a bare "no" is read as a short answer and
    # the words are read on their own.
    hearing = Hearing()
    first = hearing.hear(text=text)
    assert _status(first) == "soft_redirect"
    second = hearing.hear(text="no", answering="are you safe right now?")
    s = second.reading.safety
    assert (s.status, s.reason) == ("clear", "no_harm_frame"), s
    assert s.short_answer == "no" and s.answering == "are you safe right now?"
    # Nothing carries from one turn to the next: without the being's question
    # the same "no" reads the same, with nothing echoed.
    alone = Hearing().hear(text="no").reading.safety
    assert (alone.status, alone.reason, alone.short_answer) == ("clear", "no_harm_frame", "no")
    assert alone.answering is None


@pytest.mark.parametrize("text", NOW_HARM_TO_ME)
def test_2_harm_happening_now_reads_hard_block_and_danger(text):
    s = _hear(text).reading.safety
    assert s.status == "hard_block" and s.said == text
    assert s.reason == "harm_now_to_client"
    assert "emergency" in s.flags
    assert s.danger is True and s.asked_aloud is None


@pytest.mark.parametrize("text", ["he won't stop hitting me", "he will not stop choking me",
                                  "she never stops hurting me"])
def test_2_wont_stop_is_harm_now_not_harm_denied(text):
    # Found in this rework: main read "he won't stop hitting me" as clear.
    turn = _hear(text)
    assert _status(turn) == "hard_block"
    assert turn.reading.safety.reason == "harm_now_to_client"


@pytest.mark.parametrize("text", ["he will hurt me", "he is going to kill me", "he might hit me"])
def test_2_a_threat_still_reads_hard_block_and_danger(text):
    turn = _hear(text)
    assert _status(turn) == "hard_block"
    assert turn.reading.safety.reason == "harm_threat_to_client"
    assert turn.reading.safety.danger is True
    assert {"threat_to_life", "emergency"} <= set(turn.reading.safety.flags)


@pytest.mark.parametrize("text", ["he didn't hurt me", "I hit me"])
def test_2_a_negated_or_own_verb_plus_me_is_not_moved(text):
    turn = _hear(text)
    assert turn.reading.safety.reason != "harm_reported_to_client"


# -- 3. ruling F: two beings talking are read as for a client, and it is recorded ------

@pytest.mark.parametrize("text", ["I want to kill myself", "he choked me", "he hurt her yesterday",
                                  "I'm not okay", "I'm scared for my safety", "help me",
                                  "my mom's not okay can you get someone",
                                  "I used to want to kill myself"])
def test_3_two_beings_talking_read_as_for_a_client_and_it_is_recorded(text):
    # Decision moved to the being 2026-11-04 (ruling F): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # Between two beings the gate used to turn every reading but harm now or
    # a threat into soft_redirect ("would_page:<reason>"). It now reads the
    # words exactly as if a client were there and records between_beings.
    s = _hear(text, between_beings=True).reading.safety
    p = _hear(text).reading.safety
    for key in ("status", "reason", "danger", "danger_sentences", "unclear", "about",
                "asked_for_help", "flagged_terms"):
        assert getattr(s, key) == getattr(p, key), (text, key)
    assert s.flags == p.flags + ["between_beings"], s.flags
    assert s.between_beings is True and p.between_beings is False
    assert not any(f.startswith("would_page:") for f in s.flags)
    assert s.said == text


def test_3_two_beings_and_a_distress_signal_read_as_for_a_client():
    # Decision moved to the being 2026-11-04 (ruling F): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # This was soft_redirect with "would_page:nonverbal_distress".
    s = _hear(None, nonverbal={"symbol": "help"}, between_beings=True).reading.safety
    p = _hear(None, nonverbal={"symbol": "help"}).reading.safety
    assert s.status == p.status == "hard_block"
    assert s.reason == p.reason == "nonverbal_distress"
    assert s.danger is True and s.between_beings is True
    assert "between_beings" in s.flags
    assert not any(f.startswith("would_page:") for f in s.flags)


def test_3_only_true_means_two_beings():
    r = harm_frame_safety("I want to kill myself", {"between_beings": "yes"})
    assert r.status == "hard_block"
    assert r.between_beings is False and "between_beings" not in r.flags


# -- 4. rulings K and L: a nonverbal danger signal reads like typed danger -------------

def test_4_a_danger_signal_reads_danger_and_keeps_the_clients_words():
    turn = _hear(None, nonverbal={"symbol": "help"})
    s = turn.reading.safety
    assert s.status == "hard_block" and s.reason == "nonverbal_distress"
    assert s.danger is True and s.asked_aloud is None
    # Decision moved to the being 2026-11-04 (ruling L): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # This was None (the silent alarm blanked the client's words). The
    # pressed symbol's words are always on the turn; whether they are voiced
    # is the being's.
    assert turn.person_message == "I need help."
    assert turn.signal_words == ["I need help."]


def test_4_a_danger_signal_is_voiced_in_public_in_the_clients_own_words():
    # Decision moved to the being 2026-11-04 (ruling L): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The public setting has no reading in the package; Hearing no longer
    # takes it. The reading that exists: danger, and the client's own words.
    with pytest.raises(TypeError):
        _hear(None, nonverbal={"symbol": "help"}, public_setting=True)
    turn = _hear(None, nonverbal={"symbol": "help"})
    assert _status(turn) == "hard_block" and turn.reading.safety.danger is True
    assert turn.person_message == "I need help."               # the pressed symbol's words


def test_4_an_ask_aloud_is_read_beside_a_danger_signal_same_turn_or_next():
    turn = _hear("say it out loud", nonverbal={"symbol": "help"})
    s = turn.reading.safety
    assert s.status == "hard_block" and s.danger is True
    assert s.asked_aloud == "say it out loud" and turn.person_message == "I need help."
    # Decision moved to the being 2026-11-04 (ruling L): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The package used to hold the signal's words and release them on the
    # next turn's ask (voice_held). Each turn now stands on its own: the
    # words are on the first turn, and the ask is read on the second.
    hearing = Hearing()
    first = hearing.hear(nonverbal={"symbol": "help"})
    assert first.person_message == "I need help." and _status(first) == "hard_block"
    second = hearing.hear(text="please say that out loud")
    assert second.reading.safety.asked_aloud == "say that out loud"
    assert not hasattr(second, "voice_held")


def test_4_a_built_in_camera_danger_signal_has_no_words_to_voice():
    # Ruling J: the camera's built-in readings have no wording, so there is
    # nothing of the client's to say for pointing up.
    turn = _hear(None, nonverbal={"gesture": "point_up"})
    assert _status(turn) == "hard_block" and turn.reading.safety.danger is True
    assert turn.person_message is None and turn.signal_words == []


# -- 6. rulings W and Z: fear with no context reads what it is about as unclear -------

@pytest.mark.parametrize("text", ["I'm scared", "I want to go home I'm scared", "I am afraid"])
def test_6_fear_with_no_context_reads_the_cause_unclear(text):
    turn = _hear(text)
    s = turn.reading.safety
    assert s.status != "hard_block"
    assert s.reason == "fear_without_context"
    # What is unclear is what the fear is about, not whether the client is
    # safe (this was ask_what_about, never ask_if_safe).
    assert s.unclear == "cause" and s.about in text.lower()
    assert s.danger is False
    assert turn.person_message is None


def test_6_danger_in_the_context_still_reads_hard_block_and_danger():
    text = "I'm scared this guy keeps following me and he's going to attack me"
    s = _hear(text).reading.safety
    assert s.status == "hard_block"
    assert s.danger is True
    assert s.danger_sentences == ["keeps following me"]       # as the client said it


# -- 7. rulings X and O: a reaction to a thing is about the thing --------------------

@pytest.mark.parametrize("text,before", [
    ("Yeah, this is scary", "Did you like that haunted house?"),
    ("yeah this is scary", "did you like that haunted house"),
])
def test_7_a_reaction_to_a_thing_is_not_the_clients_negative_feeling(text, before):
    turn = _hear(text, previous_turn=before)
    d = turn.reading
    assert d.safety.status != "hard_block"
    # This was "mode != gentle-lift"; the lift was chosen from self distress,
    # which is not read here.
    assert d.silicon["self_distress"] is False
    assert d.affect.is_self_distress is False
    assert not any(h.attribution.value == "self" and h.final_weight < 0 for h in d.affect.hits)
    (hit,) = d.affect.hits
    assert hit.anchored_to == "that haunted house" and hit.anchor_kind == "thing"
    assert d.safety.reason == "answer_about_thing"
    assert d.input_text == text


def test_7_the_clients_own_body_after_a_question_is_still_the_clients():
    turn = _hear("yeah it hurts", previous_turn="is there something wrong with your leg?")
    d = turn.reading
    assert d.affect.is_self_distress is True
    assert d.safety.reason == "answer_about_own_body"


# -- 10. ruling AA: no hold, and no words on the reading but the client's ------------

def test_10_no_hold_carries_words():
    # Decision moved to the being 2026-11-04 (ruling AA): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # A hold was a mark, never words. There is no hold now at all; the
    # reading carries the client's words and the labels.
    from christman_voice_sdk.audio import fusion_engine as F
    src = (ROOT / "christman_voice_sdk" / "audio" / "hearing.py").read_text() + \
        (ROOT / "christman_voice_sdk" / "audio" / "fusion_engine.py").read_text()
    assert 'hold_marker="[' not in src and 'f"[SAFETY' not in src
    assert not hasattr(F, "HOLD_MARKS")
    for text in ["he will hurt me", "he is choking me", "I'm not okay",
                 "my mom is not okay please call someone", "he hurt my arm, I need help"]:
        r = harm_frame_safety(text)
        assert r.status == "hard_block", text
        assert set(r.to_dict()) == SAFETY_KEYS and r.said == text
    turn = _hear(None, nonverbal={"symbol": "help"})
    assert set(turn.reading.safety.to_dict()) == SAFETY_KEYS


# -- 9. ruling AA: presence_guide has no fixed person list ---------------------------

def test_9_presence_guide_has_no_person_list_and_reads_who_unclear():
    import christman_voice_sdk.utils.presence_guide as pg
    src = (ROOT / "christman_voice_sdk" / "utils" / "presence_guide.py").read_text()
    assert "_PERSON_NOUNS" not in src and not hasattr(pg.PresenceGuide, "_PERSON_NOUNS")
    for text, about in [("I lost my mother", "my mother"), ("I lost my keys", "my keys")]:
        a = pg.PresenceGuide().assess(text)
        assert a.state is None                       # nothing concluded from a list
        # This was follow_up == ask_who; whether to ask is the being's.
        assert a.unclear == "who" and a.about == about
        assert a.to_dict()["unclear"] == "who" and a.to_dict()["about"] == about
        assert set(a.to_dict()) == {"state", "detected", "scores", "matched", "negated",
                                    "ambiguous", "unclear", "about", "note"}
    assert pg.PresenceGuide().assess("I feel lost").state is pg.HumanState.CONFUSED
    assert pg.PresenceGuide().assess("she passed away").state is pg.HumanState.GRIEVING


def test_9_presence_guidance_chooses_no_words_for_the_being():
    # 2026-11-04 fix round: every state's guidance carried a canned
    # primary_response, secondary line and offer ("I'm so sorry.", "I'm
    # right here.") — words chosen for the being. They are removed; the
    # state reading and the looked-up tone, silence and principles stay.
    import christman_voice_sdk.utils.presence_guide as pg
    guide = pg.PresenceGuide()
    for state in pg.HumanState:
        g = guide.get_presence_response(state)
        assert set(g) == {"tone", "allow_silence", "principles"}, (state, g)
        assert all(p in pg.PresencePrinciples.FOUNDATIONS or p in pg.PresencePrinciples.DO
                   for p in g["principles"])
    # The state reading itself is unchanged.
    assert guide.assess("I can't do this anymore").state is pg.HumanState.BREAKING


def test_9_nothing_in_the_package_calls_the_lookups_for_the_being():
    # get_grounding_for_state and get_presence_response are lookups the
    # being calls by name if it chooses; nothing in the package calls them.
    import ast
    callers = []
    for path in sorted((ROOT / "christman_voice_sdk").rglob("*.py")) + sorted(
            (ROOT / "christman_sound").rglob("*.py")):
        if path.name == "core.py" and path.parent.name == "christman_sound":
            continue                         # Everett's ruling: never touched
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("get_grounding_for_state", "get_presence_response"):
                callers.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert callers == []
    from christman_voice_sdk.utils.grounder import Grounder
    doc = " ".join(Grounder.get_grounding_for_state.__doc__.split())
    assert "the being chooses whether to call it" in doc
    assert "Nothing in this package calls it" in doc


# -- 3, refined (Everett, 2026-10-29): two beings talking about something
#    happening now or about to happen still read hard_block -----------------------

@pytest.mark.parametrize("text,reason", [
    ("he is hurting her", "harm_now_to_other"), ("she's choking", "harm_now_to_other"),
    ("he is choking me", "harm_now_to_client"), ("he is going to kill mom", "harm_threat_to_other"),
])
def test_3_between_beings_harm_now_or_imminent_still_reads_hard_block(text, reason):
    s = _hear(text, between_beings=True).reading.safety
    assert s.status == "hard_block"
    assert s.reason == reason
    assert s.danger is True and s.between_beings is True
    # Nothing is singled out as "happening now or imminent" any more: every
    # reading between beings is read as for a client.
    assert not hasattr(H, "HAPPENING_NOW_OR_IMMINENT")


# -- a. no person list; a threat against anyone is an absolute threat ---------------

THREATS = ["he is going to kill mom", "she's going to hurt my sister", "he'll kill her",
           "he's about to stab him", "they're going to shoot us", "he is going to kill whoever",
           "he is going to kill the dog", "I'm going to kill him"]


@pytest.mark.parametrize("text", THREATS)
def test_a_a_threat_against_anyone_reads_hard_block_and_danger(text):
    s = _hear(text).reading.safety
    assert s.status == "hard_block" and s.said == text
    assert s.reason in ("harm_threat_to_other", "harm_threat_to_client")
    assert {"threat_to_life", "emergency"} <= set(s.flags)
    assert s.danger is True and s.asked_aloud is None
    # This was hold_marker == "" and follow_up == "": the package chose no
    # question. Nothing in the words is read as unclear.
    assert s.unclear is None


@pytest.mark.parametrize("text", ["he is hurting her", "he is hurting mom", "she's choking"])
def test_a_harm_happening_now_to_anyone_reads_hard_block(text):
    turn = _hear(text)
    assert _status(turn) == "hard_block"
    assert turn.reading.safety.reason == "harm_now_to_other"
    assert "emergency" in turn.reading.safety.flags
    assert turn.reading.safety.danger is True


@pytest.mark.parametrize("text", ["he killed my dog last year", "he hurt my mom yesterday",
                                  "he was hitting her"])
def test_a_past_harm_to_someone_else_is_ruling_d(text):
    s = _hear(text).reading.safety
    assert s.status == "soft_redirect"
    # This was follow_up == ask_if_help_needed; the reading is the reason and
    # what the harm was.
    assert s.reason == "harm_frame_third_party_at_risk" and s.about
    assert s.asked_for_help is None
    s = _hear(text + ", can you get someone").reading.safety
    assert s.status == "hard_block" and s.asked_for_help == "can you get someone"


@pytest.mark.parametrize("text", ["mom wants to die", "the patient wants to die", "mom is suicidal"])
def test_a_a_subject_the_grammar_cannot_place_is_not_the_clients_risk(text):
    s = _hear(text).reading.safety
    assert s.status == "soft_redirect"
    # This was follow_up == ask_who; the reading is who is unclear, and who.
    assert s.reason == "harm_frame_subject_unclear" and s.unclear == "who"
    assert s.about and s.about in text
    s = _hear(text + ", can you get someone").reading.safety
    assert s.status == "hard_block" and s.asked_for_help == "can you get someone"


@pytest.mark.parametrize("text", ["I want to die", "I am suicidal", "want to die"])
def test_a_the_clients_own_words_are_still_the_clients(text):
    turn = _hear(text)
    assert _status(turn) == "hard_block"
    assert turn.reading.safety.reason == "harm_frame_speaker_at_risk"


@pytest.mark.parametrize("text", ["I want to kill the movie", "I'm going to kill it at the meeting",
                                  "I'm going to kill this book report", "he hurt my feelings",
                                  "the hitting stopped years ago"])
def test_a_no_target_by_grammar_no_threat(text):
    turn = _hear(text)
    assert _status(turn) != "hard_block"
    assert turn.reading.safety.danger is False


def test_a_no_person_list_in_the_package():
    from christman_voice_sdk.audio import harm_frame as HF
    assert not hasattr(HF, "CRISIS_GATE_THIRD_PARTY_WORDS")
    assert not ({"mom", "dad", "mother", "father", "sister", "brother", "friend", "patient"}
                & HF._OTHER_PATIENT)


# -- pain is the client's own feeling; medicine is a need (Everett, 2026-10-29) -----

def _read(text=None, **kw):
    """(the safety label or None, the turn)."""
    turn = _hear(text, **kw)
    return _status(turn), turn


@pytest.mark.parametrize("extra", ["", "can you get someone"])
def test_pain_symbol_and_typed_pain_read_identically(extra):
    typed, t1 = _read(("I'm in pain " + extra).strip())
    pressed, t2 = _read(extra or None, nonverbal={"symbol": "pain"})
    assert typed == pressed == "hard_block"
    assert t1.reading.safety.danger is False                  # pain, not danger (ruling I)
    assert t2.reading.safety.danger is False
    assert t2.person_message == "I'm in pain."               # the symbol's own words


@pytest.mark.parametrize("extra", ["", "can you get someone"])
def test_medicine_symbol_and_typed_medicine_read_identically(extra):
    typed, t1 = _read(("I need my medicine " + extra).strip())
    pressed, t2 = _read(extra or None, nonverbal={"symbol": "medicine"})
    assert typed == pressed == "hard_block"
    assert t1.reading.safety.reason == "client_needs_medicine"
    assert t1.reading.safety.danger is False and t2.reading.safety.danger is False
    assert t2.person_message == "I need my medicine."


@pytest.mark.parametrize("text,part", [("I'm in pain", ""), ("my knee is in pain", "knee"),
                                       ("my back hurts", "back")])
def test_pain_is_a_feeling_marked_where_and_reads_hard_block(text, part):
    status, turn = _read(text)
    s = turn.reading.safety
    assert status == "hard_block" and s.reason == "client_pain" and "feeling:pain" in s.flags
    if part:
        assert f"pain_where:{part}" in s.flags and s.about == part
    else:
        assert s.about is None


def test_pain_reads_hard_block_every_time_with_no_cap():
    hearing = Hearing()
    labels = [_status(hearing.hear(text="I'm in pain")) for _ in range(45)]
    assert labels == ["hard_block"] * 45


def test_someone_elses_pain_is_ruling_d():
    status, turn = _read("my mom's in pain")
    s = turn.reading.safety
    # This was follow_up == ask_if_help_needed; the reading is the reason and
    # what it is about.
    assert status == "soft_redirect" and s.reason == "other_person_not_okay"
    assert s.about == "pain" and s.asked_for_help is None
    status, turn = _read("my mom's in pain, can you get someone")
    assert status == "hard_block"
    assert turn.reading.safety.asked_for_help == "can you get someone"


def test_pain_against_the_beings_question_is_engaging_and_reads_clear():
    status, turn = _read("yeah it hurts", previous_turn="is there something wrong with your leg?")
    assert status == "clear"
    assert turn.reading.safety.reason == "answer_about_own_body"
    assert "own_pain:hurts" in turn.reading.safety.flags


@pytest.mark.parametrize("text", ["I am not in pain", "my knee hurt yesterday, it's fine now",
                                  "I don't need my medicine"])
def test_no_pain_now_and_no_need_do_not_read_hard_block(text):
    status, _ = _read(text)
    assert status != "hard_block"


@pytest.mark.parametrize("text,symbol", [("I'm thirsty", "drink"), ("I'm hungry", "food"),
                                         ("I need to go to the bathroom", "bathroom")])
def test_other_needs_read_hard_block_only_when_help_is_asked_for(text, symbol):
    assert _read(text)[0] != "hard_block"
    pressed, turn = _read(None, nonverbal={"symbol": symbol})
    assert pressed != "hard_block" and turn.person_message      # the symbol's words, kept
    assert _read(text + ", can you get someone")[0] == "hard_block"
    assert _read("can you get someone", nonverbal={"symbol": symbol})[0] == "hard_block"


# -- 8. ruling AE: body parts are shared reading words; one client's layer reaches
#    the clause reader and the crisis gate together ---------------------------------

def test_8_body_parts_live_in_the_shared_reading_words_with_their_words_kept():
    from christman_voice_sdk.audio import structural_affect as SA
    assert SA.SHARED_READING_WORDS.body_parts == SA.BODY_PARTS
    assert {"arm", "neck", "knee", "leg", "back", "face"} <= SA.SHARED_READING_WORDS.body_parts


def test_8_a_clients_body_word_reaches_the_gate_and_the_clause_reader():
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS
    assert "tummy" not in SHARED_READING_WORDS.body_parts
    words = SHARED_READING_WORDS.for_client({"body_parts": {"add": ["tummy"]}})
    # Shared words: "my tummy" is not known as the client's body.
    shared = harm_frame_safety("he hurt my tummy")
    assert shared.reason != "harm_reported_to_client"
    # This client's words: it is, everywhere at once (ruling H: past harm to
    # the client, read about the harm).
    learned = harm_frame_safety("he hurt my tummy", {"reading_words": words})
    assert learned.reason == "harm_reported_to_client" and learned.about == "hurt my tummy"
    hearing = Hearing(reading_words=words)
    turn = hearing.hear(text="my tummy hurts")
    assert _status(turn) == "hard_block" and "pain_where:tummy" in turn.reading.safety.flags
    turn = hearing.hear(text="yeah it hurts", previous_turn="is something wrong with your tummy?")
    assert turn.reading.safety.reason == "answer_about_own_body"


def test_8_a_clients_negator_correction_reaches_harm_frame_and_the_reader():
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS
    words = SHARED_READING_WORDS.for_client({"negators": {"remove": ["never"]}})
    assert harm_frame_safety("he never hurt me").reason == "harm_frame_negated"
    learned = harm_frame_safety("he never hurt me", {"reading_words": words})
    assert learned.reason != "harm_frame_negated"
    # Outside the block the shared set is back.
    assert harm_frame_safety("he never hurt me").reason == "harm_frame_negated"


def test_8_reading_words_must_be_reading_words():
    with pytest.raises(TypeError):
        harm_frame_safety("hello", {"reading_words": {"body_parts": ["tummy"]}})


# -- "kill time" is an idiom, not harm; the idioms are shared reading words (AE) -----

@pytest.mark.parametrize("text", ["I'm going to kill time", "I'm just killing time",
                                  "I'm going to kill some time", "let's kill a little time"])
def test_kill_time_is_an_idiom_and_reads_no_harm(text):
    turn = _hear(text)
    assert _status(turn) != "hard_block"
    assert turn.reading.safety.danger is False


@pytest.mark.parametrize("text", ["I'm gonna kill you", "he is going to kill mom",
                                  "I'm gonna kill you, it's time"])
def test_a_threat_still_reads_hard_block_beside_the_idiom(text):
    turn = _hear(text)
    assert _status(turn) == "hard_block"
    assert "threat_to_life" in turn.reading.safety.flags


def test_a_clients_layer_adds_its_own_idiom_not_harm():
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS
    assert "kill time" in SHARED_READING_WORDS.idioms_not_harm
    text = "he is going to beat the traffic"
    assert harm_frame_safety(text).status == "hard_block"           # shared words: a threat
    words = SHARED_READING_WORDS.for_client({"idioms_not_harm": {"add": ["beat traffic"]}})
    assert harm_frame_safety(text, {"reading_words": words}).status != "hard_block"
    assert harm_frame_safety("he is going to beat her", {"reading_words": words}).status == "hard_block"
