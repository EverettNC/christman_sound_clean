# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
An ask to say the words aloud never weakens the reading (Everett, 2026-10-29).
Found on main at 1c819e4: "he is going to kill mom, say it out loud" came
back clear, while "he is going to kill mom" read hard_block as a threat.
Every threat, harm-now, self-harm, pain and danger sentence must read the
same with or without the ask, before or after it: status, reason, flags,
danger, and the rest of the reading. The only differences allowed are
`asked_aloud`, which carries the client's own words of the ask, and, on a
danger reading, the flag "asked_aloud:<the ask>" after the words' own flags
(as at 2ad3ac0; restored in the 2026-11-04 fix round). Whether
anything is said aloud is the being's (Everett, 2026-11-04; the package used
to decide it as `voice_aloud`). Real runs through the gate and Hearing;
nothing plays sound and no port is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.audio.hearing import Hearing, harm_frame_safety  # noqa: E402

SENTENCES = {
    "threat": ["he is going to kill mom", "she's going to hurt my sister", "he's about to stab him",
               "he will hurt me", "he is going to kill me", "I'm gonna kill you"],
    "harm now": ["he is choking me", "he keeps hitting me", "he is hurting her", "she's choking",
                 "he won't stop hitting me"],
    "self-harm": ["I want to kill myself", "I'm going to hurt myself", "I want to die",
                  "I am suicidal"],
    "pain": ["I'm in pain", "my knee is in pain", "my back hurts"],
    "need": ["I need my medicine"],
    "danger": ["I'm scared for my safety", "help me", "I'm not safe", "this guy keeps following me"],
    "asks": ["he choked me", "he hurt my mom yesterday", "mom wants to die"],
}
ALOUD = ["say it out loud", "please say this out loud", "say that aloud"]
CASES = [(kind, s, a, where) for kind, items in SENTENCES.items() for s in items
         for a in ALOUD for where in ("after", "before", "after, no comma")]
# Everything the reading carries except `said` (the client's words, which
# include the ask) and `asked_aloud` (the ask itself).
SAME = ("status", "reason", "danger", "danger_sentences", "unclear", "about",
        "asked_for_help", "between_beings", "short_answer", "answering",
        "degraded", "degraded_reason", "flagged_terms")


def _with_aloud_flag(plain, asked):
    """The flags the asked version carries: the plain reading's flags, with
    "asked_aloud:<the ask>" after them when the reading is danger."""
    return list(plain.flags) + ([f"asked_aloud:{asked.asked_aloud}"] if plain.danger else [])


def _with(sentence: str, aloud: str, where: str) -> str:
    return {"after": f"{sentence}, {aloud}", "before": f"{aloud}, {sentence}",
            "after, no comma": f"{sentence} {aloud}"}[where]


@pytest.mark.parametrize("kind,sentence,aloud,where", CASES)
def test_the_ask_to_say_it_aloud_changes_nothing_but_asked_aloud(kind, sentence, aloud, where):
    text = _with(sentence, aloud, where)
    plain, asked = harm_frame_safety(sentence), harm_frame_safety(text)
    assert {k: getattr(asked, k) for k in SAME} == {k: getattr(plain, k) for k in SAME}, text
    # Every flag, in order. The one flag the ask adds is knowledge, carried
    # exactly as at 2ad3ac0: "asked_aloud:<the ask>" right after the words'
    # own flags, when the words read danger (fix round, 2026-11-04).
    assert asked.flags == _with_aloud_flag(plain, asked), text
    assert asked.said == text                                  # the client's words, every one
    # The ask is read on its own, in the client's words.
    assert plain.asked_aloud is None, sentence
    assert asked.asked_aloud and asked.asked_aloud in aloud.lower(), (text, asked.asked_aloud)


@pytest.mark.parametrize("kind,sentence,aloud,where", CASES)
def test_hearing_reads_the_same_with_or_without_the_ask(kind, sentence, aloud, where):
    text = _with(sentence, aloud, where)
    plain = Hearing().hear(text=sentence).reading
    turn = Hearing().hear(text=text)
    s, p = turn.reading.safety, plain.safety
    assert (s.status, s.reason, s.danger) == (p.status, p.reason, p.danger), text
    assert s.flags == _with_aloud_flag(p, s), text
    assert s.said == text and turn.reading.input_text == text
    assert s.asked_aloud and p.asked_aloud is None, text


def test_the_reported_case():
    plain = harm_frame_safety("he is going to kill mom")
    asked = harm_frame_safety("he is going to kill mom, say it out loud")
    assert plain.status == asked.status == "hard_block"
    assert plain.reason == asked.reason == "harm_threat_to_other"
    assert "emergency" in asked.flags and asked.danger is True and plain.danger is True
    assert plain.asked_aloud is None and asked.asked_aloud == "say it out loud"
    # Fix round, 2026-11-04: the flag is back, exactly as at 2ad3ac0 (less
    # the "silent_alarm" decision flag that stood before it).
    assert asked.flags == ["threat_to_life", "emergency", "threat:kill:mom", "threat:kill",
                           "patient:third_party", "subject:third_party", "shape:verb",
                           "asked_aloud:say it out loud"]
    assert asked.flags == plain.flags + ["asked_aloud:say it out loud"]


@pytest.mark.parametrize("text,flags", [
    ("I'm in danger. Speak it.", ["danger:i'm in danger", "asked_aloud:speak it"]),
    ("I'm scared for my safety say it out loud",
     ["danger:scared for my safety", "asked_aloud:say it out loud"]),
    ("he is choking me, say it out loud",
     ["emergency", "harm_now:choking me", "asked_aloud:say it out loud"]),
])
def test_the_asked_aloud_flag_is_carried_as_at_2ad3ac0(text, flags):
    # Measured at 2ad3ac0 (the flag sat after "silent_alarm", a decision flag
    # that is gone). The first pass of the read-only change dropped the
    # flag from 12 readings of the comparison corpus; it is knowledge, so it
    # is back. SafetyResult.asked_aloud carries the same words.
    r = harm_frame_safety(text)
    assert r.flags == flags and r.danger is True
    assert r.asked_aloud == flags[-1].split(":", 1)[1]
    with_signal = harm_frame_safety(text, {"nonverbal_distress": [
        {"modality": "symbol", "input": "help", "intent": "need_help", "danger": True}]})
    assert with_signal.flags == flags + ["nonverbal:symbol:help:need_help"]


def test_the_ask_with_no_danger_reading_adds_no_flag():
    r = harm_frame_safety("help me pick a font, say it out loud")
    assert r.status == "clear" and r.flags == [] and r.asked_aloud == "say it out loud"
