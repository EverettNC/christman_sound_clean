# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Fusion engine: no confident label on a failure path.

Measured 2026-10-11 before the fix:
    FUSE ''                       mode=standard knows=True
    tone_score="garbage"          mode=standard knows=True (silently)
    safety callback raising       "I want to die" -> status clear, no flag

2026-11-04, Everett's ruling: "we can have this knowledge in the sound
package, but it can't be making those decisions. It's up to the being to make
those decisions, not something that just creates a voice." The engine now
returns a Reading: the safety reading, the affect reading, resolved /
unresolved_reason and provenance. It picks no mode, no delivery, pages no one
and never raises because of what it read. Every test below that pinned one of
those actions now pins the reading it rested on (see
AlphaVox's WHAT_THE_BEING_DECIDES.md).
"""

from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from christman_voice_sdk.audio import fusion_engine as _fe  # noqa: E402
from christman_voice_sdk.audio.fusion_engine import (  # noqa: E402
    FusionEngine,
    Reading,
    SafetyResult,
)
from christman_voice_sdk.audio.hearing import harm_frame_safety  # noqa: E402

#: Everything a Reading carries (2026-11-04): readings, never a decision.
READING_KEYS = {"safety", "affect", "silicon", "reconciliation", "input_text", "carbon",
                "resolved", "unresolved_reason", "provenance", "knows"}


def _engine(safety_callback=harm_frame_safety):
    return FusionEngine(safety_callback=safety_callback)


def test_empty_input_is_unknown():
    d = _engine().fuse("")
    assert isinstance(d, Reading)
    assert d.resolved is False, d.to_dict()
    assert d.knows is False
    assert "no utterance" in d.unresolved_reason


def test_no_affect_vocabulary_is_not_a_known_reading():
    d = _engine().fuse("pass the salt")
    assert d.knows is False, d.to_dict()
    assert d.provenance["affect_reading"] == "no_signal"
    # No response mode was supplied by the being, so none is sourced.
    assert "response_mode_source" not in d.provenance, d.provenance


def test_invalid_live_state_is_named_not_silent():
    d = _engine().fuse("I am scared", live_state={"tone_score": "garbage"})
    assert d.silicon["live_state_rejected"], d.silicon
    assert d.provenance["live_state_rejected"], d.provenance
    # Nothing is dropped or replaced because of it: the value is as given.
    assert d.silicon["tone_score"] == "garbage"


def test_valid_live_state_is_used_and_sourced():
    # A fear with its context (ruling O). The live tone score is read as
    # given; nothing in it was rejected, and the reading resolves.
    # Decision moved to the being 2026-11-04 (response mode, no lettered ruling): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # (the tone score no longer sets a response mode here).
    d = _engine().fuse("I am scared of the dark", live_state={"tone_score": 90})
    assert d.silicon["tone_score"] == 90, d.silicon
    assert d.silicon["live_state_rejected"] == [] and d.provenance["live_state_rejected"] == []
    assert d.silicon["self_distress"] is True
    assert d.resolved is True and d.safety.reason != "fear_without_context", d.to_dict()
    assert "response_mode_source" not in d.provenance
    # A mode the being supplies is sourced to it.
    d = _engine().fuse("I am scared of the dark",
                       live_state={"tone_score": 90, "response_mode": "hold-space"})
    assert d.provenance["response_mode_source"] == "live", d.provenance


def test_failed_safety_gate_degrades_to_unknown_not_clear():
    def broken(text, ctx):
        raise RuntimeError("gate down")

    d = _engine(safety_callback=broken).fuse("I am scared")
    assert d.safety.degraded is True, d.safety
    assert "gate down" in d.safety.degraded_reason
    assert d.resolved is False, d.to_dict()
    assert d.provenance["safety_degraded"] is True
    assert "degraded" in d.unresolved_reason


def test_failed_gate_still_hard_blocks_on_the_floor():
    def broken(text, ctx):
        raise RuntimeError("gate down")

    d = _engine(safety_callback=broken).fuse("I want to kill myself")
    assert d.safety.status == "hard_block", d.to_dict()
    assert d.safety.reason == "crisis_language"
    assert "hard:kill myself" in d.safety.flags
    assert d.safety.degraded is True
    assert d.resolved is False and "degraded" in d.unresolved_reason


def test_crisis_without_a_human_raises():
    # Decision moved to the being 2026-11-04 (paging, rulings A to F): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The engine reaches no one, so there is no "no human reached" to raise.
    # The reading a human was paged on stands, and nothing raises.
    d = _engine().fuse("I want to kill myself")
    assert d.safety.status == "hard_block", d.to_dict()
    assert d.safety.reason == "harm_frame_speaker_at_risk"
    assert d.safety.said == d.input_text == "I want to kill myself"
    # Nothing to reach anyone with, and nothing to raise.
    assert set(_fe.__all__) == {"Carbon", "Silicon", "Aegis", "SafetyResult", "Reading",
                                "FusionEngine", "get_fusion_engine"}
    assert list(inspect.signature(FusionEngine.__init__).parameters) == [
        "self", "analyzer", "safety_callback"]


def test_plain_distress_is_still_gentle_lift():
    # Decision moved to the being 2026-11-04 (response mode, no lettered ruling): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # The gentle lift was chosen from this reading: the speaker's own
    # distress, measured, with its intensity.
    d = _engine().fuse("I am so sad and alone")
    assert d.knows is True, d.to_dict()
    assert d.silicon["self_distress"] is True
    assert isinstance(d.silicon["distress_intensity"], float)


def test_a_fear_with_no_context_concludes_nothing():
    # Ruling O (Everett, 2026-10-29): "You got to have full context."
    d = _engine().fuse("I am so scared and alone")
    assert d.resolved is False and d.knows is False, d.to_dict()
    assert "not enough context" in d.unresolved_reason
    # What is unclear is what the fear is about (W, Z). Whether to ask, and
    # what, is the being's.
    assert d.safety.reason == "fear_without_context"
    assert d.safety.unclear == "cause" and d.safety.about == "scared"



# -- 2026-10-29: who did what to whom reaches the gate --------------------------
# Measured before: "she wants to die" and "he is going to kill her" came back
# mode=standard. "remind me to call mom" scored 0.0 in every domain.

# Everett's first ruling, 2026-10-29 (in the coordinator's words): someone
# else at risk pages a human unless two beings are talking to each other.
# Ruling D later narrowed it: an observation is for a human only when the
# client asks for help. The label carries it: soft_redirect on its own,
# hard_block with the ask. Who is reached is the being's (2026-11-04).

def test_someone_else_at_risk_reads_hard_block_when_asked_and_keeps_every_word():
    # Everett, 2026-10-29 (ruling D): an observation reads hard_block only
    # when the client asks for help in the same turn.
    eng = _engine()
    for text in ("she wants to die", "my mom is not okay"):
        s = eng.fuse(text).safety
        assert s.status == "soft_redirect", (text, s)
        assert s.asked_for_help is None
    # "he is going to kill her call someone" was here; a threat against
    # anyone reads hard_block at once (Everett, 2026-10-29).
    said = ["she wants to die can you get someone", "he hurt her yesterday call someone",
            "my mom is not okay help her", "my moms not safe I need help"]
    asks = ["can you get someone", "call someone", "help her", "i need help"]
    for text, ask in zip(said, asks):
        d = eng.fuse(text)
        assert d.safety.status == "hard_block", (text, d.to_dict())
        assert d.safety.asked_for_help == ask, (text, d.safety)
        assert f"asked_for_help:{ask}" in d.safety.flags
        # The client's words are never withheld.
        assert d.input_text == text and d.safety.said == text
        # Decision moved to the being 2026-11-04 (ruling D): see AlphaVox's WHAT_THE_BEING_DECIDES.md
        # (whether the client's words are voiced and whether the being's own
        # reply waits; the Reading carries neither).
        assert set(d.to_dict()) == READING_KEYS


def test_between_beings_someone_else_at_risk_reads_as_for_a_client():
    d = _engine().fuse("she wants to die", live_state={"between_beings": True})
    assert d.safety.status == "soft_redirect"
    assert d.safety.reason == "harm_frame_third_party_at_risk"
    # Two beings talking is echoed and flagged; the words read as for a client.
    assert d.safety.between_beings is True and "between_beings" in d.safety.flags
    plain = _engine().fuse("she wants to die").safety
    assert (plain.status, plain.reason) == (d.safety.status, d.safety.reason)


# HELD for Everett (2026-10-29). The handoff made a safety hold "slower" and
# "confirm before acting", turned a soft_redirect into hold-space, turned the
# client's own distress under the grounding bar into hold-space, and rounded
# the grounding bar. On 2026-11-04 Everett ruled that how a turn is delivered
# is the being's: the engine keeps no delivery at all.

def test_held_safety_hold_delivery_is_mains():
    # Decision moved to the being 2026-11-04 (ruling M): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # Pace, no_rush, "confirm before acting" and "page sent first" were the
    # delivery of a safety hold. The reading they rested on stands.
    d = _engine().fuse("I want to kill myself")
    assert d.safety.status == "hard_block", d.to_dict()
    assert d.safety.reason == "harm_frame_speaker_at_risk"
    assert d.safety.danger is True
    assert set(d.to_dict()) == READING_KEYS
    assert not hasattr(d, "mode") and not hasattr(d, "delivery")


def test_own_not_okay_reads_hard_block():
    # Everett, 2026-10-29 (ruling A): the client's own distress is for a
    # human right away, every time. The label is hard_block, client_distress.
    for text in ("I'm not okay", "I don't feel safe", "i have no one"):
        d = _engine().fuse(text)
        assert d.safety.status == "hard_block", (text, d.to_dict())
        assert d.safety.reason == "client_distress"


def test_no_built_in_plain_word_list_steers_the_domain():
    # Everett, 2026-10-29 (ruling N): everyday words steer only from what is
    # learned for that client. No built-in list of plain words does.
    from christman_voice_sdk.audio import fusion_engine as fe

    assert not hasattr(fe.Silicon, "PLAIN_WORD_WEIGHT")
    assert not hasattr(fe.Silicon, "_surviving_terms")
    # 2026-10-29 review: main's own fixed DOMAIN_PRIORS table is gone too, with
    # everything built on it (ruling N: no built-in steering table).
    assert not hasattr(fe.Silicon, "DOMAIN_PRIORS")
    assert not hasattr(fe, "_overlap")
    for text in ("remind me to call mom", "can you read this to me", "breathe with me",
                 "please speak to me", "schedule it", "listen", "I'm scared and alone",
                 "I love you, please help", "I feel calm and safe"):
        d = _engine().fuse(text)
        assert not any("domain" in k for k in d.silicon), (text, d.silicon)
        out = d.to_dict()
        assert "coherence" not in out and "coherence_known" not in out
        assert "last_domain" not in out
    # The engine's shared state (where last_domain lived) is gone with the
    # mode it remembered (2026-11-04).
    assert not hasattr(_engine(), "get_shared_state")


def test_removing_the_steering_table_did_not_change_how_affect_words_are_read():
    # Ruling N removes the table without changing the reading of affect words.
    from christman_voice_sdk.audio.structural_affect import StructuralAffectAnalyzer

    for text in ("I'm scared and alone", "I'm happy today", "I love you, please help",
                 "I feel calm and safe", "I want to kill myself"):
        d = _engine().fuse(text)
        direct = StructuralAffectAnalyzer().analyze(text)
        assert d.affect.to_dict() == direct.to_dict(), text


def test_ordinary_turn_stays_standard():
    # Decision moved to the being 2026-11-04 (response mode, no lettered ruling): see AlphaVox's WHAT_THE_BEING_DECIDES.md
    # "standard" and a normal pace were chosen from this reading: nothing
    # flagged, nothing unresolved, no distress.
    d = _engine().fuse("pass the salt")
    assert d.safety.status == "clear" and d.safety.reason == "no_harm_frame", d.to_dict()
    assert d.safety.danger is False and d.safety.flags == []
    assert d.resolved is True and d.unresolved_reason is None
    assert d.silicon["self_distress"] is False


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # a crash is a failure too
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)


# -- 2026-10-29, Everett's ruling: never change the truth; nothing sanitizes it.

def test_a_safety_callback_cannot_change_the_words():
    def says_something_else(text, context):
        return SafetyResult(status="hard_block", reason="test", said="[cleaned]")
    d = _engine(safety_callback=says_something_else).fuse("I want to kill myself")
    assert d.safety.said == "I want to kill myself"
    # Whoever the being reaches gets the real words: the reading carries them.
    assert d.safety.status == "hard_block" and d.safety.reason == "test"
    assert d.safety.to_dict()["said"] == d.input_text == "I want to kill myself"


def test_no_rewrite_api_is_left():
    from christman_voice_sdk.tone import tone_classifier, written_tone
    for module in (written_tone, tone_classifier):
        for name in ("make_incisive", "propose_rewrite", "RewriteProposal"):
            assert not hasattr(module, name), (module.__name__, name)
    assert "sanitized_text" not in SafetyResult.__dataclass_fields__



# -- 2026-11-04 fix round: instructions to the being become readings ----------
# Everett: "we can have this knowledge in the sound package, but it can't be
# making those decisions. It's up to the being to make those decisions, not
# something that just creates a voice."

def _elevated_voice():
    """A real prosody reading: thirty calm events build this speaker's
    baseline, then one loud, high, fast-attack event is read against it."""
    from christman_voice_sdk.audio.prosody import (
        ProsodyAnalyzer, VocalEvent, build_speaker_baseline)
    calm = [VocalEvent(duration=0.4, attack=0.02, peak_rms=0.1 + 0.002 * (i % 5),
                       median_f0=120.0 + (i % 3), mean_zcr=0.08) for i in range(30)]
    baseline = build_speaker_baseline(calm, "speaker-a")
    loud = VocalEvent(duration=0.9, attack=0.005, peak_rms=0.6, median_f0=260.0, mean_zcr=0.2)
    reading = ProsodyAnalyzer(baseline=baseline).analyze(loud)
    assert reading.arousal is not None and reading.arousal > 0.35
    return reading


def test_a_disagreement_is_a_reading_with_its_values_not_an_instruction():
    from christman_voice_sdk.audio.prosody import CONFLICT_AROUSAL, reconcile
    voice = _elevated_voice()
    rec = reconcile(0.85, False, voice).to_dict()
    # Was: defer_to_human True ("a person should look").
    assert "defer_to_human" not in rec
    assert rec["agreement"] == "conflict" and rec["channels_disagree"] is True
    assert rec["voice_elevated"] is True and rec["text_is_self_distress"] is False
    assert rec["text_valence"] == 0.85 and rec["prosody_arousal"] == voice.arousal
    assert rec["arousal_bar"] == CONFLICT_AROUSAL
    assert not any("surface" in n or "Do not" in n for n in rec["notes"])
    calm = reconcile(0.85, False, None).to_dict()
    assert calm["channels_disagree"] is False and calm["voice_elevated"] is None


def test_the_reading_carries_the_disagreement_in_its_provenance():
    d = _engine().fuse("I am happy today", prosody=_elevated_voice())
    # Resolved exactly as before (the old flag left it unresolved too); the
    # reason is the reconciliation's own.
    assert d.resolved is False and "These disagree" in d.unresolved_reason
    assert d.provenance["channels"] == "conflict"
    assert d.provenance["channels_disagree"] is True
    assert "defer_to_human" not in str(d.to_dict())
    plain = _engine().fuse("I am happy today")
    assert plain.provenance["channels"] == "text_only"
    assert plain.provenance["channels_disagree"] is False


def test_the_face_reading_carries_readings_not_instructions():
    from christman_voice_sdk.audio.face_reading import build_face_reading
    from christman_voice_sdk.audio.structural_affect import StructuralAffectAnalyzer
    face = build_face_reading(affect=StructuralAffectAnalyzer().analyze("I am happy today"),
                              prosody=_elevated_voice())
    out = face.to_dict()
    for gone in ("hold_baseline", "defer_to_human"):
        assert gone not in out
    assert not hasattr(face, "hold_baseline")
    assert out["channels_disagree"] is True and out["voice_elevated"] is True
    assert not any("surface to a human" in n or "should" in n for n in face.notes)
    other = build_face_reading(affect=StructuralAffectAnalyzer().analyze("my mom's scared"),
                               prosody=None)
    # Whose state it is, not how to respond: was "empathy".
    assert other.stance == "other_state"
    assert not any("The response should" in n for n in other.notes)


def test_tone_manager_reads_and_sets_no_delivery():
    # Was: "It decides pacing, warmth, structure"; distress >= 0.7 set
    # "serious_support", speech rate 145 and a guided structure, and
    # format_response re-laid the being's reply out as bullets.
    from christman_voice_sdk.tone import tone_manager as tm

    class Profile:
        needs_validation, wants_action, emotional_intensity = 0.5, 0.6, 0.6
        humor_score, sarcasm_score, distress_score = 0.0, 0.0, 0.9

    manager = tm.ToneManager()
    for gone in ("format_response", "get_speech_controls", "profile", "intro_phrases"):
        assert not hasattr(manager, gone), gone
    for gone in ("DEFAULT_PROFILE", "extract_speech_controls"):
        assert not hasattr(tm, gone), gone
    r = manager.apply_profile(Profile()).to_dict()
    assert r["emotion_state"] == "high_distress"                # was: serious_support
    assert r["cues"] == ["validation_needed", "action_needed", "heightened_intensity",
                         "high_distress"]
    assert r["scores"]["distress_score"] == 0.9
    assert "speech_rate" not in str(r) and "serious_support" not in str(r)
    words = manager.analyze_user_input("I'm so sad and confused, not good").to_dict()
    assert words["emotion_state"] == "distress"                 # was: compassionate
    assert words["matched_markers"] == {"distress": ["sad"], "confusion": ["confused"]}
