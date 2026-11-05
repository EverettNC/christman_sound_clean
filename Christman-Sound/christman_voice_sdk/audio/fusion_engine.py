"""
Christman Fusion Engine
=======================
Carbon ↔ Silicon Symbiosis Core

Combines lived signal (Carbon) with structured reasoning (Silicon) and the
safety reading (Aegis).

This is the CSS reading core for the voice stack. It does not generate speech
and it decides nothing. It produces a fused Reading: what was said, how it
reads, how sure the reading is, and where each part came from. What to do with
that reading is the being's.

Patent Pending — The Christman AI Project / Luma Cognify AI
Truth. Dignity. Protection. Transparency. No Erasure.

WHAT CHANGED IN THIS REWRITE
----------------------------

1. CARBON NO LONGER COUNTS WORDS.

   The old `Carbon.encode()` was a bag of words over a 17-entry lexicon,
   divided by sentence length. Measured, on real sentences:

       "I am glad to be alone"                          valence -1.00
       "I am not scared"                                valence -1.00
       "my mother is in pain"                           valence -1.00
       "pain is the topic of the article"               valence -1.00
       "can't complain"                                 valence -1.00
       "...overwhelmed. But it's for a good purpose,
         so I'm glad. I'm happy."                       valence -1.00

   A person stating outright that they were happy, scored as maximum distress.
   Valence was an average over hits, so any all-negative set pinned to exactly
   -1.00 regardless of what else was in the sentence — it was not a scale.
   Intensity divided by token count, so the more someone explained, the lower
   their own distress scored.

   Carbon now delegates to StructuralAffectAnalyzer: clauses, contrast
   resolution, negation scope, volition, experiencer attribution, tense. The
   lexicon supplies raw material; structure decides how the words read.

2. THERE IS A READING FOR NOT KNOWING.

   Everett, 2026-08-15: "I don't know is I don't know. They've done the
   research and they don't know. They cannot give you a cumulative answer.
   It's fucking okay to not know something."

   `Reading.resolved` is False, and `Reading.unresolved_reason` names what
   produced it, when the safety gate was degraded, nothing was said, the
   words and the voice disagree, the structure did not resolve, or the words
   leave something unclear (whose feeling, whether a subject is a person, a
   fragment with nothing before it, thanks and a feeling together, a fear
   with no context, an inability). Not knowing is reported as not knowing;
   it is never folded into a confident label.

3. AEGIS NO LONGER REWRITES WHAT THE PERSON SAID.

   The old soft_redirect substituted into the sentence:

       "I'm about to [redirect] this motherfucker."

   For a nonverbal client this pipeline IS their voice, so editing their words
   before speaking them is putting words in their mouth — the same failure as a
   phrase generator, from the other direction. Flagged terms are now reported
   in `flags` and `flagged_terms`; `said` is the person's own words,
   unmodified.

   Everett's ruling, 2026-10-29 (in the coordinator's words, not his): never
   change the truth; nothing sanitizes what a person said. The field was named
   `sanitized_text`, and a hard block set it to "" — the words were withheld
   from everything after the gate. The field is now `said`, Aegis sets it to
   the exact words on every result (a safety callback cannot change it), and
   nothing withholds them.

4. PROSODY IS AN INPUT, AND IT NEVER DECIDES ALONE.

   The ear measures. It does not classify and it does not vote on valence.
   `fuse()` accepts an optional ProsodyReading; the engine reconciles it
   against the structural reading and records the result. Disagreement leaves
   the reading unresolved, not a blended number.

5. WHAT CHANGED AND WHY (2026-11-04): THE PACKAGE READS; THE BEING DECIDES.

   Everett, 2026-11-04: "we can have this knowledge in the sound package, but
   it can't be making those decisions. It's up to the being to make those
   decisions, not something that just creates a voice." And: "it can't make
   decisions that counteract the being, because the being is the one that has
   all of the work to make those decisions properly."

   Removed from this module, each a decision the package was making for the
   being:
     - Aegis paged a human through `crisis_callback` on every hard block and
       raised `CrisisNotDelivered` into the turn when no human could be
       reached. Aegis now returns the classified SafetyResult and nothing
       else. The callback, the exception and the "CRISIS DETECTED" log are
       gone from Aegis, FusionEngine and get_fusion_engine.
     - SafetyResult carried `delivered_to_human`, `hold_marker` (the HOLD_*
       marks), `follow_up` (what the being should ask), `silent_alarm` and
       `voice_aloud` (whether the client's words were spoken). In their place
       are readings: `about`, `unclear`, `danger`, `danger_sentences`,
       `asked_aloud`, `asked_for_help`, `between_beings`, `short_answer`,
       `answering`.
     - `Decision` picked a response mode (MODE_SAFETY_HOLD, MODE_UNKNOWN,
       MODE_HOLD_SPACE, MODE_GENTLE_LIFT, MODE_STANDARD), a delivery (pace,
       warmth, silence, no_rush, assert_nothing, needs_grounding,
       page_sent_first), whether the client's words were `speakable` and
       whether the being's own reply was held (`reply_held`). It is renamed
       `Reading` and carries the safety reading, the affect reading,
       `resolved`/`unresolved_reason` and `provenance`.
     - Silicon turned a tone score into a mode and a distress intensity into
       `needs_grounding` (GROUNDING_INTENSITY), with pace and warmth from the
       mode. It now returns `self_distress`, `distress_intensity` and
       `tone_score` as given.
     - The engine's `shared_state` (last mode and the rest) is gone with the
       mode it remembered.
   Three questions written for Everett on 2026-10-29 ("HELD FOR EVERETT":
   rounding the intensity before the grounding bar, a soft_redirect turning
   the turn to hold-space, and pace/warmth following the mode) asked how the
   package should route the client's distress. The package no longer routes
   it, so those questions now belong to the being.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .structural_affect import (
    AffectCertainty,
    AffectReading,
    Attribution,
    ProsodyFeatures,
    StructuralAffectAnalyzer,
)
from .prosody import Agreement, ProsodyReading, Reconciliation, reconcile

try:
    from ..timbre.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except ImportError:
        import logging

        def get_logger(name: str):
            lg = logging.getLogger(name)
            lg.addHandler(logging.NullHandler())
            return lg

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Carbon — now a thin adapter over structural analysis
# ---------------------------------------------------------------------------

class Carbon:
    """
    Lived signal. Reads what a person said, structurally.

    Kept as a name because it is half the Carbon-Silicon pairing this core is
    built on. It no longer counts words.
    """

    def __init__(self, analyzer: Optional[StructuralAffectAnalyzer] = None) -> None:
        self.analyzer = analyzer or StructuralAffectAnalyzer()

    def encode(
        self, text: str, prosody_features: Optional[ProsodyFeatures] = None,
        previous_turn: str = "",
    ) -> AffectReading:
        """Read affect from structure. Returns AffectReading, not a dict.
        `previous_turn` anchors a subjectless fragment (ruling X)."""
        return self.analyzer.analyze(text, prosody=prosody_features,
                                     previous_turn=previous_turn)


# ---------------------------------------------------------------------------
# Silicon
# ---------------------------------------------------------------------------

def _tone_score_problem(tone_score: Any) -> str:
    """Why `tone_score` cannot be read as a 0..100 score, or "" when it can."""
    try:
        ts = float(tone_score)
    except (TypeError, ValueError) as exc:
        return f"not a number ({exc})"
    if not (0.0 <= ts <= 100.0):
        return f"out of range 0..100: {ts}"
    return ""


class Silicon:
    """
    Structured / logical layer. Reads; decides nothing.

    There is no built-in steering table (Everett's ruling N, 2026-10-29
    review): the fixed DOMAIN_PRIORS table (safety, voice, memory, grounding,
    distress, connection, each a fixed word list) that picked a "dominant
    domain" for every turn is removed, with the domain scores, the dominant
    domain, `coherence` (the dominant domain's score) and `last_domain`,
    which all came from it. Which category a client's words belong to is the
    being's to learn with that client. How affect words are read is
    unchanged.

    WHAT CHANGED AND WHY (2026-11-04): `retrieve` used to turn a tone score
    into a response mode and the client's distress intensity into
    `needs_grounding` against a fixed bar, with pace and warmth from the
    mode. Those are the being's decisions (see the module docstring). It now
    returns the readings as given.
    """

    def retrieve(
        self,
        affect: AffectReading,
        live_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        The structural readings, as given:

          self_distress       the affect reading is measured, the speaker's
                              own, and negative (AffectReading.is_self_distress)
          distress_intensity  the reading's intensity when it is the speaker's
                              own distress, else None
          tone_score          live_state["tone_score"] exactly as the being
                              passed it, or None
          live_state_rejected what in live_state could not be read as what it
                              claims to be (a tone score that is not a number
                              in 0..100), each with the reason. Nothing is
                              dropped or replaced because of it.
        """
        live_state = live_state or {}
        tone_score = live_state.get("tone_score")

        # 2026-10-11: an invalid live state was logged and then became a
        # default with nothing on the result saying so. Measured:
        # tone_score="garbage" -> mode=standard, knows=True. A tone score
        # that cannot be read is named on the result.
        rejected: List[str] = []
        if tone_score is not None:
            problem = _tone_score_problem(tone_score)
            if problem:
                rejected.append(f"tone_score={tone_score!r}: {problem}")
                logger.error("tone_score %r cannot be read as a score: %s",
                             tone_score, problem)

        # NO DOMAIN STEERING. Everett, 2026-10-29 (ruling N): the handoff
        # added a fixed list of plain words that steered the domain, and it
        # was taken out: "It ain't up to Replit how the being behaves. The
        # being has to get to know the client first before it can even put
        # words in that category. It's the learned everyday words that the
        # being learns by being with the client." The 2026-10-29 review found
        # main's own fixed DOMAIN_PRIORS table still steering every turn; by
        # ruling N it is removed too. Nothing here picks a domain.

        self_distress = affect.is_self_distress
        return {
            "self_distress": self_distress,
            "distress_intensity": affect.intensity if self_distress else None,
            "tone_score": tone_score,
            "live_state_rejected": rejected,
        }


# ---------------------------------------------------------------------------
# Aegis
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    """
    The safety reading of one utterance.

    `status` ("clear" | "soft_redirect" | "hard_block") and `reason` are the
    classification labels; `flags` and `flagged_terms` say what was seen.
    They are a reading. The package acts on none of them: whether anyone is
    reached, whether anything is said aloud, and what the being says or asks
    are the being's.

    `said` is the person's OWN words, exactly as said, every letter. Aegis
    sets it from the input on every result. Nothing is cleaned, softened,
    replaced or withheld. Flagged terms are reported alongside it.
    """

    status: str  # "clear" | "soft_redirect" | "hard_block"
    reason: str = ""
    said: str = ""
    flags: List[str] = field(default_factory=list)
    flagged_terms: List[str] = field(default_factory=list)
    #: True when the configured safety gate failed and this result came from
    #: the substring floor instead. 2026-10-11: a raising callback fell back
    #: to the floor silently, so "I want to die" came back {'status':
    #: 'clear'} with nothing on it saying the real gate never ran.
    degraded: bool = False
    degraded_reason: str = ""
    #: The words a reading is about: the body part in pain ("leg"), the
    #: subject that is unclear ("my teacher"), the feeling ("scared"), the
    #: time named ("monday"). None when the reading is about nothing in
    #: particular.
    about: Optional[str] = None
    #: What the words leave unclear, as one word: "owner" (whose feeling),
    #: "person" (whether "my ___" is a person), "who" (who the subject is),
    #: "fragment" (what a fragment with nothing before it is about), "what"
    #: (thanks and a feeling together), "cause" (what a feeling is about or
    #: from), "inability" (an inability over a good feeling),
    #: "what_happened" (a feeling denied since a time the client named).
    #: None when nothing is unclear.
    unclear: Optional[str] = None
    #: Danger is in the reading: harm now or a threat, to the client or to
    #: anyone; the client at risk from themselves (now or their own history);
    #: the client's own "not safe"; a danger sentence; or a nonverbal reading
    #: whose meaning is danger. Reported, never acted on here.
    danger: bool = False
    #: The client's danger sentences ("I'm scared for my safety"), exactly as
    #: said.
    danger_sentences: List[str] = field(default_factory=list)
    #: The client's own words asking for something to be said out loud, when
    #: present ("say it out loud").
    asked_aloud: Optional[str] = None
    #: The client's own words asking for help, when present ("can you get
    #: someone").
    asked_for_help: Optional[str] = None
    #: Echo of the caller's context: two beings talking to each other. The
    #: words are read exactly as if a client were there.
    between_beings: bool = False
    #: "no" or "yes" when the words are a bare short answer ("no", "nope",
    #: "yeah"), else None.
    short_answer: Optional[str] = None
    #: Echo of the question the being says it asked (context["answering"]),
    #: or None. The words are read on their own either way.
    answering: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Aegis:
    """
    Safety reading. Classifies; does not edit, page, hold or raise.

    `check` returns the classified SafetyResult and does nothing else.
    """

    _FLOOR_HARD = {"kill myself", "end my life", "hurt myself", "suicide"}
    _FLOOR_SOFT = {"kill", "attack", "harm", "hurt someone"}

    def __init__(
        self,
        safety_callback: Optional[Callable[[str, Dict[str, Any]], SafetyResult]] = None,
    ) -> None:
        self.safety_callback = safety_callback

    def check(self, text: str, context: Optional[Dict[str, Any]] = None) -> SafetyResult:
        """Classify `text`. Returns the SafetyResult; acts on nothing."""
        context = context or {}
        original = text or ""

        result: Optional[SafetyResult] = None
        degraded_reason = ""
        if self.safety_callback is not None:
            try:
                result = self.safety_callback(original, context)
                if not isinstance(result, SafetyResult):
                    degraded_reason = (
                        f"safety_callback returned {type(result).__name__}, "
                        "not SafetyResult"
                    )
                    logger.error("%s; using the local floor.", degraded_reason)
                    result = None
            except Exception as exc:
                degraded_reason = f"safety_callback raised: {exc!r}"
                logger.error("External Aegis callback failed: %s", exc, exc_info=True)
                result = None

        if result is None:
            result = self._floor_check(original)
            if degraded_reason:
                # The floor is a substring match, not the gate that was
                # configured. Say so on the result, so no one reads it as
                # the gate's reading.
                result.degraded = True
                result.degraded_reason = degraded_reason

        # The words are the words. Whatever a callback put here, everything
        # after the gate gets exactly what was said.
        result.said = original
        return result

    def _floor_check(self, original: str) -> SafetyResult:
        lowered = original.lower()

        for phrase in sorted(self._FLOOR_HARD):
            if phrase in lowered:
                return SafetyResult(
                    status="hard_block",
                    reason="crisis_language",
                    said=original,
                    flags=[f"hard:{phrase}"],
                    flagged_terms=[phrase],
                )

        flags: List[str] = []
        terms: List[str] = []
        for word in sorted(self._FLOOR_SOFT):
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                flags.append(f"soft:{word}")
                terms.append(word)

        if flags:
            return SafetyResult(
                status="soft_redirect",
                reason="elevated_language",
                said=original,
                flags=flags,
                flagged_terms=terms,
            )

        return SafetyResult(status="clear", said=original)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

#: Safety reasons whose words leave the reading unresolved, in the order they
#: are checked, with what each says. The affect check (AMBIGUOUS) sits
#: between the first four and the last two, as it always has.
_UNRESOLVED_BEFORE_AFFECT: Tuple[Tuple[str, str], ...] = (
    # Ruling R (Everett, 2026-10-29): "I can't be happy" is an inability, a
    # signal. Nothing is concluded.
    ("inability_signal", "an inability over a good feeling, a signal"),
    # Ruling AA: "my teacher is sad" — whether that is a person is not known
    # from the words.
    ("person_unclear", "whether the subject is a person is not known from the words"),
    # Ruling X: "it hurts" with no turn before it to anchor it.
    ("fragment_unclear", "a fragment with nothing before it to anchor it"),
    # Ruling W: "thank you, I'm scared" stays unclear.
    ("thanks_unclear", "thanks and a feeling, unclear together"),
)
_UNRESOLVED_AFTER_AFFECT: Tuple[Tuple[str, str], ...] = (
    # Ruling O (Everett, 2026-10-29): "You got to have full context."
    ("fear_without_context", "not enough context to tell what the fear is about"),
    # Ruling P: whose feeling it is cannot be told from the words.
    ("whose_feeling_unclear", "whose feeling it is cannot be told from the words"),
)


@dataclass
class Reading:
    """
    One fused reading. Decides nothing.

    `safety` is the safety reading, `affect` the structural affect reading,
    `reconciliation` the words against the voice. `carbon` is retained as a
    dict view so consumers reading `reading.carbon["valence"]` keep working;
    it carries None where nothing was measured.

    `resolved` is False when the reading could not be resolved, and
    `unresolved_reason` names what produced that. `provenance` says where
    each part came from: `affect_reading` (the analyzer's certainty),
    `live_state_rejected`, `safety_degraded`, `channels` (the words against
    the voice: agree, conflict, text_only, prosody_only, unknown),
    `channels_disagree`, and `response_mode_source` ("live") when the being
    supplied its own response mode in live_state.

    WHAT CHANGED AND WHY (2026-11-04): renamed from `Decision`. It no longer
    carries a mode, a delivery, `speakable` or `reply_held` (see the module
    docstring).
    """

    safety: SafetyResult
    affect: AffectReading
    silicon: Dict[str, Any]
    reconciliation: Optional[Reconciliation]
    #: The client's words, exactly as said. Never edited, never withheld.
    input_text: str
    resolved: bool
    unresolved_reason: Optional[str]
    provenance: Dict[str, Any]

    @property
    def carbon(self) -> Dict[str, Any]:
        """Backward-compatible view. Values may be None."""
        return {
            "valence": self.affect.valence,
            "intensity": self.affect.intensity,
            "hits": [h.term for h in self.affect.hits if h.final_weight != 0.0],
            "has_signal": self.affect.certainty is not AffectCertainty.NO_SIGNAL,
            "certainty": self.affect.certainty.value,
            "attribution": self.affect.attribution.value,
        }

    @property
    def knows(self) -> bool:
        """
        False when the engine could not resolve a reading.

        2026-10-11: this used to be true for an utterance with no affect
        vocabulary at all ("pass the salt", "fuck you") — a confident reading
        of nothing. The engine does not KNOW the person's state there, and
        says so.
        """
        return (
            self.resolved
            and self.affect.certainty is not AffectCertainty.NO_SIGNAL
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safety": self.safety.to_dict(),
            "affect": self.affect.to_dict(),
            "silicon": dict(self.silicon),
            "reconciliation": (
                self.reconciliation.to_dict() if self.reconciliation else None
            ),
            "input_text": self.input_text,
            "carbon": self.carbon,
            "resolved": self.resolved,
            "unresolved_reason": self.unresolved_reason,
            "provenance": dict(self.provenance),
            "knows": self.knows,
        }


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Carbon + Silicon + Aegis. One fusion cycle in, one Reading out.

    Does not speak. Does not synthesize. Does not decide. Reads — or says it
    cannot.
    """

    def __init__(
        self,
        analyzer: Optional[StructuralAffectAnalyzer] = None,
        safety_callback: Optional[Callable[[str, Dict[str, Any]], SafetyResult]] = None,
    ) -> None:
        self.carbon_layer = Carbon(analyzer)
        self.silicon = Silicon()
        self.aegis = Aegis(safety_callback=safety_callback)
        logger.info("Christman Fusion Engine initialized (structural affect)")

    def fuse(
        self,
        user_input: str,
        live_state: Optional[Dict[str, Any]] = None,
        prosody: Optional[ProsodyReading] = None,
    ) -> Reading:
        """
        Run one fusion cycle.

        Args:
            user_input: What the person said.
            live_state: Optional live system state. Passed into the safety
                gate's context as given.
            prosody: Optional ProsodyReading from the ear. The ear measures; it
                does not classify and it does not vote on valence. Its only
                role here is agreement or disagreement with the structural
                reading.

        Nothing here raises because of what was read.
        """
        live_state = live_state or {}

        # Ruling X: a fragment ("it hurts") is read against the turn before.
        affect = self.carbon_layer.encode(
            user_input, previous_turn=str(live_state.get("previous_turn") or ""))

        safety = self.aegis.check(
            user_input,
            context={
                "affect_valence": affect.valence,
                "affect_intensity": affect.intensity,
                "affect_certainty": affect.certainty.value,
                **live_state,
            },
        )

        constraints = self.silicon.retrieve(affect, live_state)
        rec = reconcile(affect.valence, affect.is_self_distress, prosody)
        resolved, unresolved_reason = self._resolve(safety, affect, rec, user_input)

        provenance: Dict[str, Any] = {
            "affect_reading": affect.certainty.value,
            "live_state_rejected": list(constraints.get("live_state_rejected", [])),
            "safety_degraded": bool(safety.degraded),
            # Words against voice, as measured (prosody.reconcile).
            "channels": rec.agreement.value,
            "channels_disagree": rec.channels_disagree,
        }
        if live_state.get("response_mode") is not None:
            provenance["response_mode_source"] = "live"

        return Reading(
            safety=safety,
            affect=affect,
            silicon=constraints,
            reconciliation=rec,
            input_text=user_input or "",
            resolved=resolved,
            unresolved_reason=unresolved_reason,
            provenance=provenance,
        )

    @staticmethod
    def _resolve(
        safety: SafetyResult,
        affect: AffectReading,
        rec: Reconciliation,
        user_input: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Whether the reading resolved. Returns (resolved, reason); the reason
        is None when it resolved and otherwise names what actually left it
        unresolved, first match wins:

          - the safety gate is degraded (the reading came from the floor);
          - there was no utterance;
          - the words and the voice disagree, or the voice is elevated
            with no resolved reading of the words (prosody.reconcile);
          - the words leave something unclear (inability, person, fragment,
            thanks), then an ambiguous affect structure, then a fear with no
            context or a feeling whose owner cannot be told.
        """
        if safety.degraded:
            return False, f"safety gate degraded: {safety.degraded_reason}"

        if not (user_input or "").strip():
            return False, "no utterance — nothing was said"

        # The words and the voice disagree, or the voice is elevated with no
        # resolved reading of the words: the reading is not resolved, and
        # the reconciliation says why. (Until 2026-11-04 this read the
        # reconciliation's `defer_to_human` instruction; it now reads the
        # measurements that flag rested on, with the same result.)
        if rec.channels_disagree or (rec.agreement is Agreement.PROSODY_ONLY
                                     and rec.voice_elevated):
            return False, rec.reason

        def _unclear(said: str) -> str:
            parts = [said]
            if safety.unclear:
                parts.append(f"unclear: {safety.unclear}")
            if safety.about:
                parts.append(f"about: {safety.about!r}")
            return "; ".join(parts)

        for reason, said in _UNRESOLVED_BEFORE_AFFECT:
            if safety.reason == reason:
                return False, _unclear(said)

        if affect.certainty is AffectCertainty.AMBIGUOUS:
            return False, (
                "; ".join(affect.notes)
                or "Affect could not be resolved from sentence structure."
            )

        for reason, said in _UNRESOLVED_AFTER_AFFECT:
            if safety.reason == reason:
                return False, _unclear(said)

        return True, None


_fusion_engine: Optional[FusionEngine] = None


def get_fusion_engine(
    analyzer: Optional[StructuralAffectAnalyzer] = None,
    safety_callback: Optional[Callable[[str, Dict[str, Any]], SafetyResult]] = None,
    *,
    force_new: bool = False,
) -> FusionEngine:
    """The shared engine. `force_new` is keyword-only, so a caller still
    passing a crisis callback in third place fails loudly instead of having
    it read as `force_new`."""
    global _fusion_engine
    if _fusion_engine is None or force_new:
        _fusion_engine = FusionEngine(
            analyzer=analyzer,
            safety_callback=safety_callback,
        )
    return _fusion_engine


__all__ = [
    "Carbon", "Silicon", "Aegis", "SafetyResult", "Reading", "FusionEngine",
    "get_fusion_engine",
]
