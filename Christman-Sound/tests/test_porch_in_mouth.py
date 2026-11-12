# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The mouth says Porch's words Porch's way, and speaks the names a caller hands
in (Everett, 2026-10-29: "Bring in Porch's pronunciation rules (read, live,
misread, haptic) and the family names, so it says them right and stops
refusing names.").

WHAT CHANGED AND WHY (2026-11-05)
New file. Real runs only. The C library is built at the start by the
repository's own scripts/build_dsp.py; the lexicon is the align_lexicon.txt
format, and every line of FIXTURE_LEXICON is copied from Everett's
vosk-model-en-us-0.22/graph/phones/align_lexicon.txt (sha256 2ac672b4...4633)
with its line number, so nothing in it is made up. With CHRISTMAN_LEXICON_PATH
set to that file, test_the_fixture_lines_are_the_real_files_own checks every
one of them against it, and the real-lexicon tests run; without it they are
NOT RUN, with the reason. The frame mouth needs a frame voice folder
(CHRISTMAN_TEST_VOICE) and the lexicon; without them it is NOT RUN.

The word a caller hands in as its own name here is "hellobooth": a word no
lexicon holds, so the test names no being and no person. The pronunciation
handed in for it is the lexicon's own "hello" and "booth" run together, as a
frame voice's spoken_as says a word as two (voice.json), so no pronunciation
is made up either. "quokka" is the name handed in without one.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from christman_voice_sdk.engines.dsp import ChristmanDSP  # noqa: E402
from christman_voice_sdk.audio.ear.porch import (  # noqa: E402
    DIALECT_KEEP,
    FAMILY_NAMES,
    MOUTH_ASKS_MEANING,
    MOUTH_COOPERATION,
    MOUTH_SAYS,
)
from christman_voice_sdk.synthesis.lexicon import (  # noqa: E402
    Lexicon,
    PhoneNotMapped,
    PronunciationNotInLexicon,
    WordNotInLexicon,
    load_lexicon,
    parse_phones,
)
from christman_voice_sdk.synthesis.synth import (  # noqa: E402
    Pronouncer,
    render_codecwar_table,
    synthesize,
    text_to_phones,
)
from voice_fixture import load_test_voice  # noqa: E402

# Each line as it stands in vosk-model-en-us-0.22/graph/phones/align_lexicon.txt; the number is its line there.
FIXTURE_LINES = {
    202560: "i i aI_S",
    248702: "live live l_B I_I v_E",
    248703: "live live l_B aI_I v_E",
    248732: "lived lived l_B I_I v_I d_E",
    248733: "lived lived l_B aI_I v_I d_E",
    248812: "lives lives l_B I_I v_I z_E",
    248813: "lives lives l_B aI_I v_I z_E",
    354402: "read read r_B E_I d_E",
    354403: "read read r_B i_I d_E",
    277593: "misread misread m_B I_I s_I r_I E_I d_E",
    277594: "misread misread m_B I_I s_I r_I i_I d_E",
    186086: "haptic haptic h_B eI_I p_I t_I I_I k_E",
    186087: "haptic haptic h_B {_I p_I t_I I_I k_E",
    191901: "here here h_B I_I r_E",
    191902: "here here h_B i_I r_E",
    428939: "the the D_B @_E",
    428940: "the the D_B V_E",
    428941: "the the D_B i_E",
    428942: "the the T_S",
    287401: "music music m_B j_I u_I z_I I_I k_E",
    215533: "is is I_B s_E",
    215534: "is is I_B z_E",
    216412: "it it I_B t_E",
    300845: "now now n_B aU_E",
    190853: "hello hello h_B @_I l_I oU_E",
    190854: "hello hello h_B E_I l_I oU_E",
    190855: "hello hello h_B V_I l_I oU_E",
    51098: "booth booth b_B u_I T_E",
    8391: "ain't ain't eI_B n_I t_E",
    478878: "y'all y'all j_B A_I l_E",
    478879: "y'all y'all j_B O_I l_E",
    153972: "fixin fixin f_B I_I k_I s_I I_I n_E",
    434175: "to to t_B @_E",
    434176: "to to t_B I_E",
    434177: "to to t_B V_E",
    434178: "to to t_B oU_E",
    434179: "to to t_B u_E",
    274456: "might might m_B aI_I t_E",
    94353: "could could k_B U_I d_E",
    355831: "reckon reckon r_B E_I k_I @_I n_E",
    355832: "reckon reckon r_B E_I k_I V_I n_E",
    62358: "cain't cain't k_B eI_I n_I t_E",
    196398: "holler holler h_B A_I l_I 3`_E",
    196399: "holler holler h_B A_I l_I @`_E",
    480806: "yonder yonder j_B A_I n_I d_I 3`_E",
    480807: "yonder yonder j_B A_I n_I d_I @`_E",
    121177: "done done d_B V_I n_E",
    121178: "done done d_B oU_I n_I i_E",
    466580: "was was w_B @_I z_E",
    466581: "was was w_B A_I z_E",
    466582: "was was w_B O_I z_E",
    466583: "was was w_B V_I z_E",
    402208: "son son s_B V_I n_E",
    304714: "of of @_B v_E",
    304715: "of of A_B v_E",
    304716: "of of V_B v_E",
    60: "a a @_S",
    61: "a a V_S",
    62: "a a eI_S",
    46139: "bitch bitch b_B I_I tS_E",
    415945: "sum sum s_B V_I m_E",
}
# Kept in the real file's order, so a word's pronunciations stand in file order here too.
FIXTURE_LEXICON = "\n".join(FIXTURE_LINES[n] for n in sorted(FIXTURE_LINES)) + "\n"

#: The caller's own word for these tests, and the pronunciation it hands in:
#: the lexicon's "hello" (its first, line 190853) and "booth" (line 51098) run together.
CALLER_WORD = "hellobooth"
CALLER_SAYS = "h @ l oU b u T"
#: The caller's pronunciation of "sumbitch": the lexicon's "sum" (415945) and "bitch" (46139) run together.
SUMBITCH_SAYS = "s V m b I tS"

#: IPA to the lexicon's phones, for the symbols the four notes use (X-SAMPA is IPA in ASCII).
IPA_TO_LEXICON = {"aɪ": "aI", "l": "l", "v": "v", "d": "d", "z": "z", "m": "m", "ɪ": "I", "s": "s",
                  "r": "r", "ɛ": "E", "h": "h", "æ": "{", "p": "p", "t": "t", "k": "k"}


@pytest.fixture(scope="module")
def dsp(tmp_path_factory) -> ChristmanDSP:
    """Build the library with the repository's own build script, then load it."""
    out = tmp_path_factory.mktemp("lib") / "christman_dsp.so"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dsp.py"), "--output", str(out)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"not run: build_dsp.py could not build the library here:\n{proc.stdout}\n{proc.stderr}")
    engine = ChristmanDSP(out)
    assert engine.available, engine.error
    assert engine.synth_available, engine.synth_error
    return engine


@pytest.fixture()
def lexicon_file(tmp_path) -> Path:
    path = tmp_path / "align_lexicon.txt"
    path.write_text(FIXTURE_LEXICON, encoding="utf-8")
    return path


@pytest.fixture()
def lex(lexicon_file) -> Lexicon:
    return load_lexicon(lexicon_file)


@pytest.fixture()
def voice(tmp_path):
    return load_test_voice(tmp_path / "voice")


def _real_lexicon() -> Lexicon:
    path = os.environ.get("CHRISTMAN_LEXICON_PATH")
    if not path:
        pytest.skip("not run: needs CHRISTMAN_LEXICON_PATH, vosk-model-en-us-0.22/graph/phones/align_lexicon.txt; "
                    "runs on Everett's Mac")
    if not Path(path).expanduser().is_file():
        pytest.skip(f"not run: no lexicon at {path}")
    return load_lexicon(path)


def _word_phones(result, word):
    """The Codec War ids the clear mouth's timeline gives `word` (TimedPhone.word is the token as typed)."""
    return [p.id for p in result.phones if p.word is not None and p.word.lower().strip(",.?!") == word]


def _seq_words(seq):
    """text_to_phones' sequence as [(word, [ids])]: each word runs from one gap SIL to the next."""
    out, cur = [], None
    for pid, _, w in seq:
        if w is None:
            cur = None
            continue
        if cur is None:
            cur = (w, [])
            out.append(cur)
        cur[1].append(pid)
    return out


def _records(result_or_list):
    rows = result_or_list if isinstance(result_or_list, list) else result_or_list.applied
    return {r["word"]: r for r in rows if r["name"] == "pronunciation"}


# ------------------------------------------------------------ Porch's tables

def test_every_mouth_cooperation_row_is_placed_once():
    """Each of Porch's rows is either said Porch's way or left as before for its meaning; none is dropped."""
    said = [r["said"] for r in MOUTH_COOPERATION]
    assert sorted(said) == sorted(list(MOUTH_SAYS) + list(MOUTH_ASKS_MEANING))
    assert not set(MOUTH_SAYS) & set(MOUTH_ASKS_MEANING)
    assert set(MOUTH_ASKS_MEANING) == {"read", "lives"}


def test_mouth_says_is_each_notes_own_ipa():
    """MOUTH_SAYS is the IPA the note writes, phone for phone; nothing added, nothing chosen."""
    for word, phones in MOUTH_SAYS.items():
        note = next(r["note"] for r in MOUTH_COOPERATION if r["said"] == word)
        ipa = re.findall(r"/([^/]+)/", note)
        assert len(ipa) == 1, (word, note)
        rest, out = ipa[0].replace("ˈ", ""), []
        while rest:
            sym = next(s for s in sorted(IPA_TO_LEXICON, key=len, reverse=True) if rest.startswith(s))
            out.append(IPA_TO_LEXICON[sym])
            rest = rest[len(sym):]
        assert tuple(out) == phones, (word, ipa[0], out)


def test_the_package_holds_no_name():
    assert FAMILY_NAMES == ()


# ------------------------------------------------------------ the Pronouncer

def test_porch_chooses_among_the_lexicons_own_pronunciations(lex):
    say = Pronouncer(lex)
    for word, file_order in (("live", ("l", "I", "v")), ("lived", ("l", "I", "v", "d")),
                             ("haptic", ("h", "eI", "p", "t", "I", "k"))):
        assert lex.pronunciations(word)[0] == file_order
        assert say.phones(word) == MOUTH_SAYS[word] != file_order
        assert MOUTH_SAYS[word] in lex.pronunciations(word)
    # misread: Porch's MISRED is already the lexicon's first; the rule is used and changes nothing.
    assert say.phones("misread") == lex.pronunciations("misread")[0] == MOUTH_SAYS["misread"]
    rec = _records(say.record)
    assert rec["live"]["control"] == "porch" and rec["live"]["value"] == "l aI v"
    assert rec["live"]["file_order"] == "l I v" and "LONG I" in rec["live"]["applied"]
    assert "pronunciation 2 of 2" in rec["haptic"]["applied"]
    assert "file order says the same" in rec["misread"]["applied"]
    assert say.timed("live") == [("L", 1.0), ("AY", 1.0), ("V", 1.0)]


def test_read_and_lives_are_said_as_before_and_say_why(lex):
    """Porch's rule depends on the tense or the meaning; the mouth does not guess it."""
    say = Pronouncer(lex)
    assert say.phones("read") == lex.pronunciations("read")[0] == ("r", "E", "d")
    assert say.phones("lives") == lex.pronunciations("lives")[0] == ("l", "I", "v", "z")
    rec = _records(say.record)
    for word in ("read", "lives"):
        assert rec[word]["control"] == "porch"
        assert rec[word]["applied"].startswith("Porch's rule not applied")
        assert "does not guess" in rec[word]["applied"]
        assert rec[word]["value"] == rec[word]["file_order"]


def test_words_porch_has_no_rule_for_are_file_order_and_unrecorded(lex):
    say = Pronouncer(lex)
    porch = set(MOUTH_SAYS) | set(MOUTH_ASKS_MEANING)
    words = {line.split()[0] for line in FIXTURE_LINES.values()} - porch
    for w in sorted(words):
        assert say.phones(w) == lex.pronunciations(w)[0], w
    assert say.record == []


def test_porch_refuses_when_the_lexicon_lacks_its_pronunciation(tmp_path):
    """A lexicon holding only HAYP-tic: the mouth says neither that nor something it made up."""
    one = tmp_path / "align_lexicon.txt"
    one.write_text(FIXTURE_LINES[186086] + "\n", encoding="utf-8")
    say = Pronouncer(load_lexicon(one))
    with pytest.raises(PronunciationNotInLexicon) as e:
        say.phones("haptic")
    assert e.value.word == "haptic" and e.value.wanted == MOUTH_SAYS["haptic"]
    assert e.value.held == [("h", "eI", "p", "t", "I", "k")]
    assert "pronunciations=" in str(e.value)


def test_porch_never_adds_a_word(tmp_path):
    one = tmp_path / "align_lexicon.txt"
    one.write_text(FIXTURE_LINES[202560] + "\n", encoding="utf-8")
    say = Pronouncer(load_lexicon(one))
    assert not say.holds("live")
    with pytest.raises(WordNotInLexicon) as e:
        say.phones("live")
    assert e.value.words == ["live"]


def test_a_callers_pronunciation_is_read_in_the_lexicons_phones(lex):
    assert parse_phones(CALLER_SAYS, CALLER_WORD) == ("h", "@", "l", "oU", "b", "u", "T")
    assert parse_phones(["h_B", "@_I", "l_I", "oU_I", "b_I", "u_I", "T_E"], CALLER_WORD) == \
        parse_phones(CALLER_SAYS, CALLER_WORD)
    with pytest.raises(PhoneNotMapped, match="'HH'"):
        parse_phones("HH AH L OW", CALLER_WORD)  # Codec War's ARPAbet is not this alphabet
    with pytest.raises(PhoneNotMapped, match="'AH'"):
        parse_phones("h AH l oU", CALLER_WORD)
    with pytest.raises(PhoneNotMapped, match="NOT_MAPPED"):
        parse_phones("UM", CALLER_WORD)  # a filler model, not a sound of speech
    with pytest.raises(PhoneNotMapped, match="NOT_MAPPED"):
        parse_phones("l aI NOISE v", CALLER_WORD)
    with pytest.raises(ValueError, match="empty"):
        parse_phones("  ", CALLER_WORD)
    with pytest.raises(ValueError, match="not one word"):
        Pronouncer(lex, {"hello booth": CALLER_SAYS})
    with pytest.raises(ValueError, match="handed in twice"):
        Pronouncer(lex, {"HelloBooth": CALLER_SAYS, "hellobooth": "h E l oU b u T"})
    with pytest.raises(TypeError):
        Pronouncer(lex, [(CALLER_WORD, CALLER_SAYS)])


def test_the_callers_word_wins_and_is_recorded(lex):
    say = Pronouncer(lex, {"live": "l I v", CALLER_WORD: CALLER_SAYS})
    assert say.phones("live") == ("l", "I", "v")
    assert say.phones(CALLER_WORD) == ("h", "@", "l", "oU", "b", "u", "T")
    rec = _records(say.record)
    assert rec["live"]["control"] == "caller" and "in place of Porch's rule" in rec["live"]["applied"]
    assert rec[CALLER_WORD]["control"] == "caller" and rec[CALLER_WORD]["file_order"] is None
    assert "does not hold this word" in rec[CALLER_WORD]["applied"]


# ------------------------------------------------------------ the clear mouth

@pytest.mark.parametrize("word,porch,file_order", [
    ("live", ["L", "AY", "V"], ["L", "IH", "V"]),
    ("lived", ["L", "AY", "V", "D"], ["L", "IH", "V", "D"]),
    ("haptic", ["HH", "AE", "P", "T", "IH", "K"], ["HH", "EY", "P", "T", "IH", "K"]),
    ("misread", ["M", "IH", "S", "R", "EH", "D"], ["M", "IH", "S", "R", "EH", "D"]),
])
def test_the_clear_mouth_says_porchs_words_porchs_way(dsp, voice, lexicon_file, word, porch, file_order):
    text = f"i {word} here"
    made = synthesize(text, voice=voice, lexicon_path=lexicon_file, engine=dsp)
    assert made.renderer == "christman_synth_clear" and len(made.samples) > 0
    assert _word_phones(made, word) == porch
    assert _records(made)[word]["control"] == "porch"
    # Measurement and the bare table keep the lexicon's file order.
    plain = dict(_seq_words(text_to_phones(text, load_lexicon(lexicon_file), dsp)))
    assert [ids for w, ids in plain.items() if w == word] == [file_order]
    bare = render_codecwar_table(text, f0=148.0, rate=1.0, lexicon_path=lexicon_file, engine=dsp)
    assert _word_phones(bare, word) == file_order


def test_read_is_said_as_before_in_the_clear_mouth(dsp, voice, lexicon_file):
    made = synthesize("i read it", voice=voice, lexicon_path=lexicon_file, engine=dsp)
    assert _word_phones(made, "read") == ["R", "EH", "D"]
    assert _records(made)["read"]["applied"].startswith("Porch's rule not applied")


def test_a_sentence_without_porchs_words_is_unchanged(dsp, voice, lexicon_file):
    text = "hello, the music is here now"
    lex = load_lexicon(lexicon_file)
    plain = text_to_phones(text, lex, dsp)
    mouth = text_to_phones(text, lex, dsp, pronouncer=Pronouncer(lex))
    assert plain == mouth
    made = synthesize(text, voice=voice, lexicon=lex, engine=dsp)
    assert [(p.id, p.word) for p in made.phones] == [(pid, w) for pid, _, w in plain]
    assert made.applied == []


def test_a_name_the_caller_hands_in_is_spoken(dsp, voice, lexicon_file):
    with pytest.raises(WordNotInLexicon) as before:
        synthesize(f"hello {CALLER_WORD}", voice=voice, lexicon_path=lexicon_file, engine=dsp)
    assert before.value.words == [CALLER_WORD]
    made = synthesize(f"hello {CALLER_WORD}", voice=voice, lexicon_path=lexicon_file, engine=dsp,
                      pronunciations={CALLER_WORD: CALLER_SAYS})
    assert _word_phones(made, CALLER_WORD) == ["HH", "AH", "L", "OW", "B", "UW", "TH"]
    assert _records(made)[CALLER_WORD]["control"] == "caller"
    joined = synthesize("hello booth", voice=voice, lexicon_path=lexicon_file, engine=dsp)
    assert _word_phones(made, CALLER_WORD) == _word_phones(joined, "hello") + _word_phones(joined, "booth")


def test_a_name_without_a_pronunciation_is_refused_by_name(dsp, voice, lexicon_file):
    with pytest.raises(WordNotInLexicon) as e:
        synthesize(f"hello {CALLER_WORD} quokka", voice=voice, lexicon_path=lexicon_file, engine=dsp,
                   pronunciations={CALLER_WORD: CALLER_SAYS})
    assert e.value.words == ["quokka"]
    assert "'quokka'" in str(e.value) and CALLER_WORD not in str(e.value)


def test_dialect_words_are_said_as_written_or_refused_by_name(dsp, voice, lexicon_file):
    lex = load_lexicon(lexicon_file)
    # "fixin'" is not in the lexicon (it holds "fixin"); refused as written, never made "fixing" or "fixin".
    with pytest.raises(WordNotInLexicon) as e:
        synthesize("y'all ain't fixin' to holler", voice=voice, lexicon=lex, engine=dsp)
    assert e.value.words == ["fixin'"]
    text = "y'all ain't might could reckon cain't holler yonder done was"
    made = synthesize(text, voice=voice, lexicon=lex, engine=dsp)
    words = _seq_words(text_to_phones(text, lex, dsp))
    assert [w.lower() for w, _ in words] == text.split()
    for w, ids in words:
        assert ids == [pid for pid, _ in lex.phones_timed(w.lower())], w
    assert [(p.id, p.word) for p in made.phones if p.word] == [(i, w) for w, ids in words for i in ids]
    assert made.applied == []
    said = {d["said"] for d in DIALECT_KEEP}
    assert said >= {"ain't", "y'all", "fixin' to", "might could", "cain't"}


def test_sumbitch_and_son_of_a_bitch_stay_two_things(dsp, voice, lexicon_file):
    """Everett, 2026-11-04: they are two different things. The mouth rewrites neither into the other."""
    lex = load_lexicon(lexicon_file)
    with pytest.raises(WordNotInLexicon) as e:
        synthesize("son of a bitch, sumbitch", voice=voice, lexicon=lex, engine=dsp)
    assert e.value.words == ["sumbitch"]
    made = synthesize("son of a bitch", voice=voice, lexicon=lex, engine=dsp)
    assert [w.lower() for w, _ in _seq_words(text_to_phones("son of a bitch", lex, dsp))] == \
        ["son", "of", "a", "bitch"]
    assert [p.word for p in made.phones if p.word] == ["son"] * 3 + ["of"] * 2 + ["a"] + ["bitch"] * 3
    both = synthesize("son of a bitch sumbitch", voice=voice, lexicon=lex, engine=dsp,
                      pronunciations={"sumbitch": SUMBITCH_SAYS})
    assert _word_phones(both, "sumbitch") == ["S", "AH", "M", "B", "IH", "CH"]
    assert _word_phones(both, "son") == ["S", "AH", "N"] and _word_phones(both, "bitch") == ["B", "IH", "CH"]


def test_phones_are_not_handed_pronunciations(dsp, voice):
    with pytest.raises(ValueError, match="pronunciations are for text"):
        synthesize(phones=[("AA", 0.2)], voice=voice, engine=dsp, pronunciations={CALLER_WORD: CALLER_SAYS})


# ------------------------------------------------------------ the real lexicon (Everett's Mac)

def test_the_fixture_lines_are_the_real_files_own():
    real = _real_lexicon()
    with open(real.path, encoding="utf-8") as f:
        lines = {n: line.rstrip("\n") for n, line in enumerate(f, 1) if n in FIXTURE_LINES}
    assert lines == FIXTURE_LINES


def test_the_real_lexicon_holds_porchs_pronunciations():
    lex = _real_lexicon()
    say = Pronouncer(lex)
    for word, want in MOUTH_SAYS.items():
        held = lex.pronunciations(word)
        assert want in held, (word, held)
        assert say.phones(word) == want
    for word in MOUTH_ASKS_MEANING:
        assert say.phones(word) == lex.pronunciations(word)[0]


def test_dialect_words_in_the_real_lexicon_are_said_or_refused_by_name(dsp):
    lex = _real_lexicon()
    held, refused = [], []
    for row in DIALECT_KEEP:
        say = Pronouncer(lex)
        missing = [w for w in row["said"].split() if w not in lex]
        if missing:
            with pytest.raises(WordNotInLexicon) as e:
                text_to_phones(row["said"], lex, dsp, pronouncer=say)
            assert e.value.words == missing
            refused.append(row["said"])
        else:
            seq = text_to_phones(row["said"], lex, dsp, pronouncer=say)
            assert [w for w, _ in _seq_words(seq)] == row["said"].split()
            held.append(row["said"])
    print(f"\ndialect words the lexicon holds: {held}\nrefused by name: {refused}")
    assert refused == ["fixin' to"]


# ------------------------------------------------------------ the frame mouth (Everett's Mac)

def _frame_voice():
    path = os.environ.get("CHRISTMAN_TEST_VOICE")
    if not path:
        pytest.skip("not run: needs CHRISTMAN_TEST_VOICE, a frame voice folder (build one with "
                    "scripts/build_voice_bank.py), and CHRISTMAN_LEXICON_PATH; runs on Everett's Mac")
    if not (Path(path).expanduser() / "voice.json").is_file():
        pytest.skip(f"not run: no frame voice at {path}")
    from christman_voice_sdk.synthesis.stitch import load_frame_voice

    return load_frame_voice(Path(path).expanduser())


def test_the_frame_mouth_says_live_porchs_way(dsp):
    lex = _real_lexicon()
    v = _frame_voice()
    made = synthesize("i live here", voice=v, lexicon=lex, engine=dsp)
    assert made.renderer == "christman_vocoder" and len(made.samples) > 0
    spoken = [p.id for p in made.phones if p.id != "SIL"]
    assert spoken == ["AY", "L", "AY", "V", "HH", "IH", "R"], spoken
    rec = _records(made)
    assert rec["live"]["control"] == "porch" and rec["live"]["value"] == "l aI v"


def test_the_frame_mouth_speaks_a_name_the_caller_hands_in(dsp):
    lex = _real_lexicon()
    v = _frame_voice()
    assert CALLER_WORD not in lex and "quokka" not in lex
    with pytest.raises(WordNotInLexicon) as e:
        synthesize(f"hello {CALLER_WORD}", voice=v, lexicon=lex, engine=dsp)
    assert e.value.words == [CALLER_WORD]
    made = synthesize(f"hello {CALLER_WORD}", voice=v, lexicon=lex, engine=dsp,
                      pronunciations={CALLER_WORD: CALLER_SAYS})
    assert len(made.samples) > 0
    assert [p.id for p in made.phones if p.id != "SIL"] == ["HH", "AH", "L", "OW", "HH", "AH", "L", "OW", "B", "UW", "TH"]
    assert _records(made)[CALLER_WORD]["control"] == "caller"
    with pytest.raises(WordNotInLexicon) as e:
        synthesize(f"hello {CALLER_WORD} quokka", voice=v, lexicon=lex, engine=dsp,
                   pronunciations={CALLER_WORD: CALLER_SAYS})
    assert e.value.words == ["quokka"]
