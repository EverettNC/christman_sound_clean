# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Test reporting rules for this repository (Everett, 2026-10-11).

A test that cannot run in the current environment must say so. It never
passes, and it never disappears into a bare "s". Such a test calls

    pytest.skip("not run: <what is missing and where it does run>")

and this file prints every one of them under a NOT RUN banner at the end of
the run, whatever flags pytest was given. pyproject.toml also sets `-ra`,
so pytest's own summary lists the reason as well. Two places, on purpose.

NOTHING PLAYS OUT LOUD UNLESS ASKED (2026-10-29)
------------------------------------------------
Test runs on Everett's Mac played speech through his speakers. Now a test
that plays through the speakers must carry `@pytest.mark.plays_aloud`, and
it runs only when CHRISTMAN_SOUND_PLAY_ALOUD=1 is set; otherwise it is NOT
RUN, with that reason. Every other test is guarded: if it reaches the real
christman_play (the C entry that drives the speakers), it fails at once,
before any sample is played, naming the marker it needs. A test that swaps
in its own christman_play (tests/test_playback_honest.py) is not stopped,
because nothing it plays reaches a speaker.

2026-10-29 REVIEW
-----------------
"not run:" is read whatever its case ("NOT RUN:" was missed). A test that
renders or measures through the compiled christman_dsp library carries
`@pytest.mark.needs_dsp`; where the library is not built it is NOT RUN with
the loader's own reason, instead of failing (test_formants already did this
for its whole module).

A TEST RUN WRITES NOTHING INTO THE PERSON'S OWN FOLDERS (2026-11-05)
--------------------------------------------------------------------
Test runs on Everett's Mac added lines to the log files in his own
~/.christman_ai/logs: the SDK's logger did not follow $CHRISTMAN_AI_HOME, and
no test run named a folder for it anyway. Both are closed. Before any test
module is imported, this file points the SDK's data folder
($CHRISTMAN_AI_HOME) and the Filament's folder ($FILAMENT_HOME) at a folder
made for this run, and takes away the three settings that would send one
kind of file somewhere else ($CHRISTMAN_LOG_DIR, $CHRISTMAN_VOICE_CACHE,
$CHRISTMAN_SOUND_AUDIO_CACHE_DIR). This is done whatever those settings were:
a test never writes where a person keeps their voice, profiles, logs or
boards. A test that reads a person's recordings or voice is handed their
path by name (FILAMENT_TEST_AUDIO, CHRISTMAN_VOICE_DIR and the like), never
by default. The folder is removed when the run ends.
tests/test_data_folder.py holds the checks.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile

import pytest

#: The folder this run writes in. Set before any test module is imported, so
#: a module that reads a folder setting at import reads this one.
RUN_FOLDER = tempfile.mkdtemp(prefix="christman-sound-tests-")
os.environ["CHRISTMAN_AI_HOME"] = os.path.join(RUN_FOLDER, "christman_ai")
os.environ["FILAMENT_HOME"] = os.path.join(RUN_FOLDER, "filament")
for _setting in ("CHRISTMAN_LOG_DIR", "CHRISTMAN_VOICE_CACHE", "CHRISTMAN_SOUND_AUDIO_CACHE_DIR"):
    os.environ.pop(_setting, None)


def pytest_unconfigure(config):  # noqa: ARG001
    try:
        shutil.rmtree(RUN_FOLDER)
    except OSError as exc:
        print(f"The test run's folder {RUN_FOLDER} could not be removed ({exc}); it was left there.", file=sys.stderr)


NOT_RUN_PREFIX = "not run:"
PLAY_ALOUD_ENV = "CHRISTMAN_SOUND_PLAY_ALOUD"
PLAY_ALOUD_MARK = "plays_aloud"


NEEDS_DSP_MARK = "needs_dsp"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{PLAY_ALOUD_MARK}: plays sound through the speakers; runs only when {PLAY_ALOUD_ENV}=1",
    )
    config.addinivalue_line(
        "markers",
        f"{NEEDS_DSP_MARK}: renders or measures through the compiled christman_dsp library; "
        "NOT RUN, with the reason, where it is not built",
    )


def _dsp_missing():
    """Why christman_dsp cannot be used here, or None when it can."""
    try:
        from christman_sound.dsp import get_dsp
    except ImportError as exc:
        return f"christman_sound.dsp could not be imported ({exc})"
    dsp = get_dsp()
    return None if dsp.available else dsp.error


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    # 2026-10-29 review: with christman_dsp not built, eleven tests FAILED
    # instead of saying they could not run. test_formants already reported
    # NOT RUN for this; every test that needs the library now does the same.
    if any(item.get_closest_marker(NEEDS_DSP_MARK) for item in items):
        missing = _dsp_missing()
        if missing is not None:
            skip_dsp = pytest.mark.skip(
                reason=(f"{NOT_RUN_PREFIX} christman_dsp is not built here ({missing}). "
                        "Build it with python3 scripts/build_dsp.py and run again."))
            for item in items:
                if item.get_closest_marker(NEEDS_DSP_MARK):
                    item.add_marker(skip_dsp)
    if os.environ.get(PLAY_ALOUD_ENV) == "1":
        return
    skip = pytest.mark.skip(
        reason=f"{NOT_RUN_PREFIX} plays through the speakers; set {PLAY_ALOUD_ENV}=1 to run it")
    for item in items:
        if item.get_closest_marker(PLAY_ALOUD_MARK):
            item.add_marker(skip)


def _drives_the_speakers(lib) -> bool:
    """True when lib.christman_play is the library's own entry, not a test's stand-in."""
    try:
        own = ctypes.CDLL(lib._name).christman_play
    except (OSError, AttributeError):
        return False
    here = ctypes.cast(lib.christman_play, ctypes.c_void_p).value
    return here == ctypes.cast(own, ctypes.c_void_p).value


@pytest.fixture(autouse=True)
def _speakers_only_when_marked(request, monkeypatch):
    if request.node.get_closest_marker(PLAY_ALOUD_MARK):
        yield
        return
    try:
        from christman_sound import dsp as dsp_mod
    except ImportError:
        yield
        return
    play = dsp_mod.ChristmanDSP.play

    def guarded(self, *args, **kwargs):
        if _drives_the_speakers(self._require()):
            pytest.fail(
                f"{request.node.nodeid} reached the speakers (christman_play) without "
                f"@pytest.mark.{PLAY_ALOUD_MARK}. Render to a file (play_audio=False), or mark "
                f"the test so it runs only when {PLAY_ALOUD_ENV}=1.", pytrace=False)
        return play(self, *args, **kwargs)

    monkeypatch.setattr(dsp_mod.ChristmanDSP, "play", guarded)
    yield


def _skip_message(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def _not_run_reason(message: str):
    """
    The reason after "not run:", whatever its case ("NOT RUN:", "Not run:"),
    or None. 2026-10-29 review: this matched lower case only, so a test that
    skipped with "NOT RUN: ..." was left out of the NOT RUN list.
    """
    at = message.lower().find(NOT_RUN_PREFIX)
    if at < 0:
        return None
    return message[at + len(NOT_RUN_PREFIX):].strip()


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    not_run = []
    for report in terminalreporter.stats.get("skipped", []):
        reason = _not_run_reason(_skip_message(report))
        if reason is not None:
            not_run.append((report.nodeid, reason))
    if not not_run:
        return
    terminalreporter.write_sep(
        "=", f"NOT RUN: {len(not_run)} test(s) made no claim in this environment"
    )
    for nodeid, reason in not_run:
        terminalreporter.write_line(f"NOT RUN  {nodeid}")
        terminalreporter.write_line(f"         {reason}")
