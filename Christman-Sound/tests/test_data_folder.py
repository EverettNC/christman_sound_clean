# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""Everything the SDK writes for one person follows one folder setting.

$CHRISTMAN_AI_HOME names the SDK's data folder (else ~/.christman_ai). The
logs, the voice profiles, the voice cache and the audio cache did not follow
it and wrote into ~/.christman_ai whatever it said. These checks run each
writer for real, in a child process where the folder is read at import, with
$HOME pointed at an empty folder so a write to "~" would be seen.

WHAT CHANGED AND WHY (2026-11-05)
New. Test runs on Everett's Mac added lines to his own log files; the last
check here holds this test run itself to a folder made for it.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Settings that move one kind of file; a child starts without them unless a check sets one.
ONE_KIND = ("CHRISTMAN_LOG_DIR", "CHRISTMAN_VOICE_CACHE", "CHRISTMAN_SOUND_AUDIO_CACHE_DIR")


def _child(code: str, tmp_path: Path, **settings: str) -> dict:
    """Run `code` in a fresh interpreter whose home is tmp_path/home; its last line is JSON."""
    env = {k: v for k, v in os.environ.items() if k not in ONE_KIND and k != "CHRISTMAN_AI_HOME"}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(ROOT), **settings)
    (tmp_path / "home").mkdir(exist_ok=True)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


LOG_CODE = """
import json
from christman_voice_sdk.utils.logger import get_logger, default_log_dir, logger
get_logger("a.module.of.the.sdk").info("one line")
logger.info("the package's own logger")
print(json.dumps({"dir": str(default_log_dir())}))
"""


def _files(folder: Path) -> list[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


def test_logs_follow_the_sdk_folder(tmp_path):
    got = _child(LOG_CODE, tmp_path, CHRISTMAN_AI_HOME=str(tmp_path / "chosen"))
    assert got["dir"] == str(tmp_path / "chosen" / "logs")
    assert _files(tmp_path / "chosen") == ["logs/a.module.of.the.sdk.log", "logs/christman_voice_sdk.log"]
    assert "one line" in (tmp_path / "chosen" / "logs" / "a.module.of.the.sdk.log").read_text(encoding="utf-8")
    assert _files(tmp_path / "home") == [], "a log was written under the home folder"


def test_the_log_setting_alone_still_wins(tmp_path):
    got = _child(LOG_CODE, tmp_path, CHRISTMAN_AI_HOME=str(tmp_path / "chosen"),
                 CHRISTMAN_LOG_DIR=str(tmp_path / "just-logs"))
    assert got["dir"] == str(tmp_path / "just-logs")
    assert _files(tmp_path / "just-logs") == ["a.module.of.the.sdk.log", "christman_voice_sdk.log"]
    assert not (tmp_path / "chosen").exists() and _files(tmp_path / "home") == []


def test_with_no_setting_logs_are_under_the_home_folder(tmp_path):
    got = _child(LOG_CODE, tmp_path)
    assert got["dir"] == str(tmp_path / "home" / ".christman_ai" / "logs")
    assert _files(tmp_path / "home") == [".christman_ai/logs/a.module.of.the.sdk.log",
                                         ".christman_ai/logs/christman_voice_sdk.log"]


def test_importing_and_building_loggers_writes_nothing(tmp_path):
    code = """
import json
from christman_voice_sdk.utils.logger import get_logger
quiet = get_logger("a.module.that.logs.nothing")
import christman_voice_sdk.integration.voice_capture_client, christman_voice_sdk.timbre.voicepack
print(json.dumps({"file_logging": quiet.file_logging_active}))
"""
    got = _child(code, tmp_path, CHRISTMAN_AI_HOME=str(tmp_path / "chosen"))
    assert got["file_logging"] is True
    assert not (tmp_path / "chosen").exists(), _files(tmp_path / "chosen")
    assert _files(tmp_path / "home") == []


def test_a_log_folder_that_cannot_be_written_is_said_once_and_nothing_stops(tmp_path):
    blocker = tmp_path / "a-file-not-a-folder"
    blocker.write_text("x", encoding="utf-8")
    code = """
import json
from christman_voice_sdk.utils.logger import get_logger
log = get_logger("a.module.of.the.sdk")
log.info("first"); log.info("second"); log.info("third")
print(json.dumps({"reached": True}))
"""
    env_dir = str(blocker / "logs")
    env = {k: v for k, v in os.environ.items() if k not in ONE_KIND and k != "CHRISTMAN_AI_HOME"}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(ROOT), CHRISTMAN_LOG_DIR=env_dir)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and json.loads(proc.stdout.strip().splitlines()[-1]) == {"reached": True}
    assert proc.stderr.count("File logging disabled") == 1, proc.stderr
    assert "a-file-not-a-folder" in proc.stderr and blocker.read_text(encoding="utf-8") == "x"


def test_the_folder_is_read_when_a_logger_is_built_not_at_import(tmp_path, monkeypatch):
    from christman_voice_sdk.utils import logger as logmod

    monkeypatch.delenv("CHRISTMAN_LOG_DIR", raising=False)
    monkeypatch.setenv("CHRISTMAN_AI_HOME", str(tmp_path / "first"))
    assert logmod.default_log_dir() == tmp_path / "first" / "logs"
    monkeypatch.setenv("CHRISTMAN_AI_HOME", str(tmp_path / "second"))
    assert logmod.default_log_dir() == tmp_path / "second" / "logs"
    assert not hasattr(logmod, "DEFAULT_LOG_DIR")


def test_voice_profiles_follow_the_sdk_folder(tmp_path):
    code = """
import json
from christman_voice_sdk.integration import voice_capture_client as vcc
path = vcc.save_profile("a_voice", {"fundamental_hz": 181.5})
print(json.dumps({"path": str(path), "dir": str(vcc.profile_dir()), "listed": vcc.list_profiles(),
                  "loaded": vcc.load_profile("a_voice"), "has_constant": hasattr(vcc, "PROFILE_DIR")}))
"""
    got = _child(code, tmp_path, CHRISTMAN_AI_HOME=str(tmp_path / "chosen"))
    assert got["dir"] == str(tmp_path / "chosen" / "voice_profiles")
    assert got["path"] == str(tmp_path / "chosen" / "voice_profiles" / "a_voice.json")
    assert got["listed"] == ["a_voice"] and got["loaded"] == {"fundamental_hz": 181.5}
    assert got["has_constant"] is False
    assert _files(tmp_path / "home") == [], "a profile or a log was written under the home folder"


def test_voice_profiles_with_no_setting_are_under_the_home_folder(tmp_path):
    code = """
import json
from christman_voice_sdk.integration import voice_capture_client as vcc
print(json.dumps({"dir": str(vcc.profile_dir()), "listed": vcc.list_profiles()}))
"""
    got = _child(code, tmp_path)
    assert got == {"dir": str(tmp_path / "home" / ".christman_ai" / "voice_profiles"), "listed": []}


def test_the_voice_cache_follows_the_sdk_folder(tmp_path):
    code = """
import json
from christman_voice_sdk.synthesis import voice_synthesis as vs
from christman_sound import app
print(json.dumps({"sdk": str(vs.DEFAULT_CACHE_DIR), "app": str(app._audio_cache_dir())}))
"""
    got = _child(code, tmp_path, CHRISTMAN_AI_HOME=str(tmp_path / "chosen"))
    assert got == {"sdk": str(tmp_path / "chosen" / "voice_cache"), "app": str(tmp_path / "chosen" / "voice_cache")}
    got = _child(code, tmp_path, CHRISTMAN_AI_HOME=str(tmp_path / "chosen"),
                 CHRISTMAN_VOICE_CACHE=str(tmp_path / "just-cache"))
    assert got == {"sdk": str(tmp_path / "just-cache"), "app": str(tmp_path / "just-cache")}
    got = _child(code, tmp_path)
    assert got["sdk"] == got["app"] == str(tmp_path / "home" / ".christman_ai" / "voice_cache")


def test_the_app_lists_frame_voices_from_the_sdk_folder(tmp_path):
    voice = tmp_path / "chosen" / "voices" / "a_voice"
    voice.mkdir(parents=True)
    (voice / "voice.json").write_text("{}", encoding="utf-8")
    code = """
import json
from christman_sound import app
print(json.dumps({"root": str(app._sdk_data_root())}))
"""
    got = _child(code, tmp_path, CHRISTMAN_AI_HOME=str(tmp_path / "chosen"))
    assert got["root"] == str(tmp_path / "chosen")
    source = (ROOT / "christman_sound" / "app.py").read_text(encoding="utf-8")
    assert 'voices_dir = _sdk_data_root() / "voices"' in source


def test_the_audio_cache_follows_the_sdk_folder(tmp_path, monkeypatch):
    from christman_voice_sdk.audio import enhanced_speech_recognition as esr

    monkeypatch.setenv("CHRISTMAN_AI_HOME", str(tmp_path / "chosen"))
    monkeypatch.delenv(esr.AUDIO_CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv(esr.AUDIO_CACHE_ENV, "1")
    rec = esr.EnhancedSpeechRecognition.__new__(esr.EnhancedSpeechRecognition)
    rec._audio_cache_dir = None
    path = Path(rec._cache_audio(b"\x00\x01\x02\x03"))
    assert path.parent == tmp_path / "chosen" / "audio_cache" and path.read_bytes() == b"\x00\x01\x02\x03"


def _docstring_lines(text: str) -> set[int]:
    """The line numbers taken up by the module's, classes' and functions' docstrings."""
    lines: set[int] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def test_one_copy_of_the_rule():
    """No module builds its own path to ~/.christman_ai. The rule is in
    utils/data_root.py; the Voice Creation Center keeps the one other copy
    (vcc_paths.py says why), and core.py is not touched."""
    builds_it = re.compile(r"""(home\(\)|expanduser\(["']~["']\))[^#\n]*["']\.christman_ai["']""")
    allowed = {ROOT / "christman_voice_sdk" / "utils" / "data_root.py",
               ROOT / "voice_creation_center" / "vcc_paths.py",
               ROOT / "christman_sound" / "core.py"}
    hits = []
    for package in ("christman_voice_sdk", "christman_sound", "voice_creation_center"):
        for path in sorted((ROOT / package).rglob("*.py")):
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            told = _docstring_lines(text)   # a docstring may tell what the code used to do
            for n, line in enumerate(text.splitlines(), 1):
                if n not in told and builds_it.search(line.split("#", 1)[0]):
                    hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert hits == [], "\n".join(hits)
    from christman_voice_sdk.audio.config import sdk_data_root as from_config
    from christman_voice_sdk.utils.data_root import sdk_data_root

    assert from_config is sdk_data_root


def test_this_test_run_writes_in_a_folder_made_for_it():
    import conftest
    from christman_sound.ear.board import filament_root
    from christman_voice_sdk.utils.data_root import sdk_data_root
    from christman_voice_sdk.utils.logger import default_log_dir

    run = Path(conftest.RUN_FOLDER)
    assert run.is_dir() and Path.home() not in (run, *run.parents) or run.name.startswith("christman-sound-tests-")
    assert sdk_data_root() == run / "christman_ai" != Path.home() / ".christman_ai"
    assert filament_root() == run / "filament" != Path.home() / ".filament"
    assert default_log_dir() == run / "christman_ai" / "logs"
    for setting in ONE_KIND:
        assert setting not in os.environ, setting
