"""
================================================================================
FILE: studio/test_phase2.py
PROJECT: Christman Voice Creation Center — Phase 2 Tests
AUTHOR: The Christman AI Project | Luma Cognify AI
--------------------------------------------------------------------------------
PURPOSE:
    Smoke tests for Phase 2: recorder.py + updated live_studio.py

    Tests that would open the microphone or play sound aloud are NOT RUN
    unless the person at the machine asks for them, by setting
    VCC_ALLOW_MICROPHONE=1 (records from the room) or VCC_ALLOW_PLAYBACK=1
    (plays a tone aloud), and sounddevice is installed. A test that does not
    run is reported as NOT RUN with its reason, never as a pass.

    Tests that do NOT require a microphone (device listing, stats computation,
    Phase 1 workflow with pre-existing WAV) always run.

Run with pytest, from Christman-Sound:  python3 -m pytest tests/vcc/test_phase2.py -v
================================================================================
"""

import sys
import wave
import tempfile
import time
import struct
import math
from pathlib import Path

import os

from voice_creation_center.studio.recorder import (
    Recorder, RecorderState, RecorderUnavailable, RecordingStats, SAMPLE_RATE,
)
from voice_creation_center.studio.live_studio import LiveStudio, SessionState, ValidationStatus
from signals_for_tests import write_voiced_wav

import numpy as np

try:
    import pytest
    _skip = pytest.skip
    _SkipException = pytest.skip.Exception
except ImportError:  # the standalone runner works without pytest
    class _SkipException(Exception):
        pass

    def _skip(reason):
        raise _SkipException(reason)

SPEAKER = "studio test"
BEING = "being under test"

# ── Check sounddevice availability ────────────────────────────────────────────
try:
    import sounddevice as _sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):  # OSError: sounddevice installed without PortAudio
    SOUNDDEVICE_AVAILABLE = False

#: Opening the microphone records the room; playing sound is heard by whoever
#: is there. Neither happens unless asked for, by name, at the machine.
MICROPHONE_ALLOWED = os.environ.get("VCC_ALLOW_MICROPHONE") == "1"
PLAYBACK_ALLOWED = os.environ.get("VCC_ALLOW_PLAYBACK") == "1"


def _require_microphone() -> None:
    if not MICROPHONE_ALLOWED:
        _skip("NOT RUN: would open the microphone; set VCC_ALLOW_MICROPHONE=1 to run it")
    if not SOUNDDEVICE_AVAILABLE:
        _skip("NOT RUN: sounddevice is not installed (pip install -e '.[capture]' in Christman-Sound)")


def _require_playback() -> None:
    if not PLAYBACK_ALLOWED:
        _skip("NOT RUN: would play a tone aloud; set VCC_ALLOW_PLAYBACK=1 to run it")
    if not SOUNDDEVICE_AVAILABLE:
        _skip("NOT RUN: sounddevice is not installed (pip install -e '.[capture]' in Christman-Sound)")


# ── WAV helpers ───────────────────────────────────────────────────────────────

def _write_tone_wav(path: Path, freq: float = 440.0, seconds: float = 0.5) -> None:
    """Write a real sine wave WAV — tests clipping detection with a hot signal."""
    n = int(seconds * SAMPLE_RATE)
    samples = [
        int(32767 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        for i in range(n)
    ]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(struct.pack(f"<{n}h", *samples))


def _write_clipped_wav(path: Path, seconds: float = 0.5) -> None:
    """Write a WAV at exactly max int16 — should trigger clipping detection."""
    n = int(seconds * SAMPLE_RATE)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(struct.pack(f"<{n}h", *([32767] * n)))


# ── Recorder tests ────────────────────────────────────────────────────────────

def test_recorder_initial_state():
    """Recorder starts IDLE when it has its audio library, ERROR (said loudly) when it does not."""
    r = Recorder()
    expected = RecorderState.IDLE if SOUNDDEVICE_AVAILABLE else RecorderState.ERROR
    assert r.state == expected, f"Unexpected initial state: {r.state}"
    print("PASS  test_recorder_initial_state")


def test_recorder_list_devices_no_crash():
    """list_input_devices() returns the devices; where the audio library cannot
    be used it raises RecorderUnavailable with the reason (2026-11-05: it used
    to return an empty list, the same answer as a machine with no microphone)."""
    r = Recorder()
    if not SOUNDDEVICE_AVAILABLE:
        with pytest.raises(RecorderUnavailable, match="sounddevice cannot be used here"):
            r.list_input_devices()
        print("PASS  test_recorder_list_devices_no_crash  (no audio library here: said so, by name)")
        return
    devices = r.list_input_devices()
    assert isinstance(devices, list), "Expected list from list_input_devices()"
    print(f"PASS  test_recorder_list_devices_no_crash  ({len(devices)} device(s) found)")


class _FakeSoundDevice:
    """Stands where sounddevice does, for the device questions only."""

    def __init__(self, devices=(), default=None, fail_all=None, fail_default=None, fail_index=None):
        self._devices, self._default = list(devices), default
        self._fail_all, self._fail_default, self._fail_index = fail_all, fail_default, fail_index

    def query_devices(self, device=None, kind=None):
        if kind == "input":
            if self._fail_default:
                raise self._fail_default
            return self._default
        if device is not None:
            if self._fail_index:
                raise self._fail_index
            return self._devices[device]
        if self._fail_all:
            raise self._fail_all
        return list(self._devices)


def _recorder_on(fake, device_index=None) -> Recorder:
    r = Recorder(device_index=device_index)
    r._sd, r.unavailable_reason, r._state = fake, None, RecorderState.IDLE
    return r


MIC = {"name": "Desk Microphone", "max_input_channels": 1, "default_samplerate": 48000.0}
SPEAKERS = {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000.0}


def test_an_empty_device_list_means_the_system_has_no_input_device():
    r = _recorder_on(_FakeSoundDevice(devices=[SPEAKERS], default=None))
    assert r.list_input_devices() == []
    r = _recorder_on(_FakeSoundDevice(devices=[SPEAKERS, MIC], default=MIC))
    (mic,) = r.list_input_devices()
    assert (mic.index, mic.name, mic.is_default) == (1, "Desk Microphone", True)


def test_no_audio_library_is_said_by_name_not_returned_as_no_devices(capsys):
    r = _recorder_on(None)
    r.unavailable_reason = "sounddevice cannot be used here (OSError: PortAudio library not found)."
    r._state = RecorderState.ERROR
    with pytest.raises(RecorderUnavailable) as caught:
        r.list_input_devices()
    assert "PortAudio library not found" in str(caught.value)
    r.print_devices()
    printed = capsys.readouterr().out
    assert "PortAudio library not found" in printed and "No input devices found" not in printed
    assert r.set_device(0) is False


def test_a_failed_device_query_is_said_by_name_not_returned_as_no_devices(capsys, caplog):
    r = _recorder_on(_FakeSoundDevice(fail_all=RuntimeError("Error querying device -1")))
    with pytest.raises(RecorderUnavailable) as caught:
        r.list_input_devices()
    assert "RuntimeError: Error querying device -1" in str(caught.value)
    r.print_devices()
    printed = capsys.readouterr().out
    assert "Error querying device -1" in printed and "No input devices found" not in printed
    with caplog.at_level("ERROR", logger="christman.recorder"):
        assert r.set_device(1) is False
    assert "Error querying device -1" in caplog.text


def test_no_default_input_does_not_hide_the_devices_there_are(caplog):
    fake = _FakeSoundDevice(devices=[MIC], fail_default=RuntimeError("no default input device"))
    with caplog.at_level("WARNING", logger="christman.recorder"):
        (mic,) = _recorder_on(fake).list_input_devices()
    assert mic.name == "Desk Microphone" and mic.is_default is False
    assert "no default input device" in caplog.text


def test_a_device_name_that_cannot_be_read_says_so_and_why(caplog):
    assert _recorder_on(_FakeSoundDevice(devices=[MIC], default=MIC))._get_device_name() == "Desk Microphone"
    assert _recorder_on(_FakeSoundDevice(devices=[MIC]), device_index=0)._get_device_name() == "Desk Microphone"
    with caplog.at_level("WARNING", logger="christman.recorder"):
        name = _recorder_on(_FakeSoundDevice(fail_default=OSError("device unplugged")))._get_device_name()
    assert name != "unknown" and name.startswith("name not read:")
    assert "the default input device" in name and "OSError: device unplugged" in name
    assert "device unplugged" in caplog.text
    name = _recorder_on(_FakeSoundDevice(fail_index=ValueError("no device 7")), device_index=7)._get_device_name()
    assert "device 7" in name and "ValueError: no device 7" in name
    r = _recorder_on(None)
    r.unavailable_reason = "sounddevice cannot be used here (ImportError: No module named 'sounddevice')."
    assert r._get_device_name().startswith("name not read: sounddevice cannot be used here")


def test_the_unread_name_reaches_the_take_stats(tmp_path):
    r = _recorder_on(_FakeSoundDevice(fail_default=OSError("device unplugged")))
    signal = (np.sin(np.linspace(0, 440 * 2 * np.pi, SAMPLE_RATE)) * 8000).astype("int16").reshape(-1, 1)
    stats = r._write_wav(signal, tmp_path / "take.wav", r._get_device_name())
    assert "device unplugged" in stats.device_name and stats.to_dict()["device_name"] == stats.device_name


def test_recorder_stats_computation(tmp_path):
    """
    _write_wav() computes stats correctly from a known audio array.
    Does NOT require a microphone — uses a synthetic numpy array.
    """
    r = Recorder()

    # Create a known sine wave at half amplitude
    n      = int(0.5 * SAMPLE_RATE)
    t      = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    signal = (np.sin(2 * np.pi * 440 * t) * 16383).astype(np.int16)

    out_path = tmp_path / "test_tone.wav"
    stats = r._write_wav(signal, out_path, device_name="test_device")

    assert out_path.exists(), "WAV file was not written"
    assert stats.duration_seconds > 0.4, f"Duration too short: {stats.duration_seconds}"
    assert stats.peak_db < 0.0, f"Peak dB should be negative: {stats.peak_db}"
    assert not stats.clipped, "Half-amplitude signal should not be clipped"
    assert stats.sample_rate == SAMPLE_RATE
    print(
        f"PASS  test_recorder_stats_computation  "
        f"(peak={stats.peak_db:.1f}dB, duration={stats.duration_seconds:.2f}s)"
    )


def test_recorder_clipping_detected(tmp_path):
    """_write_wav() detects clipping on a max-amplitude signal."""
    r = Recorder()
    n      = int(0.3 * SAMPLE_RATE)
    signal = np.full(n, 32767, dtype=np.int16)

    out_path = tmp_path / "clipped.wav"
    stats = r._write_wav(signal, out_path, device_name="test")

    assert stats.clipped, "Max-amplitude signal should be flagged as clipped"
    print(f"PASS  test_recorder_clipping_detected  (peak={stats.peak_db:.1f}dB)")


def test_recorder_play_existing_wav(tmp_path):
    """play() returns True for an existing WAV file. Plays a tone aloud, so it runs only when asked."""
    _require_playback()

    wav = tmp_path / "tone.wav"
    _write_tone_wav(wav)

    r = Recorder()
    started = r.play(str(wav))
    r.stop_playback()

    assert started, "play() should return True for a valid WAV file"
    print("PASS  test_recorder_play_existing_wav")


def test_recorder_play_missing_file():
    """play() returns False for a non-existent file."""
    r = Recorder()
    result = r.play("/nonexistent/path/phantom.wav")
    assert result is False, "play() should return False for a missing file"
    print("PASS  test_recorder_play_missing_file")


def test_recorder_record_fixed_requires_sounddevice():
    """
    record_fixed() without its audio library returns None, writes nothing and
    doesn't crash. The library is taken away for the test, so it runs (and no
    microphone opens) whether or not sounddevice is installed.
    """
    r = Recorder()
    r._sd = None
    out = Path(tempfile.mkdtemp()) / "should_not_exist.wav"
    result = r.record_fixed(str(out), duration_seconds=0.1, countdown=False)
    assert result is None, "record_fixed() should return None without sounddevice"
    assert not out.exists(), "nothing may be written when nothing was recorded"
    print("PASS  test_recorder_record_fixed_requires_sounddevice")


# ── LiveStudio Phase 2 tests ──────────────────────────────────────────────────

def test_live_studio_full_phase1_workflow(tmp_path):
    """
    Full Phase 1 workflow with a pre-existing WAV of voiced sound.
    Doesn't use the microphone — verifies nothing broke in Phase 2.
    """
    raw_take = write_voiced_wav(tmp_path / "raw_take.wav")

    studio = LiveStudio(export_dir=tmp_path / "out", base_dir=tmp_path)

    session = studio.create_session("I love you", speaker=SPEAKER, being=BEING)
    assert session.state == SessionState.CREATED

    studio.register_raw_take(session.id, str(raw_take))
    assert session.state == SessionState.TAKE_REGISTERED
    assert Path(session.raw_path).exists()

    studio.mark_validation(session.id, transcript="I love you")
    assert session.validation_status == ValidationStatus.PASSED, session.validation_reasons

    studio.approve(session.id)
    assert session.approved is True

    studio.export_approved(session.id, register=False)
    assert session.state == SessionState.EXPORTED
    assert Path(session.exported_to).exists()

    # Raw take must NOT have moved
    assert raw_take.exists(), "Raw take must stay in the Center"
    print("PASS  test_live_studio_full_phase1_workflow")


def test_live_studio_session_has_recording_stats(tmp_path):
    """
    A session carries the recorder's stats as a dict. The stats here are
    computed by the recorder itself (_write_wav) from a known signal, the
    same call capture_take makes, so the numbers are measured, not typed in.
    """
    r = Recorder()
    n = int(0.5 * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    signal = (np.sin(2 * np.pi * 220 * t) * 16383).astype(np.int16)
    raw = tmp_path / "raw.wav"
    stats = r._write_wav(signal, raw, device_name="no device: computed from a known signal")

    studio  = LiveStudio(export_dir=tmp_path / "out", base_dir=tmp_path)
    session = studio.create_session("You are safe", speaker=SPEAKER, being=BEING)
    studio.register_raw_take(session.id, str(raw))
    assert session.recording_stats is None, "no microphone was used, so there are no recording stats"
    session.recording_stats = stats.to_dict()

    d = session.to_dict()
    assert abs(d["recording_stats"]["peak_db"] - (-6.02)) < 0.05
    assert d["recording_stats"]["frames_recorded"] == n
    print("PASS  test_live_studio_session_has_recording_stats")


def test_live_studio_approve_requires_passing_validation(tmp_path):
    """approve() raises ValueError if validation did not pass."""
    raw = write_voiced_wav(tmp_path / "raw.wav")

    studio  = LiveStudio(export_dir=tmp_path / "out", base_dir=tmp_path)
    session = studio.create_session("test phrase", speaker=SPEAKER, being=BEING)
    studio.register_raw_take(session.id, str(raw))

    # Deliberately fail validation
    studio.mark_validation(session.id, transcript="kill yourself")
    assert session.validation_status == ValidationStatus.FAILED

    try:
        studio.approve(session.id)
        assert False, "approve() should have raised ValueError"
    except ValueError:
        pass

    print("PASS  test_live_studio_approve_requires_passing_validation")


def test_live_studio_reject_workflow(tmp_path):
    """Rejected sessions cannot be exported."""
    raw = write_voiced_wav(tmp_path / "raw.wav")

    studio  = LiveStudio(export_dir=tmp_path / "out", base_dir=tmp_path)
    session = studio.create_session("test", speaker=SPEAKER, being=BEING)
    studio.register_raw_take(session.id, str(raw))
    studio.mark_validation(session.id, transcript="test")
    studio.reject(session.id)

    assert session.state == SessionState.REJECTED
    assert session.approved is False

    try:
        studio.export_approved(session.id, register=False)
        assert False, "export_approved() should have raised ValueError"
    except ValueError:
        pass

    print("PASS  test_live_studio_reject_workflow")


def test_live_studio_finish_capture_wrong_state(tmp_path):
    """finish_capture() raises ValueError if session is not RECORDING."""
    studio  = LiveStudio(export_dir=tmp_path / "out", base_dir=tmp_path)
    session = studio.create_session("test", speaker=SPEAKER, being=BEING)

    try:
        studio.finish_capture(session.id)
        assert False, "finish_capture() should raise ValueError when not recording"
    except ValueError:
        pass

    print("PASS  test_live_studio_finish_capture_wrong_state")


def test_live_studio_list_devices_no_crash(tmp_path):
    """list_input_devices() returns the devices, or says by name why it cannot."""
    studio  = LiveStudio(export_dir=tmp_path / "out", base_dir=tmp_path)
    if not SOUNDDEVICE_AVAILABLE:
        with pytest.raises(RecorderUnavailable, match="sounddevice cannot be used here"):
            studio.list_input_devices()
        print("PASS  test_live_studio_list_devices_no_crash  (no audio library here: said so, by name)")
        return
    devices = studio.list_input_devices()
    assert isinstance(devices, list)
    print(f"PASS  test_live_studio_list_devices_no_crash  ({len(devices)} device(s))")


# ── Microphone tests — require real hardware ──────────────────────────────────
# These only run if sounddevice is installed AND a mic is present.

def test_recorder_record_fixed_real_mic(tmp_path):
    """
    Record 1 second from the real microphone.
    NOT RUN unless VCC_ALLOW_MICROPHONE=1 and sounddevice is installed.
    """
    _require_microphone()

    r   = Recorder()
    out = tmp_path / "real_take.wav"

    print("\n  🎙️  Recording 1 second from your microphone...")
    stats = r.record_fixed(str(out), duration_seconds=1.0, countdown=False)

    assert stats is not None, "the microphone was asked for and allowed, but nothing was recorded"

    assert out.exists(),              "WAV file must be written"
    assert stats.duration_seconds > 0.8, f"Duration too short: {stats.duration_seconds}"
    assert stats.sample_rate == SAMPLE_RATE
    assert stats.frames_recorded > 0

    print(
        f"PASS  test_recorder_record_fixed_real_mic  "
        f"(duration={stats.duration_seconds:.2f}s  peak={stats.peak_db:.1f}dB)"
    )


def test_live_studio_capture_fixed_real_mic(tmp_path):
    """
    capture_take() with fixed duration using the real microphone.
    NOT RUN unless VCC_ALLOW_MICROPHONE=1 and sounddevice is installed.
    """
    _require_microphone()

    studio  = LiveStudio(export_dir=tmp_path / "out", base_dir=tmp_path)
    session = studio.create_session("I love you", speaker=SPEAKER, being=BEING)

    print("\n  🎙️  Recording 'I love you' (1 second)...")
    studio.capture_take(session.id, duration_seconds=1.0, countdown=False)

    assert session.state == SessionState.TAKE_REGISTERED
    assert session.raw_path is not None
    assert Path(session.raw_path).exists()
    assert session.recording_stats is not None
    assert session.recording_stats["duration_seconds"] > 0.8

    print(
        f"PASS  test_live_studio_capture_fixed_real_mic  "
        f"(peak={session.recording_stats['peak_db']:.1f}dB)"
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_all():
    """Runs every test without pytest. A skipped test is NOT RUN, never a pass."""
    tests = [
        (test_recorder_initial_state,                       None),
        (test_recorder_list_devices_no_crash,               None),
        (test_recorder_stats_computation,                   True),
        (test_recorder_clipping_detected,                   True),
        (test_recorder_play_existing_wav,                   True),
        (test_recorder_play_missing_file,                   None),
        (test_recorder_record_fixed_requires_sounddevice,   None),
        (test_live_studio_full_phase1_workflow,             True),
        (test_live_studio_session_has_recording_stats,      True),
        (test_live_studio_approve_requires_passing_validation, True),
        (test_live_studio_reject_workflow,                  True),
        (test_live_studio_finish_capture_wrong_state,       True),
        (test_live_studio_list_devices_no_crash,            True),
        (test_recorder_record_fixed_real_mic,               True),
        (test_live_studio_capture_fixed_real_mic,           True),
    ]

    passed, failed, not_run = 0, 0, 0
    print(f"\n  Running {len(tests)} tests...\n")

    for fn, needs_tmp in tests:
        try:
            if needs_tmp:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            passed += 1
        except _SkipException as e:
            print(f"NOT RUN  {fn.__name__}: {e}")
            not_run += 1
        except Exception as e:
            print(f"FAIL     {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n  {passed} passed, {failed} failed, {not_run} not run (of {len(tests)}).\n")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)


# ==============================================================================
# © 2026 Everett Nathaniel Christman & The Christman AI Project
# Luma Cognify AI — "How can we help you love yourself more?"
# Patent Pending TCAP-2026-001 | TCAP-2026-002
# ==============================================================================
