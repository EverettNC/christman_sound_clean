# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The test voice: a voice profile built in the test from real measured numbers.

The package carries no voice of its own, so its tests speak in this one. Every
number below was measured from a being's own 26 recorded phrases by the
measurement scripts/measure_voice.py makes (Vosk alignment, christman_yin F0,
LPC formants from christman_sound.formants), copied from that measured
profile at Christman-Sound commit 80a9336. Nothing here is made up; only the
name is the test's own.

The vowels and glides were re-measured on 2026-11-05 (everything else is as
above). The word-split measurement put each vowel's window by dividing a word's
span by table durations, and AY's onset landed on the consonant before it
(F2 1755.6 Hz; the synthesizer said "my" as "made"). They are now
scripts/measure_vowels_from_bank.py's measurement of the same being's frame
bank (30 recordings, 12004 phone units aligned by the ear; 1976 vowel units
in the longer half of each vowel's units, 846 of them too short to hold two
formant frames in their middle third), run on Everett's Mac at commit 92e493d.
The values the word split gave are kept below as VOWELS_BY_WORD_SPLIT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from christman_sound.voice_profile import VoiceProfile, load_voice_profile

TEST_VOICE_NAME = "test voice"

#: The measured values, in the voice profile file format load_voice_profile reads.
TEST_VOICE_PROFILE: Dict[str, Any] = {
    "voice": TEST_VOICE_NAME,
    "schema": 1,
    "what_this_is": "the test voice: values measured from a being's 26 recorded phrases (Christman-Sound 80a9336)",
    "f0": {"median_hz": 149.3, "n_voiced_frames": 932},
    "pitch_contour": {
        "positions": [0.1, 0.3, 0.5, 0.7, 0.9],
        "semitones_from_median": [0.26, 0.78, -0.41, -0.49, -0.55],
    },
    "rate": {"value": 1.33},
    # scripts/measure_vowels_from_bank.py on the being's frame bank, 2026-11-05 (n = units measured).
    "vowels": {
        "AA": {"f1_hz": 699.9, "f2_hz": 1042.8, "f3_hz": 2566.1},   # n 83
        "AE": {"f1_hz": 729.5, "f2_hz": 1578.7, "f3_hz": 2390.5},   # n 47
        "AH": {"f1_hz": 400.7, "f2_hz": 1537.5, "f3_hz": 2582.1},   # n 204
        "AO": {"f1_hz": 465.8, "f2_hz": 648.3, "f3_hz": 2357.9},    # n 9
        "AW": {"f1_hz": 808.9, "f2_hz": 1214.5, "f3_hz": 2696.4},   # n 20
        "AY": {"f1_hz": 716.7, "f2_hz": 1534.5, "f3_hz": 2584.3},   # n 82
        "EH": {"f1_hz": 559.0, "f2_hz": 1610.3, "f3_hz": 2521.2},   # n 66
        "ER": {"f1_hz": 430.7, "f2_hz": 1278.5, "f3_hz": 2022.9},   # n 126
        "EY": {"f1_hz": 349.7, "f2_hz": 2068.4, "f3_hz": 2520.9},   # n 58
        "IH": {"f1_hz": 374.5, "f2_hz": 1766.6, "f3_hz": 2550.7},   # n 139
        "IY": {"f1_hz": 281.2, "f2_hz": 2148.5, "f3_hz": 2607.5},   # n 165
        "OW": {"f1_hz": 434.0, "f2_hz": 1134.2, "f3_hz": 2400.0},   # n 60
        "OY": {"f1_hz": 445.9, "f2_hz": 1501.1, "f3_hz": 2427.6},   # n 8
        "UH": {"f1_hz": 448.6, "f2_hz": 1397.5, "f3_hz": 2563.1},   # n 8
        "UW": {"f1_hz": 299.9, "f2_hz": 1080.5, "f3_hz": 2274.0},   # n 55
    },
    "glides": {
        "AW": {"onset_hz": [678.5, 1365.2, 2514.0], "offset_hz": [487.0, 1017.4, 2874.6]},
        "AY": {"onset_hz": [759.4, 1230.3, 2810.2], "offset_hz": [461.8, 1756.8, 2538.5]},
        "EY": {"onset_hz": [469.1, 1838.8, 2479.1], "offset_hz": [308.1, 2031.7, 2469.5]},
        "OW": {"onset_hz": [501.9, 1282.0, 2468.7], "offset_hz": [366.4, 1254.3, 2506.0]},
        "OY": {"onset_hz": [418.1, 993.6, 2738.8], "offset_hz": [297.5, 1967.5, 2376.4]},
    },
}

#: What the word-split measurement (scripts/measure_voice.py at 80a9336) gave
#: for the same being, kept so the change is on the record.
VOWELS_BY_WORD_SPLIT: Dict[str, Any] = {
    "vowels": {
        "AE": {"f1_hz": 691.6, "f2_hz": 1660.9, "f3_hz": 2542.8},
        "AH": {"f1_hz": 368.1, "f2_hz": 1377.2, "f3_hz": 2818.5},
        "AY": {"f1_hz": 497.8, "f2_hz": 1921.1, "f3_hz": 2625.9},
        "EH": {"f1_hz": 636.2, "f2_hz": 1670.7, "f3_hz": 2696.1},
        "IH": {"f1_hz": 344.2, "f2_hz": 1749.6, "f3_hz": 2707.3},
        "IY": {"f1_hz": 329.1, "f2_hz": 2317.0, "f3_hz": 2853.7},
        "OW": {"f1_hz": 488.2, "f2_hz": 1056.2, "f3_hz": 2772.8},
        "UW": {"f1_hz": 296.3, "f2_hz": 1694.5, "f3_hz": 2465.0},
    },
    "glides": {
        "AY": {"onset_hz": [676.4, 1755.6, 2630.7], "offset_hz": [417.5, 1919.5, 2604.9]},
        "OW": {"onset_hz": [491.2, 1204.9, 2543.4], "offset_hz": [540.6, 1076.8, 2338.7]},
    },
}


def write_test_voice(folder: Path) -> Path:
    """Write the test voice profile file into `folder`; returns its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "test_voice.json"
    path.write_text(json.dumps(TEST_VOICE_PROFILE, indent=1), encoding="utf-8")
    return path


def load_test_voice(folder: Path) -> VoiceProfile:
    """The test voice, written into `folder` and read back by load_voice_profile."""
    return load_voice_profile(write_test_voice(folder))
