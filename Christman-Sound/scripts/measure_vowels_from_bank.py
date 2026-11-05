#!/usr/bin/env python3
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
Re-measure a voice profile's vowels from the voice's own frame bank.

    python3 scripts/measure_vowels_from_bank.py <frame voice folder> <profile.json> <out.json>

The frame voice folder is the one scripts/build_voice_bank.py built from the
same recordings (bank.json: every phone the ear aligned in them). The profile's
vowels and glides are replaced by christman_sound.voice_creation_center.
measure_vowels_from_frame_bank (see there for how, and why: the word-split
measurement put AY's onset on the consonant before it). A vowel the bank did
not measure in at least BANK_VOWEL_MIN_UNITS units is left out, so the
synthesizer uses Codec War's table value for it, and the output says so under
"vowels_not_measured_from_bank". Everything else in the profile (pitch, contour,
rate, name) is kept as it was. Nothing is written when the bank cannot be read.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from christman_sound.voice_creation_center import (  # noqa: E402
    BANK_VOWEL_MIN_UNITS,
    measure_vowels_from_frame_bank,
)
from christman_sound.voice_profile import load_voice_profile  # noqa: E402


def main(argv: list) -> int:
    if len(argv) != 4:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    folder, src, dst = Path(argv[1]).expanduser(), Path(argv[2]).expanduser(), Path(argv[3]).expanduser()
    profile = json.loads(src.read_text(encoding="utf-8"))
    m = measure_vowels_from_frame_bank(folder)
    before = {k: dict(v) for k, v in (profile.get("vowels") or {}).items()}
    profile["vowels"] = {k: dict(v) for k, v in m["vowels"].items()}
    profile["glides"] = {k: {"onset_hz": g["onset_hz"], "offset_hz": g["offset_hz"], "n_units": g["n_units"],
                             "source": g["source"]} for k, g in m["glides"].items()}
    profile["vowels_not_measured_from_bank"] = {
        k: f"{n} units measured; fewer than BANK_VOWEL_MIN_UNITS = {BANK_VOWEL_MIN_UNITS}. Codec War's table value is used"
        for k, n in m["too_few_units"].items()
    }
    profile["vowels_measured_from_bank"] = {
        "bank": m["bank"],
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "by": "scripts/measure_vowels_from_bank.py (christman_sound.voice_creation_center.measure_vowels_from_frame_bank)",
        "units_not_measured": len(m["units_not_measured"]),
        "vowels_before": before,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(profile, indent=1), encoding="utf-8")
    load_voice_profile(dst)  # the written profile must read back, or this run failed
    for ph, v in sorted(m["vowels"].items()):
        line = f"{ph:3} n={v['n_units']:4}  F1 {v['f1_hz']:7.1f}  F2 {v['f2_hz']:7.1f}  F3 {v['f3_hz']:7.1f}"
        if ph in m["glides"]:
            g = m["glides"][ph]
            line += f"   onset {g['onset_hz']}  offset {g['offset_hz']}"
        print(line)
    for ph, n in sorted(m["too_few_units"].items()):
        print(f"{ph:3} n={n:4}  too few units; the table's value is used")
    print(f"units not measured: {len(m['units_not_measured'])}")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
