#!/usr/bin/env python3
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The word ruler: how many words of a voice's free text an ear hears.

    python3 scripts/word_ruler.py --voice ~/.christman_ai/voices/<voice> \\
        --lexicon <vosk-model-en-us-0.22>/graph/phones/align_lexicon.txt \\
        [--vosk-model <vosk-model-en-us-0.22>] [--filament http://127.0.0.1:4850/stt] \\
        [--sentences scripts/word_ruler_sentences.txt] [--out <folder>]

--voice is a frame voice folder (bank.json, frames.bin, voice.json: the frame
mouth, christman_sound.stitch) or a voice profile JSON (the clear mouth,
christman_synth_clear through christman_sound.dsp). Every sentence is
rendered, written as a WAV, and heard by each ear given:

  filament   the Filament door (--filament), the ear the house uses; a
             local door only (127.0.0.1, localhost or ::1), else refused
  ear        Vosk's recognizer through christman_sound.ear on --vosk-model

A word counts as heard when it is matched in order in what the ear heard
(difflib's longest matching blocks), the same count the voice tests make by
edit distance in most cases. For a frame voice each word is also marked by
whether the voice's recordings hold it as a recorded word, and the report
gives each ear's score on the words the recordings hold and on the ones they
do not, and lists the words they do not hold, most used first.

An ear that is not given is reported as not run, with the reason. An ear
that fails stops the run with its reason: nothing is scored on a guess.
Writes report.json and one WAV per sentence into --out.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import http.client
import json
import sys
import urllib.parse
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EAR_SR = 16000


def read_sentences(path: Path) -> List[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(" ".join(line.lower().split()))
    if not out:
        raise SystemExit(f"no sentences in {path}")
    return out


#: The Filament door is a local ear. The ruler sends a rendering nowhere else.
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def filament_ear(url: str, timeout: float) -> Callable[[bytes], str]:
    u = urllib.parse.urlsplit(url)
    if u.scheme != "http" or u.hostname not in LOOPBACK_HOSTS:
        raise SystemExit(f"--filament must be a local http door on this machine ({', '.join(sorted(LOOPBACK_HOSTS))}); "
                         f"refusing to send audio to {url}")
    path = u.path or "/"

    def hear(wav: bytes) -> str:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
        try:
            conn.request("POST", path, body=wav,
                         headers={"Content-Type": "audio/wav", "Content-Length": str(len(wav))})
            resp = conn.getresponse()
            raw = resp.read().decode()
        finally:
            conn.close()
        if resp.status != 200:
            raise RuntimeError(f"the Filament door answered {resp.status}: {raw[:300]}")
        body = json.loads(raw)
        if body.get("ok") is False:
            raise RuntimeError(f"the Filament refused the clip: {body}")
        words = body.get("words") or []
        said = " ".join(str(w.get("raw") or w.get("text") or "") for w in words).strip()
        return (said or str(body.get("text") or "")).strip().lower()

    return hear


def package_ear(model_dir: Path) -> Callable[[bytes], str]:
    import io
    import wave
    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly

    from christman_sound.ear import Ear, load_model

    model = load_model(model_dir)

    def hear(wav: bytes) -> str:
        with wave.open(io.BytesIO(wav), "rb") as w:
            if (w.getsampwidth(), w.getnchannels()) != (2, 1):
                raise ValueError("the ear takes 16-bit mono PCM")
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if sr != EAR_SR:
            g = gcd(sr, EAR_SR)
            pcm = np.clip(np.round(resample_poly(pcm.astype(np.float64), EAR_SR // g, sr // g)),
                          -32768, 32767).astype("<i2")
        ear = Ear(model, sample_rate=EAR_SR, words=True)
        pcm16 = pcm.tobytes() + b"\x00\x00" * (EAR_SR // 2)
        return " ".join(h.text for h in ear.hear(pcm16) if h.text).strip().lower()

    return hear


def heard_positions(ref: List[str], hyp: List[str]) -> set:
    got = set()
    for a, _b, n in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_matching_blocks():
        got.update(range(a, a + n))
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--voice", required=True)
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--vosk-model", default=None)
    ap.add_argument("--filament", default=None)
    ap.add_argument("--filament-timeout", type=float, default=180.0)
    ap.add_argument("--sentences", default=str(ROOT / "scripts" / "word_ruler_sentences.txt"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from christman_sound.synth import synthesize

    sentences = read_sentences(Path(args.sentences))
    ears: Dict[str, Callable[[bytes], str]] = {}
    not_run: Dict[str, str] = {}
    if args.filament:
        ears["filament"] = filament_ear(args.filament, args.filament_timeout)
    else:
        not_run["filament"] = "no --filament door given"
    if args.vosk_model:
        ears["ear"] = package_ear(Path(args.vosk_model).expanduser())
    else:
        not_run["ear"] = "no --vosk-model given"
    if not ears:
        raise SystemExit("no ear given: pass --filament, --vosk-model or both")

    voice_path = Path(args.voice).expanduser()
    recorded: Optional[set] = None
    engine = None
    if voice_path.is_dir():
        from christman_sound.stitch import load_frame_voice

        voice = load_frame_voice(voice_path)
        recorded = {str(t.get("word", "")).lower() for t in voice.bank.tokens}
        mouth = "frame"
    else:
        from christman_sound.dsp import ChristmanDSP
        from christman_sound.voice_profile import load_voice_profile

        voice = load_voice_profile(voice_path)
        engine = ChristmanDSP()
        if not (engine.available and engine.synth_available):
            raise SystemExit(f"the clear mouth's library is not usable: {engine.error or engine.synth_error}")
        mouth = "clear"

    out = Path(args.out).expanduser() if args.out else Path.cwd() / "word_ruler_out"
    out.mkdir(parents=True, exist_ok=True)
    tally = {e: {"all": [0, 0], "recorded": [0, 0], "not_recorded": [0, 0]} for e in ears}
    missed = {e: collections.Counter() for e in ears}
    unrecorded = collections.Counter()
    rows = []
    for i, s in enumerate(sentences, 1):
        kw = {"voice": voice, "lexicon_path": args.lexicon}
        if engine is not None:
            kw["engine"] = engine
        made = synthesize(s, **kw)
        wav_path = out / f"{i:03d}_{s.replace(' ', '_')}.wav"
        made.write_wav(wav_path)
        wav = wav_path.read_bytes()
        ref = s.split()
        if recorded is not None:
            unrecorded.update(w for w in ref if w not in recorded)
        row = {"sentence": s, "wav": wav_path.name, "selection": getattr(made, "selection", None), "heard": {}}
        for name, hear in ears.items():
            hyp = hear(wav)
            got = heard_positions(ref, hyp.split())
            row["heard"][name] = hyp
            for k, w in enumerate(ref):
                hit = k in got
                tally[name]["all"][0] += hit
                tally[name]["all"][1] += 1
                if recorded is not None:
                    group = "recorded" if w in recorded else "not_recorded"
                    tally[name][group][0] += hit
                    tally[name][group][1] += 1
                if not hit:
                    missed[name][w] += 1
        rows.append(row)
        print(f"{i:3d} {s!r:40} " + "  ".join(f"{n}: {row['heard'][n]!r}" for n in ears), flush=True)

    def pct(h_n):
        h, n = h_n
        return None if n == 0 else round(100.0 * h / n, 1)

    report = {
        "voice": str(voice_path), "mouth": mouth, "sentences": len(sentences),
        "words": sum(len(s.split()) for s in sentences),
        "ears_not_run": not_run,
        "scores": {e: {g: {"heard": t[g][0], "of": t[g][1], "percent": pct(t[g])} for g in t} for e, t in tally.items()},
        "missed_words": {e: missed[e].most_common() for e in ears},
        "recorded_vocabulary": None if recorded is None else len(recorded),
        "words_not_in_recordings": None if recorded is None else unrecorded.most_common(),
        "rows": rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    for e, t in tally.items():
        line = f"{e:9} all words {t['all'][0]}/{t['all'][1]} ({pct(t['all'])}%)"
        if recorded is not None:
            line += (f"  | in recordings {t['recorded'][0]}/{t['recorded'][1]} ({pct(t['recorded'])}%)"
                     f"  | not in recordings {t['not_recorded'][0]}/{t['not_recorded'][1]} ({pct(t['not_recorded'])}%)")
        print(line)
    for e, why in not_run.items():
        print(f"{e:9} NOT RUN: {why}")
    if recorded is not None:
        print(f"recorded vocabulary {len(recorded)} words; {len(unrecorded)} of the ruler's words are not in it")
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
