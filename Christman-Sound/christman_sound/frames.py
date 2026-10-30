# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_sound.frames — native frame analyzer, forced phone aligner, and bank
builder (ported from CODECWAR5TH src/lib/audio/frames.ts, align.ts, bank.ts).

Pure Python standard library + christman_dsp.so only. No numpy, no scipy.

Analyzes 22,050 Hz speech every 4.99 ms (110 samples):
  - In-place radix-2 FFT (same steps as CODECWAR5TH frames.ts & christman_vocoder.c)
  - Pitch (F0) via FFT-accelerated YIN + Viterbi path tracking (60..400 Hz)
  - 96-point CheapTrick-style pitch-synchronous spectral envelope on a mel grid
  - 8-band periodicity / aperiodicity from normalized autocorrelation at T0
  - 5-pass 3-state diagonal-Gaussian Viterbi forced phone alignment over 13
    cepstra + deltas
  - Native writer for bank.json + frames.bin consumed by christman_sound.stitch
"""

from __future__ import annotations

import array
import json
import math
import struct
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .dsp import get_dsp

__all__ = [
    "FRAME_SR",
    "HOP",
    "ENV_POINTS",
    "AP_EDGES",
    "AP_BANDS",
    "F0_MIN",
    "F0_MAX",
    "PHRASE_GAP_S",
    "EDGE_FRAMES",
    "FrameTrack",
    "AlignedPhone",
    "AlignedWord",
    "BankUnit",
    "SkippedWord",
    "BuiltBank",
    "fft",
    "hz_to_mel",
    "mel_to_hz",
    "env_grid_hz",
    "analyze_frames",
    "align_words",
    "build_bank",
    "track_ltas",
    "recording_stats",
    "pause_frames",
    "read_wav_mono_22k",
    "write_voice_bank",
]

FRAME_SR = 22050
HOP = 110
ENV_POINTS = 96
AP_EDGES: Tuple[float, ...] = (0.0, 600.0, 1200.0, 2000.0, 3000.0, 4000.0, 5500.0, 7500.0, 11025.0)
AP_BANDS = len(AP_EDGES) - 1
F0_MIN = 60.0
F0_MAX = 400.0
YIN_WIN = 512
YIN_FFT = 1024
CAND_PER_FRAME = 5
YIN_CEIL = 0.55
VOICING_COST = 0.28
JUMP_WEIGHT = 2.6
ENV_FFT = 1024
Q1 = -0.15
FLOOR_DB = -90.0
PHRASE_GAP_S = 0.15
EDGE_FRAMES = 20


def hz_to_mel(hz: float) -> float:
    return 1127.0 * math.log(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (math.exp(mel / 1127.0) - 1.0)


def env_grid_hz(sr: int = FRAME_SR) -> List[float]:
    top = hz_to_mel(sr * 0.5)
    g = [mel_to_hz((top * i) / (ENV_POINTS - 1)) for i in range(ENV_POINTS)]
    g[0] = 0.0
    g[-1] = float(sr) * 0.5
    return g


def fft(re: List[float], im: List[float], inverse: bool = False) -> None:
    """In-place radix-2 FFT of any power-of-two length (ported from CODECWAR5TH frames.ts)."""
    n = len(re)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]

    length = 2
    sign = 1.0 if inverse else -1.0
    while length <= n:
        half = length >> 1
        ang = (2.0 * math.pi / length) * sign
        wr = math.cos(ang)
        wi = math.sin(ang)
        for i in range(0, n, length):
            cr = 1.0
            ci = 0.0
            for k in range(half):
                a = i + k
                b = a + half
                rb = re[b]
                ib = im[b]
                xr = rb * cr - ib * ci
                xi = rb * ci + ib * cr
                ra = re[a]
                ia = im[a]
                re[b] = ra - xr
                im[b] = ia - xi
                re[a] = ra + xr
                im[a] = ia + xi
                nr = cr * wr - ci * wi
                ci = cr * wi + ci * wr
                cr = nr
        length <<= 1

    if inverse:
        inv_n = 1.0 / n
        for i in range(n):
            re[i] *= inv_n
            im[i] *= inv_n


@dataclass
class FrameTrack:
    sr: int
    hop: int
    n: int
    f0: List[float]
    env_db: List[List[float]]
    ap: List[List[float]]
    rms_db: List[float]
    periodicity: List[float]


def _yin_candidates_at(x: Sequence[float], center: int, sr: int) -> List[Tuple[float, float]]:
    w = YIN_WIN
    tau_min = max(2, int(sr // F0_MAX))
    tau_max = min(w - 2, int(math.ceil(sr / F0_MIN)))
    start = center - w // 2
    n_x = len(x)

    buf = [0.0] * w
    lo = max(0, start)
    hi = min(n_x, start + w)
    for idx in range(lo, hi):
        buf[idx - start] = float(x[idx])

    mean = sum(buf) / w
    e0 = 0.0
    sq = [0.0] * w
    running_sq = 0.0
    for i in range(w):
        v = buf[i] - mean
        buf[i] = v
        v2 = v * v
        e0 += v2
        running_sq += v2
        sq[i] = running_sq

    if e0 < 1e-9:
        return []

    re = [0.0] * YIN_FFT
    im = [0.0] * YIN_FFT
    re[:w] = buf
    fft(re, im, False)
    for k in range(YIN_FFT):
        re[k] = re[k] * re[k] + im[k] * im[k]
        im[k] = 0.0
    fft(re, im, True)

    cmnd = [1.0] * (tau_max + 2)
    running = 0.0
    w_last = sq[w - 1]
    for tau in range(1, tau_max + 2):
        left = sq[w - 1 - tau]
        right = w_last - sq[tau - 1]
        dt = left + right - 2.0 * re[tau]
        if dt < 0.0:
            dt = 0.0
        running += dt
        cmnd[tau] = (dt * tau) / running if running > 0.0 else 1.0

    cands: List[Tuple[float, float]] = []
    for tau in range(tau_min, tau_max + 1):
        c = cmnd[tau]
        if c < YIN_CEIL and c <= cmnd[tau - 1] and c < cmnd[tau + 1]:
            y0, y1, y2 = cmnd[tau - 1], c, cmnd[tau + 1]
            denom = 2.0 * (2.0 * y1 - y2 - y0)
            shift = (y2 - y0) / denom if abs(denom) > 1e-12 else 0.0
            best_tau = tau + max(-0.5, min(0.5, shift))
            cands.append((sr / best_tau, max(0.0, min(1.0, y1))))
    cands.sort(key=lambda item: item[1])
    return cands[:CAND_PER_FRAME]


def _track_pitch(x: Sequence[float], sr: int, hop: int, n: int, rms_db: Sequence[float]) -> Tuple[List[float], List[float]]:
    voiced_rms = [r for r in rms_db if r > -55.0]
    floor_rms = (max(rms_db) - 38.0) if voiced_rms else -55.0
    cands_list: List[List[Tuple[float, float]]] = []
    for i in range(n):
        if rms_db[i] < floor_rms:
            cands_list.append([])
        else:
            cands_list.append(_yin_candidates_at(x, i * hop, sr))

    max_s = 1 + CAND_PER_FRAME
    cost = [[math.inf] * max_s for _ in range(n)]
    back = [[0] * max_s for _ in range(n)]

    def local_cost(frame_idx: int, s: int) -> float:
        cs = cands_list[frame_idx]
        if s == 0:
            best = cs[0][1] if cs else 1.0
            return VOICING_COST + (1.0 - min(1.0, best)) * 0.35
        if s - 1 >= len(cs):
            return math.inf
        return cs[s - 1][1]

    for s in range(max_s):
        cost[0][s] = local_cost(0, s)

    for i in range(1, n):
        cs = cands_list[i]
        ps = cands_list[i - 1]
        for s in range(max_s):
            lc = local_cost(i, s)
            if not math.isfinite(lc):
                continue
            best_prev = math.inf
            best_p = 0
            for p in range(max_s):
                pc = cost[i - 1][p]
                if not math.isfinite(pc):
                    continue
                if s == 0 and p == 0:
                    tc = 0.0
                elif s == 0 or p == 0:
                    tc = 0.22
                else:
                    f_now = cs[s - 1][0]
                    f_prev = ps[p - 1][0]
                    oct_jump = abs(math.log2(f_now / f_prev))
                    tc = JUMP_WEIGHT * oct_jump * oct_jump + (0.45 if oct_jump > 0.35 else 0.0)
                if pc + tc < best_prev:
                    best_prev = pc + tc
                    best_p = p
            cost[i][s] = best_prev + lc
            back[i][s] = best_p

    last_row = cost[n - 1]
    state = min(range(max_s), key=lambda idx: last_row[idx])
    f0 = [0.0] * n
    periodicity = [0.0] * n
    for i in range(n - 1, -1, -1):
        cs = cands_list[i]
        if state > 0 and state - 1 < len(cs):
            f0[i] = cs[state - 1][0]
            periodicity[i] = 1.0 - cs[state - 1][1]
        state = back[i][state]

    f0_copy = list(f0)
    for i in range(1, n - 1):
        if f0_copy[i] > 0.0 and f0_copy[i - 1] > 0.0 and f0_copy[i + 1] > 0.0:
            vals = sorted((f0_copy[i - 1], f0_copy[i], f0_copy[i + 1]))
            f0[i] = vals[1]
        elif f0_copy[i] > 0.0 and f0_copy[i - 1] == 0.0 and f0_copy[i + 1] == 0.0:
            f0[i] = 0.0
            periodicity[i] = 0.0
    return f0, periodicity


def _envelope_at(x: Sequence[float], center: int, f0_hz: float, sr: int, grid_hz: Sequence[float]) -> List[float]:
    eff_f0 = f0_hz if f0_hz > 0.0 else 150.0
    t0 = sr / eff_f0
    half = max(64, min(ENV_FFT // 2 - 2, int(round(1.5 * t0))))
    n_fft = ENV_FFT
    n_x = len(x)

    w_len = 2 * half + 1
    w_samples = [0.0] * w_len
    w_sum = 0.0
    for idx, k in enumerate(range(-half, half + 1)):
        pos = center + k
        win = 0.5 * (1.0 + math.cos((math.pi * k) / half))
        w_sum += win
        if 0 <= pos < n_x:
            w_samples[idx] = float(x[pos]) * win

    inv_w = 1.0 / (w_sum or 1.0)
    re = [0.0] * n_fft
    im = [0.0] * n_fft
    for idx in range(w_len):
        re[idx] = w_samples[idx] * inv_w

    fft(re, im, False)
    half_bins = n_fft // 2
    p = [re[k] * re[k] + im[k] * im[k] + 1e-18 for k in range(half_bins + 1)]

    bin_hz = sr / n_fft
    w_bins = max(1.0, eff_f0 / bin_hz)
    h_bins = int(math.ceil(w_bins * 0.5))
    csum = [0.0] * (half_bins + 2)
    acc = 0.0
    for k in range(half_bins + 1):
        acc += p[k]
        csum[k + 1] = acc

    log_p = [0.0] * (half_bins + 1)
    for k in range(half_bins + 1):
        lo = max(0, k - h_bins)
        hi = min(half_bins, k + h_bins)
        avg = (csum[hi + 1] - csum[lo]) / (hi - lo + 1)
        log_p[k] = math.log(max(avg, 1e-18))

    cep_re = [0.0] * n_fft
    cep_im = [0.0] * n_fft
    for k in range(half_bins + 1):
        cep_re[k] = log_p[k]
    for k in range(1, half_bins):
        cep_re[n_fft - k] = log_p[k]
    fft(cep_re, cep_im, True)

    q_max = max(4, int(round(t0)))
    lif_re = [0.0] * n_fft
    lif_im = [0.0] * n_fft
    lif_re[0] = cep_re[0]
    for q in range(1, min(q_max, half_bins)):
        u = (math.pi * q) / t0
        sinc = math.sin(u) / u if abs(u) > 1e-9 else 1.0
        lifter = (1.0 - 2.0 * Q1) + 2.0 * Q1 * math.cos((2.0 * math.pi * q) / t0)
        w = sinc * lifter
        lif_re[q] = cep_re[q] * w
        lif_re[n_fft - q] = cep_re[n_fft - q] * w
    fft(lif_re, lif_im, False)

    to_db = 10.0 / math.log(10.0)
    out = [0.0] * len(grid_hz)
    for j, hz in enumerate(grid_hz):
        b_pos = hz / bin_hz
        b0 = max(0, min(half_bins - 1, int(math.floor(b_pos))))
        frac = b_pos - b0
        ln_val = lif_re[b0] * (1.0 - frac) + lif_re[b0 + 1] * frac
        out[j] = max(FLOOR_DB, ln_val * to_db)
    return out


def _aperiodicity_at(x: Sequence[float], center: int, f0_hz: float, sr: int) -> List[float]:
    out = [1.0] * AP_BANDS
    if f0_hz <= 0.0:
        return out
    t0 = sr / f0_hz
    lag = int(round(t0))
    win = max(128, min(512, int(round(2.5 * t0))))
    n_fft = 512
    n_x = len(x)

    a_re = [0.0] * n_fft
    a_im = [0.0] * n_fft
    b_re = [0.0] * n_fft
    b_im = [0.0] * n_fft
    base = center - win // 2
    denom_w = float(win - 1) if win > 1 else 1.0
    for idx in range(win):
        w = 0.5 - 0.5 * math.cos((2.0 * math.pi * idx) / denom_w)
        p0 = base + idx
        p1 = p0 + lag
        if 0 <= p0 < n_x:
            a_re[idx] = float(x[p0]) * w
        if 0 <= p1 < n_x:
            b_re[idx] = float(x[p1]) * w

    fft(a_re, a_im, False)
    fft(b_re, b_im, False)
    half_bins = n_fft // 2
    bin_hz = sr / n_fft

    cross = [a_re[k] * b_re[k] + a_im[k] * b_im[k] for k in range(half_bins + 1)]
    ea = [a_re[k] * a_re[k] + a_im[k] * a_im[k] for k in range(half_bins + 1)]
    eb = [b_re[k] * b_re[k] + b_im[k] * b_im[k] for k in range(half_bins + 1)]

    for band in range(AP_BANDS):
        k0 = max(1, int(math.floor(AP_EDGES[band] / bin_hz)))
        k1 = min(half_bins, int(math.ceil(AP_EDGES[band + 1] / bin_hz)))
        c_sum = sum(cross[k0 : k1 + 1])
        ea_sum = sum(ea[k0 : k1 + 1])
        eb_sum = sum(eb[k0 : k1 + 1])
        denom = math.sqrt(ea_sum * eb_sum)
        r = max(0.0, min(1.0, c_sum / denom)) if denom > 1e-18 else 0.0
        out[band] = math.sqrt(max(0.0, 1.0 - r * r))
    return out


def analyze_frames(samples: Sequence[float], sr: int = FRAME_SR, hop: int = HOP) -> FrameTrack:
    """Analyze a waveform into 4.99 ms frames (F0, 96-bin mel envelope, 8-band aperiodicity, RMS dB)."""
    x = [float(v) for v in samples]
    n = max(1, int(math.floor(len(x) / hop)))
    grid_hz = env_grid_hz(sr)
    rms_db = [FLOOR_DB] * n
    win = hop * 2
    for i in range(n):
        c = i * hop
        lo = max(0, c - win // 2)
        hi = min(len(x), c + win // 2)
        cnt = (hi - lo) or 1
        s = sum(x[k] * x[k] for k in range(lo, hi))
        rms_db[i] = max(FLOOR_DB, 10.0 * math.log10(s / cnt + 1e-18))

    f0, periodicity = _track_pitch(x, sr, hop, n, rms_db)
    env_db: List[List[float]] = []
    ap: List[List[float]] = []
    for i in range(n):
        c = i * hop
        env_db.append(_envelope_at(x, c, f0[i], sr, grid_hz))
        ap.append(_aperiodicity_at(x, c, f0[i], sr))

    return FrameTrack(
        sr=sr,
        hop=hop,
        n=n,
        f0=f0,
        env_db=env_db,
        ap=ap,
        rms_db=rms_db,
        periodicity=periodicity,
    )


N_CEP = 13
FEAT_DIM = N_CEP * 2 + 2
PASSES = 5
VAR_FLOOR = 0.05
STAY_LOGP = math.log(0.65)
STEP_LOGP = math.log(0.35)


@dataclass
class AlignedPhone:
    phone: str
    start: int
    end: int


@dataclass
class AlignedWord:
    rec: int
    word: str
    start_frame: int
    end_frame: int
    conf: float
    phones: List[AlignedPhone]


def _dct_matrix() -> List[List[float]]:
    scale = math.sqrt(2.0 / ENV_POINTS)
    return [
        [math.cos((math.pi * k * (j + 0.5)) / ENV_POINTS) * scale for j in range(ENV_POINTS)]
        for k in range(N_CEP)
    ]


_DCT = _dct_matrix()


def _track_features(t: FrameTrack) -> List[List[float]]:
    cep: List[List[float]] = []
    for i in range(t.n):
        row_env = t.env_db[i]
        cep.append([sum(row_env[j] * _DCT[k][j] for j in range(ENV_POINTS)) for k in range(N_CEP)])

    feat: List[List[float]] = []
    for i in range(t.n):
        im1 = max(0, i - 1)
        im2 = max(0, i - 2)
        ip1 = min(t.n - 1, i + 1)
        ip2 = min(t.n - 1, i + 2)
        deltas = [
            (2.0 * (cep[ip2][k] - cep[im2][k]) + (cep[ip1][k] - cep[im1][k])) / 10.0
            for k in range(N_CEP)
        ]
        voiced_flag = 1.0 if t.f0[i] > 0.0 else 0.0
        ap_low = sum(t.ap[i][:4]) * 0.25
        feat.append(cep[i] + deltas + [voiced_flag, ap_low])
    return feat


def _phone_table_specs() -> Dict[str, Dict[str, Any]]:
    dsp = get_dsp()
    if dsp.available and dsp.synth_available:
        tbl = dsp.synth_phone_table()
        fmts = dsp.synth_phone_formants()
        return {
            k: {"dur": float(v[1]), "voiced": bool(fmts.get(k, (None, False))[1])}
            for k, v in tbl.items()
        }
    return {}


def _seed_word(t: FrameTrack, start: int, end: int, phones: Sequence[str], specs: Dict[str, Dict[str, Any]]) -> List[AlignedPhone]:
    n_frames = end - start
    m = len(phones)
    weights = [max(0.03, float((specs.get(p) or {}).get("dur", 0.08))) for p in phones]
    w_sum = sum(weights) or 1.0
    bounds = [start]
    acc = 0.0
    for i in range(m - 1):
        acc += weights[i] / w_sum
        raw = start + int(round(acc * n_frames))
        lo = bounds[i] + 1
        hi = end - (m - 1 - i)
        bounds.append(max(lo, min(hi, raw)))
    bounds.append(end)

    for i in range(1, m):
        left_voiced = bool((specs.get(phones[i - 1]) or {}).get("voiced", False))
        right_voiced = bool((specs.get(phones[i]) or {}).get("voiced", False))
        if left_voiced == right_voiced:
            continue
        lo = max(bounds[i - 1] + 1, bounds[i] - 4)
        hi = min(bounds[i + 1] - 1, bounds[i] + 4)
        best = bounds[i]
        best_score = -math.inf
        for b in range(lo, hi + 1):
            score = 0.0
            for f in range(max(bounds[i - 1], b - 3), b):
                v = t.f0[f] > 0.0
                score += 1.0 if v == left_voiced else -1.0
            for f in range(b, min(bounds[i + 1], b + 3)):
                v = t.f0[f] > 0.0
                score += 1.0 if v == right_voiced else -1.0
            if score > best_score:
                best_score = score
                best = b
        bounds[i] = best

    return [AlignedPhone(phone=phones[i], start=bounds[i], end=bounds[i + 1]) for i in range(m)]


def align_words(tracks: Sequence[FrameTrack], tokens: Sequence[Dict[str, Any]], min_conf: float = 0.9) -> List[AlignedWord]:
    """5-pass 3-state diagonal-Gaussian Viterbi forced phone alignment."""
    specs = _phone_table_specs()
    feats = [_track_features(t) for t in tracks]
    words: List[AlignedWord] = []
    for tok in tokens:
        if float(tok.get("conf", 0.0)) < min_conf:
            continue
        phones = list(tok.get("phones") or [])
        if not phones:
            continue
        rec = int(tok["rec"])
        t = tracks[rec]
        sf = max(0, int(math.floor(float(tok["start"]) * t.sr / t.hop + 0.5)))
        ef = min(t.n, int(math.floor(float(tok["end"]) * t.sr / t.hop + 0.5)))
        if ef - sf < len(phones):
            continue
        words.append(
            AlignedWord(
                rec=rec,
                word=str(tok["word"]),
                start_frame=sf,
                end_frame=ef,
                conf=float(tok["conf"]),
                phones=_seed_word(t, sf, ef, phones, specs),
            )
        )
    if not words:
        return words

    g_mean = [0.0] * FEAT_DIM
    g_var = [1.0] * FEAT_DIM
    g_n = 0
    for f in feats:
        for row in f:
            for d in range(FEAT_DIM):
                g_mean[d] += row[d]
        g_n += len(f)
    if g_n > 0:
        for d in range(FEAT_DIM):
            g_mean[d] /= g_n
        acc_v = [0.0] * FEAT_DIM
        for f in feats:
            for row in f:
                for d in range(FEAT_DIM):
                    diff = row[d] - g_mean[d]
                    acc_v[d] += diff * diff
        g_var = [max(VAR_FLOOR, acc_v[d] / g_n) for d in range(FEAT_DIM)]

    two_pi = 2.0 * math.pi
    for _ in range(PASSES):
        stats: Dict[str, Tuple[List[List[float]], List[List[float]], List[float]]] = {}
        for w in words:
            f = feats[w.rec]
            for p in w.phones:
                span = max(1, p.end - p.start)
                if p.phone not in stats:
                    stats[p.phone] = (
                        [[0.0] * FEAT_DIM for _ in range(3)],
                        [[0.0] * FEAT_DIM for _ in range(3)],
                        [0.0, 0.0, 0.0],
                    )
                s_sum, s_sq, s_n = stats[p.phone]
                for t_idx in range(p.start, p.end):
                    sub = min(2, int(((t_idx - p.start) * 3) // span))
                    row = f[t_idx]
                    sum_row = s_sum[sub]
                    sq_row = s_sq[sub]
                    for d in range(FEAT_DIM):
                        rv = row[d]
                        sum_row[d] += rv
                        sq_row[d] += rv * rv
                    s_n[sub] += 1.0

        models: Dict[str, Tuple[List[List[float]], List[List[float]], List[float]]] = {}
        for ph, (s_sum, s_sq, s_n) in stats.items():
            mean = [[0.0] * FEAT_DIM for _ in range(3)]
            inv_var = [[0.0] * FEAT_DIM for _ in range(3)]
            log_z = [0.0, 0.0, 0.0]
            for s in range(3):
                cnt = s_n[s]
                if cnt >= 3.0:
                    m_vec = [s_sum[s][d] / cnt for d in range(FEAT_DIM)]
                    v_vec = [max(VAR_FLOOR, s_sq[s][d] / cnt - m_vec[d] * m_vec[d]) for d in range(FEAT_DIM)]
                else:
                    m_vec = list(g_mean)
                    v_vec = list(g_var)
                mean[s] = m_vec
                inv_var[s] = [1.0 / v_vec[d] for d in range(FEAT_DIM)]
                log_z[s] = -0.5 * sum(math.log(two_pi * v_vec[d]) for d in range(FEAT_DIM))
            models[ph] = (mean, inv_var, log_z)

        for w in words:
            f = feats[w.rec]
            m_phones = len(w.phones)
            n_frames = w.end_frame - w.start_frame
            s_total = m_phones * 3 if n_frames >= m_phones * 3 else m_phones
            per_phone = 3 if s_total == m_phones * 3 else 1
            dp = [[-math.inf] * s_total for _ in range(n_frames)]
            back = [[0] * s_total for _ in range(n_frames)]

            for s in range(s_total):
                ph_name = w.phones[s // per_phone].phone
                sub = (s % 3) if per_phone == 3 else 1
                m_arr, iv_arr, lz_arr = models[ph_name]
                m_sub = m_arr[sub]
                iv_sub = iv_arr[sub]
                lz_sub = lz_arr[sub]

                em = [0.0] * n_frames
                for t_i in range(n_frames):
                    row = f[w.start_frame + t_i]
                    quad = 0.0
                    for d in range(FEAT_DIM):
                        df = row[d] - m_sub[d]
                        quad += df * df * iv_sub[d]
                    em[t_i] = lz_sub - 0.5 * quad

                if s == 0:
                    dp[0][0] = em[0]
                    for t_i in range(1, n_frames):
                        dp[t_i][0] = dp[t_i - 1][0] + STAY_LOGP + em[t_i]
                        back[t_i][0] = 0
                else:
                    for t_i in range(s, n_frames):
                        stay = dp[t_i - 1][s] + STAY_LOGP
                        step = dp[t_i - 1][s - 1] + STEP_LOGP
                        if stay >= step:
                            dp[t_i][s] = stay + em[t_i]
                            back[t_i][s] = s
                        else:
                            dp[t_i][s] = step + em[t_i]
                            back[t_i][s] = s - 1

            state_of = [0] * n_frames
            s_cur = s_total - 1
            for t_i in range(n_frames - 1, -1, -1):
                state_of[t_i] = s_cur // per_phone
                s_cur = back[t_i][s_cur]

            cur_p = 0
            p_start = w.start_frame
            for t_i in range(1, n_frames):
                if state_of[t_i] != cur_p:
                    w.phones[cur_p].start = p_start
                    w.phones[cur_p].end = w.start_frame + t_i
                    cur_p = state_of[t_i]
                    p_start = w.start_frame + t_i
            w.phones[m_phones - 1].start = p_start
            w.phones[m_phones - 1].end = w.end_frame
    return words


@dataclass
class BankUnit:
    rec: int
    phone: str
    start: int
    end: int
    word: str
    token: int
    pos: int
    wordLen: int
    prev: str
    next: str
    phraseStart: bool
    phraseEnd: bool
    succ: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rec": self.rec,
            "phone": self.phone,
            "start": self.start,
            "end": self.end,
            "word": self.word,
            "token": self.token,
            "pos": self.pos,
            "wordLen": self.wordLen,
            "prev": self.prev,
            "next": self.next,
            "phraseStart": self.phraseStart,
            "phraseEnd": self.phraseEnd,
            "succ": self.succ,
        }


@dataclass
class SkippedWord:
    rec: int
    word: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"rec": self.rec, "word": self.word, "reason": self.reason}


@dataclass
class BuiltBank:
    tracks: List[FrameTrack]
    tokens: List[Dict[str, Any]]
    units: List[BankUnit] = field(default_factory=list)
    skipped: List[SkippedWord] = field(default_factory=list)


def build_bank(tracks: Sequence[FrameTrack], tokens: Sequence[Dict[str, Any]], min_conf: float = 0.9) -> BuiltBank:
    """Cut recordings into context-tagged phone units (ported from CODECWAR5TH src/lib/audio/bank.ts)."""
    skipped: List[SkippedWord] = []
    for tok in tokens:
        if float(tok.get("conf", 0.0)) < min_conf:
            skipped.append(SkippedWord(int(tok["rec"]), str(tok["word"]), f"conf {float(tok['conf']):.2f} < {min_conf}"))
        elif not tok.get("phones"):
            skipped.append(SkippedWord(int(tok["rec"]), str(tok["word"]), "not in lexicon"))

    aligned = align_words(tracks, tokens, min_conf=min_conf)
    by_key = {(w.rec, round(w.start_frame), round(w.end_frame)): w for w in aligned}
    units: List[BankUnit] = []

    for ti, tok in enumerate(tokens):
        rec = int(tok["rec"])
        t = tracks[rec]
        sf = max(0, int(math.floor(float(tok["start"]) * t.sr / t.hop + 0.5)))
        ef = min(t.n, int(math.floor(float(tok["end"]) * t.sr / t.hop + 0.5)))
        w = by_key.get((rec, sf, ef))
        if w is None:
            if float(tok.get("conf", 0.0)) >= min_conf and tok.get("phones"):
                skipped.append(SkippedWord(rec, str(tok["word"]), "shorter than its phones"))
            continue

        prev_tok = tokens[ti - 1] if ti > 0 else None
        next_tok = tokens[ti + 1] if ti + 1 < len(tokens) else None
        phrase_start = (
            prev_tok is None
            or int(prev_tok["rec"]) != rec
            or float(tok["start"]) - float(prev_tok["end"]) >= PHRASE_GAP_S
        )
        phrase_end = (
            next_tok is None
            or int(next_tok["rec"]) != rec
            or float(next_tok["start"]) - float(tok["end"]) >= PHRASE_GAP_S
        )
        prev_word_last = (
            prev_tok["phones"][-1]
            if (not phrase_start and prev_tok and prev_tok.get("phones"))
            else "SIL"
        )
        next_word_first = (
            next_tok["phones"][0]
            if (not phrase_end and next_tok and next_tok.get("phones"))
            else "SIL"
        )

        for pi, ph in enumerate(w.phones):
            prev_ph = w.phones[pi - 1].phone if pi > 0 else prev_word_last
            next_ph = w.phones[pi + 1].phone if pi + 1 < len(w.phones) else next_word_first
            prev_unit = units[-1] if units else None
            contiguous = prev_unit is not None and prev_unit.rec == rec and prev_unit.end == ph.start
            if contiguous and prev_unit is not None:
                prev_unit.succ = len(units)
            units.append(
                BankUnit(
                    rec=rec,
                    phone=ph.phone,
                    start=ph.start,
                    end=ph.end,
                    word=w.word,
                    token=ti,
                    pos=pi,
                    wordLen=len(w.phones),
                    prev=prev_ph,
                    next=next_ph,
                    phraseStart=phrase_start,
                    phraseEnd=phrase_end,
                    succ=-1,
                )
            )
    return BuiltBank(tracks=list(tracks), tokens=list(tokens), units=units, skipped=skipped)


def track_ltas(t: FrameTrack, start: int = 0, end: Optional[int] = None) -> List[float]:
    stop = t.n if end is None else end
    voiced_indices = [i for i in range(start, stop) if t.f0[i] > 0.0]
    if not voiced_indices:
        return [float("nan")] * ENV_POINTS
    acc = [0.0] * ENV_POINTS
    for i in voiced_indices:
        row = t.env_db[i]
        m = sum(row) / ENV_POINTS
        for j in range(ENV_POINTS):
            acc[j] += row[j] - m
    inv = 1.0 / len(voiced_indices)
    return [v * inv for v in acc]


def _quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = min(len(s) - 1, lo + 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def recording_stats(bank: BuiltBank, sessions: Sequence[str]) -> List[Dict[str, Any]]:
    by_session: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(sessions)):
        idx = [i for i, s in enumerate(sessions) if s == name]
        ltas = [0.0] * ENV_POINTS
        w = 0
        levels: List[float] = []
        for i in idx:
            t = bank.tracks[i]
            voiced_idx = [fi for fi in range(t.n) if t.f0[fi] > 0.0]
            nv = len(voiced_idx)
            if nv == 0:
                continue
            levels.extend(float(t.rms_db[fi]) for fi in voiced_idx)
            l_arr = track_ltas(t)
            for j in range(ENV_POINTS):
                ltas[j] += l_arr[j] * nv
            w += nv
        ltas_list = [float(v / w) if w else 0.0 for v in ltas]
        lvl = _quantile(levels, 0.5) if levels else -24.0
        by_session[name] = {"ltas": ltas_list, "level": lvl}
    return [by_session[s] for s in sessions]


def pause_frames(bank: BuiltBank, rec: int) -> List[int]:
    t = bank.tracks[rec]
    best: Optional[List[int]] = None
    best_level = math.inf
    toks = [k for k in bank.tokens if int(k["rec"]) == rec]
    for i in range(len(toks) - 1):
        a = int(math.floor(float(toks[i]["end"]) * t.sr / t.hop + 0.5))
        b = int(math.floor(float(toks[i + 1]["start"]) * t.sr / t.hop + 0.5))
        if (b - a) * t.hop < PHRASE_GAP_S * t.sr or (b - a) < EDGE_FRAMES:
            continue
        s = (a + b - EDGE_FRAMES) // 2
        lv = sum(t.rms_db[s : s + EDGE_FRAMES])
        if lv < best_level:
            best_level = lv
            best = [s, s + EDGE_FRAMES]
    if best is not None:
        return best
    first_start = int(math.floor(float(toks[0]["start"]) * t.sr / t.hop + 0.5)) if toks else 0
    return [max(0, first_start - EDGE_FRAMES), max(0, first_start)]


def read_wav_mono_22k(path: Union[str, Path]) -> List[float]:
    p = Path(path).expanduser()
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        if (sr, sw, ch) != (FRAME_SR, 2, 1):
            raise ValueError(
                f"{p}: must be {FRAME_SR} Hz 16-bit mono PCM (is {sr} Hz, {8 * sw}-bit, {ch} channel(s))"
            )
        raw = w.readframes(w.getnframes())
    pcm = array.array("h")
    pcm.frombytes(raw)
    if sys.byteorder != "little":
        pcm.byteswap()
    return [s / 32768.0 for s in pcm]


def _q16(v: float) -> int:
    return max(-32768, min(32767, int(math.floor(v * 100.0 + 0.5))))


def write_voice_bank(
    out_dir: Union[str, Path],
    recordings: Sequence[Dict[str, str]],
    tokens: Sequence[Dict[str, Any]],
    min_conf: float = 0.9,
) -> Dict[str, int]:
    """
    Native Python equivalent of scripts/voice_bank_driver.mjs:
    analyzes every 22,050 Hz WAV recording, runs 5-pass Viterbi forced phone
    alignment, builds context units, and writes frames.bin + bank.json.
    """
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tracks = [analyze_frames(read_wav_mono_22k(r["path"]), FRAME_SR, HOP) for r in recordings]
    bank = build_bank(tracks, tokens, min_conf=min_conf)
    sessions = [str(r["session"]) for r in recordings]
    stats = recording_stats(bank, sessions)

    fmt = struct.Struct(f"<f{ENV_POINTS}h{AP_BANDS}Hh")
    total_frames = sum(t.n for t in tracks)
    buf = bytearray(total_frames * fmt.size)
    offset = 0
    rec_meta: List[Dict[str, Any]] = []
    frame0 = 0
    for ri, t in enumerate(tracks):
        rec_meta.append({
            "path": recordings[ri]["path"],
            "session": recordings[ri]["session"],
            "n": t.n,
            "frame0": frame0,
            "stats": stats[ri],
            "pause": pause_frames(bank, ri),
        })
        for i in range(t.n):
            env_q = [_q16(float(v)) for v in t.env_db[i]]
            ap_q = [max(0, min(65535, int(math.floor(float(v) * 65535.0 + 0.5)))) for v in t.ap[i]]
            rms_q = _q16(float(t.rms_db[i]))
            fmt.pack_into(buf, offset, float(t.f0[i]), *env_q, *ap_q, rms_q)
            offset += fmt.size
        frame0 += t.n

    (out / "frames.bin").write_bytes(buf)
    meta_doc = {
        "sr": FRAME_SR,
        "hop": HOP,
        "envHz": [float(v) for v in env_grid_hz(FRAME_SR)],
        "bandEdges": list(AP_EDGES),
        "recordings": rec_meta,
        "tokens": list(tokens),
        "units": [u.to_dict() for u in bank.units],
        "skipped": [s.to_dict() for s in bank.skipped],
    }
    (out / "bank.json").write_text(json.dumps(meta_doc), encoding="utf-8")
    return {
        "frames": total_frames,
        "units": len(bank.units),
        "skipped": len(bank.skipped),
        "bytes": len(buf),
    }
