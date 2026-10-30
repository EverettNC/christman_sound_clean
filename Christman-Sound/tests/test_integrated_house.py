# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
End-to-end tests for the integrated 7-repository Christman-Sound house
architecture:
  1. christman_sound.frames (4.99 ms frame analysis, radix-2 FFT, 5-pass Viterbi
     forced alignment, native write_voice_bank -> FrameBank round-trip)
  2. christman_sound.dispersion (4-canal Dispersion & War, Refused-Mix guard)
  3. christman_sound.being (Silico-Stasis equilibrium, concentric cube/room/universe
     shape, 2 Hz shaker, psychoacoustic unit lattice, streaming codec inspector)
  4. christman_sound.ear.{mouth,board,porch} (Filament mouth map, CONF_FLOOR=0.55
     Carbon pile, Family/Residual boards, 15 dialect regions, TCAP-V-0020
     mouth-cooperation, names each being hands in, Vosk 0.22 discovery)
  5. christman_voice_sdk.audio.corti_canal (Greenwood cochlea, cycle DSP voice
     quality, SpeakerSeat >= 3 locks, grunt physics & stimming router,
     20 Hz CanalRecorder & disk writer)
  6. christman_voice_sdk.audio.lucent_reader (Gullstrand cornea optics, 4x4 head
     pose, 9-cell gaze, iris/mouth landmarks, BlinkLog, 5-part RhythmTracker,
     BodySignalFrame bridge)
  7. christman_voice_sdk.audio.whole_house (OpenSmell 21-profile classifier with
     1-scalar refusal guard, Skin haptic posture, House Fifth Stream, and
     TCAP-V-0044 Five-Element Memory Chamber)
  8. Studio API endpoints (/api/frames-analyze, /api/dispersion, /api/porch,
     /api/corti-canal, /api/lucent, /api/being, /api/whole-house,
     /api/voice-creation-center)
  9. Sovereignty check: all new modules run with numpy, scipy, soundfile,
     sounddevice, pygame, and librosa blocked via builtins.__import__.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_frames_analyze_and_native_voice_bank_roundtrip(tmp_path):
    from christman_sound.frames import FRAME_SR, HOP, analyze_frames, write_voice_bank
    from christman_sound.stitch import FrameBank, join_units, select_units
    from christman_sound.synth import render_codecwar_table, write_wav

    res = render_codecwar_table(
        None,
        phones=[("HH", 1.0), ("AH", 1.0), ("L", 1.0), ("OW", 1.0)],
        f0=140.0,
        rate=1.0,
        sample_rate=FRAME_SR,
    )
    wav_path = tmp_path / "hello.wav"
    write_wav(wav_path, res.samples, FRAME_SR)

    track = analyze_frames(res.samples, sr=FRAME_SR, hop=HOP)
    assert track.n > 15
    assert len(track.env_db) == track.n and len(track.env_db[0]) == 96
    assert len(track.ap) == track.n and len(track.ap[0]) == 8
    assert any(f > 0.0 for f in track.f0)

    out_bank = tmp_path / "bank_out"
    summary = write_voice_bank(
        out_bank,
        recordings=[{"path": str(wav_path), "session": "s1"}],
        tokens=[{
            "rec": 0,
            "word": "hello",
            "start": 0.02,
            "end": round(len(res.samples) / FRAME_SR - 0.02, 3),
            "conf": 0.99,
            "phones": ["HH", "AH", "L", "OW"],
        }],
        min_conf=0.9,
    )
    assert summary["units"] == 4
    fb = FrameBank(out_bank)
    assert len(fb.units) == 4
    sel = select_units(fb, [("hello", ["HH", "AH", "L", "OW"], False)])
    assert sel.whole_words == 1
    stitched = join_units(fb, sel)
    assert len(stitched.track.f0) > 0


def test_dispersion_four_canals_and_refused_mix():
    from christman_sound.dispersion import (
        MAX_CANALS,
        RefusedMixError,
        discern_canals,
        mix_analysis_label,
        refuse_mixed_canal,
    )

    assert MAX_CANALS == 4
    with pytest.raises(RefusedMixError):
        refuse_mixed_canal(3)
    assert "single source" in mix_analysis_label(1)

    for scene_id in ("kitchen", "two-booths", "crisis"):
        rep = discern_canals(scene_id)
        assert rep["scene"]["id"] == scene_id
        assert len(rep["canals"]) >= 2
        assert rep["comparison"]["pitch_corrupted_by_mix"] is True


def test_being_silico_stasis_shape_units_and_stream():
    from christman_sound.being import (
        CUBE_DWELL,
        CUBE_VERTS,
        LATTICE,
        LAYER_R,
        NEST,
        OPEN_FRAGMENTS,
        Qty,
        TICK_SEC,
        attach_perturbations,
        bark_to_hz,
        convert_unit,
        create_stasis_field,
        cube_centroid,
        cube_edge_length,
        empty_stasis,
        hz_to_bark,
        inspect_stream,
        intensity_of,
        layer_of,
        nearest_phone,
        place_perturbation,
        shaker_pulse,
        shannon,
        tick_stasis,
    )

    assert len(OPEN_FRAGMENTS) == 5
    assert OPEN_FRAGMENTS[0]["title"] == "54, 64, 84"
    assert NEST == {"cube": 1, "room": 2, "universe": 3}
    assert LAYER_R["center"] < LAYER_R["cube"] < LAYER_R["room"] < LAYER_R["universe"]
    assert len(CUBE_VERTS) == 8
    assert cube_edge_length() == 2.0
    c = cube_centroid()
    assert abs(c["x"]) < 1e-12 and abs(c["y"]) < 1e-12 and abs(c["z"]) < 1e-12
    assert any(q["name"] == "fs" and q["unit"] == "Hz" for q in LATTICE)

    # Settled silico-stasis rules (codec/src/lib/being/units.test.ts)
    q0 = Qty("energy", "energy", 0.0, "RMS", 0.0, 1.0, "filament")
    q_high = Qty("energy", "energy", 0.9, "RMS", 0.0, 1.0, "filament")
    assert intensity_of(q0) == 0.0
    assert layer_of(0.0, 10.0) == "center"
    assert layer_of(0.3, 0.0) == "room"
    assert layer_of(0.1, 0.0) == "universe"
    assert layer_of(0.5, CUBE_DWELL) == "cube"

    now = CUBE_DWELL + TICK_SEC
    occ = {"energy": {"arrived": 0.0, "last": now, "intensity": 0.9}}
    ps = attach_perturbations([q_high], occ, now)
    assert ps[0].layer == "cube"
    next_state = tick_stasis(empty_stasis(), ps, now, TICK_SEC)
    assert next_state.timeSense == TICK_SEC

    # Studio StasisField wrapper
    field = create_stasis_field(OPEN_FRAGMENTS)
    p = place_perturbation("test-shock", magnitude=0.6, layer="cube")
    attach_perturbations(field, [p, shaker_pulse(0.25)])
    tick_stasis(field, steps=5)
    assert field.ticks == 5
    assert 0.0 < field.quantities.coherence <= 1.0

    # Psychoacoustic conversions & vowel F1-F2 lookup
    assert abs(bark_to_hz(hz_to_bark(1000.0)) - 1000.0) < 1e-6
    assert nearest_phone(270.0, 2290.0)["phone"]["id"] == "IY"
    assert nearest_phone(730.0, 1090.0)["phone"]["id"] == "AA"
    assert shannon([1.0, 0.0, 0.0]) == 0.0
    assert abs(shannon([0.25, 0.25, 0.25, 0.25]) - math.log(4.0)) < 1e-12

    u = convert_unit(440.0, "hz")
    assert u["hz"] == 440.0
    assert u["cents"] == 0.0
    assert abs(u["period_ms"] - 2.2727) < 0.01
    round_hz = convert_unit(u["mel"], "mel")["hz"]
    assert abs(round_hz - 440.0) < 0.05

    st = inspect_stream("Christman — Luma", encoding="utf-8")
    assert st["lossless"] is True
    assert st["byte_count"] > st["char_count"]


def test_filament_mouth_board_porch_and_vosk_discovery(tmp_path, monkeypatch):
    from christman_sound.ear.board import DIALECT_MAP, add_family, add_residual, snapshot
    from christman_sound.ear.mouth import CONF_FLOOR, dress
    from christman_sound.ear.porch import reconstruct_dialect
    from christman_sound.ear.recognizer import find_graph

    monkeypatch.setenv("FILAMENT_HOME", str(tmp_path / "filament"))
    assert CONF_FLOOR == 0.55
    assert len(DIALECT_MAP) == 15

    dressed = dress({
        "ok": True,
        "final": True,
        "text": "Merry Christmas from Averett you son of a bitch",
        "words": [
            {"word": "christmas", "start": 0.1, "end": 0.4, "conf": 0.92},
            {"word": "mumble", "start": 0.5, "end": 0.8, "conf": 0.31},
        ],
    })
    # 2026-10-30: the mouth map no longer turns a client's "Christmas" into a
    # family name, or "Averett" into a person's.
    assert "Christmas" in dressed["text"] and "Christman" not in dressed["text"]
    assert "Averett" in dressed["text"] and "Everett" not in dressed["text"]
    assert "sumbitch" in dressed["text"]
    assert dressed["carbon"] is True
    assert len(dressed["unsure"]) == 1

    said = "Alpha Vox said Miss Reed felt hectic feedback and leaved early with Broxton"
    porch = reconstruct_dialect(said)
    assert "misread" in porch.as_said
    assert "haptic feedback" in porch.as_said
    assert "lived" in porch.as_said
    # The package knows no names (2026-10-30): nothing is recovered or routed
    # to a name nobody handed in.
    assert "Alpha Vox" in porch.as_said and "Broxton" in porch.as_said
    assert porch.route == []
    # A being hands in its own names and spellings.
    own = reconstruct_dialect(said, extra_lexicon=[{"said": "alpha vox", "keep": "Nova"},
                                                   {"said": "broxton", "keep": "Quill"}],
                              names=("Nova", "Quill"))
    assert "Nova" in own.as_said and "Quill" in own.as_said
    assert own.route == ["Nova", "Quill"]

    add_family("y'all", "texas")
    add_residual("unheard", "kentucky", "slipped")
    snap = snapshot()
    assert len(snap["family"]) >= 1
    assert len(snap["residual"]) >= 1

    # Verify -lgraph finds sibling vosk-model-en-us-0.22/graph/HCLG.fst
    full_dir = tmp_path / "vosk-model-en-us-0.22" / "graph"
    full_dir.mkdir(parents=True)
    (full_dir / "HCLG.fst").write_bytes(b"fst")
    lgraph_dir = tmp_path / "vosk-model-en-us-0.22-lgraph"
    lgraph_dir.mkdir(parents=True)
    assert find_graph(lgraph_dir) == full_dir


def test_corti_canal_greenwood_dsp_seat_and_alphavox(tmp_path):
    from christman_voice_sdk.audio.corti_canal import (
        CanalRecorder,
        SpeakerSeat,
        VocalTracker,
        analyze_grunt_physics,
        detect_vocal_stimming,
        freq_to_place,
        place_to_freq,
        read_canal_range,
        voice_quality_dsp,
        write_canal_frames,
    )

    p = freq_to_place(1000.0)
    assert 0.3 < p < 0.6
    assert abs(place_to_freq(p) - 1000.0) < 1e-3

    # Silence produces None for jitter/shimmer/hnr_db
    sil_q = voice_quality_dsp([0.0] * 4096, 16000)
    assert sil_q == {"jitter": None, "shimmer": None, "hnr_db": None}

    # Clean periodic wave locks jitter/shimmer/HNR
    sr = 16000
    wave_sig = [0.5 * math.sin(2.0 * math.pi * 160.0 * (i / sr)) for i in range(sr // 2)]
    q = voice_quality_dsp(wave_sig, sr, f0=160.0)
    assert q["jitter"] is not None and q["jitter"] < 0.02
    assert q["shimmer"] is not None and q["shimmer"] < 0.02
    assert q["hnr_db"] is not None and q["hnr_db"] > 15.0

    # SpeakerSeat requires >= 3 voiced locks
    seat = SpeakerSeat()
    seat.observe(130.0, 0.05)
    seat.observe(134.0, 0.06)
    assert seat.seated is False and seat.baseline() is None
    seat.observe(132.0, 0.05)
    assert seat.seated is True
    assert seat.baseline()["resting_f0_hz"] == 132.0

    # Grunt physics router: stimming vs acute distress
    stim_events = [
        {"durationMs": 300.0, "attackMs": 90.0, "peakRms": 0.06, "f0": 180.0},
        {"durationMs": 305.0, "attackMs": 92.0, "peakRms": 0.06, "f0": 181.0},
        {"durationMs": 298.0, "attackMs": 88.0, "peakRms": 0.06, "f0": 179.0},
    ]
    assert detect_vocal_stimming(stim_events) is True
    route_stim = analyze_grunt_physics(stim_events[-1], seat=seat, recent_events=stim_events)
    assert route_stim.key == "vocal_stimming"
    assert route_stim.regulatory is True and route_stim.urgent is False

    route_distress = analyze_grunt_physics({"durationMs": 190.0, "attackMs": 35.0, "peakRms": 0.18, "f0": 290.0}, seat=seat)
    assert route_distress.key == "grunt_distress" and route_distress.urgent is True

    # CanalRecorder & disk round-trip
    rec = CanalRecorder()
    for i in range(10):
        rec.record(t_ms=1000.0 + i * 50.0, rms=0.04, zcr=0.05, f0=140.0)
    recovered = rec.recover_truncated_tape(1000.0, 450.0)
    assert len(recovered) == 10 and recovered[0]["t"] == 0.0

    w_res = write_canal_frames(list(rec.frames), canal_dir=tmp_path / "canal", allow_local_dir=True)
    assert w_res["written"] == 10
    read_back = read_canal_range(1000.0, 1500.0, canal_dir=tmp_path / "canal")
    assert len(read_back) == 10

    # VocalTracker onset/offset
    vt = VocalTracker()
    assert vt.step(0.0, 0.025, 0.04, f0=145.0) is None
    assert vt.step(50.0, 0.040, 0.05, f0=148.0) is None
    closed = vt.step(120.0, 0.002, 0.03, f0=None)
    assert closed is not None and closed["f0"] == 146.5


def test_lucent_cornea_optics_rhythm_and_body_signal_bridge():
    from christman_voice_sdk.audio.body_signals import analyze_body_window
    from christman_voice_sdk.audio.lucent_reader import (
        CORNEA_PRESETS,
        BlinkLog,
        RhythmTracker,
        evaluate_optics,
        gaze_from_blendshapes,
        pose_from_matrix,
        reading_to_body_signal_frame,
        trace_meridian_rays,
    )

    emm = evaluate_optics(CORNEA_PRESETS["emmetropic"])
    kc = evaluate_optics(CORNEA_PRESETS["keratoconus"])
    assert 42.0 <= emm.d_cornea_total <= 44.0
    assert kc.d_cornea_total > emm.d_cornea_total
    rays = trace_meridian_rays(CORNEA_PRESETS["emmetropic"], n_rays=7)
    assert len(rays) == 7

    ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    pose = pose_from_matrix(ident)
    assert pose is not None and pose["pitch_deg"] == 0.0

    eyes = {k: 0.0 for k in ("eyeLookInLeft", "eyeLookOutLeft", "eyeLookInRight", "eyeLookOutRight",
                             "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
                             "eyeBlinkLeft", "eyeBlinkRight")}
    gaze = gaze_from_blendshapes({**eyes, "eyeLookOutRight": 0.6, "eyeLookInLeft": 0.6})
    assert gaze["cell"] == "middle-right"
    # 2026-10-30: a missing blendshape is not zero; nothing is read and it is named.
    partial = gaze_from_blendshapes({"eyeLookOutRight": 0.6, "eyeLookInLeft": 0.6})
    assert partial["cell"] is None and "eyeBlinkLeft" in partial["reason"]

    bl = BlinkLog()
    bl.observe(1000.0, True)
    bl.observe(1150.0, False)
    bl.observe(3000.0, True)
    assert bl.per_minute(4000.0) == 2.0

    rt = RhythmTracker()
    for i in range(20):
        t_ms = i * 100.0
        rt.push("head", t_ms, math.sin(2.0 * math.pi * 2.0 * (t_ms / 1000.0)))
    rep = rt.summary()["head"]
    assert rep is not None and 1.5 <= rep["hz"] <= 2.5

    frames = [
        reading_to_body_signal_frame(0.0, breath_freq=0.22, stare_lock=0.1),
        reading_to_body_signal_frame(1000.0, breath_freq=0.26, stare_lock=0.3),
        reading_to_body_signal_frame(2000.0, breath_freq=0.30, stare_lock=0.5),
    ]
    win = analyze_body_window(frames)
    assert win.n_frames == 3
    assert win.trends["breath_freq"].slope_per_s > 0


def test_corti_grunt_physics_measured_confidence_missing_refusal_and_threshold_reasons():
    import inspect
    import christman_voice_sdk.audio.corti_canal as cc
    from christman_voice_sdk.audio.corti_canal import (
        SpeakerSeat,
        analyze_grunt_physics,
        detect_vocal_stimming,
    )

    # 1. Every threshold has its reason written beside it in the module source
    src = inspect.getsource(cc)
    assert "CHOSEN STARTING VALUE" in src and "held until client data" in src
    for name in (
        "DISTRESS_MAX_ATTACK_MS",
        "DISTRESS_MIN_PEAK_RMS",
        "DISTRESS_MIN_F0_RATIO",
        "DISTRESS_UNSEATED_MIN_F0_HZ",
        "FRUSTRATION_MIN_DURATION_MS",
    ):
        assert hasattr(cc, name), name

    # 2. Missing measurements are NEVER filled in with 200 ms, 80 ms, or 0.02;
    #    they stay None and the component says it could not measure.
    r_none = analyze_grunt_physics(None)
    assert r_none.key == "unknown" and r_none.confidence is None
    assert "could not measure" in r_none.physics_note

    r_no_rms = analyze_grunt_physics({"durationMs": 180.0, "attackMs": 40.0, "f0": 280.0})
    assert r_no_rms.key == "unknown" and r_no_rms.confidence is None
    assert r_no_rms.peak_rms is None and "peak_rms" in r_no_rms.physics_note

    r_no_attack = analyze_grunt_physics({"durationMs": 180.0, "peakRms": 0.16, "f0": 280.0})
    assert r_no_attack.key == "unknown" and r_no_attack.confidence is None
    assert r_no_attack.attack_ms is None and "attack_ms" in r_no_attack.physics_note

    r_no_dur = analyze_grunt_physics({"attackMs": 40.0, "peakRms": 0.16, "f0": 280.0})
    assert r_no_dur.key == "unknown" and r_no_dur.confidence is None
    assert r_no_dur.duration_ms is None and "duration_ms" in r_no_dur.physics_note

    # Stimming also refuses when duration or f0 is missing on any recent event
    assert detect_vocal_stimming([
        {"f0": 180.0},
        {"f0": 180.0},
        {"f0": 180.0},
    ]) is False

    # 3. Confidence is measured from the sound, never fixed at 0.86 / 0.88 / 0.79 / 0.75
    seat = SpeakerSeat()
    for hz in (130.0, 132.0, 134.0):
        seat.observe(hz, 0.05)

    d_mild = analyze_grunt_physics({"durationMs": 190.0, "attackMs": 75.0, "peakRms": 0.125, "f0": 185.0}, seat=seat)
    d_strong = analyze_grunt_physics({"durationMs": 190.0, "attackMs": 15.0, "peakRms": 0.24, "f0": 260.0}, seat=seat)
    assert d_mild.key == "grunt_distress" and d_strong.key == "grunt_distress"
    assert d_mild.confidence is not None and d_strong.confidence is not None
    assert d_strong.confidence > d_mild.confidence

    f_mild = analyze_grunt_physics({"durationMs": 470.0, "attackMs": 120.0, "peakRms": 0.055, "f0": 140.0}, seat=seat)
    f_strong = analyze_grunt_physics({"durationMs": 850.0, "attackMs": 120.0, "peakRms": 0.14, "f0": 140.0}, seat=seat)
    assert f_mild.key == "grunt_frustration" and f_strong.key == "grunt_frustration"
    assert f_mild.confidence is not None and f_strong.confidence is not None
    assert f_strong.confidence > f_mild.confidence

    a_quiet = analyze_grunt_physics({"durationMs": 380.0, "attackMs": 95.0, "peakRms": 0.018, "f0": 132.0}, seat=seat)
    a_clear = analyze_grunt_physics({"durationMs": 160.0, "attackMs": 140.0, "peakRms": 0.09, "f0": 132.0}, seat=seat)
    assert a_quiet.key == "grunt_acknowledge" and a_clear.key == "grunt_acknowledge"
    assert a_quiet.confidence is not None and a_clear.confidence is not None
    assert a_clear.confidence != a_quiet.confidence

    stim_tight = [
        {"durationMs": 300.0, "attackMs": 90.0, "peakRms": 0.06, "f0": 180.0},
        {"durationMs": 300.0, "attackMs": 90.0, "peakRms": 0.06, "f0": 180.0},
        {"durationMs": 300.0, "attackMs": 90.0, "peakRms": 0.06, "f0": 180.0},
    ]
    stim_loose = [
        {"durationMs": 260.0, "attackMs": 90.0, "peakRms": 0.06, "f0": 172.0},
        {"durationMs": 340.0, "attackMs": 90.0, "peakRms": 0.06, "f0": 188.0},
        {"durationMs": 290.0, "attackMs": 90.0, "peakRms": 0.06, "f0": 180.0},
    ]
    r_tight = analyze_grunt_physics(stim_tight[-1], seat=seat, recent_events=stim_tight)
    r_loose = analyze_grunt_physics(stim_loose[-1], seat=seat, recent_events=stim_loose)
    assert r_tight.key == "vocal_stimming" and r_loose.key == "vocal_stimming"
    assert r_tight.confidence is not None and r_loose.confidence is not None
    assert r_tight.confidence > r_loose.confidence


def test_whole_house_smell_skin_and_memory_chamber():
    from christman_voice_sdk.audio.whole_house import (
        DEFAULT_SCENES,
        MemoryChamber,
        Scene,
        classify_smell,
        frame_from_sense,
        merge_house,
        whole_environment,
    )

    # Every MemoryChamber starts completely empty with zero default scenes
    assert DEFAULT_SCENES == ()
    chamber = MemoryChamber()
    assert chamber.scenes == []

    # Missing skin pads stay None and refuse rather than inventing [0,0,0,0] / "rest"
    empty_skin = frame_from_sense(None)
    assert empty_skin["available"] is False and empty_skin["posture"] is None
    assert "could not measure" in empty_skin["honesty_note"]

    # OpenSmell, ported from THEWHOLEHOUSE (2026-10-30): its 21 live profiles,
    # named channels, alert at 0.7. One MQ-135 reading is one voltage: no profile.
    from christman_voice_sdk.audio.whole_house import SMELL_PROFILES, classify
    assert len(SMELL_PROFILES) == 21
    one_ch = classify_smell(0.42)
    assert one_ch["available"] is True and one_ch["scalarOnly"] is True
    assert one_ch["hits"] == []
    assert one_ch["honesty"]["clinical"] is False
    assert classify_smell(None)["available"] is False

    # Named channels, scored as classify.ts scores them.
    multi = classify_smell({"channels": {"acetone": 0.9, "propanol": 0.8}})
    top = multi["hits"][0]
    assert top["profileKey"] == "ketoacidosis"
    assert abs(top["confidence"] - 1.0 * 0.85 * (2 ** 0.5 / 3 ** 0.5)) < 1e-9
    assert top["alert"] is False  # 0.694 < 0.7
    strong = classify({"acetone": 1.0, "propanol": 1.0})
    assert strong[0]["profileKey"] == "ketoacidosis" and strong[0]["alert"] is True  # 0.816 >= 0.7
    # A channel under the detect threshold is not read.
    assert classify({"acetone": 0.29}) == []

    skin = frame_from_sense({"pads": [0.8, 0.75, 0.7, 0.72], "imu_delta_g": 0.05})
    assert skin["posture"] == "press"

    # Caller explicitly adds scenes to its own chamber
    chamber.add_scene(Scene(
        id="scene-rain",
        title="Rain on the Window",
        who="listener, companion",
        what="listening to rain on the roof",
        where="quiet room window",
        when="evening",
        feeling="calm, steady",
        weight=1.35,
    ))
    chamber.add_scene(Scene(
        id="scene-bench",
        title="Workbench Frame Lock",
        who="listener, engineer",
        what="measuring a periodic waveform lock",
        where="workbench",
        when="night",
        feeling="focused, clear",
        weight=1.40,
    ))
    chamber.add_scene(Scene(
        id="scene-shadow-test",
        title="Withheld Shadow Scene",
        who="listener, caller",
        what="boundary test on a loud alarm",
        where="hallway",
        when="afternoon",
        feeling="guarded, alert",
        weight=1.25,
        shadow=True,
    ))

    # 1 door with non-unique token ("listener" is in all 3 scenes) refuses gateway
    refused = chamber.complete({"who": "listener"})
    assert refused.completed is None and "gateway refused" in refused.gate_reason

    # 2 doors complete the matching scene
    hit = chamber.complete({"who": "listener companion", "where": "quiet room window"})
    assert hit.completed is not None and hit.completed.id == "scene-rain"

    # Shadow scene returns ShadowHit somatic marker without completing until reclaimed
    sh = chamber.complete({"who": "listener caller", "what": "boundary alarm"})
    assert sh.completed is None
    assert len(sh.shadows) == 1 and sh.shadows[0].scene_id == "scene-shadow-test"

    chamber.reclaim_scene("scene-shadow-test")
    reclaimed = chamber.complete({"who": "listener caller", "what": "boundary alarm"})
    assert reclaimed.completed is not None and reclaimed.completed.id == "scene-shadow-test"

    night = chamber.apply_night(level=1)
    assert night["scenes_after"] == 3
    dream = chamber.apply_rem()
    assert dream is not None and dream.dream is True

    h = merge_house(ear={"speaker": "listener", "text": "hello"}, smell=multi, skin=skin)
    env = whole_environment(h)
    assert env["who"] == "listener"


def test_memory_chamber_starts_empty_and_no_personal_scenes_in_repo():
    from christman_voice_sdk.audio.whole_house import DEFAULT_SCENES, MemoryChamber

    assert DEFAULT_SCENES == ()
    assert MemoryChamber().scenes == []

    # Personal financial/life scene strings must appear nowhere in the repository
    forbidden = (
        "hillsboro",
        "lvnv",
        "credit one",
        "fdcpa",
        "gahanna",
        "debt collector",
        "porch-evening-coffee",
        "booth-first-lock",
        "alphavox-first-lock",
    )
    this_file = Path(__file__).resolve()
    for folder in ("christman_sound", "christman_voice_sdk", "tests"):
        for p in (ROOT / folder).rglob("*"):
            if not p.is_file() or p.resolve() == this_file or "__pycache__" in p.parts:
                continue
            if p.suffix not in (".py", ".html", ".js", ".json", ".md", ".webmanifest"):
                continue
            low = p.read_text(encoding="utf-8", errors="ignore").lower()
            for term in forbidden:
                assert term not in low, f"personal scene string {term!r} found in {p.relative_to(ROOT)}"


def test_studio_integrated_api_routes(tmp_path, monkeypatch):
    from christman_sound.app import create_app

    monkeypatch.setenv("CHRISTMAN_VOICE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FILAMENT_HOME", str(tmp_path / "filament"))
    app = create_app()
    client = app.test_client()

    r_frames = client.post("/api/frames-analyze", json={"f0": 140.0})
    assert r_frames.status_code == 200
    assert r_frames.get_json()["env_bins"] == 96

    r_disp = client.post("/api/dispersion", json={"scene_id": "kitchen"})
    assert r_disp.status_code == 200
    assert r_disp.get_json()["comparison"]["pitch_corrupted_by_mix"] is True

    r_porch = client.post("/api/porch", json={"text": "Merry Christmas from Alpha Vox and hectic feedback"})
    assert r_porch.status_code == 200
    p_data = r_porch.get_json()
    assert "Christmas" in p_data["porch"]["asSaid"] and "Christman" not in p_data["porch"]["asSaid"]
    assert "haptic feedback" in p_data["porch"]["asSaid"]

    # Without an event, /api/corti-canal refuses to guess a grunt route
    r_corti = client.post("/api/corti-canal", json={"hz": 265.0})
    assert r_corti.status_code == 200
    c_json = r_corti.get_json()
    assert 0.0 < c_json["greenwood"]["place"] < 1.0
    assert c_json["grunt_route"]["key"] == "unknown"
    assert c_json["grunt_route"]["confidence"] is None

    r_lucent = client.post("/api/lucent", json={"preset": "keratoconus"})
    assert r_lucent.status_code == 200
    l_json = r_lucent.get_json()
    assert l_json["optics"]["d_cornea_total"] > 45.0
    assert l_json["gaze"]["cell"] is None

    r_being = client.post("/api/being", json={"shaker": True, "ticks": 2, "unit_value": 440.0})
    assert r_being.status_code == 200
    assert r_being.get_json()["units"]["hz"] == 440.0

    # Whole house starts with an empty memory chamber
    r_house_empty = client.post("/api/whole-house", json={})
    assert r_house_empty.status_code == 200
    assert r_house_empty.get_json()["scenes"] == []
    assert r_house_empty.get_json()["smell"]["available"] is False
    assert r_house_empty.get_json()["skin"]["available"] is False

    # Adding a scene explicitly and querying 2 doors completes that scene
    r_house = client.post("/api/whole-house", json={
        "add_scene": {
            "id": "scene-1",
            "title": "Rain on the Roof",
            "who": "listener",
            "where": "quiet room",
            "what": "steady rain",
        },
        "memory_query": {"who": "listener", "where": "quiet room"},
    })
    assert r_house.status_code == 200
    assert r_house.get_json()["memory_completion"]["completed"]["id"] == "scene-1"


def _test_tone_wav(path, seconds=1.5, sr=22050, f0=150.0):
    """A plain harmonic test tone, written for the test: no being's recording."""
    import array
    import wave

    n = int(seconds * sr)
    pcm = array.array("h", (int(9000 * sum(math.sin(2 * math.pi * f0 * k * i / sr) / k for k in range(1, 8)) / 2.6)
                            for i in range(n)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def _no_lexicon_anywhere(tmp_path, monkeypatch):
    """No lexicon and no model folder this process can find (2026-10-30)."""
    monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))
    monkeypatch.setenv("CHRISTMAN_LEXICON_PATH", str(tmp_path / "no_lexicon_here.txt"))
    monkeypatch.setenv("VOSK_MODEL_PATH", str(tmp_path / "no_model_here"))
    monkeypatch.delenv("CHRISTMAN_VOSK_MODEL", raising=False)


def test_voice_creation_center_controls_presets_and_frames_room(tmp_path, monkeypatch):
    from christman_sound.app import create_app
    from christman_sound.voice_creation_center import (
        VOICE_CREATION_CONTROLS,
        VOICE_CREATION_PRESETS,
        resolve_vcc_controls,
    )

    assert len(VOICE_CREATION_CONTROLS) == 12
    assert len(VOICE_CREATION_PRESETS) == 6

    resolved = resolve_vcc_controls(preset="warm_grounding", overrides={"cadence": 0.42})
    assert resolved["cadence"] == 0.42
    assert resolved["resonance"] == "chest"
    assert resolved["affect"] == "neutral"

    monkeypatch.setenv("CHRISTMAN_VOICE_CACHE", str(tmp_path / "vcc_cache"))
    app = create_app()
    client = app.test_client()

    meta = client.get("/api/voice-creation-center").get_json()
    assert meta["status"] == "ok"
    assert len(meta["controls"]) == 12
    assert "warm_grounding" in meta["presets"]

    # No recording, no name: refused, nothing reached for (2026-10-30).
    r = client.post("/api/voice-creation-center", json={"action": "resynthesize_frames"})
    assert r.status_code == 400 and r.get_json()["kind"] == "NoRecording"
    r = client.post("/api/voice-creation-center", json={"action": "measure_profile"})
    assert r.status_code == 400 and r.get_json()["kind"] == "NoName"
    r = client.post("/api/voice-creation-center", json={"action": "speak_vcc", "text": "i need help"})
    assert r.get_json()["status"] == "refused"

    tone = _test_tone_wav(tmp_path / "tone.wav")
    r_resyn = client.post("/api/voice-creation-center",
                          json={"action": "resynthesize_frames", "wav_path": str(tone)}).get_json()
    assert r_resyn["status"] == "ok"
    assert r_resyn["n_frames"] > 50
    assert r_resyn["source_audio_url"].startswith("/audio/")
    assert r_resyn["rendered_audio_url"].startswith("/audio/")


def test_voice_creation_center_refuses_to_measure_without_a_lexicon_and_fills_nothing_in(tmp_path, monkeypatch):
    """
    2026-10-30: the Voice Creation Center measures or refuses. With no
    lexicon it refuses by name, writes no profile and no voice, and writes
    no lexicon of its own (it used to write a hidden ten-word one).
    """
    from christman_sound.app import create_app

    _no_lexicon_anywhere(tmp_path, monkeypatch)
    cache = tmp_path / "vcc_cache"
    monkeypatch.setenv("CHRISTMAN_VOICE_CACHE", str(cache))
    client = create_app().test_client()
    rec = tmp_path / "phrase.wav"
    rec.write_bytes(b"")  # never read: the refusal comes before any recording is opened

    r = client.post("/api/voice-creation-center", json={
        "action": "measure_profile", "name": "measured", "wav_paths": [str(rec)]}).get_json()
    assert r["status"] == "refused"
    assert r["kind"] == "LexiconUnavailable"
    assert not (cache / "profile_measured.json").exists()

    r = client.post("/api/voice-creation-center", json={
        "action": "build_frame_voice", "name": "built", "wav_paths": [str(rec)]}).get_json()
    assert r["status"] == "refused"
    assert r["kind"] == "LexiconUnavailable"
    assert not (cache / "frame_voice_built" / "voice.json").exists()
    assert not (cache / "tmp_vcc_default_lex.txt").exists()


def test_voice_creation_center_measures_and_builds_from_a_beings_recordings(tmp_path, monkeypatch):
    """
    The Voice Creation Center runs the same measurement and build as
    scripts/measure_voice.py and scripts/build_voice_bank.py (2026-10-30):
    every applied vowel was measured, every unit's word came from the ear
    with its own confidence, and the profile says what measured each value.
    Needs CHRISTMAN_VOICE_RECORDINGS (a being's recorded phrases, 22050 Hz
    16-bit mono; CHRISTMAN_VOICE_PREFIX is their file name prefix),
    CHRISTMAN_VOSK_MODEL, CHRISTMAN_LEXICON_PATH, and Praat (parselmouth).
    """
    import json

    need = {k: os.environ.get(k) for k in
            ("CHRISTMAN_VOICE_RECORDINGS", "CHRISTMAN_VOSK_MODEL", "CHRISTMAN_LEXICON_PATH")}
    missing = [k for k, v in need.items() if not v]
    if missing:
        pytest.skip("not run: needs " + ", ".join(missing) +
                    " (a being's recordings, the model folder, its lexicon); runs on Everett's Mac")
    pytest.importorskip("parselmouth", reason="not run: Praat (parselmouth) is not installed here")
    from christman_sound.voice_creation_center import build_frame_voice_center, measure_and_prove_voice_profile

    prefix = os.environ.get("CHRISTMAN_VOICE_PREFIX", "")
    recs = sorted(Path(need["CHRISTMAN_VOICE_RECORDINGS"]).expanduser().glob(prefix + "*.wav"))
    assert recs, "CHRISTMAN_VOICE_RECORDINGS holds no recordings"

    res = measure_and_prove_voice_profile("measured voice", recs, tmp_path / "voice.json",
                                          lexicon_path=need["CHRISTMAN_LEXICON_PATH"],
                                          vosk_model=need["CHRISTMAN_VOSK_MODEL"], prefix=prefix)
    prof = json.loads((tmp_path / "voice.json").read_text(encoding="utf-8"))
    assert res["vowels_measured"] == len(prof["vowels"])
    for vowel, entry in prof["vowels"].items():
        assert entry["n_tokens"] >= 2, vowel
    assert prof["rate"]["n_phrases"] >= 1
    assert res["proof"] is not None

    fv = build_frame_voice_center("built voice", recs, tmp_path / "frame_voice",
                                  lexicon_path=need["CHRISTMAN_LEXICON_PATH"],
                                  vosk_model=need["CHRISTMAN_VOSK_MODEL"])
    doc = json.loads((tmp_path / "frame_voice" / "voice.json").read_text())
    assert doc["provenance"]["builder"] == "Christman-Sound scripts/build_voice_bank.py"
    assert "Praat" in doc["provenance"]["measured_by"]["jitter, shimmer, hnr_db"]
    assert fv.measures["jitter_carried"] <= fv.measures["jitter"]
    # Every word's confidence is the ear's own: they are not one stamped value.
    confs = [t["conf"] for t in json.loads((tmp_path / "frame_voice" / "bank.json").read_text())["tokens"]]
    assert confs and len(set(confs)) > 1


def test_voice_pack_builds_from_a_beings_recordings_and_loads_back(tmp_path):
    """
    2026-10-30: the voice pack calls methods that exist. Needs
    CHRISTMAN_VOICE_RECORDINGS (a being's recordings), torch (the timbre
    modeler imports it) and Praat (voice quality).
    """
    folder = os.environ.get("CHRISTMAN_VOICE_RECORDINGS")
    if not folder:
        pytest.skip("not run: needs CHRISTMAN_VOICE_RECORDINGS (a being's recorded phrases); runs on Everett's Mac")
    pytest.importorskip("torch", reason="not run: torch is not installed here (the timbre modeler imports it)")
    pytest.importorskip("parselmouth", reason="not run: Praat (parselmouth) is not installed here")
    from christman_sound.voice_creation_center import package_voicepack_center

    recs = sorted(Path(folder).expanduser().glob(os.environ.get("CHRISTMAN_VOICE_PREFIX", "") + "*.wav"))[:4]
    res = package_voicepack_center("test pack", recs, tmp_path / "packs" / "test.voicepack")
    assert res["status"] == "ok"
    assert Path(res["voicepack_path"]).is_file()
    assert res["reference_audio_count"] == len(recs)
    assert res["metadata"]["name"] == "test pack"


def test_new_modules_import_no_numpy_scipy_or_outside_audio_packages(tmp_path):
    """Run frames, dispersion, being, voice_creation_center, corti_canal,
    lucent_reader, whole_house, and ear/porch with numpy, scipy, soundfile,
    sounddevice, pygame, and librosa blocked via builtins.__import__."""
    code = f"""
import builtins, sys
BLOCK = {{"numpy", "scipy", "soundfile", "sounddevice", "pygame", "librosa"}}
real = builtins.__import__
def guard(name, *a, **k):
    if name.split(".")[0] in BLOCK:
        raise ImportError("blocked " + name)
    return real(name, *a, **k)
builtins.__import__ = guard
sys.path.insert(0, {str(ROOT)!r})

from christman_sound.frames import analyze_frames
from christman_sound.dispersion import discern_canals
from christman_sound.being import OPEN_FRAGMENTS, convert_unit, create_stasis_field, tick_stasis
from christman_sound.voice_creation_center import resynthesize_from_frames
from christman_sound.ear.porch import reconstruct_dialect
from christman_voice_sdk.audio.corti_canal import freq_to_place, voice_quality_dsp
from christman_voice_sdk.audio.lucent_reader import CORNEA_PRESETS, evaluate_optics
from christman_voice_sdk.audio.whole_house import MemoryChamber, classify_smell

rep = discern_canals("kitchen")
assert len(rep["canals"]) == 3
import array, math, wave
tone = {str(tmp_path / "tone.wav")!r}
sr = 22050
pcm = array.array("h", (int(9000 * sum(math.sin(2 * math.pi * 150.0 * k * i / sr) / k for k in range(1, 8)) / 2.6)
                        for i in range(int(1.5 * sr))))
with wave.open(tone, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
resyn = resynthesize_from_frames(tone, {str(tmp_path / "resyn")!r}, max_seconds=1.5)
assert resyn["n_frames"] > 10
field = create_stasis_field(OPEN_FRAGMENTS)
tick_stasis(field, steps=2)
assert convert_unit(440.0, "hz")["hz"] == 440.0
assert "misread" in reconstruct_dialect("Miss Reed").as_said
assert 0.0 < freq_to_place(440.0) < 1.0
assert evaluate_optics(CORNEA_PRESETS["emmetropic"]).d_cornea_total > 40.0
assert classify_smell(0.5)["hits"] == []
loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCK)
print("OK", loaded)
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK []"
