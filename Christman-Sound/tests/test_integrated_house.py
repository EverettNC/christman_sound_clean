# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
End-to-end tests for the integrated 7-repository Christman-Sound house
architecture:
  1. christman_sound.frames (4.99 ms frame analysis, 5-pass Viterbi forced
     alignment, native write_voice_bank -> FrameBank round-trip)
  2. christman_sound.dispersion (4-canal Dispersion & War, Refused-Mix guard)
  3. christman_sound.being (Silico-Stasis orbital field, 2 Hz shaker, 9-unit
     psychoacoustic converter, streaming codec inspector)
  4. christman_sound.ear.{mouth,board,porch} (Filament mouth map, CONF_FLOOR=0.55
     Carbon pile, Family/Residual boards, 15 dialect regions, TCAP-V-0020
     mouth-cooperation, 33 family names, Vosk 0.22 discovery)
  5. christman_voice_sdk.audio.corti_canal (Greenwood cochlea, cycle DSP voice
     quality, SpeakerSeat >= 3 locks, AlphaVox grunt physics & stimming router,
     20 Hz CanalRecorder & disk writer)
  6. christman_voice_sdk.audio.lucent_reader (Gullstrand cornea optics, 4x4 head
     pose, 9-cell gaze, iris/mouth landmarks, BlinkLog, 5-part RhythmTracker,
     BodySignalFrame bridge)
  7. christman_voice_sdk.audio.whole_house (OpenSmell 21-profile classifier with
     1-scalar refusal guard, Skin haptic posture, House Fifth Stream, and
     TCAP-V-0044 Five-Element Memory Chamber)
  8. Studio API endpoints (/api/frames-analyze, /api/dispersion, /api/porch,
     /api/corti-canal, /api/lucent, /api/being, /api/whole-house)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_frames_analyze_and_native_voice_bank_roundtrip(tmp_path):
    from christman_sound.frames import FRAME_SR, HOP, analyze_frames, write_voice_bank
    from christman_sound.stitch import FrameBank, select_units, join_units
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
    assert track.env_db.shape == (track.n, 96)
    assert track.ap.shape == (track.n, 8)
    assert np.any(track.f0 > 0)

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


def test_being_silico_stasis_units_and_stream():
    from christman_sound.being import (
        OPEN_FRAGMENTS,
        attach_perturbations,
        convert_unit,
        create_stasis_field,
        inspect_stream,
        place_perturbation,
        shaker_pulse,
        tick_stasis,
    )

    field = create_stasis_field(OPEN_FRAGMENTS)
    assert len(field.nodes) == len(OPEN_FRAGMENTS)
    p = place_perturbation("test-shock", magnitude=0.4, layer="membrane")
    attach_perturbations(field, [p, shaker_pulse(0.25)])
    tick_stasis(field, steps=5)
    assert field.ticks == 5
    assert 0.0 < field.quantities.coherence <= 1.0

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
    assert "Christman" in dressed["text"]
    assert "Everett" in dressed["text"]
    assert "sumbitch" in dressed["text"]
    assert dressed["carbon"] is True
    assert len(dressed["unsure"]) == 1

    porch = reconstruct_dialect("Alpha Vox said Miss Reed felt hectic feedback and leaved early with Broxton")
    assert "AlphaVox" in porch.as_said
    assert "misread" in porch.as_said
    assert "haptic feedback" in porch.as_said
    assert "lived" in porch.as_said
    assert "Brockston" in porch.as_said
    assert "AlphaVox" in porch.route
    assert "Brockston" in porch.route

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
    sil_q = voice_quality_dsp(np.zeros(4096), 16000)
    assert sil_q == {"jitter": None, "shimmer": None, "hnr_db": None}

    # Clean periodic wave locks jitter/shimmer/HNR
    sr = 16000
    t = np.arange(sr // 2, dtype=np.float64) / sr
    wave_sig = 0.5 * np.sin(2.0 * np.pi * 160.0 * t)
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

    # AlphaVoxRouter: stimming vs acute distress
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

    gaze = gaze_from_blendshapes({"eyeLookOutRight": 0.6, "eyeLookInLeft": 0.6})
    assert gaze["cell"] == "middle-right"

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


def test_whole_house_smell_skin_and_memory_chamber():
    from christman_voice_sdk.audio.whole_house import (
        MemoryChamber,
        classify_smell,
        frame_from_sense,
        merge_house,
        whole_environment,
    )

    # 1 scalar channel refuses to invent chemical species
    one_ch = classify_smell(0.42)
    assert one_ch["channel_count"] == 1
    assert one_ch["species"] == []
    assert "cannot separate chemical species" in one_ch["honesty_note"]

    # 4-channel vector matches coffee roast
    multi = classify_smell({"channels": [0.62, 0.48, 0.18, 0.35]})
    assert multi["species"][0]["id"] == "coffee_roast"

    skin = frame_from_sense({"pads": [0.8, 0.75, 0.7, 0.72], "imu_delta_g": 0.05})
    assert skin["posture"] == "press"

    chamber = MemoryChamber()
    # 1 door with non-unique token refuses gateway
    refused = chamber.complete({"who": "Everett"})
    assert refused.completed is None and "gateway refused" in refused.gate_reason

    # 2 doors complete the porch coffee scene
    hit = chamber.complete({"who": "Everett Misty", "where": "porch coffee"})
    assert hit.completed is not None and hit.completed.id == "porch-evening-coffee"

    # Shadow scene returns ShadowHit somatic marker without completing until reclaimed
    sh = chamber.complete({"who": "Everett debt collector", "what": "FDCPA validation Hillsboro"})
    assert sh.completed is None
    assert len(sh.shadows) == 1 and sh.shadows[0].scene_id == "hillsboro-fraud-dispute"

    chamber.reclaim_scene("hillsboro-fraud-dispute")
    reclaimed = chamber.complete({"who": "Everett debt collector", "what": "FDCPA validation Hillsboro"})
    assert reclaimed.completed is not None and reclaimed.completed.id == "hillsboro-fraud-dispute"

    night = chamber.apply_night(level=1)
    assert night["scenes_after"] >= 3
    dream = chamber.apply_rem()
    assert dream is not None and dream.dream is True

    h = merge_house(ear={"speaker": "Everett", "text": "hello"}, smell=multi, skin=skin)
    env = whole_environment(h)
    assert env["who"] == "Everett"


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
    assert "Christman" in p_data["porch"]["asSaid"]
    assert "haptic feedback" in p_data["porch"]["asSaid"]

    r_corti = client.post("/api/corti-canal", json={"hz": 265.0})
    assert r_corti.status_code == 200
    assert 0.0 < r_corti.get_json()["greenwood"]["place"] < 1.0

    r_lucent = client.post("/api/lucent", json={"preset": "keratoconus"})
    assert r_lucent.status_code == 200
    assert r_lucent.get_json()["optics"]["d_cornea_total"] > 45.0

    r_being = client.post("/api/being", json={"shaker": True, "ticks": 2, "unit_value": 440.0})
    assert r_being.status_code == 200
    assert r_being.get_json()["units"]["hz"] == 440.0

    r_house = client.post("/api/whole-house", json={"memory_query": {"who": "Everett", "where": "porch coffee"}, "run_rem": True})
    assert r_house.status_code == 200
    assert r_house.get_json()["memory_completion"]["completed"]["id"] == "porch-evening-coffee"


def test_voice_creation_center_controls_presets_frames_and_bank(tmp_path, monkeypatch):
    from christman_sound.app import create_app
    from christman_sound.voice_creation_center import (
        VOICE_CREATION_CONTROLS,
        VOICE_CREATION_PRESETS,
        bundled_alphavox_wavs,
        resolve_vcc_controls,
    )

    assert len(VOICE_CREATION_CONTROLS) == 12
    assert len(VOICE_CREATION_PRESETS) == 6
    assert len(bundled_alphavox_wavs()) == 2

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

    # 1. Frames Room A/B Resynthesis on bundled AlphaVox recording
    r_resyn = client.post("/api/voice-creation-center", json={"action": "resynthesize_frames"}).get_json()
    assert r_resyn["status"] == "ok"
    assert r_resyn["n_frames"] > 50
    assert r_resyn["source_audio_url"].startswith("/audio/")
    assert r_resyn["rendered_audio_url"].startswith("/audio/")

    # 2. Measure & Prove VoiceProfile
    r_prof = client.post("/api/voice-creation-center", json={"action": "measure_profile", "name": "AlphaVox_Test"}).get_json()
    assert r_prof["status"] == "ok"
    assert r_prof["f0_median_hz"] > 60.0
    assert len(r_prof["fingerprint"]) == 64

    # 3. Build native FrameVoice bank from bundled AlphaVox recordings and speak with VCC controls
    r_fv = client.post("/api/voice-creation-center", json={"action": "build_frame_voice", "name": "AlphaVox_FV"}).get_json()
    assert r_fv["status"] == "ok"
    assert r_fv["units_count"] > 0

    r_spk = client.post("/api/voice-creation-center", json={
        "action": "speak_vcc",
        "text": "i need help please",
        "voice": r_fv["frame_voice_dir"],
        "preset": "warm_grounding",
    }).get_json()
    assert r_spk["status"] == "ok"
    assert r_spk["renderer"] == "christman_vocoder"
    assert r_spk["audio_url"].startswith("/audio/")
