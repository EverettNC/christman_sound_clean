# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_sound.app — Local installable application server, PWA studio host,
and native desktop bundle installer (macOS .app / Linux .desktop).

Pure Python standard library + C engine. Binds strictly to loopback (127.0.0.1)
by default. Wires the entire Christman-Sound + 7-repository house architecture
without stubs or outside engines:
  - Sovereign voice synthesis & Frame Voices (christman_sound.synth, stitch, frames)
  - Codec War bare table verification (render_codecwar_table)
  - Multi-canal Dispersion & War bench (christman_sound.dispersion)
  - Silico-Stasis equilibrium, 2 Hz shaker & psychoacoustic units (christman_sound.being)
  - Hearing, structural affect, harm frame, crisis gate & nonverbal (Hearing)
  - Corti Greenwood cochlea, cycle DSP quality, SpeakerSeat, AlphaVoxRouter &
    20 Hz CanalRecorder (christman_voice_sdk.audio.corti_canal)
  - Lucent Gullstrand cornea optics, 5-part RhythmTracker & BodySignalFrame
    bridge (christman_voice_sdk.audio.lucent_reader & body_signals)
  - Whole House OpenSmell 21-profile classifier, Skin haptic posture, House
    Fifth Stream & TCAP-V-0044 Five-Element Memory Chamber (whole_house)
  - The Ear + Filament Mouth Map + Carbon Pile + Family/Residual Boards +
    Porch TCAP-V-0020 Mouth-Cooperation (christman_sound.ear)
  - Auto-discovery for ~/Downloads/vosk-model-en-us-0.22
"""

from __future__ import annotations

import argparse
import email.parser
import json
import logging
import mimetypes
import os
import re
import stat
import subprocess
import sys
import tempfile
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from . import dsp as dsp_mod
from .being import (
    ENCODINGS,
    OPEN_FRAGMENTS,
    StasisField,
    attach_perturbations,
    convert_unit,
    create_stasis_field,
    inspect_stream,
    place_perturbation,
    shaker_pulse,
    tick_stasis,
)
from .dispersion import SCENES, discern_canals, fuse_pcm, render_canal_source
from .dsp import DSPUnavailable, PlaybackUnavailable, SynthRefused, SynthUnavailable, get_dsp
from .ear.board import DIALECT_MAP, add_family, add_residual, snapshot as board_snapshot
from .ear.mouth import CONF_FLOOR, dress as dress_filament, load_carbon, save_carbon
from .ear.porch import FAMILY_NAMES, MOUTH_COOPERATION, reconstruct_dialect, recover_mangles
from .ear.recognizer import discover_vosk_model
from .formants import measure_formants
from .frames import analyze_frames
from .lexicon import (
    LEXICON_ENV,
    Lexicon,
    LexiconFormatError,
    LexiconUnavailable,
    PhoneNotMapped,
    WordNotInLexicon,
    discover_align_lexicon,
    load_lexicon,
)
from .stitch import FrameVoiceUnavailable, StitchRefused
from .synth import (
    CHRISTMAN_SOUND_SAMPLE_RATE,
    CODECWAR_SAMPLE_RATE,
    FRAME_CONTROLS,
    NoVoice,
    NothingToSay,
    UnknownPhone,
    render_codecwar_table,
    speak,
    synthesize,
    write_wav,
)
from .voice_creation_center import (
    BOOTH_SCRIPT_SAMPLES,
    VOICE_CREATION_CONTROLS,
    VOICE_CREATION_PRESETS,
    build_frame_voice_center,
    bundled_alphavox_wavs,
    measure_and_prove_voice_profile,
    package_voicepack_center,
    resolve_vcc_controls,
    resynthesize_from_frames,
)
from .voice_profile import VoiceProfileError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
_STATIC_DIR = _PACKAGE_DIR / "static"

DEFAULT_APP_HOST = os.getenv("CHRISTMAN_APP_HOST", "127.0.0.1")
DEFAULT_APP_PORT = int(os.getenv("CHRISTMAN_APP_PORT", "5480"))

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _audio_cache_dir() -> Path:
    base = Path(os.environ.get("CHRISTMAN_VOICE_CACHE") or (Path.home() / ".christman_ai" / "voice_cache")).expanduser()
    base.mkdir(parents=True, mode=0o700, exist_ok=True)
    return base


def _parse_multipart(body: bytes, content_type: str) -> Tuple[Dict[str, str], Dict[str, bytes]]:
    fields: Dict[str, str] = {}
    files: Dict[str, bytes] = {}
    header_bytes = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    msg = email.parser.BytesParser().parsebytes(header_bytes + body)
    if not msg.is_multipart():
        return fields, files
    for part in msg.iter_parts():
        disp = part.get("Content-Disposition", "")
        name_m = re.search(r'name="([^"]+)"', disp)
        if not name_m:
            continue
        field_name = name_m.group(1)
        payload = part.get_payload(decode=True) or b""
        if 'filename="' in disp:
            files[field_name] = payload
        else:
            fields[field_name] = payload.decode("utf-8", errors="replace")
    return fields, files


@dataclass
class AppResponse:
    status_code: int
    data: bytes
    content_type: str = "application/json"

    def get_json(self) -> Any:
        return json.loads(self.data.decode("utf-8"))


def _json_resp(payload: Any, status_code: int = 200) -> AppResponse:
    return AppResponse(
        status_code=status_code,
        data=json.dumps(payload).encode("utf-8"),
        content_type="application/json; charset=utf-8",
    )


class ChristmanSoundApp:
    """Pure-stdlib application router with an in-memory test client and HTTP server."""

    def __init__(self) -> None:
        from christman_voice_sdk.audio.corti_canal import CanalRecorder, SpeakerSeat, VocalTracker
        from christman_voice_sdk.audio.lucent_reader import BlinkLog, RhythmTracker
        from christman_voice_sdk.audio.whole_house import MemoryChamber

        self.stasis_field: StasisField = create_stasis_field(OPEN_FRAGMENTS)
        self.vocal_tracker = VocalTracker()
        self.speaker_seat = SpeakerSeat()
        self.canal_recorder = CanalRecorder()
        self.blink_log = BlinkLog()
        self.rhythm_tracker = RhythmTracker()
        self.memory_chamber = MemoryChamber()

    def dispatch(
        self,
        method: str,
        raw_path: str,
        body: bytes = b"",
        headers: Optional[Dict[str, str]] = None,
    ) -> AppResponse:
        headers = headers or {}
        parsed = urlsplit(raw_path)
        path = parsed.path

        if method == "GET":
            if path == "/" or path == "/index.html":
                html_path = _STATIC_DIR / "index.html"
                return AppResponse(200, html_path.read_bytes(), "text/html; charset=utf-8")
            if path == "/manifest.webmanifest":
                m_path = _STATIC_DIR / "manifest.webmanifest"
                return AppResponse(200, m_path.read_bytes(), "application/manifest+json; charset=utf-8")
            if path == "/sw.js":
                sw_path = _STATIC_DIR / "sw.js"
                return AppResponse(200, sw_path.read_bytes(), "application/javascript; charset=utf-8")
            if path.startswith("/static/"):
                rel = path[len("/static/"):]
                if ".." in rel or rel.startswith("/"):
                    return _json_resp({"error": "invalid path"}, 400)
                target = (_STATIC_DIR / rel).resolve()
                if not str(target).startswith(str(_STATIC_DIR.resolve())) or not target.is_file():
                    return _json_resp({"error": "not found"}, 404)
                mime, _ = mimetypes.guess_type(str(target))
                return AppResponse(200, target.read_bytes(), mime or "application/octet-stream")
            if path.startswith("/audio/"):
                fname = path[len("/audio/"):]
                if not _SAFE_FILENAME.match(fname) or not fname.endswith(".wav"):
                    return _json_resp({"error": "invalid filename"}, 400)
                wav_path = _audio_cache_dir() / fname
                if not wav_path.is_file():
                    return _json_resp({"error": "audio not found"}, 404)
                return AppResponse(200, wav_path.read_bytes(), "audio/wav")
            if path == "/api/status" or path == "/status":
                return self._handle_status()
            if path == "/api/voice-creation-center":
                return self._handle_vcc_get()
            if path == "/api/porch":
                return _json_resp({
                    "status": "ok",
                    "carbon_floor": CONF_FLOOR,
                    "carbon_pile": load_carbon(),
                    "board": board_snapshot(),
                    "family_names": list(FAMILY_NAMES),
                    "mouth_cooperation": list(MOUTH_COOPERATION),
                }, 200)
            return _json_resp({"error": "not found"}, 404)

        if method == "POST":
            ctype = headers.get("Content-Type") or headers.get("content-type") or ""
            if path == "/api/build-engines":
                return self._handle_build_engines()
            if path == "/api/install-desktop":
                return self._handle_install_desktop()
            if path == "/api/synthesize":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_synthesize(data)
            if path == "/api/codecwar-table":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_codecwar_table(data)
            if path == "/api/hear":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_hear(data)
            if path == "/api/measure-wav":
                return self._handle_measure_wav(body, ctype)
            if path == "/api/recognize-wav":
                return self._handle_recognize_wav(body, ctype)
            if path == "/api/voice-creation-center":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_vcc_post(data)
            if path == "/api/frames-analyze":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_frames_analyze(data)
            if path == "/api/dispersion":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_dispersion(data)
            if path == "/api/voice-creation-center":
                return self._handle_vcc_get()
            if path == "/api/porch":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_porch(data)
            if path == "/api/corti-canal":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_corti_canal(data)
            if path == "/api/lucent":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_lucent(data)
            if path == "/api/being":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_being(data)
            if path == "/api/whole-house":
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                return self._handle_whole_house(data)
            return _json_resp({"error": "not found"}, 404)

        return _json_resp({"error": "method not allowed"}, 405)

    def _handle_status(self) -> AppResponse:
        dsp = get_dsp()
        ear_lib = _PACKAGE_DIR / "ear" / ("libear.dylib" if sys.platform == "darwin" else "libear.so")
        disc_lex = discover_align_lexicon()
        disc_vosk = discover_vosk_model()
        lex_env = os.environ.get(LEXICON_ENV) or (str(disc_lex) if disc_lex else "")
        vosk_env = os.environ.get("VOSK_MODEL_PATH") or (str(disc_vosk) if disc_vosk else "")

        profiles: List[str] = []
        try:
            from christman_voice_sdk.integration.voice_capture_client import list_profiles
            profiles = list_profiles()
        except Exception:
            profiles = []

        phones: List[str] = []
        if dsp.available and dsp.synth_available:
            try:
                phones = list(dsp.synth_phone_table().keys())
            except Exception:
                phones = []

        return _json_resp({
            "status": "online",
            "version": "1.1.0",
            "cloud_dependencies": False,
            "dsp": {
                "available": dsp.available,
                "synth_available": dsp.synth_available,
                "path": str(dsp.path) if dsp.path else None,
                "error": dsp.error,
                "synth_error": dsp.synth_error,
                "play_backend": dsp.play_backend() if dsp.available else None,
                "phones": phones,
            },
            "ear": {
                "library_built": ear_lib.is_file(),
                "library_path": str(ear_lib),
                "vosk_model_path": vosk_env or None,
                "vosk_model_present": bool(vosk_env and Path(vosk_env).expanduser().is_dir()),
            },
            "lexicon": {
                "env_path": lex_env or None,
                "present": bool(lex_env and Path(lex_env).expanduser().is_file()),
            },
            "organs": {
                "mouth_synth": dsp.available and dsp.synth_available,
                "frames_stitch": True,
                "dispersion_war": list(SCENES.keys()),
                "filament_porch": {
                    "dialect_regions": len(DIALECT_MAP),
                    "family_names": len(FAMILY_NAMES),
                    "carbon_floor": CONF_FLOOR,
                },
                "corti_canal": {
                    "seated": self.speaker_seat.seated,
                    "canal_frames": len(self.canal_recorder.frames),
                },
                "lucent_cornea": True,
                "being_stasis": self.stasis_field.quantities.to_dict(),
                "whole_house_memory": {
                    "scenes": len(self.memory_chamber.scenes),
                },
            },
            "profiles": profiles,
            "frame_controls": list(FRAME_CONTROLS),
            "sample_rates": [CODECWAR_SAMPLE_RATE, CHRISTMAN_SOUND_SAMPLE_RATE],
            "timestamp": datetime.now().isoformat(),
        })

    def _handle_build_engines(self) -> AppResponse:
        results: Dict[str, Any] = {}
        build_dsp_script = _REPO_ROOT / "scripts" / "build_dsp.py"
        if build_dsp_script.is_file():
            proc = subprocess.run(
                [sys.executable, str(build_dsp_script), "--check"],
                capture_output=True, text=True, cwd=str(_REPO_ROOT),
            )
            results["dsp"] = {
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        else:
            results["dsp"] = {"returncode": 1, "error": f"{build_dsp_script} not found"}

        proc_ear = subprocess.run(
            [sys.executable, "-m", "christman_sound.ear.build", "--check"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT),
        )
        results["ear"] = {
            "returncode": proc_ear.returncode,
            "stdout": proc_ear.stdout.strip(),
            "stderr": proc_ear.stderr.strip(),
        }
        dsp_mod._instance = None
        dsp = get_dsp()
        results["dsp_loaded"] = dsp.available and dsp.synth_available
        code = 200 if results["dsp"].get("returncode") == 0 and results["ear"].get("returncode") == 0 else 500
        return _json_resp(results, code)

    def _handle_install_desktop(self) -> AppResponse:
        try:
            installed = install_desktop_bundle()
            return _json_resp({"status": "ok", "installed": installed}, 200)
        except Exception as exc:
            return _json_resp({"status": "error", "reason": str(exc)}, 500)

    def _handle_synthesize(self, data: Dict[str, Any]) -> AppResponse:
        text = data.get("text")
        phones_raw = data.get("phones")
        voice_arg = data.get("voice")
        voice_json_inline = data.get("voice_profile_json")
        disc_lex = discover_align_lexicon()
        lexicon_path = data.get("lexicon_path") or os.environ.get(LEXICON_ENV) or (str(disc_lex) if disc_lex else None)
        lexicon_inline = data.get("lexicon_text")
        sample_rate = data.get("sample_rate")
        play_aloud = bool(data.get("play_aloud", False))
        being_reply = bool(data.get("being_reply", False))
        silent_alarm = bool(data.get("silent_alarm", False))

        if being_reply and text:
            try:
                from christman_voice_sdk.synthesis.reply_guard import ForbiddenReply, check_being_reply
                check_being_reply(text, silent_alarm=silent_alarm)
            except ForbiddenReply as exc:
                return _json_resp({
                    "available": False,
                    "status": "forbidden_reply",
                    "kind": "ForbiddenReply",
                    "reason": str(exc),
                }, 422)

        tmp_files: List[Path] = []
        try:
            if voice_json_inline and not voice_arg:
                tf = Path(tempfile.mktemp(suffix=".json", prefix="cs_voice_"))
                tf.write_text(
                    voice_json_inline if isinstance(voice_json_inline, str) else json.dumps(voice_json_inline),
                    encoding="utf-8",
                )
                tmp_files.append(tf)
                voice_arg = str(tf)
            elif isinstance(voice_arg, str) and voice_arg and not Path(voice_arg).expanduser().exists():
                try:
                    from christman_voice_sdk.integration.voice_capture_client import PROFILES_DIR
                    cand = PROFILES_DIR / f"{voice_arg}.json"
                    if cand.is_file():
                        voice_arg = str(cand)
                except Exception:
                    pass

            lex_obj: Optional[Lexicon] = None
            if lexicon_inline and isinstance(lexicon_inline, str) and lexicon_inline.strip():
                lf = Path(tempfile.mktemp(suffix=".txt", prefix="cs_lex_"))
                lf.write_text(lexicon_inline, encoding="utf-8")
                tmp_files.append(lf)
                lex_obj = load_lexicon(lf)

            kwargs: Dict[str, Any] = {}
            for ctl in FRAME_CONTROLS:
                if ctl in data and data[ctl] is not None and data[ctl] != "":
                    val = data[ctl]
                    kwargs[ctl] = val if ctl in ("resonance", "affect") else float(val)
            if data.get("rate") is not None and data.get("rate") != "":
                kwargs["rate"] = float(data["rate"])
            if data.get("f0") is not None and data.get("f0") != "":
                kwargs["f0"] = float(data["f0"])
            if sample_rate is not None and sample_rate != "":
                kwargs["sample_rate"] = int(sample_rate)

            phones_seq = None
            if phones_raw is not None:
                phones_seq = [(str(p[0]), float(p[1])) for p in phones_raw]

            if play_aloud:
                res = speak(
                    text if phones_seq is None else None,
                    voice=voice_arg,
                    phones=phones_seq,
                    lexicon=lex_obj,
                    lexicon_path=lexicon_path if lex_obj is None else None,
                    **kwargs,
                )
            else:
                res = synthesize(
                    text if phones_seq is None else None,
                    voice=voice_arg,
                    phones=phones_seq,
                    lexicon=lex_obj,
                    lexicon_path=lexicon_path if lex_obj is None else None,
                    **kwargs,
                )

            fname = f"synth_{int(datetime.now().timestamp() * 1000)}.wav"
            out_path = _audio_cache_dir() / fname
            res.write_wav(out_path)

            return _json_resp({
                "status": "ok",
                "available": True,
                "audio_url": f"/audio/{fname}",
                "duration": res.duration,
                "sample_rate": res.sample_rate,
                "samples_count": len(res.samples),
                "voice": res.voice,
                "renderer": res.renderer,
                "voice_fingerprint": res.voice_fingerprint,
                "rate": res.rate,
                "f0": list(res.f0) if isinstance(res.f0, tuple) else res.f0,
                "phones": [{"id": p.id, "start": round(p.start, 4), "dur": round(p.dur, 4), "word": p.word} for p in res.phones],
                "applied": res.applied,
                "selection": res.selection,
            }, 200)
        except (NoVoice, VoiceProfileError, FrameVoiceUnavailable, LexiconUnavailable, LexiconFormatError, DSPUnavailable, SynthUnavailable) as exc:
            return _json_resp({
                "available": False,
                "status": "unavailable",
                "kind": type(exc).__name__,
                "reason": str(exc),
            }, 503)
        except (WordNotInLexicon, PhoneNotMapped, NothingToSay, UnknownPhone, SynthRefused, StitchRefused, ValueError) as exc:
            body_out: Dict[str, Any] = {
                "available": False,
                "status": "refused",
                "kind": type(exc).__name__,
                "reason": str(exc),
            }
            if hasattr(exc, "words"):
                body_out["words"] = list(exc.words)
            return _json_resp(body_out, 422)
        except PlaybackUnavailable as exc:
            return _json_resp({
                "available": False,
                "status": "playback_failed",
                "kind": "PlaybackUnavailable",
                "reason": str(exc),
            }, 503)
        finally:
            for tf in tmp_files:
                tf.unlink(missing_ok=True)

    def _handle_codecwar_table(self, data: Dict[str, Any]) -> AppResponse:
        text = data.get("text")
        phones_raw = data.get("phones")
        f0 = float(data.get("f0", 148.0))
        rate = float(data.get("rate", 1.0))
        sample_rate = int(data.get("sample_rate", CODECWAR_SAMPLE_RATE))
        disc_lex = discover_align_lexicon()
        lexicon_path = data.get("lexicon_path") or os.environ.get(LEXICON_ENV) or (str(disc_lex) if disc_lex else None)
        lexicon_inline = data.get("lexicon_text")

        tmp_files: List[Path] = []
        try:
            lex_obj: Optional[Lexicon] = None
            if lexicon_inline and isinstance(lexicon_inline, str) and lexicon_inline.strip():
                lf = Path(tempfile.mktemp(suffix=".txt", prefix="cs_lex_"))
                lf.write_text(lexicon_inline, encoding="utf-8")
                tmp_files.append(lf)
                lex_obj = load_lexicon(lf)

            phones_seq = None
            if phones_raw is not None:
                phones_seq = [(str(p[0]), float(p[1])) for p in phones_raw]

            res = render_codecwar_table(
                text if phones_seq is None else None,
                phones=phones_seq,
                f0=f0,
                rate=rate,
                sample_rate=sample_rate,
                lexicon=lex_obj,
                lexicon_path=lexicon_path if lex_obj is None else None,
            )
            fname = f"cw_{int(datetime.now().timestamp() * 1000)}.wav"
            out_path = _audio_cache_dir() / fname
            res.write_wav(out_path)

            dsp = get_dsp()
            rms_val = dsp.rms(res.samples)
            zcr_val = dsp.zcr(res.samples)
            yin_val = dsp.yin(res.samples[:min(len(res.samples), 2048)], sample_rate) if len(res.samples) >= 64 else 0.0

            return _json_resp({
                "status": "ok",
                "audio_url": f"/audio/{fname}",
                "duration": res.duration,
                "sample_rate": res.sample_rate,
                "samples_count": len(res.samples),
                "renderer": res.renderer,
                "f0": res.f0,
                "rate": res.rate,
                "measurements": {
                    "rms": round(rms_val, 5),
                    "zcr": round(zcr_val, 5),
                    "yin_f0_hz": round(yin_val, 2),
                },
                "phones": [{"id": p.id, "start": round(p.start, 4), "dur": round(p.dur, 4), "word": p.word} for p in res.phones],
            }, 200)
        except Exception as exc:
            return _json_resp({
                "available": False,
                "status": "refused",
                "kind": type(exc).__name__,
                "reason": str(exc),
            }, 422)
        finally:
            for tf in tmp_files:
                tf.unlink(missing_ok=True)

    def _handle_hear(self, data: Dict[str, Any]) -> AppResponse:
        from christman_voice_sdk.audio.corti_canal import analyze_grunt_physics
        from christman_voice_sdk.audio.hearing import Hearing
        from christman_voice_sdk.utils.grounder import Grounder
        from christman_voice_sdk.utils.presence_guide import PresenceGuide

        text = data.get("text")
        previous_turn = data.get("previous_turn") or None
        between_beings = bool(data.get("between_beings", False))
        public_setting = bool(data.get("public_setting", False))
        acoustic_key = data.get("acoustic_key") or None
        nonverbal = data.get("nonverbal") or None
        corti_event = data.get("corti_event") or None
        porch_result = None

        if isinstance(text, str) and text.strip():
            porch_layer = reconstruct_dialect(text)
            porch_result = porch_layer.to_dict()

        alphavox_route = None
        if isinstance(corti_event, dict) and not acoustic_key and not (text and str(text).strip()):
            route_obj = analyze_grunt_physics(corti_event, seat=self.speaker_seat)
            alphavox_route = route_obj.to_dict()
            if data.get("auto_route_grunt"):
                acoustic_key = route_obj.key

        paged: List[Dict[str, Any]] = []

        def crisis_pager(safety_result, context) -> bool:
            paged.append({
                "status": safety_result.status,
                "reason": safety_result.reason,
                "said": safety_result.said,
                "flags": list(safety_result.flags),
                "silent_alarm": safety_result.silent_alarm,
                "voice_aloud": safety_result.voice_aloud,
            })
            return True

        hearing = Hearing(crisis_callback=crisis_pager)
        turn = hearing.hear(
            text=text,
            corti_event=corti_event,
            nonverbal=nonverbal,
            acoustic_key=acoustic_key,
            previous_turn=previous_turn,
            between_beings=between_beings,
            public_setting=public_setting if public_setting else None,
        )

        presence = None
        if isinstance(text, str) and text.strip():
            try:
                presence = PresenceGuide().assess(text).to_dict()
            except Exception as exc:
                presence = {"error": str(exc)}

        grounding = None
        if turn.decision and turn.decision.delivery.get("needs_grounding"):
            try:
                grounding = Grounder().select_script(stress_score=0.75)
            except Exception:
                grounding = None

        return _json_resp({
            "status": "ok",
            "turn": turn.to_dict(),
            "paged_human": len(paged) > 0,
            "pager_events": paged,
            "presence_guide": presence,
            "grounding_script": grounding,
            "porch": porch_result,
            "alphavox_route": alphavox_route,
        }, 200)

    def _handle_measure_wav(self, body: bytes, content_type: str) -> AppResponse:
        from christman_voice_sdk.audio.corti_canal import freq_to_place, hz_to_note, voice_quality_dsp
        from christman_voice_sdk.integration.voice_capture_client import extract_frequency_signature
        from christman_voice_sdk.tone import voice_measure as vm

        tmp_wav: Optional[Path] = None
        try:
            if "multipart/form-data" in content_type:
                _, files = _parse_multipart(body, content_type)
                if "file" not in files:
                    return _json_resp({"error": "missing file upload"}, 400)
                tmp_wav = Path(tempfile.mktemp(suffix=".wav", prefix="cs_meas_"))
                tmp_wav.write_bytes(files["file"])
                wav_path = tmp_wav
            else:
                data = json.loads(body.decode("utf-8") or "{}") if body else {}
                raw_path = data.get("path", "")
                if not raw_path or not Path(raw_path).expanduser().is_file():
                    return _json_resp({"error": "provide a WAV file upload or a valid local path"}, 400)
                wav_path = Path(raw_path).expanduser()

            y, sr = vm.read_wav(str(wav_path))
            dsp = get_dsp()
            rms_val = dsp.rms(y) if dsp.available else None
            zcr_val = dsp.zcr(y) if dsp.available else None
            track = vm.pitch_contour(y, sr).summary() if dsp.available else {"available": False, "reason": dsp.error}
            sig = extract_frequency_signature(y, sr)

            try:
                qual = vm.voice_quality(y, sr).to_dict()
            except Exception as exc:
                qual = {"available": False, "reason": str(exc)}

            med_f0 = track.get("f0_median_hz") if isinstance(track, dict) else None
            dsp_qual = voice_quality_dsp(y, sr, f0=med_f0)
            greenwood_place = round(freq_to_place(med_f0), 4) if med_f0 else None
            note_info = hz_to_note(med_f0)

            try:
                f_meas = measure_formants(y, sr)
                formants_dict = {
                    "f1_hz": f_meas.f1_hz,
                    "f2_hz": f_meas.f2_hz,
                    "f3_hz": f_meas.f3_hz,
                    "b1_hz": f_meas.b1_hz,
                    "b2_hz": f_meas.b2_hz,
                    "b3_hz": f_meas.b3_hz,
                    "voiced_frames": f_meas.voiced_frames,
                    "total_frames": f_meas.total_frames,
                }
            except Exception as exc:
                formants_dict = {"available": False, "reason": str(exc)}

            return _json_resp({
                "status": "ok",
                "sample_rate": sr,
                "samples": int(len(y)),
                "duration_s": round(len(y) / sr, 4) if sr > 0 else 0,
                "rms": rms_val,
                "zcr": zcr_val,
                "pitch_contour": track,
                "frequency_signature": sig,
                "voice_quality": qual,
                "dsp_voice_quality": dsp_qual,
                "greenwood_place": greenwood_place,
                "musical_note": note_info,
                "formants": formants_dict,
            }, 200)
        except Exception as exc:
            return _json_resp({
                "status": "error",
                "kind": type(exc).__name__,
                "reason": str(exc),
            }, 422)
        finally:
            if tmp_wav is not None:
                tmp_wav.unlink(missing_ok=True)

    def _handle_recognize_wav(self, body: bytes, content_type: str) -> AppResponse:
        from christman_voice_sdk.audio.speech_recognition_engine import EarBackend, SpeechRecognitionEngine
        from christman_voice_sdk.audio.word_ear_receiver import WordEarReceiver, result_payload

        disc_vosk = discover_vosk_model()
        model_dir = os.environ.get("VOSK_MODEL_PATH") or (str(disc_vosk) if disc_vosk else None)
        raw_bytes: Optional[bytes] = None
        if "multipart/form-data" in content_type:
            fields, files = _parse_multipart(body, content_type)
            model_dir = fields.get("model_path") or model_dir
            raw_bytes = files.get("file")
        else:
            data = json.loads(body.decode("utf-8") or "{}") if body else {}
            model_dir = data.get("model_path") or model_dir
            p = data.get("path")
            if p and Path(p).expanduser().is_file():
                raw_bytes = Path(p).expanduser().read_bytes()

        if not raw_bytes:
            return _json_resp({"error": "provide a WAV file or local path"}, 400)

        backend = EarBackend(model_dir) if model_dir else EarBackend()
        engine = SpeechRecognitionEngine(backend=backend)
        receiver = WordEarReceiver(engine=engine)
        res = receiver.take_wav(raw_bytes)
        payload = result_payload(res)
        if res.status.value == "ok":
            dressed = dress_filament({"ok": True, "text": payload.get("text", ""), "words": payload.get("words", [])})
            porch = reconstruct_dialect(dressed.get("text", "")).to_dict()
            payload["filament"] = dressed
            payload["porch"] = porch
        code = 200 if res.status.value == "ok" else 503
        return _json_resp(payload, code)


    def _handle_vcc_get(self) -> AppResponse:
        wavs = [str(p) for p in bundled_alphavox_wavs()]
        voices_dir = Path.home() / ".christman_ai" / "voices"
        frame_voices = sorted(str(p) for p in voices_dir.glob("*") if (p / "voice.json").is_file()) if voices_dir.is_dir() else []
        return _json_resp({
            "status": "ok",
            "controls": list(VOICE_CREATION_CONTROLS),
            "presets": VOICE_CREATION_PRESETS,
            "script_samples": list(BOOTH_SCRIPT_SAMPLES),
            "bundled_alphavox_wavs": wavs,
            "frame_voices": frame_voices,
            "plaque_url": "/static/plaque.jpg",
        }, 200)

    def _handle_vcc_post(self, data: Dict[str, Any]) -> AppResponse:
        action = str(data.get("action") or "resynthesize_frames")
        cache = _audio_cache_dir()
        default_wavs = bundled_alphavox_wavs()
        try:
            if action == "resynthesize_frames":
                wav_in = data.get("wav_path") or (str(default_wavs[0]) if default_wavs else None)
                if not wav_in or not Path(wav_in).expanduser().is_file():
                    return _json_resp({"error": "no WAV file found to resynthesize"}, 400)
                rep = resynthesize_from_frames(wav_in, cache)
                rep["source_audio_url"] = f"/audio/{Path(rep['source_wav_path']).name}"
                rep["rendered_audio_url"] = f"/audio/{Path(rep['rendered_wav_path']).name}"
                return _json_resp({"status": "ok", **rep}, 200)

            if action == "measure_profile":
                name = str(data.get("name") or "AlphaVox_Measured")
                wav_list = data.get("wav_paths") or [str(p) for p in default_wavs]
                out_file = cache / f"profile_{name}.json"
                lex_obj = None
                if data.get("lexicon_text"):
                    tf = cache / "tmp_vcc_lex.txt"
                    tf.write_text(str(data["lexicon_text"]), encoding="utf-8")
                    lex_obj = load_lexicon(tf)
                res = measure_and_prove_voice_profile(
                    name=name,
                    wav_paths=wav_list,
                    out_path=out_file,
                    lexicon=lex_obj,
                    test_phrase=str(data.get("test_phrase") or "i need help please"),
                )
                return _json_resp({"status": "ok", **res}, 200)

            if action == "build_frame_voice":
                name = str(data.get("name") or "AlphaVox_FrameVoice")
                wav_list = data.get("wav_paths") or [str(p) for p in default_wavs]
                out_dir = cache / f"frame_voice_{name}"
                lex_obj = None
                if data.get("lexicon_text"):
                    tf = cache / "tmp_vcc_lex.txt"
                    tf.write_text(str(data["lexicon_text"]), encoding="utf-8")
                    lex_obj = load_lexicon(tf)
                elif not discover_align_lexicon():
                    tf = cache / "tmp_vcc_default_lex.txt"
                    tf.write_text(
                        "i i aI_S\nneed need n_B i_I d_E\nhelp help h_B E_I l_I p_E\nplease please p_B l_I i_I z_E\n"
                        "want want w_B A_I n_I t_E\nto to t_B u_E\ngo go g_B oU_E\nhome home h_B oU_I m_E\n"
                        "hello hello h_B @_I l_I oU_E\nchristman christman k_B r_I I_I s_I t_I m_I V_I n_E\n",
                        encoding="utf-8",
                    )
                    lex_obj = load_lexicon(tf)
                fv = build_frame_voice_center(
                    name=name,
                    wav_paths=wav_list,
                    out_dir=out_dir,
                    lexicon=lex_obj,
                )
                return _json_resp({
                    "status": "ok",
                    "name": fv.name,
                    "frame_voice_dir": str(fv.path),
                    "units_count": len(fv.bank.units),
                    "recordings_count": len(fv.bank.recordings),
                    "f0_median_hz": fv.measures["f0_median_hz"],
                    "syllable_rate": fv.measures["syllable_rate"],
                    "hnr_db": fv.measures["hnr_db"],
                }, 200)

            if action == "speak_vcc":
                preset = data.get("preset")
                overrides = data.get("controls") or {}
                controls = resolve_vcc_controls(preset=preset, overrides=overrides)
                voice_arg = data.get("voice")
                if not voice_arg:
                    default_fv = cache / "frame_voice_AlphaVox_FrameVoice"
                    if (default_fv / "voice.json").is_file():
                        voice_arg = str(default_fv)
                lex_obj = None
                if data.get("lexicon_text"):
                    tf = cache / "tmp_vcc_lex.txt"
                    tf.write_text(str(data["lexicon_text"]), encoding="utf-8")
                    lex_obj = load_lexicon(tf)
                elif not discover_align_lexicon() and not os.environ.get(LEXICON_ENV):
                    tf = cache / "tmp_vcc_default_lex.txt"
                    tf.write_text(
                        "i i aI_S\nneed need n_B i_I d_E\nhelp help h_B E_I l_I p_E\nplease please p_B l_I i_I z_E\n"
                        "want want w_B A_I n_I t_E\nto to t_B u_E\ngo go g_B oU_E\nhome home h_B oU_I m_E\n"
                        "hello hello h_B @_I l_I oU_E\nchristman christman k_B r_I I_I s_I t_I m_I V_I n_E\n",
                        encoding="utf-8",
                    )
                    lex_obj = load_lexicon(tf)
                text = str(data.get("text") or "i need help please")
                res = synthesize(text, voice=voice_arg, lexicon=lex_obj, **controls)
                fname = f"vcc_{int(datetime.now().timestamp() * 1000)}.wav"
                res.write_wav(cache / fname)
                return _json_resp({
                    "status": "ok",
                    "audio_url": f"/audio/{fname}",
                    "voice": res.voice,
                    "renderer": res.renderer,
                    "duration": res.duration,
                    "sample_rate": res.sample_rate,
                    "controls_resolved": controls,
                    "applied": res.applied,
                    "selection": res.selection,
                }, 200)

            if action == "package_voicepack":
                name = str(data.get("name") or "AlphaVox_Pack")
                wav_list = data.get("wav_paths") or [str(p) for p in default_wavs]
                out_vp = cache / f"{name}.voicepack"
                vp_res = package_voicepack_center(name=name, wav_paths=wav_list, out_path=out_vp)
                return _json_resp(vp_res, 200)

            return _json_resp({"error": f"unknown action {action!r}"}, 400)
        except Exception as exc:
            return _json_resp({
                "status": "refused",
                "kind": type(exc).__name__,
                "reason": str(exc),
            }, 422)

    def _handle_frames_analyze(self, data: Dict[str, Any]) -> AppResponse:
        """Analyze a 22,050 Hz waveform into 4.99 ms frames (F0, 96-bin mel envelope, 8-band aperiodicity)."""
        phones_raw = data.get("phones") or [("HH", 1.0), ("AH", 1.0), ("L", 1.0), ("OW", 1.0)]
        f0 = float(data.get("f0", 142.0))
        rate = float(data.get("rate", 1.0))
        rendered = render_codecwar_table(
            None,
            phones=[(str(p[0]), float(p[1])) for p in phones_raw],
            f0=f0,
            rate=rate,
            sample_rate=CHRISTMAN_SOUND_SAMPLE_RATE,
        )
        track = analyze_frames(rendered.samples, sr=CHRISTMAN_SOUND_SAMPLE_RATE)
        voiced_mask = track.f0 > 0
        voiced_count = int(voiced_mask.sum())
        med_f0 = round(float( sorted(track.f0[voiced_mask])[voiced_count // 2] ), 2) if voiced_count else None
        mean_ap = [round(float(v), 4) for v in track.ap[voiced_mask].mean(axis=0)] if voiced_count else [1.0] * 8
        return _json_resp({
            "status": "ok",
            "sr": track.sr,
            "hop": track.hop,
            "n_frames": track.n,
            "voiced_frames": voiced_count,
            "f0_median_hz": med_f0,
            "mean_voiced_aperiodicity_8band": mean_ap,
            "env_bins": int(track.env_db.shape[1]),
            "f0_preview": [round(float(v), 1) for v in track.f0[:40]],
        }, 200)

    def _handle_dispersion(self, data: Dict[str, Any]) -> AppResponse:
        scene_id = str(data.get("scene_id") or "kitchen")
        if scene_id not in SCENES:
            return _json_resp({"error": f"unknown scene {scene_id!r}", "available_scenes": list(SCENES)}, 400)
        report = discern_canals(scene_id=scene_id, sr=CODECWAR_SAMPLE_RATE)
        scene = SCENES[scene_id]
        pcm_list = [render_canal_source(s, sr=CODECWAR_SAMPLE_RATE) for s in scene.sources]
        for idx, (s, pcm) in enumerate(zip(scene.sources, pcm_list)):
            fname = f"canal_{scene_id}_{s.id}_{int(datetime.now().timestamp() * 1000)}.wav"
            write_wav(_audio_cache_dir() / fname, pcm.tolist(), CODECWAR_SAMPLE_RATE)
            report["canals"][idx]["audio_url"] = f"/audio/{fname}"
        fused = fuse_pcm(pcm_list)
        fused_name = f"fused_{scene_id}_{int(datetime.now().timestamp() * 1000)}.wav"
        write_wav(_audio_cache_dir() / fused_name, fused.tolist(), CODECWAR_SAMPLE_RATE)
        report["fused_audio_url"] = f"/audio/{fused_name}"
        return _json_resp({"status": "ok", **report}, 200)

    def _handle_porch(self, data: Dict[str, Any]) -> AppResponse:
        action = str(data.get("action") or "reconstruct")
        if action == "clear_carbon":
            save_carbon([])
            return _json_resp({"status": "ok", "carbon_pile": []}, 200)
        if action == "add_board":
            tok = str(data.get("token") or "").strip()
            region = str(data.get("region") or "")
            board_kind = str(data.get("board") or "family").lower()
            row = add_residual(tok, region, "manual") if board_kind == "residual" else add_family(tok, region)
            return _json_resp({"status": "ok", "row": row, "board": board_snapshot()}, 200)

        raw_text = str(data.get("text") or "")
        extra = data.get("extra_lexicon") or []
        layer = reconstruct_dialect(raw_text, extra_lexicon=extra)
        words = data.get("words") or []
        dressed = dress_filament({"ok": True, "final": True, "text": raw_text, "words": words})
        return _json_resp({
            "status": "ok",
            "porch": layer.to_dict(),
            "filament": dressed,
            "board": board_snapshot(),
            "carbon_pile": load_carbon(),
        }, 200)

    def _handle_corti_canal(self, data: Dict[str, Any]) -> AppResponse:
        from christman_voice_sdk.audio.corti_canal import (
            analyze_grunt_physics,
            detect_vocal_stimming,
            freq_to_place,
            hz_to_note,
            place_to_freq,
        )

        hz = float(data.get("hz", 220.0))
        place = freq_to_place(hz)
        inv_hz = place_to_freq(place)
        note = hz_to_note(hz)

        if data.get("observe_lock"):
            lock = data["observe_lock"]
            self.speaker_seat.observe(
                lock.get("f0"),
                lock.get("peak_rms"),
                lock.get("f1"),
                lock.get("f2"),
            )

        if data.get("record_canal_frame"):
            cf = data["record_canal_frame"]
            self.canal_recorder.record(
                t_ms=float(cf.get("t_ms", 0.0)),
                rms=float(cf.get("rms", 0.02)),
                zcr=float(cf.get("zcr", 0.05)),
                f0=cf.get("f0"),
                f1=cf.get("f1"),
                f2=cf.get("f2"),
            )

        event = data.get("event") or {"durationMs": 220.0, "attackMs": 45.0, "peakRms": 0.14, "f0": hz}
        recent_events = data.get("recent_events") or []
        route = analyze_grunt_physics(event, seat=self.speaker_seat, recent_events=recent_events)
        stimming = detect_vocal_stimming(recent_events) if recent_events else False

        return _json_resp({
            "status": "ok",
            "greenwood": {
                "hz": round(hz, 2),
                "place": round(place, 4),
                "roundtrip_hz": round(inv_hz, 2),
                "note": note,
            },
            "speaker_seat": self.speaker_seat.baseline() or {"seated": False, "locks": len(self.speaker_seat.f0_locks), "min_locks_required": self.speaker_seat.MIN_LOCKS},
            "alphavox_route": route.to_dict(),
            "vocal_stimming_detected": stimming,
            "canal_buffer_frames": len(self.canal_recorder.frames),
        }, 200)

    def _handle_lucent(self, data: Dict[str, Any]) -> AppResponse:
        from christman_voice_sdk.audio.body_signals import analyze_body_window
        from christman_voice_sdk.audio.lucent_reader import (
            CORNEA_LAYERS,
            CORNEA_PRESETS,
            CorneaParams,
            evaluate_optics,
            gaze_from_blendshapes,
            local_stamp,
            reading_to_body_signal_frame,
            trace_meridian_rays,
        )

        preset_name = str(data.get("preset") or "emmetropic")
        params = CORNEA_PRESETS.get(preset_name, CORNEA_PRESETS["emmetropic"])
        if isinstance(data.get("params"), dict):
            p_raw = data["params"]
            params = CorneaParams(
                r_anterior_mm=float(p_raw.get("r_anterior_mm", params.r_anterior_mm)),
                r_posterior_mm=float(p_raw.get("r_posterior_mm", params.r_posterior_mm)),
                cct_um=float(p_raw.get("cct_um", params.cct_um)),
                asphericity_q=float(p_raw.get("asphericity_q", params.asphericity_q)),
                astigmatism_d=float(p_raw.get("astigmatism_d", params.astigmatism_d)),
                axis_deg=float(p_raw.get("axis_deg", params.axis_deg)),
                axial_length_mm=float(p_raw.get("axial_length_mm", params.axial_length_mm)),
                lens_power_d=float(p_raw.get("lens_power_d", params.lens_power_d)),
                endothelial_density=float(p_raw.get("endothelial_density", params.endothelial_density)),
            )

        optics = evaluate_optics(params)
        rays = trace_meridian_rays(params, n_rays=int(data.get("n_rays", 7)))
        gaze = gaze_from_blendshapes(data.get("blendshapes") or {})

        body_window_out = None
        if isinstance(data.get("body_frames"), list):
            b_frames = [
                reading_to_body_signal_frame(
                    t_ms=float(f["t_ms"]),
                    breath_freq=f.get("breath_freq"),
                    breath_amp=f.get("breath_amp"),
                    eye_jitter=f.get("eye_jitter"),
                    stare_lock=f.get("stare_lock"),
                    tremor=f.get("tremor"),
                )
                for f in data["body_frames"]
            ]
            body_window_out = analyze_body_window(b_frames).to_dict()

        return _json_resp({
            "status": "ok",
            "preset": preset_name,
            "available_presets": list(CORNEA_PRESETS.keys()),
            "layers": list(CORNEA_LAYERS),
            "optics": optics.to_dict(),
            "rays": rays,
            "gaze": gaze,
            "body_window": body_window_out,
            "stamp": local_stamp(),
        }, 200)

    def _handle_being(self, data: Dict[str, Any]) -> AppResponse:
        if data.get("perturbation"):
            p = data["perturbation"]
            pert = place_perturbation(
                label=str(p.get("label") or "stimulus"),
                magnitude=float(p.get("magnitude", 0.3)),
                layer=str(p.get("layer") or "membrane"),
                angle=float(p.get("angle", 0.0)),
            )
            attach_perturbations(self.stasis_field, [pert])
        if data.get("shaker"):
            attach_perturbations(self.stasis_field, [shaker_pulse(float(self.stasis_field.ticks) * 0.05)])

        steps = int(data.get("ticks", 1))
        if steps > 0:
            tick_stasis(self.stasis_field, steps=steps)

        unit_val = float(data.get("unit_value", 220.0))
        unit_from = str(data.get("unit_from", "hz"))
        units_out = convert_unit(unit_val, from_unit=unit_from)

        stream_text = str(data.get("stream_text") or "Christman Sound — Truth. Dignity. Protection.")
        stream_enc = str(data.get("stream_encoding") or "utf-8")
        stream_out = inspect_stream(stream_text, encoding=stream_enc)

        return _json_resp({
            "status": "ok",
            "stasis": self.stasis_field.to_dict(),
            "units": units_out,
            "stream": stream_out,
            "encodings": [{"id": k, "label": v} for k, v in ENCODINGS],
        }, 200)

    def _handle_whole_house(self, data: Dict[str, Any]) -> AppResponse:
        from christman_voice_sdk.audio.whole_house import (
            classify_smell,
            describe_house,
            frame_from_sense,
            merge_house,
            whole_environment,
        )

        smell_raw = data.get("smell", {"channels": [0.62, 0.48, 0.18, 0.35], "temp_c": 22.5, "humidity_pct": 44.0})
        skin_raw = data.get("skin", {"pads": [0.28, 0.32, 0.25, 0.29], "temp_c": 34.2, "imu_delta_g": 0.04})
        smell_res = classify_smell(smell_raw)
        skin_res = frame_from_sense(skin_raw)

        if data.get("reclaim_scene_id"):
            self.memory_chamber.reclaim_scene(str(data["reclaim_scene_id"]))
        if data.get("night_level"):
            night_report = self.memory_chamber.apply_night(int(data["night_level"]))
        else:
            night_report = None
        if data.get("run_rem"):
            rem_scene = self.memory_chamber.apply_rem()
        else:
            rem_scene = None

        query = data.get("memory_query") or {"who": "Everett", "where": "porch coffee"}
        completion = self.memory_chamber.complete(query)

        house_state = merge_house(
            ear=data.get("ear") or {"speaker": "Everett", "text": query.get("what", "")},
            cornea=data.get("cornea"),
            smell=smell_res,
            skin=skin_res,
            porch=data.get("porch"),
        )
        return _json_resp({
            "status": "ok",
            "smell": smell_res,
            "skin": skin_res,
            "house_environment": whole_environment(house_state),
            "house_description": describe_house(house_state),
            "memory_completion": completion.to_dict(),
            "night_report": night_report,
            "rem_dream": rem_scene.to_dict() if rem_scene else None,
            "scenes": [s.to_dict() for s in self.memory_chamber.scenes],
        }, 200)

    def test_client(self) -> "AppTestClient":
        return AppTestClient(self)

    def run(self, host: str = DEFAULT_APP_HOST, port: int = DEFAULT_APP_PORT) -> None:
        app_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                resp = app_ref.dispatch("GET", self.path, headers=dict(self.headers))
                self.send_response(resp.status_code)
                self.send_header("Content-Type", resp.content_type)
                self.send_header("Content-Length", str(len(resp.data)))
                self.end_headers()
                self.wfile.write(resp.data)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b""
                resp = app_ref.dispatch("POST", self.path, body=body, headers=dict(self.headers))
                self.send_response(resp.status_code)
                self.send_header("Content-Type", resp.content_type)
                self.send_header("Content-Length", str(len(resp.data)))
                self.end_headers()
                self.wfile.write(resp.data)

            def log_message(self, format: str, *args: Any) -> None:
                logger.info(format, *args)

        server = ThreadingHTTPServer((host, port), Handler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


class AppTestClient:
    def __init__(self, app: ChristmanSoundApp) -> None:
        self.app = app

    def get(self, path: str) -> AppResponse:
        return self.app.dispatch("GET", path)

    def post(self, path: str, json_body: Any = None, **kwargs: Any) -> AppResponse:
        if "json" in kwargs:
            json_body = kwargs["json"]
        raw = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
        return self.app.dispatch("POST", path, body=raw, headers={"Content-Type": "application/json"})


def create_app() -> ChristmanSoundApp:
    return ChristmanSoundApp()


def install_desktop_bundle(target_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Create a native desktop application bundle:
      - macOS: `Christman Sound.app` in ~/Applications (or target_dir)
      - Linux: `christman-sound.desktop` in ~/.local/share/applications (or target_dir)
    Also writes a portable copy under <repo>/dist/ when writable.
    """
    python_exe = sys.executable
    icon_png = _STATIC_DIR / "icons" / "icon-512.png"
    created: Dict[str, str] = {}

    app_roots: List[Path] = []
    if target_dir is not None:
        app_roots.append(Path(target_dir))
    else:
        if sys.platform == "darwin":
            home_apps = Path.home() / "Applications"
            home_apps.mkdir(parents=True, exist_ok=True)
            app_roots.append(home_apps)
        dist_dir = _REPO_ROOT / "dist"
        try:
            dist_dir.mkdir(parents=True, exist_ok=True)
            app_roots.append(dist_dir)
        except OSError:
            pass

    for root in app_roots:
        bundle = root / "Christman Sound.app"
        contents = bundle / "Contents"
        macos_dir = contents / "MacOS"
        res_dir = contents / "Resources"
        macos_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)

        plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>Christman Sound</string>
  <key>CFBundleDisplayName</key>
  <string>Christman Sound</string>
  <key>CFBundleIdentifier</key>
  <string>org.christmanaiproject.christmansound</string>
  <key>CFBundleVersion</key>
  <string>1.1.0</string>
  <key>CFBundleShortVersionString</key>
  <string>1.1.0</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleExecutable</key>
  <string>ChristmanSound</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon.png</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
"""
        (contents / "Info.plist").write_text(plist, encoding="utf-8")
        if icon_png.is_file():
            (res_dir / "AppIcon.png").write_bytes(icon_png.read_bytes())

        launcher = macos_dir / "ChristmanSound"
        launcher_script = f"""#!/bin/sh
set -e
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
BUNDLE_PARENT="$(cd "$(dirname "$0")/../../.." && pwd)"
if [ -f "$BUNDLE_PARENT/christman_dsp.c" ]; then
  REPO_DIR="$BUNDLE_PARENT"
elif [ -f "$BUNDLE_PARENT/../christman_dsp.c" ]; then
  REPO_DIR="$(cd "$BUNDLE_PARENT/.." && pwd)"
else
  REPO_DIR="{_REPO_ROOT}"
fi
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
if [ -z "$VOSK_MODEL_PATH" ] && [ -d "$HOME/Downloads/vosk-model-en-us-0.22" ]; then
  export VOSK_MODEL_PATH="$HOME/Downloads/vosk-model-en-us-0.22"
fi
if [ -z "$CHRISTMAN_LEXICON_PATH" ] && [ -f "$HOME/Downloads/vosk-model-en-us-0.22/graph/phones/align_lexicon.txt" ]; then
  export CHRISTMAN_LEXICON_PATH="$HOME/Downloads/vosk-model-en-us-0.22/graph/phones/align_lexicon.txt"
fi
PY_BIN="{python_exe}"
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="$(command -v python3)"
fi
if [ ! -f "$REPO_DIR/christman_sound/christman_dsp.so" ] && [ ! -f "$REPO_DIR/christman_sound/christman_dsp.dylib" ]; then
  "$PY_BIN" "$REPO_DIR/scripts/build_dsp.py"
fi
if [ ! -f "$REPO_DIR/christman_sound/ear/libear.dylib" ] && [ ! -f "$REPO_DIR/christman_sound/ear/libear.so" ]; then
  "$PY_BIN" -m christman_sound.ear.build
fi
exec "$PY_BIN" -m christman_sound.app --open-browser
"""
        launcher.write_text(launcher_script, encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        created["macos_app"] = str(bundle)

    if sys.platform.startswith("linux"):
        desktop_dir = Path(target_dir) if target_dir else (Path.home() / ".local" / "share" / "applications")
        try:
            desktop_dir.mkdir(parents=True, exist_ok=True)
            desktop_file = desktop_dir / "christman-sound.desktop"
            desktop_content = f"""[Desktop Entry]
Type=Application
Name=Christman Sound
Comment=Sovereign Audio, Voice, and Speech Studio
Exec="{python_exe}" -m christman_sound.app --open-browser
Icon={icon_png}
Terminal=false
Categories=AudioVideo;Audio;
"""
            desktop_file.write_text(desktop_content, encoding="utf-8")
            created["linux_desktop"] = str(desktop_file)
        except OSError:
            pass

    return created


def main_install(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build C engines and install the Christman Sound desktop bundle.")
    ap.add_argument("--target-dir", default=None, help="Optional folder for the desktop bundle")
    args = ap.parse_args(argv)

    app = create_app()
    build_res = app._handle_build_engines().get_json()
    print(json.dumps(build_res, indent=2))
    target = Path(args.target_dir).expanduser() if args.target_dir else None
    installed = install_desktop_bundle(target)
    print("Installed desktop bundle:", json.dumps(installed, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the local Christman Sound Studio application (127.0.0.1).")
    ap.add_argument("--host", default=DEFAULT_APP_HOST, help="Loopback bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=DEFAULT_APP_PORT, help=f"Port (default {DEFAULT_APP_PORT})")
    ap.add_argument("--open-browser", action="store_true", help="Open the studio in the default browser")
    ap.add_argument("--install-desktop", action="store_true", help="Install the desktop app bundle and exit")
    args = ap.parse_args(argv)

    if args.install_desktop:
        return main_install([])

    disc_vosk = discover_vosk_model()
    if disc_vosk and not os.environ.get("VOSK_MODEL_PATH"):
        os.environ["VOSK_MODEL_PATH"] = str(disc_vosk)
    disc_lex = discover_align_lexicon()
    if disc_lex and not os.environ.get(LEXICON_ENV):
        os.environ[LEXICON_ENV] = str(disc_lex)

    app = create_app()
    url = f"http://{args.host}:{args.port}"
    print(f"Christman Sound Studio listening on {url} (100% offline)")
    if args.open_browser:
        webbrowser.open(url)
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
