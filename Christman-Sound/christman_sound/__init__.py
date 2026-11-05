# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound — the package beings import, and where the voice lives.

The voice and its integrated acoustic organs live here, in plain Python and C:
  - dsp.py          the one loader for christman_dsp.so (built by
                    scripts/build_dsp.py from christman_dsp.c,
                    christman_synth.c and christman_vocoder.c)
  - synth.py        synthesize() and speak(): text -> phones -> sound
  - stitch.py       frame voices, spoken from a voice's own recordings
  - voice_creation_center.py  Voice Creation Center controls, presets, Frames Room A/B resynthesis, profile proof & bank builder
  - frames.py       4.99 ms frame analysis, 5-pass Viterbi forced phone
                    alignment, and native bank builder (from Codec War)
  - dispersion.py   4-canal Dispersion & War engine and Refused-Mix guard
  - being.py        Silico-Stasis equilibrium field, 2 Hz shaker,
                    psychoacoustic unit converter, and text stream inspector
  - lexicon.py      words -> phones (with auto-discovery for
                    vosk-model-en-us-0.22/graph/phones/align_lexicon.txt)
  - formants.py     formant measurement from recordings
  - voice_profile.py  a voice's measured profile
  - app.py          local installable PWA studio server & desktop app builder
  - logger.py       logging
  - ear/            offline speech-to-text, Filament mouth map, Carbon pile,
                    Family/Residual boards, 15-region dialect map, and Porch
                    TCAP-V-0020 mouth-cooperation

The package has no voice of its own: a being passes its own voice (a voice
profile file it measured with scripts/measure_voice.py, or a frame voice
folder) to synth.synthesize() and speak().

`christman_sound.core` is kept for the record. Everett's standing ruling: it
is called by nothing, is never wired into anything, and is not touched.

The modular SDK is `christman_voice_sdk`: hearing, Corti canal & Greenwood
cochlea (`corti_canal.py`), Lucent cornea optics & 5-part rhythm reader
(`lucent_reader.py`), Whole House OpenSmell / Skin / TCAP-V-0044 Five-Element
Memory Chamber (`whole_house.py`), recognition, tone, timbre, and the speaking
engine (`synthesis/voice_synthesis.py`).
"""
