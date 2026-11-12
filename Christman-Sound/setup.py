# WHAT CHANGED AND WHY (2026-10-28, Lane 6)
# This file used to carry its own dependency list. pyproject.toml carries one
# too, and setuptools reads pyproject.toml's [project] table, so the two lists
# drifted apart (this one had no Intel Mac markers for torch, and both carried
# librosa, which nothing imports and which broke the install on Intel Mac).
# Everything now lives in pyproject.toml. This shim stays so tools that still
# call setup.py work.
#
# WHAT CHANGED AND WHY (2026-11-05)
# pip install now builds the C library (Everett: "I want people to be able to
# just download it and install it"). It is still not done here: setup() takes
# no arguments. pyproject.toml's [tool.setuptools.cmdclass] adds the step
# (scripts/pip_build.py), and scripts/build_dsp.py compiles christman_dsp.c,
# christman_synth.c and christman_vocoder.c into
# christman_voice_sdk/engines/christman_dsp.so, the file christman_voice_sdk/engines/dsp.py loads.
#
# 2026-10-17: pyttsx3 was removed (Everett's ruling cuts every outside speech
# engine). pygame was removed on 2026-10-13.
from setuptools import setup

setup()
