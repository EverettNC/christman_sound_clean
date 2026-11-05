#!/bin/sh
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
# One-step installer for Christman Sound from a source folder (macOS & Linux).
# Run it with the Python you mean to use (an activated venv, for example):
#   1. Seats ~/Downloads/vosk-model-en-us-0.22 for this run if present
#   2. pip install -e . : builds christman_dsp.so (christman_dsp.c,
#      christman_synth.c, christman_vocoder.c) and installs the package with
#      its dependencies, vosk among them
#   3. Proves what pip built: the smoke test on the library, and vosk imports
#   4. Creates the native desktop launcher (Christman Sound.app / .desktop)
#
# 2026-11-05 (Everett): the ear is Vosk again. Step 3 used to compile
# christman_sound/ear/libear, the decoder a Claude Code session wrote in Vosk's
# place; that decoder is held, not installed, in held/claude-ear-2026-10-28.
#
# WHAT CHANGED AND WHY (2026-11-05)
# Everett asked that the package install from a download with one pip
# install. pip now builds the C library itself (scripts/pip_build.py), so
# this script no longer compiles first and installs after. It used to run
# build_dsp.py --check before pip (which needed numpy already installed),
# install vosk on its own, then pip install -e . with --no-deps and
# --no-build-isolation (which needed every dependency and a recent setuptools
# already present). Now it is the same single pip install a newcomer runs,
# then the smoke test on the library that install built (--check-only
# compiles nothing). Step 4 is christman-sound-install, which in a checkout
# rebuilds the library and smoke-tests it again before it writes the launcher,
# as it always has.

set -e
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

if [ -d "$HOME/Downloads/vosk-model-en-us-0.22" ]; then
  export VOSK_MODEL_PATH="${VOSK_MODEL_PATH:-$HOME/Downloads/vosk-model-en-us-0.22}"
  if [ -f "$HOME/Downloads/vosk-model-en-us-0.22/graph/phones/align_lexicon.txt" ]; then
    export CHRISTMAN_LEXICON_PATH="${CHRISTMAN_LEXICON_PATH:-$HOME/Downloads/vosk-model-en-us-0.22/graph/phones/align_lexicon.txt}"
  fi
  echo "==> Seated vosk-model-en-us-0.22 from $VOSK_MODEL_PATH"
fi

echo "==> Installing Christman Sound (pip builds the C DSP, synthesizer and frame vocoder)..."
python3 -m pip install -e .

echo "==> Checking the library pip built..."
python3 scripts/build_dsp.py --check-only

echo "==> Checking the ear's recognizer (vosk)..."
python3 -c "import vosk" || { echo "vosk did not import: the ear cannot hear" >&2; exit 1; }

echo "==> Installing Christman Sound desktop application bundle..."
python3 -m christman_sound.app --install-desktop

echo "==> Done! Launch 'Christman Sound.app' or run: christman-sound --open-browser"
