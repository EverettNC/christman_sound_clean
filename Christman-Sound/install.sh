#!/bin/sh
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
# One-step offline installer for Christman Sound (macOS & Linux):
#   1. Auto-detects ~/Downloads/vosk-model-en-us-0.22 if present
#   2. Compiles christman_dsp.so (christman_dsp.c, christman_synth.c, christman_vocoder.c)
#   3. Installs the vosk library, the ear's recognizer (nothing to compile)
#   4. Installs the Python package in editable mode (--no-build-isolation --no-deps)
#   5. Creates the native desktop launcher (Christman Sound.app / .desktop)
#
# 2026-11-05 (Everett): the ear is Vosk again. Step 3 used to compile
# christman_sound/ear/libear, the decoder a Claude Code session wrote in Vosk's
# place; that decoder is held, not installed, in held/claude-ear-2026-10-28.

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

echo "==> Building Christman C DSP, Synthesizer, and Frame Vocoder..."
python3 scripts/build_dsp.py --check

echo "==> Installing the ear's recognizer (vosk)..."
python3 -m pip install vosk
python3 -c "import vosk" || { echo "vosk did not import: the ear cannot hear" >&2; exit 1; }

echo "==> Installing christman-sound CLI entry points..."
python3 -m pip install --no-build-isolation --no-deps -e .

echo "==> Installing Christman Sound desktop application bundle..."
python3 -m christman_sound.app --install-desktop

echo "==> Done! Launch 'Christman Sound.app' or run: christman-sound --open-browser"
