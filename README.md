# christman_sound_clean

The sound package of The Christman AI Project, Christman Sound, is the `Christman-Sound` folder of this repository. Its own README (`Christman-Sound/README.md`) says what it holds and how to use it.

## Install it

WHAT CHANGED AND WHY (2026-11-05): Everett asked that the sound package install from a download with one pip install. pip now builds the package's C library itself; nothing is run after it.

Download this repository (a clone, or its archive unpacked), then point pip at the package's folder:

```bash
pip install ./Christman-Sound             # an installed copy
pip install -e ./Christman-Sound          # editable: edits in the folder take effect at once
```

It needs Python 3.10 or later and a C compiler (Xcode Command Line Tools on a Mac, gcc or clang on Linux).

Not in the package, downloaded separately: the Vosk model folder `vosk-model-en-us-0.22`, which the ear hears through. Set `VOSK_MODEL_PATH` to it. The studio (`christman-sound`) also finds it on its own in `~/Downloads/vosk-model-en-us-0.22` or `~/models/vosk-model-en-us-0.22`; the speech recognition engine and the word-ear receiver read `VOSK_MODEL_PATH` only.
