# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
What pip runs to build the C library when the package is installed.

WHAT CHANGED AND WHY (2026-11-05)
Everett: "I want the sound package to be an installable package. I want
people to be able to just download it and install it." Before, pip built
nothing in C: pyproject.toml asked for `python3 scripts/build_dsp.py --check`
by hand after installing, and a package installed without that step could
neither measure nor speak. Now one `pip install` builds the library:

  * pip install <folder or git URL>  (a wheel): christman_dsp.so is compiled
    into the wheel's christman_voice_sdk/engines/ and installed beside dsp.py.
  * pip install -e <folder>  (editable): it is compiled into the checkout's
    christman_voice_sdk/engines/, where scripts/build_dsp.py writes it.

pyproject.toml names these classes under [tool.setuptools.cmdclass]; nothing
is passed to setup(), so setup.py stays a bare shim. The compile line itself
is not here: build() in scripts/build_dsp.py is the one place it is written,
and this file calls it, then load_check() on what was built.

Why a step of its own and not an override of build_py: in an editable
install setuptools runs a replaced build_py inside a try that turns any
error into a warning, so a failed compile would have installed a package
that cannot speak, quietly. A step of its own runs plainly: if the compiler
is missing or the library does not load whole, pip install fails and says why.

This file is build tooling. It is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

from setuptools import Command
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel  # setuptools >= 70.1
from setuptools.command.build import build as _build

_SCRIPTS = Path(__file__).resolve().parent
#: The library's path inside the installed package, as dsp.py looks for it.
# The same name scripts/build_dsp.py chooses (LIB_NAME): .dll on Windows, .so elsewhere.
_RELATIVE_OUTPUT = Path("christman_voice_sdk") / "engines" / ("christman_dsp.dll" if sys.platform == "win32" else "christman_dsp.so")


def _build_dsp():
    """scripts/build_dsp.py, loaded by its path (scripts/ is not a package)."""
    path = _SCRIPTS / "build_dsp.py"
    spec = importlib.util.spec_from_file_location("christman_build_dsp", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class build_christman_dsp(Command):
    """Compile the C library (scripts/build_dsp.py) and prove it loads."""

    description = ("compile christman_dsp.c, christman_synth.c and christman_vocoder.c into "
                   "christman_voice_sdk/engines/christman_dsp.so with scripts/build_dsp.py")
    user_options: List = []

    #: Set to True by setuptools for an editable install (PEP 660).
    editable_mode: bool = False

    def initialize_options(self) -> None:
        self.build_lib = None

    def finalize_options(self) -> None:
        self.set_undefined_options("build", ("build_lib", "build_lib"))

    def _wheel_path(self) -> Path:
        return Path(self.build_lib) / _RELATIVE_OUTPUT

    def _checkout_path(self) -> Path:
        return _build_dsp().OUTPUT

    def run(self) -> None:
        build_dsp = _build_dsp()
        output = build_dsp.OUTPUT if self.editable_mode else self._wheel_path()
        output.parent.mkdir(parents=True, exist_ok=True)
        build_dsp.build(output)
        print(build_dsp.load_check(output))

    def get_source_files(self) -> List[str]:
        return ["christman_dsp.c", "christman_synth.c", "christman_vocoder.c", "scripts/build_dsp.py"]

    def get_outputs(self) -> List[str]:
        # Editable: the library is written into the checkout, not build_lib.
        return [] if self.editable_mode else [str(self._wheel_path())]

    def get_output_mapping(self) -> Dict[str, str]:
        # Editable: where the library would sit in a wheel -> where it is.
        return {str(self._wheel_path()): str(self._checkout_path())} if self.editable_mode else {}


class build(_build):
    """setuptools' build, with the C library as its last step."""

    sub_commands = _build.sub_commands + [("build_christman_dsp", None)]


class bdist_wheel(_bdist_wheel):
    """
    The wheel carries a compiled library, so it is tagged for this platform,
    not "any". The library is loaded by ctypes and is not a Python extension,
    so the tag names no Python version: py3-none-<platform>.
    """

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _python, _abi, platform = super().get_tag()
        return "py3", "none", platform
