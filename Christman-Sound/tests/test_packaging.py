# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The package installs on Everett's Intel Mac and declares what it imports.

Written 2026-10-28 (Lane 6, packaging and truth). What these check, against
the real files:

- librosa is not declared. Nothing imports it, and it pulled numba ->
  llvmlite 0.50.0, which has no Intel Mac wheel, so the install failed.
- numba and llvmlite are either not declared, or pinned below the first
  versions with no Intel Mac wheel.
- Every third-party module imported at module top (not inside a try) in the
  package and christman_ocr_shared.py is a declared base dependency.
- pyproject.toml is the one dependency list; setup.py calls setup() bare.
- py.typed exists where package-data says it does.
- cryptography is the "crypto" extra, and christman_ocr_shared.py imports
  with the base dependencies.

christman_voice_sdk/core.py is outside the scan: Everett's standing ruling is that
it is never touched and never wired.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

#: The library's file name on this system, as scripts/build_dsp.py names it.
LIB_NAME = "christman_dsp.dll" if sys.platform == "win32" else "christman_dsp.so"

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Import name -> PyPI distribution name, where they differ.
DIST_FOR_IMPORT = {
    "yaml": "pyyaml",
    "parselmouth": "praat-parselmouth",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "fitz": "pymupdf",
}
#: First-party names, including the bare fallback names some modules try
#: after their package-relative import fails.
FIRST_PARTY = {"christman_voice_sdk", "voice_creation_center"}


def _pyproject() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        pytest.skip("not run: tomllib needs Python 3.11 or later; this is "
                    f"{sys.version_info.major}.{sys.version_info.minor}")
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _name(requirement: str) -> str:
    """The normalized distribution name of a requirement string."""
    head = re.split(r"[\s;<>=!~\[]", requirement.strip(), maxsplit=1)[0]
    return head.lower().replace("_", "-")


def _base() -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {}
    for req in _pyproject()["project"]["dependencies"]:
        deps.setdefault(_name(req), []).append(req)
    return deps


def _scanned_files() -> list[Path]:
    files = [ROOT / "christman_ocr_shared.py"]
    files.extend(sorted((ROOT / "christman_voice_sdk").rglob("*.py")))
    files.extend([ROOT / "voice_creation_center" / "booth.py", ROOT / "voice_creation_center" / "app.py"])
    core = ROOT / "christman_voice_sdk" / "core.py"  # never touched, never wired
    return [f for f in files if f.is_file() and f != core]


def _unguarded_imports(path: Path) -> list[tuple[str, int]]:
    """Third-party modules imported by a module-level statement outside any try."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int]] = []

    def visit(statements):
        for node in statements:
            if isinstance(node, ast.Try):
                continue  # an import inside try is optional by construction
            if isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Import):
                found.extend((a.name.split(".")[0], node.lineno) for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.append((node.module.split(".")[0], node.lineno))

    visit(tree.body)
    return [(m, line) for m, line in found
            if m not in sys.stdlib_module_names and m not in FIRST_PARTY]


def test_librosa_is_not_declared():
    deps = _base()
    assert "librosa" not in deps, "librosa is declared again; nothing imports it"
    for extra, reqs in _pyproject()["project"].get("optional-dependencies", {}).items():
        assert "librosa" not in {_name(r) for r in reqs}, f"librosa is in the {extra!r} extra"


def test_numba_and_llvmlite_are_absent_or_pinned_for_intel_mac():
    deps = _base()
    ceiling = {"numba": "0.61", "llvmlite": "0.44"}
    for dist, below in ceiling.items():
        for req in deps.get(dist, []):
            assert f"<{below}" in req.replace(" ", ""), (
                f"{req!r}: {dist} must be pinned <{below} on Intel Mac "
                f"(the newer releases have no x86_64 macOS wheel)")


def test_every_unguarded_third_party_import_is_declared():
    deps = _base()
    files = _scanned_files()
    assert len(files) > 50, f"the scan found only {len(files)} files; the tree moved"
    missing = []
    for path in files:
        for module, line in _unguarded_imports(path):
            dist = DIST_FOR_IMPORT.get(module, module).lower().replace("_", "-")
            if dist not in deps:
                missing.append(f"{path.relative_to(ROOT)}:{line} imports {module} "
                               f"({dist} is not a base dependency)")
    assert missing == [], "\n".join(missing)


def test_the_scan_sees_the_imports_that_were_missing():
    """The scan itself works: it finds the imports that broke installs.

    websockets was the fourth; since 2026-10-29 nothing in the package imports
    it, and it is no longer declared."""
    seen = {m for f in _scanned_files() for m, _ in _unguarded_imports(f)}
    assert {"yaml", "scipy", "numpy"} <= seen


def test_setup_py_is_a_bare_shim():
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "setup"]
    assert len(calls) == 1, "setup.py must call setup() exactly once"
    assert not calls[0].args and not calls[0].keywords, (
        "setup.py passes arguments to setup(); pyproject.toml is the one list")
    assert "ext_modules" not in (ROOT / "setup.py").read_text(encoding="utf-8").split("from setuptools")[1]


def test_py_typed_ships_where_package_data_says():
    data = _pyproject()["tool"]["setuptools"]["package-data"]
    assert "py.typed" in data["christman_voice_sdk"]
    assert (ROOT / "christman_voice_sdk" / "py.typed").is_file()


def test_cryptography_is_the_crypto_extra_and_optional_in_voicepack():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "cryptography" in {_name(r) for r in extras["crypto"]}
    assert "cryptography" not in _base(), "cryptography is optional; it belongs in the extra"
    voicepack = ROOT / "christman_voice_sdk" / "timbre" / "voicepack.py"
    assert "cryptography" not in {m for m, _ in _unguarded_imports(voicepack)}


def _import_in_child(module: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); import {module}"],
        capture_output=True, text=True, env=env, timeout=120, cwd=str(ROOT),
    )


def test_christman_ocr_shared_imports_with_the_base_dependencies(tmp_path):
    proc = _import_in_child("christman_ocr_shared", {"CHRISTMAN_AI_HOME": str(tmp_path)})
    assert proc.returncode == 0, proc.stderr[-2000:]


# -- one pip install, nothing after it (2026-11-05) -----------------------------
#
# WHAT CHANGED AND WHY (2026-11-05): Everett, "I want the sound package to be
# an installable package. I want people to be able to just download it and
# install it." pip install now compiles the C library (scripts/pip_build.py
# runs scripts/build_dsp.py). These tests hold that in place: every console
# script names a function that exists (christman-build-dsp named a module that
# is never installed), and a wheel pip builds carries a library that loads.


def test_every_console_script_names_a_function_that_exists():
    import importlib

    broken = []
    for name, target in _pyproject()["project"]["scripts"].items():
        module_name, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            broken.append(f"{name} = {target}: {module_name} does not import ({exc})")
            continue
        if not callable(getattr(module, attr, None)):
            broken.append(f"{name} = {target}: {module_name} has no {attr}")
    assert broken == [], "\n".join(broken)


def test_the_build_compiles_the_library_as_its_own_step():
    cmdclass = _pyproject()["tool"]["setuptools"]["cmdclass"]
    assert cmdclass["build"] == "scripts.pip_build.build"
    assert cmdclass["build_christman_dsp"] == "scripts.pip_build.build_christman_dsp"
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for needed in ("christman_dsp.c", "christman_synth.c", "christman_vocoder.c",
                   "scripts/build_dsp.py", "scripts/pip_build.py"):
        assert needed in manifest, f"{needed} is not in MANIFEST.in: an sdist could not build"
    for package, data in _pyproject()["tool"]["setuptools"]["package-data"].items():
        assert not {"*.so", "*.dylib"} & set(data), (
            f"{package}: a library left in a checkout would ride into the wheel")


def _setuptools_too_old() -> str | None:
    try:
        import setuptools
    except ImportError:
        return "setuptools is not installed in this environment"
    major_minor = tuple(int(p) for p in re.findall(r"\d+", setuptools.__version__)[:2])
    if major_minor < (70, 1):
        return (f"setuptools {setuptools.__version__} here is older than 70.1, which pyproject.toml "
                "requires (pip install builds in its own environment with a newer one)")
    return None


def test_a_wheel_pip_builds_carries_the_library_and_it_loads(tmp_path):
    """pip wheel on a copy of the source: the library is in the wheel, tagged
    for this platform, held/ and scripts/ are not, and the library from the
    wheel loads whole and measures."""
    import shutil
    import zipfile

    if not (ROOT / "christman_voice_sdk").is_dir():
        pytest.skip(f"not run: {ROOT} holds no source packages (these tests are running against an "
                    "installed copy); a wheel is built from the source folder")
    too_old = _setuptools_too_old()
    if too_old:
        pytest.skip(f"not run: {too_old}")
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode != 0:
        pytest.skip(f"not run: pip is not available to {sys.executable}")
    if not (shutil.which(os.environ.get("CC") or "cc") or shutil.which("clang") or shutil.which("gcc")):
        pytest.skip("not run: no C compiler here; pip install needs one to build the library")

    source = tmp_path / "Christman-Sound"
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        "build", "dist", "*.egg-info", "__pycache__", "*.so", "*.dylib", ".pytest_cache", "tests"))
    wheels = tmp_path / "wheels"
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(wheels), str(source)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    built = sorted(wheels.glob("christman_sound-*.whl"))
    assert len(built) == 1, built
    assert re.match(r"christman_sound-[^-]+-py3-none-(?!any)\w+\.whl$", built[0].name), built[0].name

    with zipfile.ZipFile(built[0]) as wheel:
        names = wheel.namelist()
        # The wheel is not pure, so setuptools files the packages under
        # <name>.data/purelib/; pip installs that into site-packages as usual.
        # What matters is that the library sits beside dsp.py, which loads it.
        installed = {re.sub(r"^christman_sound-[^/]+\.data/(pure|plat)lib/", "", n): n for n in names}
        assert f"christman_voice_sdk/engines/{LIB_NAME}" in installed, sorted(installed)[:20]
        assert "christman_voice_sdk/engines/dsp.py" in installed
        library, loader = installed[f"christman_voice_sdk/engines/{LIB_NAME}"], installed["christman_voice_sdk/engines/dsp.py"]
        assert library.rsplit("/", 1)[0] == loader.rsplit("/", 1)[0], (library, loader)
        assert not [n for n in installed if n.startswith(("held/", "scripts/", "tests/"))]
        for package in ("christman_voice_sdk/", "voice_creation_center/"):
            assert any(n.startswith(package) for n in installed), f"{package} is not in the wheel"
        wheel.extract(library, tmp_path / "unpacked")

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import build_dsp
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    summary = build_dsp.load_check(tmp_path / "unpacked" / library)
    assert summary.startswith("load check passed"), summary
