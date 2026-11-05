# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""Every Filament page the package ships sends its own stream name on /live.

Brought from THEFILAMENT ear/tests/test_pages_session.py. The pages' own
JavaScript lines are run with node, so the check is on the code the browser
runs. The frozen filament.html is checked byte for byte in
test_filament_door.py, and here to hold no stream name.

WHAT CHANGED AND WHY (2026-11-05)
New in the package: it reads the pages from christman_sound/filament/pages,
where the server serves them from. THEFILAMENT's copies under public/ belong
to its own app and are not in the package.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from filament_helpers import ROOT

PAGES_DIR = ROOT / "christman_sound" / "filament" / "pages"
PAGES = ["filament-door.html", "filament-housing.html", "filament-live.html"]
NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="not run: node is not on this machine")


def run_lines(lines: list[str], expr: str):
    js = "\n".join(lines) + f"\nprocess.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@needs_node
@pytest.mark.parametrize("page", PAGES)
def test_each_page_sends_its_own_stream(page):
    text = (PAGES_DIR / page).read_text(encoding="utf-8")
    posts = re.findall(r"fetch\(EAR\+'/live'\+([^,]+),", text)
    assert posts == ["LQ(q||'')"], page
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith(("const SESSION", "const LQ"))]
    assert len(lines) == 2, page
    got = run_lines(lines, "[SESSION, LQ('?reset=1'), LQ(''), LQ('?end=1')]")
    other = run_lines(lines, "SESSION")
    name = got[0]
    assert NAME.match(name), name
    assert name != other, f"{page}: two page loads must get two names"
    assert got[1:] == [f"?reset=1&session={name}", f"?session={name}", f"?end=1&session={name}"]


@needs_node
def test_simple_page_sends_its_own_stream():
    text = (PAGES_DIR / "ui.html").read_text(encoding="utf-8")
    assert 'fetch("/live" + (query ? query + "&" : "?") + "session=" + SESSION,' in text
    line = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("const SESSION")]
    assert len(line) == 1
    a, b = run_lines(line, "SESSION"), run_lines(line, "SESSION")
    assert NAME.match(a)
    assert a != b


def test_frozen_page_holds_no_stream_name():
    assert "session" not in (PAGES_DIR / "filament.html").read_text(encoding="utf-8")
