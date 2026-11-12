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
New in the package: it reads the pages from christman_voice_sdk/audio/filament/pages,
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

PAGES_DIR = ROOT / "christman_voice_sdk" / "audio" / "filament" / "pages"
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


# ------------------------------------------------------- which ear a page calls
#
# 2026-11-05. The three pages held
#     const EAR = location.port === '4850' ? '' : 'http://127.0.0.1:4850';
# so a copy served by a door on any other port (a test copy) sent its sound,
# and its board and Carbon posts, to the ear on 4850. The page's own EAR
# lines are run with node against the address each case gives.

def _ear_for(page: str, href: str) -> str:
    text = (PAGES_DIR / page).read_text(encoding="utf-8")
    start = text.index("  const EAR = (() => {")
    block = text[start: text.index("  })();", start) + len("  })();")]
    js = f"const location = new URL({json.dumps(href)});\n{block}\nprocess.stdout.write(JSON.stringify(EAR));"
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


@needs_node
@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("href, ear", [
    ("http://127.0.0.1:4850/", ""),                              # the usual door: itself
    ("http://127.0.0.1:4861/housing", ""),                       # a door on another port: itself, not 4850
    ("http://localhost:59999/live.html", ""),
    ("file:///Users/someone/filament-door.html", "http://127.0.0.1:4850"),   # opened as a file: the ear here
    ("http://127.0.0.1:4861/?ear-port=4850", "http://127.0.0.1:4850"),       # named on purpose
    ("http://127.0.0.1:4861/?ear-port=4861", ""),
    ("http://127.0.0.1:4861/?ear-port=0", ""),                   # not a port: the page's own door
    ("http://127.0.0.1:4861/?ear-port=70000", ""),
    ("http://127.0.0.1:4861/?ear-port=evil.example", ""),        # no other machine can be named
    ("http://127.0.0.1:4861/?ear-port=4850@evil.example", ""),
    ("http://127.0.0.1:4861/?ear=http://evil.example", ""),
])
def test_a_page_calls_the_door_that_served_it(page, href, ear):
    assert _ear_for(page, href) == ear


@pytest.mark.parametrize("page", PAGES)
def test_no_page_falls_back_to_4850_when_served_from_another_port(page):
    text = (PAGES_DIR / page).read_text(encoding="utf-8")
    assert "location.port === '4850'" not in text
    # 4850 is written once in code, for a page opened as a file, and nowhere else outside comments.
    code = [ln for ln in text.splitlines() if "4850" in ln and not ln.strip().startswith("//")]
    assert code == ["    return location.protocol === 'file:' ? 'http://127.0.0.1:4850' : '';"], code


def test_the_simple_page_shows_the_door_it_was_served_from():
    text = (PAGES_DIR / "ui.html").read_text(encoding="utf-8")
    assert '<span id="door-port">4850</span>' in text
    assert 'if (h.door) document.getElementById("door-port").textContent = h.door;' in text
    assert 'Allow microphone for " + (location.host || "127.0.0.1:4850")' in text
    assert "Allow microphone for 127.0.0.1:4850" not in text


# ------------------------------------------------------ every page's script runs
#
# 2026-11-05. ui.html's escapeHtml held '"': """ (its entities had been turned
# back into the characters), a syntax error: the browser threw the whole script
# away, and the simple page sat at "probing" with its button off. It came that
# way from THEFILAMENT. No test parsed the pages' scripts; these do, with node.

ALL_PAGES = ["filament-door.html", "filament-housing.html", "filament-live.html", "filament.html", "ui.html"]


def _inline_scripts(page: str) -> list[str]:
    text = (PAGES_DIR / page).read_text(encoding="utf-8")
    return re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", text, re.S)


@needs_node
@pytest.mark.parametrize("page", ALL_PAGES + ["dialect-map.js"])
def test_every_script_the_package_ships_is_javascript_a_browser_will_run(page, tmp_path):
    scripts = [(PAGES_DIR / page).read_text(encoding="utf-8")] if page.endswith(".js") else _inline_scripts(page)
    assert scripts, f"{page} holds no script"
    for n, script in enumerate(scripts):
        target = tmp_path / f"{n}.js"
        target.write_text(script, encoding="utf-8")
        proc = subprocess.run(["node", "--check", str(target)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{page}, script {n + 1}: {proc.stderr.strip().splitlines()[-3:]}"


@needs_node
def test_the_simple_page_escapes_what_it_writes():
    (script,) = _inline_scripts("ui.html")
    start = script.index("function escapeHtml(s) {")
    fn = script[start: script.index("\n      }", start) + len("\n      }")]
    got = run_lines([fn], "escapeHtml(`<b onclick=\"x('y')\">Tom & Jerry</b>`)")
    assert got == "&lt;b onclick=&quot;x(&#39;y&#39;)&quot;&gt;Tom &amp; Jerry&lt;/b&gt;"
