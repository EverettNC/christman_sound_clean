"""
================================================================================
FILE: vcc_paths.py
PROJECT: Christman Voice Creation Center
AUTHOR: The Christman AI Project | Luma Cognify AI
--------------------------------------------------------------------------------
PURPOSE:
    The one place every VCC path comes from.

    WHAT CHANGED AND WHY (added 2026-10-28)
    The VCC was split in two. Voice_registry.py, voice_express.py and
    voice_ingestor.py sit at the repo root but built their paths from
    Path(__file__).parent.parent, so they read and wrote packs/, inventory/,
    express/, incoming/ and the rest in the folder ABOVE the repo, while
    engines/voice_loader.py, engines/voice_path_registrar.py and the studio
    used the repo itself. The two halves never met: the loader could not see
    what the registry wrote, and three writers kept two different express
    indexes.

    Now there is one base directory. It is, in order:
      1. the base_dir a caller passes in, else
      2. the folder named by $VCC_HOME, else
      3. default_base(): this folder in a Christman-Sound checkout, and
         $CHRISTMAN_AI_HOME/voice_creation_center (default
         ~/.christman_ai/voice_creation_center) in an installed copy.
    Every folder below is a fixed name under it, the same for every module.

    WHAT CHANGED AND WHY (2026-11-05)
    Everett asked that the sound package install from a download with one pip
    install. Installed that way (not editable), this folder is inside
    site-packages, and the VCC would have written client recordings, packs,
    the express index and its logs among the installed code: a folder pip
    replaces on the next upgrade and removes on uninstall, and one a person
    may not be able to write to at all. So step 3 now asks where this folder
    is. In a checkout (an editable install, or code run from the source
    folder) the base stays this folder, exactly as before. In an installed
    copy it is the per-user folder beside the SDK's own data
    ($CHRISTMAN_AI_HOME, else ~/.christman_ai: the same rule as
    christman_voice_sdk.audio.config.sdk_data_root), under
    voice_creation_center/. $VCC_HOME and base_dir still win over both.
    Nothing is created here; each writer makes its own folders when it writes.
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

#: The folder this file sits in: the base when nothing else is given and this
#: is a checkout.
REPO_ROOT = Path(__file__).resolve().parent

#: Environment variable that moves the whole VCC's data somewhere else.
HOME_ENV = "VCC_HOME"

#: The family's data folder for an installed copy, the variable the SDK reads
#: (christman_voice_sdk.audio.config.sdk_data_root), else ~/.christman_ai.
AI_HOME_ENV = "CHRISTMAN_AI_HOME"

#: The VCC's folder under it.
INSTALLED_SUBDIR = "voice_creation_center"


def is_source_checkout(package_dir: Union[str, Path] = REPO_ROOT) -> bool:
    """
    True when package_dir sits in a Christman-Sound source folder: the folder
    above it holds pyproject.toml and christman_dsp.c. Neither is ever
    installed, so an installed copy (in site-packages) answers False, and an
    editable install, which runs from the checkout, answers True.
    """
    source = Path(package_dir).parent
    return (source / "pyproject.toml").is_file() and (source / "christman_dsp.c").is_file()


def installed_default_base() -> Path:
    """$CHRISTMAN_AI_HOME/voice_creation_center, else ~/.christman_ai/voice_creation_center."""
    raw = os.environ.get(AI_HOME_ENV)
    root = Path(raw).expanduser() if raw else Path.home() / ".christman_ai"
    return root / INSTALLED_SUBDIR


def default_base(package_dir: Union[str, Path] = REPO_ROOT) -> Path:
    """The base when no base_dir and no $VCC_HOME are given: package_dir in a
    checkout, else the per-user folder (installed_default_base)."""
    return Path(package_dir) if is_source_checkout(package_dir) else installed_default_base()


class VCCPaths:
    """Every folder and file the VCC reads or writes, under one base directory."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        chosen = base_dir if base_dir is not None else os.environ.get(HOME_ENV) or default_base()
        self.base = Path(chosen).expanduser().resolve()

        # Voice packs and the registry's inventory (Voice_registry.py, engines/voice_loader.py)
        self.packs = self.base / "packs"
        self.inventory = self.base / "inventory"
        self.inventory_index = self.inventory / "__index__.json"

        # The express lane (voice_express.py, engines/voice_path_registrar.py, voice_ingestor.py)
        self.express = self.base / "express"
        self.express_index = self.express / "express_index.json"
        self.phrase_cache = self.express / "phrase_cache"
        self.priority_queue = self.express / "priority_queue"

        # The ingestor's front door (voice_ingestor.py)
        self.incoming = self.base / "incoming"
        self.processed = self.base / "processed"
        self.failed = self.base / "failed"
        self.ingested_phrases = self.packs / "phrases"

        # Audit logs (engines/voice_logger.py)
        self.logs = self.base / "logs"

        # The studio's raw takes (studio/live_studio.py)
        self.studio_raw = self.base / "studio" / "raw"

    def __repr__(self) -> str:
        return f"VCCPaths(base={str(self.base)!r})"


def vcc_paths(base_dir: Optional[Union[str, Path]] = None) -> VCCPaths:
    """The VCC's paths under `base_dir`, else $VCC_HOME, else default_base()."""
    return VCCPaths(base_dir)
