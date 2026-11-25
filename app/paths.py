"""Where the app's files live: run from the repository, or installed with Setup.exe.

The installer puts an ``INSTALLED`` file next to ``app/`` (the program folder, replaced by every
update). An installed copy keeps everything it writes in ``%LOCALAPPDATA%\\AI Reel Maker``:
the data folder, its optional ``settings.env``, the prepared Remotion tools and the logs. Run
from the repository nothing changes: ``data/``, ``.env`` and ``remotion/`` next to ``app/``.

``AIREEL_HOME`` (environment) chooses another home for an installed copy (used by tests).
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "AI Reel Maker"
MARKER = "INSTALLED"
# The installed copy's port, so it can run next to a copy started from the repository (5000).
INSTALLED_PORT = 5050


def program_dir() -> Path:
    """The folder holding ``app/``, ``run.py`` and ``frontend/`` (the repository or install)."""
    return Path(__file__).resolve().parent.parent


def is_installed() -> bool:
    return (program_dir() / MARKER).is_file()


def user_home() -> Path:
    """The installed copy's own folder for everything it writes."""
    chosen = os.environ.get("AIREEL_HOME")
    if chosen:
        return Path(chosen)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_NAME


def home() -> Path:
    """Where data, settings.env, logs and the prepared tools go."""
    return user_home() if is_installed() else program_dir()


def default_data_dir() -> Path:
    return home() / "data"


def env_file() -> Path:
    return user_home() / "settings.env" if is_installed() else program_dir() / ".env"


def default_host() -> str:
    """The installed copy answers only this computer (no firewall question on first start);
    the repository copy keeps listening on every network card as before."""
    return "127.0.0.1" if is_installed() else "0.0.0.0"


def default_port() -> int:
    return INSTALLED_PORT if is_installed() else 5000


def logs_dir() -> Path:
    return home() / ("logs" if is_installed() else "data")


def remotion_source() -> Path:
    """The composition's code as shipped (read only once installed)."""
    return program_dir() / "remotion"


def remotion_dir() -> Path:
    """Where the composition is prepared (packages installed, bundle built): a copy in the
    user's folder once installed, because the program folder is replaced by updates."""
    return user_home() / "remotion" if is_installed() else remotion_source()


def tools_dir() -> Path:
    """Bundled programs (ffmpeg, node) of an installed copy."""
    return program_dir() / "tools"


def bundled_tool_dirs() -> list[Path]:
    """Folders of bundled programs that exist, in the order they go first on PATH."""
    folders = [tools_dir() / "ffmpeg", tools_dir() / "node"]
    return [folder for folder in folders if folder.is_dir()]


def put_bundled_tools_first() -> list[Path]:
    """Put the bundled ffmpeg and node first on PATH (the versions the release was tested with).
    Programs the app finds on PATH (Ollama, Claude) are unaffected. Returns the folders added."""
    folders = bundled_tool_dirs()
    if not folders:
        return []
    current = os.environ.get("PATH", "").split(os.pathsep)
    added = [str(folder) for folder in folders]
    os.environ["PATH"] = os.pathsep.join([*added, *[p for p in current if p and p not in added]])
    return folders
