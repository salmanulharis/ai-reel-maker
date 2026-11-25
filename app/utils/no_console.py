"""Stop console windows from flashing up when the windowless app starts a program.

The installed app runs under ``pythonw.exe`` (no console). Windows then gives every console
program it starts (FFmpeg, node, npm, Ollama, Claude) a new console window of its own, one
flash per FFmpeg call. ``hide_console_windows`` makes every new process start without one, unless
the caller chose its own creation flags. Only the launcher calls it; ``run.py`` in a console
does not need it.
"""

from __future__ import annotations

import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000
_patched = False


def hide_console_windows() -> bool:
    """Patch subprocess.Popen once (Windows only). Returns True when it is patched."""
    global _patched
    if sys.platform != "win32":
        return False
    if _patched:
        return True
    original = subprocess.Popen.__init__

    def __init__(self, *args, **kwargs):  # noqa: N807 - replaces Popen.__init__
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = CREATE_NO_WINDOW
        original(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__  # type: ignore[method-assign]
    _patched = True
    return True
