#!/usr/bin/env python3
"""Stop the running app (started with start.bat or run.py) and the FFmpeg work it started.

A job that was running continues where it stopped the next time the app starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.instance import stop_app  # noqa: E402

MESSAGES = {
    "stopped": "AI Reel Maker stopped. A running job continues where it stopped next time you start the app.",
    "not_running": "AI Reel Maker was not running.",
    "not_ours": "AI Reel Maker was not running (the process it recorded is now another program, left alone).",
}


def main() -> int:
    outcome = stop_app(get_settings().data_dir)
    print(MESSAGES[outcome])
    return 0


if __name__ == "__main__":
    sys.exit(main())
