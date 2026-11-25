"""FFmpegRunner.run_streaming: the runner for Node and the Claude CLI (real child processes)."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from app.video.ffmpeg_runner import CancelledError, CommandError, FFmpegRunner


def test_lines_stdin_and_exit_code():
    seen: list[str] = []
    script = "import sys; data = sys.stdin.read(); print('got', len(data)); print('line2'); sys.stderr.write('warn'); sys.exit(3)"
    code, lines, errors = FFmpegRunner().run_streaming([sys.executable, "-c", script], stdin_text="x" * 100_000, on_line=seen.append)
    assert code == 3 and lines == ["got 100000", "line2"] == seen and errors == "warn"


def test_timeout_and_missing_program():
    with pytest.raises(CommandError, match="did not finish"):
        FFmpegRunner().run_streaming([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    with pytest.raises(CommandError, match="Could not start"):
        FFmpegRunner().run_streaming(["definitely-not-a-program-xyz"])


def test_cancel_stops_the_whole_tree():
    psutil = pytest.importorskip("psutil")
    runner = FFmpegRunner()
    child = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); print('started', flush=True); time.sleep(60)"
    found: dict = {}

    def watch(line: str) -> None:
        if line == "started":
            me = psutil.Process()
            found["tree"] = [p for p in me.children(recursive=True)]
            threading.Thread(target=runner.cancel, daemon=True).start()

    began = time.monotonic()
    with pytest.raises(CancelledError):
        runner.run_streaming([sys.executable, "-c", child], on_line=watch, timeout=50)
    assert time.monotonic() - began < 30
    time.sleep(0.5)
    assert len(found["tree"]) >= 2 and not any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in found["tree"])
