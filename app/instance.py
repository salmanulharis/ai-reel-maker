"""One running copy of the app, and finding and stopping it from the outside.

``run.py`` holds an exclusive lock on ``data/app.lock`` for as long as it runs; a second copy
cannot take it and knows the app is already running (or still starting). The operating system
releases the lock when the process ends in any way, so a crash or a power cut never blocks the
next start. ``data/app.pid`` records which process holds the lock, for ``scripts/stop_app.py``.

The record is a separate file because a Windows lock also stops other programs from reading
the locked bytes.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, BinaryIO

LOCK_NAME = "app.lock"
RECORD_NAME = "app.pid"
# The command line of the app's process contains this (see is_app_process).
SCRIPT_NAME = "run.py"
# How long to wait for the app to answer before giving up on opening the browser.
READY_TIMEOUT = 90.0
# Two start times this close are the same process (the record keeps a rounded time).
_SAME_START_SECONDS = 2.0


class InstanceLock:
    """The exclusive lock of the running copy, and its process record."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.lock_path = self.data_dir / LOCK_NAME
        self.record_path = self.data_dir / RECORD_NAME
        self._handle: BinaryIO | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """Take the lock; False when another copy of the app holds it."""
        if self._handle is not None:
            return True
        self.data_dir.mkdir(parents=True, exist_ok=True)
        handle = open(self.lock_path, "a+b")  # noqa: SIM115 - kept open while the app runs
        try:
            _lock(handle)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def write_record(self, port: int) -> None:
        """Note which process runs the app (only while holding the lock)."""
        if self._handle is None:
            raise RuntimeError("the instance lock is not held")
        record = {
            "pid": os.getpid(),
            "port": int(port),
            "started": process_start_time(os.getpid()),
            "script": SCRIPT_NAME,
        }
        temp = self.record_path.with_suffix(".tmp")
        temp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(temp, self.record_path)

    def release(self) -> None:
        """Remove the record and free the lock. Safe to call more than once."""
        if self._handle is None:
            return
        record = read_record(self.data_dir)
        if record and record.get("pid") == os.getpid():
            self.record_path.unlink(missing_ok=True)
        try:
            _unlock(self._handle)
        except OSError:
            pass
        self._handle.close()
        self._handle = None


def _lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_record(data_dir: Path) -> dict[str, Any] | None:
    """The running copy's record, or None when there is none or it cannot be read."""
    try:
        data = json.loads((Path(data_dir) / RECORD_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pid"), int):
        return None
    return data


def process_start_time(pid: int) -> float | None:
    """When a process started (seconds since the epoch), or None without psutil."""
    try:
        import psutil

        return round(psutil.Process(pid).create_time(), 2)
    except Exception:  # noqa: BLE001 - only used to tell processes apart
        return None


def is_app_process(process, record: dict[str, Any]) -> bool:
    """Whether ``process`` (a psutil.Process) is the app named in ``record``: same ID, the same
    start time, and run.py on its command line. A process that later reused the ID is not."""
    try:
        if process.pid != record.get("pid"):
            return False
        started = record.get("started")
        if started is not None and abs(process.create_time() - float(started)) > _SAME_START_SECONDS:
            return False
        return any(Path(part).name == record.get("script", SCRIPT_NAME) for part in process.cmdline())
    except Exception:  # noqa: BLE001 - gone or not readable: not ours to stop
        return False


def stop_app(data_dir: Path, timeout: float = 10.0) -> str:
    """Stop the running copy and everything it started (FFmpeg). Returns "stopped",
    "not_running", or "not_ours" (the recorded process is something else; left alone)."""
    import psutil

    record = read_record(data_dir)
    if record is None:
        return "not_running"
    try:
        process = psutil.Process(record["pid"])
    except psutil.Error:
        _remove_stale_record(data_dir, record)
        return "not_running"
    if not is_app_process(process, record):
        _remove_stale_record(data_dir, record)
        return "not_ours"
    try:
        children = process.children(recursive=True)
    except psutil.Error:
        children = []
    # Children first, so no FFmpeg is left running after the app is gone.
    for item in [*children, process]:
        try:
            item.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs([*children, process], timeout=timeout)
    for item in alive:
        try:
            item.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=timeout)
    _remove_stale_record(data_dir, record)
    return "stopped"


def _remove_stale_record(data_dir: Path, record: dict[str, Any]) -> None:
    """Remove the record when it still names the process it named before."""
    current = read_record(data_dir)
    if current is not None and current.get("pid") == record.get("pid"):
        (Path(data_dir) / RECORD_NAME).unlink(missing_ok=True)


def local_url(port: int) -> str:
    return f"http://localhost:{int(port)}"


def wait_until_ready(port: int, timeout: float = READY_TIMEOUT, interval: float = 0.5) -> bool:
    """Whether the app answers its health check within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{int(port)}/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - fixed local URL
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(interval)
    return False


def open_when_ready(port: int, timeout: float = READY_TIMEOUT) -> bool:
    """Open the app in the browser once it answers; False when it never did."""
    if not wait_until_ready(port, timeout):
        return False
    webbrowser.open(local_url(port))
    return True
