"""One running copy of the app, starting it again, and stopping it from the outside."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import run as run_module
from app import instance
from app.config import Settings
from app.instance import InstanceLock, is_app_process, read_record, stop_app, wait_until_ready

ROOT = Path(__file__).resolve().parent.parent


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT)}


def _holder(data_dir: Path, *, crash: bool = False) -> subprocess.Popen:
    """Another process that takes the lock, says so, then waits (or ends without cleaning up)."""
    code = textwrap.dedent(
        f"""
        import os, sys, time
        from pathlib import Path
        from app.instance import InstanceLock
        lock = InstanceLock(Path({str(data_dir)!r}))
        if not lock.acquire():
            print("busy", flush=True)
            sys.exit(1)
        lock.write_record(5000)
        print("got", flush=True)
        if {crash!r}:
            os._exit(1)
        time.sleep(60)
        """
    )
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=_env())
    assert process.stdout.readline().strip() == "got"
    return process


def test_a_second_copy_cannot_take_the_lock(tmp_path):
    holder = _holder(tmp_path)
    try:
        assert not InstanceLock(tmp_path).acquire()
        assert read_record(tmp_path)["pid"] == holder.pid
    finally:
        holder.kill()
        holder.wait()
    assert InstanceLock(tmp_path).acquire(), "the lock is free once that process has ended"


def test_a_crash_or_power_cut_never_blocks_the_next_start(tmp_path):
    crashed = _holder(tmp_path, crash=True)
    crashed.wait()
    assert read_record(tmp_path) is not None, "the record is left behind by the crash"
    lock = InstanceLock(tmp_path)
    assert lock.acquire()
    lock.write_record(5001)
    assert read_record(tmp_path)["pid"] == os.getpid() and read_record(tmp_path)["port"] == 5001
    lock.release()
    assert read_record(tmp_path) is None
    lock.release()  # safe to call twice


def test_the_record_cannot_be_written_without_the_lock(tmp_path):
    with pytest.raises(RuntimeError):
        InstanceLock(tmp_path).write_record(5000)


def test_unreadable_records_count_as_none(tmp_path):
    (tmp_path / "app.pid").write_text("{not json", encoding="utf-8")
    assert read_record(tmp_path) is None
    (tmp_path / "app.pid").write_text(json.dumps({"pid": "12"}), encoding="utf-8")
    assert read_record(tmp_path) is None
    assert read_record(tmp_path / "missing") is None


# Telling the app apart from other processes


def _fake(pid=10, started=100.0, cmdline=("python", "C:\\app\\run.py")):
    return SimpleNamespace(pid=pid, create_time=lambda: started, cmdline=lambda: list(cmdline))


def test_only_the_recorded_app_process_counts():
    record = {"pid": 10, "started": 100.0, "script": "run.py"}
    assert is_app_process(_fake(), record)
    assert not is_app_process(_fake(pid=11), record)
    assert not is_app_process(_fake(started=500.0), record), "the ID was reused by a newer process"
    assert not is_app_process(_fake(cmdline=("notepad.exe",)), record)
    assert not is_app_process(_fake(cmdline=("python", "other_run.py")), record)

    def gone():
        raise psutil.NoSuchProcess(10)

    assert not is_app_process(SimpleNamespace(pid=10, create_time=gone, cmdline=gone), record)


def test_stopping_with_nothing_running(tmp_path):
    assert stop_app(tmp_path) == "not_running"
    (tmp_path / "app.pid").write_text(json.dumps({"pid": 2**22 + 12345, "started": 1.0}), encoding="utf-8")
    assert stop_app(tmp_path) == "not_running"
    assert not (tmp_path / "app.pid").exists(), "a record of a process that is gone is removed"


def test_another_program_with_the_recorded_id_is_left_alone(tmp_path):
    record = {"pid": os.getpid(), "started": instance.process_start_time(os.getpid()), "script": "run.py"}
    (tmp_path / "app.pid").write_text(json.dumps(record), encoding="utf-8")
    assert stop_app(tmp_path) == "not_ours"
    assert psutil.pid_exists(os.getpid())
    assert not (tmp_path / "app.pid").exists()


def test_stop_ends_the_app_and_the_work_it_started(tmp_path):
    data = tmp_path / "data"
    script = tmp_path / "run.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys, time
            from pathlib import Path
            from app.instance import InstanceLock
            lock = InstanceLock(Path({str(data)!r}))
            lock.acquire()
            lock.write_record(5000)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            print(child.pid, flush=True)
            time.sleep(60)
            """
        ),
        encoding="utf-8",
    )
    app = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True, env=_env())
    child_pid = int(app.stdout.readline())
    try:
        assert stop_app(data) == "stopped"
        assert app.wait(timeout=10) is not None
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        assert read_record(data) is None
        assert InstanceLock(data).acquire(), "the next start is not blocked"
    finally:
        for pid in (app.pid, child_pid):
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass


# run.py


@pytest.fixture
def run_settings(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", app_port=_free_port())
    monkeypatch.setattr(run_module, "get_settings", lambda: settings)
    return settings


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_starting_again_only_points_to_the_running_copy(run_settings, monkeypatch, capsys):
    holder = _holder(run_settings.data_dir)
    try:
        opened = []
        monkeypatch.setattr(run_module, "open_when_ready", lambda port: opened.append(port) or True)
        monkeypatch.setattr("app.create_app", lambda *a, **k: pytest.fail("a second copy must not start the app"))
        assert run_module.main(["--open-browser"]) == 0
        assert "already running at http://localhost:5000" in capsys.readouterr().out
        assert opened == [5000], "the port of the running copy, from its record"
    finally:
        holder.kill()
        holder.wait()


def test_ctrl_c_stops_cleanly_and_frees_the_lock(run_settings, monkeypatch, capsys):
    import app

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(app, "create_app", lambda settings: object())
    monkeypatch.setattr(app.socketio, "run", interrupted)
    assert run_module.main([]) == 0
    assert "stopped" in capsys.readouterr().out
    assert read_record(run_settings.data_dir) is None
    assert InstanceLock(run_settings.data_dir).acquire()


def test_a_busy_port_is_explained_and_the_lock_freed(run_settings, monkeypatch, capsys):
    import app

    def busy(*args, **kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr(app, "create_app", lambda settings: object())
    monkeypatch.setattr(app.socketio, "run", busy)
    assert run_module.main([]) == 1
    assert "APP_PORT" in capsys.readouterr().err
    assert InstanceLock(run_settings.data_dir).acquire()


def test_nothing_is_opened_when_the_app_never_answers():
    started = time.monotonic()
    assert not wait_until_ready(_free_port(), timeout=0.6, interval=0.1)
    assert time.monotonic() - started < 5


def test_batch_files_keep_windows_line_endings():
    for name in ("start.bat", "stop.bat"):
        data = (ROOT / name).read_bytes()
        assert data.count(b"\n") == data.count(b"\r\n"), name
