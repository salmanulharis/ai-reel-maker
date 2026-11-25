"""The installed (Setup.exe) copy: where it keeps its files, the Remotion source copy, the
windowless launcher's helpers, and the self-test."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import paths
from app.compose import remotion

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def installed(tmp_path, monkeypatch):
    """Pretend this copy was installed: a marker in a fake program folder, home in tmp_path."""
    program = tmp_path / "program"
    program.mkdir()
    (program / paths.MARKER).write_text("x")
    monkeypatch.setattr(paths, "program_dir", lambda: program)
    monkeypatch.setenv("AIREEL_HOME", str(tmp_path / "home"))
    return tmp_path


def test_repository_copy_keeps_its_files_next_to_the_code(monkeypatch):
    monkeypatch.delenv("AIREEL_HOME", raising=False)
    assert not paths.is_installed()
    assert paths.default_data_dir() == ROOT / "data"
    assert paths.env_file() == ROOT / ".env"
    assert paths.default_port() == 5000 and paths.default_host() == "0.0.0.0"
    assert paths.remotion_dir() == paths.remotion_source() == ROOT / "remotion"


def test_installed_copy_writes_only_to_its_home(installed):
    home = installed / "home"
    assert paths.is_installed()
    assert paths.default_data_dir() == home / "data"
    assert paths.env_file() == home / "settings.env"
    assert paths.logs_dir() == home / "logs"
    assert paths.remotion_dir() == home / "remotion"
    assert paths.default_port() == paths.INSTALLED_PORT and paths.default_host() == "127.0.0.1"
    from app.config import Settings

    settings = Settings()
    assert settings.data_dir == (home / "data").resolve() and settings.app_port == paths.INSTALLED_PORT
    assert settings.app_host == "127.0.0.1"


def test_user_home_defaults_to_local_app_data(monkeypatch, tmp_path):
    monkeypatch.delenv("AIREEL_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert paths.user_home() == tmp_path / "AI Reel Maker"


def test_bundled_tools_go_first_on_path(installed, monkeypatch):
    program = installed / "program"
    (program / "tools" / "ffmpeg").mkdir(parents=True)
    (program / "tools" / "node").mkdir(parents=True)
    monkeypatch.setenv("PATH", os.pathsep.join(["C:\\other", str(program / "tools" / "node")]))
    added = paths.put_bundled_tools_first()
    parts = os.environ["PATH"].split(os.pathsep)
    assert added == [program / "tools" / "ffmpeg", program / "tools" / "node"]
    assert parts[:3] == [str(program / "tools" / "ffmpeg"), str(program / "tools" / "node"), "C:\\other"]


def test_no_bundled_tools_leaves_path_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "program_dir", lambda: tmp_path)
    monkeypatch.setenv("PATH", "C:\\only")
    assert paths.put_bundled_tools_first() == [] and os.environ["PATH"] == "C:\\only"


def _source(folder: Path, lock: str = "1") -> Path:
    (folder / "src").mkdir(parents=True)
    (folder / "src" / "index.ts").write_text("export {};")
    (folder / "scripts").mkdir()
    (folder / "scripts" / "render.mjs").write_text("// render")
    (folder / "package.json").write_text("{}")
    (folder / "package-lock.json").write_text(lock)
    (folder / "node_modules").mkdir()
    (folder / "test").mkdir()
    return folder


def test_remotion_source_is_copied_once_and_again_when_it_changes(tmp_path):
    source, target = _source(tmp_path / "shipped"), tmp_path / "prepared"
    assert remotion.sync_source(source, target) is True  # first copy: packages to install
    assert (target / "src" / "index.ts").is_file() and (target / "scripts" / "render.mjs").is_file()
    assert not (target / "node_modules").exists() and not (target / "test").exists()
    assert remotion.source_hash(target) == remotion.source_hash(source)
    assert remotion.sync_source(source, target) is False  # unchanged: nothing to do
    (target / "node_modules").mkdir()
    (source / "src" / "index.ts").write_text("export const changed = 1;")
    assert remotion.sync_source(source, target) is False  # code changed, same packages
    assert "changed" in (target / "src" / "index.ts").read_text() and (target / "node_modules").is_dir()
    (source / "package-lock.json").write_text("2")
    assert remotion.sync_source(source, target) is True  # new packages: install again


def test_remotion_source_is_not_copied_onto_itself(tmp_path):
    source = _source(tmp_path / "remotion")
    assert remotion.sync_source(source, source) is False
    assert not (source / ".source.json").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_console_windows_are_hidden_for_new_processes(monkeypatch):
    from app.utils import no_console

    seen = []
    original = subprocess.Popen.__init__

    def record(self, *args, **kwargs):
        seen.append(kwargs.get("creationflags"))
        raise OSError("not started")

    monkeypatch.setattr(subprocess.Popen, "__init__", record)
    monkeypatch.setattr(no_console, "_patched", False)
    assert no_console.hide_console_windows() is True
    for flags in (None, 0x10):
        with pytest.raises(OSError):
            subprocess.Popen(["x"], creationflags=flags) if flags else subprocess.Popen(["x"])
    assert seen == [no_console.CREATE_NO_WINDOW, 0x10]
    assert subprocess.Popen.__init__ is not original


def _launcher():
    spec = importlib.util.spec_from_file_location("launcher", ROOT / "launcher.pyw")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_opens_an_app_window_with_its_own_profile(tmp_path):
    launcher = _launcher()
    command = launcher.window_command("msedge.exe", "http://localhost:5050", tmp_path / "window")
    assert command[0] == "msedge.exe" and "--app=http://localhost:5050" in command
    assert f"--user-data-dir={tmp_path / 'window'}" in command
    assert not launcher.window_open(tmp_path / "no-such-profile")


def test_record_names_the_launcher(tmp_path):
    from app.instance import InstanceLock, read_record

    lock = InstanceLock(tmp_path)
    assert lock.acquire()
    try:
        lock.write_record(5050, script="launcher.pyw")
        assert read_record(tmp_path)["script"] == "launcher.pyw"
    finally:
        lock.release()


def test_storage_report_names_the_version(tmp_path):
    from app.config import Settings
    from app.services.storage import storage_report
    from app.version import __version__

    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_directories()
    report = storage_report(settings)
    assert report["version"] == __version__ and report["installed"] is False


def test_self_test_reports_each_check(monkeypatch):
    from app import selftest

    monkeypatch.setattr(selftest, "_app", lambda: "version x")
    monkeypatch.setattr(selftest, "_program", lambda name, args, first_line=None: f"{name} ok")
    out = io.StringIO()
    assert selftest.run(out) == 0
    text = out.getvalue()
    assert "All checks passed." in text and "ok    ffmpeg: ffmpeg ok" in text
    monkeypatch.setattr(selftest, "_voice", lambda: (_ for _ in ()).throw(RuntimeError("no espeak")))
    out = io.StringIO()
    assert selftest.run(out) == 1 and "FAIL  narration voice: no espeak" in out.getvalue()


def test_lockfile_pins_every_package_and_leaves_out_test_tools():
    lines = [line for line in (ROOT / "requirements-lock.txt").read_text().splitlines() if line and not line.startswith("#")]
    assert lines and all("==" in line for line in lines)
    names = {line.split("==")[0].lower() for line in lines}
    assert "pytest" not in names and "torch" not in names and "opencv-python" not in names
    assert {"flask", "faster-whisper", "kokoro-onnx", "opencv-python-headless", "onnxruntime"} <= names


def test_tools_file_is_valid():
    tools = json.loads((ROOT / "packaging" / "tools.json").read_text(encoding="utf-8"))
    assert tools["python"]["version"].startswith("3.12.") and tools["node"]["url"].startswith("https://nodejs.org/")


def test_stop_creates_nothing(installed):
    """The uninstaller runs "launcher.pyw --stop --quiet": it must not create the data folder."""
    launcher = _launcher()
    assert launcher.main(["--stop", "--quiet"]) == 0
    assert not (installed / "home").exists()
