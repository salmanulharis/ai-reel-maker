"""Start the installed AI Reel Maker in its own window. Closing the window stops the app.

The Start-menu shortcut runs this with ``pythonw.exe`` (no console). It puts the bundled FFmpeg
and Node first on PATH, writes everything the app prints to ``logs\\app.log``, starts the
server, and opens the app in an Edge (or Chrome) app window with its own profile. When the last
window of that profile closes, the app stops with the programs it started; a job that was
running continues where it stopped the next time the app starts.

``launcher.pyw --stop`` (the "Stop AI Reel Maker" shortcut) stops a running copy.

Starting it again while it runs only opens another window. Without Edge or Chrome the default
browser is used and the app runs until "Stop AI Reel Maker" is chosen.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app import paths  # noqa: E402

SCRIPT = "launcher.pyw"
LOG_LIMIT = 5 * 1024 * 1024
# How often to look whether the app window is still open.
WINDOW_POLL_SECONDS = 2.0


def open_log() -> Path:
    """Send print() and errors to logs/app.log (pythonw has nowhere to show them)."""
    folder = paths.logs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "app.log"
    try:
        if log.is_file() and log.stat().st_size > LOG_LIMIT:
            os.replace(log, log.with_suffix(".log.1"))
    except OSError:
        pass
    stream = open(log, "a", encoding="utf-8", errors="replace", buffering=1)  # noqa: SIM115 - open while the app runs
    sys.stdout = sys.stderr = stream
    print(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting AI Reel Maker from {ROOT}")
    return log


def message(text: str) -> None:
    """A Windows message box (the app has no console)."""
    print(text)
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, paths.APP_NAME, 0x40)
    except Exception:  # noqa: BLE001 - not on Windows, or no desktop
        pass


def find_browser() -> str | None:
    """Edge (always on Windows 11), else Chrome: both open an app window with --app."""
    candidates = []
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidates += [
                Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe",
            ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def window_profile() -> Path:
    return paths.user_home() / "window"


def window_command(browser: str, url: str, profile: Path) -> list[str]:
    return [
        browser,
        f"--app={url}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-mode",
        "--window-size=1440,920",
    ]


def window_open(profile: Path) -> bool:
    """Whether a browser process with the app's own profile is still running."""
    import psutil

    marker = f"--user-data-dir={profile}".lower()
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            if any(part.lower() == marker for part in process.info.get("cmdline") or []):
                return True
        except Exception:  # noqa: BLE001 - gone or not readable
            continue
    return False


def open_window(url: str) -> bool:
    """Open the app window. True when it can be watched (an app window was opened)."""
    import subprocess
    import webbrowser

    browser = find_browser()
    if not browser:
        webbrowser.open(url)
        return False
    profile = window_profile()
    profile.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(window_command(browser, url, profile))  # noqa: S603 - fixed program and arguments
    return True


def wait_for_window_to_close() -> None:
    profile = window_profile()
    # The browser takes a moment to start; then wait while any process of the profile runs.
    deadline = time.monotonic() + 30
    while not window_open(profile) and time.monotonic() < deadline:
        time.sleep(0.5)
    while window_open(profile):
        time.sleep(WINDOW_POLL_SECONDS)


def stop_children() -> None:
    """End every program the app started (FFmpeg, node), so nothing is left running."""
    import psutil

    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass


def stop(quiet: bool = False) -> int:
    """The "Stop AI Reel Maker" shortcut (``--quiet``: no message, used by the installer).
    Only reads the running copy's record: it creates no folders or log (the uninstaller runs it)."""
    from app.config import Settings
    from app.instance import stop_app

    outcome = stop_app(Settings().data_dir)
    if quiet:
        return 0
    message(
        "AI Reel Maker stopped. A job that was running continues where it stopped next time."
        if outcome == "stopped"
        else "AI Reel Maker was not running."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    from app.utils.no_console import hide_console_windows

    args = sys.argv[1:] if argv is None else argv
    if "--stop" in args:
        return stop(quiet="--quiet" in args)
    open_log()
    hide_console_windows()
    paths.put_bundled_tools_first()

    from app.config import get_settings
    from app.instance import InstanceLock, local_url, read_record, wait_until_ready

    settings = get_settings()
    settings.ensure_directories()
    lock = InstanceLock(settings.data_dir)
    if not lock.acquire():
        record = read_record(settings.data_dir) or {}
        port = int(record.get("port") or settings.app_port)
        if wait_until_ready(port):
            open_window(local_url(port))
        else:
            message("AI Reel Maker is starting or not answering. Try again in a moment.")
        return 0

    failure: list[BaseException] = []
    try:
        lock.write_record(settings.app_port, script=SCRIPT)
        from app import create_app, socketio

        application = create_app(settings)

        def serve() -> None:
            try:
                socketio.run(application, host=settings.app_host, port=settings.app_port, allow_unsafe_werkzeug=True)
            except BaseException as exc:  # noqa: BLE001 - reported by the main thread
                failure.append(exc)

        server = threading.Thread(target=serve, daemon=True, name="server")
        server.start()
        deadline = time.monotonic() + 90
        while not failure and time.monotonic() < deadline:
            if wait_until_ready(settings.app_port, timeout=1):
                break
        if failure or not wait_until_ready(settings.app_port, timeout=1):
            reason = f": {failure[0]}" if failure else "."
            message(
                f"AI Reel Maker could not start on port {settings.app_port}{reason}\n\n"
                f"Another program may be using that port. Details are in {paths.logs_dir() / 'app.log'}"
            )
            return 1
        url = local_url(settings.app_port)
        print(f"AI Reel Maker is running at {url}")
        if open_window(url):
            wait_for_window_to_close()
            print("The app window was closed; stopping.")
        else:
            server.join()
    except Exception as exc:  # noqa: BLE001 - shown to the user, details in the log
        import traceback

        traceback.print_exc()
        message(f"AI Reel Maker stopped with an error: {exc}\n\nDetails are in {paths.logs_dir() / 'app.log'}")
        return 1
    finally:
        stop_children()
        lock.release()
    print("AI Reel Maker stopped.")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # The server thread does not stop by itself; end the process (the worker's jobs continue
    # where they stopped next time).
    os._exit(code)
