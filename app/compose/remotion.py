"""Remotion: animated text, titles and captions drawn over a narrated video.

The ``remotion/`` package (next to ``app/``) holds one composition, ``Reel``, that draws a
render_spec. Preparing it is one button in Settings: install its packages (npm), build the
bundle (cached under data/cache, rebuilt only when the package's code changes) and download
the headless browser Remotion renders with. Nothing here downloads anything by itself.

Rendering is video only (the sound is mixed by FFmpeg afterwards) and done in chunks of about
ten seconds; finished chunks are kept with a fingerprint, so a paused or interrupted render
continues where it stopped. The base video is served to the renderer from a small local web
server that only this computer can reach and that only serves the render's own folder.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import http.server
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

from app import paths
from app.services.download_progress import transfer_fields
from app.utils.atomic import atomic_write_json, read_json
from app.utils.security import safe_path_argument
from app.video.ffmpeg_runner import CommandError, FFmpegRunner

# The shipped composition, and where it is prepared (the same folder in the repository; a copy
# in the user's folder once installed, see sync_source).
SOURCE = paths.remotion_source()
ROOT = paths.remotion_dir()
ENTRY = ROOT / "src" / "index.ts"
# What is copied from SOURCE to ROOT (never node_modules or tests).
_SOURCE_ITEMS = ("src", "scripts", "package.json", "package-lock.json", "tsconfig.json")
_SYNC_FILE = ".source.json"
COMPOSITION = "Reel"
MIN_NODE = (18, 0)
CHUNK_SECONDS = 10.0
CHUNKS_FILE = "chunks_done.json"
_BROWSER_MARKER = "remotion-browser.json"

_prepare_lock = threading.Lock()
_prepare_state: dict[str, Any] = {"state": "idle", "step": "", "error": None, "log": [], **transfer_fields()}


def node_executable() -> str | None:
    return shutil.which("node")


def npm_executable() -> str | None:
    return shutil.which("npm")


def node_version(runner: FFmpegRunner | None = None) -> tuple[int, int] | None:
    node = node_executable()
    if not node:
        return None
    try:
        code, lines, _ = (runner or FFmpegRunner()).run_streaming([node, "--version"], timeout=20)
    except (CommandError, OSError):
        return None
    text = (lines[0] if lines else "").strip().lstrip("v")
    try:
        major, minor = (int(part) for part in text.split(".")[:2])
    except ValueError:
        return None
    return (major, minor) if code == 0 else None


def packages_installed() -> bool:
    return (ROOT / "node_modules" / "@remotion" / "renderer" / "package.json").is_file()


def source_hash(root: Path | None = None) -> str:
    """Fingerprint of the composition's code and package versions (the bundle's cache key).
    Always of the shipped code, so an update that changes it makes the tools need preparing."""
    root = root or SOURCE
    digest = hashlib.sha256()
    files = sorted(p for p in (root / "src").rglob("*") if p.is_file())
    for path in [*files, root / "package.json", root / "package-lock.json"]:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else ""


def sync_source(source: Path | None = None, target: Path | None = None) -> bool:
    """Copy the shipped composition to where it is prepared, when they differ (installed copy
    only; in the repository both are the same folder). Returns True when the package list
    changed, so the packages must be installed again."""
    source, target = source or SOURCE, target or ROOT
    if source.resolve() == target.resolve():
        return False
    wanted = source_hash(source)
    state = read_json(target / _SYNC_FILE, {})
    state = state if isinstance(state, dict) else {}
    lock = _file_hash(source / "package-lock.json")
    if state.get("hash") == wanted and (target / "src").is_dir():
        return False
    target.mkdir(parents=True, exist_ok=True)
    for name in _SOURCE_ITEMS:
        item, copy = source / name, target / name
        if item.is_dir():
            shutil.rmtree(copy, ignore_errors=True)
            shutil.copytree(item, copy)
        elif item.is_file():
            shutil.copy2(item, copy)
    atomic_write_json(target / _SYNC_FILE, {"hash": wanted, "lock": lock})
    return state.get("lock") != lock


def bundle_dir(settings) -> Path:
    return Path(settings.cache_dir) / "remotion-bundle" / source_hash()


def bundle_ready(settings) -> bool:
    return (bundle_dir(settings) / "index.html").is_file()


def browser_ready(settings) -> bool:
    marker = read_json(Path(settings.cache_dir) / _BROWSER_MARKER, None)
    return isinstance(marker, dict) and bool(marker.get("ok")) and packages_installed()


def status(settings) -> dict[str, Any]:
    version = node_version()
    installed = packages_installed()
    state = dict(_prepare_state)
    result = {
        "node": ".".join(str(n) for n in version) if version else None,
        "node_ok": bool(version and version >= MIN_NODE),
        "installed": installed,
        "bundle": installed and bundle_ready(settings),
        "browser": browser_ready(settings),
        "prepare": state,
    }
    result["ready"] = bool(result["node_ok"] and result["installed"] and result["bundle"] and result["browser"])
    return result


def ready(settings) -> bool:
    return bool(packages_installed() and bundle_ready(settings) and browser_ready(settings) and node_executable())


# Preparing the tools (Settings → Prepare video tools)


def start_prepare(settings) -> dict[str, Any]:
    with _prepare_lock:
        if _prepare_state.get("state") == "running":
            return dict(_prepare_state)
        _prepare_state.update({"state": "running", "step": "starting", "error": None, "log": [], **transfer_fields(phase="starting")})
    threading.Thread(target=_prepare, args=(settings,), daemon=True, name="remotion-prepare").start()
    return dict(_prepare_state)


def _note(step: str, line: str = "") -> None:
    with _prepare_lock:
        _prepare_state["step"] = step
        if line:
            _prepare_state["log"] = [*_prepare_state.get("log", []), line[:300]][-30:]


def _prepare(settings) -> None:
    runner = FFmpegRunner()
    try:
        version = node_version(runner)
        if not version or version < MIN_NODE:
            raise CommandError("Node.js 18 or newer is needed. Install it from nodejs.org, then try again.")
        packages_changed = sync_source()
        if packages_changed or not packages_installed():
            npm = npm_executable()
            if not npm:
                raise CommandError("npm was not found. It comes with Node.js; reinstall Node.js.")
            _note("install", "Installing the video tools' packages (one time, about 200 MB)…")
            command = [npm, "ci" if (ROOT / "package-lock.json").is_file() else "install", "--no-audit", "--no-fund"]
            code, lines, errors = runner.run_streaming(command, cwd=ROOT, timeout=1800, on_line=lambda l: _note("install", l))
            if code != 0:
                raise CommandError(f"Installing the packages failed: {(errors or chr(10).join(lines[-5:]))[-400:]}")
        node = node_executable() or "node"
        if not bundle_ready(settings):
            _note("bundle", "Building the composition…")
            out = bundle_dir(settings)
            out.parent.mkdir(parents=True, exist_ok=True)
            code, lines, errors = runner.run_streaming(
                [node, str(ROOT / "scripts" / "bundle.mjs"), str(ENTRY), str(out)],
                cwd=ROOT,
                timeout=900,
                on_line=lambda l: _note("bundle", l),
            )
            if code != 0 or not bundle_ready(settings):
                raise CommandError(f"Building the composition failed: {(errors or chr(10).join(lines[-5:]))[-400:]}")
            for old in out.parent.iterdir():
                if old.is_dir() and old != out:
                    shutil.rmtree(old, ignore_errors=True)
        if not browser_ready(settings):
            _note("browser", "Downloading the headless browser Remotion draws with (one time)…")
            code, lines, errors = runner.run_streaming(
                [node, str(ROOT / "scripts" / "browser.mjs")], cwd=ROOT, timeout=1800, on_line=lambda l: _note("browser", l)
            )
            if code != 0:
                raise CommandError(f"Downloading the browser failed: {(errors or chr(10).join(lines[-5:]))[-400:]}")
            atomic_write_json(Path(settings.cache_dir) / _BROWSER_MARKER, {"ok": True, "hash": source_hash()})
        with _prepare_lock:
            _prepare_state.update({"state": "done", "step": "done", "error": None, **transfer_fields(phase="done")})
    except Exception as exc:  # noqa: BLE001
        with _prepare_lock:
            _prepare_state.update({"state": "error", "error": str(exc)[:500], **transfer_fields(phase="error")})


# Rendering


def concurrency(settings) -> int:
    chosen = int(getattr(settings, "remotion_concurrency", 0) or 0)
    if chosen > 0:
        return min(16, chosen)
    return max(1, min(6, (os.cpu_count() or 4) // 2))


def _error_event(lines: list[str]) -> dict | None:
    """The last {"type": "error"} line a script printed, if any."""
    for line in reversed(lines):
        if line.startswith('{"type":"error"'):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return {"message": line[:400]}
            return event if isinstance(event, dict) else {"message": line[:400]}
    return None


@contextlib.contextmanager
def serve_folder(folder: Path) -> Iterator[str]:
    """A web server on 127.0.0.1 (a free port) serving only ``folder``, while the block runs."""

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:  # noqa: D401 - silence per-request logging
            return

    handler = functools.partial(Quiet, directory=str(folder))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="remotion-assets")
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def render_picture(
    spec: dict,
    work_dir: Path,
    settings,
    runner: FFmpegRunner,
    *,
    on_progress: Callable[[float], None] | None = None,
    log: Callable[[str, str], None] | None = None,
) -> Path:
    """Draw ``spec`` (whose video src is a file name inside ``work_dir``) as a silent video.
    Returns its path. Raises CommandError when Remotion cannot render."""
    log = log or (lambda message, level="INFO": None)
    node = node_executable()
    if not node or not ready(settings):
        raise CommandError("The Remotion video tools are not prepared (Settings → Video tools).")
    fps = float(spec["fps"])
    total = max(1, int(round(float(spec["duration"]) * fps)))
    per_chunk = max(1, int(round(CHUNK_SECONDS * fps)))
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    done = read_json(chunks_dir / CHUNKS_FILE, {})
    done = done if isinstance(done, dict) else {}
    bundle = bundle_dir(settings)
    workers = concurrency(settings)
    spec_text = json.dumps(spec, sort_keys=True)
    ranges = [(start, min(total, start + per_chunk) - 1) for start in range(0, total, per_chunk)]
    paths: list[Path] = []
    with serve_folder(work_dir) as base_url:
        served = json.loads(spec_text)
        if served["video"]["kind"] == "base" and served["video"]["src"]:
            served["video"]["src"] = f"{base_url}/{served['video']['src']}"
        for number, (first, last) in enumerate(ranges):
            out = chunks_dir / f"chunk_{number:04d}.mp4"
            fingerprint = hashlib.sha256(f"{spec_text}|{bundle.name}|{first}-{last}".encode()).hexdigest()[:24]
            if done.get(out.name) == fingerprint and out.is_file() and out.stat().st_size > 0:
                paths.append(out)
                if on_progress:
                    on_progress((last + 1) / total)
                continue
            job_file = chunks_dir / f"chunk_{number:04d}.json"
            atomic_write_json(
                job_file,
                {
                    "bundle": str(bundle),
                    "composition": COMPOSITION,
                    "props": served,
                    "frameRange": [first, last],
                    "output": str(out.with_suffix(".part.mp4")),
                    "concurrency": workers,
                    "crf": 16,
                },
            )

            def report(line: str, first: int = first, last: int = last) -> None:
                if not line.startswith("{"):
                    return
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    return
                if event.get("type") == "progress" and on_progress:
                    share = float(event.get("value") or 0)
                    on_progress((first + share * (last - first + 1)) / total)

            code, lines, errors = runner.run_streaming(
                [node, str(ROOT / "scripts" / "render.mjs"), str(job_file)], cwd=ROOT, on_line=report, timeout=3600
            )
            failure = _error_event(lines)
            if code != 0 or failure:
                message = (failure or {}).get("message") or errors[-400:] or f"exit code {code}"
                raise CommandError(f"Remotion could not draw the video: {message}")
            out.with_suffix(".part.mp4").replace(out)
            done[out.name] = fingerprint
            atomic_write_json(chunks_dir / CHUNKS_FILE, done)
            job_file.unlink(missing_ok=True)
            paths.append(out)
    picture = work_dir / "picture.mp4"
    listing = chunks_dir / "chunks.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in paths), encoding="utf-8")
    runner.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            safe_path_argument(listing),
            "-c",
            "copy",
            "-an",
            safe_path_argument(picture),
        ]
    )
    log(f"Remotion drew {total} frames in {len(ranges)} part(s) with {workers} worker(s)", "INFO")
    return picture
