"""
Watch folder: videos dropped into a chosen folder are made into highlights automatically, with a
saved preset, and the finished videos are copied to a "highlights" folder inside it.

A file is picked up once its size has stopped changing (it has finished copying). Each file is
processed once; the list of handled files is kept in data/watch_state.json.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path

from app.models.schemas import HighlightConfig
from app.services.presets import list_presets
from app.services.settings_store import read_setting, write_setting
from app.utils.atomic import atomic_write_json
from app.utils.security import is_allowed_extension, sanitize_filename

_log = logging.getLogger(__name__)
SETTING_KEY = "watch_folder"
OUTPUT_FOLDER = "highlights"
POLL_SECONDS = 10


def read_config() -> dict:
    try:
        data = json.loads(read_setting(SETTING_KEY) or "{}")
    except json.JSONDecodeError:
        data = {}
    return {
        "enabled": bool(data.get("enabled")),
        "folder": str(data.get("folder") or ""),
        "preset": str(data.get("preset") or ""),
    }


def save_config(enabled, folder, preset) -> dict:
    folder = str(folder or "").strip()
    preset = str(preset or "").strip()
    if enabled:
        if not folder:
            raise ValueError("choose a folder to watch")
        path = Path(folder).expanduser()
        if not path.is_dir():
            raise ValueError("that folder does not exist")
        folder = str(path.resolve())
        if preset and not any(p["name"] == preset for p in list_presets()):
            raise ValueError("that preset does not exist")
    config = {"enabled": bool(enabled), "folder": folder, "preset": preset}
    write_setting(SETTING_KEY, json.dumps(config))
    return config


def preset_config(name: str) -> HighlightConfig:
    """A highlight config from a saved preset (Create-page form values), or the defaults."""
    form = next((p["form"] for p in list_presets() if p["name"] == name), {}) if name else {}
    values = {key: value for key, value in form.items() if key in HighlightConfig.model_fields}
    try:
        config = HighlightConfig.model_validate(values)
    except Exception:  # noqa: BLE001 - a broken preset falls back to the defaults
        config = HighlightConfig()
    return config.model_copy(update={"auto_render": True})


class WatchState:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            self.data = {}
        self.sizes: dict[str, int] = {}

    def seen(self, file: Path, stat) -> bool:
        return self.data.get(str(file)) == f"{stat.st_size}:{int(stat.st_mtime)}"

    def mark(self, file: Path, stat, job_id: str) -> None:
        self.data[str(file)] = f"{stat.st_size}:{int(stat.st_mtime)}"
        self.data[f"job:{job_id}"] = str(file)
        atomic_write_json(self.path, self.data, indent=1)

    def ready(self, file: Path, size: int) -> bool:
        """True once the size is the same as on the previous look (copying has finished)."""
        key = str(file)
        previous = self.sizes.get(key)
        self.sizes[key] = size
        return previous == size and size > 0


def scan_once(manager, state: WatchState) -> list[str]:
    """Create jobs for new, finished files in the watched folder. Returns the new job ids."""
    config = read_config()
    if not config["enabled"] or not config["folder"]:
        return []
    folder = Path(config["folder"])
    if not folder.is_dir():
        return []
    created = []
    for file in sorted(folder.iterdir()):
        if not file.is_file() or not is_allowed_extension(file.name):
            continue
        stat = file.stat()
        if state.seen(file, stat) or not state.ready(file, stat.st_size):
            continue
        try:
            job = create_job_from_file(manager, file, preset_config(config["preset"]))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the watcher
            _log.warning("Watch folder: could not start %s: %s", file.name, exc)
            continue
        state.mark(file, stat, job.id)
        created.append(job.id)
    return created


def create_job_from_file(manager, file: Path, config: HighlightConfig):
    """A queued job made from a copy of `file` (the original stays where it is)."""
    from app.video.ffmpeg_runner import FFmpegRunner
    from app.video.quality import source_frame_rate
    from app.models.schemas import VideoMetadata

    job_id = uuid.uuid4().hex[:12]
    job_dir = manager.settings.jobs_dir / job_id
    source_dir = job_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    filename = sanitize_filename(file.name)
    dest = source_dir / f"v0_{filename}"
    shutil.copy2(file, dest)
    try:
        meta = FFmpegRunner().probe(dest)
    except Exception:  # noqa: BLE001
        meta = VideoMetadata(filename=filename, size_bytes=dest.stat().st_size)
    entry = {
        "id": "v0",
        "index": 0,
        "filename": filename,
        "path": str(dest.resolve()),
        "analysis_path": str(dest.resolve()),
        "duration": meta.duration,
        "width": meta.width,
        "height": meta.height,
        "fps": source_frame_rate(meta),
        "size_bytes": meta.size_bytes or dest.stat().st_size,
        "watched_from": str(file),
    }
    return manager._create_record(job_id, job_dir, [entry], meta, float(meta.duration or 0), config, "queued")


def copy_outputs(job_dir: Path, outputs: list[Path]) -> list[Path]:
    """Copy a watched job's finished videos to <watched folder>/highlights."""
    from app.video.sources import load_sources

    sources = load_sources(job_dir)
    origin = next((s.get("watched_from") for s in sources if s.get("watched_from")), None)
    if not origin:
        return []
    target = Path(origin).parent / OUTPUT_FOLDER
    target.mkdir(parents=True, exist_ok=True)
    stem = Path(origin).stem
    copied = []
    for path in outputs:
        name = f"{stem}-{path.stem}{path.suffix}" if path.stem != "highlight" else f"{stem}-highlight{path.suffix}"
        dest = target / name
        shutil.copy2(path, dest)
        copied.append(dest)
    return copied


def start_watcher(app) -> threading.Thread:
    from app.services.job_manager import get_job_manager

    def loop() -> None:
        with app.app_context():
            manager = get_job_manager()
            state = WatchState(manager.settings.data_dir / "watch_state.json")
            while True:
                try:
                    created = scan_once(manager, state)
                    if created:
                        _log.info("Watch folder: started %s job(s)", len(created))
                except Exception:  # noqa: BLE001
                    _log.exception("Watch folder scan failed")
                time.sleep(POLL_SECONDS)

    thread = threading.Thread(target=loop, name="watch-folder", daemon=True)
    thread.start()
    return thread
