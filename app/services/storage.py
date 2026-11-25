from __future__ import annotations

import shutil
from pathlib import Path

from app import paths
from app.config import Settings, get_settings
from app.version import __version__


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def storage_report(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    jobs = settings.jobs_dir
    tmp = settings.tmp_dir
    cache = settings.cache_dir
    outputs = 0
    for job in jobs.glob("*"):
        out = job / "outputs"
        outputs += dir_size(out)
    return {
        "jobs_bytes": dir_size(jobs),
        "tmp_bytes": dir_size(tmp),
        "cache_bytes": dir_size(cache),
        "outputs_bytes": outputs,
        "total_bytes": dir_size(settings.data_dir),
        "data_dir": str(settings.data_dir),
        "version": __version__,
        "installed": paths.is_installed(),
    }


def clear_tmp(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    _clear_dir(settings.tmp_dir)


# Prepared tools kept in the cache folder: not analysis cache, so "clear cache" keeps them
# (removing them would silently switch narrated videos to the FFmpeg drawing).
_TOOLS = {"remotion-bundle", "remotion-browser.json"}


def clear_cache(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    _clear_dir(settings.cache_dir, keep=_TOOLS)


def delete_job_files(job_id: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    path = settings.jobs_dir / job_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _clear_dir(path: Path, keep: set[str] | None = None) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.name == ".gitkeep" or child.name in (keep or set()):
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass
