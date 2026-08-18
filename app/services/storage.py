from __future__ import annotations

import shutil
from pathlib import Path

from app.config import Settings, get_settings


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
    }


def clear_tmp(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    _clear_dir(settings.tmp_dir)


def clear_cache(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    _clear_dir(settings.cache_dir)


def delete_job_files(job_id: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    path = settings.jobs_dir / job_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass
