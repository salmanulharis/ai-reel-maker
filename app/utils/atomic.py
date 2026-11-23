"""Writes that survive a sudden power cut.

A file is first written next to its final name, forced to disk, and then renamed over the final
name. A crash at any moment leaves either the previous complete file or the new complete file,
never a half-written one that a later run would read as finished.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def part_path(path: Path) -> Path:
    """Where a file is written before it is complete. The extension stays last, so FFmpeg still
    picks the right format; the leading dot keeps it out of lookups by name."""
    path = Path(path)
    return path.with_name(f".{path.stem}.part{path.suffix}")


def replace_file(source: Path, dest: Path, attempts: int = 10) -> None:
    """Move ``source`` over ``dest``. On Windows the old file can be briefly held open (for
    example while it is being streamed or scanned), so a few retries are allowed."""
    for attempt in range(attempts):
        try:
            os.replace(source, dest)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)


def sync_file(path: Path) -> None:
    """Force a finished file's contents to disk (best effort)."""
    try:
        with open(path, "rb+") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(temp, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any, **dumps_options: Any) -> None:
    atomic_write_text(path, json.dumps(data, **dumps_options))


def read_json(path: Path, default: Any = None) -> Any:
    """A JSON file's contents, or ``default`` when it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
