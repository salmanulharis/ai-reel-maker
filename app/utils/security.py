from __future__ import annotations

import os
import re
from pathlib import Path, PurePath


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ALLOWED_MIME_PREFIXES = ("video/",)
MAX_FILENAME_LEN = 180


_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def valid_job_id(value: str | None) -> bool:
    """Job ids are short hex strings; anything with dots or slashes (either kind) is refused so a
    URL can never point outside the jobs folder."""
    return bool(value) and bool(_JOB_ID.fullmatch(str(value)))


def sanitize_filename(name: str) -> str:
    base = PurePath(name).name
    base = re.sub(r"[^\w.\- ]+", "_", base, flags=re.UNICODE)
    base = base.strip(" .") or "video"
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".mp4"
    stem = stem[: MAX_FILENAME_LEN - len(ext)] or "video"
    return stem + ext


def is_allowed_extension(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTENSIONS


def safe_path_argument(path: str | Path) -> str:
    """Return a filesystem path as a single argv token. Never for shell interpolation."""
    return os.fspath(path)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_job_relative(root: Path, *parts: str) -> Path:
    candidate = (root.joinpath(*parts)).resolve()
    if not is_relative_to(candidate, root):
        raise ValueError("Path escapes job directory")
    return candidate
