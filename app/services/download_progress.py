"""Byte progress, speed and time left for model downloads.

Every download status shares these fields: ``bytes_done``, ``bytes_total`` (0 when unknown),
``speed`` (bytes per second, smoothed), ``eta`` (seconds, or None) and ``phase``.

Hugging Face downloads (Whisper, picture and sound models) do not report progress through a
stable public API, so ``HubWatcher`` measures the files growing in the Hugging Face cache
instead. That works with every huggingface_hub version and download backend.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

_log = logging.getLogger(__name__)

# Share of a new speed sample in the smoothed speed.
_SPEED_SMOOTHING = 0.3
# A speed is only reported after this long, so the first burst does not promise a quick finish.
_SPEED_MIN_SECONDS = 1.5
WATCH_INTERVAL = 0.5


class RateMeter:
    """Smoothed transfer speed and time left from successive byte counts."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.begun = clock()
        self._last: tuple[float, float] | None = None
        self.speed = 0.0

    def update(self, done: float) -> None:
        now = self.clock()
        if self._last is not None:
            then, before = self._last
            if now - then >= 0.2 and done >= before:
                sample = (done - before) / (now - then)
                self.speed = sample if self.speed <= 0 else self.speed + _SPEED_SMOOTHING * (sample - self.speed)
            elif now - then < 0.2:
                return
        self._last = (now, done)

    def fields(self, done: float, total: float) -> dict[str, Any]:
        self.update(done)
        steady = self.clock() - self.begun >= _SPEED_MIN_SECONDS and self.speed > 0
        eta = None
        if steady and total > 0 and done < total:
            eta = round((total - done) / self.speed, 1)
        return {
            "bytes_done": int(done),
            "bytes_total": int(total),
            "speed": round(self.speed, 1) if steady else 0.0,
            "eta": eta,
        }


def transfer_fields(done: float = 0, total: float = 0, phase: str = "") -> dict[str, Any]:
    """The shared fields for a status with no speed (starting, finished, failed)."""
    return {"bytes_done": int(done), "bytes_total": int(total), "speed": 0.0, "eta": None, "phase": phase}


def hub_cache_folder(repo_id: str) -> Path | None:
    """The Hugging Face cache folder of a model repository, or None without huggingface_hub."""
    try:
        from huggingface_hub import constants
        from huggingface_hub.file_download import repo_folder_name
    except ImportError:
        return None
    return Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id=repo_id, repo_type="model")


def hub_missing_bytes(repo_id: str, files: Iterable[str] = (), patterns: Iterable[str] = ()) -> int:
    """Total size of the files of ``repo_id`` that still have to be downloaded: the ``files``
    named, or those matching ``patterns``. 0 when it cannot be found out (shown as unknown)."""
    try:
        from huggingface_hub import HfApi, try_to_load_from_cache

        names = list(files)
        sizes: dict[str, int] = {}
        info = HfApi().model_info(repo_id, files_metadata=True)
        for sibling in info.siblings or []:
            name = sibling.rfilename
            wanted = name in names if names else any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
            if wanted and sibling.size:
                sizes[name] = int(sibling.size)
        missing = 0
        for name, size in sizes.items():
            cached = try_to_load_from_cache(repo_id, name)
            if not (isinstance(cached, str) and Path(cached).is_file()):
                missing += size
        return missing
    except Exception:  # noqa: BLE001 - offline or an old huggingface_hub: the size stays unknown
        _log.debug("Could not read the download size of %s", repo_id, exc_info=True)
        return 0


class HubWatcher:
    """Counts the bytes written to a repository's cache folder while a download runs and passes
    the shared progress fields to ``on_update`` twice a second. Files that were already there
    when the watch began are not counted."""

    def __init__(self, repo_id: str, total: int, on_update: Callable[[dict[str, Any]], None]):
        self.folder = hub_cache_folder(repo_id)
        self.total = max(0, int(total))
        self.on_update = on_update
        self.meter = RateMeter()
        self.stop = threading.Event()
        self.baseline = self._sizes()
        self.thread = threading.Thread(target=self._run, daemon=True, name="download-progress")

    def __enter__(self) -> "HubWatcher":
        self.on_update({**self.meter.fields(0, self.total), "phase": "downloading"})
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def written(self) -> int:
        """Bytes written since the watch began."""
        now = self._sizes()
        done = 0
        for path, size in now.items():
            before = self.baseline.get(path, 0)
            if size > before:
                done += size - before
        return min(done, self.total) if self.total else done

    def _sizes(self) -> dict[str, int]:
        if self.folder is None or not self.folder.is_dir():
            return {}
        sizes: dict[str, int] = {}
        # Downloads are written into blobs/ (as *.incomplete while running); snapshots only link.
        for path in (self.folder / "blobs").glob("*"):
            try:
                sizes[path.name.removesuffix(".incomplete")] = max(
                    sizes.get(path.name.removesuffix(".incomplete"), 0), path.stat().st_size
                )
            except OSError:
                continue
        return sizes

    def _run(self) -> None:
        while not self.stop.wait(WATCH_INTERVAL):
            try:
                self.on_update({**self.meter.fields(self.written(), self.total), "phase": "downloading"})
            except Exception:  # noqa: BLE001 - reporting never disturbs the download
                _log.debug("Download progress update failed", exc_info=True)


class LayeredProgress:
    """Ollama reports a download per layer, one after another. This adds the layers up, so the
    bar runs once from 0 to 100% instead of restarting for every layer."""

    def __init__(self):
        self.layers: dict[str, tuple[int, int]] = {}
        self.meter = RateMeter()

    def update(self, status: str, digest: str, completed: int, total: int) -> dict[str, Any]:
        if digest and total:
            self.layers[digest] = (int(completed or 0), int(total))
        done = sum(completed for completed, _ in self.layers.values())
        size = sum(total for _, total in self.layers.values())
        fields = self.meter.fields(done, size)
        fields["phase"] = ollama_phase(status)
        return fields


def ollama_phase(status: str) -> str:
    text = (status or "").lower()
    if text.startswith("pulling manifest") or not text:
        return "starting"
    if text.startswith("pulling") or text.startswith("downloading"):
        return "downloading"
    if text.startswith("verifying"):
        return "verifying"
    if text == "success":
        return "done"
    return "finishing"
