"""Small ONNX models downloaded from Hugging Face on request (picture and sound matching).

Each model is a set of files in one repository. They are downloaded in the background from
Settings, found again in the Hugging Face cache, and run with onnxruntime: on a GPU when an
onnxruntime build with CUDA or DirectML is installed, otherwise (or when the GPU fails to
start) on the CPU.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.services.download_progress import HubWatcher, hub_missing_bytes, transfer_fields

_log = logging.getLogger(__name__)

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def runtime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401 - only checks that it is installed
        import tokenizers  # noqa: F401 - only checks that it is installed
    except ImportError:
        return False
    return True


def providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    return [name for name in GPU_PROVIDERS if name in available] + ["CPUExecutionProvider"]


def session(path: str, threads: int, label: str = "The model"):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    chosen = providers()
    try:
        return ort.InferenceSession(path, options, providers=chosen)
    except Exception:  # noqa: BLE001 - a GPU provider that cannot start falls back to the CPU
        if chosen == ["CPUExecutionProvider"]:
            raise
        _log.warning("%s could not use the graphics card; using the processor", label)
        return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def on_gpu(onnx_session) -> bool:
    return any(name in GPU_PROVIDERS for name in onnx_session.get_providers())


class HubModel:
    """The files of one model on Hugging Face, and their background download."""

    def __init__(self, repo: str, files: tuple[str, ...], size_mb: int, label: str):
        self.repo = repo
        self.files = files
        self.size_mb = size_mb
        self.label = label
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "state": "idle",
            "files_done": 0,
            "files_total": len(files),
            "error": None,
            **transfer_fields(),
        }

    def cached_path(self, filename: str) -> str | None:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return None
        path = try_to_load_from_cache(self.repo, filename)
        return path if isinstance(path, str) and Path(path).is_file() else None

    def installed(self) -> bool:
        return all(self.cached_path(name) for name in self.files)

    def status(self, installed: bool, runtime: bool) -> dict[str, Any]:
        with self.lock:
            state = dict(self.state)
        state["installed"] = installed
        state["size_mb"] = self.size_mb
        state["runtime"] = runtime
        return state

    def begin(self) -> bool:
        """Mark a download as running; False when one already is."""
        with self.lock:
            if self.state.get("state") == "running":
                return False
            self.state.update({"state": "running", "files_done": 0, "error": None, **transfer_fields(phase="starting")})
            return True

    def download(self) -> None:
        def update(fields: dict[str, Any]) -> None:
            with self.lock:
                self.state.update(fields)

        try:
            from huggingface_hub import hf_hub_download

            total = hub_missing_bytes(self.repo, files=self.files)
            with HubWatcher(self.repo, total, update) as watcher:
                for index, name in enumerate(self.files, start=1):
                    hf_hub_download(self.repo, name)
                    with self.lock:
                        self.state["files_done"] = index
                written = watcher.written()
            result = {"state": "done", "error": None, **transfer_fields(written, total or written, "done")}
        except Exception as exc:  # noqa: BLE001
            result = {
                "state": "error",
                "error": f"Could not download the {self.label}: {exc}"[:300],
                **transfer_fields(phase="error"),
            }
        update(result)
