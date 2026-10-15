"""Fast picture matching with SigLIP (ONNX), a first filter before the slow vision model.

SigLIP turns a frame and a short description into vectors whose similarity says how well
they match. It cannot judge what happens in a scene, so it is used only to rank and filter:
which scenes deserve the vision model, which scenes look alike, which match the user's focus,
and a better quick score when the vision model is off or skipped a scene.

Runs on the CPU with onnxruntime (already installed with faster-whisper), or on a GPU when an
onnxruntime build with CUDA or DirectML is installed; a GPU that fails falls back to the CPU.
Measured on a 6-core CPU: about 0.34 s per frame, about 0.1 s per description. The full
precision model is used: the 8-bit one ranked frames noticeably differently (Spearman
0.56-0.94 against full precision).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

MODEL_REPO = "Xenova/siglip-base-patch16-224"  # ONNX export of google/siglip-base-patch16-224 (Apache-2.0)
MODEL_FILES = (
    "tokenizer.json",
    "onnx/text_model.onnx",
    "onnx/vision_model.onnx",
)
MODEL_DOWNLOAD_MB = 780
_IMAGE_SIZE = 224
_TEXT_LENGTH = 64  # SigLIP was trained with text padded to exactly 64 tokens.
_IMAGE_BATCH = 8

# What makes a frame worth a closer look, and what makes it unlikely to be a highlight.
POSITIVE_PROMPTS = (
    "an exciting action moment",
    "people laughing or celebrating",
    "a person speaking expressively to the camera",
    "a product being demonstrated up close",
    "a dramatic or beautiful scene",
    "fast movement or sports",
    "a surprising moment",
)
NEGATIVE_PROMPTS = (
    "a black screen",
    "a blurry out of focus frame",
    "an empty room with nobody in it",
    "a title card or logo",
    "an end screen asking viewers to subscribe",
)
# Frames this similar (cosine) show essentially the same picture.
DUPLICATE_SIMILARITY = 0.95
_GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")

_lock = threading.Lock()
_matcher: "ImageMatcher | None" = None
_download_state: dict[str, Any] = {"state": "idle", "files_done": 0, "files_total": len(MODEL_FILES), "error": None}


# Model files


def _cached_path(filename: str) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(MODEL_REPO, filename)
    return path if isinstance(path, str) and Path(path).is_file() else None


def model_installed() -> bool:
    return all(_cached_path(name) for name in MODEL_FILES)


def runtime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def download_status() -> dict[str, Any]:
    with _lock:
        state = dict(_download_state)
    state["installed"] = model_installed()
    state["size_mb"] = MODEL_DOWNLOAD_MB
    state["runtime"] = runtime_available()
    return state


def start_download() -> dict[str, Any]:
    with _lock:
        if _download_state.get("state") == "running":
            return dict(_download_state)
        _download_state.update({"state": "running", "files_done": 0, "error": None})
    threading.Thread(target=_download, daemon=True, name="image-model-download").start()
    return download_status()


def _download() -> None:
    try:
        from huggingface_hub import hf_hub_download

        for index, name in enumerate(MODEL_FILES, start=1):
            hf_hub_download(MODEL_REPO, name)
            with _lock:
                _download_state["files_done"] = index
        state = {"state": "done", "error": None}
    except Exception as exc:  # noqa: BLE001
        state = {"state": "error", "error": f"Could not download the image matching model: {exc}"[:300]}
    with _lock:
        _download_state.update(state)


# Inference


def _providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    return [name for name in _GPU_PROVIDERS if name in available] + ["CPUExecutionProvider"]


def _session(path: str, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    providers = _providers()
    try:
        return ort.InferenceSession(path, options, providers=providers)
    except Exception:  # noqa: BLE001 - a GPU provider that cannot start falls back to the CPU
        if providers == ["CPUExecutionProvider"]:
            raise
        _log.warning("Image matching could not use the graphics card; using the processor")
        return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


class ImageMatcher:
    """SigLIP vision and text encoders. Loaded once and shared; safe to call from one job at a time."""

    def __init__(self, threads: int = 0):
        from tokenizers import Tokenizer

        paths = {name: _cached_path(name) for name in MODEL_FILES}
        if not all(paths.values()):
            raise FileNotFoundError("The image matching model is not downloaded")
        self.tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
        pad_id = self.tokenizer.token_to_id("</s>") or 1
        self.tokenizer.enable_padding(length=_TEXT_LENGTH, pad_id=pad_id, pad_token="</s>")
        self.tokenizer.enable_truncation(_TEXT_LENGTH)
        self.vision = _session(paths["onnx/vision_model.onnx"], threads)
        self.text = _session(paths["onnx/text_model.onnx"], threads)
        self.device = "GPU" if any(p in _GPU_PROVIDERS for p in self.vision.get_providers()) else "CPU"
        self._text_cache: dict[str, np.ndarray] = {}

    def embed_images(self, paths: list[str]) -> tuple[np.ndarray, list[bool]]:
        """Unit vectors for frames, and which frames could be read (unreadable ones are zero)."""
        from PIL import Image

        vectors = np.zeros((len(paths), 768), dtype=np.float32)
        readable = [False] * len(paths)
        for start in range(0, len(paths), _IMAGE_BATCH):
            batch, rows = [], []
            for offset, path in enumerate(paths[start : start + _IMAGE_BATCH]):
                try:
                    with Image.open(path) as image:
                        image = image.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.BICUBIC)
                        pixels = (np.asarray(image, dtype=np.float32) / 255.0 - 0.5) / 0.5
                    batch.append(pixels.transpose(2, 0, 1))
                    rows.append(start + offset)
                except (OSError, ValueError):
                    continue
            if not batch:
                continue
            output = self.vision.run(["pooler_output"], {"pixel_values": np.stack(batch)})[0]
            for row, vector in zip(rows, output):
                vectors[row] = _unit(vector)
                readable[row] = True
        return vectors, readable

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        missing = [text for text in dict.fromkeys(texts) if text not in self._text_cache]
        if missing:
            ids = np.array([encoding.ids for encoding in self.tokenizer.encode_batch(missing)], dtype=np.int64)
            output = self.text.run(["pooler_output"], {"input_ids": ids})[0]
            for text, vector in zip(missing, output):
                self._text_cache[text] = _unit(vector)
        return np.stack([self._text_cache[text] for text in texts])


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def get_matcher(threads: int = 0) -> ImageMatcher | None:
    """The shared matcher, or None when the model or runtime is missing or fails to load."""
    global _matcher
    if not runtime_available() or not model_installed():
        return None
    with _lock:
        if _matcher is None:
            try:
                _matcher = ImageMatcher(threads)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Image matching unavailable: %s", exc)
                return None
        return _matcher


# Scene scoring


def _spread(values: np.ndarray) -> np.ndarray:
    """Rescale to 0-1 between the 10th and 90th percentile of this video."""
    if values.size == 0:
        return values
    low, high = np.percentile(values, 10), np.percentile(values, 90)
    if high - low < 1e-6:
        return np.full_like(values, 0.5)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def score_scenes(matcher: ImageMatcher, scene_frames: dict[int, str], focus: str = "") -> dict[int, dict]:
    """Per scene: interest (0-1, relative to this video), focus match (0-1, or None without a
    focus), and the scene it duplicates, if any."""
    scene_ids = list(scene_frames)
    if not scene_ids:
        return {}
    vectors, readable = matcher.embed_images([scene_frames[scene_id] for scene_id in scene_ids])
    keep = [i for i, ok in enumerate(readable) if ok]
    if not keep:
        return {}
    vectors = vectors[keep]
    ids = [scene_ids[i] for i in keep]
    positive = matcher.embed_texts(list(POSITIVE_PROMPTS))
    negative = matcher.embed_texts(list(NEGATIVE_PROMPTS))
    raw = (vectors @ positive.T).max(axis=1) - (vectors @ negative.T).max(axis=1)
    interest = _spread(raw)
    focus_scores = None
    if focus.strip():
        focus_scores = _spread(vectors @ matcher.embed_texts([focus.strip()])[0])
    result: dict[int, dict] = {}
    for row, scene_id in enumerate(ids):
        duplicate_of = None
        if row:
            similarity = vectors[:row] @ vectors[row]
            best = int(np.argmax(similarity))
            if float(similarity[best]) >= DUPLICATE_SIMILARITY:
                duplicate_of = ids[best]
        result[scene_id] = {
            "interest": round(float(interest[row]), 3),
            "focus": round(float(focus_scores[row]), 3) if focus_scores is not None else None,
            "duplicate_of": duplicate_of,
        }
    return result
