"""Sound understanding with CLAP (ONNX): what can be heard, moment by moment.

CLAP turns a few seconds of sound and a short description into vectors whose similarity says
how well they match, like SigLIP does for pictures. The sound is cut into overlapping 5 s
windows; each window gets a vector, which is kept with the analysis. From the vectors come
labels for common sounds (music, laughter, applause, a crowd cheering, a dog barking, …) and,
for prompt mode, how well each window matches the sounds the user asked for. Matching new
words later needs no new pass over the audio.

Model: laion/clap-htsat-unfused (Apache-2.0), ONNX export by Xenova, full precision. Runs on
the CPU (about 0.1 s per window on a 6-core processor, so a few minutes per hour of sound) or
on a GPU with an onnxruntime GPU build. Sound is read at 48 kHz, the rate the model was trained
on (a separate copy of the sound is extracted for it and removed afterwards).
"""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.ai.onnx_models import HubModel, on_gpu, session
from app.ai.onnx_models import runtime_available as onnx_runtime_available
from app.analysis.clap_features import SAMPLE_RATE, log_mel

_log = logging.getLogger(__name__)

MODEL_REPO = "Xenova/clap-htsat-unfused"  # ONNX export of laion/clap-htsat-unfused (Apache-2.0)
MODEL_FILES = ("tokenizer.json", "onnx/audio_model.onnx", "onnx/text_model.onnx")
MODEL_DOWNLOAD_MB = 620
WINDOW_SECONDS = 5.0
HOP_SECONDS = 2.5
_BATCH = 8
# Similarity scale for labels (as in CLIP-style models); a softmax over the labels at this scale.
_LABEL_SCALE = 100.0
# A label is shown for a window when its share is at least this, best three at most.
_LABEL_MIN_SHARE = 0.2
_TEMPLATE = "This is a sound of {}."
VERSION = 1

# Short key -> description given to the model. The keys are what the AI editor and the log see.
SOUND_LABELS: dict[str, str] = {
    "speech": "people talking",
    "music": "music playing",
    "singing": "someone singing",
    "laughter": "people laughing",
    "applause": "applause and clapping",
    "cheering": "a crowd cheering",
    "crowd": "the noise of a crowd",
    "children": "children playing",
    "baby": "a baby crying",
    "dog": "a dog barking",
    "cat": "a cat meowing",
    "birds": "birds singing",
    "water": "water and waves",
    "wind": "wind blowing",
    "rain": "rain falling",
    "traffic": "cars and traffic",
    "engine": "an engine running",
    "siren": "a siren",
    "bang": "an explosion or a loud bang",
    "footsteps": "footsteps",
    "whistle": "a whistle blowing",
    "ball": "a ball being kicked or hit",
    "bell": "a bell ringing",
    "phone": "a phone ringing",
    "typing": "typing on a keyboard",
    "cooking": "cooking in a kitchen",
    "silence": "silence",
}

_MODEL = HubModel(MODEL_REPO, MODEL_FILES, MODEL_DOWNLOAD_MB, "sound model")
_lock = threading.Lock()
_matcher: "SoundMatcher | None" = None


def model_installed() -> bool:
    return _MODEL.installed()


def runtime_available() -> bool:
    return onnx_runtime_available()


def download_status() -> dict[str, Any]:
    return _MODEL.status(model_installed(), runtime_available())


def start_download() -> dict[str, Any]:
    if _MODEL.begin():
        threading.Thread(target=_MODEL.download, daemon=True, name="sound-model-download").start()
    return download_status()


class SoundMatcher:
    """CLAP sound and text encoders. Loaded once and shared; used by one job at a time."""

    def __init__(self, threads: int = 0):
        from tokenizers import Tokenizer

        paths = {name: _MODEL.cached_path(name) for name in MODEL_FILES}
        if not all(paths.values()):
            raise FileNotFoundError("The sound model is not downloaded")
        self.tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(77)
        self.audio = session(paths["onnx/audio_model.onnx"], threads, "Sound matching")
        self.text = session(paths["onnx/text_model.onnx"], threads, "Sound matching")
        self.device = "GPU" if on_gpu(self.audio) else "CPU"
        self._text_cache: dict[str, np.ndarray] = {}

    def embed_audio(self, windows: list[np.ndarray]) -> np.ndarray:
        """Unit vectors for windows of 48 kHz samples."""
        features = np.stack([log_mel(samples)[None] for samples in windows])
        return _unit_rows(self.audio.run(["audio_embeds"], {"input_features": features})[0])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        for text in dict.fromkeys(texts):
            if text not in self._text_cache:
                # One at a time: the exported model takes no attention mask, so padding would
                # change the result.
                ids = np.array([self.tokenizer.encode(text).ids], dtype=np.int64)
                self._text_cache[text] = _unit_rows(self.text.run(["text_embeds"], {"input_ids": ids})[0])[0]
        return np.stack([self._text_cache[text] for text in texts])


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


def get_sound_matcher(threads: int = 0) -> SoundMatcher | None:
    """The shared matcher, or None when the model or runtime is missing or fails to load."""
    global _matcher
    if not runtime_available() or not model_installed():
        return None
    with _lock:
        if _matcher is None:
            try:
                _matcher = SoundMatcher(threads)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Sound matching unavailable: %s", exc)
                return None
        return _matcher


def release_sound_matcher() -> bool:
    """Drop the loaded model so its memory is free for the AI models. It reloads on next use."""
    global _matcher
    import gc

    with _lock:
        loaded = _matcher is not None
        _matcher = None
    if loaded:
        gc.collect()
    return loaded


# Analysis


def embed_recording(
    matcher: SoundMatcher, wav_path: Path, check_cancel: Callable[[], None] | None = None
) -> tuple[list[float], np.ndarray]:
    """Window start times and one unit vector per window of a 48 kHz 16-bit WAV."""
    starts: list[float] = []
    vectors: list[np.ndarray] = []
    with wave.open(str(wav_path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("expected 16-bit audio")
        rate, channels, total = handle.getframerate(), handle.getnchannels(), handle.getnframes()
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio")
        size, hop = int(WINDOW_SECONDS * rate), int(HOP_SECONDS * rate)
        positions = list(range(0, max(1, total - size + hop), hop)) if total > 0 else []
        for begin in range(0, len(positions), _BATCH):
            if check_cancel:
                check_cancel()
            windows = []
            for position in positions[begin : begin + _BATCH]:
                handle.setpos(position)
                samples = np.frombuffer(handle.readframes(size), dtype="<i2").astype(np.float32) / 32768.0
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                windows.append(samples)
                starts.append(round(position / rate, 3))
            vectors.append(matcher.embed_audio(windows))
    if not vectors:
        return [], np.zeros((0, 512), dtype=np.float32)
    return starts, np.concatenate(vectors)


def label_windows(matcher: SoundMatcher, vectors: np.ndarray) -> list[dict[str, float]]:
    """Per window, the common sounds that can be heard, with their share (best three at most)."""
    if vectors.size == 0:
        return []
    keys = list(SOUND_LABELS)
    texts = matcher.embed_texts([_TEMPLATE.format(SOUND_LABELS[key]) for key in keys])
    logits = (vectors @ texts.T) * _LABEL_SCALE
    logits -= logits.max(axis=1, keepdims=True)
    shares = np.exp(logits)
    shares /= shares.sum(axis=1, keepdims=True)
    labels: list[dict[str, float]] = []
    for row in shares:
        best = np.argsort(-row)[:3]
        labels.append({keys[i]: round(float(row[i]), 3) for i in best if row[i] >= _LABEL_MIN_SHARE})
    return labels


def match_texts(matcher: SoundMatcher, vectors: np.ndarray, texts: list[str]) -> list[float]:
    """Per window, how well it matches the best of ``texts`` (0-1, relative to this recording)."""
    texts = [text.strip() for text in texts if text and text.strip()]
    if vectors.size == 0 or not texts:
        return []
    similarity = (vectors @ matcher.embed_texts([_TEMPLATE.format(text) for text in texts]).T).max(axis=1)
    low, high = np.percentile(similarity, 10), np.percentile(similarity, 95)
    if high - low < 1e-6:
        return [0.0] * len(similarity)
    return [round(float(value), 3) for value in np.clip((similarity - low) / (high - low), 0.0, 1.0)]


def sounds_between(events: dict | None, start: float, end: float) -> dict[str, float]:
    """The sounds heard between two times (best share per label), and "cue" for how well the
    moment matches the sounds the user asked for, from saved ``sound_events.json`` data."""
    if not isinstance(events, dict):
        return {}
    window = float(events.get("window") or WINDOW_SECONDS)
    heard: dict[str, float] = {}
    for item in events.get("windows") or []:
        begin = float(item.get("start") or 0)
        if begin + window <= start or begin >= end:
            continue
        for key, share in (item.get("labels") or {}).items():
            heard[key] = max(heard.get(key, 0.0), float(share))
        if item.get("cue") is not None:
            heard["cue"] = max(heard.get("cue", 0.0), float(item["cue"]))
    return heard
