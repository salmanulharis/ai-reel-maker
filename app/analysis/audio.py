from __future__ import annotations

import threading
from pathlib import Path

from app.models.schemas import Transcript, TranscriptSegment, WordTimestamp
from app.video.ffmpeg_runner import FFmpegRunner, FFmpegError

_whisper_models: dict[tuple[str, str, str, int, int], object] = {}
_whisper_lock = threading.Lock()


def extract_audio(runner: FFmpegRunner, video_path: Path, wav_path: Path) -> Path:
    return runner.extract_audio(video_path, wav_path)


def _get_whisper_model(model_size: str, device: str, compute: str, cpu_threads: int, num_workers: int):
    key = (model_size, device, compute, cpu_threads, num_workers)
    with _whisper_lock:
        model = _whisper_models.get(key)
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
            _whisper_models[key] = model
        return model


def transcribe(
    wav_path: Path,
    model_size: str = "small",
    device: str = "auto",
    cpu_threads: int = 6,
    num_workers: int = 1,
) -> Transcript:
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        return Transcript(segments=[], words=[], language=None)

    chosen_device = "cpu"
    compute = "int8"
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                chosen_device = "cuda"
                compute = "float16"
        except Exception:
            chosen_device = "cpu"
    elif device in {"cuda", "cpu"}:
        chosen_device = device
        compute = "float16" if device == "cuda" else "int8"

    model = _get_whisper_model(model_size, chosen_device, compute, cpu_threads, num_workers)
    segments_iter, info = model.transcribe(str(wav_path), word_timestamps=True, vad_filter=True)
    segments = []
    words = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        segments.append(TranscriptSegment(start=float(seg.start), end=float(seg.end), text=text))
        for word in getattr(seg, "words", None) or []:
            words.append(
                WordTimestamp(
                    word=word.word.strip(),
                    start=float(word.start),
                    end=float(word.end),
                )
            )
    language = getattr(info, "language", None)
    return Transcript(segments=segments, words=words, language=language)


def detect_silence(runner: FFmpegRunner, media_path: Path, noise_db: float, min_duration: float) -> list[dict]:
    try:
        return runner.detect_silence(media_path, noise_db, min_duration)
    except FFmpegError:
        return []
