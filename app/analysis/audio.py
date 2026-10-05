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
    transcript, _info = transcribe_detailed(wav_path, model_size, device, cpu_threads, num_workers)
    return transcript


def preload_whisper(model_size: str, cpu_threads: int = 6, num_workers: int = 1):
    return _get_whisper_model(model_size, "cpu", "int8", cpu_threads, num_workers)


def transcribe_detailed(
    wav_path: Path,
    model_size: str = "small",
    device: str = "auto",
    cpu_threads: int = 6,
    num_workers: int = 1,
) -> tuple[Transcript, dict]:
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        empty = Transcript(segments=[], words=[], language=None)
        return empty, {"ok": False, "error": "faster-whisper is not installed", "device": "cpu", "compute_type": None}

    chosen_device, compute = _select_whisper_device(device)
    try:
        transcript = _transcribe_with(wav_path, model_size, chosen_device, compute, cpu_threads, num_workers)
        return transcript, {"ok": True, "device": chosen_device, "compute_type": compute, "fallback": None, "error": None}
    except Exception as exc:  # noqa: BLE001
        if chosen_device != "cuda":
            empty = Transcript(segments=[], words=[], language=None)
            return empty, {
                "ok": False,
                "device": chosen_device,
                "compute_type": compute,
                "fallback": None,
                "error": str(exc),
            }
        try:
            transcript = _transcribe_with(wav_path, model_size, "cpu", "int8", cpu_threads, num_workers)
            return transcript, {
                "ok": True,
                "device": "cpu",
                "compute_type": "int8",
                "fallback": "cpu",
                "error": None,
                "warning": "CUDA transcription failed, so Whisper continued on CPU.",
            }
        except Exception as retry_exc:  # noqa: BLE001
            empty = Transcript(segments=[], words=[], language=None)
            return empty, {
                "ok": False,
                "device": "cpu",
                "compute_type": "int8",
                "fallback": "cpu",
                "error": str(retry_exc),
            }


def _select_whisper_device(device: str) -> tuple[str, str]:
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", "float16"
        except Exception:
            return "cpu", "int8"
        return "cpu", "int8"
    if device == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"


def _transcribe_with(
    wav_path: Path,
    model_size: str,
    device: str,
    compute: str,
    cpu_threads: int,
    num_workers: int,
) -> Transcript:
    model = _get_whisper_model(model_size, device, compute, cpu_threads, num_workers)
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
