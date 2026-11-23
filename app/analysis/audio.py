from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

from app.models.schemas import Transcript, TranscriptSegment, WordTimestamp
from app.video.ffmpeg_runner import CancelledError, FFmpegRunner, FFmpegError

# Whisper works at this sample rate; a resumed transcription skips this many samples per second.
WHISPER_SAMPLE_RATE = 16000
# How often a running transcription hands its finished segments to ``on_progress``.
PROGRESS_SAVE_SECONDS = 20.0
# Less audio than this after the saved point is not worth starting Whisper for.
_MIN_RESUME_AUDIO_SECONDS = 0.5

_whisper_models: dict[tuple[str, str, str, int, int], object] = {}
_whisper_lock = threading.Lock()

# Hugging Face repos behind faster-whisper's model names, for when the installed
# faster-whisper is missing or older than the catalog.
_WHISPER_REPOS = {
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "large": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def whisper_repo(model: str) -> str:
    try:
        from faster_whisper.utils import _MODELS

        repo = _MODELS.get(model)
        if repo:
            return repo
    except Exception:  # noqa: BLE001
        pass
    if "/" in model:
        return model
    return _WHISPER_REPOS.get(model, f"Systran/faster-whisper-{model}")


def _hf_hub_cache() -> Path:
    try:
        from huggingface_hub import constants

        cache = getattr(constants, "HF_HUB_CACHE", None) or getattr(constants, "HUGGINGFACE_HUB_CACHE", None)
        if cache:
            return Path(cache)
    except Exception:  # noqa: BLE001
        pass
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache:
        return Path(cache)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def whisper_model_cached(model: str) -> bool:
    """True when a complete copy of the model is in the Hugging Face cache."""
    local = Path(model)
    if local.is_dir():
        return (local / "model.bin").exists()
    snapshots = _hf_hub_cache() / ("models--" + whisper_repo(model).replace("/", "--")) / "snapshots"
    if not snapshots.is_dir():
        return False
    return any((snapshot / "model.bin").exists() for snapshot in snapshots.iterdir())


def download_whisper_model(model: str) -> str:
    """Download model files without loading them into memory."""
    from faster_whisper import download_model

    return download_model(model)


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


def transcribe_detailed(
    wav_path: Path,
    model_size: str = "small",
    device: str = "auto",
    cpu_threads: int = 6,
    num_workers: int = 1,
    gpu_compute: str | None = None,
    resume: dict | None = None,
    on_progress: Callable[[dict], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> tuple[Transcript, dict]:
    """Transcribe ``wav_path``. ``resume`` is a saved partial transcription (see
    ``on_progress``): the audio before its ``until`` time is skipped and its segments are kept.
    ``on_progress`` receives such a partial now and then, so an interrupted run can continue
    where it stopped. ``check_cancel`` is called between segments and stops the run by raising."""
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        empty = Transcript(segments=[], words=[], language=None)
        return empty, {"ok": False, "error": "faster-whisper is not installed", "device": "cpu", "compute_type": None}

    chosen_device, compute = _select_whisper_device(device, gpu_compute)
    # The most recent saved point, so a retry on the CPU continues from it too.
    latest = {"state": _valid_resume(resume)}

    def progress(state: dict) -> None:
        latest["state"] = state
        if on_progress:
            on_progress(state)

    def run(run_device: str, run_compute: str) -> Transcript:
        return _transcribe_with(
            wav_path,
            model_size,
            run_device,
            run_compute,
            cpu_threads,
            num_workers,
            resume=latest["state"],
            on_progress=progress,
            check_cancel=check_cancel,
        )

    try:
        transcript = run(chosen_device, compute)
        return transcript, {"ok": True, "device": chosen_device, "compute_type": compute, "fallback": None, "error": None}
    except CancelledError:
        raise
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
            transcript = run("cpu", "int8")
            return transcript, {
                "ok": True,
                "device": "cpu",
                "compute_type": "int8",
                "fallback": "cpu",
                "error": None,
                "warning": "CUDA transcription failed, so Whisper continued on CPU.",
            }
        except CancelledError:
            raise
        except Exception as retry_exc:  # noqa: BLE001
            empty = Transcript(segments=[], words=[], language=None)
            return empty, {
                "ok": False,
                "device": "cpu",
                "compute_type": "int8",
                "fallback": "cpu",
                "error": str(retry_exc),
            }


def _select_whisper_device(device: str, gpu_compute: str | None = None) -> tuple[str, str]:
    """(device, compute type). "auto" uses an NVIDIA card when faster-whisper's engine can see
    one. ``gpu_compute`` overrides the precision on the card (int8_float16 uses less memory)."""
    if device == "auto":
        from app.services.hardware import whisper_cuda_available

        device = "cuda" if whisper_cuda_available() else "cpu"
    if device == "cuda":
        return "cuda", gpu_compute or "float16"
    return "cpu", "int8"


def release_whisper_models() -> int:
    """Drop loaded Whisper models so their memory is free for the AI models. Returns how many."""
    import gc

    with _whisper_lock:
        count = len(_whisper_models)
        _whisper_models.clear()
    if count:
        gc.collect()
    return count


def _transcribe_with(
    wav_path: Path,
    model_size: str,
    device: str,
    compute: str,
    cpu_threads: int,
    num_workers: int,
    resume: dict | None = None,
    on_progress: Callable[[dict], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> Transcript:
    model = _get_whisper_model(model_size, device, compute, cpu_threads, num_workers)
    segments: list[TranscriptSegment] = []
    words: list[WordTimestamp] = []
    language = None
    offset = 0.0
    audio = str(wav_path)
    options: dict = {}
    resume = _valid_resume(resume)
    if resume:
        rest = _audio_after(wav_path, float(resume["until"]))
        if rest is not None:
            offset = float(resume["until"])
            segments = [TranscriptSegment.model_validate(item) for item in resume["segments"]]
            words = [WordTimestamp.model_validate(item) for item in resume.get("words") or []]
            language = resume.get("language")
            if language:
                # The rest is in the language already heard; detecting it again could differ.
                options["language"] = language
            audio = rest
    if not isinstance(audio, str) and len(audio) < WHISPER_SAMPLE_RATE * _MIN_RESUME_AUDIO_SECONDS:
        return Transcript(segments=segments, words=words, language=language)

    segments_iter, info = model.transcribe(audio, word_timestamps=True, vad_filter=True, **options)
    language = language or getattr(info, "language", None)
    last_saved = time.monotonic()
    for seg in segments_iter:
        if check_cancel:
            check_cancel()
        text = (seg.text or "").strip()
        segments.append(TranscriptSegment(start=offset + float(seg.start), end=offset + float(seg.end), text=text))
        for word in getattr(seg, "words", None) or []:
            words.append(
                WordTimestamp(
                    word=word.word.strip(),
                    start=offset + float(word.start),
                    end=offset + float(word.end),
                )
            )
        if on_progress and time.monotonic() - last_saved >= PROGRESS_SAVE_SECONDS:
            last_saved = time.monotonic()
            on_progress(_partial_state(segments, words, language))
    return Transcript(segments=segments, words=words, language=language)


def _partial_state(segments: list[TranscriptSegment], words: list[WordTimestamp], language: str | None) -> dict:
    """A saved point: everything up to the end of the last finished segment."""
    return {
        "until": segments[-1].end if segments else 0.0,
        "segments": [item.model_dump() for item in segments],
        "words": [item.model_dump() for item in words],
        "language": language,
    }


def _valid_resume(resume: dict | None) -> dict | None:
    if not isinstance(resume, dict) or not isinstance(resume.get("segments"), list) or not resume["segments"]:
        return None
    try:
        until = float(resume.get("until") or 0)
    except (TypeError, ValueError):
        return None
    return resume if until > 0 else None


def _audio_after(wav_path: Path, seconds: float):
    """The audio after ``seconds`` as Whisper's samples, or None when it cannot be read (the
    transcription then starts over)."""
    try:
        from faster_whisper import decode_audio
    except ImportError:
        return None
    try:
        samples = decode_audio(str(wav_path), sampling_rate=WHISPER_SAMPLE_RATE)
    except Exception:  # noqa: BLE001 - starting over is always possible
        return None
    return samples[int(seconds * WHISPER_SAMPLE_RATE):]


def detect_silence(runner: FFmpegRunner, media_path: Path, noise_db: float, min_duration: float) -> list[dict]:
    try:
        return runner.detect_silence(media_path, noise_db, min_duration)
    except FFmpegError:
        return []
