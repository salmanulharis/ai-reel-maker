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
# Batched transcription writes down pieces of up to 30 s between pauses. They are split again
# after sentence ends and at pauses at least this long, so a segment stays about one sentence,
# as with the one-at-a-time method.
SENTENCE_PAUSE_SECONDS = 1.0
# A piece Whisper wrote without punctuation is split at its longest pauses until no part is
# longer than this (the one-at-a-time method's segments are rarely longer than about 6 s).
MAX_SEGMENT_SECONDS = 10.0
# Shorter gaps between words are not treated as pauses to split at.
_MIN_SPLIT_GAP = 0.2
_SENTENCE_ENDS = (".", "!", "?", "…", "。", "！", "？", "؟", "।", "॥", "።", "။")
_CLOSING = "\"'”’)]}」』"

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


# The files faster-whisper downloads for a model (see faster_whisper.utils.download_model).
WHISPER_FILE_PATTERNS = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*")


def download_whisper_model(model: str) -> str:
    """Download model files without loading them into memory."""
    from faster_whisper import download_model

    return download_model(model)


def whisper_repo(model: str) -> str | None:
    """The Hugging Face repository faster-whisper downloads ``model`` from, when known."""
    if "/" in model:
        return model
    try:
        from faster_whisper.utils import _MODELS  # not public; only used for the download size
    except ImportError:
        return None
    repo = _MODELS.get(model) if isinstance(_MODELS, dict) else None
    return str(repo) if repo else None


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
    batch_size: int = 0,
) -> Transcript:
    transcript, _info = transcribe_detailed(
        wav_path, model_size, device, cpu_threads, num_workers, batch_size=batch_size
    )
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
    on_position: Callable[[float], None] | None = None,
    language: str | None = None,
    batch_size: int = 0,
    task: str = "transcribe",
) -> tuple[Transcript, dict]:
    """Transcribe ``wav_path``. ``resume`` is a saved partial transcription (see
    ``on_progress``): the audio before its ``until`` time is skipped and its segments are kept.
    ``on_progress`` receives such a partial now and then, so an interrupted run can continue
    where it stopped. ``check_cancel`` is called between segments and stops the run by raising.
    ``on_position`` receives the time in the audio reached so far, after every segment.
    ``language`` (a Whisper code) skips language detection. ``batch_size`` above 0 writes down
    that many pieces of speech at once (see ``_transcribe_with``). ``task="translate"`` writes
    an English translation instead of the words as spoken."""
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
            on_position=on_position,
            language=language,
            batch_size=batch_size,
            task=task,
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
    on_position: Callable[[float], None] | None = None,
    language: str | None = None,
    batch_size: int = 0,
    task: str = "transcribe",
) -> Transcript:
    """With ``batch_size`` above 0, faster-whisper's batched mode is used: the speech is cut
    into pieces of up to 30 s at pauses (voice detection) and several pieces are written down
    at once, without re-reading unclear parts at higher temperatures. On the processor this is
    several times faster, and it avoids the repetition loops the one-at-a-time method falls
    into on languages the model knows poorly. Its pieces are split again into sentences."""
    model = _get_whisper_model(model_size, device, compute, cpu_threads, num_workers)
    segments: list[TranscriptSegment] = []
    words: list[WordTimestamp] = []
    chosen = language or None
    language = chosen
    probability: float | None = None
    offset = 0.0
    audio = str(wav_path)
    options: dict = {"language": chosen} if chosen else {}
    if task == "translate":
        options["task"] = "translate"
    resume = _valid_resume(resume)
    if resume:
        rest = _audio_after(wav_path, float(resume["until"]))
        if rest is not None:
            offset = float(resume["until"])
            segments = [TranscriptSegment.model_validate(item) for item in resume["segments"]]
            words = [WordTimestamp.model_validate(item) for item in resume.get("words") or []]
            language = chosen or resume.get("language")
            probability = resume.get("language_probability")
            if language:
                # The rest is in the language already heard; detecting it again could differ.
                options["language"] = language
            audio = rest
    if not isinstance(audio, str) and len(audio) < WHISPER_SAMPLE_RATE * _MIN_RESUME_AUDIO_SECONDS:
        return Transcript(segments=segments, words=words, language=language, language_probability=probability)

    batched = _batched_pipeline(model) if batch_size > 0 else None
    if batched is not None:
        segments_iter, info = batched.transcribe(
            audio, word_timestamps=True, vad_filter=True, batch_size=int(batch_size), **options
        )
        segments_iter = _sentence_segments(segments_iter)
    else:
        segments_iter, info = model.transcribe(audio, word_timestamps=True, vad_filter=True, **options)
    if not language:
        language = getattr(info, "language", None)
        detected = getattr(info, "language_probability", None)
        probability = round(float(detected), 3) if isinstance(detected, (int, float)) else None
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
        if on_position:
            on_position(offset + float(seg.end))
        if on_progress and time.monotonic() - last_saved >= PROGRESS_SAVE_SECONDS:
            last_saved = time.monotonic()
            on_progress(_partial_state(segments, words, language, probability))
    return Transcript(segments=segments, words=words, language=language, language_probability=probability)


def _batched_pipeline(model):
    """faster-whisper's batched pipeline around ``model``; None when this faster-whisper has none
    (then the one-at-a-time method is used)."""
    try:
        from faster_whisper import BatchedInferencePipeline
    except ImportError:
        return None
    return BatchedInferencePipeline(model=model)


def _sentence_segments(segments):
    """Whisper segments split after sentence ends and at long pauses, using their word times.
    A segment without word times is passed on unchanged. The text of a part is its words
    joined as Whisper wrote them (with their own spaces), so languages written without spaces
    stay intact."""
    for seg in segments:
        words = [word for word in (getattr(seg, "words", None) or []) if (word.word or "").strip()]
        if not words:
            yield seg
            continue
        part: list = []
        for index, word in enumerate(words):
            part.append(word)
            following = words[index + 1] if index + 1 < len(words) else None
            if following is None or _ends_sentence(word.word) or (
                float(following.start) - float(word.end) >= SENTENCE_PAUSE_SECONDS
            ):
                for piece in _split_long(part):
                    yield _WordSegment(piece)
                part = []


def _split_long(words: list) -> list[list]:
    """``words`` split at their longest pause, again and again, until no part is longer than
    MAX_SEGMENT_SECONDS or a part has no pause left to split at."""
    if float(words[-1].end) - float(words[0].start) <= MAX_SEGMENT_SECONDS or len(words) < 2:
        return [words]
    gaps = [float(words[i + 1].start) - float(words[i].end) for i in range(len(words) - 1)]
    widest = max(range(len(gaps)), key=lambda i: gaps[i])
    if gaps[widest] < _MIN_SPLIT_GAP:
        return [words]
    return _split_long(words[: widest + 1]) + _split_long(words[widest + 1 :])


class _WordSegment:
    """A part of a Whisper segment, shaped like faster-whisper's segments."""

    def __init__(self, words: list):
        self.words = words
        self.start = float(words[0].start)
        self.end = float(words[-1].end)
        self.text = "".join(word.word for word in words)


def _ends_sentence(word: str) -> bool:
    text = (word or "").strip().rstrip(_CLOSING)
    return text.endswith(_SENTENCE_ENDS)


def _partial_state(
    segments: list[TranscriptSegment],
    words: list[WordTimestamp],
    language: str | None,
    probability: float | None = None,
) -> dict:
    """A saved point: everything up to the end of the last finished segment."""
    return {
        "until": segments[-1].end if segments else 0.0,
        "segments": [item.model_dump() for item in segments],
        "words": [item.model_dump() for item in words],
        "language": language,
        "language_probability": probability,
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
