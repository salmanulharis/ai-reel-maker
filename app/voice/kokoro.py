"""Text to speech with Kokoro v1.0 (82M parameters, ONNX), on this computer.

The model (about 340 MB) is downloaded from Settings like the picture and sound models, and runs
with onnxruntime through ``kokoro-onnx`` (which brings its own espeak-ng, so nothing else needs
installing). Each narration line is spoken once and kept in the job's ``narration/`` folder,
keyed by its text, voice, speed and model: editing one line on the Review page only re-voices
that line.

Word timings (for captions) come from running faster-whisper on the spoken line and matching
the words it hears to the known text; without Whisper they are spread by word length.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from app.ai.onnx_models import HubModel, session
from app.utils.atomic import atomic_write_json, part_path, read_json, replace_file

_log = logging.getLogger(__name__)

MODEL_REPO = "fastrtc/kokoro-onnx"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
MODEL_DOWNLOAD_MB = 338
SAMPLE_RATE = 24000
VERSION = 1
NARRATION_DIR = "narration"
MANIFEST = "narration.json"

# English voices (Kokoro v1.0). "a" = American, "b" = British; f/m = female/male.
VOICES: dict[str, str] = {
    "af_heart": "Heart (US, female) - recommended",
    "af_bella": "Bella (US, female)",
    "af_nicole": "Nicole (US, female, soft)",
    "am_michael": "Michael (US, male)",
    "am_fenrir": "Fenrir (US, male)",
    "am_puck": "Puck (US, male, lively)",
    "bf_emma": "Emma (UK, female)",
    "bm_george": "George (UK, male)",
    "bm_fable": "Fable (UK, male)",
}
DEFAULT_VOICE = "af_heart"

_MODEL = HubModel(MODEL_REPO, MODEL_FILES, MODEL_DOWNLOAD_MB, "narration voice model")
_narrator = None
_narrator_lock = threading.Lock()
_speak_lock = threading.Lock()
# One narration run per job folder at a time (the Review preview and a render can overlap).
_folder_locks: dict[str, threading.Lock] = {}
_folder_locks_guard = threading.Lock()


def _folder_lock(folder: Path) -> threading.Lock:
    with _folder_locks_guard:
        return _folder_locks.setdefault(str(Path(folder).resolve()), threading.Lock())


def runtime_available() -> bool:
    try:
        import kokoro_onnx  # noqa: F401 - only checks that it is installed
        import onnxruntime  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        return False
    return True


def installed() -> bool:
    return _MODEL.installed()


def download_status() -> dict:
    status = _MODEL.status(installed(), runtime_available())
    status["voices"] = VOICES
    return status


def start_download() -> dict:
    if _MODEL.begin():
        threading.Thread(target=_MODEL.download, daemon=True, name="voice-model-download").start()
    return download_status()


def valid_voice(name: str) -> str:
    return name if name in VOICES else DEFAULT_VOICE


def language_for(voice: str) -> str:
    return "en-gb" if voice.startswith("b") else "en-us"


def speakable(text: str) -> bool:
    """Whether ``text`` is something the English voices can read (mostly Latin letters)."""
    letters = [char for char in str(text or "") if char.isalpha()]
    if not letters:
        return False
    latin = sum(1 for char in letters if ord(char) < 0x250)
    return latin / len(letters) >= 0.85


class Narrator:
    """The loaded voice model. Shared; one line is spoken at a time."""

    def __init__(self, threads: int = 0):
        from kokoro_onnx import Kokoro

        paths = {name: _MODEL.cached_path(name) for name in MODEL_FILES}
        if not all(paths.values()):
            raise FileNotFoundError("The narration voice model is not downloaded. Download it in Settings.")
        self.kokoro = Kokoro.from_session(session(paths[MODEL_FILES[0]], threads, "Narration"), paths[MODEL_FILES[1]])

    def speak(self, text: str, voice: str, speed: float) -> np.ndarray:
        voice = valid_voice(voice)
        with _speak_lock:
            samples, rate = self.kokoro.create(
                text, voice=voice, speed=float(min(1.5, max(0.6, speed))), lang=language_for(voice)
            )
        samples = np.asarray(samples, dtype=np.float32)
        if rate != SAMPLE_RATE:
            raise RuntimeError(f"Unexpected sample rate {rate} from the voice model")
        return samples


def get_narrator(threads: int = 0) -> Narrator:
    global _narrator
    with _narrator_lock:
        if _narrator is None:
            _narrator = Narrator(threads)
        return _narrator


def release_narrator() -> bool:
    """Free the voice model (before memory-hungry steps such as rendering). True when one was loaded."""
    global _narrator
    with _narrator_lock:
        loaded = _narrator is not None
        _narrator = None
    return loaded


def line_key(text: str, voice: str, speed: float) -> str:
    data = [VERSION, MODEL_FILES[0], " ".join(str(text).split()), valid_voice(voice), round(float(speed), 2)]
    return hashlib.sha1(json.dumps(data).encode("utf-8")).hexdigest()[:20]


def narrate(
    job_dir: Path,
    lines: list[tuple[str, str]],
    *,
    voice: str,
    speed: float,
    settings=None,
    speak: Callable[[str, str, float], np.ndarray] | None = None,
    align: Callable[[Path, str], list[dict]] | None = None,
    check_cancel: Callable[[], None] | None = None,
    on_line: Callable[[int, int], None] | None = None,
) -> dict[str, dict]:
    """Spoken versions of ``lines`` ((beat id, text)), as {beat id: {file, duration, words,
    text, key}}. Lines spoken before with the same text, voice and speed are reused. ``speak``
    and ``align`` replace the voice model and Whisper (tests)."""
    folder = Path(job_dir) / NARRATION_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with _folder_lock(folder):
        return _narrate(folder, lines, voice, speed, settings, speak, align, check_cancel, on_line)


def _narrate(folder: Path, lines, voice, speed, settings, speak, align, check_cancel, on_line) -> dict[str, dict]:
    import soundfile

    manifest_path = folder / MANIFEST
    manifest = read_json(manifest_path, {})
    manifest = manifest if isinstance(manifest, dict) else {}
    if speak is None:
        threads = int(getattr(settings, "tts_threads", 0) or 0)
        speak = lambda text, v, s: get_narrator(threads).speak(text, v, s)  # noqa: E731
    if align is None:
        align = lambda path, text: align_words(path, text, settings)  # noqa: E731
    result: dict[str, dict] = {}
    wanted = [(beat, text) for beat, text in lines if str(text or "").strip()]
    for number, (beat, text) in enumerate(wanted, start=1):
        if check_cancel:
            check_cancel()
        key = line_key(text, voice, speed)
        entry = manifest.get(key)
        path = folder / f"{key}.wav"
        if not (isinstance(entry, dict) and path.is_file()):
            samples = speak(text, voice, speed)
            temp = part_path(path)
            soundfile.write(str(temp), samples, SAMPLE_RATE, subtype="PCM_16", format="WAV")
            replace_file(temp, path)
            duration = round(len(samples) / SAMPLE_RATE, 3)
            try:
                words = align(path, text)
            except Exception:  # noqa: BLE001 - timings spread by length are still usable
                _log.debug("Word timing failed for a narration line", exc_info=True)
                words = []
            entry = {
                "file": path.name,
                "duration": duration,
                "text": text,
                "words": words or spread_words(text, duration),
            }
            manifest[key] = entry
            atomic_write_json(manifest_path, manifest)
        result[beat] = {**entry, "key": key}
        if on_line:
            on_line(number, len(wanted))
    keep = {item["key"] for item in result.values()}
    if set(manifest) - keep:
        # Lines no longer in the brief: their files are removed, the manifest keeps only these.
        for key in set(manifest) - keep:
            (folder / f"{key}.wav").unlink(missing_ok=True)
        atomic_write_json(manifest_path, {key: manifest[key] for key in keep})
    return result


def spread_words(text: str, duration: float) -> list[dict]:
    """Word times spread over ``duration`` by word length (no recognition)."""
    words = str(text or "").split()
    if not words or duration <= 0:
        return []
    weights = [len(word) + 1 for word in words]
    total = float(sum(weights))
    result, cursor = [], 0.0
    for word, weight in zip(words, weights):
        length = duration * weight / total
        result.append({"text": word, "start": round(cursor, 3), "end": round(cursor + length, 3)})
        cursor += length
    return result


def align_words(path: Path, text: str, settings=None) -> list[dict]:
    """The words of ``text`` with the times Whisper hears them in ``path``."""
    from app.analysis.audio import transcribe_detailed

    transcript, info = transcribe_detailed(
        Path(path),
        model_size=str(getattr(settings, "whisper_model", "base") or "base"),
        device=str(getattr(settings, "whisper_device", "auto") or "auto"),
        cpu_threads=int(getattr(settings, "whisper_cpu_threads", 4) or 4),
        num_workers=1,
    )
    heard = [(w.word, float(w.start), float(w.end)) for w in transcript.words]
    if not heard or not info.get("ok", True):
        return []
    return match_words(text, heard)


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def match_words(text: str, heard: list[tuple[str, float, float]]) -> list[dict]:
    """Times for the known words of ``text`` from recognised words: matched words take the
    recognised times; the rest are spread between their matched neighbours."""
    words = str(text or "").split()
    if not words:
        return []
    known = [_norm(w) for w in words]
    seen = [_norm(w) for w, _, _ in heard]
    times: list[tuple[float, float] | None] = [None] * len(words)
    matcher = difflib.SequenceMatcher(a=known, b=seen, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            _, start, end = heard[block.b + offset]
            times[block.a + offset] = (start, end)
    end_of_audio = max(end for _, _, end in heard)
    result = []
    index = 0
    while index < len(words):
        if times[index] is not None:
            start, end = times[index]
            result.append({"text": words[index], "start": round(start, 3), "end": round(end, 3)})
            index += 1
            continue
        gap_end = index
        while gap_end < len(words) and times[gap_end] is None:
            gap_end += 1
        left = result[-1]["end"] if result else 0.0
        right = times[gap_end][0] if gap_end < len(words) else max(left, end_of_audio)
        span = max(0.05 * (gap_end - index), right - left)
        step = span / (gap_end - index)
        for position in range(index, gap_end):
            start = left + step * (position - index)
            result.append({"text": words[position], "start": round(start, 3), "end": round(start + step, 3)})
        index = gap_end
    return result
