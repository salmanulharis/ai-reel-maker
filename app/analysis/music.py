"""Music-led editing: cut the video to the beat of the user's music.

The track is read once (numpy only, no extra packages) into a beat grid:
- an onset curve from the spectral flux of a short-time Fourier transform,
- the tempo from the curve's autocorrelation, weighted towards common tempos,
- beat times by dynamic programming (Ellis, "Beat tracking by dynamic programming", 2007): beats
  fall on strong onsets and stay about one period apart,
- bars by choosing which of every four beats carries the most accent (the downbeat),
- the energy of every beat (loudness, smoothed over two bars, 0-1 within the track) and how lively
  the track is overall (tempo and loudness).

The result is cached in ``music_analysis.json`` by the music file's name, size and time.

Planning then works in "slots": stretches of the music from one beat to a later one, starting on
the first downbeat. Where the music is energetic a slot is short (quick cuts), where it is calm a
slot is long. The planner chooses enough moments, and each is fitted to its slot exactly, so every
cut of the video lands on a beat. The video ends at the end of a 4-bar phrase when it can.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from app.utils.atomic import atomic_write_json, read_json
from app.utils.security import safe_path_argument

MUSIC_ANALYSIS_FILE = "music_analysis.json"
VERSION = 1
_RATE = 22050
_FRAME = 2048
_HOP = 512
_MIN_BPM, _MAX_BPM, _PRIOR_BPM = 60.0, 200.0, 120.0
_TIGHTNESS = 100.0
# The doubled tempo wins when its autocorrelation is at least this share of the slower one.
_DOUBLE_TEMPO_SHARE = 0.5
# Seconds a clip should last for music at this energy (0-1): quick cuts where it is loud.
_CALM_CLIP, _LIVELY_CLIP = 5.0, 2.2
# The fewest beats in a slot, and the choices (multiples of two keep cuts on strong beats).
_SLOT_BEATS = (2, 4, 8, 16)
_MIN_SLOT_SECONDS = 1.2
_PHRASE_BEATS = 16
# A phrase end is preferred when it keeps at least this share of the length aimed for.
_PHRASE_KEEP = 0.8


class MusicError(RuntimeError):
    pass


# Reading the track


def analyze_file(path: Path, runner, work_dir: Path) -> dict:
    """The beat grid of the music file at ``path`` (see the module docstring)."""
    wav = Path(work_dir) / ".music_analysis.wav"
    try:
        runner.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                safe_path_argument(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(_RATE),
                "-c:a",
                "pcm_s16le",
                safe_path_argument(wav),
            ]
        )
        samples = _read_wav(wav)
    finally:
        wav.unlink(missing_ok=True)
    return analyze_samples(samples, _RATE)


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data


def analyze_samples(samples: np.ndarray, rate: int = _RATE) -> dict:
    samples = np.asarray(samples, dtype=np.float32)
    duration = len(samples) / float(rate)
    if duration < 4.0:
        raise MusicError("The music is shorter than 4 seconds.")
    onset = _onset_curve(samples)
    fps = rate / _HOP
    period, confidence = _tempo(onset, fps)
    frames = _track_beats(onset, period)
    # A frame's flux peaks when the onset is near the middle of its window.
    beats = [round((float(f) * _HOP + _FRAME / 2) / rate, 3) for f in frames]
    if len(beats) < 8:
        raise MusicError("No steady beat was found in the music.")
    intervals = np.diff(beats)
    steady = intervals[np.abs(intervals - np.median(intervals)) < 0.25 * np.median(intervals)]
    bpm = 60.0 / float(np.mean(steady if len(steady) else intervals))
    strength = [float(onset[min(len(onset) - 1, int(f))]) for f in frames]
    rms = _beat_rms(samples, rate, beats)
    phase = _downbeat_phase(strength, rms)
    energy = _energy(rms)
    loudness_db = 20 * math.log10(max(1e-6, float(np.sqrt(np.mean(samples**2)))))
    lively = 0.6 * _clamp((bpm - 80) / 60) + 0.4 * _clamp((loudness_db + 30) / 18)
    return {
        "version": VERSION,
        "duration": round(duration, 3),
        "bpm": round(bpm, 1),
        "confidence": round(confidence, 3),
        "beats": beats,
        "downbeat_phase": phase,
        "start": beats[phase],
        "energy": [round(value, 3) for value in energy],
        "lively": round(lively, 3),
    }


def _frames(samples: np.ndarray) -> np.ndarray:
    if len(samples) < _FRAME:
        samples = np.pad(samples, (0, _FRAME - len(samples)))
    count = 1 + (len(samples) - _FRAME) // _HOP
    strides = (samples.strides[0] * _HOP, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=(count, _FRAME), strides=strides)


def _onset_curve(samples: np.ndarray) -> np.ndarray:
    window = np.hanning(_FRAME).astype(np.float32)
    frames = _frames(samples)
    curve = np.zeros(len(frames), dtype=np.float32)
    previous = None
    # In blocks, so a long track never holds the whole spectrum in memory.
    for begin in range(0, len(frames), 2048):
        block = np.abs(np.fft.rfft(frames[begin : begin + 2048] * window, axis=1))
        block = np.log1p(100.0 * block)
        if previous is not None:
            block = np.vstack([previous[None, :], block])
            flux = np.maximum(0.0, np.diff(block, axis=0)).sum(axis=1)
            curve[begin : begin + len(flux)] = flux
        else:
            flux = np.maximum(0.0, np.diff(block, axis=0)).sum(axis=1)
            curve[begin + 1 : begin + 1 + len(flux)] = flux
        previous = block[-1]
    # Remove the slowly moving level, keep the peaks.
    width = 16
    kernel = np.ones(width, dtype=np.float32) / width
    local = np.convolve(curve, kernel, mode="same")
    curve = np.maximum(0.0, curve - local)
    peak = float(curve.max())
    return curve / peak if peak > 0 else curve


def _tempo(onset: np.ndarray, fps: float) -> tuple[float, float]:
    """(beat period in frames, confidence 0-1)."""
    centred = onset - onset.mean()
    size = 1 << int(math.ceil(math.log2(max(2, len(centred) * 2))))
    spectrum = np.fft.rfft(centred, size)
    auto = np.fft.irfft(spectrum * np.conj(spectrum))[: len(centred)]
    if auto[0] <= 0:
        return fps * 60.0 / _PRIOR_BPM, 0.0
    auto = auto / auto[0]
    low = int(fps * 60.0 / _MAX_BPM)
    high = min(len(auto) - 1, int(fps * 60.0 / _MIN_BPM))
    lags = np.arange(max(1, low), max(low + 2, high))
    bpm = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpm / _PRIOR_BPM) / 1.0) ** 2)
    weighted = auto[lags] * prior
    best = int(lags[int(np.argmax(weighted))])
    # Half the tempo is the usual mistake (every other beat lines up too): when the beats in
    # between are clearly there as well, the faster tempo is the beat.
    half = int(round(best / 2))
    if half >= max(1, low):
        near = auto[max(1, half - 1) : half + 2]
        if len(near) and float(near.max()) >= _DOUBLE_TEMPO_SHARE * float(auto[best]):
            best = max(1, half - 1) + int(np.argmax(near))
    # Parabolic refinement around the best lag.
    if 1 <= best < len(auto) - 1:
        a, b, c = auto[best - 1], auto[best], auto[best + 1]
        denominator = a - 2 * b + c
        shift = 0.5 * (a - c) / denominator if denominator else 0.0
    else:
        shift = 0.0
    return float(best + max(-0.5, min(0.5, shift))), float(max(0.0, min(1.0, auto[best])))


def _track_beats(onset: np.ndarray, period: float) -> list[int]:
    count = len(onset)
    score = onset.astype(np.float64).copy()
    back = np.full(count, -1, dtype=np.int64)
    low, high = int(round(period / 2)), int(round(period * 2))
    offsets = np.arange(low, high + 1)
    penalty = -_TIGHTNESS * (np.log(offsets / period) ** 2)
    for t in range(low, count):
        candidates = t - offsets
        valid = candidates >= 0
        if not valid.any():
            continue
        values = score[candidates[valid]] + penalty[valid]
        index = int(np.argmax(values))
        if values[index] > 0:
            score[t] = onset[t] + values[index]
            back[t] = candidates[valid][index]
    # Start from the best score in the last period, then follow the chain back.
    tail = max(0, count - int(round(period)))
    t = tail + int(np.argmax(score[tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = int(back[t])
    beats.reverse()
    return beats


def _beat_rms(samples: np.ndarray, rate: int, beats: list[float]) -> list[float]:
    edges = [int(b * rate) for b in beats] + [len(samples)]
    values = []
    for start, end in zip(edges, edges[1:]):
        piece = samples[start:end]
        values.append(float(np.sqrt(np.mean(piece**2))) if len(piece) else 0.0)
    return values


def _downbeat_phase(strength: list[float], rms: list[float]) -> int:
    best, best_value = 0, -1.0
    for phase in range(4):
        picked = range(phase, len(strength), 4)
        value = sum(strength[i] + rms[i] for i in picked) / max(1, len(picked))
        if value > best_value:
            best, best_value = phase, value
    return best


def _energy(rms: list[float]) -> list[float]:
    values = np.array(rms, dtype=np.float64)
    if not len(values):
        return []
    smooth = np.convolve(values, np.ones(8) / 8, mode="same")
    low, high = np.percentile(smooth, 10), np.percentile(smooth, 90)
    if high - low < 1e-9:
        return [0.5] * len(values)
    return [_clamp((v - low) / (high - low)) for v in smooth]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


# Cache


def load_or_analyze(
    job_dir: Path, music: Path | None, runner, log: Callable[[str, str], None] | None = None
) -> dict | None:
    """The beat grid of the job's music (cached), or None without music or when it cannot be
    read (the video is then planned as usual)."""
    log = log or (lambda message, level="INFO": None)
    if music is None or not music.is_file():
        return None
    stat = music.stat()
    key = {"name": music.name, "size": stat.st_size, "mtime": round(stat.st_mtime, 3)}
    cached = read_json(Path(job_dir) / MUSIC_ANALYSIS_FILE, None)
    if isinstance(cached, dict) and cached.get("key") == key and cached.get("version") == VERSION:
        return cached
    try:
        analysis = analyze_file(music, runner, Path(job_dir))
    except MusicError as exc:
        log(f"Music-led editing: {exc} The video is planned without the beat.", "WARNING")
        return None
    except Exception as exc:  # noqa: BLE001 - an unreadable file never stops the job
        log(f"Music-led editing: the music could not be read ({str(exc)[:160]}).", "WARNING")
        return None
    analysis["key"] = key
    analysis["name"] = music.name.split("_", 1)[-1]
    atomic_write_json(Path(job_dir) / MUSIC_ANALYSIS_FILE, analysis)
    log(
        f"Music: {analysis['bpm']:.0f} beats per minute, {len(analysis['beats'])} beats over "
        f"{analysis['duration']:.0f}s; the first bar starts at {analysis['start']:.2f}s",
        "INFO",
    )
    return analysis


# Planning


def target_seconds(analysis: dict, config, source_duration: float) -> float:
    """How long the music-led video should be: the music from its first bar, within the length
    asked for (automatic length: at most a quarter of the video)."""
    available = float(analysis["beats"][-1]) - float(analysis["start"])
    if config.length_mode == "auto":
        wanted = max(float(config.min_duration), source_duration * 0.25 if source_duration > 0 else available)
    else:
        wanted = float(config.max_duration)
    return max(_MIN_SLOT_SECONDS, min(available, wanted))


def slots(analysis: dict, target: float) -> list[tuple[float, float]]:
    """Music stretches (start, end), beat to beat from the first downbeat, as long as
    ``target`` in total: short where the music is energetic, long where it is calm; ending on a
    phrase (16 beats) when that keeps most of the length."""
    beats = [float(b) for b in analysis["beats"]]
    energy = list(analysis.get("energy") or [0.5] * len(beats))
    first = int(analysis.get("downbeat_phase") or 0)
    start = beats[first]
    result: list[tuple[float, float, int]] = []
    index = first
    while index < len(beats) - 1 and beats[index] - start < target - 0.25:
        level = energy[min(index, len(energy) - 1)]
        wanted = _CALM_CLIP + (_LIVELY_CLIP - _CALM_CLIP) * level
        beat = (beats[min(index + 4, len(beats) - 1)] - beats[index]) / 4 or 0.5
        count = min(_SLOT_BEATS, key=lambda n: abs(n * beat - wanted))
        while count * beat < _MIN_SLOT_SECONDS and count < _SLOT_BEATS[-1]:
            count *= 2
        # Never past the length aimed for: a shorter slot, or the end.
        while count > _SLOT_BEATS[0] and beats[min(len(beats) - 1, index + count)] - start > target + 0.25:
            count //= 2
        end_index = min(len(beats) - 1, index + count)
        if end_index == index or beats[end_index] - start > target + 0.25:
            break
        result.append((beats[index], beats[end_index], end_index))
        index = end_index
    # Prefer ending on a phrase.
    for cut in range(len(result), 0, -1):
        end, end_index = result[cut - 1][1], result[cut - 1][2]
        if (end_index - first) % _PHRASE_BEATS == 0 and end - start >= min(target, result[-1][1] - start) * _PHRASE_KEEP:
            result = result[:cut]
            break
    return [(round(a, 3), round(b, 3)) for a, b, _ in result]


def bias_candidates(candidates: list[dict], analysis: dict) -> list[dict]:
    """Lively music prefers lively moments and calm music calm ones (a small score shift)."""
    lively = float(analysis.get("lively") or 0.5)
    result = []
    for candidate in candidates:
        row = dict(candidate)
        metrics = row.get("metrics") or {}
        action = max(float(metrics.get("emotional_intensity") or 0.3), float(metrics.get("visual_importance") or 0.3))
        shift = 0.1 * (2 * lively - 1) * (2 * action - 1)
        row["score"] = round(_clamp(float(row.get("score") or 0) + shift), 4)
        result.append(row)
    return result


def planning_config(config, plan_slots: list[tuple[float, float]]):
    """The config the planner gets for music-led editing: about the slots' length in total (a
    little more, so there is enough to choose from), clips around the slot lengths."""
    total = sum(b - a for a, b in plan_slots)
    longest = max((b - a for a, b in plan_slots), default=4.0)
    shortest = min((b - a for a, b in plan_slots), default=2.0)
    return config.model_copy(
        update={
            "length_mode": "range",
            "min_duration": round(total, 2),
            "max_duration": round(total * 1.4 + longest, 2),
            "sequence_duration_mode": "custom",
            "max_sequence_duration": round(max(longest * 1.5, 2.0), 2),
            "min_sequence_duration": round(max(0.5, shortest * 0.5), 2),
            "shorts_count": 0,
        }
    )


def fit_clips(
    clips: list[dict],
    plan_slots: list[tuple[float, float]],
    candidates: dict[str, dict],
    source_lengths: dict[str, float],
) -> tuple[list[dict], list[tuple[float, float]]]:
    """Each planned clip cut to the length of its slot, in plan order, so every cut is on a
    beat. A clip longer than its slot keeps the part around its peak, starting at a cut point
    when one fits; a shorter one takes in the footage around it. Extra clips are dropped
    (weakest first, never the first or the last); with too few clips, the music ends earlier.
    Returns the fitted clips and the slots used."""
    clips = [dict(c) for c in clips]
    while len(clips) > len(plan_slots) and len(clips) > 2:
        middle = range(1, len(clips) - 1)
        weakest = min(middle, key=lambda i: float(clips[i].get("score") or 0))
        clips.pop(weakest)
    fitted: list[dict] = []
    used: list[tuple[str, float, float]] = []
    queue = list(clips)
    for slot_start, slot_end in plan_slots:
        length = slot_end - slot_start
        while queue:
            clip = queue.pop(0)
            placed = _fit_one(clip, length, candidates, source_lengths, used)
            if placed is not None:
                fitted.append(placed)
                used.append((str(placed.get("source_id") or ""), placed["start"], placed["end"]))
                break
        else:
            break
    return fitted, plan_slots[: len(fitted)]


def _fit_one(clip: dict, length: float, candidates: dict, source_lengths: dict, used: list) -> dict | None:
    sid = str(clip.get("source_id") or "")
    start, end = float(clip["start"]), float(clip["end"])
    candidate = candidates.get(str(clip.get("candidate_id") or "")) or {}
    peak = float(candidate.get("peak") or (start + end) / 2)
    limit = float(source_lengths.get(sid) or source_lengths.get("") or 0) or None
    if end - start >= length:
        options = [start] + [float(t) for t, _w in candidate.get("cuts") or [] if start < float(t) <= end - length]
        new_start = min(options, key=lambda s: abs(s + length / 2 - peak))
        new_end = new_start + length
    else:
        extra = length - (end - start)
        new_start, new_end = start - extra / 2, end + extra / 2
        if new_start < 0:
            new_end -= new_start
            new_start = 0.0
        if limit is not None and new_end > limit:
            new_start -= new_end - limit
            new_end = limit
        if new_start < 0:
            return None
    span = (round(new_start, 3), round(new_start + length, 3))
    if any(sid == other and span[0] < b and a < span[1] for other, a, b in used):
        return None
    result = dict(clip)
    result["start"], result["end"] = span
    result["join_previous"] = False
    result["duration_reason"] = f"{length:.2f}s — fitted to the beat of the music"
    return result
