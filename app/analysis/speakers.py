"""
Light speaker separation: who is speaking in each transcript segment, without large models.

Each spoken segment gets a voice print (MFCC means and spreads plus a pitch estimate) computed
with numpy from the analysis audio; segments are then grouped with agglomerative clustering.
The number of speakers (1-4) is the one whose groups separate best (silhouette score). It works
well when voices clearly differ (for example a man and a woman) and less well for similar
voices. Results are cached in speakers.json; speaker ids are per video ("v0:A", "v1:A", ...),
because the same letter in two recordings is not known to be the same person.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from app.utils.atomic import atomic_write_json

RESULT_FILE = "speakers.json"
VERSION = 1
SAMPLE_RATE = 16000
MAX_SPEAKERS = 4
MIN_SEGMENT = 0.8  # seconds of speech needed for a voice print
MIN_SILHOUETTE = 0.12  # below this, the "groups" are not really different voices
# Groups must be clearly farther apart than the spread inside them. One voice split in two
# stays near 1 (the split only follows noise); different voices are well above.
MIN_SEPARATION = 1.8
LETTERS = "ABCD"


# ---------------------------------------------------------------------------- audio features
def read_wav(path: Path) -> np.ndarray:
    """Mono float samples from a 16-bit PCM WAV (as the analysis writes it)."""
    with wave.open(str(path), "rb") as handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("expected 16-bit audio")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        # Rare (the analysis writes 16 kHz); a plain resample is enough for voice prints.
        positions = np.linspace(0, len(samples) - 1, int(len(samples) * SAMPLE_RATE / rate))
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    return samples


def _mel_filterbank(n_filters: int = 26, n_fft: int = 512, rate: int = SAMPLE_RATE) -> np.ndarray:
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    points = to_hz(np.linspace(to_mel(60), to_mel(rate / 2 - 200), n_filters + 2))
    bins = np.floor((n_fft + 1) * points / rate).astype(int)
    bank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left, centre, right = bins[i - 1], bins[i], bins[i + 1]
        for k in range(left, centre):
            bank[i - 1, k] = (k - left) / max(1, centre - left)
        for k in range(centre, right):
            bank[i - 1, k] = (right - k) / max(1, right - centre)
    return bank


_BANK = _mel_filterbank()
_N = np.arange(26)
_DCT = np.cos(np.pi / 26 * (_N[None, :] + 0.5) * np.arange(13)[:, None])


def voice_print(samples: np.ndarray) -> np.ndarray | None:
    """MFCC 1-12 mean and spread, plus pitch, over the louder (voiced) frames of a segment."""
    frame, hop = 400, 160  # 25 ms frames every 10 ms
    if len(samples) < frame * 5:
        return None
    emphasized = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    count = 1 + (len(emphasized) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    frames = emphasized[index] * np.hamming(frame)
    energy = (frames**2).mean(axis=1)
    voiced = energy > np.percentile(energy, 40)
    if voiced.sum() < 5:
        return None
    frames = frames[voiced]
    spectrum = np.abs(np.fft.rfft(frames, 512)) ** 2 / 512
    mel = np.log(np.maximum(spectrum @ _BANK.T, 1e-10))
    mfcc = (mel @ _DCT.T)[:, 1:13]
    # Pitch: the strongest autocorrelation lag between 60 and 400 Hz, on up to 60 voiced
    # frames spread over the segment (FFT autocorrelation keeps long recordings fast).
    rows = index[voiced]
    rows = rows[np.linspace(0, len(rows) - 1, min(60, len(rows))).astype(int)]
    raw = samples[rows]
    centred = raw - raw.mean(axis=1, keepdims=True)
    auto = np.fft.irfft(np.abs(np.fft.rfft(centred, 1024)) ** 2, 1024)
    low, high = SAMPLE_RATE // 400, SAMPLE_RATE // 60
    pitch = SAMPLE_RATE / (low + np.argmax(auto[:, low:high], axis=1))
    return np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0), [np.log(np.median(pitch))]])


# ---------------------------------------------------------------------------- grouping
def group_voices(prints: np.ndarray, max_speakers: int = MAX_SPEAKERS) -> np.ndarray:
    """A speaker number per voice print: the grouping (1-4 groups) that separates best."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    count = len(prints)
    if count < 4:
        return np.zeros(count, dtype=int)
    scaled = StandardScaler().fit_transform(prints)
    best_labels, best_score = np.zeros(count, dtype=int), MIN_SILHOUETTE
    for k in range(2, min(max_speakers, count - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
        if min(np.bincount(labels)) < 2:
            continue
        score = silhouette_score(scaled, labels)
        if separation(scaled, labels) < MIN_SEPARATION:
            continue
        if score > best_score + 0.02 * (k - 2):  # a little proof is needed for each extra speaker
            best_labels, best_score = labels, score
    return best_labels


def separation(points: np.ndarray, labels: np.ndarray) -> float:
    """The closest two group centres, divided by the average distance of points to their own
    group's centre."""
    groups = sorted(set(int(label) for label in labels))
    centres = np.array([points[labels == g].mean(axis=0) for g in groups])
    spread = np.mean([np.linalg.norm(points[labels == g] - centres[i], axis=1).mean() for i, g in enumerate(groups)])
    closest = min(np.linalg.norm(centres[i] - centres[j]) for i in range(len(groups)) for j in range(i + 1, len(groups)))
    return float(closest / max(spread, 1e-9))


def separate(samples: np.ndarray, segments: list[dict], source_id: str = "v0") -> dict:
    """Speakers of one video's transcript segments. Short segments take the label of the
    nearest longer one (they are usually the same person finishing a sentence)."""
    usable, prints = [], []
    for i, segment in enumerate(segments):
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < MIN_SEGMENT:
            continue
        piece = samples[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        found = voice_print(piece)
        if found is not None and np.all(np.isfinite(found)):
            usable.append(i)
            prints.append(found)
    labels_by_index: dict[int, int] = {}
    if prints:
        for i, label in zip(usable, group_voices(np.array(prints))):
            labels_by_index[i] = int(label)
    # Name speakers A, B, ... by how much they talk.
    talk: dict[int, float] = {}
    for i, label in labels_by_index.items():
        talk[label] = talk.get(label, 0.0) + float(segments[i]["end"]) - float(segments[i]["start"])
    order = {label: LETTERS[n] for n, (label, _) in enumerate(sorted(talk.items(), key=lambda kv: -kv[1]))}
    result_segments = []
    for i, segment in enumerate(segments):
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if i in labels_by_index:
            label = labels_by_index[i]
        elif labels_by_index:
            nearest = min(labels_by_index, key=lambda j: abs(j - i))
            label = labels_by_index[nearest]
        else:
            continue
        result_segments.append({"source_id": source_id, "start": start, "end": end, "speaker": f"{source_id}:{order[label]}"})
    speakers = []
    for label, letter in order.items():
        own = [s for s in result_segments if s["speaker"] == f"{source_id}:{letter}"]
        sample = max(own, key=lambda s: s["end"] - s["start"]) if own else None
        speakers.append(
            {
                "id": f"{source_id}:{letter}",
                "letter": letter,
                "source_id": source_id,
                "seconds": round(sum(s["end"] - s["start"] for s in own), 1),
                "sample": {"start": sample["start"], "end": min(sample["end"], sample["start"] + 8)} if sample else None,
            }
        )
    return {"speakers": speakers, "segments": result_segments}


# ---------------------------------------------------------------------------- job level
def _audio_for(job_dir: Path, source: dict, multi: bool, runner=None) -> Path | None:
    folder = (job_dir / "work" / str(source.get("id") or "v0")) if multi else job_dir
    wav = folder / "source" / "audio.wav"
    if wav.is_file():
        return wav
    from app.video.sources import working_path

    video = working_path(source)
    if not video.is_file():
        return None
    from app.video.ffmpeg_runner import FFmpegRunner
    from app.video.processor import VideoProcessor

    wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        (runner or FFmpegRunner()).run(VideoProcessor.extract_audio_command(video, wav))
    except Exception:  # noqa: BLE001 - no audio, no speakers
        return None
    return wav if wav.is_file() else None


def job_speakers(job_dir: Path, runner=None, refresh: bool = False) -> dict:
    """Speakers of a job (all its videos), computed once and cached in speakers.json."""
    from app.video.captions import load_transcript
    from app.video.sources import fallback_source, load_sources

    cached = job_dir / RESULT_FILE
    if cached.is_file() and not refresh:
        try:
            data = json.loads(cached.read_text(encoding="utf-8"))
            if data.get("version") == VERSION:
                return data
        except (OSError, json.JSONDecodeError):
            pass
    sources = load_sources(job_dir) or [fallback_source(job_dir / "source" / "missing", "")]
    multi = len(sources) > 1
    result = {"version": VERSION, "speakers": [], "segments": [], "note": ""}
    for source in sources:
        source_id = str(source.get("id") or "v0")
        segments = load_transcript(job_dir, source_id, multi).get("segments") or []
        if not segments:
            continue
        wav = _audio_for(job_dir, source, multi, runner)
        if wav is None:
            continue
        part = separate(read_wav(wav), segments, source_id)
        result["speakers"].extend(part["speakers"])
        result["segments"].extend(part["segments"])
    if not result["segments"]:
        result["note"] = "no speech"
    atomic_write_json(cached, result)
    return result


def dominant_speaker(segments: list[dict], start: float, end: float, source_id: str = "") -> str | None:
    """The speaker who talks most within start..end of one video (None without speech)."""
    talk: dict[str, float] = {}
    for segment in segments:
        if source_id and segment.get("source_id") not in (source_id, None):
            continue
        overlap = min(end, segment["end"]) - max(start, segment["start"])
        if overlap > 0:
            talk[segment["speaker"]] = talk.get(segment["speaker"], 0.0) + overlap
    return max(talk, key=talk.get) if talk else None


def filter_candidates(candidates: list[dict], speakers: dict, speaker_id: str, first_source: str = "v0") -> list[dict]:
    """Only the moments where `speaker_id` is the main speaker."""
    segments = speakers.get("segments") or []
    kept = []
    for candidate in candidates:
        source_id = str(candidate.get("source_id") or first_source)
        who = dominant_speaker(segments, float(candidate["start"]), float(candidate["end"]), source_id)
        if who == speaker_id:
            kept.append(candidate)
    return kept
