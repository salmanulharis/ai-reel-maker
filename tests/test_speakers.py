"""Light speaker separation (numpy voice prints + clustering)."""

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from app import create_app
from app.analysis.pipeline import plan_from_cache
from app.analysis.speakers import (
    SAMPLE_RATE,
    dominant_speaker,
    filter_candidates,
    job_speakers,
    read_wav,
    separate,
    voice_print,
)
from app.config import Settings
from app.models.schemas import HighlightConfig

RNG = np.random.default_rng(7)


def voice(pitch: float, weights, seconds: float) -> np.ndarray:
    """A crude voiced sound: harmonics of `pitch` with a timbre, wobbling like speech."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wobble = pitch * (1 + 0.03 * np.sin(2 * np.pi * 3 * t))
    phase = 2 * np.pi * np.cumsum(wobble) / SAMPLE_RATE
    sound = sum(w * np.sin((k + 1) * phase) for k, w in enumerate(weights))
    envelope = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 2.5 * t))
    return (0.25 * sound * envelope / max(1, len(weights)) + 0.003 * RNG.standard_normal(len(t))).astype(np.float32)


LOW = (110, [1.0, 0.8, 0.6, 0.4, 0.3, 0.2])
HIGH = (240, [1.0, 0.3, 0.1, 0.05])


def conversation(turns):
    """Audio and transcript segments for turns of (voice, seconds), with short gaps."""
    pieces, segments, cursor = [], [], 0.0
    for (pitch, weights), seconds in turns:
        pieces.append(voice(pitch, weights, seconds))
        segments.append({"start": round(cursor, 3), "end": round(cursor + seconds, 3), "text": "words"})
        cursor += seconds
        pieces.append(np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32))
        cursor += 0.3
    return np.concatenate(pieces), segments


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


TURNS = [(LOW, 3), (HIGH, 2.5), (LOW, 4), (HIGH, 3), (LOW, 2.5), (HIGH, 2), (LOW, 3), (HIGH, 0.5)]


def test_two_voices_are_told_apart():
    audio, segments = conversation(TURNS)
    result = separate(audio, segments)
    labels = [s["speaker"] for s in result["segments"]]
    assert len(set(labels)) == 2
    low = {labels[i] for i, (v, _) in enumerate(TURNS) if v is LOW}
    high = {labels[i] for i, (v, _) in enumerate(TURNS[:-1]) if v is HIGH}
    assert len(low) == 1 and len(high) == 1 and low != high
    # The one who talks most is A; the short last turn borrows its neighbour's label.
    assert low == {"v0:A"} and labels[-1] == labels[-2]
    speakers = {s["id"]: s for s in result["speakers"]}
    assert speakers["v0:A"]["seconds"] > speakers["v0:B"]["seconds"]
    assert speakers["v0:A"]["sample"]["end"] - speakers["v0:A"]["sample"]["start"] <= 8


def test_one_voice_is_one_speaker():
    audio, segments = conversation([(LOW, 3)] * 6)
    result = separate(audio, segments, "v1")
    assert {s["speaker"] for s in result["segments"]} == {"v1:A"} and len(result["speakers"]) == 1


def test_silence_and_short_audio_have_no_voice_print():
    assert voice_print(np.zeros(100, dtype=np.float32)) is None
    assert separate(np.zeros(SAMPLE_RATE, dtype=np.float32), [{"start": 0, "end": 0.5, "text": "x"}]) == {"speakers": [], "segments": []}


def test_dominant_speaker_and_filter():
    segments = [
        {"source_id": "v0", "start": 0, "end": 5, "speaker": "v0:A"},
        {"source_id": "v0", "start": 5, "end": 6, "speaker": "v0:B"},
        {"source_id": "v1", "start": 0, "end": 9, "speaker": "v1:A"},
    ]
    assert dominant_speaker(segments, 3, 6, "v0") == "v0:A"
    assert dominant_speaker(segments, 5, 6, "v0") == "v0:B"
    assert dominant_speaker(segments, 20, 30, "v0") is None
    candidates = [{"start": 0, "end": 4}, {"start": 5, "end": 6}, {"start": 1, "end": 2, "source_id": "v1"}]
    assert filter_candidates(candidates, {"segments": segments}, "v0:B") == [candidates[1]]


def test_read_wav_stereo_and_rate(tmp_path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(np.full(1600, 16384, dtype="<i2").tobytes())
    samples = read_wav(path)
    assert len(samples) == 1600 and np.allclose(samples, 0.5, atol=0.01)


@pytest.fixture()
def client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9", enable_whisper=False)
    settings.ensure_directories()
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_job_speakers_route_and_cache(client):
    from app.services.job_manager import get_job_manager

    data = {"file": (io.BytesIO(b"v"), "talk.mp4"), "config": json.dumps({})}
    job_id = client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()["id"]
    job_dir = get_job_manager().job_dir(job_id)
    audio, segments = conversation(TURNS)
    write_wav(job_dir / "source" / "audio.wav", audio)
    (job_dir / "transcript.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")
    found = client.get(f"/api/jobs/{job_id}/speakers").get_json()
    assert [s["id"] for s in found["speakers"]] == ["v0:A", "v0:B"]
    assert (job_dir / "speakers.json").is_file()
    # Cached: the audio is no longer needed.
    (job_dir / "source" / "audio.wav").unlink()
    assert client.get(f"/api/jobs/{job_id}/speakers").get_json() == found
    assert client.get("/api/jobs/missing/speakers").status_code == 404


def test_no_speech_gives_no_speakers(tmp_path):
    result = job_speakers(tmp_path)
    assert result["speakers"] == [] and result["note"] == "no speech"


def test_planning_for_one_speaker():
    def cand(cid, start, score):
        return {"id": cid, "start": start, "end": start + 4, "score": score, "text": f"moment {cid} words",
                "reason": "r", "metrics": {"importance": score}}

    cache = {
        "candidates": [cand("a", 0, 0.9), cand("b", 10, 0.8), cand("c", 20, 0.7)],
        "speakers": {"segments": [
            {"source_id": "v0", "start": 0, "end": 5, "speaker": "v0:A"},
            {"source_id": "v0", "start": 10, "end": 15, "speaker": "v0:B"},
            {"source_id": "v0", "start": 20, "end": 25, "speaker": "v0:A"},
        ]},
        "metadata": {"duration": 60},
    }
    settings = Settings(ollama_base_url="http://127.0.0.1:9")
    config = HighlightConfig(min_duration=4, max_duration=12, max_sequence_duration=5, sequence_duration_mode="custom")
    only_b = plan_from_cache(dict(cache), config.model_copy(update={"focus_speaker": "v0:B"}), settings)
    assert [c["start"] for c in only_b["clips"]] == [10]
    nobody = plan_from_cache(dict(cache), config.model_copy(update={"focus_speaker": "v0:C"}), settings)
    # Nobody matches: everyone's moments are used, and the plan says so.
    assert nobody["clips"] and nobody["notes"].startswith("No strong moments")
