"""Music-led editing: the beat grid of a track, slots that follow its energy, clips fitted
exactly to them, the planner and render wiring, and re-planning when the music changes."""

from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from app.analysis import music
from app.analysis.optimizer import ClipOptimizer
from app.analysis.pipeline import music_led_plan, plan_from_cache
from app.models.schemas import HighlightConfig
from app.video.audio_mix import music_mix_command

RATE = 22050


def click_track(bpm=120.0, seconds=60.0, first=0.73, loud_from=30.0, seed=0):
    """Clicks on every beat, accented every fourth, twice as loud after ``loud_from``."""
    count = int(RATE * seconds)
    x = np.random.default_rng(seed).normal(0, 0.01, count).astype(np.float32)
    length = int(0.05 * RATE)
    shape = np.exp(-np.arange(length) / (0.01 * RATE))
    for i, beat in enumerate(np.arange(first, seconds - 0.1, 60.0 / bpm)):
        n = int(beat * RATE)
        accent = i % 4 == 0
        tone = np.sin(2 * np.pi * (1000 if accent else 600) * np.arange(length) / RATE)
        x[n : n + length] += (1.0 if accent else 0.4) * tone * shape
    x[int(loud_from * RATE) :] *= 2.5
    return x


@pytest.fixture(scope="module")
def grid():
    return music.analyze_samples(click_track(), RATE)


def test_the_beat_grid_follows_the_music(grid):
    assert grid["bpm"] == pytest.approx(120, abs=1.5)
    intervals = np.diff(grid["beats"])
    assert np.median(intervals) == pytest.approx(0.5, abs=0.02)
    assert grid["start"] == pytest.approx(0.73, abs=0.06), "the first accented beat starts the first bar"
    early, late = np.mean(grid["energy"][10:40]), np.mean(grid["energy"][-40:-10])
    assert late > early + 0.3, "the louder half has more energy"


@pytest.mark.parametrize("bpm", [90.0, 140.0])
def test_other_tempos_are_found(bpm):
    grid = music.analyze_samples(click_track(bpm=bpm, seconds=40, loud_from=40), RATE)
    assert grid["bpm"] == pytest.approx(bpm, rel=0.03)


def test_silence_or_too_short_music_is_refused():
    with pytest.raises(music.MusicError):
        music.analyze_samples(np.zeros(RATE * 2, dtype=np.float32), RATE)


def test_slots_are_quicker_where_the_music_is_energetic(grid):
    slots = music.slots(grid, 50)
    lengths = [b - a for a, b in slots]
    calm = [b - a for a, b in slots if a < 28]
    lively = [b - a for a, b in slots if a > 32]
    assert max(lively) < min(calm)
    assert slots[-1][1] - slots[0][0] <= 50.25, "never longer than asked"
    for a, b in slots:
        assert min(abs(np.array(grid["beats"]) - a)) < 1e-3 and min(abs(np.array(grid["beats"]) - b)) < 1e-3
    assert all(length >= 1.2 for length in lengths)


def test_slots_end_on_a_phrase_when_close_enough(grid):
    slots = music.slots(grid, 20)
    assert slots[-1][1] - slots[0][0] == pytest.approx(16, abs=0.1), "16 s is a 4-bar phrase and 80% of 20 s"


def _candidate(cid, start, end, score=0.6, **extra):
    return {"id": cid, "start": start, "end": end, "score": score, "text": "", "source_id": "", "metrics": {}, **extra}


def test_clips_are_fitted_exactly_to_their_slots():
    slots = [(0.7, 4.7), (4.7, 6.7), (6.7, 8.7)]
    clips = [
        {"candidate_id": "a", "start": 10.0, "end": 20.0, "score": 0.9, "source_id": ""},
        {"candidate_id": "b", "start": 30.0, "end": 31.0, "score": 0.5, "source_id": ""},
        {"candidate_id": "c", "start": 50.0, "end": 53.0, "score": 0.7, "source_id": ""},
    ]
    candidates = {
        "a": _candidate("a", 10, 20, peak=15.0, cuts=[[12.0, 1.0], [13.1, 1.0]]),
        "b": _candidate("b", 30, 31),
        "c": _candidate("c", 50, 53),
    }
    fitted, used = music.fit_clips(clips, slots, candidates, {"": 100.0})
    assert [round(c["end"] - c["start"], 3) for c in fitted] == [4.0, 2.0, 2.0]
    assert fitted[0]["start"] == 13.1, "starts at the cut point nearest the peak"
    assert (fitted[1]["start"], fitted[1]["end"]) == (29.5, 31.5), "a short clip takes in the footage around it"
    assert used == slots


def test_extra_clips_drop_the_weakest_middle_one():
    clips = [{"candidate_id": str(i), "start": i * 10.0, "end": i * 10.0 + 5, "score": s, "source_id": ""} for i, s in enumerate([0.9, 0.2, 0.8, 0.5])]
    fitted, used = music.fit_clips(clips, [(0, 2), (2, 4), (4, 6)], {}, {"": 100.0})
    assert [c["candidate_id"] for c in fitted] == ["0", "2", "3"]


def test_too_few_clips_end_the_music_earlier():
    clips = [{"candidate_id": "a", "start": 0.0, "end": 5.0, "score": 0.9, "source_id": ""}]
    fitted, used = music.fit_clips(clips, [(0, 2), (2, 4)], {}, {"": 100.0})
    assert len(fitted) == 1 and used == [(0, 2)]


def test_lively_music_prefers_lively_moments():
    calm_moment = _candidate("calm", 0, 5, 0.6, metrics={"emotional_intensity": 0.1, "visual_importance": 0.2})
    lively_moment = _candidate("lively", 10, 15, 0.6, metrics={"emotional_intensity": 0.9})
    biased = {c["id"]: c["score"] for c in music.bias_candidates([calm_moment, lively_moment], {"lively": 0.9})}
    assert biased["lively"] > 0.6 > biased["calm"]
    calm = {c["id"]: c["score"] for c in music.bias_candidates([calm_moment, lively_moment], {"lively": 0.1})}
    assert calm["calm"] > calm["lively"]


def test_music_led_plan_cuts_every_clip_on_a_beat(grid):
    candidates = [_candidate(f"c{i}", i * 12.0, i * 12.0 + 6, 0.5 + (i % 3) / 10) for i in range(40)]
    config = HighlightConfig(music_led=True, min_duration=20, max_duration=40)
    plan = music_led_plan(
        candidates, config, grid, ClipOptimizer(), source_duration=480, source_lengths={"": 480}, multi_source=False
    )
    assert plan is not None and plan.clips
    beats = np.array(grid["beats"])
    position = grid["start"]
    for clip in plan.clips:
        position += clip.end - clip.start
        assert min(abs(beats - position)) < 0.01, "each cut of the video is on a beat"
    assert plan.estimated_duration <= 40.3 and "beat" in plan.notes


def test_plan_from_cache_uses_the_music_and_falls_back_without_it(grid):
    candidates = [_candidate(f"c{i}", i * 12.0, i * 12.0 + 6, 0.6) for i in range(30)]
    cache = {"candidates": candidates, "candidates_all": candidates, "metadata": {"duration": 360}, "music_analysis": grid}
    config = HighlightConfig(music_led=True, min_duration=15, max_duration=25)
    with_music = plan_from_cache(cache, config, _settings())
    assert "beat" in with_music["notes"]
    without = plan_from_cache({**cache, "music_analysis": None}, config, _settings())
    assert "beat" not in (without["notes"] or "")


def _settings():
    from app.config import Settings

    return Settings(_env_file=None)


def test_music_led_turns_off_what_would_break_the_beat():
    config = HighlightConfig(music_led=True, transition="fade", tighten="pauses", shorts_count=3, video_brief=True, narration=True)
    assert (config.transition, config.tighten, config.shorts_count, config.video_brief, config.narration) == ("cut", "off", 0, False, False)


def test_the_mix_starts_on_the_first_bar_and_lowers_the_video(tmp_path):
    cmd = music_mix_command(tmp_path / "v.mp4", tmp_path / "m.mp3", tmp_path / "o.mp4", 30, 0.9, False, True, start=0.72, fade_in=0.02, source_volume=0.25)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "atrim=0.720:30.720" in graph and "asetpts=PTS-STARTPTS" in graph
    assert "afade=t=in:st=0:d=0.02" in graph and "volume=0.250" in graph
    muted = music_mix_command(tmp_path / "v.mp4", tmp_path / "m.mp3", tmp_path / "o.mp4", 30, 0.9, False, True, source_volume=0.0)
    assert "[0:a]" not in muted[muted.index("-filter_complex") + 1], "muted: music only"
    default = music_mix_command(tmp_path / "v.mp4", tmp_path / "m.mp3", tmp_path / "o.mp4", 30, 0.25, True, True)
    assert "atrim=0.000:30.000" in default[default.index("-filter_complex") + 1], "background music as before"


class WavRunner:
    """Writes the click track where the decode command wants its WAV."""

    def run(self, cmd):
        out = cmd[-1]
        data = (np.clip(click_track(seconds=30, loud_from=30), -1, 1) * 32767).astype(np.int16)
        with wave.open(out, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(RATE)
            handle.writeframes(data.tobytes())


def test_the_analysis_is_cached_by_the_music_file(tmp_path):
    track = tmp_path / "music" / "music_song.mp3"
    track.parent.mkdir()
    track.write_bytes(b"x" * 100)
    first = music.load_or_analyze(tmp_path, track, WavRunner())
    assert first["name"] == "song.mp3" and (tmp_path / music.MUSIC_ANALYSIS_FILE).exists()

    class Fails:
        def run(self, cmd):
            raise AssertionError("cached analysis must be reused")

    assert music.load_or_analyze(tmp_path, track, Fails())["beats"] == first["beats"]
    assert music.load_or_analyze(tmp_path, None, Fails()) is None
    assert not list(tmp_path.glob("*.wav")), "the decoded copy is removed"


def test_replacing_the_music_refits_the_clips(tmp_path, monkeypatch):
    import app.services.job_manager as jm
    from app import create_app
    from app.config import Settings, get_settings, set_settings
    from app.db import get_session
    from app.models.db_models import Job

    previous = get_settings()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9")
    settings.ensure_directories()
    set_settings(settings)
    try:
        app = create_app(settings, start_background=False)
        monkeypatch.setattr(jm, "_manager", None)
        monkeypatch.setattr("app.video.ffmpeg_runner.FFmpegRunner.run", lambda self, cmd, *a, **k: WavRunner().run(cmd))
        job_dir = settings.jobs_dir / "j1"
        job_dir.mkdir(parents=True)
        candidates = [_candidate(f"c{i}", i * 12.0, i * 12.0 + 6, 0.6) for i in range(30)]
        for name in ("candidates.json", "candidates_all.json"):
            (job_dir / name).write_text(json.dumps(candidates), encoding="utf-8")
        (job_dir / "metadata.json").write_text(json.dumps({"duration": 360}), encoding="utf-8")
        session = get_session()
        session.add(
            Job(
                id="j1",
                status="planning",
                analysis_status="completed",
                original_filename="v.mp4",
                input_path=str(tmp_path / "v.mp4"),
                config_json=HighlightConfig(music_led=True, min_duration=10, max_duration=20).model_dump_json(),
            )
        )
        session.commit()
        session.close()
        from io import BytesIO

        client = app.test_client()
        response = client.post("/api/jobs/j1/music", data={"file": (BytesIO(b"x" * 200), "song.mp3")})
        body = response.get_json()
        assert response.status_code == 200 and body["replanned"] is True and "beat" in body["notes"]
        plan = json.loads((job_dir / "clip_plan.json").read_text(encoding="utf-8"))
        assert plan["clips"] and "beat" in plan["notes"]
        removed = client.delete("/api/jobs/j1/music").get_json()
        assert removed["replanned"] is True and "beat" not in (removed["notes"] or "")
    finally:
        set_settings(previous)
