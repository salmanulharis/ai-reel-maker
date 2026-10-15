"""Regression tests for option handling, rendering, re-planning, cancel and settings fixes."""

import io
import json
import threading
from pathlib import Path

import pytest

from app.analysis.candidates import generate_candidates, rescore_candidate
from app.analysis.optimizer import ClipOptimizer
from app.analysis.pipeline import plan_from_cache
from app.analysis.scorer import HighlightScorer
from app.config import Settings
from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, RenderSettings, Transcript, VideoMetadata
from app.video.ffmpeg_runner import parse_probe
from app.video.quality import choose_canvas, clip_normalize_filters
from app.video.renderer import Renderer, build_clip_srt, build_srt


# Rendering


class _Runner:
    """Records commands, writes each output, and reports ``output_meta`` for the joined file."""

    def __init__(self, source_meta: dict[str, VideoMetadata], output_meta: VideoMetadata, fail_on=None):
        self.commands: list[list[str]] = []
        self.source_meta = source_meta
        self.output_meta = output_meta
        self.fail_on = fail_on

    def run(self, cmd):
        self.commands.append(cmd)
        if self.fail_on and self.fail_on(cmd):
            from app.video.ffmpeg_runner import FFmpegError

            raise FFmpegError("failed")
        Path(cmd[-1]).write_bytes(b"x" * 2000)

    def probe(self, path):
        key = Path(path).name
        if key in self.source_meta:
            return self.source_meta[key]
        return self.output_meta


def _meta(w=1920, h=1080, fps=30.0, audio=True):
    return VideoMetadata(duration=20, width=w, height=h, fps=fps, avg_fps=fps, has_audio=audio, size_bytes=2000)


@pytest.mark.parametrize(
    ("ratio", "size"),
    [("9:16", (608, 1080)), ("1:1", (1080, 1080)), ("4:5", (864, 1080)), ("16:9", (1920, 1080))],
)
def test_aspect_ratio_sets_the_output_canvas(ratio, size):
    canvas = choose_canvas([_meta()], RenderSettings(aspect_ratio=ratio))
    assert (canvas.width, canvas.height) == size
    assert canvas.fill is True


def test_original_aspect_keeps_letterboxing():
    canvas = choose_canvas([_meta()], RenderSettings())
    assert canvas.fill is False
    filters = clip_normalize_filters(_meta(1280, 720), canvas)
    assert any("decrease" in f for f in filters) and any(f.startswith("pad=") for f in filters)


def test_aspect_render_fills_canvas_and_passes_validation(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    runner = _Runner({"in.mp4": _meta()}, _meta(608, 1080))
    plan = ClipPlan(target_duration=5, estimated_duration=5, clips=[ClipPlanItem(start=0, end=5, reason="m", score=1)])
    result = Renderer(runner).render(
        source, plan, tmp_path / "out.mp4", RenderSettings(aspect_ratio="9:16"), tmp_path / "work", metadata=_meta()
    )
    assert result["validation"]["ok"] is True
    vf = next(cmd for cmd in runner.commands if "clip_000.mp4" in cmd[-1])
    chain = vf[vf.index("-vf") + 1]
    assert "force_original_aspect_ratio=increase" in chain and "crop=608:1080" in chain
    assert (tmp_path / "out.mp4").exists()
    assert not (tmp_path / ".out.rendering.mp4").exists()


def test_failed_validation_keeps_the_previous_output(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"good" * 1000)
    runner = _Runner({"in.mp4": _meta()}, _meta(320, 240))
    plan = ClipPlan(target_duration=5, estimated_duration=5, clips=[ClipPlanItem(start=0, end=5, reason="m", score=1)])
    from app.video.ffmpeg_runner import FFmpegError

    with pytest.raises(FFmpegError):
        Renderer(runner).render(source, plan, output, RenderSettings(), tmp_path / "work", metadata=_meta())
    assert output.read_bytes() == b"good" * 1000


def test_failed_join_leaves_no_partial_file(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    runner = _Runner({"in.mp4": _meta()}, _meta(), fail_on=lambda cmd: "concat" in cmd)
    plan = ClipPlan(target_duration=5, estimated_duration=5, clips=[ClipPlanItem(start=0, end=5, reason="m", score=1)])
    from app.video.ffmpeg_runner import FFmpegError

    with pytest.raises(FFmpegError):
        Renderer(runner).render(source, plan, tmp_path / "out.mp4", RenderSettings(), tmp_path / "w", metadata=_meta())
    assert not any(path.name.endswith(".rendering.mp4") for path in tmp_path.iterdir())


def test_burned_subtitles_are_timed_per_clip_and_per_source(tmp_path: Path):
    one, two = tmp_path / "one.mp4", tmp_path / "two.mp4"
    one.write_bytes(b"x" * 2000)
    two.write_bytes(b"x" * 2000)
    runner = _Runner({"one.mp4": _meta(), "two.mp4": _meta()}, _meta())
    plan = ClipPlan(
        target_duration=5,
        estimated_duration=5,
        clips=[
            ClipPlanItem(start=10, end=13, reason="m", score=1, source_id="v0"),
            ClipPlanItem(start=50, end=52, reason="m", score=1, source_id="v1"),
            ClipPlanItem(start=80, end=82, reason="m", score=1, source_id="v0"),
        ],
    )
    transcripts = {
        "v0": [{"start": 10.2, "end": 12.5, "text": "FIRST"}],
        "v1": [{"start": 50.1, "end": 51.8, "text": "SECOND"}],
    }
    work = tmp_path / "work"
    Renderer(runner).render(
        one,
        plan,
        tmp_path / "out.mp4",
        RenderSettings(subtitles="burned"),
        work,
        metadata=_meta(),
        source_lookup={"v0": one, "v1": two},
        subtitle_segments=transcripts,
    )
    first = (work / "clip_000.srt").read_text(encoding="utf-8")
    second = (work / "clip_001.srt").read_text(encoding="utf-8")
    assert "FIRST" in first and "00:00:00,200 --> 00:00:02,500" in first
    assert "SECOND" in second and "00:00:00,100 --> 00:00:01,800" in second
    # No speech in the third clip: no subtitle filter, because FFmpeg cannot open an empty file.
    third = next(cmd for cmd in runner.commands if "clip_002.mp4" in cmd[-1])
    assert "subtitles=" not in third[third.index("-vf") + 1]


def test_sidecar_srt_uses_each_clips_own_transcript(tmp_path: Path):
    plan = ClipPlan(
        target_duration=5,
        estimated_duration=5,
        clips=[
            ClipPlanItem(start=10, end=13, reason="m", score=1, source_id="v0"),
            ClipPlanItem(start=10, end=12, reason="m", score=1, source_id="v1"),
        ],
    )
    dest = build_srt(
        plan,
        {"v0": [{"start": 10, "end": 11, "text": "zero"}], "v1": [{"start": 10, "end": 11, "text": "one"}]},
        tmp_path / "out.srt",
    )
    text = dest.read_text(encoding="utf-8")
    assert "zero" in text and "one" in text
    assert "00:00:03,000 --> 00:00:04,000\none" in text


def test_single_source_srt_without_source_id(tmp_path: Path):
    clip = ClipPlanItem(start=5, end=8, reason="m", score=1)
    assert build_clip_srt(clip, [{"start": 6, "end": 7, "text": "hi"}], tmp_path / "c.srt")
    assert "00:00:01,000 --> 00:00:02,000" in (tmp_path / "c.srt").read_text(encoding="utf-8")
    plan = ClipPlan(target_duration=3, estimated_duration=3, clips=[clip])
    assert "hi" in build_srt(plan, {"v0": [{"start": 6, "end": 7, "text": "hi"}]}, tmp_path / "s.srt").read_text(
        encoding="utf-8"
    )


def test_video_without_sound_gets_silent_track_when_others_have_sound(tmp_path: Path):
    loud, quiet = tmp_path / "loud.mp4", tmp_path / "quiet.mp4"
    loud.write_bytes(b"x" * 2000)
    quiet.write_bytes(b"x" * 2000)
    runner = _Runner({"loud.mp4": _meta(), "quiet.mp4": _meta(audio=False)}, _meta())
    plan = ClipPlan(
        target_duration=5,
        estimated_duration=5,
        clips=[
            ClipPlanItem(start=0, end=3, reason="m", score=1, source_id="v0"),
            ClipPlanItem(start=0, end=3, reason="m", score=1, source_id="v1"),
        ],
    )
    Renderer(runner).render(
        loud, plan, tmp_path / "out.mp4", RenderSettings(), tmp_path / "w", metadata=_meta(),
        source_lookup={"v0": loud, "v1": quiet},
    )
    first = next(cmd for cmd in runner.commands if "clip_000.mp4" in cmd[-1])
    second = next(cmd for cmd in runner.commands if "clip_001.mp4" in cmd[-1])
    assert "anullsrc=r=48000:cl=stereo" not in first
    assert "anullsrc=r=48000:cl=stereo" in second
    # The silent input comes before the output options, so they apply to the output.
    assert second.index("anullsrc=r=48000:cl=stereo") < second.index("-t")


def test_hardware_fallback_reencodes_every_clip(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    runner = _Runner(
        {"in.mp4": _meta()},
        _meta(),
        fail_on=lambda cmd: "h264_nvenc" in cmd and "clip_001.mp4" in cmd[-1],
    )
    plan = ClipPlan(
        target_duration=6,
        estimated_duration=6,
        clips=[ClipPlanItem(start=0, end=3, reason="m", score=1), ClipPlanItem(start=5, end=8, reason="m", score=1)],
    )
    Renderer(runner).render(
        source, plan, tmp_path / "out.mp4", RenderSettings(quality="high"), tmp_path / "w",
        metadata=_meta(), hardware_encoder="h264_nvenc",
    )
    final = {}
    for cmd in runner.commands:
        if cmd[-1].endswith(("clip_000.mp4", "clip_001.mp4")):
            final[Path(cmd[-1]).name] = cmd[cmd.index("-c:v") + 1]
    assert final == {"clip_000.mp4": "libx264", "clip_001.mp4": "libx264"}


def test_rotated_phone_video_reports_its_displayed_size(tmp_path: Path):
    payload = {
        "format": {"duration": "5"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]}
        ],
    }
    meta = parse_probe(payload, tmp_path / "phone.mp4")
    assert (meta.width, meta.height) == (1080, 1920)
    assert meta.rotation == -90


# Options and planning


def test_no_limit_length_keeps_the_typed_minimum():
    config = HighlightConfig(min_duration=60, max_duration=30, length_mode="auto")
    assert config.min_duration == 60 and config.max_duration == 60
    ranged = HighlightConfig(min_duration=60, max_duration=30)
    assert (ranged.min_duration, ranged.max_duration) == (30, 60)


def test_priority_weight_shares_from_the_ui_keep_their_order():
    low = HighlightConfig(content_priorities=["action", "speech"], priority_weights={"action": 0.2, "speech": 0.6})
    assert low.priority_weights["action"] < low.priority_weights["speech"]
    # Older clients that send the 1-5 scale still work for 2-5.
    legacy = HighlightConfig(content_priorities=["action"], priority_weights={"action": 5})
    assert legacy.priority_weights["action"] == 1.0


def _scene_candidates(config: HighlightConfig):
    scenes = [
        {"scene_id": 0, "start": 0.0, "end": 6.0, "duration": 6.0},
        {"scene_id": 1, "start": 6.0, "end": 12.0, "duration": 6.0},
    ]
    visual = {0: {"visual_importance": 0.5}, 1: {"visual_importance": 0.7}}
    quality = {0: {"quality": 0.1}, 1: {"quality": 0.9}}
    return generate_candidates(
        Transcript(),
        scenes,
        visual,
        segment_scores=[],
        silences=[],
        source_duration=12.0,
        config=config,
        scorer=HighlightScorer(),
        scene_quality=quality,
    )


def test_replanning_scores_match_the_first_analysis():
    config = HighlightConfig(min_duration=5, max_duration=10)
    candidates = _scene_candidates(config)
    assert candidates
    scorer = HighlightScorer()
    for cand in candidates:
        assert rescore_candidate(cand, scorer, config) == pytest.approx(cand["score"], abs=1e-6)


def test_replanning_keeps_focus_boosts_and_uses_candidates_before_repeat_removal():
    config = HighlightConfig(min_duration=4, max_duration=8, remove_repetition=False, allow_repeated_content=True)
    base = {"metrics": {"importance": 0.5}, "score_factor": 1.0, "audio_signals": ["balanced"]}
    all_candidates = [
        {**base, "id": "a", "start": 0.0, "end": 4.0, "text": "", "topic": "a", "focus_match": 1.0},
        {**base, "id": "b", "start": 10.0, "end": 14.0, "text": "", "topic": "b"},
    ]
    cache = {"candidates_all": all_candidates, "candidates": all_candidates[1:], "metadata": {"duration": 30}}
    plan = plan_from_cache(cache, config, Settings(ollama_embed_model=""))
    scores = {clip["start"]: clip["score"] for clip in plan["clips"]}
    assert 0.0 in scores  # "a" was removed as a repeat earlier but is available again
    # Same metrics, so only the stored focus match can put "a" ahead of "b".
    assert scores[0.0] > scores.get(10.0, 0)


def test_replanning_old_scene_candidates_keeps_the_scene_floor():
    config = HighlightConfig()
    scorer = HighlightScorer()
    old_scene = {"source": "visual", "metrics": {"importance": 0.0, "production_quality": 0.7}}
    assert rescore_candidate(old_scene, scorer, config) == pytest.approx(0.42)
    old_window = {"source": "fallback", "metrics": {}, "score": 0.45}
    assert rescore_candidate(old_window, scorer, config) == 0.45


def test_audio_focus_without_its_signal_does_not_lower_scores():
    plain = _scene_candidates(HighlightConfig(min_duration=5, max_duration=10))
    natural = _scene_candidates(HighlightConfig(min_duration=5, max_duration=10, audio_focus="natural"))
    assert [c["score"] for c in plain] == [c["score"] for c in natural]


def test_short_combined_footage_with_one_useful_video_stays_inside_it():
    candidates = [{"id": "a", "start": 0.0, "end": 4.0, "score": 0.6, "source_id": "v0", "text": "", "topic": "a"}]
    plan = ClipOptimizer().select(
        candidates, min_duration=30, max_duration=30, source_duration=16.0, multi_source=True
    )
    assert [(clip.source_id, clip.end) for clip in plan.clips] == [("v0", 4.0)]


def test_short_combined_footage_keeps_each_video():
    candidates = [
        {"id": "a", "start": 0.0, "end": 4.0, "score": 0.6, "source_id": "v0", "text": "", "topic": "a"},
        {"id": "b", "start": 0.0, "end": 4.0, "score": 0.7, "source_id": "v1", "text": "", "topic": "b"},
    ]
    plan = ClipOptimizer().select(candidates, min_duration=30, max_duration=30, source_duration=16.0)
    assert sorted(clip.source_id for clip in plan.clips) == ["v0", "v1"]
    assert all(clip.end <= 4.0 for clip in plan.clips)


# Settings and API


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app import create_app
    from app.services import job_manager

    monkeypatch.setattr(job_manager, "_manager", None)
    settings = Settings(
        data_dir=tmp_path / "data",
        ollama_base_url="http://127.0.0.1:9",
        enable_whisper=False,
        enable_visual_analysis=False,
    )
    settings.ensure_directories()
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
    monkeypatch.setattr(job_manager, "_manager", None)


def _upload(client) -> str:
    data = {"file": (io.BytesIO(b"one"), "one.mp4"), "config": json.dumps({"min_duration": 20, "max_duration": 30})}
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 201
    return response.get_json()["id"]


def test_settings_reject_decimals_and_out_of_range_values(client):
    assert client.post("/api/settings", json={"ollama_vision_batch_size": 2.5}).status_code == 400
    assert client.post("/api/settings", json={"ollama_timeout": 0}).status_code == 400
    assert client.post("/api/settings", json={"output_crf": 60}).status_code == 400
    ok = client.post("/api/settings", json={"ollama_vision_batch_size": "3"})
    assert ok.status_code == 200
    assert ok.get_json()["settings"]["ollama_vision_batch_size"] == 3
    assert client.get("/api/settings").status_code == 200


def test_a_bad_saved_value_does_not_break_settings(client):
    from app.models.db_models import SettingRow, get_session

    session = get_session()
    session.merge(SettingRow(key="ollama_vision_batch_size", value="2.5"))
    session.merge(SettingRow(key="output_crf", value="21"))
    session.commit()
    session.close()
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.get_json()["settings"]["output_crf"] == 21


def test_settings_hide_the_secret_key(client):
    assert "secret_key" not in client.get("/api/settings").get_json()["settings"]


def test_other_websites_cannot_change_anything(client):
    blocked = client.post("/api/settings", json={"output_crf": 20}, headers={"Origin": "https://evil.example"})
    assert blocked.status_code == 403
    own = client.post("/api/settings", json={"output_crf": 20}, headers={"Origin": "http://localhost"})
    assert own.status_code == 200
    dev = client.post("/api/settings", json={"output_crf": 20}, headers={"Origin": "http://localhost:5173"})
    assert dev.status_code == 200


def test_render_refuses_a_clip_that_ends_before_it_starts(client):
    job_id = _upload(client)
    response = client.post(f"/api/jobs/{job_id}/render", json={"clips": [{"start": 5, "end": 2}]})
    assert response.status_code == 400
    assert "Clip 1" in response.get_json()["error"]


def test_shorter_lowers_a_fixed_length_and_updates_the_job(client):
    from app.services.job_manager import get_job_manager

    job_id = _upload(client)
    manager = get_job_manager()
    config = HighlightConfig(min_duration=30, max_duration=30)
    manager.update(job_id, config_json=config.model_dump_json(), analysis_status="completed", status="planning")
    job_dir = manager.job_dir(job_id)
    candidates = [
        {"id": f"c{i}", "start": i * 5.0, "end": i * 5.0 + 4, "score": 0.8, "metrics": {"importance": 0.8},
         "text": "", "topic": f"t{i}"}
        for i in range(12)
    ]
    (job_dir / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
    (job_dir / "metadata.json").write_text(json.dumps({"duration": 60}), encoding="utf-8")
    response = client.post(f"/api/jobs/{job_id}/regenerate", json={"preset": "more_concise"})
    assert response.status_code == 200
    job = response.get_json()["job"]
    assert job["max_duration"] == 21 and job["min_duration"] == 21
    assert job["config"]["max_duration"] == 21


def test_running_job_cannot_be_deleted(client):
    from app.services.job_manager import get_job_manager

    job_id = _upload(client)
    get_job_manager().update(job_id, status="rendering")
    assert client.delete(f"/api/jobs/{job_id}").status_code == 409
    get_job_manager().update(job_id, status="failed")
    # Cancelled but its run is still stopping: other actions wait for it.
    runner = object()
    get_job_manager().register_runner(job_id, runner)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 409
    assert client.post(f"/api/jobs/{job_id}/render", json={}).status_code == 409
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 409
    get_job_manager().unregister_runner(job_id, runner)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200


def test_cancel_stops_registered_runners():
    from app.services.job_manager import JobManager

    manager = JobManager(Settings())
    stopped = threading.Event()

    class FakeRunner:
        def cancel(self):
            stopped.set()

    runner = FakeRunner()
    manager.register_runner("job", runner)
    manager.update = lambda *args, **kwargs: None
    manager.cancel("job")
    assert stopped.wait(2)
    assert manager.cancel_event("job").is_set()
    manager.unregister_runner("job", runner)
    assert "job" not in manager._runners


def test_transcripts_load_per_source(tmp_path: Path):
    from app.workers.worker import load_transcripts

    (tmp_path / "work" / "v1").mkdir(parents=True)
    (tmp_path / "work" / "v1" / "transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "x"}]}), encoding="utf-8"
    )
    loaded = load_transcripts(tmp_path, [{"id": "v0"}, {"id": "v1"}])
    assert loaded == {"v0": [], "v1": [{"start": 0, "end": 1, "text": "x"}]}
    (tmp_path / "transcript.json").write_text(json.dumps({"segments": [{"start": 2, "end": 3, "text": "y"}]}), encoding="utf-8")
    assert load_transcripts(tmp_path, [{"id": "v0"}]) == {"v0": [{"start": 2, "end": 3, "text": "y"}]}


# New styles and priorities


NEW_STYLES = ["cinematic", "funny", "story", "trailer", "calm"]
NEW_PRIORITIES = ["people", "scenery", "fun", "pets", "music", "sports"]


def test_new_styles_and_priorities_are_accepted():
    for style in NEW_STYLES:
        assert HighlightConfig(style=style).style == style
    config = HighlightConfig(content_priorities=NEW_PRIORITIES)
    assert config.content_priorities == NEW_PRIORITIES


def test_every_style_and_priority_has_weights():
    from app.analysis.scorer import PRIORITY_EMPHASIS, STYLE_EMPHASIS

    for style in NEW_STYLES:
        assert STYLE_EMPHASIS[style]
    for name in NEW_PRIORITIES:
        assert PRIORITY_EMPHASIS[name]


def test_style_still_counts_when_priorities_are_chosen():
    from app.models.schemas import CandidateScore

    scorer = HighlightScorer()
    calm_shot = CandidateScore(visual_importance=0.9, production_quality=0.9, emotional_intensity=0.2)
    loud_peak = CandidateScore(visual_importance=0.5, production_quality=0.4, emotional_intensity=0.95, hook_strength=0.9)
    for priorities in (["connection"], ["action", "visual"], ["people"]):
        calm = [scorer.finalize(m, style="calm", priorities=priorities) for m in (calm_shot, loud_peak)]
        trailer = [scorer.finalize(m, style="trailer", priorities=priorities) for m in (calm_shot, loud_peak)]
        assert calm[0] - calm[1] > trailer[0] - trailer[1]
    # Auto style leaves the priority weights exactly as before.
    assert scorer._weights_for("auto", ["action"]).visual_importance == 0.26


def test_topic_priorities_steer_the_vision_prompt_and_picture_matching():
    from app.ai.prompts import picture_focus_texts, visual_analysis_focus, visual_focus_priorities

    assert visual_focus_priorities(["pets", "action", "connection"]) == ["connection", "pets"]
    focus = visual_analysis_focus("the beach", ["scenery"])
    assert focus.startswith("the beach.") and "landscapes" in focus
    assert picture_focus_texts("", ["action"]) == []
    assert picture_focus_texts("the beach", ["pets"]) == ["the beach", "a pet or an animal"]


def test_new_style_is_described_in_the_transcript_prompt():
    from app.ai.prompts import transcript_batch_prompt_fast

    assert "jokes" in transcript_batch_prompt_fast("funny", "", [])
    assert "Style: informative." in transcript_batch_prompt_fast("informative", "", [])


def test_visual_scores_are_redone_when_vision_priorities_change():
    from app.analysis.pipeline import _stored_visual_focus

    assert _stored_visual_focus(True) == ["connection"]
    assert _stored_visual_focus(False) == []
    assert _stored_visual_focus(None) == []
    assert _stored_visual_focus(["pets"]) == ["pets"]


def test_style_changes_clip_pacing():
    from app.analysis.sequence_limits import desired_auto_length

    candidate = {"start": 0, "end": 20, "metrics": {"importance": 0.6, "visual_importance": 0.6}, "text": ""}
    trailer, _ = desired_auto_length(candidate, 20, style="trailer")
    cinematic, reason = desired_auto_length(candidate, 20, style="cinematic")
    assert trailer <= 3.0 < cinematic
    assert "cinematic pacing" in reason


def test_several_focus_texts_count_by_the_best_match():
    import numpy as np

    from app.analysis import image_match

    class Matcher:
        def embed_images(self, paths):
            return np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]), [True, True, True]

        def embed_texts(self, texts):
            table = {"cat": [1.0, 0.0], "sea": [0.0, 1.0]}
            vectors = [table.get(text, [0.5, 0.5]) for text in texts]
            return np.array(vectors, dtype=np.float32)

    result = image_match.score_scenes(Matcher(), {0: "a", 1: "b", 2: "c"}, ["cat", "sea"])
    assert result[0]["focus"] >= result[2]["focus"] and result[1]["focus"] >= result[2]["focus"]


# What to leave out


def _cand(cid, start, end, text="", score=0.6, **metrics):
    base = {"importance": 0.6, "production_quality": 0.7}
    base.update(metrics)
    return {"id": cid, "start": start, "end": end, "text": text, "score": score, "metrics": base, "topic": cid}


def test_leave_out_config_is_cleaned():
    config = HighlightConfig(avoid=["text", "sponsor", "nonsense", "text"], avoid_custom="  the car ")
    assert config.avoid == ["sponsor", "text"]
    assert config.avoid_custom == "the car"
    assert HighlightConfig(avoid=["sponsor", "text"], use_speech=False).avoid == ["text"]


def test_speech_detectors_find_sponsor_swearing_filler_and_custom_words():
    from app.analysis import avoid

    cands = [
        _cand("a", 30, 34, "This video is sponsored by Acme, use code SAVE"),
        _cand("b", 40, 44, "what the hell, that was shit"),
        _cand("c", 50, 54, "um yeah", filler_score=0.85),
        _cand("d", 60, 64, "and then we drove the car home"),
        _cand("e", 70, 74, "a perfectly normal sentence about the weather today"),
    ]
    avoid.annotate(cands, ["sponsor", "swearing", "filler"], "the car", [], {}, {}, 200)
    by_id = {c["id"]: c["avoid_match"] for c in cands}
    assert by_id["a"]["sponsor"] == 1.0
    assert by_id["b"]["swearing"] == 1.0
    assert by_id["c"]["filler"] == 1.0
    assert by_id["d"]["custom"] == 1.0
    assert max(by_id["e"].values()) == 0.0


def test_intro_outro_needs_position_and_counts_signs_fully():
    from app.analysis import avoid

    cands = [
        _cand("hi", 1, 5, "hey guys welcome back to the channel"),
        _cand("start", 2, 6, "the river was calm"),
        _cand("mid", 100, 104, "thanks for watching"),
        _cand("end", 195, 199, "thanks for watching and subscribe"),
    ]
    avoid.annotate(cands, ["intro_outro"], "", [], {}, {}, 200)
    values = {c["id"]: c["avoid_match"]["intro_outro"] for c in cands}
    assert values == {"hi": 1.0, "start": 0.5, "mid": 0.0, "end": 1.0}


def test_picture_and_vision_matches_come_from_the_scenes_a_moment_covers():
    from app.analysis import avoid

    scenes = [{"scene_id": 0, "start": 0, "end": 10}, {"scene_id": 1, "start": 10, "end": 20}]
    match = {0: {"avoid": {"text": 0.9}}, 1: {"avoid": {"text": 0.1}}}
    visual = {0: {"avoid": 0.2}, 1: {"avoid": 0.95}}
    cands = [_cand("first", 1, 9), _cand("second", 11, 19)]
    avoid.annotate(cands, ["text"], "", scenes, visual, match, 100)
    assert cands[0]["avoid_match"]["text"] == 0.9 and cands[0]["avoid_match"]["vision"] == 0.2
    assert cands[1]["avoid_match"]["text"] == 0.1 and cands[1]["avoid_match"]["vision"] == 0.95


def test_avoid_ranks_lower_and_never_removes_with_a_safe_fallback():
    from app.analysis import avoid

    bad = {**_cand("bad", 0, 4, score=0.9), "avoid_match": {"sponsor": 1.0}}
    good = {**_cand("good", 10, 14, score=0.6), "avoid_match": {"sponsor": 0.0}}
    kept, removed, fell_back = avoid.apply([bad, good], ["sponsor"], "", "avoid")
    assert [c["id"] for c in kept] == ["good", "bad"] and removed == 0 and not fell_back
    kept, removed, _ = avoid.apply([bad, good], ["sponsor"], "", "never")
    assert [c["id"] for c in kept] == ["good"] and removed == 1
    kept, removed, fell_back = avoid.apply([bad], ["sponsor"], "", "never")
    assert len(kept) == 1 and removed == 0 and fell_back


def test_replanning_applies_leave_out_from_stored_matches():
    config = HighlightConfig(min_duration=4, max_duration=4, avoid=["sponsor"], avoid_mode="never")
    base = {"metrics": {"importance": 0.9}, "score_factor": 1.0, "audio_signals": ["balanced"], "text": ""}
    cands = [
        {**base, "id": "ad", "start": 0.0, "end": 4.0, "topic": "ad", "avoid_match": {"sponsor": 1.0}},
        {**base, "id": "ok", "start": 10.0, "end": 14.0, "topic": "ok", "metrics": {"importance": 0.5},
         "avoid_match": {"sponsor": 0.0}},
    ]
    plan = plan_from_cache({"candidates_all": cands, "metadata": {"duration": 60}}, config, Settings(ollama_embed_model=""))
    assert [clip["start"] for clip in plan["clips"]] == [10.0]


def test_picture_avoid_scores_use_contrast_margins():
    import numpy as np

    from app.analysis import avoid

    class Matcher:
        def embed_texts(self, texts):
            out = []
            for text in texts:
                out.append([1.0, 0.0] if "text" in text or "slide" in text else [0.0, 1.0])
            return np.array(out, dtype=np.float32)

    vectors = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    rows = avoid.picture_avoid_scores(Matcher(), vectors, avoid.picture_avoid_texts(["text"], ""))
    assert rows[0]["text"] == 1.0 and rows[1]["text"] == 0.0


def test_vision_prompt_and_schema_ask_for_avoid_only_when_needed():
    from app.ai.prompts import visual_batch_prompt, visual_batch_schema
    from app.analysis.avoid import vision_avoid_text

    assert vision_avoid_text(["sponsor"], "") == ""
    text = vision_avoid_text(["screens"], "the car")
    assert "screens" in text and "the car" in text
    assert '"avoid"' not in visual_batch_prompt([], "", "")
    assert '"avoid":0-1' in visual_batch_prompt([], "", text)
    assert "avoid" in visual_batch_schema(True)["properties"]["scenes"]["items"]["properties"]
    assert "avoid" not in visual_batch_schema(False)["properties"]["scenes"]["items"]["properties"]


def test_speech_model_is_told_what_to_rate_low():
    from app.ai.prompts import transcript_batch_prompt_fast
    from app.analysis.avoid import speech_avoid_text

    hint = speech_avoid_text(["sponsor", "dark"], "prices")
    assert hint == "sponsor messages and ads; prices"
    assert "Give low importance to: sponsor messages and ads; prices." in transcript_batch_prompt_fast("auto", "", [], hint)


# Batch 1 of the second audit


def test_probe_reads_non_ascii_file_names(tmp_path: Path):
    import shutil
    import subprocess

    from app.video.ffmpeg_runner import FFmpegRunner

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is not installed")
    clip = tmp_path / "മലയാളം_Łódź.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=10:duration=1",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    meta = FFmpegRunner().probe(clip)
    assert meta.duration > 0.5 and meta.width == 64


def test_nan_never_becomes_the_top_score():
    from app.ai.json_util import clamp_unit
    from app.utils.timefmt import clamp

    assert clamp(float("nan")) == 0.0
    assert clamp_unit("nan") == 0.0
    assert clamp(float("inf")) == 1.0


def test_intro_outro_ignores_position_only_scene_flags():
    from app.analysis import avoid

    scenes = [{"scene_id": 0, "start": 0, "end": 10}, {"scene_id": 1, "start": 590, "end": 600}]
    visual = {0: {"appears_intro": True, "score_source": "ai"}, 1: {"appears_ending": True}}
    cands = [_cand("open", 2, 9, "this mountain view is amazing"), _cand("goal", 591, 599, "what a goal")]
    avoid.annotate(cands, ["intro_outro"], "", scenes, visual, {}, 600)
    assert [c["avoid_match"]["intro_outro"] for c in cands] == [0.5, 0.5]


def test_talking_heads_reach_full_strength_without_picture_matching():
    from app.analysis import avoid

    scenes = [{"scene_id": 0, "start": 20, "end": 40}]
    visual = {0: {"action_strength": 0.1}}
    cand = _cand("talk", 22, 38, "so the main thing I want to explain is this", speech_presence=0.95)
    avoid.annotate([cand], ["talking_heads"], "", scenes, visual, {0: {}}, 120)
    assert cand["avoid_match"]["talking_heads"] >= avoid.NEVER_FROM


def test_out_of_range_vision_avoid_never_makes_scores_negative():
    from app.analysis import avoid
    from app.models.schemas import VisualAnalysis

    assert VisualAnalysis(avoid=8).avoid == 1.0
    assert VisualAnalysis(avoid="unknown").avoid is None
    stored = {**_cand("x", 0, 4, score=0.8), "avoid_match": {"vision": 8}}
    kept, _, _ = avoid.apply([stored], ["text"], "", "avoid")
    assert kept[0]["score"] >= 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("you can use the code from the repo", 0.0),
        ("the team's sponsors cheered", 0.0),
        ("this video is sponsored by Acme", 1.0),
        ("use code SAVE20 at checkout", 1.0),
        ("get 20 % off today", 1.0),
    ],
)
def test_sponsor_detector_avoids_common_false_positives(text, expected):
    from app.analysis import avoid

    cand = _cand("c", 50, 54, text)
    avoid.annotate([cand], ["sponsor"], "", [], {}, {}, 200)
    assert cand["avoid_match"]["sponsor"] == expected


def test_mild_words_and_mid_video_greetings_are_weaker_matches():
    from app.analysis import avoid

    mild = _cand("m", 50, 54, "oh damn that was close")
    hello = _cand("h", 100, 104, "and then she said hi everyone, look at this huge waterfall over here")
    avoid.annotate([mild], ["swearing"], "", [], {}, {}, 200)
    avoid.annotate([hello], ["filler"], "", [], {}, {}, 200)
    assert mild["avoid_match"]["swearing"] == 0.5
    assert hello["avoid_match"]["filler"] == 0.0


def test_single_video_keeps_its_full_metadata(tmp_path: Path):
    from app.analysis.pipeline import PipelineProgress, VideoAnalyzer

    analyzer = VideoAnalyzer(Settings(data_dir=tmp_path / "data", enable_repetition_analysis=False), runner=None, ollama=None)
    full = {"duration": 30.0, "width": 640, "height": 360, "fps": 30.0, "has_audio": True, "video_codec": "h264"}
    analyzer.analyze = lambda *args, **kwargs: {"metadata": dict(full), "candidates": []}
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    cache = analyzer.analyze_sources(
        tmp_path, [{"id": "v0", "path": str(source), "filename": "in.mp4"}], HighlightConfig(), PipelineProgress(lambda *_: None), {}
    )
    assert cache["metadata"]["has_audio"] is True and cache["metadata"]["video_codec"] == "h264"
    assert cache["metadata"]["source_count"] == 1


def test_job_routes_refuse_ids_that_leave_the_jobs_folder(client):
    assert client.get("/api/jobs/..%5C..%5Cx/logs").status_code == 404
    assert client.get("/api/media/..%5Cx/srt").status_code == 404


def test_invalid_config_values_are_refused_with_a_clear_message(client):
    data = {
        "file": (io.BytesIO(b"one"), "one.mp4"),
        "config": '{"min_duration": NaN, "max_duration": 30}',
    }
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "min duration must be a positive number"
    config = HighlightConfig(must_include=[[5, 1], [2, 4]], custom_focus="x" * 900)
    assert config.must_include == [(2.0, 4.0)] and len(config.custom_focus) == 500


def test_render_keeps_removed_clips_and_needs_one_kept(client):
    from app.services.job_manager import get_job_manager

    job_id = _upload(client)
    removed_only = [{"start": 1, "end": 3, "kept": False}]
    assert client.post(f"/api/jobs/{job_id}/render", json={"clips": removed_only}).status_code == 400
    both = [{"start": 1, "end": 3, "kept": False}, {"start": 5, "end": 8, "kept": True}]
    assert client.post(f"/api/jobs/{job_id}/render", json={"clips": both}).status_code == 200
    saved = get_job_manager().serialize(get_job_manager().get(job_id))["clips"]
    assert [(c["start"], c["kept"]) for c in saved] == [(1.0, False), (5.0, True)]
