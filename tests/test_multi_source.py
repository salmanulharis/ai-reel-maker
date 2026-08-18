from pathlib import Path

from app.analysis.optimizer import ClipOptimizer
from app.models.schemas import ClipPlan, ClipPlanItem, RenderSettings, VideoMetadata
from app.video.renderer import Renderer
from app.video.sources import tag_candidates


def test_tag_candidates_adds_source_identity():
    tagged = tag_candidates(
        [{"id": "t0", "start": 1, "end": 4, "topic": "scene-2-1", "text": "hello"}],
        {"id": "v1", "filename": "b-roll.mp4", "path": "/tmp/b-roll.mp4"},
    )
    assert tagged[0]["source_id"] == "v1"
    assert tagged[0]["source_name"] == "b-roll.mp4"
    assert tagged[0]["id"] == "v1:t0"
    assert tagged[0]["topic"].startswith("v1-")


def test_optimizer_skips_weak_video_and_duplicate_across_sources():
    optimizer = ClipOptimizer()
    weak = {
        "id": "v0:a",
        "start": 2,
        "end": 8,
        "score": 0.32,
        "topic": "filler",
        "text": "um welcome thanks for watching please subscribe",
        "reason": "Intro",
        "source_id": "v0",
        "source_name": "intro.mp4",
        "metrics": {"filler_score": 0.8, "importance": 0.2},
    }
    best = {
        "id": "v1:a",
        "start": 10,
        "end": 18,
        "score": 0.94,
        "topic": "jump",
        "text": "the rider clears the gap",
        "reason": "Action",
        "source_id": "v1",
        "source_name": "action.mp4",
        "metrics": {"emotional_intensity": 0.9, "visual_importance": 0.88},
    }
    copycat = {
        "id": "v2:a",
        "start": 4,
        "end": 12,
        "score": 0.9,
        "topic": "jump",
        "text": "another rider clearing the same gap",
        "reason": "Action",
        "source_id": "v2",
        "source_name": "action-take2.mp4",
        "metrics": {"emotional_intensity": 0.85, "visual_importance": 0.8},
    }
    extra = {
        "id": "v1:b",
        "start": 30,
        "end": 36,
        "score": 0.82,
        "topic": "landing",
        "text": "the landing is clean and the crowd reacts",
        "reason": "Reaction",
        "source_id": "v1",
        "source_name": "action.mp4",
        "metrics": {"emotional_intensity": 0.7, "importance": 0.7},
    }
    plan = optimizer.select(
        [weak, best, copycat, extra],
        min_duration=8,
        max_duration=20,
        sequence_duration_mode="auto",
        allow_repeated_content=False,
        similarity_fn=lambda x, y: 0.86 if x["topic"] == y["topic"] else 0.1,
    )
    names = {c.source_name for c in plan.clips}
    ids = {c.candidate_id for c in plan.clips}
    assert "action.mp4" in names
    assert "intro.mp4" not in names
    assert "v1:a" in ids
    assert "v2:a" not in ids
    assert all(c.source_id for c in plan.clips)


def test_optimizer_keeps_overlapping_times_from_different_videos():
    optimizer = ClipOptimizer()
    plan = optimizer.select(
        [
            {
                "id": "v0:a",
                "start": 10,
                "end": 16,
                "score": 0.9,
                "topic": "leaf",
                "text": "a leaf falls through the forest",
                "reason": "Visual",
                "source_id": "v0",
                "source_name": "dolby.mp4",
                "metrics": {"visual_importance": 0.9},
            },
            {
                "id": "v1:a",
                "start": 10,
                "end": 16,
                "score": 0.88,
                "topic": "swing",
                "text": "spider-man swings between buildings",
                "reason": "Action",
                "source_id": "v1",
                "source_name": "spiderman.mp4",
                "metrics": {"emotional_intensity": 0.9},
            },
        ],
        min_duration=8,
        max_duration=20,
        sequence_duration_mode="auto",
        allow_repeated_content=False,
    )
    ids = {c.candidate_id for c in plan.clips}
    assert "v0:a" in ids
    assert "v1:a" in ids


def test_renderer_cuts_each_clip_from_its_source(tmp_path: Path):
    first = tmp_path / "v0.mp4"
    second = tmp_path / "v1.mp4"
    first.write_bytes(b"a" * 2000)
    second.write_bytes(b"b" * 2000)
    output = tmp_path / "out.mp4"
    work = tmp_path / "work"
    commands: list[list[str]] = []
    inputs: list[str] = []

    class FakeRunner:
        def run(self, cmd):
            commands.append(cmd)
            if cmd and cmd[0] == "ffmpeg" and "concat" not in cmd:
                inputs.append(cmd[cmd.index("-i") + 1])
            Path(cmd[-1]).write_bytes(b"x" * 2000)

        def probe(self, path):
            return VideoMetadata(duration=5.0, width=1920, height=1080, fps=24, has_audio=True, size_bytes=2000)

    plan = ClipPlan(
        target_duration=8,
        estimated_duration=8,
        clips=[
            ClipPlanItem(start=1, end=5, reason="hook", score=0.9, source_id="v0", source_name="one.mp4"),
            ClipPlanItem(start=3, end=7, reason="action", score=0.8, source_id="v1", source_name="two.mp4"),
        ],
    )
    Renderer(FakeRunner()).render(
        first,
        plan,
        output,
        RenderSettings(codec="h264"),
        work,
        metadata=VideoMetadata(duration=20, width=1920, height=1080, has_audio=True),
        source_lookup={"v0": first, "v1": second},
    )
    assert str(first) in inputs
    assert str(second) in inputs
    concat = next(cmd for cmd in commands if "concat" in cmd)
    assert "copy" in concat
    joined = " ".join(" ".join(cmd) for cmd in commands)
    assert "fps=30" not in joined
    assert "fps=24" in joined
    cut = next(cmd for cmd in commands if cmd[0] == "ffmpeg" and "concat" not in cmd)
    assert cut[cut.index("-preset") + 1] == "slow"
    assert cut[cut.index("-crf") + 1] == "14"
