"""Several separate shorts from one video."""

import io
import json
import zipfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from app import create_app
from app.analysis.optimizer import ClipOptimizer
from app.config import Settings
from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, VideoMetadata
from app.services.outputs import ShortParamError, find_output, parse_short
from app.workers.worker import remove_stale_outputs, render_groups


# Unrelated sentences, so no moment counts as a repeat of another.
WORDS = {
    "a1": "rockets launch from coastal pads", "a2": "bread rises overnight in cool kitchens",
    "a3": "glaciers carve deep valleys slowly", "b1": "violins need careful humidity control",
    "b2": "marathon runners pace their first mile", "b3": "honeybees dance to share directions",
    "c1": "tides follow the pull of the moon", "c2": "chess openings reward early development",
    "c3": "volcanic soil grows rich vineyards",
}


def _c(cid, start, end, score, text=None, source_id=""):
    return {
        "id": cid,
        "start": start,
        "end": end,
        "score": score,
        "text": text or WORDS[cid],
        "topic": cid,
        "reason": "Key moment",
        "source_id": source_id,
    }


# Three separate areas of a long video, each with a few strong moments.
CANDIDATES = [
    _c("a1", 10, 18, 0.95), _c("a2", 20, 28, 0.7), _c("a3", 30, 38, 0.6),
    _c("b1", 400, 408, 0.9), _c("b2", 410, 418, 0.65), _c("b3", 420, 428, 0.55),
    _c("c1", 800, 808, 0.85), _c("c2", 810, 818, 0.6), _c("c3", 820, 828, 0.5),
]
OPTIONS = dict(min_duration=15, max_duration=25, max_sequence_duration=8, sequence_duration_mode="custom")


def test_shorts_are_separate_local_and_best_first():
    plan = ClipOptimizer().select_shorts(CANDIDATES, 3, **OPTIONS)
    groups = {}
    for clip in plan.clips:
        groups.setdefault(clip.short_index, []).append(clip)
    assert sorted(groups) == [0, 1, 2]
    # Short 1 is built around the best moment, short 2 around the next best area, and so on.
    assert groups[0][0].start < 100 and groups[1][0].start > 300 and groups[2][0].start > 700
    for clips in groups.values():
        assert 15 * 0.75 <= sum(c.duration for c in clips) <= 25 + 0.01
        # Each short stays in one area of the video, in time order.
        assert max(c.end for c in clips) - min(c.start for c in clips) <= 3 * 25
        assert [c.start for c in clips] == sorted(c.start for c in clips)
    spans = sorted((c.start, c.end) for c in plan.clips)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert plan.notes == "3 shorts"


def test_fewer_shorts_when_there_is_not_enough_material():
    plan = ClipOptimizer().select_shorts(CANDIDATES[:3], 4, **OPTIONS)
    assert {c.short_index for c in plan.clips} == {0}
    assert "asked for 4" in plan.notes
    assert ClipOptimizer().select_shorts([], 3, **OPTIONS).clips == []


def test_must_exclude_applies_to_shorts():
    plan = ClipOptimizer().select_shorts(CANDIDATES, 3, **{**OPTIONS, "must_exclude": [(0, 100)]})
    assert all(c.start >= 100 for c in plan.clips)


def test_plan_dispatches_on_the_shorts_setting():
    config = HighlightConfig(min_duration=15, max_duration=25, max_sequence_duration=8, sequence_duration_mode="custom")
    single = ClipOptimizer().plan(CANDIDATES, config)
    assert {c.short_index for c in single.clips} == {0}
    shorts = ClipOptimizer().plan(CANDIDATES, config.model_copy(update={"shorts_count": 3}))
    assert {c.short_index for c in shorts.clips} == {0, 1, 2}


@pytest.mark.parametrize("given, expected", [(0, 0), (1, 0), (2, 2), (5, 5), (15, 10), (-3, 0)])
def test_shorts_count_is_kept_in_range(given, expected):
    assert HighlightConfig(shorts_count=given).shorts_count == expected


def test_shorts_always_use_a_length_range():
    assert HighlightConfig(shorts_count=3, length_mode="auto").length_mode == "range"
    assert HighlightConfig(length_mode="auto").length_mode == "auto"


def test_render_groups(tmp_path):
    def item(start, short=0):
        return ClipPlanItem(start=start, end=start + 2, reason="r", short_index=short)

    single = render_groups(ClipPlan(target_duration=4, estimated_duration=4, clips=[item(0), item(5)]), HighlightConfig(), tmp_path, "mp4")
    assert [(g["label"], g["kind"], len(g["clips"]), g["output"].name, g["srt"].name) for g in single] == [
        ("highlight", "highlight", 2, "highlight.mp4", "highlight.srt")
    ]
    # Short 2 (index 1) lost all its clips: the others keep their numbers.
    plan = ClipPlan(target_duration=4, estimated_duration=6, clips=[item(0, 0), item(9, 2), item(5, 0)])
    groups = render_groups(plan, HighlightConfig(shorts_count=3), tmp_path, "webm")
    assert [(g["label"], g["short"], [c.start for c in g["clips"]], g["output"].name) for g in groups] == [
        ("short_01", 0, [0, 5], "short_01.webm"),
        ("short_03", 2, [9], "short_03.webm"),
    ]
    assert render_groups(ClipPlan(target_duration=1, estimated_duration=0, clips=[]), HighlightConfig(), tmp_path, "mp4") == []


def test_remove_stale_outputs(tmp_path):
    for name in ("highlight.mp4", "highlight.srt", "short_01.mp4", "short_02.mp4", "short_02.srt", "post_text.json"):
        (tmp_path / name).write_bytes(b"x")
    remove_stale_outputs(tmp_path, {"short_01.mp4"})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["post_text.json", "short_01.mp4"]


def test_parse_short_and_find_output(tmp_path):
    assert parse_short(None) is None and parse_short("") is None and parse_short("3") == 3
    for bad in ("0", "11", "x", "-1"):
        with pytest.raises(ShortParamError):
            parse_short(bad)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "highlight.mp4").write_bytes(b"x")
    (outputs / "short_02.webm").write_bytes(b"x")
    (outputs / "short_02.srt").write_bytes(b"x")
    assert find_output(tmp_path).name == "highlight.mp4"
    assert find_output(tmp_path, 2).name == "short_02.webm"
    assert find_output(tmp_path, 3) is None
    assert find_output(tmp_path / "missing") is None


def test_old_database_gets_the_short_column(tmp_path):
    path = tmp_path / "old.db"
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE selected_clips (id INTEGER PRIMARY KEY, job_id VARCHAR(40), start FLOAT, \"end\" FLOAT, "
                "score FLOAT, reason TEXT, transcript TEXT, sort_order INTEGER, kept BOOLEAN)"
            )
        )
        conn.execute(text("INSERT INTO selected_clips (job_id, start, \"end\", score, reason, transcript, sort_order, kept) VALUES ('j', 1, 2, 0.5, 'r', '', 0, 1)"))
    engine.dispose()
    from app.models import db_models

    db_models.init_db(path)
    with db_models.engine.begin() as conn:
        names = {row[1] for row in conn.execute(text("PRAGMA table_info(selected_clips)")).fetchall()}
        value = conn.execute(text("SELECT short_index FROM selected_clips")).scalar()
    assert "short_index" in names and value == 0


# ------------------------------------------------------------------------------ routes


@pytest.fixture()
def client(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9", enable_whisper=False)
    settings.ensure_directories()
    import app.services.editor_export as export

    monkeypatch.setattr(
        export,
        "_default_probe",
        lambda path: VideoMetadata(duration=900, width=1280, height=720, fps=25, has_audio=True, channels=2),
    )
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _job(client, config) -> str:
    data = {"file": (io.BytesIO(b"video"), "talk.mp4"), "config": json.dumps(config)}
    return client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()["id"]


def test_clips_keep_their_short(client):
    job_id = _job(client, {"shorts_count": 2})
    clips = [
        {"start": 10, "end": 12, "reason": "r", "short_index": 0},
        {"start": 400, "end": 403, "reason": "r", "short_index": 1},
        {"start": 50, "end": 52, "reason": "r", "short_index": "bad"},
    ]
    client.post(f"/api/jobs/{job_id}/clips", json={"clips": clips})
    saved = client.get(f"/api/jobs/{job_id}").get_json()["clips"]
    assert [c["short_index"] for c in saved] == [0, 1, 0]


def test_output_routes_take_a_short_number(client):
    from app.services.job_manager import get_job_manager

    job_id = _job(client, {"shorts_count": 2})
    outputs = get_job_manager().job_dir(job_id) / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "short_02.mp4").write_bytes(b"second short")
    assert client.get(f"/api/media/{job_id}/output?short=2").data == b"second short"
    assert client.get(f"/api/media/{job_id}/output?short=1").status_code == 404
    assert client.get(f"/api/media/{job_id}/output?short=x").status_code == 400
    download = client.get(f"/api/jobs/{job_id}/download?short=2")
    assert download.status_code == 200 and "short_02.mp4" in download.headers["Content-Disposition"]
    assert client.get(f"/api/jobs/{job_id}/download?short=99").status_code == 400
    assert client.get(f"/api/jobs/{job_id}/cover?t=1&short=1").get_json()["error"] == "no output"


def test_editor_export_has_a_timeline_per_short(client):
    job_id = _job(client, {"shorts_count": 2, "max_sequence_duration": 60})
    clips = [
        {"start": 10, "end": 12, "reason": "r", "short_index": 0},
        {"start": 400, "end": 403, "reason": "r", "short_index": 1},
        {"start": 20, "end": 21, "reason": "r", "short_index": 0},
    ]
    client.post(f"/api/jobs/{job_id}/clips", json={"clips": clips})
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job_id}/export/editor").data))
    names = set(archive.namelist())
    assert {"README.txt", "short_01/highlight.xml", "short_01/highlight.fcpxml", "short_02/highlight.edl"} <= names
    assert "2 shorts" in archive.read("README.txt").decode()
    assert "talk short 2" in archive.read("short_02/highlight.fcpxml").decode()
    edl = archive.read("short_01/highlight.edl").decode()
    assert edl.count("* FROM CLIP NAME") == 2


def test_post_text_per_short(client, monkeypatch):
    from app.services.job_manager import get_job_manager
    import app.services.ollama_runtime as runtime

    monkeypatch.setattr(runtime, "get_ollama_client", lambda settings=None: (_ for _ in ()).throw(RuntimeError("offline")))
    job_id = _job(client, {"shorts_count": 2})
    job_dir = get_job_manager().job_dir(job_id)
    (job_dir / "transcript.json").write_text(
        json.dumps({"segments": [{"start": 10, "end": 12, "text": "About bicycles."}, {"start": 400, "end": 403, "text": "About gardens."}]}),
        encoding="utf-8",
    )
    clips = [
        {"start": 10, "end": 12, "reason": "r", "short_index": 0},
        {"start": 400, "end": 403, "reason": "r", "short_index": 1},
    ]
    client.post(f"/api/jobs/{job_id}/clips", json={"clips": clips})
    second = client.post(f"/api/jobs/{job_id}/post-text?short=2").get_json()["suggestion"]
    assert "gardens" in second["description"].lower() and "bicycles" not in second["description"].lower()
    assert client.get(f"/api/jobs/{job_id}/post-text?short=2").get_json()["suggestion"] == second
    assert client.get(f"/api/jobs/{job_id}/post-text?short=1").get_json()["suggestion"] is None
    assert client.post(f"/api/jobs/{job_id}/post-text?short=3").status_code == 400
