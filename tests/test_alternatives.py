"""More moments: strong moments the plan left out, offered on the Review page."""

import io
import json

import pytest

from app import create_app
from app.analysis.optimizer import ClipOptimizer, sequence_settings
from app.config import Settings


def _c(cid, start, end, score, text="", source_id="", metrics=None):
    return {
        "id": cid,
        "start": start,
        "end": end,
        "score": score,
        "text": text or f"moment {cid} words",
        "reason": "Key explanation",
        "source_id": source_id,
        "metrics": metrics or {},
    }


def _alternatives(candidates, taken, **kwargs):
    options = {"min_duration": 20, "max_duration": 30, "max_sequence_duration": 60}
    options.update(kwargs)
    return ClipOptimizer().alternatives(candidates, taken, **options)


def test_best_unused_moments_first_and_none_overlap_the_plan():
    candidates = [
        _c("a", 10, 15, 0.95),
        _c("b", 40, 45, 0.70),
        _c("c", 60, 65, 0.90),
        _c("d", 13, 18, 0.99),  # overlaps the planned clip at 10-15
    ]
    items = _alternatives(candidates, [{"start": 10, "end": 15}])
    assert [i.candidate_id for i in items] == ["c", "b"]


def test_removed_clips_still_count_as_taken():
    candidates = [_c("a", 10, 15, 0.9), _c("b", 30, 35, 0.8)]
    items = _alternatives(candidates, [{"start": 10, "end": 15, "kept": False}])
    assert [i.candidate_id for i in items] == ["b"]


def test_overlapping_alternatives_keep_only_the_stronger():
    candidates = [_c("a", 10, 16, 0.6), _c("b", 14, 20, 0.8), _c("c", 50, 55, 0.5)]
    items = _alternatives(candidates, [])
    assert [i.candidate_id for i in items] == ["b", "c"]


def test_limit_must_exclude_and_low_value_are_respected():
    candidates = [
        _c("a", 10, 15, 0.9),
        _c("b", 20, 25, 0.8, metrics={"filler_score": 0.9}),  # filler
        _c("c", 30, 35, 0.7),
        _c("d", 40, 45, 0.6),
        _c("e", 50, 55, 0.5),
    ]
    items = _alternatives(candidates, [], must_exclude=[(29, 31)], limit=2)
    assert [i.candidate_id for i in items] == ["a", "d"]


def test_moments_are_fitted_to_the_clip_length_limit():
    items = _alternatives([_c("long", 100, 140, 0.9)], [], max_sequence_duration=6)
    assert len(items) == 1
    assert items[0].duration == pytest.approx(6, abs=0.01)
    assert 100 <= items[0].start and items[0].end <= 140
    assert items[0].duration_reason


def test_sources_are_compared_separately_unless_single_source():
    candidates = [_c("other-video", 10, 15, 0.9, source_id="v1")]
    taken = [{"start": 10, "end": 15, "source_id": "v0"}]
    assert [i.candidate_id for i in _alternatives(candidates, taken)] == ["other-video"]
    # Older single-video plans may lack the id on one side; times alone decide then.
    taken = [{"start": 10, "end": 15, "source_id": ""}]
    candidates = [_c("same", 10, 15, 0.9, source_id="v0")]
    assert _alternatives(candidates, taken, single_source=True) == []


def test_sequence_settings_matches_planner_rules():
    assert sequence_settings(20, 30, 0.2, None, "custom") == (1.0, None, "custom")
    assert sequence_settings(20, 30, 3, 4, "custom") == (4, 4, "custom")
    cap, _, mode = sequence_settings(20, 30, 5, None, "auto", "range")
    assert mode == "auto" and cap == 30
    assert sequence_settings(20, 30, 5, None, "anything")[2] == "custom"


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        ollama_base_url="http://127.0.0.1:9",
        enable_whisper=False,
        enable_visual_analysis=False,
    )
    settings.ensure_directories()
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _create_job(client) -> str:
    data = {
        "file": (io.BytesIO(b"video"), "talk.mp4"),
        "config": json.dumps({"min_duration": 20, "max_duration": 30, "max_sequence_duration": 60}),
    }
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 201
    return response.get_json()["id"]


def _job_dir(client, job_id: str):
    from app.services.job_manager import get_job_manager

    return get_job_manager().job_dir(job_id)


def test_moments_endpoint_without_analysis(client):
    job_id = _create_job(client)
    response = client.get(f"/api/jobs/{job_id}/moments")
    assert response.status_code == 200
    assert response.get_json() == {"moments": [], "available": False}


def test_moments_endpoint_unknown_job(client):
    assert client.get("/api/jobs/missing/moments").status_code == 404


def test_moments_endpoint_leaves_out_current_clips(client):
    job_id = _create_job(client)
    strong = {"importance": 0.9, "hook_strength": 0.8, "information_density": 0.8}
    weak = {"importance": 0.4}
    candidates = [
        _c("planned", 10, 15, 0, text="the planned moment here", metrics=strong),
        _c("strong", 40, 45, 0, text="a strong idea about pricing", metrics=strong),
        _c("weaker", 70, 75, 0, text="another point on the roadmap", metrics=weak),
    ]
    (_job_dir(client, job_id) / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
    saved = client.post(f"/api/jobs/{job_id}/clips", json={"clips": [{"start": 10, "end": 15, "reason": "r"}]})
    assert saved.status_code == 200

    body = client.get(f"/api/jobs/{job_id}/moments").get_json()
    assert body["available"] is True
    ids = [m["candidate_id"] for m in body["moments"]]
    assert ids == ["strong", "weaker"]
    first = body["moments"][0]
    assert {"start", "end", "score", "reason", "transcript"} <= set(first)
    assert "source_path" not in first
    assert first["transcript"] == "a strong idea about pricing"

    limited = client.get(f"/api/jobs/{job_id}/moments?limit=1").get_json()
    assert len(limited["moments"]) == 1
    # Out-of-range limits are clamped rather than rejected.
    assert client.get(f"/api/jobs/{job_id}/moments?limit=0").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/moments?limit=abc").status_code == 200
