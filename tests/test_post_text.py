"""Suggested titles, description and hashtags for posting the highlight."""

import io
import json
from types import SimpleNamespace

import pytest

from app import create_app
from app.config import Settings
from app.services import post_text
from app.services.post_text import clean, gather_material, simple_suggestion, suggest
from app.models.schemas import HighlightConfig


TRANSCRIPT = {
    "segments": [
        {"start": 10.0, "end": 14.0, "text": "Solar panels now cost far less than ten years ago."},
        {"start": 30.0, "end": 34.0, "text": "Batteries make solar panels useful at night."},
        {"start": 50.0, "end": 52.0, "text": "Not in any clip."},
    ]
}


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def role_options(self, role):
        return {"num_ctx": 4096}

    def chat(self, model, prompt, schema=None, options=None, **kwargs):
        self.prompts.append((prompt, options))
        if self.error:
            raise self.error
        return self.reply


def _job(tmp_path, config=None):
    (tmp_path / "transcript.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    return SimpleNamespace(config_json=json.dumps(config or {}), original_filename="solar_talk.mp4")


CLIPS = [
    {"start": 10, "end": 14, "kept": True},
    {"start": 20, "end": 22, "kept": False},
    {"start": 30, "end": 34},
]
SETTINGS = SimpleNamespace(ollama_text_model="qwen3:4b-instruct-2507-q4_K_M")


def test_material_is_what_kept_clips_say(tmp_path):
    _job(tmp_path)
    spoken, seen = gather_material(tmp_path, CLIPS, HighlightConfig())
    assert spoken == [TRANSCRIPT["segments"][0]["text"], TRANSCRIPT["segments"][1]["text"]]
    assert seen == []
    # Without speech, nothing is read from the transcript.
    assert gather_material(tmp_path, CLIPS, HighlightConfig(use_speech=False))[0] == []


def test_scene_descriptions_are_used(tmp_path):
    (tmp_path / "scenes.json").write_text(json.dumps([{"scene_id": 1, "start": 9, "end": 15}, {"scene_id": 2, "start": 40, "end": 45}]), encoding="utf-8")
    (tmp_path / "visual_analysis.json").write_text(json.dumps({"1": {"visual_summary": "A roof with solar panels"}, "2": {"visual_summary": "Unused"}}), encoding="utf-8")
    assert gather_material(tmp_path, CLIPS, HighlightConfig(use_speech=False))[1] == ["A roof with solar panels"]


def test_ai_reply_is_cleaned_and_saved(tmp_path):
    job = _job(tmp_path, {"style": "informative", "custom_focus": "costs"})
    client = FakeClient(
        {
            "titles": ['"Solar got cheap"', "Solar got cheap", "  Night power  ", "Four", "Five"],
            "description": "Solar panels   cost less, and batteries help at night.",
            "hashtags": ["#Solar", "clean energy", "solar", "2024", "batteries!"],
        }
    )
    result = suggest(job, tmp_path, CLIPS, SETTINGS, client)
    assert result == {
        "titles": ["Solar got cheap", "Night power", "Four"],
        "description": "Solar panels cost less, and batteries help at night.",
        "hashtags": ["solar", "cleanenergy", "batteries"],
        "source": "ai",
    }
    prompt, options = client.prompts[0]
    assert "Solar panels now cost far less" in prompt and "Not in any clip" not in prompt
    assert "costs" in prompt and "informative" in prompt
    assert options["num_predict"] == 500
    assert post_text.saved(tmp_path) == result


@pytest.mark.parametrize(
    "client",
    [FakeClient(error=RuntimeError("connection refused")), FakeClient({"titles": [], "description": ""}), FakeClient("not a dict")],
)
def test_falls_back_to_a_simple_suggestion(tmp_path, client):
    result = suggest(_job(tmp_path), tmp_path, CLIPS, SETTINGS, client)
    assert result["source"] == "simple" and result["note"]
    assert result["titles"][0] == "Solar talk: highlights"
    assert "solar" in result["hashtags"] and "panels" in result["hashtags"]
    assert result["description"].startswith("Solar panels now cost")


def test_no_kept_clips(tmp_path):
    with pytest.raises(post_text.PostTextError):
        suggest(_job(tmp_path), tmp_path, [{"start": 1, "end": 2, "kept": False}], SETTINGS, FakeClient())


def test_nothing_to_work_from_gives_a_simple_suggestion_without_the_model(tmp_path):
    job = _job(tmp_path, {"use_speech": False})
    client = FakeClient({"titles": ["x"], "description": "y", "hashtags": []})
    result = suggest(job, tmp_path, CLIPS, SETTINGS, client)
    assert client.prompts == [] and result["source"] == "simple"
    assert result["description"] == "The best moments from Solar talk."


def test_clean_limits_lengths():
    long = clean({"titles": ["x" * 300], "description": "y" * 900, "hashtags": [f"t{i}" for i in range(20)]})
    assert len(long["titles"][0]) == 100 and len(long["description"]) == 500 and len(long["hashtags"]) == 8
    assert clean({}) == {"titles": [], "description": "", "hashtags": []}


def test_simple_suggestion_shortens_long_sentences():
    result = simple_suggestion(["word " * 40 + "end."], [], "x.mp4")
    assert result["titles"][1].endswith("…") and len(result["titles"][1]) <= 70


@pytest.fixture()
def client(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9", enable_whisper=False)
    settings.ensure_directories()
    import app.services.ollama_runtime as runtime

    monkeypatch.setattr(runtime, "get_ollama_client", lambda settings=None: FakeClient(error=RuntimeError("offline")))
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_endpoints(client):
    from app.services.job_manager import get_job_manager

    assert client.get("/api/jobs/missing/post-text").status_code == 404
    data = {"file": (io.BytesIO(b"v"), "talk.mp4"), "config": json.dumps({})}
    job_id = client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()["id"]
    assert client.get(f"/api/jobs/{job_id}/post-text").get_json() == {"suggestion": None}
    assert client.post(f"/api/jobs/{job_id}/post-text").status_code == 400
    (get_job_manager().job_dir(job_id) / "transcript.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    client.post(f"/api/jobs/{job_id}/clips", json={"clips": [{"start": 10, "end": 14, "reason": "r"}]})
    made = client.post(f"/api/jobs/{job_id}/post-text").get_json()["suggestion"]
    assert made["source"] == "simple" and "offline" in made["note"]
    assert client.get(f"/api/jobs/{job_id}/post-text").get_json()["suggestion"] == made
