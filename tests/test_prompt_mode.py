"""Prompt mode: reading a description, sound labels, the AI editor, re-planning with changes,
the analysis switches, the settings it adds, and the smoother cuts."""

from __future__ import annotations

import json
import re
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from app import create_app
from app.ai.intent import interpret, read_rules
from app.ai.ollama import OllamaError
from app.analysis import audio_events, clap_features, score_source
from app.analysis.director import apply_director, content_entry, run_director
from app.analysis.optimizer import ClipOptimizer
from app.analysis.pipeline import PipelineProgress, VideoAnalyzer
from app.config import Settings, get_settings, set_settings
from app.models.db_models import Job, get_session
from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, RenderSettings, Transcript, TranscriptSegment, VideoMetadata
from app.services import job_manager as jm
from app.services.job_manager import get_job_manager
from app.video.renderer import Renderer
from app.workers import worker


# Settings model


def test_existing_configs_stay_manual_and_unchanged():
    config = HighlightConfig.model_validate({"style": "funny", "transition": "fade"})
    assert (config.mode, config.prompt, config.requirements, config.use_visuals) == ("manual", "", None, True)
    assert config.analysis_depth == "balanced" and config.refine_notes == []


def test_prompt_mode_needs_a_description_and_one_kind_of_analysis():
    # Nothing described yet (the video is uploaded first): the options are used as they are.
    assert HighlightConfig(mode="prompt", prompt="  ").mode == "manual"
    with pytest.raises(ValueError):
        HighlightConfig(use_visuals=False, use_speech=False)
    config = HighlightConfig(mode="prompt", prompt="x", requirements={"keep": [" goals ", "Goals", "", "saves"]})
    assert config.requirements.keep == ["goals", "saves"]
    assert HighlightConfig(transition="auto").transition == "auto"


def test_refine_notes_keep_the_newest_ten():
    config = HighlightConfig(mode="prompt", prompt="x", refine_notes=[f"change {i}" for i in range(14)] + ["  "])
    assert config.refine_notes[0] == "change 4" and len(config.refine_notes) == 10


# Reading the description


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A 30 second TikTok with captions, skip the first 2 minutes", {"must_exclude": [(0.0, 120.0)], "max_duration": 30.0, "aspect_ratio": "9:16", "subtitles": "burned"}),
        ("about a minute, keep 1:20 to 2:05, start with the best moment", {"must_include": [(80.0, 125.0)], "min_duration": 51, "max_duration": 69, "order_mode": "hook"}),
        ("make 3 shorts under 45 seconds, remove the ums", {"shorts_count": 3, "max_duration": 45.0, "tighten": "fillers"}),
        ("a square video as long as it needs", {"length_mode": "auto", "aspect_ratio": "1:1"}),
        ("cut 0:00-0:30 and keep 5:00-5:40", {"must_exclude": [(0.0, 30.0)], "must_include": [(300.0, 340.0)]}),
        ("skip the last two minutes, no subtitles", {"must_exclude": [(480.0, 600.0)], "subtitles": "none"}),
        ("two to three minutes", {"min_duration": 120.0, "max_duration": 180.0}),
    ],
)
def test_rules_read_lengths_times_and_formats(text, expected):
    rules = read_rules(text, 600)
    for key, value in expected.items():
        assert rules[key] == value, (key, rules)


def test_words_are_not_read_as_lengths():
    assert "min_duration" not in read_rules("as long as it shows the dog", 600)


class IntentClient:
    def __init__(self, reply=None, fail=None):
        self.reply, self.fail, self.prompts = reply, fail, []

    def role_options(self, role):
        return {"num_ctx": 4096}

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
        self.prompts.append(prompt)
        if self.fail:
            raise self.fail
        return self.reply


def test_the_model_reads_the_meaning_and_the_rules_win_on_numbers():
    reply = {
        "summary": "Best goals as a vertical clip",
        "min_seconds": 10,
        "max_seconds": 90,
        "format": "9:16",
        "style": "exciting",
        "priorities": ["sports", "nonsense"],
        "leave_out": ["intro_outro"],
        "order": "story",
        "keep": ["goals", "celebrations"],
        "remove": ["warm-up"],
        "moments": ["the winning goal"],
        "visual_cues": ["a ball going into the net"],
        "audio_cues": ["crowd cheering"],
    }
    result = interpret("A 30 second clip of the best goals", duration=600, client=IntentClient(reply), model="m")
    settings = result["settings"]
    assert (settings["min_duration"], settings["max_duration"]) == (26, 30.0), "numbers come from the rules"
    assert settings["aspect_ratio"] == "9:16" and settings["reframe"] == "subject"
    assert settings["style"] == "exciting" and settings["content_priorities"] == ["sports"]
    assert settings["avoid"] == ["intro_outro"] and settings["order_mode"] == "ai"
    assert settings["custom_focus"] == "goals; celebrations; the winning goal"
    assert settings["avoid_custom"] == "warm-up"
    assert settings["transition"] == "auto" and settings["level_audio"] is True
    assert result["requirements"]["audio_cues"] == ["crowd cheering"]
    assert result["source"] == "ai" and result["warning"] is None
    sources = {change["field"]: change["source"] for change in result["changes"]}
    assert sources["max_duration"] == "rules" and sources["style"] == "ai" and sources["transition"] == "default"
    HighlightConfig.model_validate({"mode": "prompt", "prompt": "x", **settings})


def test_without_the_model_the_basics_and_the_focus_still_work():
    result = interpret(
        "45 seconds of the funniest bits with captions",
        duration=600,
        client=IntentClient(fail=OllamaError("model not found")),
        model="m",
    )
    assert result["source"] == "rules" and result["warning"]
    assert result["settings"]["max_duration"] == 45.0 and result["settings"]["subtitles"] == "burned"
    assert result["settings"]["custom_focus"].startswith("45 seconds of the funniest")


def test_speech_settings_are_left_out_when_audio_analysis_is_off():
    result = interpret("vertical with captions, cut the ums", client=None, model="", use_speech=False)
    assert "subtitles" not in result["settings"] and "tighten" not in result["settings"]


def test_unusable_values_from_the_model_are_dropped():
    result = interpret("anything", client=IntentClient({"summary": "s", "shorts": 50, "min_seconds": -4, "keep_times": [{"start": 9, "end": 3}]}), model="m")
    assert "shorts_count" not in result["settings"] and "min_duration" not in result["settings"]
    assert "must_include" not in result["settings"]


# Sound


def test_clap_features_have_the_trained_shape():
    filters = clap_features.mel_filters()
    assert filters.shape == (513, 64) and (filters >= 0).all()
    window = clap_features.fit_window(np.ones(48000, dtype=np.float32))
    assert window.size == clap_features.WINDOW_SAMPLES and window[47999] == 1 and window[48000] == 1
    assert clap_features.log_mel(np.zeros(1000, dtype=np.float32)).shape == (1001, 64)


class FakeSoundMatcher:
    """Windows whose first sample is loud sound like "music"; quiet ones like "silence"."""

    device = "CPU"

    def __init__(self):
        self.windows: list[int] = []

    def embed_audio(self, windows):
        self.windows.extend(len(w) for w in windows)
        rows = [[1.0, 0.0, 0.0] if abs(float(w[0])) > 0.1 else [0.0, 1.0, 0.0] for w in windows]
        return np.array(rows, dtype=np.float32)

    def embed_texts(self, texts):
        rows = []
        for text in texts:
            if "music" in text:
                rows.append([1.0, 0.0, 0.0])
            elif "silence" in text:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.array(rows, dtype=np.float32)


def _wav(path: Path, seconds: float, loud_from: float) -> Path:
    rate = audio_events.SAMPLE_RATE
    samples = np.zeros(int(seconds * rate), dtype=np.int16)
    samples[int(loud_from * rate) :] = 20000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


def test_sound_windows_are_labelled_and_matched(tmp_path):
    matcher = FakeSoundMatcher()
    starts, vectors = audio_events.embed_recording(matcher, _wav(tmp_path / "a.wav", 12.0, 5.0))
    assert starts == [0.0, 2.5, 5.0, 7.5]
    assert matcher.windows[0] == 5 * audio_events.SAMPLE_RATE
    labels = audio_events.label_windows(matcher, vectors)
    assert list(labels[0]) == ["silence"] and list(labels[2]) == ["music"]
    match = audio_events.match_texts(matcher, vectors, ["music"])
    assert match[2] > match[0]
    events = {"window": 5.0, "windows": [{"start": s, "labels": l, "cue": c} for s, l, c in zip(starts, labels, match)]}
    heard = audio_events.sounds_between(events, 6.0, 7.0)
    assert "music" in heard and heard["cue"] == max(match)


def test_a_stop_during_sound_labelling_is_honoured(tmp_path):
    from app.video.ffmpeg_runner import CancelledError

    def stop():
        raise CancelledError("paused")

    with pytest.raises(CancelledError):
        audio_events.embed_recording(FakeSoundMatcher(), _wav(tmp_path / "a.wav", 30.0, 0.0), stop)


# The AI editor


def _candidates():
    rows = []
    for i in range(6):
        rows.append(
            {
                "id": f"v0:t{i}",
                "start": i * 10.0,
                "end": i * 10.0 + 6,
                "score": 0.4 + i * 0.05,
                "text": ["the dog runs", "a goal is scored", "people laugh loudly", "rain on the window", "a cake is baked", "the band starts playing"][i],
                "reason": "Meaningful moment",
                "source_id": "v0",
                "metrics": {"importance": 0.5},
            }
        )
    return rows


CONTENT = {
    "v0": content_entry(
        [{"scene_id": 0, "start": 0.0, "end": 12.0}, {"scene_id": 1, "start": 12.0, "end": 60.0}],
        {"0": {"caption": "a dog catches a ball"}, "1": {"caption": ""}},
        {"window": 5.0, "windows": [{"start": 0.0, "labels": {"music": 0.8}, "cue": 0.9}]},
    )
}


class EditorClient:
    def __init__(self, picks=None, num_ctx=16384, fail_think=False, fail=None):
        self.picks = picks or (lambda ids: ids[:2])
        self.num_ctx = num_ctx
        self.fail_think = fail_think
        self.fail = fail
        self.calls: list[dict] = []

    def role_options(self, role):
        assert role == "director"
        return {"num_ctx": self.num_ctx}

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True, think=False):
        self.calls.append({"prompt": prompt, "think": think, "options": options})
        if self.fail:
            raise self.fail
        if think and self.fail_think:
            raise OllamaError('HTTP 400: "m" does not support thinking')
        ids = re.findall(r"^(m\d+)(?: \(c\d+\))? \[", prompt, re.MULTILINE)
        return {
            "understanding": "Dog moments first",
            "left_out": "talking",
            "order": "story",
            "clips": [{"id": alias, "role": "hook", "why": f"chosen {alias}"} for alias in self.picks(ids)] + [{"id": "m999", "role": "key", "why": "x"}],
        }


def _config(**extra):
    return HighlightConfig(mode="prompt", prompt="the dog", requirements={"keep": ["dog"], "audio_cues": ["music"]}, **extra)


def test_the_editor_reads_what_is_said_seen_and_heard_and_picks_by_number():
    client = EditorClient(picks=lambda ids: [ids[-1], ids[0]])
    plan = run_director(client, "m", Settings(_env_file=None), _config(), _candidates(), CONTENT)
    prompt = client.calls[0]["prompt"]
    assert 'says: "the dog runs"' in prompt and "sees: a dog catches a ball" in prompt
    assert "hears: music (matches the wanted sounds)" in prompt and "Keep: dog" in prompt
    assert [c["id"] for c in plan["clips"]] == ["v0:t5", "v0:t0"], "unknown numbers are ignored"
    assert plan["order"] == "story" and plan["understanding"] == "Dog moments first"


def test_an_unchanged_request_reuses_the_saved_plan():
    client = EditorClient()
    first = run_director(client, "m", Settings(_env_file=None), _config(), _candidates(), CONTENT)
    again = run_director(client, "m", Settings(_env_file=None), _config(), _candidates(), CONTENT, cached=first)
    assert again is first and len(client.calls) == 1
    changed = run_director(
        client, "m", Settings(_env_file=None), _config(refine_notes=["shorter"]), _candidates(), CONTENT, cached=first
    )
    assert changed["key"] != first["key"] and len(client.calls) == 2
    assert "shorter" in client.calls[-1]["prompt"]


def test_long_videos_are_read_in_parts_then_chosen_from():
    client = EditorClient(num_ctx=4096, picks=lambda ids: ids[:1])
    many = [{**c, "id": f"v0:x{i}", "start": i * 7.0, "end": i * 7.0 + 5, "text": "word " * 60} for i, c in enumerate(_candidates() * 20)]
    plan = run_director(client, "m", Settings(_env_file=None, ollama_director_num_predict=256), _config(), many, CONTENT)
    assert len(client.calls) > 3, "an outline, several parts and a final choice"
    assert "planning the story" in client.calls[0]["prompt"], "the story is planned first"
    assert "This is one part of a longer video" in client.calls[1]["prompt"]
    assert "Choose the moments for the final video" in client.calls[-1]["prompt"]
    assert plan and plan["clips"]


def test_thinking_falls_back_when_the_model_cannot_think():
    client = EditorClient(fail_think=True)
    plan = run_director(client, "m", Settings(_env_file=None, director_think=True), _config(), _candidates(), CONTENT)
    assert [call["think"] for call in client.calls] == [True, False]
    assert plan and plan["clips"]


def test_no_plan_when_the_model_fails():
    client = EditorClient(fail=OllamaError("HTTP 404: model not found"))
    assert run_director(client, "m", Settings(_env_file=None), _config(), _candidates(), CONTENT) is None


def test_the_editors_picks_come_first_and_keep_their_reasons():
    plan = {"order": "story", "clips": [{"id": "v0:t1", "role": "hook", "why": "the dog"}, {"id": "v0:t0", "role": "ending", "why": "bye"}]}
    rows = apply_director(_candidates(), plan)
    assert [r["id"] for r in rows[:2]] == ["v0:t1", "v0:t0"]
    assert rows[0]["role"] == "hook" and rows[0]["reason"] == "the dog" and rows[0]["director_order"] == 0
    other = next(r for r in rows if r["id"] == "v0:t5")
    assert other["score"] == pytest.approx(0.65 * 0.7)
    assert apply_director(_candidates(), None) == _candidates()


def test_the_planner_keeps_picks_it_would_skip_and_follows_the_story_order():
    filler = {**_candidates()[0], "metrics": {"filler_score": 0.9}}
    rows = apply_director([filler, *_candidates()[1:]], {"order": "story", "clips": [{"id": "v0:t3", "role": "hook", "why": "a"}, {"id": "v0:t0", "role": "ending", "why": "b"}]})
    config = HighlightConfig(mode="prompt", prompt="x", min_duration=10, max_duration=14, order_mode="ai")
    plan = ClipOptimizer().plan(rows, config, source_duration=60)
    assert [c.candidate_id for c in plan.clips] == ["v0:t3", "v0:t0"]
    assert [c.role for c in plan.clips] == ["hook", "ending"]


# Analysis


class VisionClient:
    def __init__(self):
        self.prompts: list[str] = []

    def role_options(self, role):
        return {"num_ctx": 16384, "num_thread": 4}

    def encode_image(self, path):
        return "image"

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
        if images:
            self.prompts.append(prompt)
            ids = [int(m) for m in re.findall(r"scene_id=(\d+)", prompt)]
            caption = '"caption"' in prompt
            return {"scenes": [{"scene_id": i, "visual_importance": 0.9, **({"caption": f"scene {i} shows a dog"} if caption else {})} for i in ids]}
        count = len(re.findall(r"^\d+\. \[", prompt, re.MULTILINE))
        return {"segments": [{"i": i, "importance": 0.8, "novelty": 0.7, "information_density": 0.6} for i in range(count)]}


class Runner:
    def __init__(self):
        self.cancel_event = threading.Event()

    def extract_frame(self, source, out, stamp, duration=None):
        Path(out).write_bytes(b"jpg")


def _job_dir(tmp_path):
    job_dir = tmp_path / "job"
    (job_dir / "source").mkdir(parents=True)
    (job_dir / "source" / "audio.wav").write_bytes(b"RIFF")
    meta = {"duration": 10.0, "width": 640, "height": 360, "fps": 30.0, "has_audio": False}
    scenes = [{"scene_id": 0, "start": 0.0, "end": 5.0, "duration": 5.0}, {"scene_id": 1, "start": 5.0, "end": 10.0, "duration": 5.0}]
    for name, value in (("metadata.json", meta), ("scenes.json", scenes), ("silences.json", [])):
        (job_dir / name).write_text(json.dumps(value), encoding="utf-8")
    return job_dir


def _analyze(tmp_path, job_dir, config, client):
    from app.workers.worker import load_cache

    settings = Settings(data_dir=tmp_path / "data", vision_scope="all", image_matching=False, ollama_vision_batch_size=2)
    analyzer = VideoAnalyzer(settings, Runner(), client, vision_enabled=True, text_enabled=True)
    return analyzer.analyze(job_dir, job_dir / "v.mp4", config, PipelineProgress(lambda *_: None), load_cache(job_dir), plan=False)


@pytest.fixture
def speech(monkeypatch):
    transcript = Transcript(segments=[TranscriptSegment(start=0.5, end=4.0, text="Here is the dog."), TranscriptSegment(start=5.5, end=9.0, text="It catches the ball.")])
    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", lambda *a, **k: (transcript, {"ok": True}))


def test_prompt_mode_asks_the_vision_model_to_describe_scenes(tmp_path, speech):
    job_dir = _job_dir(tmp_path)
    manual = VisionClient()
    _analyze(tmp_path, job_dir, HighlightConfig(), manual)
    assert manual.prompts and all('"caption"' not in p for p in manual.prompts)

    prompt = VisionClient()
    result = _analyze(tmp_path, job_dir, HighlightConfig(mode="prompt", prompt="the dog"), prompt)
    assert prompt.prompts, "scenes scored before prompt mode are looked at again for their description"
    assert result["visual"]["1"]["caption"] == "scene 1 shows a dog"
    assert result["visual"]["1"]["score_source"] == score_source.AI


def test_video_analysis_off_skips_the_vision_model(tmp_path, speech):
    job_dir = _job_dir(tmp_path)
    client = VisionClient()
    result = _analyze(tmp_path, job_dir, HighlightConfig(use_visuals=False), client)
    assert client.prompts == []
    assert all(entry["score_source"] == score_source.FALLBACK for entry in result["visual"].values())


def test_ai_scores_are_not_replaced_by_quick_scores_while_waiting_for_a_description(tmp_path, speech):
    job_dir = _job_dir(tmp_path)
    ai = score_source.tag({"scene_id": 0, "visual_importance": 0.77})
    (job_dir / "visual_analysis.json").write_text(json.dumps({"0": ai, "1": ai}), encoding="utf-8")
    (job_dir / "visual_connection_focus.json").write_text("[]", encoding="utf-8")
    result = _analyze(tmp_path, job_dir, HighlightConfig(mode="prompt", prompt="x", use_visuals=True), None)
    assert result["visual"]["0"]["visual_importance"] == 0.77


# Jobs: re-planning, stages, routes


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    previous = get_settings()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9")
    settings.ensure_directories()
    set_settings(settings)
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    monkeypatch.setattr(jm, "_manager", None)
    job_dir = settings.jobs_dir / "j1"
    job_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidates()
    (job_dir / "candidates_all.json").write_text(json.dumps(candidates), encoding="utf-8")
    (job_dir / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
    (job_dir / "content_index.json").write_text(json.dumps(CONTENT), encoding="utf-8")
    (job_dir / "metadata.json").write_text(json.dumps({"duration": 60}), encoding="utf-8")
    (job_dir / "clip_plan.json").write_text(json.dumps({"clips": [{"start": 50, "end": 56, "candidate_id": "v0:t5"}]}), encoding="utf-8")
    (job_dir / "sources.json").write_text(json.dumps([{"id": "v0", "path": str(tmp_path / "v.mp4"), "filename": "v.mp4"}]), encoding="utf-8")
    config = HighlightConfig(mode="prompt", prompt="the dog", min_duration=10, max_duration=14)
    session = get_session()
    session.add(
        Job(
            id="j1",
            status="planning",
            analysis_status="completed",
            original_filename="v.mp4",
            input_path=str(tmp_path / "v.mp4"),
            config_json=config.model_dump_json(),
        )
    )
    session.commit()
    session.close()
    yield app, job_dir
    set_settings(previous)


class HealthyEditor(EditorClient):
    settings = None

    def health(self):
        return {"connected": True, "text_present": True, "vision_ready": True}

    def reachable(self):
        return True


def test_asking_for_a_change_plans_again_without_analyzing(app_env, monkeypatch):
    app, job_dir = app_env
    client = HealthyEditor(picks=lambda ids: [ids[1], ids[2]])
    monkeypatch.setattr(worker, "get_ollama_client", lambda settings: client)
    monkeypatch.setattr(worker, "_prepare_all_sources", lambda *a: pytest.fail("analysis must not run again"))
    http = app.test_client()
    assert http.post("/api/jobs/j1/refine", json={"request": ""}).status_code == 400
    queued = http.post("/api/jobs/j1/refine", json={"request": "make it about the ball"})
    assert queued.status_code == 200 and queued.get_json()["analysis_status"] == "refining"
    worker._run_analysis(app, "j1")
    job = get_job_manager().get("j1")
    assert (job.status, job.analysis_status) == ("planning", "completed"), job.error
    assert json.loads(job.config_json)["refine_notes"] == ["make it about the ball"]
    prompt = client.calls[-1]["prompt"]
    assert "make it about the ball" in prompt and "The last plan used, in order: m6" in prompt
    clips = get_job_manager().serialize(job)["clips"]
    # A story opens with one hook and closes with one ending.
    assert [c["role"] for c in clips] == ["hook", "ending"]
    info = http.get("/api/jobs/j1/ai-plan").get_json()
    assert info["plan"]["understanding"] == "Dog moments first" and info["refine_notes"] == ["make it about the ball"]
    names = [s["name"] for s in get_job_manager().stages("j1")]
    assert names.index("director") < names.index("planning")


def test_a_change_on_a_manual_highlight_becomes_the_description(app_env):
    app, _ = app_env
    get_job_manager().update("j1", config_json=HighlightConfig().model_dump_json())
    app.test_client().post("/api/jobs/j1/refine", json={"request": "only the dog"})
    config = json.loads(get_job_manager().get("j1").config_json)
    assert (config["mode"], config["prompt"], config["refine_notes"]) == ("prompt", "only the dog", [])


def test_a_change_needs_a_finished_analysis(app_env):
    app, _ = app_env
    get_job_manager().update("j1", status="analyzing", analysis_status="running")
    assert app.test_client().post("/api/jobs/j1/refine", json={"request": "x"}).status_code == 409


def test_the_create_page_can_read_a_description_without_ollama(app_env):
    app, _ = app_env
    http = app.test_client()
    assert http.post("/api/prompt/interpret", json={"prompt": ""}).status_code == 400
    result = http.post("/api/prompt/interpret", json={"prompt": "30 second reel with captions", "job_id": "j1"}).get_json()
    assert result["source"] == "rules" and result["settings"]["aspect_ratio"] == "9:16"
    status = http.get("/api/system/sound-model").get_json()
    assert {"installed", "size_mb", "state"} <= set(status)


def test_manual_jobs_do_not_show_the_prompt_steps(app_env):
    manager = get_job_manager()
    manager.prepare_stages("j1", "prompt")
    assert {"sounds", "director"} <= {s["name"] for s in manager.stages("j1")}
    manager.prepare_stages("j1", "manual")
    names = [s["name"] for s in manager.stages("j1")]
    assert "sounds" not in names and "director" not in names and names[-1] == "rendering"


# Settings


def test_the_editor_settings_can_be_saved_and_are_recommended():
    from app.routes.settings import EDITABLE
    from app.services.model_catalog import recommend_setup

    assert {"sound_matching", "ollama_director_model", "ollama_director_num_ctx", "director_think"} <= EDITABLE
    cpu = recommend_setup(0, 16, cpu_threads=12)
    assert (cpu["director_num_ctx"], cpu["director_think"]) == (16384, False)
    gpu = recommend_setup(12, 32, cpu_threads=12, goal="quality")
    assert gpu["director_think"] is (gpu["text_model"] != "qwen3:4b-instruct-2507-q4_K_M")


def test_the_editor_role_gets_its_own_context():
    from app.ai.ollama import OllamaClient

    client = OllamaClient(Settings(_env_file=None, ollama_text_num_ctx=4096, ollama_director_num_ctx=16384))
    assert client.role_options("director")["num_ctx"] == 16384
    assert client.role_options("text")["num_ctx"] == 4096


# Smoother cuts


class RenderRunner:
    def __init__(self):
        self.cancel_event = threading.Event()
        self.commands: list[list[str]] = []

    def run(self, cmd):
        self.commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"x" * 2000)

    def probe(self, path):
        return VideoMetadata(duration=60, width=1280, height=720, fps=30.0, avg_fps=30.0, has_audio=True, size_bytes=2000)


def _cuts(tmp_path, transition, clips):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    runner = RenderRunner()
    plan = ClipPlan(target_duration=10, estimated_duration=10, clips=clips)
    Renderer(runner).render(source, plan, tmp_path / "out.mp4", RenderSettings(transition=transition), tmp_path / "work", metadata=runner.probe(source))
    cuts = [c for c in runner.commands if re.search(r"clip_\d+\.mp4$", c[-1])]
    return [c[c.index("-vf") + 1] for c in cuts], [c[c.index("-af") + 1] for c in cuts]


def test_automatic_transitions_fade_only_where_the_video_jumps(tmp_path):
    clips = [
        ClipPlanItem(start=0, end=4, reason="a"),
        ClipPlanItem(start=4.2, end=8, reason="flows on"),
        ClipPlanItem(start=30, end=34, reason="jumps"),
    ]
    video, _audio = _cuts(tmp_path, "auto", clips)
    assert "fade=t=in" in video[0] and "fade=t=out" not in video[0]
    assert "fade=t=in" not in video[1] and "fade=t=out" in video[1]
    assert "fade=t=in" in video[2] and "fade=t=out" in video[2]


def test_plain_cuts_get_a_tiny_sound_fade_against_clicks(tmp_path):
    video, audio = _cuts(tmp_path, "cut", [ClipPlanItem(start=0, end=4, reason="a"), ClipPlanItem(start=10, end=14, reason="b")])
    assert all("fade=t=" not in v for v in video)
    assert all("afade=t=in:st=0:d=0.020" in a and "afade=t=out:st=3.980:d=0.020" in a for a in audio)


def test_a_video_can_be_uploaded_before_the_description_is_written(app_env):
    import io

    app, _ = app_env
    config = {"mode": "prompt", "prompt": "", "use_visuals": True, "use_speech": True}
    data = {"file": (io.BytesIO(b"not really a video"), "clip.mp4"), "config": json.dumps(config)}
    response = app.test_client().post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 201 or response.status_code == 200, response.get_json()
    assert response.get_json()["config"]["mode"] == "manual"
