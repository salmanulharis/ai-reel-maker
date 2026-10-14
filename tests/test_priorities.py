import json

from app.analysis.optimizer import ClipOptimizer
from app.analysis.sequence_limits import desired_auto_length
from app.analysis.scorer import HighlightScorer
from app.ai.prompts import visual_analysis_focus
from app.analysis.candidates import populate_candidate_audio_metrics
from app.models.schemas import CandidateScore, HighlightConfig, Transcript
from app.routes.jobs import REGEN_PRESETS


def _score(**overrides):
    base = dict(
        importance=0.5,
        novelty=0.4,
        visual_importance=0.4,
        information_density=0.4,
        hook_strength=0.3,
        repetition_score=0.05,
        filler_score=0.05,
        context_dependency=0.2,
        emotional_intensity=0.3,
        production_quality=0.7,
    )
    base.update(overrides)
    return CandidateScore(**base)


def test_action_priority_prefers_visual_action_over_speech():
    scorer = HighlightScorer()
    action = _score(emotional_intensity=0.95, visual_importance=0.9, importance=0.3, information_density=0.2)
    speech = _score(importance=0.95, information_density=0.9, emotional_intensity=0.2, visual_importance=0.2)
    action_first = scorer.finalize(action, priorities=["action"]) > scorer.finalize(speech, priorities=["action"])
    speech_first = scorer.finalize(speech, priorities=["speech"]) > scorer.finalize(action, priorities=["speech"])
    assert action_first
    assert speech_first


def test_connection_priority_prefers_affectionate_visual_moments():
    scorer = HighlightScorer()
    affectionate = _score(emotional_intensity=0.9, visual_importance=0.9, importance=0.3, information_density=0.2)
    explanation = _score(importance=0.95, information_density=0.9, emotional_intensity=0.2, visual_importance=0.2)
    assert scorer.finalize(affectionate, priorities=["connection"]) > scorer.finalize(
        explanation, priorities=["connection"]
    )


def test_audio_focus_ranks_natural_sound_and_dialogue_without_changing_balanced():
    scorer = HighlightScorer()
    natural = _score(natural_audio=0.95, speech_presence=0.05)
    dialogue = _score(natural_audio=0.05, speech_presence=0.95)
    assert scorer.finalize(natural, audio_focus="natural") > scorer.finalize(dialogue, audio_focus="natural")
    assert scorer.finalize(dialogue, audio_focus="dialogue") > scorer.finalize(natural, audio_focus="dialogue")
    assert scorer.finalize(natural) == scorer.finalize(dialogue)


def test_review_focus_presets_set_the_requested_content_priority():
    assert REGEN_PRESETS["more_visual"]["content_priorities"] == ["visual"]
    assert REGEN_PRESETS["focus_on_speech"]["content_priorities"] == ["speech"]
    assert REGEN_PRESETS["focus_on_action"]["content_priorities"] == ["action"]


def test_review_regeneration_uses_selected_focus_to_change_cached_plan(tmp_path, monkeypatch):
    from app import create_app
    from app.config import Settings, get_settings, set_settings
    from app.models.db_models import Job, get_session
    from app.services.job_manager import get_job_manager

    previous = get_settings()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", ollama_embed_model="")
    try:
        app = create_app(settings, start_background=False)
        monkeypatch.setattr("app.routes.jobs.get_settings", lambda: settings)
        session = get_session()
        config = HighlightConfig(
            min_duration=8,
            max_duration=8,
            max_sequence_duration=8,
            min_sequence_duration=8,
            sequence_duration_mode="custom",
            remove_repetition=False,
        )
        session.add(
            Job(
                id="focus-job",
                status="planning",
                original_filename="video.mp4",
                input_path=str(tmp_path / "video.mp4"),
                style="auto",
                min_duration=8,
                max_duration=8,
                config_json=config.model_dump_json(),
            )
        )
        session.commit()
        session.close()

        job_dir = get_job_manager().job_dir("focus-job")
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "metadata.json").write_text(json.dumps({"duration": 60}), encoding="utf-8")
        (job_dir / "candidates.json").write_text(
            json.dumps(
                [
                    {
                        "id": "visual",
                        "start": 10,
                        "end": 18,
                        "score": 0.5,
                        "topic": "visual",
                        "text": "sunset coastline ocean",
                        "metrics": {
                            "importance": 0.25,
                            "information_density": 0.2,
                            "visual_importance": 0.98,
                            "novelty": 0.55,
                            "hook_strength": 0.3,
                            "emotional_intensity": 0.96,
                            "production_quality": 0.8,
                        },
                    },
                    {
                        "id": "speech",
                        "start": 30,
                        "end": 38,
                        "score": 0.5,
                        "topic": "speech",
                        "text": "here are the important details and explanation",
                        "metrics": {
                            "importance": 0.98,
                            "information_density": 0.98,
                            "visual_importance": 0.2,
                            "novelty": 0.5,
                            "hook_strength": 0.2,
                            "emotional_intensity": 0.15,
                            "production_quality": 0.8,
                        },
                    },
                ]
            ),
            encoding="utf-8",
        )

        client = app.test_client()
        visual = client.post("/api/jobs/focus-job/regenerate", json={"preset": "more_visual"})
        speech = client.post("/api/jobs/focus-job/regenerate", json={"preset": "focus_on_speech"})
        assert visual.status_code == speech.status_code == 200
        assert visual.get_json()["plan"]["clips"][0]["candidate_id"] == "visual"
        assert speech.get_json()["plan"]["clips"][0]["candidate_id"] == "speech"
    finally:
        set_settings(previous)


def test_legacy_candidates_get_audio_metrics_for_replanning():
    candidates = [
        {"start": 2, "end": 4, "text": "", "metrics": {}},
        {"start": 2, "end": 4, "text": "a few spoken words", "metrics": {}},
    ]
    energy = [0.1] * 8 + [0.9] * 8 + [0.1] * 8
    populate_candidate_audio_metrics(candidates, Transcript(), energy)
    assert candidates[0]["metrics"]["natural_audio"] > candidates[1]["metrics"]["natural_audio"]
    assert candidates[1]["metrics"]["speech_presence"] > candidates[0]["metrics"]["speech_presence"]


def test_connection_priority_focus_is_appended_to_custom_focus():
    focus = visual_analysis_focus("focus on the couple", ["connection"])
    assert focus.startswith("focus on the couple.")
    assert "affectionate or romantic interactions" in focus
    assert "kisses" in focus
    assert "do not infer details that are not visible" in focus


def test_connection_priority_gives_strong_interactions_room_to_breathe():
    candidate = {
        "start": 10,
        "end": 20,
        "text": "",
        "metrics": {
            "importance": 0.1,
            "information_density": 0.1,
            "visual_importance": 0.2,
            "emotional_intensity": 0.5,
            "hook_strength": 0.1,
        },
    }
    length, reason = desired_auto_length(candidate, 20, priorities=["connection"])
    assert length == 6
    assert "connection priority" in reason


def test_balanced_does_not_collapse_to_a_single_type():
    scorer = HighlightScorer()
    action = scorer.finalize(_score(emotional_intensity=0.9, visual_importance=0.85), priorities=["balanced"])
    info = scorer.finalize(_score(importance=0.9, information_density=0.9), priorities=["balanced"])
    assert abs(action - info) < 0.25


def test_highlight_config_normalizes_balanced_and_weights():
    config = HighlightConfig(content_priorities=["action", "balanced", "speech"], priority_weights={"action": 5})
    assert config.content_priorities == ["balanced"]
    config = HighlightConfig(content_priorities=["action", "speech"], priority_weights={"action": 8, "speech": 0})
    assert config.content_priorities == ["action", "speech"]
    assert 0.2 <= config.priority_weights["action"] <= 1.0
    assert config.priority_weights["speech"] >= 0.2
    config = HighlightConfig(content_priorities=["connection"], priority_weights={"connection": 5})
    assert config.content_priorities == ["connection"]
    assert config.priority_weights["connection"] == 1.0


def test_optimizer_keeps_only_the_strongest_similar_action():
    optimizer = ClipOptimizer()
    candidates = [
        {
            "id": "weak",
            "start": 10,
            "end": 18,
            "score": 0.72,
            "topic": "jump",
            "text": "the rider jumps the gap",
            "reason": "Action",
            "metrics": {"emotional_intensity": 0.7, "visual_importance": 0.7},
        },
        {
            "id": "strong",
            "start": 40,
            "end": 50,
            "score": 0.94,
            "topic": "jump",
            "text": "another jump over the same gap",
            "reason": "Action",
            "metrics": {"emotional_intensity": 0.95, "visual_importance": 0.9},
        },
        {
            "id": "talk",
            "start": 70,
            "end": 78,
            "score": 0.8,
            "topic": "story",
            "text": "here is why this trail matters",
            "reason": "Speech",
            "metrics": {"importance": 0.85, "information_density": 0.8},
        },
    ]
    plan = optimizer.select(
        candidates,
        min_duration=10,
        max_duration=20,
        sequence_duration_mode="auto",
        allow_repeated_content=False,
        similarity_fn=lambda x, y: 0.88 if x["topic"] == y["topic"] else 0.1,
    )
    ids = [c.candidate_id for c in plan.clips]
    assert "strong" in ids
    assert "weak" not in ids
    assert "talk" in ids
