"""Automatic optimization: review, apply, turn off, undo, warnings, and suggestions from runs."""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings, set_settings
from app.models.db_models import init_db
from app.services import auto_tune
from app.services.ai_profile import APPLIED_FIELDS
from app.services.model_catalog import recommend_setup


def hardware(*, ram=16.0, available=None, vram=None, free_vram=None, gpu=None, cores=6, threads=12, disk=200.0, mode=None):
    recommendation = recommend_setup(vram, ram, free_vram, cores)
    return {
        "cpu": "Test CPU",
        "cpu_cores": cores,
        "cpu_threads": threads,
        "system": "Windows",
        "ram_gb": ram,
        "ram_available_gb": ram * 0.6 if available is None else available,
        "disk_free_gb": disk,
        "gpu": gpu,
        "gpu_vendor": "nvidia" if gpu else None,
        "vram_gb": vram,
        "vram_free_gb": free_vram,
        "cuda": bool(gpu),
        "inference_mode": mode or ("gpu" if vram and vram >= 2 else "cpu"),
        "recommendation": recommendation,
    }


class OfflineClient:
    def health(self):
        return {"connected": False, "version": None, "models": [], "running": [], "error": "offline"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    previous = get_settings()
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        ai_config_mode="custom",
        ollama_vision_model="qwen3-vl:8b",
        ollama_text_model="qwen3:8b",
        ollama_num_thread=6,
    )
    settings.ensure_directories()
    set_settings(settings)
    init_db(settings.db_path)
    state = {"hw": hardware()}
    monkeypatch.setattr("app.services.ai_profile.detect_hardware", lambda **_kw: state["hw"])
    monkeypatch.setattr("app.services.ai_profile.get_ollama_client", lambda settings=None: OfflineClient())
    yield settings, state
    set_settings(previous)


def test_managed_settings_match_what_automatic_mode_applies():
    assert set(auto_tune.MANAGED) == set(APPLIED_FIELDS)


# Recommendations per kind of computer


def _values(plan):
    return {item["key"]: item["recommended"] for item in plan["items"]}


def test_cpu_only_low_memory_uses_small_models_and_says_why(env):
    settings, state = env
    state["hw"] = hardware(ram=12.0)
    plan = auto_tune.build_plan(settings, refresh=True)
    values = _values(plan)
    assert values["ollama_vision_model"] == "qwen3-vl:2b-instruct"
    assert values["whisper_device"] == "cpu"
    assert values["ollama_num_thread"] == 5  # one core left free
    assert any("smaller video model" in item for item in plan["tradeoffs"])
    assert "no graphics card" in plan["hardware"]["text"]


def test_cpu_only_plenty_of_memory_does_not_pick_large_models(env):
    settings, state = env
    state["hw"] = hardware(ram=64.0, cores=16)
    values = _values(auto_tune.build_plan(settings, refresh=True))
    assert values["ollama_vision_model"] == "qwen3-vl:4b-instruct"
    assert values["ollama_text_model"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert values["ollama_num_thread"] == 8  # capped


def test_gpu_picks_larger_models_without_trade_off_notes(env):
    settings, state = env
    state["hw"] = hardware(ram=32.0, vram=12.0, free_vram=11.0, gpu="RTX 4070")
    plan = auto_tune.build_plan(settings, refresh=True)
    values = _values(plan)
    assert values["ollama_vision_model"] == "qwen3-vl:8b-instruct"
    assert plan["tradeoffs"] == []
    assert plan["hardware"]["uses_gpu"] is True
    assert any("graphics memory" in item["reason"] for item in plan["items"])


def test_small_gpu_stays_within_its_memory(env):
    settings, state = env
    state["hw"] = hardware(ram=16.0, vram=4.0, free_vram=3.0, gpu="GTX 1650")
    values = _values(auto_tune.build_plan(settings, refresh=True))
    assert values["ollama_vision_model"] in {"qwen3-vl:2b-instruct", "qwen3-vl:4b-instruct"}
    assert values["whisper_device"] == "cpu"


def test_unknown_hardware_still_gives_a_safe_plan(env):
    settings, state = env
    state["hw"] = hardware(ram=None, cores=None, threads=None, available=0, disk=None)
    state["hw"]["ram_available_gb"] = None
    plan = auto_tune.build_plan(settings, refresh=True)
    assert _values(plan)["ollama_vision_model"] == "qwen3-vl:2b-instruct"
    assert plan["warnings"] == []


# Safety warnings


def test_warns_when_free_memory_or_disk_is_low(env):
    settings, state = env
    state["hw"] = hardware(ram=16.0, available=1.5, disk=4.0)
    warnings = auto_tune.build_plan(settings, refresh=True)["warnings"]
    assert any("1.5 GB of memory is free" in item for item in warnings)
    assert any("4.0 GB of disk space" in item for item in warnings)


def test_warns_when_a_gpu_elsewhere_runs_the_models():
    hw = hardware(ram=16.0)
    runtime = {"placement": {"mode": "gpu"}}
    found = auto_tune.warnings(hw["recommendation"], hw, runtime, Settings(_env_file=None))
    assert any("isn't in this computer" in item for item in found)


def test_builtin_graphics_on_this_computer_is_not_called_remote():
    hw = hardware(ram=14.8, gpu="AMD Radeon (TM) Graphics")
    runtime = {"placement": {"mode": "gpu"}}
    settings = Settings(_env_file=None, ollama_base_url="http://127.0.0.1:11434")
    found = auto_tune.warnings(hw["recommendation"], hw, runtime, settings)
    assert not any("isn't in this computer" in item for item in found)
    assert "built-in graphics (AMD Radeon (TM) Graphics)" in auto_tune.builtin_graphics_note(hw, runtime, settings)


def test_remote_ollama_gpu_is_still_called_remote_with_builtin_graphics_here():
    hw = hardware(ram=14.8, gpu="AMD Radeon (TM) Graphics")
    runtime = {"placement": {"mode": "gpu"}}
    settings = Settings(_env_file=None, ollama_base_url="http://192.168.1.50:11434")
    found = auto_tune.warnings(hw["recommendation"], hw, runtime, settings)
    assert any("isn't in this computer" in item for item in found)
    assert auto_tune.builtin_graphics_note(hw, runtime, settings) is None


# Turning it on and off, and undo


def test_review_lists_only_settings_that_change(env):
    settings, _state = env
    plan = auto_tune.build_plan(settings, refresh=True)
    keys = {item["key"] for item in plan["changes"]}
    assert "ollama_vision_model" in keys and "ollama_text_model" in keys
    assert all(item["changed"] for item in plan["changes"])
    assert plan["enabled"] is False


def test_enable_applies_and_remembers_previous_values(env):
    settings, _state = env
    before = {key: getattr(settings, key) for key in auto_tune.MANAGED}
    plan = auto_tune.enable(settings)
    assert plan["enabled"] is True
    assert settings.ai_config_mode == "auto"
    assert settings.ollama_vision_model == "qwen3-vl:4b-instruct"
    assert plan["changes"] == []
    assert plan["can_undo"] is True
    assert auto_tune.read_snapshot()["values"] == before


def test_enable_never_touches_unrelated_preferences(env):
    settings, _state = env
    settings.output_crf = 23
    settings.enable_visual_analysis = False
    settings.ollama_base_url = "http://127.0.0.1:11435"
    auto_tune.enable(settings)
    assert settings.output_crf == 23
    assert settings.enable_visual_analysis is False
    assert settings.ollama_base_url == "http://127.0.0.1:11435"


def test_disable_keeps_current_values(env):
    settings, _state = env
    auto_tune.enable(settings)
    applied = settings.ollama_vision_model
    plan = auto_tune.disable(settings)
    assert plan["enabled"] is False
    assert settings.ai_config_mode == "custom"
    assert settings.ollama_vision_model == applied


def test_undo_restores_previous_values_and_switches_to_manual(env):
    settings, _state = env
    auto_tune.enable(settings)
    plan = auto_tune.undo(settings)
    assert settings.ollama_vision_model == "qwen3-vl:8b"
    assert settings.ollama_text_model == "qwen3:8b"
    assert settings.ollama_num_thread == 6
    assert settings.ai_config_mode == "custom"
    assert plan["can_undo"] is False


def test_enabling_when_already_optimal_keeps_the_older_undo_point(env):
    settings, _state = env
    auto_tune.enable(settings)
    first = auto_tune.read_snapshot()
    auto_tune.disable(settings)
    auto_tune.enable(settings)
    assert auto_tune.read_snapshot() == first


# Learning from runs


class FakeProgress:
    def __init__(self, vision_seconds=40.0, text_near_limit=0):
        self.timings = {"visual": 120.0, "content": 60.0}
        self.usage = {
            "vision": {"calls": 3, "prompt_sec": vision_seconds * 3 / 2, "output_sec": vision_seconds * 3 / 2,
                       "output_tokens": 240, "near_context_limit": 0, "cut_off": 0},
            "text": {"calls": 2, "prompt_sec": 4, "output_sec": 50, "output_tokens": 500,
                     "near_context_limit": text_near_limit, "cut_off": 0},
        }


def test_runs_are_recorded_and_capped(env):
    settings, _state = env
    for _ in range(7):
        auto_tune.record_run(FakeProgress(), {"metadata": {"duration": 138.0}}, settings)
    history = auto_tune.read_history()
    assert len(history) == 5
    assert history[0]["vision"]["seconds_per_call"] == pytest.approx(40.0)
    assert history[0]["text"]["tokens_per_sec"] == pytest.approx(10.0)
    assert history[0]["video_seconds"] == 138.0


def test_slow_vision_on_cpu_suggests_turning_it_off(env):
    settings, state = env
    state["hw"] = hardware(ram=16.0)
    auto_tune.record_run(FakeProgress(vision_seconds=35.0), {"metadata": {"duration": 60}}, settings)
    tips = auto_tune.build_plan(settings, refresh=True)["tips"]
    assert any("turning it off" in tip for tip in tips)
    # Suggestions only: nothing was changed.
    assert settings.enable_visual_analysis is True


def test_full_context_suggests_more_memory_when_ram_allows(env):
    settings, state = env
    state["hw"] = hardware(ram=32.0)
    auto_tune.record_run(FakeProgress(vision_seconds=5.0, text_near_limit=2), {"metadata": {}}, settings)
    tips = auto_tune.build_plan(settings, refresh=True)["tips"]
    assert any(str(settings.ollama_text_num_ctx * 2) in tip for tip in tips)


def test_fast_runs_give_no_tips(env):
    settings, _state = env
    progress = FakeProgress(vision_seconds=4.0)
    auto_tune.record_run(progress, {"metadata": {}}, settings)
    assert auto_tune.build_plan(settings)["tips"] == []


# API


def test_optimization_api(env):
    from app import create_app

    settings, _state = env
    client = create_app(settings, start_background=False).test_client()
    plan = client.get("/api/system/ai/optimization").get_json()
    assert plan["enabled"] is False and plan["changes"]
    assert client.post("/api/system/ai/optimization", json={"enabled": "yes"}).status_code == 400
    assert client.post("/api/system/ai/optimization/undo").status_code == 400
    assert client.post("/api/system/ai/optimization", json={"enabled": True}).get_json()["enabled"] is True
    assert client.post("/api/system/ai/optimization/undo").get_json()["enabled"] is False
    assert settings.ollama_vision_model == "qwen3-vl:8b"
