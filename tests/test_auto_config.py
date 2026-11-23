"""Automatic configuration: detection, goals, pinned choices, held downloads, run plans,
recovery from running out of memory, and measured speed."""

from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest

from app.ai.context_fit import fitting_vision_batch, vision_ctx_for_batch
from app.ai.ollama import OllamaError
from app.analysis import score_source
from app.analysis.visual import analyze_scenes_visual
from app.config import Settings, get_settings, set_settings
from app.models.db_models import init_db
from app.services import ai_profile, auto_tune, calibration, hardware
from app.services.model_catalog import nominal_ram_gb, recommend_setup
from app.services.run_plan import build_run_plan
from app.services.settings_store import read_setting, write_setting

TEXT_4B = "qwen3:4b-instruct-2507-q4_K_M"


# Recommendations


@pytest.mark.parametrize(
    ("reading", "label"),
    [(15.8, 16), (14.8, 16), (7.7, 8), (31.9, 32), (12.0, 12), (13.0, 13.0), (None, 0.0), (300.0, 300.0)],
)
def test_memory_reads_as_the_size_on_the_label(reading, label):
    assert nominal_ram_gb(reading) == label


HARDWARE = [
    (None, 8, None, 4),
    (None, 15.8, None, 6),
    (None, 64, None, 16),
    (4, 16, 3.5, 4),
    (6, 32, 5.5, 8),
    (8, 32, 7.5, 8),
    (12, 32, 11, 8),
    (24, 64, 20, 16),
    (24, 64, 4, 16),
]


@pytest.mark.parametrize("goal", ["quality", "balanced", "fast"])
@pytest.mark.parametrize("hw", HARDWARE)
def test_recommended_batch_always_fits_its_context(hw, goal):
    setup = recommend_setup(*hw, goal=goal)
    batch, num_ctx = setup["vision_batch_size"], setup["vision_num_ctx"]
    assert fitting_vision_batch(batch, num_ctx, 256) == batch
    assert setup["goal"] == goal


def test_context_for_batch_is_the_smallest_that_fits():
    for batch in (1, 2, 3, 4):
        num_ctx = vision_ctx_for_batch(batch)
        assert fitting_vision_batch(batch, num_ctx, 256) == batch
        if num_ctx > 2048:  # 2,048 is the smallest window recommended
            assert fitting_vision_batch(batch, num_ctx - 1024, 256) < batch


def test_sixteen_gigabyte_pc_without_graphics_card():
    setup = recommend_setup(None, 15.8, None, 6)
    assert setup["vision_model"] == "qwen3-vl:4b-instruct"
    assert setup["suggest_disable_vision"] is False
    assert setup["vision_batch_size"] == 1


def test_goals_on_a_processor_only_computer():
    assert recommend_setup(None, 64, None, 8, goal="quality")["vision_model"] == "qwen3-vl:8b-instruct"
    assert recommend_setup(None, 15.8, None, 6, goal="quality")["text_model"] == "qwen3:8b"
    assert recommend_setup(None, 15.8, None, 6, goal="fast")["vision_model"] == "qwen3-vl:2b-instruct"
    # Best quality never puts a model on the processor that does not fit beside the rest.
    assert recommend_setup(None, 8, None, 4, goal="quality")["vision_model"] == "qwen3-vl:2b-instruct"


def test_best_quality_on_a_small_card_can_split_the_larger_model():
    setup = recommend_setup(6, 32, 5.5, 8, goal="quality")
    assert setup["vision_model"] == "qwen3-vl:8b-instruct"
    assert setup["warm_both"] is False
    assert any("runs on the processor" in note for note in setup["notes"])
    # Not enough system memory for the rest: no split.
    assert recommend_setup(6, 8, 5.5, 8, goal="quality")["vision_model"] == "qwen3-vl:4b-instruct"


def test_fastest_on_a_large_card_steps_down_once():
    setup = recommend_setup(12, 32, 11, 8, goal="fast")
    assert setup["vision_model"] == "qwen3-vl:4b-instruct"
    assert setup["text_model"] == TEXT_4B


def test_speech_to_text_stays_on_the_processor_without_cuda_support():
    setup = recommend_setup(24, 64, 20, 16, whisper_cuda=False)
    assert setup["whisper_device"] == "cpu"
    assert setup["whisper_model"] == "small"
    assert any("CUDA" in note for note in setup["notes"])


def test_unknown_goal_is_balanced():
    assert recommend_setup(None, 16, goal="whatever")["goal"] == "balanced"


# Hardware detection


def test_nvidia_smi_rows_are_read():
    gpus = hardware.parse_nvidia_smi("NVIDIA GeForce RTX 4070, 12282, 11000, 551.23, 47\nbad\n")
    assert gpus == [
        {"gpu": "NVIDIA GeForce RTX 4070", "vram_gb": 12.0, "vram_free_gb": 10.7, "driver": "551.23", "gpu_temp_c": 47.0}
    ]


def test_rocm_memory_is_read_from_json():
    text = 'WARNING: noise\n{"card0": {"VRAM Total Memory (B)": "17163091968", "VRAM Total Used Memory (B)": "1073741824"}}'
    assert hardware.parse_rocm_vram(text) == (16.0, 15.0)
    assert hardware.parse_rocm_vram("not json") == (None, None)


def test_windows_adapters_tell_built_in_graphics_from_cards():
    gpus = hardware.adapters_to_gpus(
        [
            {"name": "Intel(R) UHD Graphics 730", "memory_bytes": 128 * 1024**2},
            {"name": "Intel(R) Arc(TM) A770 Graphics", "memory_bytes": 16 * 1024**3},
            {"name": "AMD Radeon RX 7800 XT", "memory_bytes": 16 * 1024**3},
            {"name": "AMD Radeon(TM) Graphics", "memory_bytes": 512 * 1024**2},
            {"name": "Microsoft Basic Display Adapter", "memory_bytes": None},
        ]
    )
    summary = [(gpu["gpu_vendor"], gpu["gpu_integrated"], gpu["vram_gb"]) for gpu in gpus]
    assert summary == [("intel", True, None), ("intel", False, 16.0), ("amd", False, 16.0), ("amd", True, None)]
    assert hardware._pick_gpu(gpus)["gpu"] == "Intel(R) Arc(TM) A770 Graphics"


def test_removed_cards_left_in_the_registry_are_ignored():
    registry = [{"name": "NVIDIA GeForce GTX 1060", "memory_bytes": 6 * 1024**3}, {"name": "AMD Radeon RX 6600", "memory_bytes": 8 * 1024**3}]
    present = [{"name": "AMD Radeon RX 6600", "memory_bytes": None}]
    assert hardware.present_adapters(registry, present) == [registry[1]]
    assert hardware.present_adapters(registry, []) == registry
    assert hardware.present_adapters([], present) == present


def _core_record(efficiency_class: int) -> bytes:
    # RelationProcessorCore (0), record size 48, flags, efficiency class, padding.
    return struct.pack("<IIBB", 0, 48, 0, efficiency_class) + bytes(38)


def test_performance_and_efficiency_cores_are_counted():
    buffer = b"".join(_core_record(1) for _ in range(6)) + b"".join(_core_record(0) for _ in range(4))
    assert hardware.split_core_classes(hardware.parse_core_classes(buffer)) == (6, 4)
    same = b"".join(_core_record(0) for _ in range(6))
    assert hardware.split_core_classes(hardware.parse_core_classes(same)) == (None, None)
    assert hardware.count_cpu_list("0-11,16") == 13


@pytest.fixture
def env(tmp_path, monkeypatch):
    previous = get_settings()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", ai_config_mode="custom")
    settings.ensure_directories()
    set_settings(settings)
    init_db(settings.db_path)
    yield settings
    set_settings(previous)


def _detect_with(monkeypatch, gpus, memory=None):
    monkeypatch.setattr("app.services.hardware._all_gpus", lambda _ram: [dict(gpu) for gpu in gpus])
    monkeypatch.setattr(
        "app.services.hardware._memory",
        lambda: memory or {"cpu": "Test CPU", "ram_gb": 32, "ram_available_gb": 20, "cores": 8, "threads": 16},
    )
    return hardware.detect_hardware()


def test_amd_card_on_windows_counts_once_ollama_has_used_it(env, monkeypatch):
    card = {"gpu": "AMD Radeon RX 7800 XT", "gpu_vendor": "amd", "gpu_integrated": False, "vram_gb": 16.0}
    first = _detect_with(monkeypatch, [card])
    assert first["inference_mode"] == "cpu"
    assert first["gpu_unconfirmed"] is True
    assert first["recommendation"]["inference_mode"] == "cpu"
    write_setting(hardware.GPU_CONFIRMED_KEY, "AMD Radeon RX 7800 XT")
    second = _detect_with(monkeypatch, [card])
    assert second["inference_mode"] == "gpu"
    assert second["recommendation"]["vision_model"] == "qwen3-vl:8b-instruct"


def test_built_in_graphics_never_select_gpu_models(env, monkeypatch):
    igpu = {"gpu": "Intel(R) UHD Graphics 730", "gpu_vendor": "intel", "gpu_integrated": True, "vram_gb": None}
    info = _detect_with(monkeypatch, [igpu])
    assert info["inference_mode"] == "cpu" and info["gpu_unconfirmed"] is False


def test_threads_follow_performance_cores(env, monkeypatch):
    memory = {
        "cpu": "Hybrid CPU", "ram_gb": 32, "ram_available_gb": 20, "cores": 14, "threads": 20,
        "performance_cores": 6, "efficiency_cores": 8,
    }
    info = _detect_with(monkeypatch, [], memory)
    assert info["recommendation"]["num_thread"] == 5


def test_ollama_seen_on_the_card_is_remembered(env, monkeypatch):
    class LocalClient:
        def is_local(self):
            return True

    monkeypatch.setattr("app.services.ai_profile.get_ollama_client", lambda settings=None: LocalClient())
    detected = {"gpu": "AMD Radeon RX 7800 XT", "gpu_unconfirmed": True}
    running = {"running": [{"name": "qwen3-vl:4b-instruct", "size": 100, "size_vram": 100}]}
    assert ai_profile._confirm_gpu(detected, {"running": []}, env) is False
    assert ai_profile._confirm_gpu(detected, running, env) is True
    assert read_setting(hardware.GPU_CONFIRMED_KEY) == "AMD Radeon RX 7800 XT"


def test_automatic_whisper_device_uses_ctranslate2(monkeypatch):
    from app.analysis import audio

    monkeypatch.setattr("app.services.hardware.whisper_cuda_available", lambda: True)
    assert audio._select_whisper_device("auto") == ("cuda", "float16")
    assert audio._select_whisper_device("auto", "int8_float16") == ("cuda", "int8_float16")
    monkeypatch.setattr("app.services.hardware.whisper_cuda_available", lambda: False)
    assert audio._select_whisper_device("auto") == ("cpu", "int8")


def test_loaded_whisper_models_can_be_released():
    from app.analysis import audio

    audio._whisper_models[("tiny", "cpu", "int8", 1, 1)] = object()
    assert audio.release_whisper_models() == 1
    assert audio._whisper_models == {}


# Automatic mode: pinned choices, held downloads, Ollama elsewhere


class HealthClient:
    def __init__(self, models=None, connected=True, local=True):
        self.models = models or []
        self.connected = connected
        self.local = local

    def health(self):
        return {"connected": self.connected, "version": "0.13.0", "models": list(self.models), "running": []}

    def shares_cpu(self):
        return self.local

    def is_local(self):
        return self.local


def _cpu_hardware(ram=15.8):
    return {
        "cpu": "Test CPU", "cpu_cores": 6, "cpu_threads": 12, "ram_gb": ram, "ram_available_gb": 8.0,
        "disk_free_gb": 100.0, "gpu": None, "vram_gb": None, "inference_mode": "cpu",
        "recommendation": recommend_setup(None, ram, None, 6),
    }


@pytest.fixture
def auto_env(env, monkeypatch):
    state = {"hw": _cpu_hardware(), "client": HealthClient(models=["qwen3-vl:2b-instruct", TEXT_4B])}
    env.ollama_vision_model = "qwen3-vl:2b-instruct"
    env.ollama_text_model = TEXT_4B

    def detect(**kwargs):
        return {**state["hw"], "recommendation": hardware.recommend_for(state["hw"], kwargs.get("goal", "balanced"))}

    monkeypatch.setattr("app.services.ai_profile.detect_hardware", detect)
    monkeypatch.setattr("app.services.ai_profile.get_ollama_client", lambda settings=None: state["client"])
    monkeypatch.setattr("app.services.auto_tune.get_ollama_client", lambda settings=None: state["client"])
    return env, state


def test_a_model_that_is_not_downloaded_waits_and_the_current_one_keeps_working(auto_env):
    settings, state = auto_env
    plan = auto_tune.enable(settings)
    assert settings.ai_config_mode == "auto"
    assert settings.ollama_vision_model == "qwen3-vl:2b-instruct"
    assert ai_profile.held_fields() == ["ollama_vision_model"]
    vision = next(item for item in plan["items"] if item["key"] == "ollama_vision_model")
    assert vision["needs_download"] is True and vision["recommended"] == "qwen3-vl:4b-instruct"
    # Other settings were applied as usual.
    assert settings.ollama_vision_num_ctx == vision_ctx_for_batch(1)

    state["client"].models.append("qwen3-vl:4b-instruct")
    ai_profile.sync_hardware_profile(settings)
    assert settings.ollama_vision_model == "qwen3-vl:4b-instruct"
    assert ai_profile.held_fields() == []


def test_with_ollama_offline_models_are_applied_as_before(auto_env):
    settings, state = auto_env
    state["client"] = HealthClient(connected=False)
    auto_tune.enable(settings)
    assert settings.ollama_vision_model == "qwen3-vl:4b-instruct"


def test_a_setting_changed_by_hand_is_kept_and_automatic_mode_stays_on(auto_env):
    from app import create_app

    settings, state = auto_env
    state["client"].models.append("qwen3-vl:4b-instruct")
    auto_tune.enable(settings)
    client = create_app(settings, start_background=False).test_client()
    assert client.post("/api/settings", json={"ollama_num_thread": 3}).status_code == 200
    assert settings.ai_config_mode == "auto"
    assert ai_profile.pinned_fields(settings) == {"ollama_num_thread"}
    # Saving an unchanged value does not pin it.
    client.post("/api/settings", json={"ollama_text_num_ctx": settings.ollama_text_num_ctx})
    assert ai_profile.pinned_fields(settings) == {"ollama_num_thread"}

    # A new recommendation (different hardware) leaves the user's choice alone.
    state["hw"] = _cpu_hardware(ram=64.0)
    ai_profile.sync_hardware_profile(settings, refresh=True)
    assert settings.ollama_num_thread == 3

    plan = auto_tune.build_plan(settings)
    threads = next(item for item in plan["items"] if item["key"] == "ollama_num_thread")
    assert threads["pinned"] is True
    assert all(item["key"] != "ollama_num_thread" for item in plan["changes"])

    plan = client.post("/api/system/ai/optimization", json={"pin": {"key": "ollama_num_thread", "pinned": False}}).get_json()
    assert settings.ollama_num_thread == 5
    assert ai_profile.pinned_fields(settings) == set()


def test_in_manual_mode_editing_does_not_pin(auto_env):
    from app import create_app

    settings, _state = auto_env
    client = create_app(settings, start_background=False).test_client()
    client.post("/api/settings", json={"ollama_num_thread": 3})
    assert settings.ai_config_mode == "custom"
    assert ai_profile.pinned_fields(settings) == set()


def test_turning_off_or_undoing_clears_pins(auto_env):
    settings, state = auto_env
    state["client"].models.append("qwen3-vl:4b-instruct")
    auto_tune.enable(settings)
    auto_tune.set_pin("ollama_num_thread", True, settings)
    auto_tune.disable(settings)
    assert ai_profile.pinned_fields(settings) == set()


def test_ollama_on_another_computer_keeps_its_model_settings(auto_env):
    settings, state = auto_env
    state["client"] = HealthClient(models=["qwen3-vl:2b-instruct"], local=False)
    settings.ollama_vision_model = "qwen3-vl:8b-instruct"
    settings.ollama_vision_num_ctx = 8192
    plan = auto_tune.enable(settings)
    assert settings.ollama_vision_model == "qwen3-vl:8b-instruct"
    assert settings.ollama_vision_num_ctx == 8192
    assert settings.ollama_num_thread == 5  # local settings are still managed
    vision = next(item for item in plan["items"] if item["key"] == "ollama_vision_model")
    assert vision["managed"] is False
    assert all(item["key"] != "ollama_vision_model" for item in plan["changes"])


def test_choosing_a_goal_applies_it_with_an_undo_point(auto_env):
    settings, state = auto_env
    state["client"].models += ["qwen3-vl:4b-instruct", "qwen3:8b"]
    auto_tune.enable(settings)
    plan = auto_tune.set_goal("quality", settings)
    assert plan["goal"] == "quality" and settings.ai_goal == "quality"
    assert settings.ollama_text_model == "qwen3:8b"
    assert plan["can_undo"] is True
    with pytest.raises(ValueError):
        auto_tune.set_goal("turbo", settings)


def test_goal_while_off_changes_only_the_recommendation(auto_env):
    settings, _state = auto_env
    plan = auto_tune.set_goal("fast", settings)
    assert settings.ai_config_mode == "custom"
    assert settings.ollama_vision_model == "qwen3-vl:2b-instruct"
    assert plan["enabled"] is False


def test_optimization_api_checks_its_input(auto_env):
    from app import create_app

    settings, _state = auto_env
    client = create_app(settings, start_background=False).test_client()
    assert client.post("/api/system/ai/optimization", json={"goal": "turbo"}).status_code == 400
    assert client.post("/api/system/ai/optimization", json={"pin": {"key": "output_crf", "pinned": True}}).status_code == 400
    assert client.post("/api/system/ai/optimization", json={"pin": {"key": "ollama_num_thread"}}).status_code == 400
    assert client.post("/api/system/ai/optimization", json={"goal": "fast"}).get_json()["goal"] == "fast"


# Run plans


def _settings(**overrides):
    values = dict(
        _env_file=None, ollama_vision_model="qwen3-vl:4b-instruct", ollama_text_model=TEXT_4B, whisper_model="base"
    )
    values.update(overrides)
    return Settings(**values)


def test_plenty_of_memory_keeps_models_loaded():
    plan = build_run_plan(
        _settings(), {"ram_gb": 32, "inference_mode": "cpu"}, ollama_here=True,
        readings={"ram_available_gb": 20, "cpu_percent": 10, "on_battery": False},
    )
    assert plan.one_at_a_time is False and plan.notes == []


def test_little_free_memory_loads_models_one_at_a_time():
    plan = build_run_plan(
        _settings(), {"ram_gb": 15.8, "inference_mode": "cpu"}, ollama_here=True,
        readings={"ram_available_gb": 5.0, "cpu_percent": 90, "on_battery": True},
    )
    assert plan.one_at_a_time is True
    text = " ".join(plan.notes)
    assert "one at a time" in text and "90%" in text and "battery" in text


def test_big_models_on_a_small_card_take_turns_with_speech_to_text():
    plan = build_run_plan(
        _settings(ollama_vision_model="qwen3-vl:8b-instruct", ollama_text_model="qwen3:8b", whisper_model="large-v3-turbo"),
        {"ram_gb": 32, "inference_mode": "gpu", "vram_gb": 8},
        ollama_here=True,
        whisper_on_gpu=True,
        readings={"ram_available_gb": 20, "cpu_percent": 10},
    )
    assert plan.one_at_a_time and plan.free_gpu_for_speech
    assert plan.whisper_gpu_compute == "int8_float16"


def test_ollama_elsewhere_does_not_compete_for_this_memory():
    plan = build_run_plan(
        _settings(), {"ram_gb": 8, "inference_mode": "cpu"}, ollama_here=False,
        readings={"ram_available_gb": 3.0, "cpu_percent": 10},
    )
    assert plan.one_at_a_time is False


# Running out of memory


class MemoryClient:
    """Fails with an out-of-memory error until ``free_after`` relief calls, or for requests
    larger than ``max_images``."""

    def __init__(self, max_images=99, always=False):
        self.max_images = max_images
        self.always = always
        self.calls: list[int] = []

    def role_options(self, role):
        return {"num_ctx": 16384}

    def encode_image(self, path):
        return "image"

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
        self.calls.append(len(images or []))
        if self.always or len(images or []) > self.max_images:
            raise OllamaError("HTTP 500: model requires more system memory (5.1 GiB) than is available")
        ids = [int(match) for match in re.findall(r"scene_id=(\d+)", prompt)]
        return {"scenes": [{"scene_id": scene_id, "visual_importance": 0.8} for scene_id in ids]}


def _scenes(tmp_path: Path, count: int):
    scenes = [{"scene_id": i, "start": i * 5.0, "end": i * 5.0 + 5.0, "duration": 5.0} for i in range(count)]
    frames = {}
    for scene in scenes:
        image = tmp_path / f"m{scene['scene_id']}.jpg"
        image.write_bytes(b"jpg")
        frames[str(scene["scene_id"])] = [{"timestamp": scene["start"] + 1, "image_path": str(image)}]
    return scenes, frames


def test_out_of_memory_frees_models_and_continues_with_smaller_batches(tmp_path):
    scenes, frames = _scenes(tmp_path, 6)
    client = MemoryClient(max_images=1)
    relieved = []
    results, _errors, reasons = analyze_scenes_visual(
        client, "m", scenes, frames, 30.0, settings=Settings(ollama_vision_batch_size=4),
        on_memory_error=lambda: relieved.append(1),
    )
    assert reasons == {}
    assert len(results) == 6
    assert relieved
    # After the first failure, later batches start at the size that worked.
    assert client.calls[-2:] == [1, 1]


def test_lasting_out_of_memory_stops_asking_and_marks_scenes_for_a_retry(tmp_path):
    scenes, frames = _scenes(tmp_path, 4)
    client = MemoryClient(always=True)
    results, errors, reasons = analyze_scenes_visual(
        client, "m", scenes, frames, 20.0, settings=Settings(ollama_vision_batch_size=1),
        on_memory_error=lambda: None,
    )
    assert len(results) == 4
    assert set(reasons.values()) == {score_source.AI_FAILED}
    assert score_source.AI_FAILED in score_source.RETRYABLE
    assert client.calls == [1, 1]  # the scene once, then once more after freeing memory
    assert any("ran out of memory" in message for message in errors)


def test_without_a_memory_handler_an_out_of_memory_batch_falls_back_as_before(tmp_path):
    scenes, frames = _scenes(tmp_path, 2)
    client = MemoryClient(always=True)
    _results, _errors, reasons = analyze_scenes_visual(
        client, "m", scenes, frames, 10.0, settings=Settings(ollama_vision_batch_size=2)
    )
    assert set(reasons.values()) == {score_source.AI_FAILED}
    assert client.calls == [2]


class TextMemoryClient:
    def __init__(self, fail_times=1):
        self.fail_times = fail_times
        self.calls: list[str] = []
        self.unloaded: list[str | None] = []

    def role_options(self, role):
        return {"num_ctx": 8192}

    def unload_others(self, keep=None):
        self.unloaded.append(keep)
        return []

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
        self.calls.append(prompt)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OllamaError("HTTP 500: cudaMalloc failed: out of memory")
        count = len(re.findall(r"^\d+\. \[", prompt, re.MULTILINE))
        return {"segments": [{"i": i, "importance": 0.8, "novelty": 0.7, "information_density": 0.6} for i in range(count)]}


def test_speech_scoring_recovers_from_running_out_of_memory(monkeypatch):
    import threading

    from app.analysis.pipeline import PipelineProgress, VideoAnalyzer
    from app.models.schemas import HighlightConfig, TranscriptSegment

    class Runner:
        cancel_event = threading.Event()

    released = []
    monkeypatch.setattr("app.analysis.pipeline.release_whisper_models", lambda: released.append("whisper"))
    monkeypatch.setattr("app.analysis.pipeline.release_matcher", lambda: released.append("matcher"))
    client = TextMemoryClient()
    settings = _settings()
    analyzer = VideoAnalyzer(settings, Runner(), client)
    logs = []
    progress = PipelineProgress(lambda kind, payload: logs.append(payload) if kind == "log" else None)
    segments = [TranscriptSegment(start=i * 2.0, end=i * 2.0 + 1.5, text=f"Sentence number {i} explains the idea.") for i in range(4)]
    scored = analyzer._score_text_batch(segments, HighlightConfig(), progress)
    assert all(item["score_source"] == score_source.AI for item in scored)
    assert client.unloaded == [TEXT_4B]
    assert released == ["whisper", "matcher"]
    assert progress.adjustments and any("ran out of memory" in entry["message"] for entry in logs)


# Measured speed


def test_estimate_from_measured_speed():
    settings = _settings()
    speeds = {"whisper_realtime_factor": 10.0, "text_tokens_per_sec": 8.0, "vision_seconds_per_image": 30.0}
    estimate = calibration.estimate(speeds, settings)
    assert estimate["complete"] is True
    assert estimate["parts"] == {"speech_to_text": 60, "speech_understanding": 720, "video_understanding": 900}
    partial = calibration.estimate({"vision_seconds_per_image": 30.0}, settings)
    assert partial["complete"] is False
    assert calibration.estimate({}, settings) is None


def test_fast_computer_is_offered_the_larger_video_model():
    settings = _settings()
    speeds = {"whisper_realtime_factor": 40.0, "text_tokens_per_sec": 40.0, "vision_seconds_per_image": 4.0}
    hw = {"ram_gb": 32, "inference_mode": "cpu"}
    tips = calibration.suggestions(settings, hw, "balanced", speeds)
    assert tips and "qwen3-vl:8b-instruct" in tips[0]
    # Best quality already chooses by size, and a small computer has no room for it.
    assert calibration.suggestions(settings, hw, "quality", speeds) == []
    assert calibration.suggestions(settings, {"ram_gb": 15.8, "inference_mode": "cpu"}, "balanced", speeds) == []


def test_slow_computer_is_offered_the_fastest_goal():
    settings = _settings()
    speeds = {"whisper_realtime_factor": 2.0, "text_tokens_per_sec": 3.0, "vision_seconds_per_image": 60.0}
    tips = calibration.suggestions(settings, {"ram_gb": 16, "inference_mode": "cpu"}, "balanced", speeds)
    assert tips and "Fastest" in tips[0]
    assert calibration.suggestions(settings, {"ram_gb": 16}, "fast", speeds) == []


def test_speeds_come_from_the_latest_run_with_the_same_models(env):
    env.ollama_text_model, env.ollama_vision_model, env.whisper_model = TEXT_4B, "qwen3-vl:4b-instruct", "base"
    write_setting(
        "perf_profile",
        '{"text": {"model": "%s", "tokens_per_sec": 5}, "vision": {"model": "qwen3-vl:2b-instruct", "seconds_per_image": 9},'
        ' "whisper": {"model": "base", "realtime_factor": 12}}' % TEXT_4B,
    )
    history = [{"text_model": TEXT_4B, "vision_model": "qwen3-vl:4b-instruct",
                "text": {"tokens_per_sec": 7.5}, "vision": {"seconds_per_image": 20.0}}]
    speeds = calibration.current_speeds(env, history)
    assert speeds == {"text_tokens_per_sec": 7.5, "vision_seconds_per_image": 20.0, "whisper_realtime_factor": 12.0}


def test_measuring_is_refused_while_a_video_is_processed(env, monkeypatch):
    from app import create_app

    monkeypatch.setattr("app.services.calibration._job_running", lambda: True)
    client = create_app(env, start_background=False).test_client()
    response = client.post("/api/system/ai/measure")
    assert response.status_code == 409
    assert client.get("/api/system/ai/measure").get_json()["state"] in {"idle", "done", "error"}


def test_measurement_stores_what_could_be_measured(env, monkeypatch):
    class ProbeClient:
        def health(self):
            return {"connected": True, "text_present": True, "vision_ready": True}

        def probe_chat(self, model, prompt, images=None, num_predict=32, role=None):
            if images:
                return {"total_sec": 12.5, "load_sec": 2.5}
            return {"tokens_per_sec": 6.4}

        def running_models(self):
            return [{"name": "qwen3-vl:4b-instruct", "size": 100, "size_vram": 0}]

    monkeypatch.setattr("app.services.calibration.get_ollama_client", lambda settings=None: ProbeClient())
    monkeypatch.setattr("app.services.calibration._measure_whisper", lambda settings: None)
    profile = calibration.measure(env)
    assert profile["text"]["tokens_per_sec"] == 6.4
    assert profile["vision"] == {"model": env.ollama_vision_model, "seconds_per_image": 10.0, "placement": "cpu"}
    assert profile["whisper"] is None


def test_runs_record_time_per_image_and_memory_trouble(env):
    class Progress:
        timings = {"visual": 60.0}
        usage = {"vision": {"calls": 2, "prompt_sec": 20, "output_sec": 40, "prompt_tokens": 4600, "output_tokens": 300}}
        adjustments = ["The video model ran out of memory."]

    auto_tune.record_run(Progress(), {"metadata": {"duration": 60}}, env)
    last = auto_tune.read_history()[0]
    assert last["vision"]["seconds_per_image"] == 15.0
    assert last["adjustments"] == ["The video model ran out of memory."]
    tips = auto_tune.tips(auto_tune.read_history(), env, {"inference_mode": "gpu"})
    assert any("ran out of memory" in tip for tip in tips)
