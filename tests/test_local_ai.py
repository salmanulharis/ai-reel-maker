from app.ai.ollama import describe_placement
from app.config import Settings, get_settings, set_settings
from app.db import init_db
from app.services.ai_profile import apply_recommendation, fingerprint, resolve_runtime, sync_hardware_profile
from app.services.ai_test import explain_ai_error
from app.services.hardware import detect_hardware
from app.services.model_catalog import recommend_setup
from app.services.settings_store import apply_values


def test_cpu_ram_does_not_select_large_models():
    setup = recommend_setup(None, 64, None, 8)
    assert setup["inference_mode"] == "cpu"
    assert setup["vision_model"] == "qwen3-vl:4b-instruct"
    assert setup["text_model"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert setup["whisper_device"] == "cpu"
    assert setup["warm_both"] is False
    assert "qwen3-vl:8b-instruct" in setup["alternatives"]["vision"]


def test_low_ram_cpu_uses_smallest_vision_model():
    setup = recommend_setup(0, 8, 0, 4)
    assert setup["vision_model"] == "qwen3-vl:2b-instruct"
    assert setup["whisper_model"] == "base"
    assert setup["suggest_disable_vision"] is True


def test_small_gpu_prefers_4b_and_offers_8b():
    setup = recommend_setup(8, 32, 7.5, 8)
    assert setup["vision_model"] == "qwen3-vl:4b-instruct"
    assert setup["text_model"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert "qwen3-vl:8b-instruct" in setup["alternatives"]["vision"]


def test_mid_gpu_selects_8b_pair():
    setup = recommend_setup(12, 32, 11, 8)
    assert setup["vision_model"] == "qwen3-vl:8b-instruct"
    assert setup["text_model"] == "qwen3:8b"
    assert setup["whisper_model"] == "small"


def test_large_gpu_can_select_30b_vision():
    setup = recommend_setup(24, 64, 20, 16)
    assert setup["vision_model"] == "qwen3-vl:30b-a3b-instruct"
    assert setup["text_model"] == "qwen3:8b"
    assert setup["whisper_model"] == "large-v3-turbo"
    assert setup["whisper_device"] == "cuda"
    assert "distil-large-v3" in setup["alternatives"]["whisper"]


def test_large_card_with_little_free_memory_steps_down():
    setup = recommend_setup(24, 64, 4, 16)
    assert setup["vision_model"] == "qwen3-vl:4b-instruct"


def test_low_vram_stays_on_cpu_whisper():
    setup = recommend_setup(4, 16, 3.5, 4)
    assert setup["vision_model"] == "qwen3-vl:2b-instruct"
    assert setup["whisper_model"] == "base"
    assert setup["whisper_device"] == "cpu"


def test_fast_detect_skips_encoder_probe(monkeypatch):
    def fail_probe():
        raise AssertionError("encoder probe should not run")

    monkeypatch.setattr("app.services.hardware.preferred_hw_encoder", fail_probe)
    monkeypatch.setattr("app.services.hardware._nvidia_gpus", lambda: [])
    monkeypatch.setattr("app.services.hardware._apple_gpu", lambda _ram: None)
    monkeypatch.setattr("app.services.hardware._rocm_gpu", lambda: None)
    monkeypatch.setattr("app.services.hardware._windows_gpu", lambda: None)
    monkeypatch.setattr("app.services.hardware._sysfs_gpu", lambda: None)
    monkeypatch.setattr(
        "app.services.hardware._memory",
        lambda: {"cpu": "Test CPU", "ram_gb": 64, "ram_available_gb": 32, "cores": 8, "threads": 16},
    )
    info = detect_hardware()
    assert info["encode_acceleration"] == "Not checked"
    assert info["inference_mode"] == "cpu"
    assert info["inference_label"] == "CPU-only"
    assert info["recommended_vision_model"] == "qwen3-vl:4b-instruct"
    assert info["acceleration"] == info["encode_acceleration"]


def test_encode_status_is_separate_from_inference(monkeypatch):
    monkeypatch.setattr("app.services.hardware.detect_ffmpeg_encoders", lambda: ["h264_nvenc", "libx264"])
    monkeypatch.setattr("app.services.hardware.preferred_hw_encoder", lambda: "h264_nvenc")
    monkeypatch.setattr("app.services.hardware._nvidia_gpus", lambda: [])
    monkeypatch.setattr("app.services.hardware._apple_gpu", lambda _ram: None)
    monkeypatch.setattr("app.services.hardware._rocm_gpu", lambda: None)
    monkeypatch.setattr("app.services.hardware._windows_gpu", lambda: None)
    monkeypatch.setattr("app.services.hardware._sysfs_gpu", lambda: None)
    monkeypatch.setattr(
        "app.services.hardware._memory",
        lambda: {"cpu": "Test CPU", "ram_gb": 32, "ram_available_gb": 16, "cores": 8, "threads": 8},
    )
    info = detect_hardware(include_encoders=True)
    assert info["inference_mode"] == "cpu"
    assert info["encode_acceleration"] == "NVIDIA NVENC"


def test_placement_reports_gpu_cpu_and_split():
    assert describe_placement([{"name": "m", "size": 100, "size_vram": 100}])["mode"] == "gpu"
    assert describe_placement([{"name": "m", "size": 100, "size_vram": 0}])["mode"] == "cpu"
    split = describe_placement([{"name": "m", "size": 100, "size_vram": 40}])
    assert split["mode"] == "split"
    assert split["measured"] is True


def test_integrated_graphics_without_memory_stay_on_cpu():
    runtime = resolve_runtime(
        {"gpu": "Intel UHD Graphics", "gpu_vendor": "intel", "vram_gb": None, "inference_mode": "cpu"},
        [{"name": "qwen3:4b", "size": 100, "size_vram": 0}],
    )
    assert runtime["label"] == "CPU-only"
    assert "CPU" in runtime["hint"]


def test_runtime_warns_when_gpu_is_present_but_model_is_on_cpu():
    runtime = resolve_runtime(
        {"gpu": "RTX 3060", "gpu_vendor": "nvidia", "vram_gb": 12, "cuda": True, "inference_mode": "gpu"},
        [{"name": "qwen3-vl:8b", "size": 100, "size_vram": 0}],
    )
    assert runtime["measured"] is True
    assert runtime["label"] == "GPU detected, running on CPU"
    assert "smaller model" in runtime["hint"]


def test_error_copy_names_the_next_step():
    assert "Install" in explain_ai_error("connection refused")
    assert "Install" in explain_ai_error("model 'qwen3-vl:8b' not found, try pulling it")
    assert "smaller model" in explain_ai_error("cuda out of memory")
    assert "OLLAMA_VULKAN" in explain_ai_error("runner stopped", gpu_present=True, placement_mode="cpu")


def test_auto_mode_applies_once_and_custom_mode_sticks(tmp_path, monkeypatch):
    previous = get_settings()
    settings = Settings(data_dir=tmp_path / "data", ai_config_mode="auto", ollama_vision_model="qwen3-vl:8b")
    settings.ensure_directories()
    set_settings(settings)
    init_db(settings.db_path)

    class FakeClient:
        def health(self):
            return {
                "connected": False,
                "version": None,
                "models": [],
                "running": [],
                "error": "offline",
                "model_details": [],
            }

    cpu = _hardware(recommend_setup(None, 64, None, 8), gpu=None, ram=64)
    monkeypatch.setattr("app.services.ai_profile.detect_hardware", lambda **_kwargs: cpu)
    monkeypatch.setattr("app.services.ai_profile.get_ollama_client", lambda settings=None: FakeClient())
    try:
        first = sync_hardware_profile(settings, refresh=True)
        assert settings.ollama_vision_model == "qwen3-vl:4b-instruct"
        assert settings.whisper_device == "cpu"
        assert first["profile"]["applied_now"] is True
        assert "CPU-only" in first["profile"]["auto_summary"]

        apply_values({"ai_config_mode": "custom", "ollama_vision_model": "qwen3-vl:8b"}, settings)
        gpu = _hardware(recommend_setup(24, 64, 20, 8), gpu="RTX 4090", ram=64, vram=24)
        monkeypatch.setattr("app.services.ai_profile.detect_hardware", lambda **_kwargs: gpu)
        second = sync_hardware_profile(settings, refresh=True)
        assert settings.ai_config_mode == "custom"
        assert settings.ollama_vision_model == "qwen3-vl:8b"
        assert second["profile"]["applied_now"] is False
        assert fingerprint(cpu, None) != fingerprint(gpu, None)
    finally:
        set_settings(previous)


def test_apply_recommendation_sets_auto_mode(tmp_path):
    previous = get_settings()
    settings = Settings(data_dir=tmp_path / "data", ai_config_mode="custom")
    settings.ensure_directories()
    set_settings(settings)
    init_db(settings.db_path)
    try:
        summary = apply_recommendation(settings, recommend_setup(12, 32, 11, 8))
        assert settings.ai_config_mode == "auto"
        assert settings.ollama_vision_model == "qwen3-vl:8b-instruct"
        assert "GPU" in summary
    finally:
        set_settings(previous)


def _hardware(recommendation, *, gpu, ram, vram=None):
    return {
        "cpu": "Test CPU",
        "ram_gb": ram,
        "vram_gb": vram,
        "gpu": gpu,
        "gpu_vendor": "nvidia" if gpu else None,
        "cuda": bool(gpu),
        "inference_mode": "gpu" if vram else "cpu",
        "ffmpeg": True,
        "ffprobe": True,
        "encoders_checked": False,
        "recommendation": recommendation,
        "disk_free_gb": 100,
    }
