from pathlib import Path

import requests

from app.ai.ollama import OllamaClient, _model_present
from app.config import Settings, default_ollama_base_url
from app.services.hardware import recommend_models


def test_parse_model_ref_splits_library_tags():
    from app.ai.registry_pull import parse_model_ref

    assert parse_model_ref("qwen3-vl:4b") == ("library", "qwen3-vl", "4b")
    assert parse_model_ref("qwen3") == ("library", "qwen3", "latest")


def test_model_present_requires_matching_tag():
    models = ["qwen3-vl:4b", "qwen3:4b"]
    assert _model_present("qwen3-vl:4b", models)
    assert not _model_present("qwen3-vl:8b", models)
    assert not _model_present("qwen3:8b", models)


def test_model_present_accepts_tagged_alias():
    assert _model_present("qwen3-vl:8b", ["qwen3-vl:8b"])
    assert _model_present("qwen3-vl:8b", ["qwen3-vl:8b:latest"])


def test_recommend_models_for_16gb_ram_no_vram():
    vision, text = recommend_models(None, 14.8)
    assert vision == "qwen3-vl:4b"
    assert text == "qwen3:4b"


def test_native_default_ollama_url_is_localhost(monkeypatch):
    monkeypatch.setattr("app.config.Path.exists", lambda self: False)
    assert default_ollama_base_url() == "http://127.0.0.1:11434"


def test_docker_default_ollama_url_uses_host_gateway(monkeypatch):
    original = Path.exists

    def fake_exists(self):
        if str(self) in {"/.dockerenv", "\\.dockerenv"} or self.as_posix() == "/.dockerenv":
            return True
        return original(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert default_ollama_base_url() == "http://host.docker.internal:11434"


def test_client_falls_back_from_unresolvable_docker_host(monkeypatch):
    settings = Settings(ollama_base_url="http://host.docker.internal:11434")
    client = OllamaClient(settings)
    calls: list[str] = []

    class FakeResp:
        ok = True
        status_code = 200

        def json(self):
            return {"version": "0.32.14", "models": [{"name": "qwen3-vl:4b"}]}

        def raise_for_status(self):
            return None

        @property
        def text(self):
            return ""

    def fake_request(self, method, url, **kwargs):
        calls.append(url)
        if "host.docker.internal" in url:
            raise requests.exceptions.ConnectionError("Failed to resolve 'host.docker.internal'")
        return FakeResp()

    monkeypatch.setattr("app.ai.ollama.requests.Session.request", fake_request)
    health = client.health()
    assert health["connected"] is True
    assert any("127.0.0.1" in url for url in calls)
    assert client.root.startswith("http://127.0.0.1")


def test_pull_model_uses_hour_timeout_and_streams(monkeypatch):
    settings = Settings(ollama_base_url="http://127.0.0.1:11434", ollama_timeout=300)
    client = OllamaClient(settings)
    captured: dict = {}

    class FakeResp:
        ok = True
        status_code = 200
        text = ""

        def iter_lines(self):
            yield b'{"status":"pulling manifest"}'
            yield b'{"status":"success"}'

    def fake_request(self, method, url, **kwargs):
        captured.update(kwargs)
        captured["url"] = url
        captured["method"] = method
        return FakeResp()

    monkeypatch.setattr("app.ai.ollama.requests.Session.request", fake_request)
    client.pull_model("qwen3-vl:4b")
    assert captured["timeout"] >= 3600
    assert captured.get("stream") is True
    assert captured["json"]["stream"] is True
