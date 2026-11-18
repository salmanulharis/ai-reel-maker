"""Saved highlight presets."""

import pytest

from app import create_app
from app.config import Settings


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


def test_save_list_replace_and_delete(client):
    assert client.get("/api/presets").get_json() == {"presets": []}
    form = {"style": "exciting", "min_duration": 15, "crop": {"x": 0.1}, "cropEnabled": True, "must_include": "0-5"}
    saved = client.post("/api/presets", json={"name": "  Weekly   vlog ", "form": form})
    assert saved.status_code == 200
    presets = saved.get_json()["presets"]
    assert [p["name"] for p in presets] == ["Weekly vlog"]
    # Choices tied to one video are not kept.
    assert presets[0]["form"] == {"style": "exciting", "min_duration": 15}

    client.post("/api/presets", json={"name": "Another", "form": {"style": "social"}})
    replaced = client.post("/api/presets", json={"name": "weekly VLOG", "form": {"style": "informative"}})
    names = [p["name"] for p in replaced.get_json()["presets"]]
    assert names == ["Another", "weekly VLOG"]

    deleted = client.delete("/api/presets?name=ANOTHER")
    assert [p["name"] for p in deleted.get_json()["presets"]] == ["weekly VLOG"]
    assert client.delete("/api/presets?name=missing").status_code == 404


def test_presets_do_not_appear_in_settings(client):
    client.post("/api/presets", json={"name": "A", "form": {"style": "social"}})
    assert "highlight_presets" not in client.get("/api/settings").get_json()["settings"]


@pytest.mark.parametrize(
    "body, message",
    [
        ({"name": "", "form": {}}, "preset name is required"),
        ({"name": "x" * 61, "form": {}}, "preset name is too long (at most 60 characters)"),
        ({"name": "A", "form": "nope"}, "preset settings are missing"),
        ({"name": "A", "form": {"custom_focus": "x" * 25_000}}, "preset settings are too large"),
    ],
)
def test_invalid_presets_are_rejected(client, body, message):
    response = client.post("/api/presets", json=body)
    assert response.status_code == 400
    assert response.get_json()["error"] == message


def test_preset_limit(client):
    for i in range(30):
        assert client.post("/api/presets", json={"name": f"P{i:02d}", "form": {}}).status_code == 200
    full = client.post("/api/presets", json={"name": "One more", "form": {}})
    assert full.status_code == 400
    # Replacing an existing one still works at the limit.
    assert client.post("/api/presets", json={"name": "p00", "form": {"style": "social"}}).status_code == 200


def test_damaged_storage_reads_as_empty(client):
    from app.services.presets import SETTING_KEY, list_presets
    from app.services.settings_store import write_setting

    write_setting(SETTING_KEY, "{broken")
    assert list_presets() == []
    write_setting(SETTING_KEY, '[{"name": 3}, {"name": "ok", "form": {}}, "junk"]')
    assert [p["name"] for p in list_presets()] == ["ok"]
