"""Saved highlight presets: named sets of Create-page choices, kept in the settings table."""

from __future__ import annotations

import json
import time

from app.services.settings_store import read_setting, write_setting

SETTING_KEY = "highlight_presets"
MAX_PRESETS = 30
MAX_NAME = 60
MAX_FORM_BYTES = 20_000
# Choices that belong to one video, not to a way of working: never stored in a preset.
VIDEO_SPECIFIC = {"crop", "cropEnabled", "must_include", "must_exclude"}


class PresetError(ValueError):
    """A preset that cannot be saved; the message is shown to the user."""


def list_presets() -> list[dict]:
    raw = read_setting(SETTING_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    presets = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("form"), dict):
            presets.append({"name": item["name"], "form": item["form"], "saved_at": float(item.get("saved_at") or 0)})
    return presets


def _store(presets: list[dict]) -> None:
    write_setting(SETTING_KEY, json.dumps(presets))


def clean_name(name) -> str:
    text = " ".join(str(name or "").split())
    if not text:
        raise PresetError("preset name is required")
    if len(text) > MAX_NAME:
        raise PresetError(f"preset name is too long (at most {MAX_NAME} characters)")
    return text


def save_preset(name, form) -> list[dict]:
    """Add a preset, or replace the one with the same name (ignoring case)."""
    name = clean_name(name)
    if not isinstance(form, dict):
        raise PresetError("preset settings are missing")
    form = {key: value for key, value in form.items() if isinstance(key, str) and key not in VIDEO_SPECIFIC}
    if len(json.dumps(form)) > MAX_FORM_BYTES:
        raise PresetError("preset settings are too large")
    presets = [p for p in list_presets() if p["name"].casefold() != name.casefold()]
    if len(presets) >= MAX_PRESETS:
        raise PresetError(f"you can keep up to {MAX_PRESETS} presets; delete one first")
    presets.append({"name": name, "form": form, "saved_at": time.time()})
    presets.sort(key=lambda p: p["name"].casefold())
    _store(presets)
    return presets


def delete_preset(name) -> list[dict]:
    name = clean_name(name)
    presets = list_presets()
    kept = [p for p in presets if p["name"].casefold() != name.casefold()]
    if len(kept) == len(presets):
        raise PresetError("preset not found")
    _store(kept)
    return kept
