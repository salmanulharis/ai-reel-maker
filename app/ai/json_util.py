from __future__ import annotations

import json
import re
from typing import Any

from app.utils.timefmt import clamp


def clamp_unit(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        text = value.strip().lower().replace("%", "")
        try:
            number = float(text)
        except ValueError:
            return 0.0
        if number > 1.0:
            number = number / 100.0
        return clamp(number)
    try:
        return clamp(float(value))
    except (TypeError, ValueError):
        return 0.0


def _repair_json(text: str) -> str:
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = text.replace("'", '"')
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def parse_model_json(raw: str) -> dict[str, Any]:
    if not raw or not str(raw).strip():
        raise ValueError("Empty model response")
    text = str(raw).strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model response")
    blob = _repair_json(match.group(0))
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("Model JSON was not an object")
    return data
