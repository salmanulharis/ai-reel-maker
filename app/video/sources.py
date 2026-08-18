from __future__ import annotations

import json
from pathlib import Path

SOURCES_FILE = "sources.json"


def source_id_for(index: int) -> str:
    return f"v{index}"


def save_sources(job_dir: Path, sources: list[dict]) -> None:
    path = Path(job_dir) / SOURCES_FILE
    path.write_text(json.dumps(sources, indent=2), encoding="utf-8")


def load_sources(job_dir: Path) -> list[dict]:
    path = Path(job_dir) / SOURCES_FILE
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def fallback_source(input_path: str | Path, filename: str = "") -> dict:
    path = Path(input_path)
    return {
        "id": "v0",
        "index": 0,
        "filename": filename or path.name,
        "path": str(path),
        "analysis_path": str(path),
        "duration": 0.0,
        "width": 0,
        "height": 0,
    }


def working_path(source: dict) -> Path:
    return Path(source.get("analysis_path") or source.get("path") or "")


def tag_candidates(candidates: list[dict], source: dict) -> list[dict]:
    tagged = []
    sid = str(source.get("id") or "v0")
    name = str(source.get("filename") or working_path(source).name)
    path = str(working_path(source))
    for item in candidates:
        row = dict(item)
        row["source_id"] = sid
        row["source_name"] = name
        row["source_path"] = path
        raw_id = str(row.get("id") or "clip")
        if not raw_id.startswith(f"{sid}:"):
            row["id"] = f"{sid}:{raw_id}"
        topic = str(row.get("topic") or "")
        if topic.startswith("scene-"):
            row["topic"] = f"{sid}-{topic}"
        tagged.append(row)
    return tagged


def lookup_path(sources: list[dict], source_id: str | None, default: Path | None = None) -> Path | None:
    if source_id:
        for item in sources:
            if item.get("id") == source_id:
                return working_path(item)
    if sources:
        return working_path(sources[0])
    return default
