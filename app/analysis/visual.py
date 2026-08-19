from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from collections.abc import Callable

from app.ai.ollama import OllamaClient, OllamaError
from app.ai.prompts import VISUAL_BATCH_SCHEMA, visual_batch_prompt
from app.models.schemas import Scene, VisualAnalysis
from app.utils.timefmt import clamp_timestamp

if TYPE_CHECKING:
    from app.config import Settings


def sample_timestamps(
    scene: Scene,
    source_duration: float | None = None,
    density: str = "auto",
) -> list[float]:
    duration = max(scene.duration, 0.01)
    media_end = min(scene.end, source_duration) if source_duration else scene.end
    if density == "lite":
        if duration < 4:
            points = [0.5]
        else:
            points = [0.0, 0.5, 1.0]
    elif duration < 1.5:
        points = [0.5]
    elif duration < 4:
        points = [0.0, 0.5, 1.0]
    elif duration < 12:
        points = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        extra = min(4, int(duration // 8))
        steps = 5 + extra
        points = [i / (steps - 1) for i in range(steps)]
    stamps = []
    for point in points:
        raw = scene.start + point * duration
        stamps.append(clamp_timestamp(raw, media_end, pad=0.15))
    unique: list[float] = []
    for stamp in stamps:
        if not unique or abs(stamp - unique[-1]) > 0.04:
            unique.append(stamp)
    return unique


def motion_score(frame_paths: list[Path]) -> float:
    if len(frame_paths) < 2:
        return 0.2
    prev = None
    diffs = []
    try:
        import cv2
    except ImportError:
        return 0.4
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        small = cv2.resize(image, (64, 64))
        if prev is not None:
            diffs.append(float(np.mean(np.abs(small.astype(float) - prev.astype(float)))) / 255.0)
        prev = small
    if not diffs:
        return 0.2
    return float(min(1.0, np.mean(diffs) * 4))


def fallback_visual(frame_paths: list[Path], scene: Scene, source_duration: float) -> VisualAnalysis:
    motion = motion_score(frame_paths)
    intro = scene.start < max(8.0, source_duration * 0.08)
    ending = scene.end > source_duration * 0.9
    return VisualAnalysis(
        visual_summary=f"Scene {scene.scene_id} at {scene.start:.1f}–{scene.end:.1f}s. Deterministic visual fallback.",
        visual_importance=min(1.0, 0.35 + motion * 0.5),
        action_strength=motion,
        information_density=0.4 + (0.1 if not intro else 0.0),
        visual_change=motion,
        repetition_likelihood=0.2 if motion < 0.15 else 0.08,
        context_dependency=0.4 if intro else 0.25,
        appears_intro=intro,
        appears_ending=ending,
    )


def _pick_representative_frame(frame_paths: list[Path]) -> Path | None:
    existing = [path for path in frame_paths if path.exists()]
    if not existing:
        return None
    return existing[len(existing) // 2]


def _expand_visual_score(raw: dict, fallback: VisualAnalysis, scene: Scene) -> VisualAnalysis:
    importance = float(raw.get("visual_importance", fallback.visual_importance))
    action = float(raw.get("action_strength", fallback.action_strength))
    density = float(raw.get("information_density", fallback.information_density))
    if importance > 0.7 or action > 0.65:
        summary = f"Strong visual moment in scene {scene.scene_id} ({scene.start:.1f}-{scene.end:.1f}s)."
    elif importance > 0.4:
        summary = f"Moderate visual content in scene {scene.scene_id} ({scene.start:.1f}-{scene.end:.1f}s)."
    else:
        summary = fallback.visual_summary
    return VisualAnalysis.model_validate(
        {
            **fallback.model_dump(),
            **raw,
            "visual_summary": raw.get("visual_summary") or summary,
            "visual_importance": importance,
            "action_strength": action,
            "information_density": density,
            "visual_change": raw.get("visual_change", fallback.visual_change),
            "repetition_likelihood": raw.get("repetition_likelihood", fallback.repetition_likelihood),
            "context_dependency": raw.get("context_dependency", fallback.context_dependency),
            "appears_intro": raw.get("appears_intro", fallback.appears_intro),
            "appears_ending": raw.get("appears_ending", fallback.appears_ending),
        }
    )


def analyze_scenes_visual(
    client: OllamaClient | None,
    model: str,
    scenes: list[dict],
    frame_map: dict[str, list[dict]],
    source_duration: float,
    custom_focus: str = "",
    enabled: bool = True,
    settings: Settings | None = None,
    on_batch_start: Callable[[int, int, int], None] | None = None,
) -> tuple[dict[int, VisualAnalysis], list[str]]:
    from app.config import get_settings

    cfg = settings or get_settings()
    results: dict[int, VisualAnalysis] = {}
    errors: list[str] = []
    ai_queue: list[tuple[Scene, list[Path], Path, VisualAnalysis]] = []

    for scene in scenes:
        scene_model = Scene.model_validate(scene)
        paths = [Path(entry["image_path"]) for entry in frame_map.get(str(scene_model.scene_id), [])]
        fallback = fallback_visual(paths, scene_model, source_duration)
        if not enabled or client is None:
            results[scene_model.scene_id] = fallback
            continue
        if scene_model.duration < cfg.ollama_vision_min_scene_duration:
            results[scene_model.scene_id] = fallback
            errors.append(
                f"Scene {scene_model.scene_id} skipped ({scene_model.duration:.1f}s < "
                f"{cfg.ollama_vision_min_scene_duration:.1f}s); using motion fallback."
            )
            continue
        frame = _pick_representative_frame(paths)
        if frame is None:
            results[scene_model.scene_id] = fallback
            errors.append(f"Scene {scene_model.scene_id}: no frame image; using motion fallback.")
            continue
        ai_queue.append((scene_model, paths, frame, fallback))

    if not ai_queue or client is None:
        return results, errors

    batch_size = max(1, cfg.ollama_vision_batch_size)
    vision_options = {
        "num_ctx": cfg.ollama_vision_num_ctx,
        "num_predict": cfg.ollama_vision_num_predict,
    }
    batch_count = (len(ai_queue) + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, len(ai_queue), batch_size), start=1):
        batch = ai_queue[start : start + batch_size]
        if on_batch_start:
            on_batch_start(batch_index, batch_count, len(batch))
        scene_meta = [
            {"scene_id": scene.scene_id, "start": scene.start, "end": scene.end}
            for scene, _, _, _ in batch
        ]
        try:
            images = [client.encode_image(str(frame)) for _, _, frame, _ in batch]
            data = client.chat(
                model=model,
                prompt=visual_batch_prompt(scene_meta, custom_focus),
                images=images,
                schema=VISUAL_BATCH_SCHEMA,
                options=vision_options,
                schema_first=False,
            )
            by_id = {
                int(row.get("scene_id")): row
                for row in (data.get("scenes") or [])
                if row.get("scene_id") is not None
            }
            for scene_model, paths, _, fallback in batch:
                raw = by_id.get(scene_model.scene_id, {})
                if not raw:
                    results[scene_model.scene_id] = fallback
                    errors.append(f"Scene {scene_model.scene_id}: missing AI score; using motion fallback.")
                    continue
                results[scene_model.scene_id] = _expand_visual_score(raw, fallback, scene_model)
        except (OllamaError, Exception) as exc:  # noqa: BLE001
            detail = str(exc).replace("\n", " ").strip()[:220]
            errors.append(f"Visual batch {batch_index}/{batch_count} failed ({detail}); using motion fallback.")
            for scene_model, paths, _, fallback in batch:
                results[scene_model.scene_id] = fallback

    return results, errors
