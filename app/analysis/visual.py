from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ai.ollama import OllamaClient, OllamaError
from app.ai.prompts import VISUAL_SCHEMA, visual_prompt
from app.models.schemas import Scene, VisualAnalysis
from app.utils.timefmt import clamp_timestamp
from app.video.ffmpeg_runner import FFmpegRunner


def sample_timestamps(scene: Scene, source_duration: float | None = None) -> list[float]:
    duration = max(scene.duration, 0.01)
    media_end = min(scene.end, source_duration) if source_duration else scene.end
    if duration < 1.5:
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


def _pick_frames(frame_paths: list[Path], limit: int = 3) -> list[Path]:
    if len(frame_paths) <= limit:
        return list(frame_paths)
    mid = len(frame_paths) // 2
    picked = [frame_paths[0], frame_paths[mid], frame_paths[-1]]
    unique: list[Path] = []
    for path in picked:
        if path not in unique:
            unique.append(path)
    return unique[:limit]


def analyze_scene_visual(
    client: OllamaClient | None,
    model: str,
    frame_paths: list[Path],
    scene: Scene,
    source_duration: float,
    custom_focus: str = "",
    enabled: bool = True,
) -> tuple[VisualAnalysis, str | None]:
    fallback = fallback_visual(frame_paths, scene, source_duration)
    if not enabled or client is None or not frame_paths:
        return fallback, None
    try:
        images = [client.encode_image(str(p)) for p in _pick_frames(frame_paths) if p.exists()]
        data = client.chat(
            model=model,
            prompt=visual_prompt(custom_focus),
            images=images,
            schema=VISUAL_SCHEMA,
        )
        parsed = VisualAnalysis.model_validate({**fallback.model_dump(), **data})
        return parsed, None
    except (OllamaError, Exception) as exc:  # noqa: BLE001
        return fallback, str(exc)
