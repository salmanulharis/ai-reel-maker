from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from collections.abc import Callable

from app.ai.context_fit import (
    VISION_CAPTION_TOKENS,
    VISION_TOKENS_PER_SCENE,
    fitting_scenes_per_image,
    fitting_vision_batch,
    predict_budget,
    vision_ctx_for_batch,
)
from app.ai.ollama import OllamaClient, OllamaConnectionError, OllamaError, OllamaTruncatedError, is_memory_error
from app.ai.prompts import visual_batch_prompt, visual_batch_schema
from app.analysis import frame_grid, score_source
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


def vision_reply_tokens(describe: bool) -> int:
    """Reply tokens per scene, with room for a description when one is asked for."""
    return VISION_TOKENS_PER_SCENE + (VISION_CAPTION_TOKENS if describe else 0)


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
            "caption": " ".join(str(raw.get("caption") or "").split())[:200],
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
    shortlist: set[int] | None = None,
    avoid: str = "",
    on_memory_error: Callable[[], None] | None = None,
    on_batch_done: Callable[[dict[int, VisualAnalysis], dict[int, str]], None] | None = None,
    describe: bool = False,
) -> tuple[dict[int, VisualAnalysis], list[str], dict[int, str]]:
    """Score scenes with the vision model, falling back to motion scoring per scene.

    ``avoid`` lists content the user wants left out; the model then also rates how clearly
    each scene shows it. ``on_memory_error`` frees memory when the model runs out of it; the
    request is then retried in smaller parts. ``on_batch_done`` receives the scores and fallback
    reasons so far after every batch, so finished batches can be saved as they complete. With
    ``describe`` the model also says in a few words what each scene shows ("caption").

    Up to ``ollama_vision_scenes_per_image`` scenes share one picture (a labelled grid, see
    app.analysis.frame_grid), fewer when the context window is too small for their replies.
    A scene whose frame cannot be opened is sent on its own, one frame per picture.

    Returns the scores, log messages, and the fallback reason for every scene that did not get
    an AI score (see app.analysis.score_source)."""
    from app.config import get_settings

    cfg = settings or get_settings()
    results: dict[int, VisualAnalysis] = {}
    errors: list[str] = []
    reasons: dict[int, str] = {}
    ai_queue: list[tuple[Scene, list[Path], Path, VisualAnalysis]] = []

    def use_fallback(scene_model: Scene, fallback: VisualAnalysis, reason: str) -> None:
        results[scene_model.scene_id] = fallback
        reasons[scene_model.scene_id] = reason

    for scene in scenes:
        scene_model = Scene.model_validate(scene)
        paths = [Path(entry["image_path"]) for entry in frame_map.get(str(scene_model.scene_id), [])]
        fallback = fallback_visual(paths, scene_model, source_duration)
        if not enabled or client is None:
            use_fallback(scene_model, fallback, score_source.AI_UNAVAILABLE)
            continue
        if shortlist is not None and scene_model.scene_id not in shortlist:
            use_fallback(scene_model, fallback, score_source.NOT_SHORTLISTED)
            continue
        if scene_model.duration < cfg.ollama_vision_min_scene_duration:
            use_fallback(scene_model, fallback, score_source.SHORT_SCENE)
            errors.append(
                f"Scene {scene_model.scene_id} skipped ({scene_model.duration:.1f}s < "
                f"{cfg.ollama_vision_min_scene_duration:.1f}s); using motion fallback."
            )
            continue
        frame = _pick_representative_frame(paths)
        if frame is None:
            use_fallback(scene_model, fallback, score_source.NO_FRAME)
            errors.append(f"Scene {scene_model.scene_id}: no frame image; using motion fallback.")
            continue
        ai_queue.append((scene_model, paths, frame, fallback))

    if not ai_queue or client is None:
        return results, errors, reasons

    num_ctx = client.role_options("vision")["num_ctx"]
    per_image = vision_scenes_per_image(cfg, num_ctx, describe)
    tiled = [item for item in ai_queue if per_image > 1 and frame_grid.can_tile(item[2])]
    tiled_ids = {item[0].scene_id for item in tiled}
    # (scene, scenes per picture): grids first, then the scenes sent one frame per picture.
    queue = [(item, per_image) for item in tiled] + [
        (item, 1) for item in ai_queue if item[0].scene_id not in tiled_ids
    ]
    # Pictures per request. A request that ran out of memory lowers this for the batches that follow.
    limit = {
        "size": fitting_vision_batch(
            cfg.ollama_vision_batch_size,
            num_ctx,
            cfg.ollama_vision_num_predict,
            vision_reply_tokens(describe),
            per_image if tiled else 1,
        )
    }
    unreachable = False
    out_of_memory = False
    position = 0
    batch_index = 0

    while position < len(queue):
        grid = queue[position][1]
        batch = []
        while position < len(queue) and queue[position][1] == grid and len(batch) < limit["size"] * grid:
            batch.append(queue[position][0])
            position += 1
        batch_index += 1
        batch_count = batch_index + _requests_left(queue, position, limit["size"])
        if on_batch_start:
            on_batch_start(batch_index, batch_count, len(batch))
        if unreachable or out_of_memory:
            for scene_model, _, _, fallback in batch:
                use_fallback(scene_model, fallback, score_source.AI_FAILED)
            if on_batch_done:
                on_batch_done(results, reasons)
            continue
        try:
            by_id = _score_visual_batch(
                client, model, batch, custom_focus, cfg, avoid, on_memory_error, limit, describe=describe, grid=grid
            )
            for scene_model, _, _, fallback in batch:
                raw = by_id.get(scene_model.scene_id, {})
                if not raw:
                    use_fallback(scene_model, fallback, score_source.AI_FAILED)
                    errors.append(f"Scene {scene_model.scene_id}: missing AI score; using motion fallback.")
                    continue
                results[scene_model.scene_id] = _expand_visual_score(raw, fallback, scene_model)
        except OllamaConnectionError as exc:
            unreachable = True
            detail = str(exc).replace("\n", " ").strip()[:220]
            errors.append(
                f"Ollama stopped responding during visual batch {batch_index}/{batch_count} ({detail}); "
                "using motion fallback for the remaining scenes."
            )
            for scene_model, _, _, fallback in batch:
                use_fallback(scene_model, fallback, score_source.AI_FAILED)
        except Exception as exc:  # noqa: BLE001
            detail = str(exc).replace("\n", " ").strip()[:220]
            if is_memory_error(str(exc)):
                # Every later request would fail the same way; a later analysis redoes these scenes.
                out_of_memory = True
                errors.append(
                    f"The video model ran out of memory at batch {batch_index}/{batch_count} ({detail}). "
                    "The remaining scenes use motion scoring and are analyzed again next time. "
                    "Close other programs or choose a smaller video model."
                )
            else:
                errors.append(f"Visual batch {batch_index}/{batch_count} failed ({detail}); using motion fallback.")
            for scene_model, _, _, fallback in batch:
                use_fallback(scene_model, fallback, score_source.AI_FAILED)
        if on_batch_done:
            on_batch_done(results, reasons)

    return results, errors, reasons


def vision_scenes_per_image(cfg: "Settings", num_ctx: int, describe: bool) -> int:
    """Scenes tiled into one picture: the configured number, or fewer when one picture and the
    replies for its scenes would not fit the context window."""
    return fitting_scenes_per_image(
        getattr(cfg, "ollama_vision_scenes_per_image", 1),
        num_ctx,
        cfg.ollama_vision_num_predict,
        vision_reply_tokens(describe),
    )


def _requests_left(queue: list[tuple[tuple, int]], position: int, pictures: int) -> int:
    """Requests still needed for the scenes after ``position`` at ``pictures`` per request."""
    left = 0
    for grid in {grid for _, grid in queue[position:]}:
        scenes = sum(1 for _, each in queue[position:] if each == grid)
        per_request = max(1, pictures * grid)
        left += (scenes + per_request - 1) // per_request
    return left


def _score_visual_batch(
    client: OllamaClient,
    model: str,
    batch: list[tuple[Scene, list[Path], Path, VisualAnalysis]],
    custom_focus: str,
    cfg: Settings,
    avoid: str = "",
    on_memory_error: Callable[[], None] | None = None,
    limit: dict[str, int] | None = None,
    retried: bool = False,
    describe: bool = False,
    grid: int = 1,
) -> dict[int, dict]:
    """Raw AI rows by scene id. A reply cut off by the token limit is retried as two smaller batches.

    With ``grid`` above 1, up to that many scenes share one picture (labelled tiles); otherwise
    every scene is its own picture. When the model runs out of memory and ``on_memory_error``
    is given, it is called to free memory, and the batch is retried in halves, then once as a
    single picture with the smallest context window. ``limit["size"]`` (pictures per request)
    is lowered so later batches start at the size that worked."""
    scene_meta = [
        {"scene_id": scene.scene_id, "start": scene.start, "end": scene.end}
        for scene, _, _, _ in batch
    ]
    tiles = max(1, int(grid))
    options = client.role_options("vision")
    if retried:
        smallest = (
            vision_ctx_for_batch(1, cfg.ollama_vision_num_predict)
            if tiles == 1
            else vision_ctx_for_batch(1, cfg.ollama_vision_num_predict, vision_reply_tokens(describe), len(batch))
        )
        options["num_ctx"] = min(options["num_ctx"], smallest)
    options["num_predict"] = predict_budget(
        cfg.ollama_vision_num_predict, len(batch), vision_reply_tokens(describe), options["num_ctx"]
    )
    if tiles > 1:
        quality = int(getattr(cfg, "ollama_vision_jpeg_quality", 75) or 75)
        images = [
            frame_grid.encode(
                frame_grid.compose(
                    [(scene.scene_id, frame) for scene, _, frame, _ in batch[start : start + tiles]],
                    cfg.ollama_vision_max_pixels,
                ),
                quality,
            )
            for start in range(0, len(batch), tiles)
        ]
    else:
        images = [client.encode_image(str(frame)) for _, _, frame, _ in batch]

    def halves() -> dict[int, dict]:
        # Between pictures when there are several, otherwise between the scenes of the one picture.
        middle = (len(images) // 2) * tiles if len(images) > 1 else len(batch) // 2
        return {
            **_score_visual_batch(
                client, model, batch[:middle], custom_focus, cfg, avoid, on_memory_error, limit,
                describe=describe, grid=grid,
            ),
            **_score_visual_batch(
                client, model, batch[middle:], custom_focus, cfg, avoid, on_memory_error, limit,
                describe=describe, grid=grid,
            ),
        }

    try:
        data = client.chat(
            model=model,
            prompt=visual_batch_prompt(scene_meta, custom_focus, avoid, describe, grid=tiles > 1),
            images=images,
            schema=visual_batch_schema(bool(avoid), describe),
            options=options,
            schema_first=False,
        )
    except OllamaTruncatedError:
        if len(batch) <= 1:
            raise
        return halves()
    except OllamaError as exc:
        if on_memory_error is None or not is_memory_error(str(exc)) or (len(images) <= 1 and retried):
            raise
        on_memory_error()
        if limit is not None:
            limit["size"] = max(1, len(images) // 2)
        if len(images) > 1:
            return halves()
        # One picture (a single frame or one grid): smaller grids would cost as much memory.
        return _score_visual_batch(
            client, model, batch, custom_focus, cfg, avoid, on_memory_error, limit,
            retried=True, describe=describe, grid=grid,
        )
    by_id: dict[int, dict] = {}
    for row in data.get("scenes") or []:
        if not isinstance(row, dict) or row.get("scene_id") is None:
            continue
        try:
            by_id[int(row["scene_id"])] = row
        except (TypeError, ValueError):
            continue
    return by_id


# Shortlist sizing: cover this many times the highlight length with the best-scored speech,
# and never shortlist fewer scenes than this.
_SHORTLIST_COVERAGE = 3.0
_SHORTLIST_MIN_SCENES = 6
# A scene with less speech than this share of its length counts as visual-only content.
_VISUAL_ONLY_SPEECH = 0.3
# When the shortlist would hold this share of scenes anyway, analyze them all.
_SHORTLIST_ALL_SHARE = 0.8


def shortlist_scenes(
    scenes: list[dict],
    segment_scores: list[dict],
    highlight_seconds: float,
    scene_quality: dict[int, dict] | None = None,
    visual_signal: dict[int, float] | None = None,
    match: dict[int, dict] | None = None,
) -> set[int] | None:
    """Scenes worth the vision model's time, or None to analyze every scene.

    With speech: every scene with little speech (its content is visual), then the scenes behind
    the best-scored speech, until about three times the highlight length is covered. Without
    speech: the scenes with the strongest quick picture, motion, and sound signal, to the same
    budget. Black scenes and scenes that look like an earlier one are left to quick scoring.
    Without speech or quick signals, every scene is analyzed."""
    if len(scenes) <= _SHORTLIST_MIN_SCENES:
        return None
    if not segment_scores and not visual_signal:
        return None
    quality = {int(k): v for k, v in (scene_quality or {}).items() if isinstance(v, dict)}
    match = match or {}
    signal = visual_signal or {}
    picked: set[int] = set()
    excluded: set[int] = set()
    ranked: list[tuple[float, int, float]] = []
    for scene in scenes:
        scene_id = int(scene["scene_id"])
        start, end = float(scene["start"]), float(scene["end"])
        duration = max(end - start, 0.01)
        if quality.get(scene_id, {}).get("black") or (match.get(scene_id) or {}).get("duplicate_of") is not None:
            excluded.add(scene_id)
            continue
        if not segment_scores:
            ranked.append((float(signal.get(scene_id, 0.0)), scene_id, duration))
            continue
        spoken = 0.0
        best = 0.0
        for item in segment_scores:
            overlap = min(end, float(item["end"])) - max(start, float(item["start"]))
            if overlap <= 0:
                continue
            spoken += overlap
            score = item.get("score") if isinstance(item.get("score"), dict) else {}
            best = max(best, float(score.get("importance") or 0) * 0.6 + float(score.get("information_density") or 0) * 0.4)
        if spoken / duration < _VISUAL_ONLY_SPEECH:
            picked.add(scene_id)
            continue
        interest = (match.get(scene_id) or {}).get("interest")
        if interest is not None:
            best = 0.75 * best + 0.25 * float(interest)
        ranked.append((best, scene_id, duration))
    budget = max(1.0, highlight_seconds) * _SHORTLIST_COVERAGE
    covered = 0.0
    for best, scene_id, duration in sorted(ranked, reverse=True):
        if covered >= budget and len(picked) >= _SHORTLIST_MIN_SCENES:
            break
        picked.add(scene_id)
        covered += duration
    if len(picked) >= len(scenes) * _SHORTLIST_ALL_SHARE:
        # Nearly everything qualifies: analyze every scene, except black and lookalike ones.
        return {int(scene["scene_id"]) for scene in scenes} - excluded if excluded else None
    return picked
