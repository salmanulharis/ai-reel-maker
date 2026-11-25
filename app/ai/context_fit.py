"""How many frames fit in one vision request, shared by the pipeline and the recommendations.

Both must use the same sums: a recommended batch size that does not fit its context window is
silently reduced when a job runs, so Settings would show values that are never used.
"""

from __future__ import annotations

# Upper bound on reply tokens per scene in a batched visual reply.
VISION_TOKENS_PER_SCENE = 80
# Extra reply tokens per scene when the vision model also describes it (prompt mode, at most
# 15 words).
VISION_CAPTION_TOKENS = 32
# Prompt tokens per picture. Ollama scales Qwen3-VL images up to about a megapixel, so a picture
# costs about 1,050 tokens whatever its resolution (measured with 256-512 px frames, and with a
# 2x2 grid of 512 px frames). This is why several scenes are tiled into one picture.
VISION_PROMPT_TOKENS_PER_IMAGE = 1100
# Most scenes per picture (a 3x3 grid; larger grids pass a megapixel and cost more tokens).
VISION_MAX_SCENES_PER_IMAGE = 9
# Share of a one-frame request spent reading the picture, the rest writing the reply (measured on
# the processor: 1,933 s reading and 484 s writing for 52 one-frame requests). Tiling divides
# only the reading part between the scenes of a picture.
VISION_IMAGE_SHARE = 0.8
# Instructions plus one "scene_id=... [start-end]" line per scene.
VISION_PROMPT_BASE_TOKENS = 200
VISION_PROMPT_TOKENS_PER_LINE = 16
VISION_CONTEXT_SAFETY = 0.9
# The default reply budget for vision requests (Settings.ollama_vision_num_predict).
VISION_DEFAULT_PREDICT = 256
_CONTEXT_STEP = 1024
_MIN_CONTEXT = 2048


def predict_budget(configured: int, items: int, per_item: int, num_ctx: int) -> int:
    """num_predict for a batched JSON reply: room for every item, but never more than half the
    context window so the prompt still fits. Never below the configured value."""
    wanted = max(configured, per_item * max(1, items) + 64)
    return min(wanted, max(configured, num_ctx // 2))


def fitting_vision_batch(
    configured: int,
    num_ctx: int,
    configured_predict: int,
    per_scene: int = VISION_TOKENS_PER_SCENE,
    scenes_per_image: int = 1,
) -> int:
    """The largest number of pictures per request, up to the configured size, whose pictures
    and reply fit the context. Each picture holds ``scenes_per_image`` scenes.

    A batch that overflows the context loses part of its prompt or gets its reply cut off.
    Smaller batches cost almost nothing extra: each image is read once either way."""
    limit = num_ctx * VISION_CONTEXT_SAFETY
    scenes = max(1, int(scenes_per_image))
    for size in range(max(1, configured), 0, -1):
        prompt = VISION_PROMPT_BASE_TOKENS + size * (
            VISION_PROMPT_TOKENS_PER_IMAGE + scenes * VISION_PROMPT_TOKENS_PER_LINE
        )
        if prompt + predict_budget(configured_predict, size * scenes, per_scene, num_ctx) <= limit:
            return size
    return 1


def fitting_scenes_per_image(
    configured: int, num_ctx: int, configured_predict: int, per_scene: int = VISION_TOKENS_PER_SCENE
) -> int:
    """The most scenes, up to the configured number, that one picture can hold while one picture
    and the replies for its scenes fit the context. At least 1 (one frame per picture)."""
    wanted = min(max(1, int(configured or 1)), VISION_MAX_SCENES_PER_IMAGE)
    limit = num_ctx * VISION_CONTEXT_SAFETY
    for scenes in range(wanted, 1, -1):
        prompt = VISION_PROMPT_BASE_TOKENS + VISION_PROMPT_TOKENS_PER_IMAGE + scenes * VISION_PROMPT_TOKENS_PER_LINE
        if prompt + predict_budget(configured_predict, scenes, per_scene, num_ctx) <= limit:
            return scenes
    return 1


def vision_scene_cost(scenes_per_image: int) -> float:
    """Time for one scene relative to a one-frame request, when ``scenes_per_image`` scenes
    share a picture: the reading is shared, each scene still has its own reply."""
    scenes = max(1, int(scenes_per_image or 1))
    return VISION_IMAGE_SHARE / scenes + (1.0 - VISION_IMAGE_SHARE)


def vision_ctx_for_batch(
    batch: int,
    configured_predict: int = VISION_DEFAULT_PREDICT,
    per_scene: int = VISION_TOKENS_PER_SCENE,
    scenes_per_image: int = 1,
) -> int:
    """The smallest context window (in steps of 1,024 tokens) that holds ``batch`` pictures of
    ``scenes_per_image`` scenes each."""
    batch = max(1, int(batch))
    num_ctx = _MIN_CONTEXT
    while (
        fitting_vision_batch(batch, num_ctx, configured_predict, per_scene, scenes_per_image) < batch
        and num_ctx < 65536
    ):
        num_ctx += _CONTEXT_STEP
    return num_ctx
