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
# Prompt tokens per frame. Ollama scales Qwen3-VL images up to a minimum size, so a frame costs
# about 1,050 tokens whatever its resolution (measured with 256-512 px frames).
VISION_PROMPT_TOKENS_PER_IMAGE = 1100
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
    configured: int, num_ctx: int, configured_predict: int, per_scene: int = VISION_TOKENS_PER_SCENE
) -> int:
    """The largest batch, up to the configured size, whose images and reply fit the context.

    A batch that overflows the context loses part of its prompt or gets its reply cut off.
    Smaller batches cost almost nothing extra: each image is read once either way."""
    limit = num_ctx * VISION_CONTEXT_SAFETY
    for size in range(max(1, configured), 0, -1):
        prompt = VISION_PROMPT_BASE_TOKENS + size * (VISION_PROMPT_TOKENS_PER_IMAGE + VISION_PROMPT_TOKENS_PER_LINE)
        if prompt + predict_budget(configured_predict, size, per_scene, num_ctx) <= limit:
            return size
    return 1


def vision_ctx_for_batch(batch: int, configured_predict: int = VISION_DEFAULT_PREDICT) -> int:
    """The smallest context window (in steps of 1,024 tokens) that holds ``batch`` frames."""
    batch = max(1, int(batch))
    num_ctx = _MIN_CONTEXT
    while fitting_vision_batch(batch, num_ctx, configured_predict) < batch and num_ctx < 65536:
        num_ctx += _CONTEXT_STEP
    return num_ctx
