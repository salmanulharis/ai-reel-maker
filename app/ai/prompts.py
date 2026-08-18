from __future__ import annotations

VISUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "visual_summary": {"type": "string"},
        "visual_importance": {"type": "number"},
        "action_strength": {"type": "number"},
        "information_density": {"type": "number"},
        "visual_change": {"type": "number"},
        "repetition_likelihood": {"type": "number"},
        "context_dependency": {"type": "number"},
        "has_person": {"type": "boolean"},
        "has_text": {"type": "boolean"},
        "appears_intro": {"type": "boolean"},
        "appears_ending": {"type": "boolean"},
        "surprising": {"type": "boolean"},
    },
    "required": [
        "visual_summary",
        "visual_importance",
        "action_strength",
        "information_density",
        "visual_change",
        "repetition_likelihood",
        "context_dependency",
    ],
}

SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "importance": {"type": "number"},
        "novelty": {"type": "number"},
        "information_density": {"type": "number"},
        "hook_strength": {"type": "number"},
        "repetition_score": {"type": "number"},
        "filler_score": {"type": "number"},
        "context_dependency": {"type": "number"},
        "emotional_intensity": {"type": "number"},
        "label": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["importance", "novelty", "information_density"],
}

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "integer"}},
        "rationale": {"type": "string"},
    },
    "required": ["order"],
}


def visual_prompt(custom_focus: str = "") -> str:
    extra = f"\nUser focus: {custom_focus}" if custom_focus else ""
    return f"""Analyze these representative frames from one video scene.
Return JSON only.
Determine: what is happening, visual importance, whether a person is speaking,
whether this is a product demo, whether there is on-screen text, whether the
action is meaningful, whether the scene looks repetitive, introductory, or an ending.
{extra}
Score every numeric field from 0.0 to 1.0."""


def transcript_batch_prompt(style: str, custom_focus: str, segments: list[dict]) -> str:
    extra = f"\nCustom instruction: {custom_focus}" if custom_focus else ""
    lines = "\n".join(
        f"{idx}. [{item['start']:.1f}-{item['end']:.1f}] {item['text']}"
        for idx, item in enumerate(segments)
    )
    return f"""Score each transcript segment for highlight value.
Style: {style}.{extra}
For each segment return importance, novelty, information_density, hook_strength,
repetition_score, filler_score, context_dependency, emotional_intensity, label, reason.
Labels should be human readable such as Strong hook, Key explanation, Unique information,
Important demonstration, Strong reaction, Conclusion, Filler, Introduction, Repeated idea.
Prefer complete thoughts over greetings and repetition.
Segments:
{lines}
Return JSON: {{"segments":[...in the same order...]}}."""


def order_prompt(clips: list[dict]) -> str:
    lines = "\n".join(
        f"{idx}. {c['start']:.1f}-{c['end']:.1f} {c.get('reason','')} :: {c.get('text','')[:180]}"
        for idx, c in enumerate(clips)
    )
    return f"""Recommend a natural viewing order for these highlight clips.
Prefer temporal coherence. A narrative arc of hook → context → key point → conclusion
is allowed only if it still feels natural.
Return JSON {{"order": [indices], "rationale": "..."}}.
Clips:
{lines}"""
