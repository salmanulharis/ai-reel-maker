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

VISUAL_SCENE_SCHEMA_FAST = {
    "type": "object",
    "properties": {
        "scene_id": {"type": "integer"},
        "visual_importance": {"type": "number"},
        "action_strength": {"type": "number"},
        "information_density": {"type": "number"},
        "visual_change": {"type": "number"},
        "repetition_likelihood": {"type": "number"},
        "context_dependency": {"type": "number"},
    },
    "required": ["scene_id", "visual_importance", "action_strength", "information_density"],
}

VISUAL_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": VISUAL_SCENE_SCHEMA_FAST,
        }
    },
    "required": ["scenes"],
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

TRANSCRIPT_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": SEGMENT_SCHEMA,
        }
    },
    "required": ["segments"],
}

TRANSCRIPT_BATCH_SCHEMA_FAST = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "importance": {"type": "number"},
                    "novelty": {"type": "number"},
                    "information_density": {"type": "number"},
                },
                "required": ["i", "importance", "novelty", "information_density"],
            },
        }
    },
    "required": ["segments"],
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


def visual_analysis_focus(custom_focus: str, priorities: list[str] | None = None) -> str:
    focus = custom_focus.strip()
    if "connection" in (priorities or []):
        connection = (
            "Prioritize clearly visible affectionate or romantic interactions, meaningful gestures, "
            "embraces, kisses, and non-explicit intimate moments. Judge the action and interaction, "
            "not merely the presence of people, and do not infer details that are not visible."
        )
        focus = f"{focus}. {connection}" if focus else connection
    return focus


def visual_batch_prompt(scenes: list[dict], custom_focus: str = "") -> str:
    extra = f" Focus: {custom_focus}." if custom_focus else ""
    lines = "\n".join(
        f"{idx}. scene_id={scene['scene_id']} [{scene['start']:.1f}-{scene['end']:.1f}s]"
        for idx, scene in enumerate(scenes)
    )
    return f"""Analyze each scene from its frame image. Image order matches scene order.{extra}
Return compact JSON only:
{{"scenes":[{{"scene_id":n,"visual_importance":0-1,"action_strength":0-1,"information_density":0-1,"visual_change":0-1,"repetition_likelihood":0-1,"context_dependency":0-1}}]}}
Use the same scene_id values. Score 0.0-1.0. Penalize intros, endings, repetition, and empty frames.
Scenes:
{lines}"""


def transcript_batch_prompt(style: str, custom_focus: str, segments: list[dict]) -> str:
    extra = f" Focus: {custom_focus}." if custom_focus else ""
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


def transcript_batch_prompt_fast(style: str, custom_focus: str, segments: list[dict]) -> str:
    extra = f" Focus: {custom_focus}." if custom_focus else ""
    lines = "\n".join(
        f"{idx}. [{item['start']:.1f}-{item['end']:.1f}] {item['text']}"
        for idx, item in enumerate(segments)
    )
    return f"""Score highlight value for each segment (0.0-1.0). Style: {style}.{extra}
Return compact JSON only: {{"segments":[{{"i":n,"importance":n,"novelty":n,"information_density":n}}]}}
One entry per segment, in the same order. "i" is the segment number shown before it. Penalize filler, greetings, and repetition.
Segments:
{lines}"""


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
