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


# What the vision model should rate higher for a "What matters most" choice. Priorities that are
# not listed (action, speech, ...) only change how the scores are weighted.
PRIORITY_VISUAL_HINTS = {
    "connection": (
        "Prioritize clearly visible affectionate or romantic interactions, meaningful gestures, "
        "embraces, kisses, and non-explicit intimate moments. Judge the action and interaction, "
        "not merely the presence of people, and do not infer details that are not visible."
    ),
    "people": (
        "Prioritize scenes where people's faces, expressions, and interactions are clearly visible; "
        "rate empty or faceless scenes lower."
    ),
    "scenery": (
        "Prioritize beautiful landscapes, cityscapes, wide views, sunsets, and well-composed shots "
        "of places; rate cluttered or close-up indoor shots lower."
    ),
    "fun": (
        "Prioritize people laughing, playing, joking, or celebrating, and visibly funny or joyful moments."
    ),
    "pets": "Prioritize scenes where pets or other animals are clearly visible, especially moving or playing.",
    "music": "Prioritize people singing, dancing, playing instruments, or performing on stage.",
    "sports": (
        "Prioritize sports and games in motion: shots, goals, tricks, races, and the reactions to them."
    ),
}

# Short picture descriptions for fast picture matching (SigLIP) per priority.
PRIORITY_PICTURE_TEXT = {
    "connection": "two people hugging, holding hands, or kissing",
    "people": "a close-up of people's faces and expressions",
    "scenery": "a beautiful landscape or scenic view",
    "fun": "people laughing, playing, or celebrating",
    "pets": "a pet or an animal",
    "music": "people singing, dancing, or playing music",
    "sports": "people playing a sport",
}

# A sentence about each newer style for the transcript prompt; older styles are named only.
STYLE_PROMPT_HINTS = {
    "cinematic": "prefer vivid, descriptive, emotional lines over explanations",
    "funny": "prefer jokes, witty lines, laughter, and surprising reactions",
    "story": "prefer lines that move the story forward and are understandable on their own",
    "trailer": "prefer short, striking, dramatic lines that make people want to watch",
    "calm": "prefer gentle, reflective lines; avoid shouting and hype",
}


def visual_focus_priorities(priorities: list[str] | None) -> list[str]:
    """The chosen priorities that change what the vision model looks for, in a stable order."""
    chosen = set(priorities or [])
    return [name for name in PRIORITY_VISUAL_HINTS if name in chosen]


def visual_analysis_focus(custom_focus: str, priorities: list[str] | None = None) -> str:
    focus = custom_focus.strip()
    for name in visual_focus_priorities(priorities):
        hint = PRIORITY_VISUAL_HINTS[name]
        focus = f"{focus}. {hint}" if focus else hint
    return focus


def picture_focus_texts(custom_focus: str, priorities: list[str] | None = None) -> list[str]:
    """What fast picture matching compares scenes with: the user's focus and the chosen
    priorities' picture descriptions."""
    texts = [custom_focus.strip()] if custom_focus.strip() else []
    texts += [PRIORITY_PICTURE_TEXT[name] for name in PRIORITY_PICTURE_TEXT if name in set(priorities or [])]
    return texts


def _style_label(style: str) -> str:
    hint = STYLE_PROMPT_HINTS.get(style)
    return f"{style} ({hint})" if hint else style


def visual_batch_schema(with_avoid: bool = False) -> dict:
    """The batch reply schema; with ``with_avoid`` each scene may also carry an "avoid" score."""
    if not with_avoid:
        return VISUAL_BATCH_SCHEMA
    item = {**VISUAL_SCENE_SCHEMA_FAST, "properties": {**VISUAL_SCENE_SCHEMA_FAST["properties"], "avoid": {"type": "number"}}}
    return {**VISUAL_BATCH_SCHEMA, "properties": {"scenes": {"type": "array", "items": item}}}


def visual_batch_prompt(scenes: list[dict], custom_focus: str = "", avoid: str = "") -> str:
    extra = f" Focus: {custom_focus}." if custom_focus else ""
    avoid_key = ',"avoid":0-1' if avoid else ""
    if avoid:
        extra += f' "avoid" is how clearly the scene shows any of: {avoid}. Use 0 when none is visible.'

    lines = "\n".join(
        f"{idx}. scene_id={scene['scene_id']} [{scene['start']:.1f}-{scene['end']:.1f}s]"
        for idx, scene in enumerate(scenes)
    )
    return f"""Analyze each scene from its frame image. Image order matches scene order.{extra}
Return compact JSON only:
{{"scenes":[{{"scene_id":n,"visual_importance":0-1,"action_strength":0-1,"information_density":0-1,"visual_change":0-1,"repetition_likelihood":0-1,"context_dependency":0-1{avoid_key}}}]}}
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


def transcript_batch_prompt_fast(style: str, custom_focus: str, segments: list[dict], avoid: str = "") -> str:
    extra = f" Focus: {custom_focus}." if custom_focus else ""
    if avoid:
        extra += f" Give low importance to: {avoid}."
    lines = "\n".join(
        f"{idx}. [{item['start']:.1f}-{item['end']:.1f}] {item['text']}"
        for idx, item in enumerate(segments)
    )
    return f"""Score highlight value for each segment (0.0-1.0). Style: {_style_label(style)}.{extra}
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
