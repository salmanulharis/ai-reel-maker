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
        # Nothing reads the model's value (the motion measurement is used), but it stays: removing
        # it from the reply changed the other scores and captions on a real job.
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


def visual_batch_schema(with_avoid: bool = False, with_caption: bool = False) -> dict:
    """The batch reply schema; with ``with_avoid`` each scene may also carry an "avoid" score,
    with ``with_caption`` a short description of what it shows."""
    if not with_avoid and not with_caption:
        return VISUAL_BATCH_SCHEMA
    properties = dict(VISUAL_SCENE_SCHEMA_FAST["properties"])
    required = list(VISUAL_SCENE_SCHEMA_FAST["required"])
    if with_avoid:
        properties["avoid"] = {"type": "number", "minimum": 0, "maximum": 1}
    if with_caption:
        properties["caption"] = {"type": "string"}
        required.append("caption")
    item = {**VISUAL_SCENE_SCHEMA_FAST, "properties": properties, "required": required}
    return {**VISUAL_BATCH_SCHEMA, "properties": {"scenes": {"type": "array", "items": item}}}


def visual_batch_prompt(
    scenes: list[dict], custom_focus: str = "", avoid: str = "", describe: bool = False, grid: bool = False
) -> str:
    """With ``grid`` the images are grids of scene frames, each frame labelled with its scene id
    (see app.analysis.frame_grid); otherwise there is one frame per scene, in scene order."""
    extra = f" Focus: {custom_focus}." if custom_focus else ""
    avoid_key = ',"avoid":0-1' if avoid else ""
    if avoid:
        extra += f' "avoid" is how clearly the scene shows any of: {avoid}. Use 0 when none is visible.'
    if describe:
        avoid_key += ',"caption":"..."'
        extra += ' "caption" says plainly what is visible and happening, in at most 15 words (who, what, where).'

    lines = "\n".join(
        f"{idx}. scene_id={scene['scene_id']} [{scene['start']:.1f}-{scene['end']:.1f}s]"
        for idx, scene in enumerate(scenes)
    )
    if grid:
        intro = (
            "Each image is a grid of scene frames. The label above each frame names its scene "
            "(\"Scene 12\" means scene 12). Analyze each scene from its own frame only."
        )
    else:
        intro = "Analyze each scene from its frame image. Image order matches scene order."
    return f"""{intro}{extra}
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


# Prompt mode: understanding the user's description

PROMPT_STYLES = ("auto", "informative", "exciting", "social", "educational", "cinematic", "funny", "story", "trailer", "calm")
PROMPT_PRIORITIES = ("action", "speech", "visual", "information", "connection", "people", "scenery", "fun", "pets", "music", "sports")
PROMPT_LEAVE_OUT = ("filler", "sponsor", "swearing", "talking_heads", "dark", "text", "screens", "empty", "intro_outro", "silence")

_PHRASES = {"type": "array", "items": {"type": "string"}, "maxItems": 8}
_TIME_RANGES = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"start": {"type": "number"}, "end": {"type": "number"}},
        "required": ["start", "end"],
    },
    "maxItems": 8,
}

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "min_seconds": {"type": "number"},
        "max_seconds": {"type": "number"},
        "length_auto": {"type": "boolean"},
        "format": {"type": "string", "enum": ["original", "16:9", "9:16", "1:1", "4:5"]},
        "subtitles": {"type": "string", "enum": ["none", "burned", "srt"]},
        "style": {"type": "string", "enum": list(PROMPT_STYLES)},
        "shorts": {"type": "integer"},
        "priorities": {"type": "array", "items": {"type": "string", "enum": list(PROMPT_PRIORITIES)}},
        "leave_out": {"type": "array", "items": {"type": "string", "enum": list(PROMPT_LEAVE_OUT)}},
        "order": {"type": "string", "enum": ["time", "story", "hook"]},
        "transition": {"type": "string", "enum": ["cut", "fade", "auto"]},
        "tighten": {"type": "string", "enum": ["off", "pauses", "fillers"]},
        "keep": _PHRASES,
        "remove": _PHRASES,
        "moments": _PHRASES,
        "visual_cues": _PHRASES,
        "audio_cues": _PHRASES,
        "keep_times": _TIME_RANGES,
        "remove_times": _TIME_RANGES,
    },
    "required": ["summary", "keep", "remove", "moments", "visual_cues", "audio_cues"],
}


def intent_prompt(request: str, duration: float, video_count: int) -> str:
    length = f"{duration:.0f} seconds" if duration > 0 else "unknown length"
    videos = f"{video_count} videos, {length} in total" if video_count > 1 else f"one video of {length}"
    return f"""A person wants a short highlight cut from {videos}. Turn their request into settings and editing requirements.
Request: <<<{request}>>>

Rules:
- "summary": one sentence saying what you understood.
- Add a setting ONLY when the request asks for it or clearly implies it. Leave every other setting out.
  Examples: TikTok, Reels, Shorts, vertical -> "format":"9:16"; captions or subtitles (only when asked for) -> "subtitles":"burned"; "about a minute" -> "min_seconds":50,"max_seconds":70; "as long as needed" -> "length_auto":true; "5 clips" or "5 shorts" -> "shorts":5.
- "order": "story" to tell it as a story, "hook" to open with the strongest moment, "time" to keep the original order.
- "priorities" and "leave_out" use only the listed values.
- "keep", "remove", "moments": short phrases about content to keep, to cut, and specific moments that must be in it.
- "visual_cues": short descriptions of what wanted moments LOOK like (for example "a dog catching a ball"). "audio_cues": what they SOUND like (for example "crowd cheering").
- "keep_times" / "remove_times": only times the request states, in seconds from the start of the video.
- Do not invent content the request does not mention. Empty lists are fine.
Return JSON only."""


# Prompt mode: the AI editor

DIRECTOR_ROLES = ("hook", "context", "key", "highlight", "reaction", "ending")

DIRECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "understanding": {"type": "string"},
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "role": {"type": "string", "enum": list(DIRECTOR_ROLES)},
                    "why": {"type": "string"},
                },
                "required": ["id", "role", "why"],
            },
        },
        "order": {"type": "string", "enum": ["time", "story"]},
        "left_out": {"type": "string"},
    },
    "required": ["understanding", "clips", "order"],
}

_REQUIREMENT_LABELS = (
    ("keep", "Keep"),
    ("remove", "Cut"),
    ("moments", "Must include"),
    ("visual_cues", "Looks like"),
    ("audio_cues", "Sounds like"),
)


def _request_block(request: str, requirements: dict, notes: list[str], previous: list[str] | None = None) -> str:
    wants = [
        f"{label}: {'; '.join(requirements[key])}" for key, label in _REQUIREMENT_LABELS if requirements.get(key)
    ]
    asked = "\n".join(wants) or "(nothing specific)"
    changes = ""
    if notes:
        changes = "\nChanges asked for after the last plan (newest last; these win over the request):\n" + "\n".join(
            f"- {note}" for note in notes
        )
    before = f"\nThe last plan used, in order: {', '.join(previous)}" if previous else ""
    return f"""Request: <<<{request}>>>
Requirements:
{asked}{changes}{before}"""


def director_prompt(
    request: str,
    requirements: dict,
    notes: list[str],
    lines: list[str],
    target: str,
    pick_up_to: int,
    previous: list[str] | None = None,
    final: bool = True,
    outline: list[str] | None = None,
) -> str:
    if final:
        task = f"Choose the moments for the final video ({target}). Pick up to {pick_up_to}."
    else:
        task = (
            f"This is one part of a longer video. Pick up to {pick_up_to} moments worth considering "
            f"for the final video ({target})."
        )
    listing = "\n".join(lines)
    plan = ""
    order_rule = (
        '- "order": "story" only when another order tells it better than the original timeline; '
        "then list clips in viewing order."
    )
    if outline:
        plan = "\nThe story outline (beats in viewing order, each from a chapter c1, c2, ...):\n" + "\n".join(outline) + "\n"
        order_rule = (
            "- Follow the outline: pick moments for every beat from its chapter (shown after each id), "
            'and list the clips in the outline\'s order. Set "order" to "story".\n'
            "- When a moment needs the one just before it to be understood, pick that one too."
        )
    return f"""You are a video editor. {task}
{_request_block(request, requirements, notes, previous if final else None)}
{plan}
Each moment: id [start-end, length] score | says: speech | sees: picture | hears: sounds.
"score" is the automatic highlight value (0-1); use it when the request does not decide.
Rules:
- Follow the request and requirements first. Never pick a moment that matches "Cut".
- The video must tell one clear story: a hook that opens it, the context a viewer needs, the key moments, and a clear ending.
- Pick moments that are complete and understandable; avoid cutting mid-sentence.
- Do not pick two moments with the same content.
- "role" is the moment's part in the video: one "hook" (first) and one "ending" (last). "why" says in a few words why it is in (refer to the request).
{order_rule}
- "understanding": one sentence about what the final video shows. "left_out": what you cut, briefly.
Moments:
{listing}
Return JSON only, using ids from the list."""


# Prompt mode: the AI editor's outline (the story first, then the moments)

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "understanding": {"type": "string"},
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter": {"type": "string"},
                    "role": {"type": "string", "enum": list(DIRECTOR_ROLES)},
                    "purpose": {"type": "string"},
                    "seconds": {"type": "number"},
                },
                "required": ["chapter", "role", "purpose", "seconds"],
            },
        },
    },
    "required": ["understanding", "beats"],
}


def outline_prompt(
    request: str,
    requirements: dict,
    notes: list[str],
    chapters: list[str],
    target: str,
    max_beats: int,
    previous: list[str] | None = None,
) -> str:
    listing = "\n".join(chapters)
    return f"""You are a video editor planning the story of a short video cut from a longer one ({target}).
{_request_block(request, requirements, notes, previous)}

The longer video in chapters, in time order. Each chapter: id [start-end] | says: what is said | sees: what is shown.
{listing}

Plan the story before choosing any moment. Rules:
- Follow the request and requirements first. Leave out chapters that do not serve it.
- Write 3 to {max_beats} beats in viewing order. Each beat uses one chapter; a chapter may serve more than one beat.
- The first beat is the "hook" (what makes people keep watching); the last is the "ending" (a clear conclusion, result or payoff). Between them: "context" a viewer needs to follow, then "key" moments and "highlight" or "reaction" moments.
- Keep the original order unless another order clearly tells the story better.
- "purpose": a few words on what the beat shows. "seconds": how long it should run; together about the length asked for.
- "understanding": one sentence about what the final video shows.
Return JSON only, using chapter ids from the list."""


# Prompt mode: the AI editor checks its own cut

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "remove": {"type": "array", "items": {"type": "string"}},
        "add": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "before": {"type": "string"}, "why": {"type": "string"}},
                "required": ["id", "before"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["remove", "add", "notes"],
}


def review_prompt(request: str, requirements: dict, notes: list[str], cut: list[str], context: list[str]) -> str:
    plan = "\n".join(cut)
    extra = "\n".join(context) or "(none)"
    return f"""You are a video editor checking a cut before it is made.
{_request_block(request, requirements, notes)}

The cut, in viewing order (id and role | says | sees | hears):
{plan}

Moments just before some of them in the original video, that could be added:
{extra}

Watch the cut in your head. Does it tell one clear story that fits the request, from the hook to the ending?
- "remove": ids in the cut that repeat another, confuse the story or do not fit the request (few or none).
- "add": a moment from the second list to put right before the moment "before", only when that moment cannot be understood without it.
- "notes": one sentence on the story.
Return JSON only, using ids from the lists."""
