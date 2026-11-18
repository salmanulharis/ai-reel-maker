"""
Suggested text for posting the highlight: three titles, a short description and hashtags.

The local text model writes them from what is said in the kept clips (or, without speech, from
the vision model's scene descriptions). When the model is not available or replies badly, a
simple suggestion is made from the most used words, so the button always gives a result.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from app.models.schemas import HighlightConfig
from app.video.captions import clip_cues, load_transcript
from app.video.sources import load_sources

RESULT_FILE = "post_text.json"
MAX_SPOKEN_CHARS = 4000
MAX_SCENES = 30
MAX_TITLES = 3
MAX_TITLE = 100
MAX_DESCRIPTION = 500
MAX_HASHTAGS = 8

SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["titles", "description", "hashtags"],
}

_STOPWORDS = set(
    """about above after again against also although always among another anyone anything around
    because been before being below between both cannot could didn does doing done down during each
    either enough even every from further gonna have having here hers herself himself however into
    itself just know like made make many maybe might more most much must myself need never only other
    others ourselves over really right same should since some something still such than that their
    theirs them themselves then there these they thing things this those though through thus together
    under until very want well were what whatever when where whether which while with within without
    would yeah your yours yourself okay going think actually basically""".split()
)


class PostTextError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _scene_summaries(folder: Path, start: float, end: float) -> list[str]:
    try:
        scenes = json.loads((folder / "scenes.json").read_text(encoding="utf-8"))
        visual = json.loads((folder / "visual_analysis.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    found = []
    for scene in scenes if isinstance(scenes, list) else []:
        try:
            s_start, s_end = float(scene["start"]), float(scene["end"])
            entry = visual.get(str(scene["scene_id"])) if isinstance(visual, dict) else None
        except (KeyError, TypeError, ValueError):
            continue
        summary = " ".join(str((entry or {}).get("visual_summary") or "").split())
        if summary and s_end > start and s_start < end:
            found.append(summary)
    return found


def gather_material(job_dir: Path, clips: list[dict], config: HighlightConfig) -> tuple[list[str], list[str]]:
    """What is said in each kept clip, and descriptions of what is seen, in highlight order."""
    sources = load_sources(job_dir)
    multi = len(sources) > 1
    known = {str(s.get("id") or "v0") for s in sources}
    first = str(sources[0].get("id") or "v0") if sources else "v0"
    spoken: list[str] = []
    seen: list[str] = []
    transcripts: dict[str, dict] = {}
    for clip in clips:
        if clip.get("kept") is False:
            continue
        try:
            start, end = float(clip["start"]), float(clip["end"])
        except (KeyError, TypeError, ValueError):
            continue
        source_id = str(clip.get("source_id") or "")
        source_id = source_id if source_id in known else first
        if config.use_speech:
            if source_id not in transcripts:
                transcripts[source_id] = load_transcript(job_dir, source_id, multi)
            text = " ".join(cue.text for cue in clip_cues(transcripts[source_id], start, end, "plain"))
            text = text or " ".join(str(clip.get("transcript") or "").split())
            if text:
                spoken.append(text)
        folder = job_dir / "work" / source_id if multi else job_dir
        for summary in _scene_summaries(folder, start, end):
            if summary not in seen:
                seen.append(summary)
    return spoken, seen[:MAX_SCENES]


def _clean_title(text) -> str:
    title = " ".join(str(text or "").split()).strip("\"'“”‘’ ")
    return title[:MAX_TITLE].rstrip()


def _clean_hashtag(text) -> str:
    return re.sub(r"[^\w]", "", str(text or "").lower().lstrip("#"), flags=re.UNICODE)[:40]


def clean(data: dict) -> dict:
    """Keep only usable, short, unique items from a model reply."""
    titles: list[str] = []
    for item in data.get("titles") or []:
        title = _clean_title(item)
        if title and title.lower() not in {t.lower() for t in titles}:
            titles.append(title)
    hashtags: list[str] = []
    for item in data.get("hashtags") or []:
        tag = _clean_hashtag(item)
        if tag and not tag.isdigit() and tag not in hashtags:
            hashtags.append(tag)
    description = " ".join(str(data.get("description") or "").split())[:MAX_DESCRIPTION].rstrip()
    return {"titles": titles[:MAX_TITLES], "description": description, "hashtags": hashtags[:MAX_HASHTAGS]}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]


def simple_suggestion(spoken: list[str], seen: list[str], name: str) -> dict:
    """Titles, description and hashtags without a model: the first things said, and the most
    used meaningful words."""
    stem = re.sub(r"[_\-.]+", " ", Path(name or "video").stem).strip() or "Video"
    stem = stem[:1].upper() + stem[1:]
    text = " ".join(spoken) or " ".join(seen)
    sentences = _sentences(text)
    titles = [f"{stem}: highlights"]
    for sentence in sentences[:2]:
        titles.append(sentence if len(sentence) <= 70 else sentence[:67].rsplit(" ", 1)[0] + "…")
    description = " ".join(sentences[:2])[:240] or f"The best moments from {stem}."
    words = [w for w in re.findall(r"[^\W\d_]{4,}", text.lower(), flags=re.UNICODE) if w not in _STOPWORDS]
    hashtags = [word for word, _ in Counter(words).most_common(6)] + ["highlights"]
    return clean({"titles": titles, "description": description, "hashtags": hashtags})


def _prompt(spoken: list[str], seen: list[str], config: HighlightConfig, name: str) -> str:
    said = ""
    for index, text in enumerate(spoken, 1):
        line = f"{index}. {text}\n"
        if len(said) + len(line) > MAX_SPOKEN_CHARS:
            break
        said += line
    scenes = "\n".join(f"- {s}" for s in seen)
    focus = f"\nThe person asked to focus on: {config.custom_focus.strip()}" if config.custom_focus.strip() else ""
    return (
        "Write text for posting a short highlight video on social media.\n"
        "Use only what is in the material below; do not invent names, places, numbers or facts.\n"
        "Write in the same language as the spoken words (English if there are none).\n"
        f"Highlight style: {config.style}.{focus}\n"
        f"Video file name: {name}\n\n"
        f"What is said in the highlight, clip by clip:\n{said or '(no speech)'}\n"
        f"What is seen:\n{scenes or '(no descriptions)'}\n\n"
        "Reply with JSON only: "
        '{"titles": [three different catchy titles, each under 70 characters], '
        '"description": "two or three sentences, under 300 characters", '
        '"hashtags": [five to eight hashtags without the # sign, lowercase, no spaces]}'
    )


def _result_path(job_dir: Path, short: int | None) -> Path:
    name = RESULT_FILE if short is None else f"post_text_short_{short:02d}.json"
    return job_dir / "outputs" / name


def suggest(job, job_dir: Path, clips: list[dict], settings, client=None, short: int | None = None) -> dict:
    """Suggestions for posting this highlight (or short number `short`); saved with the job."""
    if short is not None:
        clips = [c for c in clips if int(c.get("short_index") or 0) == short - 1]
    if not any(c.get("kept") is not False for c in clips):
        raise PostTextError("no clips to describe")
    config = HighlightConfig.model_validate(json.loads(job.config_json or "{}"))
    spoken, seen = gather_material(job_dir, clips, config)
    name = Path(job.original_filename or "video").name
    result = None
    note = ""
    if spoken or seen:
        try:
            if client is None:
                from app.services.ollama_runtime import get_ollama_client

                client = get_ollama_client(settings)
            options = client.role_options("text")
            options["num_predict"] = 500
            reply = client.chat(settings.ollama_text_model, _prompt(spoken, seen, config, name), schema=SCHEMA, options=options)
            cleaned = clean(reply if isinstance(reply, dict) else {})
            if cleaned["titles"] and cleaned["description"]:
                result = {**cleaned, "source": "ai"}
            else:
                note = "The AI reply was not usable."
        except Exception as exc:  # noqa: BLE001 - a simple suggestion is always possible
            note = f"The AI model could not be used ({str(exc)[:160]})."
    else:
        note = "There is no speech or scene description to work from."
    if result is None:
        result = {**simple_suggestion(spoken, seen, name), "source": "simple", "note": note}
    path = _result_path(job_dir, short)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def saved(job_dir: Path, short: int | None = None) -> dict | None:
    path = _result_path(job_dir, short)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
