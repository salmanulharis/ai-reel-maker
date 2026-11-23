from __future__ import annotations

from typing import Any

from app.ai.context_fit import vision_ctx_for_batch

GOALS = ("quality", "balanced", "fast")
DEFAULT_GOAL = "balanced"
# Memory sizes computers are sold with. Windows reports a little less than the label (15.8 GB
# for 16 GB, about 14.8 GB when built-in graphics reserve a share), so a reading just under a
# size counts as that size.
_RAM_SIZES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
_RAM_TOLERANCE = 0.9
# Approximate memory a loaded Whisper model uses.
WHISPER_MEMORY_GB = {
    "tiny": 0.4,
    "base": 0.5,
    "small": 1.0,
    "medium": 2.6,
    "large-v3-turbo": 2.6,
    "distil-large-v3": 2.4,
    "large-v3": 4.5,
}
# The share of a model that must sit in graphics memory before a split between the graphics
# card and the processor is worth choosing (Best quality only).
_SPLIT_MIN_FRACTION = 0.6


def recommended_threads(count: int | None) -> int:
    if not count or count < 1:
        return 4
    reserve = 1 if count > 2 else 0
    return max(1, min(8, int(count) - reserve))


# Pinned tag: the bare qwen3:4b alias now points to the thinking-only 2507 build.
TEXT_4B = "qwen3:4b-instruct-2507-q4_K_M"

VISION_FIT_GB = {
    "qwen3-vl:30b-a3b-instruct": 18.0,
    "qwen3-vl:8b-instruct": 7.0,
    "qwen3-vl:4b-instruct": 4.0,
    "qwen3-vl:2b-instruct": 2.5,
}

VISION_MODELS: list[dict[str, Any]] = [
    {
        "name": "qwen3-vl:2b-instruct",
        "kind": "vision",
        "disk_gb": 1.9,
        "min_vram_gb": 3,
        "min_ram_gb": 8,
        "summary": "Smallest vision model for slides and scenes.",
        "speed": "Fastest option. Weakest on small on-screen text.",
    },
    {
        "name": "qwen3-vl:4b-instruct",
        "kind": "vision",
        "disk_gb": 3.3,
        "min_vram_gb": 6,
        "min_ram_gb": 16,
        "summary": "Balanced vision model for demos, lectures, and UI.",
        "speed": "Comfortable on a 6 GB GPU. Slow on CPU.",
    },
    {
        "name": "qwen3-vl:8b-instruct",
        "kind": "vision",
        "disk_gb": 6.1,
        "min_vram_gb": 8,
        "min_ram_gb": 16,
        "summary": "Best quality for on-screen text, slides, and product UI.",
        "speed": "Comfortable when about 8 GB of VRAM is free. Slow on CPU.",
    },
    {
        "name": "qwen3-vl:30b-a3b-instruct",
        "kind": "vision",
        "disk_gb": 20.0,
        "min_vram_gb": 24,
        "min_ram_gb": 32,
        "summary": "Highest vision quality. For large GPUs only.",
        "speed": "Needs about 20 GB of VRAM. A large amount of system RAM does not make this a CPU default.",
    },
]

TEXT_MODELS: list[dict[str, Any]] = [
    {
        "name": TEXT_4B,
        "kind": "text",
        "disk_gb": 2.5,
        "min_vram_gb": 4,
        "min_ram_gb": 8,
        "summary": "Scores transcript moments for importance, hooks, and filler.",
        "speed": "Practical on CPU and small GPUs.",
    },
    {
        "name": "qwen3:8b",
        "kind": "text",
        "disk_gb": 5.2,
        "min_vram_gb": 8,
        "min_ram_gb": 16,
        "summary": "Clearer transcript scoring for lectures and interviews.",
        "speed": "Comfortable on an 8 GB GPU. Noticeably slower on CPU.",
    },
    {
        "name": "qwen3:30b",
        "kind": "text",
        "disk_gb": 18.0,
        "min_vram_gb": 20,
        "min_ram_gb": 32,
        "summary": "Largest transcript model. Optional on big GPUs.",
        "speed": "Needs about 20 GB of VRAM and should not stay loaded beside a 30B vision model.",
    },
]

WHISPER_MODELS: list[dict[str, Any]] = [
    {
        "name": "tiny",
        "kind": "whisper",
        "disk_gb": 0.08,
        "min_vram_gb": 1,
        "min_ram_gb": 4,
        "summary": "Fastest speech-to-text.",
        "speed": "Lowest accuracy. Suited to short or quiet CPU machines.",
    },
    {
        "name": "base",
        "kind": "whisper",
        "disk_gb": 0.15,
        "min_vram_gb": 1,
        "min_ram_gb": 8,
        "summary": "Default transcription on CPU.",
        "speed": "A practical balance of speed and accuracy without a GPU.",
    },
    {
        "name": "small",
        "kind": "whisper",
        "disk_gb": 0.5,
        "min_vram_gb": 2,
        "min_ram_gb": 8,
        "summary": "Better accuracy for interviews and tutorials.",
        "speed": "Still practical on CPU. Light on an NVIDIA GPU.",
    },
    {
        "name": "medium",
        "kind": "whisper",
        "disk_gb": 1.5,
        "min_vram_gb": 4,
        "min_ram_gb": 16,
        "summary": "High transcription accuracy.",
        "speed": "Heavy on CPU. Prefer an NVIDIA GPU.",
    },
    {
        "name": "large-v3-turbo",
        "kind": "whisper",
        "disk_gb": 1.6,
        "min_vram_gb": 3,
        "min_ram_gb": 8,
        "summary": "Close to large-v3 accuracy in all languages, several times faster.",
        "speed": "Best on NVIDIA GPUs when a couple of GB stay free. Slow on CPU.",
    },
    {
        "name": "distil-large-v3",
        "kind": "whisper",
        "disk_gb": 1.5,
        "min_vram_gb": 3,
        "min_ram_gb": 8,
        "summary": "Fast large-model accuracy. English only.",
        "speed": "Best on NVIDIA GPUs when a couple of GB stay free. Use it only for English videos.",
    },
    {
        "name": "large-v3",
        "kind": "whisper",
        "disk_gb": 3.1,
        "min_vram_gb": 6,
        "min_ram_gb": 16,
        "summary": "Highest transcription accuracy.",
        "speed": "Uses the most memory and competes with vision models for VRAM.",
    },
]

CATALOG: dict[str, list[dict[str, Any]]] = {
    "vision": VISION_MODELS,
    "text": TEXT_MODELS,
    "whisper": WHISPER_MODELS,
}


def catalog_entry(name: str) -> dict[str, Any] | None:
    for items in CATALOG.values():
        for item in items:
            if item["name"] == name:
                return item
    return None


def nominal_ram_gb(ram_gb: float | None) -> float:
    """Installed memory as the size on the label, for comparing against requirements."""
    ram = float(ram_gb or 0)
    for size in _RAM_SIZES:
        if size * _RAM_TOLERANCE <= ram < size:
            return float(size)
    return ram


def model_memory_gb(name: str | None) -> float:
    """Approximate memory an Ollama model needs once loaded (weights plus working memory)."""
    entry = catalog_entry(str(name or ""))
    if not entry or entry.get("kind") == "whisper":
        return 0.0
    return round(float(entry.get("disk_gb") or 0) * 1.15 + 0.6, 1)


def whisper_memory_gb(name: str | None) -> float:
    return WHISPER_MEMORY_GB.get(str(name or ""), 1.0)


def next_larger(name: str | None) -> str | None:
    """The next model up in the same family, or None for the largest or an unknown model."""
    for items in (VISION_MODELS, TEXT_MODELS):
        names = [item["name"] for item in items]
        if name in names:
            position = names.index(name)
            return names[position + 1] if position + 1 < len(names) else None
    return None


def normalize_goal(goal: str | None) -> str:
    value = str(goal or "").strip().lower()
    return value if value in GOALS else DEFAULT_GOAL


def recommend_setup(
    vram_gb: float | None,
    ram_gb: float | None,
    free_vram_gb: float | None = None,
    cpu_threads: int | None = None,
    *,
    goal: str = DEFAULT_GOAL,
    whisper_cuda: bool | None = None,
) -> dict[str, Any]:
    """Pick models and speed settings for this computer.

    ``goal`` chooses between the best quality that fits, a balance, and the fastest setup.
    Models come from GPU memory: system RAM never selects 30B, and selects 8B only for Best
    quality. ``whisper_cuda`` is False when speech to text cannot use the graphics card
    (None: not checked)."""
    total = float(vram_gb or 0)
    free = float(free_vram_gb) if free_vram_gb is not None else total
    budget = free if free > 0 else total
    threads = recommended_threads(cpu_threads)
    ram = nominal_ram_gb(ram_gb)
    goal = normalize_goal(goal)
    if total < 2 and budget < 2:
        setup = _cpu_setup(ram, threads, goal)
    else:
        setup = _gpu_setup(total, budget, threads, goal, ram)
    if whisper_cuda is False and setup["whisper_device"] == "cuda":
        setup["whisper_device"] = "cpu"
        if setup["whisper_model"] not in {"tiny", "base", "small"}:
            setup["whisper_model"] = "small"
        setup["notes"].append(
            "Speech to text runs on the processor: its graphics card support (CUDA 12 with cuDNN) was not found."
        )
    setup.update(_director_setup(setup, ram, budget, goal))
    setup["goal"] = goal
    return setup


def _director_setup(setup: dict[str, Any], ram: float, budget: float, goal: str) -> dict[str, Any]:
    """The AI editor (prompt mode) reads every candidate moment at once, so it gets a larger
    context window than transcript scoring. Reasoning first improves its choices but costs
    thousands of reply tokens: worth it on a graphics card with a hybrid model, too slow on the
    processor, and not possible with the 4B instruct model."""
    on_gpu = setup.get("inference_mode") == "gpu"
    if on_gpu:
        num_ctx = 32768 if budget >= 16 else 16384
    else:
        num_ctx = 16384 if ram >= 16 else 8192
    think = on_gpu and goal != "fast" and setup.get("text_model") != TEXT_4B
    return {"director_num_ctx": num_ctx, "director_think": think}


def _cpu_setup(ram: float, threads: int, goal: str) -> dict[str, Any]:
    low = ram < 16
    vision = "qwen3-vl:2b-instruct" if low else "qwen3-vl:4b-instruct"
    text = TEXT_4B
    whisper = "tiny" if ram < 8 else "base"
    notes: list[str] = []
    if low:
        notes.append(
            "CPU-only. Highlights still run. Vision is slower, and you can turn it off if you want a faster pass."
        )
        alternatives = {
            "vision": ["qwen3-vl:4b-instruct"] if ram >= 12 else [],
            "text": [],
            "whisper": ["base"] if whisper == "tiny" else ["small"],
        }
    else:
        notes.append(
            "CPU-only. The 4B vision model is the default. Larger models stay available, and they are slow without a GPU."
        )
        alternatives = {"vision": ["qwen3-vl:8b-instruct"], "text": ["qwen3:8b"], "whisper": ["small"]}

    if goal == "quality":
        if ram >= 32:
            vision, text, whisper = "qwen3-vl:8b-instruct", "qwen3:8b", "medium"
        elif ram >= 16:
            text, whisper = "qwen3:8b", "small"
        elif ram >= 12:
            vision, whisper = "qwen3-vl:4b-instruct", "small"
        if ram >= 12:
            notes.append(
                "Best quality: larger models are chosen because they fit in this computer's memory. "
                "On the processor they take noticeably longer."
            )
    elif goal == "fast":
        if vision != "qwen3-vl:2b-instruct":
            vision = "qwen3-vl:2b-instruct"
            notes.append("Fastest: the smallest video model is chosen. It is weaker at reading small on-screen text.")

    return _pack(
        inference_mode="cpu",
        vision=vision,
        text=text,
        whisper=whisper,
        whisper_device="cpu",
        # One frame per request: on the processor a larger batch saves no time.
        batch=1,
        vision_ctx=vision_ctx_for_batch(1),
        text_ctx=2048 if low else 4096,
        threads=threads,
        warm_both=False,
        suggest_disable_vision=low and goal != "quality",
        notes=notes,
        alternatives=alternatives,
    )


def _gpu_setup(total: float, budget: float, threads: int, goal: str, ram: float = 0.0) -> dict[str, Any]:
    notes: list[str] = []
    alternatives: dict[str, list[str]] = {"vision": [], "text": [], "whisper": []}
    if total >= 24 and budget >= VISION_FIT_GB["qwen3-vl:30b-a3b-instruct"]:
        vision = "qwen3-vl:30b-a3b-instruct"
        text = "qwen3:8b"
        alternatives["vision"] = ["qwen3-vl:8b-instruct"]
        alternatives["text"] = ["qwen3:30b"]
        notes.append(
            "30B vision is selected because this GPU has room for it. Transcript scoring stays on the 8B text model so both large models are not loaded together."
        )
    elif total >= 10 and budget >= VISION_FIT_GB["qwen3-vl:8b-instruct"]:
        vision = "qwen3-vl:8b-instruct"
        text = "qwen3:8b"
        notes.append("The vision and text models run one at a time.")
        if total >= 20:
            alternatives["vision"] = ["qwen3-vl:30b-a3b-instruct"]
            alternatives["text"] = ["qwen3:30b"]
    elif budget >= VISION_FIT_GB["qwen3-vl:4b-instruct"] or total >= 4:
        vision = "qwen3-vl:4b-instruct" if budget >= VISION_FIT_GB["qwen3-vl:4b-instruct"] else "qwen3-vl:2b-instruct"
        text = TEXT_4B
        if budget >= VISION_FIT_GB["qwen3-vl:8b-instruct"]:
            alternatives["vision"] = ["qwen3-vl:8b-instruct"]
            notes.append("qwen3-vl:8b-instruct can fit, and 4B stays the default so vision remains reliable.")
    else:
        vision = "qwen3-vl:2b-instruct"
        text = TEXT_4B
        notes.append("Very little free VRAM. The smallest vision model is selected.")

    split = False
    if goal == "quality":
        vision, text, split = _quality_step(vision, text, budget, ram, notes)
    elif goal == "fast":
        vision, text = _fast_step(vision, text, notes)

    whisper, whisper_device, whisper_alts = _whisper_choice(vision, total, budget)
    if goal == "quality" and not split and total >= 6 and budget >= 4 and whisper not in {"large-v3-turbo", "large-v3"}:
        # Speech to text finishes before the AI models load, so it can have the card to itself.
        whisper, whisper_device = "large-v3-turbo", "cuda"
        notes.append(
            "Best quality: speech to text uses a large model on the graphics card and is unloaded before the AI models need the memory."
        )
    alternatives["whisper"] = [name for name in whisper_alts if name != whisper]
    if whisper_device == "cpu" and total >= 6:
        notes.append("Whisper stays on CPU so it does not compete with the vision model for VRAM.")

    warm_both = total >= 20 and "30b" not in vision and "30b" not in text and not split
    tight = budget < 8 or split
    batch = 1 if tight else 2
    return _pack(
        inference_mode="gpu",
        vision=vision,
        text=text,
        whisper=whisper,
        whisper_device=whisper_device,
        batch=batch,
        vision_ctx=vision_ctx_for_batch(batch) if tight else max(4096, vision_ctx_for_batch(batch)),
        text_ctx=4096,
        threads=threads,
        warm_both=warm_both,
        suggest_disable_vision=False,
        notes=notes,
        alternatives=alternatives,
    )


def _quality_step(vision: str, text: str, budget: float, ram: float, notes: list[str]) -> tuple[str, str, bool]:
    """Best quality on a graphics card: one model size up when it fits, or when most of it fits
    and the rest can sit in system memory."""
    larger = next_larger(vision)
    if not larger or "30b" in larger:
        return vision, text, False
    need = VISION_FIT_GB[larger]
    if budget >= need:
        notes.append(f"Best quality: {larger} fits in the free graphics memory, so it is chosen over the smaller default.")
        return larger, _text_for_quality(larger, text), False
    if budget >= need * _SPLIT_MIN_FRACTION and ram >= 16:
        notes.append(
            f"Best quality: {larger} is chosen. About {round(budget / need * 100)}% of it fits in graphics memory "
            "and the rest runs on the processor, which is slower than a model that fits completely."
        )
        return larger, _text_for_quality(larger, text), True
    return vision, text, False


def _text_for_quality(vision: str, text: str) -> str:
    # The models run one at a time, so the text model can match the vision model's size.
    return "qwen3:8b" if "8b" in vision and text == TEXT_4B else text


def _fast_step(vision: str, text: str, notes: list[str]) -> tuple[str, str]:
    """Fastest on a graphics card: one model size down, never below 4B unless already there."""
    smaller = {"qwen3-vl:30b-a3b-instruct": "qwen3-vl:8b-instruct", "qwen3-vl:8b-instruct": "qwen3-vl:4b-instruct"}
    if vision in smaller:
        notes.append(f"Fastest: {smaller[vision]} is chosen over {vision}. It understands a little less.")
        return smaller[vision], (TEXT_4B if smaller[vision] == "qwen3-vl:4b-instruct" else text)
    return vision, text


def _whisper_choice(vision: str, total: float, budget: float) -> tuple[str, str, list[str]]:
    spare = budget - VISION_FIT_GB.get(vision, 4)
    # large-v3-turbo is the large default: distil-large-v3 is faster but transcribes English only.
    if total >= 16 and budget >= 8:
        alts = ["large-v3", "distil-large-v3"] if total >= 24 else ["medium", "small", "distil-large-v3"]
        if spare >= 2 or budget >= 10:
            return "large-v3-turbo", "cuda", alts
        return "small", "cpu", alts
    if total >= 10:
        alts = ["large-v3-turbo"] if budget >= 12 else []
        if spare >= 2:
            return "small", "cuda", alts
        return "small", "cpu", alts
    if total >= 6 or budget >= 6:
        if spare >= 2:
            return "small", "cuda", []
        return "small", "cpu", []
    return "base", "cpu", []


def _pack(
    *,
    inference_mode: str,
    vision: str,
    text: str,
    whisper: str,
    whisper_device: str,
    batch: int,
    vision_ctx: int,
    text_ctx: int,
    threads: int,
    warm_both: bool,
    suggest_disable_vision: bool,
    notes: list[str],
    alternatives: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "inference_mode": inference_mode,
        "vision_model": vision,
        "text_model": text,
        "whisper_model": whisper,
        "whisper_device": whisper_device,
        "vision_batch_size": batch,
        "vision_num_ctx": vision_ctx,
        "text_num_ctx": text_ctx,
        "num_thread": threads,
        "whisper_threads": threads,
        "warm_both": warm_both,
        "suggest_disable_vision": suggest_disable_vision,
        "notes": notes,
        "alternatives": alternatives,
    }
