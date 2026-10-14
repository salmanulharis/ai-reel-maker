from __future__ import annotations

from typing import Any


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


def recommend_setup(
    vram_gb: float | None,
    ram_gb: float | None,
    free_vram_gb: float | None = None,
    cpu_threads: int | None = None,
) -> dict[str, Any]:
    """Pick models from GPU memory. System RAM never selects 8B or 30B by itself."""
    total = float(vram_gb or 0)
    free = float(free_vram_gb) if free_vram_gb is not None else total
    budget = free if free > 0 else total
    threads = recommended_threads(cpu_threads)
    ram = float(ram_gb or 0)
    if total < 2 and budget < 2:
        return _cpu_setup(ram, threads)
    return _gpu_setup(total, budget, threads)


def _cpu_setup(ram: float, threads: int) -> dict[str, Any]:
    if ram < 16:
        vision = "qwen3-vl:2b-instruct"
        whisper = "tiny" if ram < 8 else "base"
        batch, vision_ctx, text_ctx = 1, 2048, 2048
        notes = [
            "CPU-only. Highlights still run. Vision is slower, and you can turn it off if you want a faster pass.",
        ]
        alternatives = {
            "vision": ["qwen3-vl:4b-instruct"] if ram >= 12 else [],
            "text": [],
            "whisper": ["base"] if whisper == "tiny" else ["small"],
        }
        suggest_disable = True
    else:
        vision = "qwen3-vl:4b-instruct"
        whisper = "base"
        batch, vision_ctx, text_ctx = 2, 2048, 4096
        notes = [
            "CPU-only. The 4B vision model is the default. Larger models stay available, and they are slow without a GPU.",
        ]
        alternatives = {
            "vision": ["qwen3-vl:8b-instruct"],
            "text": ["qwen3:8b"],
            "whisper": ["small"],
        }
        suggest_disable = False
    return _pack(
        inference_mode="cpu",
        vision=vision,
        text=TEXT_4B,
        whisper=whisper,
        whisper_device="cpu",
        batch=batch,
        vision_ctx=vision_ctx,
        text_ctx=text_ctx,
        threads=threads,
        warm_both=False,
        suggest_disable_vision=suggest_disable,
        notes=notes,
        alternatives=alternatives,
    )


def _gpu_setup(total: float, budget: float, threads: int) -> dict[str, Any]:
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

    whisper, whisper_device, whisper_alts = _whisper_choice(vision, total, budget)
    alternatives["whisper"] = whisper_alts
    if whisper_device == "cpu" and total >= 6:
        notes.append("Whisper stays on CPU so it does not compete with the vision model for VRAM.")

    warm_both = total >= 20 and "30b" not in vision and "30b" not in text
    tight = budget < 8
    return _pack(
        inference_mode="gpu",
        vision=vision,
        text=text,
        whisper=whisper,
        whisper_device=whisper_device,
        batch=2 if tight else 4,
        vision_ctx=2048 if tight else 4096,
        text_ctx=4096,
        threads=threads,
        warm_both=warm_both,
        suggest_disable_vision=False,
        notes=notes,
        alternatives=alternatives,
    )


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
