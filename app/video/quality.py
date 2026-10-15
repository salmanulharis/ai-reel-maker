from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import RenderSettings, VideoMetadata


@dataclass(frozen=True)
class OutputCanvas:
    width: int
    height: int
    fps: float
    # True when an aspect ratio is chosen: every clip is scaled to cover the canvas and the
    # overflow is cropped from the centre, instead of fitting it inside black bars.
    fill: bool = False


ASPECT_RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5}


def even_dim(value: int | float) -> int:
    number = max(2, int(round(float(value))))
    return number - (number % 2)


def choose_canvas(metas: list[VideoMetadata], settings: RenderSettings | None = None) -> OutputCanvas:
    """Pick a consistent output size/fps from source videos without inventing pixels."""
    settings = settings or RenderSettings()
    usable = [item for item in metas if int(item.width or 0) > 0 and int(item.height or 0) > 0]
    if not usable:
        return OutputCanvas(1920, 1080, 30.0)
    best = max(usable, key=lambda item: int(item.width) * int(item.height))
    width = even_dim(best.width)
    height = even_dim(best.height)
    ratio = ASPECT_RATIOS.get(str(getattr(settings, "aspect_ratio", "original") or "original"))
    if ratio:
        # The largest box of the chosen shape that fits inside the main source.
        if width / height > ratio:
            width = even_dim(height * ratio)
        else:
            height = even_dim(width / ratio)
    fps = target_frame_rate(best, usable, getattr(settings, "frame_rate", "source"))
    cap = {"1080p": 1080, "720p": 720}.get(str(settings.resolution or ""), None)
    if cap and height > cap:
        scale = cap / height
        width = even_dim(width * scale)
        height = even_dim(cap)
    return OutputCanvas(width=width, height=height, fps=fps, fill=bool(ratio))


# NTSC rates and how far a probed value may be from them. Tight enough that a true 24 or 30 fps
# source (0.024 / 0.03 away) keeps its own rate instead of being slowed to 23.976 / 29.97.
_NTSC_RATES = ((24000, 1001), (30000, 1001), (60000, 1001))
_NTSC_TOLERANCE = 0.01


def fps_filter(fps: float) -> str:
    return f"fps={rate_expression(fps)}"


def rate_expression(fps: float) -> str:
    """FFmpeg rate for a frame rate: exact fractions for NTSC rates, whole numbers when close."""
    for num, den in _NTSC_RATES:
        if abs(fps - num / den) < _NTSC_TOLERANCE or abs(fps - round(num / den, 2)) < _NTSC_TOLERANCE:
            return f"{num}/{den}"
    if abs(fps - round(fps)) < 0.05:
        return str(int(round(fps)))
    return f"{fps:.3f}"


def snap_rate(fps: float) -> float:
    """The exact value of a probed rate: 23.98 -> 23.976..., 30.0 -> 30.0, others unchanged."""
    expression = rate_expression(fps)
    if "/" in expression:
        num, den = expression.split("/")
        return int(num) / int(den)
    return float(expression)


# Choices offered for the output, besides keeping the source rate.
FRAME_RATE_CHOICES = {"24": 24.0, "25": 25.0, "30": 30.0, "50": 50.0, "60": 60.0}
# A declared rate this far from the measured average marks a variable-frame-rate source
# (phone recordings, screen captures, game captures).
_VFR_TOLERANCE = 0.02
_MIN_FPS, _MAX_FPS = 1.0, 120.0
_COMMON_RATES = (24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 48.0, 50.0, 60000 / 1001, 60.0, 120.0)
_VFR_SNAP = 0.03


def is_variable_rate(meta: VideoMetadata) -> bool:
    declared, average = float(meta.fps or 0), float(meta.avg_fps or 0)
    if declared <= 0 or average <= 0:
        return False
    return abs(declared - average) / declared > _VFR_TOLERANCE


def source_frame_rate(meta: VideoMetadata) -> float:
    """The rate a source plays at. For a variable-rate source the average is the honest value:
    its declared rate is often a ceiling (60) or a doubled artifact (47.952 for 23.976 video)."""
    variable = is_variable_rate(meta)
    rate = float(meta.avg_fps or 0) if variable else float(meta.fps or meta.avg_fps or 0)
    if rate <= 0:
        return 0.0
    rate = min(_MAX_FPS, max(_MIN_FPS, rate))
    if variable:
        # Averages are noisy (a 30 fps phone clip may average 29.87): use the nearest standard
        # rate when it is close, and keep unusual rates (a 12 fps screen capture) as they are.
        nearest = min(_COMMON_RATES, key=lambda common: abs(common - rate))
        if abs(nearest - rate) / nearest <= _VFR_SNAP:
            return nearest
        return round(rate, 2)
    return snap_rate(rate)


def target_frame_rate(primary: VideoMetadata, sources: list[VideoMetadata], choice: str | None) -> float:
    """Output rate: the chosen rate, or the main (largest) source's own rate."""
    if choice in FRAME_RATE_CHOICES:
        return FRAME_RATE_CHOICES[str(choice)]
    rate = source_frame_rate(primary)
    if rate:
        return rate
    return max((source_frame_rate(item) for item in sources), default=0.0) or 30.0


def same_rate(a: float, b: float) -> bool:
    return abs(a - b) < _NTSC_TOLERANCE


def frame_rate_plan(source: VideoMetadata, target: float, smooth: bool = False) -> str:
    """How one source reaches the output rate.

    "keep": constant rate that already matches; no filter, so no frame is duplicated or dropped.
    "convert": the fps filter. It keeps every frame on its own timestamp, so duration and audio
        sync are preserved; whole-number reductions (60->30, 50->25) drop frames evenly.
    "smooth": motion interpolation, only when asked for and when plain conversion would judder
        (rates that are not whole multiples of each other, or raising the rate)."""
    rate = source_frame_rate(source)
    if rate and same_rate(rate, target) and not is_variable_rate(source):
        return "keep"
    if smooth and rate and not _even_reduction(rate, target):
        return "smooth"
    return "convert"


def _even_reduction(rate: float, target: float) -> bool:
    if target > rate or same_rate(rate, target):
        return False
    ratio = rate / target
    return abs(ratio - round(ratio)) < 0.01


def frame_rate_filters(source: VideoMetadata, target: float, smooth: bool = False) -> list[str]:
    plan = frame_rate_plan(source, target, smooth)
    if plan == "keep":
        return []
    if plan == "smooth":
        # Motion-compensated interpolation: creates in-between frames instead of repeating or
        # skipping them. Much slower than "convert", so it is opt-in.
        return [
            f"minterpolate=fps={rate_expression(target)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        ]
    return [fps_filter(target)]


def describe_frame_rate(sources: list[VideoMetadata], target: float, smooth: bool = False) -> str:
    """One log line about what happens to the frame rate."""
    plans = [(source_frame_rate(item), frame_rate_plan(item, target, smooth), is_variable_rate(item)) for item in sources]
    if plans and all(plan == "keep" for _, plan, _ in plans):
        return f"Frame rate: keeping the original {target:.3f} fps (no conversion)."
    parts = []
    for rate, plan, variable in plans:
        source = f"{rate:.3f} fps{' (variable)' if variable else ''}" if rate else "unknown rate"
        how = {"keep": "kept", "convert": "converted", "smooth": "converted with motion smoothing"}[plan]
        parts.append(f"{source} {how}")
    return f"Frame rate: output {target:.3f} fps; " + ", ".join(dict.fromkeys(parts)) + "."


def clip_normalize_filters(
    source: VideoMetadata,
    canvas: OutputCanvas,
    extra: list[str] | None = None,
    smooth_motion: bool = False,
) -> list[str]:
    filters: list[str] = []
    src_w = even_dim(source.width or canvas.width)
    src_h = even_dim(source.height or canvas.height)
    if canvas.fill and (src_w != canvas.width or src_h != canvas.height):
        filters.append(
            f"scale={canvas.width}:{canvas.height}:force_original_aspect_ratio=increase:flags=lanczos"
        )
        filters.append(f"crop={canvas.width}:{canvas.height}")
    elif src_w != canvas.width or src_h != canvas.height:
        filters.append(
            f"scale={canvas.width}:{canvas.height}:force_original_aspect_ratio=decrease:flags=lanczos"
        )
        filters.append(f"pad={canvas.width}:{canvas.height}:(ow-iw)/2:(oh-ih)/2:color=black")
    filters.extend(frame_rate_filters(source, canvas.fps, smooth_motion))
    filters.extend(extra or [])
    filters.append("format=yuv420p")
    filters.append("setpts=PTS-STARTPTS")
    return filters


def encode_crf(quality: str, crf: int) -> int:
    mapping = {"source": 14, "very_high": 16, "high": 18}
    if quality in mapping:
        return mapping[quality]
    return max(12, min(int(crf or 18), 28))


def encode_preset(codec: str, quality: str) -> str:
    if codec == "h264_nvenc":
        return "p7" if quality in {"source", "very_high"} else "p5"
    if quality in {"source", "very_high"}:
        return "slow"
    return "medium"


def video_encode_args(codec: str, quality: str = "high", crf: int = 18) -> list[str]:
    codec = codec or "libx264"
    quality = quality or "high"
    if codec == "h264_nvenc":
        qp = {"source": 16, "very_high": 18, "high": 20}.get(quality, 20)
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            encode_preset(codec, quality),
            "-rc",
            "constqp",
            "-qp",
            str(qp),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    if codec in {"h264_qsv", "h264_amf", "h264_videotoolbox"}:
        return [
            "-c:v",
            codec,
            "-q:v",
            "3" if quality in {"source", "very_high"} else "5",
            "-pix_fmt",
            "yuv420p",
        ]
    args = [
        "-c:v",
        codec,
        "-preset",
        encode_preset(codec, quality),
        "-crf",
        str(encode_crf(quality, crf)),
        "-pix_fmt",
        "yuv420p",
    ]
    if codec == "libx264":
        args += ["-profile:v", "high"]
    return args


def audio_encode_args(quality: str = "high", audio_bitrate: str = "192k") -> list[str]:
    bitrate = "320k" if quality == "source" else (audio_bitrate or "192k")
    return ["-c:a", "aac", "-b:a", bitrate, "-ar", "48000", "-ac", "2"]


def quality_software_first(quality: str) -> bool:
    return (quality or "source") in {"source", "very_high"}
