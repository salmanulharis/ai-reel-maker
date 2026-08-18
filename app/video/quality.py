from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import RenderSettings, VideoMetadata


@dataclass(frozen=True)
class OutputCanvas:
    width: int
    height: int
    fps: float


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
    fps = float(best.fps or 0) or max((float(item.fps or 0) for item in usable), default=0.0) or 30.0
    cap = {"1080p": 1080, "720p": 720}.get(str(settings.resolution or ""), None)
    if cap and height > cap:
        scale = cap / height
        width = even_dim(width * scale)
        height = even_dim(cap)
    return OutputCanvas(width=width, height=height, fps=fps)


def fps_filter(fps: float) -> str:
    if abs(fps - 24000 / 1001) < 0.05 or abs(fps - 23.976) < 0.04:
        return "fps=24000/1001"
    if abs(fps - 30000 / 1001) < 0.05 or abs(fps - 29.97) < 0.04:
        return "fps=30000/1001"
    if abs(fps - round(fps)) < 0.05:
        return f"fps={int(round(fps))}"
    return f"fps={fps:.3f}"


def clip_normalize_filters(
    source: VideoMetadata,
    canvas: OutputCanvas,
    extra: list[str] | None = None,
) -> list[str]:
    filters: list[str] = []
    src_w = even_dim(source.width or canvas.width)
    src_h = even_dim(source.height or canvas.height)
    if src_w != canvas.width or src_h != canvas.height:
        filters.append(
            f"scale={canvas.width}:{canvas.height}:force_original_aspect_ratio=decrease:flags=lanczos"
        )
        filters.append(f"pad={canvas.width}:{canvas.height}:(ow-iw)/2:(oh-ih)/2:color=black")
    filters.append(fps_filter(canvas.fps))
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
