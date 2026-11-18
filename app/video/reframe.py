"""
Smart framing for 9:16, 1:1 and 4:5 output: instead of always keeping the middle of the picture,
each clip's crop is centred on the people in it (the largest face, found with OpenCV in a few
frames of the clip). The framing is fixed for a clip, so it never wobbles. When no face is found
the crop stays in the middle, as before.
"""

from __future__ import annotations

from pathlib import Path

ASPECT_RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5}
# Where a face sits in a vertical crop: a little above the middle looks natural.
FACE_HEIGHT_IN_CROP = 0.4
SAMPLE_POINTS = (0.2, 0.5, 0.8)
DETECT_WIDTH = 640

_cascade = None


def _detector():
    global _cascade
    if _cascade is None:
        import cv2

        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _cascade = cascade if not cascade.empty() else False
    return _cascade or None


def largest_face(image) -> tuple[float, float, float] | None:
    """(centre x, centre y, area) of the largest face in an image, as shares of its size."""
    import cv2

    detector = _detector()
    if detector is None or image is None or getattr(image, "size", 0) == 0:
        return None
    height, width = image.shape[:2]
    if width > DETECT_WIDTH:
        scale = DETECT_WIDTH / width
        image = cv2.resize(image, (DETECT_WIDTH, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    side = max(24, int(min(width, height) * 0.06))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(side, side))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return (x + w / 2) / width, (y + h / 2) / height, (w * h) / (width * height)


def subject_center(images: list) -> tuple[float, float] | None:
    """Where to centre a clip's crop: the largest face of each frame, weighted by its size."""
    found = [face for face in (largest_face(image) for image in images) if face]
    if not found:
        return None
    total = sum(area for _, _, area in found)
    return (
        sum(x * area for x, _, area in found) / total,
        sum(y * area for _, y, area in found) / total,
    )


def clip_frames(runner, source: Path, start: float, end: float, work_dir: Path, index: int) -> list:
    """A few frames from a clip, read as images. Frames that cannot be read are skipped."""
    import cv2

    images = []
    length = max(0.0, end - start)
    for n, share in enumerate(SAMPLE_POINTS):
        path = work_dir / f"reframe_{index:03d}_{n}.jpg"
        try:
            runner.run(_frame_command(source, path, start + length * share))
            image = cv2.imread(str(path))
        except Exception:  # noqa: BLE001 - framing is a nicety; the middle is a fine fallback
            image = None
        finally:
            path.unlink(missing_ok=True)
        if image is not None:
            images.append(image)
    return images


def _frame_command(source: Path, dest: Path, at: float) -> list[str]:
    from app.video.processor import VideoProcessor

    return VideoProcessor.extract_frame_command(source, dest, at, width=DETECT_WIDTH)


def to_canvas(
    center: tuple[float, float], src_w: int, src_h: int, canvas_w: int, canvas_h: int
) -> tuple[float, float]:
    """A point in the source picture as a point in the output canvas, which the source is
    scaled into and padded to (letterboxed) when their shapes differ."""
    cx, cy = center
    if src_w <= 0 or src_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
        return cx, cy
    scale = min(canvas_w / src_w, canvas_h / src_h)
    content_w, content_h = src_w * scale, src_h * scale
    pad_x, pad_y = (canvas_w - content_w) / 2, (canvas_h - content_h) / 2
    return (pad_x + cx * content_w) / canvas_w, (pad_y + cy * content_h) / canvas_h


def aspect_crop_filter(aspect_ratio: str, center: tuple[float, float] | None = None) -> str | None:
    """FFmpeg crop to the output shape: in the middle, or around `center` (shares of the
    picture), always kept inside the picture."""
    ratio = ASPECT_RATIOS.get(aspect_ratio)
    if ratio is None:
        return None
    crop = f"crop='if(gte(iw/ih,{ratio}),ih*{ratio},iw)':'if(gte(iw/ih,{ratio}),ih,iw/{ratio})'"
    if center is None:
        return crop
    cx = min(1.0, max(0.0, float(center[0])))
    cy = min(1.0, max(0.0, float(center[1])))
    x = f"max(0,min(iw-ow,iw*{cx:.4f}-ow/2))"
    y = f"max(0,min(ih-oh,ih*{cy:.4f}-oh*{FACE_HEIGHT_IN_CROP}))"
    return f"{crop}:'{x}':'{y}'"
