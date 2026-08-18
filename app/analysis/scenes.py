from __future__ import annotations

from pathlib import Path

from app.models.schemas import Scene
from app.video.ffmpeg_runner import FFmpegRunner


def detect_scenes(
    video_path: str | Path,
    threshold: float = 27.0,
    runner: FFmpegRunner | None = None,
) -> list[Scene]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except ImportError:
        return _fallback_windows(video_path, runner)

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    try:
        manager.add_detector(AdaptiveDetector())
    except Exception:
        pass
    manager.detect_scenes(video)
    scene_list = manager.get_scene_list()
    scenes: list[Scene] = []
    for idx, (start, end) in enumerate(scene_list):
        start_s = start.get_seconds()
        end_s = end.get_seconds()
        scenes.append(Scene(scene_id=idx, start=start_s, end=end_s, duration=end_s - start_s))
    if not scenes and runner:
        meta = runner.probe(video_path)
        scenes = [Scene(scene_id=0, start=0, end=meta.duration, duration=meta.duration)]
    return scenes


def _fallback_windows(video_path: str | Path, runner: FFmpegRunner | None) -> list[Scene]:
    duration = 60.0
    if runner:
        duration = runner.probe(video_path).duration or duration
    window = 8.0
    scenes = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(duration, start + window)
        scenes.append(Scene(scene_id=idx, start=start, end=end, duration=end - start))
        start = end
        idx += 1
    return scenes or [Scene(scene_id=0, start=0, end=duration, duration=duration)]
