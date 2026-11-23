from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from app.models.schemas import Scene
from app.video.ffmpeg_runner import FFmpegRunner


def detect_scenes(
    video_path: str | Path,
    threshold: float = 27.0,
    runner: FFmpegRunner | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[Scene]:
    """Scenes of the video. ``on_progress`` receives the share of the video read so far."""
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
    if on_progress is None:
        manager.detect_scenes(video)
    else:
        with _watch_position(video, on_progress):
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


class _watch_position:
    """Reports how far PySceneDetect has read, twice a second, from a helper thread."""

    def __init__(self, video, on_progress: Callable[[float], None]):
        self.video = video
        self.on_progress = on_progress
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="scene-progress")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        try:
            total = float(self.video.duration.get_seconds())
        except Exception:  # noqa: BLE001 - no length: nothing to report against
            return
        if total <= 0:
            return
        while not self.stop.wait(0.5):
            try:
                self.on_progress(min(1.0, float(self.video.position.get_seconds()) / total))
            except Exception:  # noqa: BLE001 - reporting never disturbs the detection
                pass


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
