"""How far a job has come: one 0-100 figure, counts inside the step that runs, and time left.

A job is a list of weighted steps. A step's weight is its expected cost in seconds, so a long
transcription moves the bar more than reading the video's details. Steps that are already
finished from an earlier run (cached) or that do not apply are taken out of the total, so they
neither hold the bar back nor make it jump. The figure that is shown never goes backwards.

The time left uses how fast the running step is going once it has a steady speed, plus the
remaining steps corrected by the speed seen so far in this run. It is None until it is
reliable enough to show.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Hashable

STAGES = [
    ("prepare", "Preparing video"),
    ("metadata", "Video metadata"),
    ("scenes", "Scene detection"),
    ("audio", "Audio extraction"),
    ("transcript", "Transcription"),
    ("silence", "Silence detection"),
    ("frames", "Keyframe extraction"),
    ("sounds", "Understanding sounds"),
    ("content", "Finding important moments"),
    ("visual", "Visual AI analysis"),
    ("repetition", "Removing repetition"),
    ("director", "Planning with AI"),
    ("planning", "Planning highlight"),
]

# Steps that run once per source video; the others run once per job.
PER_SOURCE = ("metadata", "scenes", "audio", "transcript", "silence", "frames", "sounds", "content", "visual")
BEFORE_SOURCES = ("prepare",)
AFTER_SOURCES = ("repetition", "director", "planning")

# Expected cost of each step in seconds: (fixed, per second of video) on a CPU-only computer.
# Only the ratios matter for the percentage; the time left is corrected by the speed of this run.
_COST: dict[str, tuple[float, float]] = {
    "prepare": (1.0, 0.15),
    "metadata": (0.5, 0.0),
    "scenes": (1.0, 0.08),
    "audio": (1.0, 0.01),
    "transcript": (2.0, 0.33),
    "silence": (0.5, 0.01),
    "frames": (2.0, 0.03),
    "sounds": (2.0, 0.05),
    "content": (2.0, 0.8),
    "visual": (2.0, 0.8),
    "repetition": (1.0, 0.01),
    "director": (15.0, 0.0),
    "planning": (1.0, 0.0),
}
# Used until a video's length is known.
DEFAULT_VIDEO_SECONDS = 600.0
# Reply tokens per transcript segment and frames per scene, to turn measured speeds into costs.
TEXT_TOKENS_PER_SEGMENT = 48
FRAMES_PER_VISION_SCENE = 1.5

# A new figure is sent at most this often per step (a finished step is always sent).
EMIT_INTERVAL = 0.5
# Time left is shown once this share of the job is done and the job has run this long.
_ETA_MIN_FRACTION = 0.03
_ETA_MIN_SECONDS = 10.0
# The running step's own speed is trusted from this share and this many seconds.
_STEP_RATE_MIN_FRACTION = 0.05
_STEP_RATE_MIN_SECONDS = 5.0
# How strongly the expected costs are trusted before this run's speed is known (in seconds).
_PRIOR_SECONDS = 60.0
# Smoothing of the time left between updates (share of the new estimate).
_ETA_SMOOTHING = 0.3

Key = Hashable


@dataclass
class _Unit:
    weight: float
    fraction: float = 0.0
    state: str = "pending"  # pending, running, done, removed
    started: float | None = None
    spent: float = 0.0
    touched: bool = False


class ProgressModel:
    """Weighted steps in order. Pure bookkeeping: no clock of its own and no output."""

    def __init__(self, units: list[tuple[Key, float]] | None = None):
        self.units: dict[Key, _Unit] = {}
        for key, weight in units or []:
            self.add(key, weight)

    def add(self, key: Key, weight: float) -> None:
        if key not in self.units:
            self.units[key] = _Unit(weight=max(0.01, float(weight)))

    def set_weight(self, key: Key, weight: float) -> None:
        unit = self.units.get(key)
        if unit is not None:
            unit.weight = max(0.01, float(weight))

    def start(self, key: Key, now: float) -> None:
        unit = self.units.get(key)
        if unit is None:
            return
        self.remove_untouched_before(key)
        if unit.state != "running":
            unit.state = "running"
            unit.started = now
            unit.fraction = 0.0
        unit.touched = True

    def set_fraction(self, key: Key, fraction: float, now: float) -> None:
        unit = self.units.get(key)
        if unit is None:
            return
        if unit.state != "running":
            self.start(key, now)
        unit.fraction = min(1.0, max(0.0, float(fraction)))

    def finish(self, key: Key, now: float) -> None:
        """A step that ended. One that never ran was finished earlier (cached): it is taken out."""
        unit = self.units.get(key)
        if unit is None:
            return
        if unit.state == "running":
            unit.spent += max(0.0, now - (now if unit.started is None else unit.started))
            unit.state = "done"
            unit.fraction = 1.0
        elif unit.state != "done":
            unit.state = "removed"
        unit.touched = True
        unit.started = None

    def remove(self, key: Key) -> None:
        unit = self.units.get(key)
        if unit is not None and unit.state != "done":
            unit.state = "removed"
            unit.touched = True
            unit.started = None

    def remove_untouched_before(self, key: Key) -> None:
        """Steps listed before ``key`` that never started were not needed this time."""
        for other, unit in self.units.items():
            if other == key:
                break
            if not unit.touched and unit.state == "pending":
                unit.state = "removed"
                unit.touched = True

    def running(self) -> tuple[Key, _Unit] | None:
        for key, unit in self.units.items():
            if unit.state == "running":
                return key, unit
        return None

    def fraction(self) -> float:
        active = [unit for unit in self.units.values() if unit.state != "removed"]
        total = sum(unit.weight for unit in active)
        if total <= 0:
            return 0.0
        return min(1.0, sum(unit.weight * unit.fraction for unit in active) / total)

    def remaining_seconds(self, now: float) -> float:
        """Expected seconds left: the running step from its own speed when steady, the rest from
        their costs scaled by the speed of the steps finished in this run."""
        finished = [unit for unit in self.units.values() if unit.state == "done" and unit.spent > 0]
        expected = sum(unit.weight for unit in finished)
        actual = sum(unit.spent for unit in finished)
        speed = (actual + _PRIOR_SECONDS) / (expected + _PRIOR_SECONDS)
        left = 0.0
        for unit in self.units.values():
            if unit.state == "pending":
                left += speed * unit.weight
            elif unit.state == "running":
                elapsed = max(0.0, now - (now if unit.started is None else unit.started))
                if unit.fraction >= _STEP_RATE_MIN_FRACTION and elapsed >= _STEP_RATE_MIN_SECONDS:
                    left += elapsed * (1.0 - unit.fraction) / unit.fraction
                else:
                    left += speed * unit.weight * (1.0 - unit.fraction)
        return max(0.0, left)


class _Reporter:
    """Shared output of the job reporters: the figure never goes back, updates are spaced out,
    and the time left is smoothed."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.model = ProgressModel()
        self.lock = threading.RLock()
        self.begun = clock()
        self._shown = 0.0
        self._eta: float | None = None
        self._eta_at = 0.0
        self._last_emit: dict[Any, float] = {}

    def _overall(self) -> float:
        self._shown = max(self._shown, self.model.fraction())
        return self._shown

    def _time_left(self, now: float) -> float | None:
        overall = self._shown
        if overall < _ETA_MIN_FRACTION or now - self.begun < _ETA_MIN_SECONDS or overall >= 1.0:
            return None if overall < 1.0 else 0.0
        estimate = self.model.remaining_seconds(now)
        if self._eta is None:
            self._eta = estimate
        else:
            carried = max(0.0, self._eta - (now - self._eta_at))
            self._eta = carried + _ETA_SMOOTHING * (estimate - carried)
        self._eta_at = now
        return round(self._eta, 1)

    def _due(self, key: Any, now: float, final: bool) -> bool:
        if final or now - self._last_emit.get(key, -1e9) >= EMIT_INTERVAL:
            self._last_emit[key] = now
            return True
        return False


class PipelineProgress(_Reporter):
    """Progress of an analysis, sent through ``emit(kind, payload)`` as "progress" and "log"."""

    def __init__(
        self,
        emit: Callable[[str, dict], None],
        speeds: dict[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(clock)
        self.emit = emit
        self.speeds = dict(speeds or {})
        self.started = time.time()
        # Seconds spent in each stage that actually ran (cached stages are not counted).
        self.timings: dict[str, float] = {}
        self._running: dict[str, float] = {}
        # Model work per AI stage ("vision", "text"), summed over every source video.
        self.usage: dict[str, dict[str, float]] = {}
        # What had to be adjusted while running (for example after running out of memory).
        self.adjustments: list[str] = []
        self.source_index = 0
        self.source_count = 1
        self._durations: list[float | None] = [None]
        self._sizes: list[float] = [0.0]
        self._overrides: dict[Key, float] = {}
        self._steps: dict[str, tuple[float, float, str]] = {}
        self._build()

    # Plan of the job

    def configure_sources(self, sizes: list[float], durations: list[float | None] | None = None) -> None:
        """The source videos of the job (file sizes, and lengths where known), before any runs."""
        with self.lock:
            count = max(1, len(sizes))
            self.source_count = count
            self._sizes = [float(size or 0) for size in sizes] or [0.0]
            given = list(durations or [])
            self._durations = [
                (float(given[i]) if i < len(given) and given[i] else None) for i in range(count)
            ]
            # Steps that already ran (preparing the videos) keep how far they got.
            earlier = self.model.units
            self._build()
            for key, unit in earlier.items():
                if key in self.model.units and unit.touched:
                    weight = self.model.units[key].weight
                    self.model.units[key] = unit
                    unit.weight = weight

    def begin_source(self, index: int) -> None:
        with self.lock:
            self.source_index = max(0, min(index, self.source_count - 1))
            for key, unit in self.model.units.items():
                if isinstance(key, tuple) and key[1] is not None and key[1] < self.source_index:
                    if unit.state == "running":
                        self.model.finish(key, self.clock())
                    elif unit.state == "pending":
                        self.model.remove(key)

    def source_duration(self, seconds: float) -> None:
        """The current video's length, once known: costs that depend on it are worked out again."""
        with self.lock:
            if seconds and seconds > 0:
                self._durations[self.source_index] = float(seconds)
                self._reweigh()

    def expect(self, stage: str, seconds: float) -> None:
        """A better cost for a step of the current video (or of the job), once it is known."""
        with self.lock:
            key = self._key(stage)
            self._overrides[key] = max(0.01, float(seconds))
            self.model.set_weight(key, self._overrides[key])

    def skip(self, *stages: str) -> None:
        """Steps that will not run for the current video (or the job): finished earlier or off."""
        with self.lock:
            for stage in stages:
                self.model.remove(self._key(stage))

    def only(self, *stages: str) -> None:
        """Only these job steps run (planning again from a finished analysis)."""
        with self.lock:
            keep = {self._key(stage) for stage in stages}
            for key in list(self.model.units):
                if key not in keep:
                    self.model.remove(key)

    # Reports

    def log(self, message: str, level: str = "INFO") -> None:
        self.emit("log", {"level": level, "message": message, "ts": time.time()})

    def stage(self, name: str, status: str, detail: str = "", index: int = 0, total: int = 1) -> None:
        """A step started, ended or was skipped. ``index``/``total`` (when total > 1) is a count."""
        with self.lock:
            now = self.clock()
            self._time_stage(name, status)
            key = self._key(name)
            step = (0.0, 0.0, "")
            if status == "running":
                self.model.start(key, now)
                if total > 1:
                    step = (float(index), float(total), "")
                    self._steps[name] = step
                    self.model.set_fraction(key, index / total, now)
                else:
                    self._steps.pop(name, None)
            elif status == "done":
                self.model.finish(key, now)
                _, count, unit = self._steps.pop(name, step)
                step = (count, count, unit)
            else:
                self.model.remove(key)
                self._steps.pop(name, None)
            self._last_emit[name] = now
            self._send(name, status, detail, now, step)

    def step(self, name: str, done: float, total: float, unit: str = "", detail: str = "") -> None:
        """How far the running step ``name`` is: ``done`` of ``total`` ``unit`` ("time" for seconds
        of video; "" for a plain share). Spaced out to a few updates a second."""
        if total <= 0:
            return
        with self.lock:
            now = self.clock()
            done = min(float(done), float(total))
            self._steps[name] = (done, float(total), unit)
            key = self._key(name)
            if key not in self.model.units or self.model.units[key].state != "running":
                self._time_stage(name, "running")
            self.model.set_fraction(key, done / total, now)
            if self._due(name, now, done >= total):
                self._send(name, "running", detail, now, self._steps[name])

    def share(self, name: str, unit: str = "time", total: float = 0.0, detail: str = "") -> Callable[[float], None]:
        """A callback taking a share 0..1 of step ``name`` (for FFmpeg and similar progress)."""

        def report(fraction: float) -> None:
            if total > 0:
                self.step(name, fraction * total, total, unit, detail)
            else:
                self.step(name, fraction * 100, 100, "", detail)

        return report

    # Output

    def _send(self, name: str, status: str, detail: str, now: float, step: tuple[float, float, str]) -> None:
        overall = self._overall()
        done, count, unit = step
        payload = {
            "stage": name,
            "status": status,
            "detail": detail,
            "index": int(done),
            "total": int(count) if count else 1,
            "step_done": round(done, 2),
            "step_total": round(count, 2),
            "step_unit": unit,
            "overall": round(overall * 100, 1),
            "eta": self._time_left(now),
            "source_index": self.source_index,
            "source_count": self.source_count,
            "elapsed": time.time() - self.started,
        }
        self.emit("progress", payload)

    # Stage timings

    def _time_stage(self, name: str, status: str) -> None:
        now = time.perf_counter()
        if status == "running":
            self._running.setdefault(name, now)
        elif status == "done" and name in self._running:
            spent = now - self._running.pop(name)
            self.timings[name] = round(self.timings.get(name, 0.0) + spent, 2)

    def timing_summary(self) -> str:
        labels = dict(STAGES)
        parts = [f"{labels.get(name, name)} {seconds:.1f}s" for name, seconds in self.timings.items() if seconds >= 0.05]
        return ", ".join(parts)

    # Weights

    def _key(self, stage: str) -> Key:
        if stage in PER_SOURCE:
            return (stage, self.source_index)
        return (stage, None)

    def _build(self) -> None:
        self.model = ProgressModel()
        for stage in BEFORE_SOURCES:
            self.model.add((stage, None), 1.0)
        for index in range(self.source_count):
            for stage in PER_SOURCE:
                self.model.add((stage, index), 1.0)
        for stage in AFTER_SOURCES:
            self.model.add((stage, None), 1.0)
        self._reweigh()

    def _duration(self, index: int) -> float:
        known = self._durations[index] if index < len(self._durations) else None
        if known:
            return known
        # Not probed yet: in proportion to the file size of the videos whose length is known.
        rates = [
            (self._durations[i] or 0) / self._sizes[i]
            for i in range(len(self._durations))
            if self._durations[i] and i < len(self._sizes) and self._sizes[i] > 0
        ]
        size = self._sizes[index] if index < len(self._sizes) else 0
        if rates and size > 0:
            return sum(rates) / len(rates) * size
        known_lengths = [d for d in self._durations if d]
        return sum(known_lengths) / len(known_lengths) if known_lengths else DEFAULT_VIDEO_SECONDS

    def _cost(self, stage: str, seconds: float) -> float:
        fixed, per_second = _COST.get(stage, (1.0, 0.0))
        rtf = self.speeds.get("whisper_realtime_factor")
        if stage == "transcript" and rtf:
            return fixed + seconds / rtf
        tokens = self.speeds.get("text_tokens_per_sec")
        if stage == "content" and tokens:
            return fixed + seconds / 5.0 * TEXT_TOKENS_PER_SEGMENT / tokens
        per_image = self.speeds.get("vision_seconds_per_image")
        if stage == "visual" and per_image:
            return fixed + seconds / 20.0 * FRAMES_PER_VISION_SCENE * per_image
        return fixed + per_second * seconds

    def _reweigh(self) -> None:
        total = sum(self._duration(i) for i in range(self.source_count))
        for key in self.model.units:
            stage, index = key
            if key in self._overrides:
                weight = self._overrides[key]
            elif index is None:
                weight = self._cost(stage, total)
            else:
                weight = self._cost(stage, self._duration(index))
            self.model.set_weight(key, weight)

    # Estimates from counts

    def vision_seconds(self, scenes: int) -> float:
        return max(1.0, scenes * FRAMES_PER_VISION_SCENE * self.speeds.get("vision_seconds_per_image", 6.0))

    def text_seconds(self, segments: int) -> float:
        return max(1.0, segments * TEXT_TOKENS_PER_SEGMENT / self.speeds.get("text_tokens_per_sec", 10.0))


class RenderProgress(_Reporter):
    """Progress of creating the videos: every clip weighted by its length, then the extra pass
    for background music. ``emit(payload)`` receives the same fields as an analysis update."""

    # Mixing music re-encodes the sound only: a fraction of the clip encoding time.
    MUSIC_COST = 0.15

    def __init__(
        self,
        groups: list[list[float]],
        music: bool,
        emit: Callable[[dict], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(clock)
        self.emit = emit
        self.groups = groups
        self.clip_total = sum(len(group) for group in groups)
        self._clips_done = 0
        for g, clips in enumerate(groups):
            for c, seconds in enumerate(clips):
                self.model.add(("clip", g, c), max(0.05, seconds))
            if music:
                self.model.add(("music", g), max(0.5, sum(clips) * self.MUSIC_COST))
        self._detail = ""

    def clip(self, group: int, index: int, fraction: float, detail: str = "") -> None:
        """Clip ``index`` of video ``group`` is ``fraction`` encoded (1.0 when done or reused)."""
        with self.lock:
            now = self.clock()
            key = ("clip", group, index)
            unit = self.model.units.get(key)
            if unit is None or unit.state == "done":
                # A clip encoded again in software after the hardware encoder failed stays done.
                return
            final = fraction >= 1.0
            if final:
                if unit.state != "done":
                    if unit.state != "running":
                        # Reused from an earlier run: it counts as done straight away.
                        self.model.start(key, now)
                        unit.started = now
                    self.model.finish(key, now)
                    self._clips_done += 1
            else:
                self.model.set_fraction(key, fraction, now)
            self._detail = detail or self._detail
            if self._due("render", now, final):
                self._send(now)

    def music(self, group: int, fraction: float) -> None:
        with self.lock:
            now = self.clock()
            key = ("music", group)
            if key not in self.model.units:
                return
            final = fraction >= 1.0
            if final:
                if self.model.units[key].state != "running":
                    self.model.start(key, now)
                self.model.finish(key, now)
            else:
                self.model.set_fraction(key, fraction, now)
            if self._due("render", now, final):
                self._send(now)

    def skip_music(self, group: int) -> None:
        with self.lock:
            self.model.remove(("music", group))

    def detail(self, text: str) -> None:
        with self.lock:
            self._detail = text
            self._send(self.clock())

    def _send(self, now: float) -> None:
        overall = self._overall()
        self.emit(
            {
                "stage": "rendering",
                "status": "running",
                "detail": self._detail,
                "step_done": self._clips_done,
                "step_total": self.clip_total,
                "step_unit": "clips",
                "overall": round(overall * 100, 1),
                "eta": self._time_left(now),
            }
        )
