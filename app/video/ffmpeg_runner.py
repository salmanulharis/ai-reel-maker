from __future__ import annotations

import json
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from app.models.schemas import VideoMetadata
from app.utils.timefmt import clamp_timestamp
from app.video.processor import VideoProcessor


class CancelledError(Exception):
    pass


class FFmpegError(RuntimeError):
    pass


# Where FFmpeg commands run by this thread report how far they are (see report_progress).
_sink = threading.local()


class FFmpegRunner:
    def __init__(self, cancel_event: threading.Event | None = None):
        self.cancel_event = cancel_event or threading.Event()
        # Several commands can run at once (parallel frame extraction), so cancel stops them all.
        self._procs: set[subprocess.Popen] = set()
        self._procs_lock = threading.Lock()
        self.processor = VideoProcessor()

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._procs_lock:
            procs = list(self._procs)
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def run(
        self,
        command: list[str],
        timeout: int | None = None,
        collect_stderr: bool = True,
    ) -> subprocess.CompletedProcess:
        if self.cancel_event.is_set():
            raise CancelledError("Job cancelled")
        sink = getattr(_sink, "target", None)
        if sink is not None and _is_ffmpeg(command):
            return self._run_reporting(command, timeout, collect_stderr, *sink)
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if collect_stderr else subprocess.DEVNULL,
            text=True,
            # FFmpeg writes UTF-8 (file names included). The Windows default code page cannot
            # decode names in other scripts, which silently emptied every probe.
            encoding="utf-8",
            errors="replace",
        )
        with self._procs_lock:
            self._procs.add(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise FFmpegError(f"Command timed out: {command[0]}")
        finally:
            code = proc.returncode
            with self._procs_lock:
                self._procs.discard(proc)
        if self.cancel_event.is_set():
            raise CancelledError("Job cancelled")
        result = subprocess.CompletedProcess(command, code, stdout, stderr)
        if code != 0:
            raise FFmpegError(_short_ffmpeg_error(stderr or stdout or f"exit {code}"))
        return result

    def _run_reporting(
        self,
        command: list[str],
        timeout: int | None,
        collect_stderr: bool,
        duration: float,
        on_progress: Callable[[float], None],
    ) -> subprocess.CompletedProcess:
        """Run an FFmpeg command that writes to a file, reading its progress output as it goes."""
        proc = subprocess.Popen(
            [command[0], "-progress", "pipe:1", "-nostats", *command[1:]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if collect_stderr else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        with self._procs_lock:
            self._procs.add(proc)
        errors: list[str] = []
        reader = None
        if collect_stderr:
            # Read alongside, so a full stderr pipe never blocks FFmpeg.
            reader = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
            reader.start()
        timed_out = threading.Event()
        timer = None
        if timeout:
            timer = threading.Timer(timeout, lambda: (timed_out.set(), proc.kill()))
            timer.daemon = True
            timer.start()
        try:
            for line in proc.stdout:
                share = parse_progress_line(line, duration)
                if share is not None:
                    try:
                        on_progress(share)
                    except Exception:  # noqa: BLE001 - reporting never disturbs the command
                        pass
            proc.wait()
        finally:
            if timer is not None:
                timer.cancel()
            if reader is not None:
                reader.join(timeout=5)
            code = proc.returncode
            with self._procs_lock:
                self._procs.discard(proc)
        if timed_out.is_set():
            raise FFmpegError(f"Command timed out: {command[0]}")
        if self.cancel_event.is_set():
            raise CancelledError("Job cancelled")
        stderr = "".join(errors)
        if code != 0:
            raise FFmpegError(_short_ffmpeg_error(stderr or f"exit {code}"))
        return subprocess.CompletedProcess(command, code, "", stderr)

    def probe(self, input_path: str | Path) -> VideoMetadata:
        result = self.run(VideoProcessor.probe_command(input_path))
        payload = json.loads(result.stdout or "{}")
        return parse_probe(payload, Path(input_path))

    def extract_audio(self, input_path: str | Path, output_path: str | Path, sample_rate: int = 16000) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.run(VideoProcessor.extract_audio_command(input_path, output, sample_rate))
        return output

    def extract_frame(
        self,
        input_path: str | Path,
        output_path: str | Path,
        timestamp: float,
        duration: float | None = None,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        safe = clamp_timestamp(timestamp, duration)
        try:
            self.run(VideoProcessor.extract_frame_command(input_path, output, safe))
        except FFmpegError:
            fallback = clamp_timestamp((duration or safe) * 0.5 if duration else max(0.0, safe - 0.4), duration)
            if abs(fallback - safe) < 0.05:
                raise
            self.run(VideoProcessor.extract_frame_command(input_path, output, fallback))
        return output

    def detect_silence(self, input_path: str | Path, noise_db: float, min_duration: float) -> list[dict]:
        try:
            result = self.run(
                VideoProcessor.silencedetect_command(input_path, noise_db, min_duration)
            )
            text = (result.stderr or "") + (result.stdout or "")
        except FFmpegError as exc:
            text = str(exc)
        return parse_silence(text)


@contextmanager
def report_progress(runner, duration: float | None, on_progress: Callable[[float], None] | None) -> Iterator[None]:
    """While active, FFmpeg commands that this thread runs through ``runner`` report the share of
    ``duration`` (seconds of media) they have written. Only for commands that write to a file or
    to null: their progress goes to stdout. Other runners (test doubles) are left alone."""
    if on_progress is None or not duration or duration <= 0 or not isinstance(runner, FFmpegRunner):
        yield
        return
    previous = getattr(_sink, "target", None)
    _sink.target = (float(duration), on_progress)
    try:
        yield
    finally:
        _sink.target = previous


def parse_progress_line(line: str, duration: float) -> float | None:
    """The share done from one line of FFmpeg's ``-progress`` output, or None for other lines."""
    key, _, value = line.strip().partition("=")
    if key == "progress" and value == "end":
        return 1.0
    # out_time_ms is in microseconds too (a long-standing FFmpeg quirk).
    if key not in {"out_time_us", "out_time_ms"} or duration <= 0:
        return None
    try:
        seconds = int(value) / 1_000_000
    except ValueError:
        return None
    return min(1.0, max(0.0, seconds / duration))


def _is_ffmpeg(command: list[str]) -> bool:
    name = Path(str(command[0])).stem.lower() if command else ""
    return name.startswith("ffmpeg")


def _parse_frame_rate(rate: object) -> float:
    if isinstance(rate, (int, float)):
        return float(rate)
    text = str(rate or "")
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            return float(num) / max(float(den), 1e-6)
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _sane_fps(r_frame_rate: object, avg_frame_rate: object) -> float:
    declared = _parse_frame_rate(r_frame_rate)
    average = _parse_frame_rate(avg_frame_rate)
    if average > 1 and (declared <= 1 or declared > 120 or (average > 1 and declared / average >= 4)):
        return average
    if 1 < declared <= 120:
        return declared
    if 1 < average <= 120:
        return average
    return average or declared or 0.0


def parse_probe(payload: dict, path: Path) -> VideoMetadata:
    fmt = payload.get("format") or {}
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fps = _sane_fps(video.get("r_frame_rate"), video.get("avg_frame_rate"))
    rotation = 0
    for side in video.get("side_data_list") or []:
        if "rotation" in side:
            try:
                rotation = int(float(side["rotation"]))
            except (TypeError, ValueError):
                rotation = 0
    tags = video.get("tags") or {}
    if "rotate" in tags:
        try:
            rotation = int(float(tags["rotate"]))
        except (TypeError, ValueError):
            pass
    duration = float(fmt.get("duration") or video.get("duration") or 0)
    size = int(fmt.get("size") or 0)
    if path.exists() and not size:
        size = path.stat().st_size
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if abs(rotation) % 180 == 90:
        # Phones store portrait video as landscape plus a rotation flag. FFmpeg turns the
        # frames upright when decoding, so the size everything works with is the displayed one.
        width, height = height, width
    return VideoMetadata(
        duration=duration,
        width=width,
        height=height,
        fps=round(fps, 3),
        avg_fps=round(_parse_frame_rate(video.get("avg_frame_rate")), 3),
        bitrate=int(fmt.get("bit_rate") or 0),
        video_codec=str(video.get("codec_name") or ""),
        audio_codec=str(audio.get("codec_name") or ""),
        audio_sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        rotation=rotation,
        pixel_format=str(video.get("pix_fmt") or ""),
        has_audio=bool(audio),
        size_bytes=size,
        filename=path.name,
    )


def parse_silence(stderr: str) -> list[dict]:
    starts = []
    ends = []
    for line in stderr.splitlines():
        if "silence_start:" in line:
            try:
                starts.append(float(line.split("silence_start:")[-1].strip().split()[0]))
            except ValueError:
                continue
        if "silence_end:" in line:
            try:
                chunk = line.split("silence_end:")[-1].strip().split()[0]
                ends.append(float(chunk))
            except ValueError:
                continue
    ranges = []
    for idx, start in enumerate(starts):
        end = ends[idx] if idx < len(ends) else start
        ranges.append({"start": start, "end": end, "duration": max(0.0, end - start)})
    return ranges


def _short_ffmpeg_error(text: str) -> str:
    skip = ("ffmpeg version", "built with", "configuration:", "libavutil", "libavcodec", "libavformat", "libavdevice", "libavfilter", "libswscale", "libswresample", "libpostproc")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) or stripped.startswith(prefix.strip()) for prefix in skip):
            continue
        if stripped.startswith("  lib"):
            continue
        lines.append(stripped)
    useful = lines[-12:] or [text.strip()[-500:]]
    return "\n".join(useful)
