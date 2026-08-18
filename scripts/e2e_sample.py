from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:18527"
CONFIG = {
    "min_duration": 20,
    "max_duration": 30,
    "style": "exciting",
    "preserve_context": "balanced",
    "remove_repetition": True,
    "allow_repeated_content": False,
    "max_sequence_duration": 5,
    "sequence_duration_mode": "auto",
    "content_priorities": ["balanced"],
    "max_repeated_duration": 15,
    "repetition_mode": "balanced",
    "order_mode": "chronological",
    "social_short": False,
    "aspect_ratio": "original",
    "subtitles": "none",
    "transition": "cut",
    "codec": "auto",
    "quality": "source",
    "resolution": "original",
}
CENTER_CROP = {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}


def _sample_videos() -> list[Path]:
    folder = ROOT / "samples"
    videos: list[Path] = []
    for ext in (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"):
        videos.extend(p for p in folder.glob(f"*{ext}") if ".multi-e2e" not in p.parts)
    videos.sort(key=lambda item: item.name.lower())
    return videos


def _resolve_sample() -> Path:
    args = [a for a in sys.argv[1:] if not a.startswith("http") and a not in {"--crop", "--multi"}]
    urls = [a for a in sys.argv[1:] if a.startswith("http")]
    global BASE
    if urls:
        BASE = urls[0]
    if "--crop" in sys.argv:
        CONFIG["crop"] = CENTER_CROP
    if args:
        path = Path(args[0])
        if path.is_file():
            return path
    folder = ROOT / "samples"
    preferred = folder / "sample.mp4"
    if preferred.is_file():
        return preferred
    videos = _sample_videos()
    if not videos:
        raise SystemExit(f"No sample video found in {folder}")
    return max(videos, key=lambda item: item.stat().st_mtime)


SAMPLE = _resolve_sample()
MULTI = "--multi" in sys.argv


def main() -> int:
    if not SAMPLE.is_file():
        print(f"Missing sample video: {SAMPLE}", file=sys.stderr)
        return 1
    uploads = [SAMPLE]
    if MULTI:
        folder_videos = _sample_videos()
        if len(folder_videos) >= 2:
            uploads = folder_videos[:8]
            print(f"Uploading {len(uploads)} sample videos to {BASE}")
            for item in uploads:
                print(f"  - {item.name}")
        else:
            split_dir = ROOT / "samples" / ".multi-e2e"
            uploads = _split_sample(SAMPLE, split_dir)
            print(f"Uploading {len(uploads)} clips split from {SAMPLE.name} to {BASE}")
    else:
        print(f"Uploading {SAMPLE.name} to {BASE}")
    if CONFIG.get("crop"):
        print(f"Crop enabled: {CONFIG['crop']}")
    before_sizes = {path.resolve(): path.stat().st_size for path in uploads}
    job = _upload(uploads)
    job_id = job["id"]
    sources = job.get("sources") or []
    print(
        f"Created job {job_id} status={job.get('status')} duration={job.get('metadata', {}).get('duration')} sources={len(sources)}"
    )
    if MULTI and len(sources) < 2:
        print("Expected multiple source videos on the job", file=sys.stderr)
        return 1
    job = _post(f"/api/jobs/{job_id}/analyze", CONFIG)
    print(f"Queued analysis status={job.get('status')}")
    job = _wait(job_id)
    print(f"Finished status={job.get('status')} error={job.get('error')!r} progress={job.get('progress')}")
    if job.get("status") == "planning" and job.get("analysis_status") == "completed":
        print("Analysis complete; starting render")
        job = _post(f"/api/jobs/{job_id}/render", {"clips": job.get("clips") or []})
        job = _wait(job_id)
        print(f"Render finished status={job.get('status')} error={job.get('error')!r}")
    if job.get("status") != "completed":
        logs = _get(f"/api/jobs/{job_id}/logs")
        for line in (logs.get("logs") or [])[-12:]:
            print(f"[{line.get('level')}] {line.get('message')}")
        return 1
    outputs = job.get("outputs") or []
    print(f"Highlight ready: {len(outputs)} output(s), {len(job.get('clips') or [])} clips")
    durations = []
    used_sources = set()
    for clip in job.get("clips") or []:
        start = float(clip.get("start") or 0)
        end = float(clip.get("end") or 0)
        durations.append(end - start)
        source = clip.get("source_name") or clip.get("source_id") or ""
        if source:
            used_sources.add(source)
        print(
            f"  {start:6.2f}–{end:6.2f}s ({end - start:.1f}s) "
            f"[{source or 'source?'}] {clip.get('reason') or ''}"
        )
        if clip.get("duration_reason"):
            print(f"           {clip.get('duration_reason')}")
    if MULTI:
        missing = [c for c in (job.get("clips") or []) if not (c.get("source_id") or c.get("source_name"))]
        if missing:
            print("Clips are missing source video labels", file=sys.stderr)
            return 1
        print(f"Used source videos: {', '.join(sorted(used_sources)) or 'none'}")
        if len(outputs) != 1:
            print("Expected a single combined highlight output", file=sys.stderr)
            return 1
    if CONFIG.get("sequence_duration_mode") == "auto" and durations:
        print("Auto sequence lengths: " + ", ".join(f"{item:.1f}s" for item in durations))
        if len(durations) >= 2 and all(abs(item - 5.0) < 0.05 for item in durations):
            print("Auto mode produced uniform 5s clips", file=sys.stderr)
            return 1
    if outputs:
        out = outputs[0]
        print(f"Output {out.get('width')}x{out.get('height')} {out.get('duration')}s {out.get('codec') or ''}")
        out_dur = float(out.get("duration") or 0)
        clip_total = sum(durations)
        if abs(out_dur - clip_total) > 3:
            print(
                f"Output duration {out_dur:.1f}s does not match selected clips {clip_total:.1f}s",
                file=sys.stderr,
            )
            return 1
        if out_dur > float(CONFIG["max_duration"]) + 2.5:
            print(
                f"Output {out_dur:.1f}s exceeds max duration {CONFIG['max_duration']}s",
                file=sys.stderr,
            )
            return 1
        if not CONFIG.get("crop"):
            width = int(out.get("width") or 0)
            height = int(out.get("height") or 0)
            expected_w = int((job.get("metadata") or {}).get("width") or 0)
            expected_h = int((job.get("metadata") or {}).get("height") or 0)
            if expected_w and width and (width < expected_w - 2 or height < expected_h - 2):
                print(
                    f"Unexpected downscale: output {width}x{height} from {expected_w}x{expected_h}",
                    file=sys.stderr,
                )
                return 1
        if CONFIG.get("crop"):
            width = int(out.get("width") or 0)
            height = int(out.get("height") or 0)
            if width >= 1920 or height >= 1080:
                print(f"Crop was ignored: output is still {width}x{height}", file=sys.stderr)
                return 1
            print("Crop applied: highlight is smaller than the original 1920x1080 sample")
        changed = [path.name for path in uploads if path.stat().st_size != before_sizes[path.resolve()]]
        if changed:
            print(f"Sample file(s) on disk changed; originals must stay untouched: {', '.join(changed)}", file=sys.stderr)
            return 1
        print(f"Download: {BASE}/api/jobs/{job['id']}/download")
        print(f"Review:   {BASE}/jobs/{job['id']}/result")
    return 0


def _wait(job_id: str, timeout: int = 900) -> dict:
    started = time.time()
    last = ""
    while time.time() - started < timeout:
        job = _get(f"/api/jobs/{job_id}")
        status = f"{job.get('status')} {job.get('last_stage')} {job.get('progress')}"
        if status != last:
            print(status)
            last = status
        if job.get("status") in {"completed", "failed", "cancelled", "planning"}:
            if job.get("status") != "planning" or job.get("analysis_status") == "completed":
                return job
        time.sleep(3)
    raise TimeoutError(f"Job {job_id} did not finish in {timeout}s")


def _upload(paths: list[Path] | None = None) -> dict:
    paths = paths or [SAMPLE]
    boundary = "----ReelMakerBoundary"
    body = _multipart(boundary, paths, json.dumps(CONFIG))
    req = urllib.request.Request(
        f"{BASE}/api/jobs",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return _send(req)


def _split_sample(sample: Path, dest_dir: Path) -> list[Path]:
    import subprocess

    dest_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(sample)
    first_len = min(22.0, max(8.0, duration * 0.4))
    second_start = min(duration * 0.55, max(first_len + 2.0, duration - first_len))
    part_a = dest_dir / "part-a.mp4"
    part_b = dest_dir / "part-b.mp4"
    _ffmpeg(["-i", str(sample), "-t", f"{first_len:.3f}", "-c", "copy", str(part_a)])
    _ffmpeg(["-ss", f"{second_start:.3f}", "-i", str(sample), "-c", "copy", str(part_b)])
    if part_a.stat().st_size < 1000 or part_b.stat().st_size < 1000:
        _ffmpeg(["-i", str(sample), "-t", f"{first_len:.3f}", "-c:v", "libx264", "-c:a", "aac", str(part_a)])
        _ffmpeg(["-ss", f"{second_start:.3f}", "-i", str(sample), "-c:v", "libx264", "-c:a", "aac", str(part_b)])
    return [part_a, part_b]


def _probe_duration(path: Path) -> float:
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip() or 0)


def _ffmpeg(args: list[str]) -> None:
    import subprocess

    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def _multipart(boundary: str, paths: list[Path], config: str) -> bytes:
    chunks = []
    for path in paths:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode()
        )
        chunks.append(b"Content-Type: video/mp4\r\n\r\n")
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(b'Content-Disposition: form-data; name="config"\r\n\r\n')
    chunks.append(config.encode())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    return _send(req)


def _get(path: str) -> dict:
    return _send(urllib.request.Request(f"{BASE}{path}"))


def _send(req: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"{exc.code} {req.full_url} {body}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
