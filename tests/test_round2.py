"""Second feature round: even sound, hook-first order, picture extras, music, translated
subtitles, watch folder, and the routes behind them."""

import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import create_app
from app.analysis.optimizer import HOOK_REASON, ClipOptimizer
from app.config import Settings
from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, RenderSettings, VideoMetadata
from app.services import translate, watch_folder
from app.video import audio_mix
from app.video.overlays import Extras, build_graph, output_size, punch_in_filters
from app.video.processor import VideoProcessor
from app.video.renderer import Renderer

HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------- even sound
def test_loudness_report_and_gain():
    report = 'noise\n{\n "input_i" : "-23.40",\n "input_tp" : "-3.1"\n}\n'
    assert audio_mix.parse_loudness(report) == -23.4
    assert audio_mix.parse_loudness("no json") is None
    assert audio_mix.parse_loudness('{"input_i": "-inf"}') is None
    assert audio_mix.leveling_gain(-23.4) == 9.4
    assert audio_mix.leveling_gain(-40) == 12.0  # boost is capped
    assert audio_mix.leveling_gain(-5) == -9.0
    assert audio_mix.leveling_gain(-70) is None and audio_mix.leveling_gain(None) is None
    assert audio_mix.leveling_filters(9.4) == ["volume=9.4dB", "alimiter=limit=0.891:level=disabled"]
    assert audio_mix.leveling_filters(0.05) == [] and audio_mix.leveling_filters(None) == []


class FakeRunner:
    def __init__(self, loudness=-24.0):
        self.commands = []
        self.loudness = loudness

    def run(self, cmd):
        self.commands.append(cmd)
        if "null" in cmd:
            return SimpleNamespace(stderr=f'{{"input_i": "{self.loudness}"}}')
        Path(cmd[-1]).write_bytes(b"x" * 2000)
        return SimpleNamespace(stderr="")

    def probe(self, path):
        return VideoMetadata(duration=6, width=1920, height=1080, fps=30, has_audio=True, size_bytes=2000)


def _render(tmp_path, settings, extras=None, clips=None, runner=None):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    runner = runner or FakeRunner()
    clips = clips or [ClipPlanItem(start=0, end=2, reason="a"), ClipPlanItem(start=4, end=6, reason="b")]
    import app.video.renderer as module

    original = module.validate_output
    module.validate_output = lambda *a, **k: {"ok": True, "error": None}
    try:
        result = Renderer(runner).render(
            source, ClipPlan(target_duration=4, estimated_duration=4, clips=clips), tmp_path / "out.mp4", settings, tmp_path / "w", extras=extras
        )
    finally:
        module.validate_output = original
    cuts = [c for c in runner.commands if Path(c[-1]).name.startswith("clip_")]
    return result, cuts


def test_render_levels_each_clip(tmp_path):
    result, cuts = _render(tmp_path, RenderSettings(level_audio=True))
    assert all("volume=10.0dB" in c[c.index("-af") + 1] for c in cuts)
    assert result["leveled"] == {"clips": 2, "of": 2}
    result, cuts = _render(tmp_path, RenderSettings())
    assert all("volume=" not in c[c.index("-af") + 1] for c in cuts) and result["leveled"] is None


# ---------------------------------------------------------------------------- strongest hook first
def _c(cid, start, score, hook):
    return {"id": cid, "start": start, "end": start + 4, "score": score, "text": f"unique words {cid} {start}",
            "topic": cid, "reason": "Key moment", "metrics": {"hook_strength": hook}}


def test_hook_first_order():
    candidates = [_c("a", 10, 0.9, 0.1), _c("b", 50, 0.7, 0.95), _c("c", 90, 0.8, 0.2)]
    plan = ClipOptimizer().select(candidates, min_duration=10, max_duration=14, order_mode="hook", max_sequence_duration=8)
    assert plan.clips[0].candidate_id == "b" and plan.clips[0].reason.startswith(HOOK_REASON)
    assert [c.start for c in plan.clips[1:]] == sorted(c.start for c in plan.clips[1:])
    plain = ClipOptimizer().select(candidates, min_duration=10, max_duration=14, max_sequence_duration=8)
    assert [c.start for c in plain.clips] == sorted(c.start for c in plain.clips)
    assert not any(c.reason.startswith(HOOK_REASON) for c in plain.clips)


# ---------------------------------------------------------------------------- picture extras
def test_output_size_matches_ffmpeg_crop_rounding():
    assert output_size(1280, 720, "9:16") == (404, 720)
    assert output_size(1920, 1080, "1:1") == (1080, 1080)
    assert output_size(1080, 1920, "16:9") == (1080, 606)
    assert output_size(1921, 1081, "original") == (1920, 1080)


def test_build_graph():
    assert build_graph("a,b", Extras(), 1280, 720, 30, 0, 10) == "a,b"
    logo = Path("C:/brand/logo.png")
    graph = build_graph("a,b", Extras(logo=logo, logo_position="top-left", progress_bar=True), 1280, 720, 30, 2.5, 10)
    assert graph.startswith("a,b[b0];movie='C\\:/brand/logo.png'")
    assert "[b0][logo]overlay=" in graph and "[b1][bar]overlay=x='-1280+1280*(2.500+t)/10.000'" in graph
    assert "scale=152:-1" in graph  # 12% of 1280, rounded down to even
    off = build_graph("a", Extras(logo=logo, logo_position="off"), 100, 100, 30, 0, 1)
    assert off == "a"
    assert punch_in_filters(404, 720)[1] == "scale=404:720:flags=lanczos"


def test_punch_in_alternates_by_moment_and_extras_reach_the_command(tmp_path):
    clips = [
        ClipPlanItem(start=0, end=2, reason="a"),
        ClipPlanItem(start=3, end=4, reason="a", join_previous=True),
        ClipPlanItem(start=6, end=8, reason="b"),
        ClipPlanItem(start=9, end=11, reason="c"),
    ]
    _, cuts = _render(tmp_path, RenderSettings(aspect_ratio="9:16"), Extras(punch_in=True, progress_bar=True), clips)
    graphs = [c[c.index("-vf") + 1] for c in cuts]
    closer = ["crop='trunc(iw/1.1" in g for g in graphs]
    assert closer == [False, False, True, False]  # moments 0 (two pieces), 1, 2
    assert all("[bar]overlay" in g for g in graphs)
    assert "(0.000+t)/7.000" in graphs[0] and "(3.000+t)/7.000" in graphs[2]


# ---------------------------------------------------------------------------- music
def test_music_command(tmp_path):
    cmd = audio_mix.music_mix_command(Path("v.mp4"), Path("m.mp3"), Path("o.mp4"), 20, 0.25, True, True)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "sidechaincompress" in graph and "volume=0.250" in graph and "afade=t=out:st=18.500" in graph
    assert cmd[cmd.index("-stream_loop") + 1] == "-1" and cmd[cmd.index("-c:v") + 1] == "copy"
    no_duck = audio_mix.music_mix_command(Path("v.mp4"), Path("m.mp3"), Path("o.mp4"), 20, 0.25, False, True)
    assert "sidechaincompress" not in no_duck[no_duck.index("-filter_complex") + 1]
    silent = audio_mix.music_mix_command(Path("v.mp4"), Path("m.mp3"), Path("o.mp4"), 20, 0.25, True, False)
    assert "[0:a]" not in silent[silent.index("-filter_complex") + 1]
    (tmp_path / "music").mkdir()
    assert audio_mix.find_music(tmp_path) is None
    (tmp_path / "music" / "music_song.mp3").write_bytes(b"x")
    assert audio_mix.find_music(tmp_path).name == "music_song.mp3"


# ---------------------------------------------------------------------------- translated subtitles
class FakeClient:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def role_options(self, role):
        return {"num_ctx": 4096}

    def chat(self, model, prompt, schema=None, options=None, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("offline")
        lines = [line.split(". ", 1)[1] for line in prompt.splitlines() if line[:1].isdigit() and ". " in line]
        return {"lines": [f"ES:{line}" for line in lines]}


def test_translation_only_for_kept_clips_and_cached(tmp_path):
    transcript = {"segments": [{"start": 0, "end": 2, "text": "Hello there"}, {"start": 10, "end": 12, "text": "Not used"}],
                  "words": [{"word": "Hello", "start": 0, "end": 1}]}
    cache = tmp_path / "t.json"
    client = FakeClient()
    out, done, needed = translate.translated_transcript(transcript, [(0, 3)], "Spanish", client, "m", cache)
    assert [s["text"] for s in out["segments"]] == ["ES:Hello there", "Not used"] and out["words"] == []
    assert (done, needed) == (1, 1)
    translate.translated_transcript(transcript, [(0, 3)], "Spanish", client, "m", cache)
    assert client.calls == 1  # second time from the cache
    out, done, _ = translate.translated_transcript(transcript, [(0, 3)], "French", FakeClient(fail=True), "m", tmp_path / "f.json")
    assert out["segments"][0]["text"] == "Hello there" and done == 0


def test_config_validation():
    config = HighlightConfig(subtitle_language="  Brazilian   Portuguese ", logo_size=99, music_volume=150, order_mode="hook")
    assert (config.subtitle_language, config.logo_size, config.music_volume) == ("Brazilian Portuguese", 40.0, 100.0)
    defaults = HighlightConfig()
    assert (defaults.level_audio, defaults.punch_in, defaults.progress_bar, defaults.logo_position, defaults.auto_render) == (False, False, False, "off", False)


# ---------------------------------------------------------------------------- app routes and watch folder
@pytest.fixture()
def client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9", enable_whisper=False)
    settings.ensure_directories()
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _job(client, config=None):
    data = {"file": (io.BytesIO(b"video"), "talk.mp4"), "config": json.dumps(config or {})}
    return client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()["id"]


def test_music_routes(client):
    job_id = _job(client)
    assert client.get(f"/api/jobs/{job_id}/music").get_json() == {"name": None}
    bad = client.post(f"/api/jobs/{job_id}/music", data={"file": (io.BytesIO(b"x"), "song.exe")}, content_type="multipart/form-data")
    assert bad.status_code == 400
    ok = client.post(f"/api/jobs/{job_id}/music", data={"file": (io.BytesIO(b"x"), "my song.mp3")}, content_type="multipart/form-data")
    assert ok.get_json()["name"] == "my song.mp3" == client.get(f"/api/jobs/{job_id}/music").get_json()["name"]
    odd = client.post(f"/api/jobs/{job_id}/music", data={"file": (io.BytesIO(b"x"), "../<bad>.FLAC")}, content_type="multipart/form-data")
    assert odd.get_json()["name"] == client.get(f"/api/jobs/{job_id}/music").get_json()["name"]
    assert odd.get_json()["name"].endswith(".flac") and "/" not in odd.get_json()["name"] and "<" not in odd.get_json()["name"]
    options = client.post(f"/api/jobs/{job_id}/render-options", json={"music_volume": 60, "music_duck": False, "style": "x"}).get_json()
    assert options == {"config": {"music_volume": 60.0, "music_duck": False, "narration": False}}
    assert client.get(f"/api/jobs/{job_id}").get_json()["config"]["style"] == "auto"
    assert client.delete(f"/api/jobs/{job_id}/music").get_json() == {"name": None}


def test_tighten_spans_route(client):
    from app.services.job_manager import get_job_manager

    job_id = _job(client, {"tighten": "pauses"})
    words = [{"word": "a", "start": 1.0, "end": 1.5}, {"word": "b", "start": 5.0, "end": 5.5}]
    (get_job_manager().job_dir(job_id) / "transcript.json").write_text(json.dumps({"words": words}), encoding="utf-8")
    data = client.post(f"/api/jobs/{job_id}/tighten-spans", json={"clips": [{"start": 0, "end": 10}, {"start": 3, "end": 2}]}).get_json()
    assert data["mode"] == "pauses" and data["spans"] == [[[0.88, 1.62], [4.88, 5.62]], []]
    other = _job(client)
    assert client.post(f"/api/jobs/{other}/tighten-spans", json={"clips": [{"start": 0, "end": 10}]}).get_json()["spans"] == [[[0, 10]]]


def test_srt_for_a_short(client):
    from app.services.job_manager import get_job_manager

    job_id = _job(client)
    outputs = get_job_manager().job_dir(job_id) / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "short_02.srt").write_bytes(b"1\n")
    assert client.get(f"/api/media/{job_id}/srt?short=2").data == b"1\n"
    assert client.get(f"/api/media/{job_id}/srt?short=1").status_code == 404
    assert client.get(f"/api/media/{job_id}/srt?short=x").status_code == 400


def test_logo_routes(client):
    assert client.get("/api/brand/logo").status_code == 404
    assert client.post("/api/brand/logo", data={"file": (io.BytesIO(b"x"), "a.gif")}, content_type="multipart/form-data").status_code == 400
    assert client.post("/api/brand/logo", data={"file": (io.BytesIO(b"png"), "Logo.PNG")}, content_type="multipart/form-data").status_code == 200
    assert client.get("/api/brand/logo").data == b"png"
    client.post("/api/brand/logo", data={"file": (io.BytesIO(b"jpg"), "b.jpg")}, content_type="multipart/form-data")
    assert client.get("/api/brand/logo").data == b"jpg"
    assert client.delete("/api/brand/logo").get_json() == {"logo": False}
    assert client.get("/api/brand/logo").status_code == 404


def test_watch_folder(client, tmp_path):
    from app.services.job_manager import get_job_manager

    assert client.get("/api/watch-folder").get_json() == {"enabled": False, "folder": "", "preset": ""}
    assert client.post("/api/watch-folder", json={"enabled": True, "folder": ""}).status_code == 400
    assert client.post("/api/watch-folder", json={"enabled": True, "folder": str(tmp_path / "nope")}).status_code == 400
    watched = tmp_path / "inbox"
    watched.mkdir()
    client.post("/api/presets", json={"name": "Quick", "form": {"style": "exciting", "min_duration": 15, "must_include": "x"}})
    assert client.post("/api/watch-folder", json={"enabled": True, "folder": str(watched), "preset": "Missing"}).status_code == 400
    saved = client.post("/api/watch-folder", json={"enabled": True, "folder": str(watched), "preset": "Quick"}).get_json()
    assert saved["enabled"] and saved["preset"] == "Quick"

    manager = get_job_manager()
    state = watch_folder.WatchState(tmp_path / "state.json")
    (watched / "trip.mp4").write_bytes(b"video data")
    (watched / "notes.txt").write_bytes(b"x")
    assert watch_folder.scan_once(manager, state) == []  # size not settled yet
    created = watch_folder.scan_once(manager, state)
    assert len(created) == 1
    job = client.get(f"/api/jobs/{created[0]}").get_json()
    assert job["status"] == "queued" and job["config"]["style"] == "exciting" and job["config"]["auto_render"] is True
    assert (watched / "trip.mp4").exists()  # the original stays
    assert watch_folder.scan_once(manager, state) == []  # handled once

    from app.workers.worker import _wants_auto_render

    assert _wants_auto_render(created[0]) is True
    out = manager.job_dir(created[0]) / "outputs"
    out.mkdir(parents=True)
    (out / "highlight.mp4").write_bytes(b"v")
    copied = watch_folder.copy_outputs(manager.job_dir(created[0]), [out / "highlight.mp4"])
    assert [p.name for p in copied] == ["trip-highlight.mp4"] and copied[0].parent == watched / "highlights"


# ---------------------------------------------------------------------------- real FFmpeg
@pytest.mark.skipif(not HAS_FFMPEG, reason="needs FFmpeg")
def test_filters_run_in_real_ffmpeg(tmp_path):
    def make(args, out):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args, str(out)], check=True)

    video = tmp_path / "in dir" / "in.mp4"
    video.parent.mkdir()
    make(["-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=2", "-f", "lavfi", "-i", "sine=f=440:duration=2",
          "-shortest", "-c:a", "aac"], video)
    logo = tmp_path / "brand dir" / "logo.png"
    logo.parent.mkdir()
    make(["-f", "lavfi", "-i", "color=c=red:s=200x100", "-frames:v", "1"], logo)
    music = tmp_path / "song.mp3"
    make(["-f", "lavfi", "-i", "sine=f=220:duration=1"], music)

    measured = subprocess.run(audio_mix.measure_command(video, 0, 2), capture_output=True, text=True)
    assert audio_mix.parse_loudness(measured.stderr) is not None

    width, height = output_size(640, 360, "9:16")
    chain = ",".join(["crop='if(gte(iw/ih,0.5625),ih*0.5625,iw)':'if(gte(iw/ih,0.5625),ih,iw/0.5625)'", "format=yuv420p", "setpts=PTS-STARTPTS"]
                     + punch_in_filters(width, height))
    graph = build_graph(chain, Extras(logo=logo, logo_position="bottom-right", progress_bar=True), width, height, 25, 1, 4)
    out = tmp_path / "clip.mp4"
    cmd = VideoProcessor.cut_clip_command(video, out, 0, 2, extra=["-vf", graph, "-af", "volume=3dB,alimiter=limit=0.891:level=disabled"])
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-400:]
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                       "-of", "json", str(out)], capture_output=True, text=True).stdout)["streams"][0]
    assert (probe["width"], probe["height"]) == (width, height)

    mixed = tmp_path / "mixed.mp4"
    result = subprocess.run(audio_mix.music_mix_command(out, music, mixed, 2, 0.3, True, True), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-400:]
    assert mixed.stat().st_size > 1000
