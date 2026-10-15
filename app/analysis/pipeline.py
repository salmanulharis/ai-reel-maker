from __future__ import annotations

import json
import os
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.ai.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaTruncatedError,
    describe_usage,
    predict_budget,
    usage_delta,
)
from app.ai.prompts import (
    TRANSCRIPT_BATCH_SCHEMA_FAST,
    picture_focus_texts,
    transcript_batch_prompt_fast,
    visual_analysis_focus,
    visual_focus_priorities,
)
from app.analysis import score_source
from app.analysis.audio import detect_silence, extract_audio, transcribe_detailed
from app.analysis import avoid as leave_out
from app.analysis.avoid import picture_avoid_texts, speech_avoid_text, vision_avoid_text
from app.analysis.candidates import generate_candidates, populate_candidate_audio_metrics, rescore_candidate
from app.analysis.optimizer import ClipOptimizer
from app.analysis.repetition import RepetitionAnalyzer
from app.analysis.scenes import detect_scenes
from app.analysis.scorer import HighlightScorer
from app.analysis.semantic import FOCUS_BOOST, Embedder, apply_focus, repetition_embedding_fn
from app.analysis.signals import ENERGY_HOP, audio_energy, scene_quality
from app.analysis.image_match import get_matcher, score_scenes
from app.analysis.visual import (
    analyze_scenes_visual,
    fitting_vision_batch,
    motion_score,
    sample_timestamps,
    shortlist_scenes,
)
from app.config import Settings
from app.models.schemas import CandidateScore, HighlightConfig, Scene, Transcript
from app.video.ffmpeg_runner import CancelledError, FFmpegRunner
from app.video.sources import save_sources, tag_candidates, working_path


STAGES = [
    ("metadata", "Video metadata"),
    ("scenes", "Scene detection"),
    ("audio", "Audio extraction"),
    ("transcript", "Transcription"),
    ("silence", "Silence detection"),
    ("frames", "Keyframe extraction"),
    ("content", "Finding important moments"),
    ("visual", "Visual AI analysis"),
    ("repetition", "Removing repetition"),
    ("planning", "Planning highlight"),
]

# Candidate moments kept for planning: the configured number, or more for long videos so
# their later parts are not crowded out (one per ~8 s), up to this ceiling.
_MAX_CANDIDATES_CEILING = 200
_SECONDS_PER_CANDIDATE = 8.0

# Upper bound on reply tokens per segment in a batched transcript-scoring reply. qwen3:4b
# pretty-prints the JSON and was measured at 42-46 tokens per segment; a lower budget cuts
# full batches off, and each cut-off reply is generated again as two halves.
_TEXT_TOKENS_PER_SEGMENT = 48


class PipelineProgress:
    def __init__(self, emit):
        self.emit = emit
        self.started = time.time()
        # Seconds spent in each stage that actually ran (cached stages are not counted).
        self.timings: dict[str, float] = {}
        self._running: dict[str, float] = {}
        # Model work per AI stage ("vision", "text"), summed over every source video.
        self.usage: dict[str, dict[str, float]] = {}

    def log(self, message: str, level: str = "INFO") -> None:
        self.emit("log", {"level": level, "message": message, "ts": time.time()})

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

    def stage(self, name: str, status: str, detail: str = "", index: int = 0, total: int = 1) -> None:
        self._time_stage(name, status)
        elapsed = time.time() - self.started
        known = [s[0] for s in STAGES]
        pos = known.index(name) if name in known else 0
        overall = ((pos + (0.5 if status == "running" else 1 if status == "done" else 0)) / max(len(known), 1)) * 90
        self.emit(
            "progress",
            {
                "stage": name,
                "status": status,
                "detail": detail,
                "index": index,
                "total": total,
                "overall": round(overall, 1),
                "elapsed": elapsed,
            },
        )


class VideoAnalyzer:
    def __init__(
        self,
        settings: Settings,
        runner: FFmpegRunner,
        ollama: OllamaClient | None,
        vision_enabled: bool = True,
        text_enabled: bool = True,
    ):
        self.settings = settings
        self.runner = runner
        self.ollama = ollama
        self.vision_enabled = vision_enabled
        self.text_enabled = text_enabled
        self.scorer = HighlightScorer()
        self.repetition = RepetitionAnalyzer()
        self.optimizer = ClipOptimizer()
        # Meaning-based repeat detection and focus matching; unavailable without Ollama or
        # without the embedding model, in which case word-based similarity is used as before.
        self.embedder = Embedder(ollama, getattr(settings, "ollama_embed_model", ""))

    def analyze(
        self,
        job_dir: Path,
        source: Path,
        config: HighlightConfig,
        progress: PipelineProgress,
        cache: dict,
        plan: bool = True,
    ) -> dict:
        meta = cache.get("metadata")
        if not meta:
            progress.stage("metadata", "running")
            meta = self.runner.probe(source).model_dump()
            _write(job_dir / "metadata.json", meta)
            progress.log(f"Video loaded ({meta.get('width')}x{meta.get('height')}, {meta.get('duration'):.1f}s)")
            progress.stage("metadata", "done")
        cache["metadata"] = meta
        source_duration = float(meta.get("duration") or 0)

        scenes = cache.get("scenes")
        if not scenes:
            progress.stage("scenes", "running")
            detected = detect_scenes(source, self.settings.scene_threshold, self.runner)
            scenes = [s.model_dump() for s in detected]
            _write(job_dir / "scenes.json", scenes)
            progress.log(f"{len(scenes)} scenes detected")
            progress.stage("scenes", "done")
        cache["scenes"] = scenes

        wav = job_dir / "source" / "audio.wav"
        # Speech can be turned off for the whole app or for one highlight. Audio is still
        # extracted when the video has sound: loudness and pauses come from it.
        want_speech = bool(self.settings.enable_whisper) and bool(getattr(config, "use_speech", True))
        if not wav.exists() and meta.get("has_audio", True):
            progress.stage("audio", "running")
            try:
                extract_audio(self.runner, source, wav)
                progress.log("Audio extracted")
            except CancelledError:
                # A cancel stops FFmpeg mid-file; a partial file would be reused as finished audio.
                wav.unlink(missing_ok=True)
                raise
            except Exception as exc:  # noqa: BLE001
                wav.unlink(missing_ok=True)
                progress.log(f"Audio extraction failed: {exc}", "WARNING")
            progress.stage("audio", "done")
        elif not wav.exists():
            progress.stage("audio", "skipped", "No sound in this video")
        else:
            progress.stage("audio", "done", "cached")

        transcript_data = cache.get("transcript")
        if not want_speech:
            # Not written to disk, so turning speech back on transcribes (or reuses) as usual.
            transcript_data = Transcript().model_dump()
            progress.stage("transcript", "skipped", "Speech is turned off")
            progress.log("Speech is turned off: moments are chosen from pictures and sound")
        elif transcript_data is None:
            progress.stage("transcript", "running")
            transcribed = True
            if wav.exists():
                transcript_data, transcribed = self._transcribe(wav, progress)
            else:
                # Not cached: if the audio can be extracted on a later run, it is transcribed then.
                transcribed = False
                transcript_data = Transcript().model_dump()
                progress.log("No transcript available; continuing with visual/scene analysis", "WARNING")
            if transcribed:
                _write(job_dir / "transcript.json", transcript_data)
            progress.stage(
                "transcript", "done", "" if transcribed or not wav.exists() else "Transcription failed"
            )
        cache["transcript"] = transcript_data
        transcript = Transcript.model_validate(transcript_data)

        silences = cache.get("silences")
        if silences is None:
            progress.stage("silence", "running")
            target = wav if wav.exists() else source
            silences = detect_silence(
                self.runner, target, self.settings.silence_threshold_db, self.settings.silence_min_duration
            )
            _write(job_dir / "silences.json", silences)
            progress.stage("silence", "done")
        cache["silences"] = silences

        frames_dir = job_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        frame_map = cache.get("frames") or {}
        has_frame_files = any(
            Path(p["image_path"]).exists()
            for paths in frame_map.values()
            for p in paths
        )
        if not has_frame_files:
            progress.stage("frames", "running")
            use_lite_frames = self.settings.enable_visual_analysis and bool(self.ollama) and self.vision_enabled
            frame_density = "lite" if use_lite_frames else "auto"
            frame_map = self._extract_frames(source, scenes, frames_dir, source_duration, frame_density, progress)
            _write(job_dir / "frames.json", frame_map)
            progress.log(f"Representative frames sampled ({sum(len(v) for v in frame_map.values())} frames)")
            progress.stage("frames", "done")
        cache["frames"] = frame_map
        quality = self._scene_quality(job_dir, scenes, frame_map, cache)
        energy = self._audio_energy(job_dir, wav, cache)
        if config.audio_focus == "natural" and not energy:
            progress.log("Natural-sound focus has no loudness data; ranking will use the other available signals.", "WARNING")
        elif config.audio_focus == "dialogue" and not (transcript.words or transcript.segments):
            progress.log("Dialogue focus has no transcript data; ranking will use the other available signals.", "WARNING")

        match = self._image_match(job_dir, scenes, frame_map, config, cache, progress)
        signals = _visual_signals(scenes, frame_map, quality, energy, match)

        if want_speech:
            progress.stage("content", "running")
            usage_before = self._usage()
            segment_scores, scores_changed = self._score_transcript(
                transcript, config, progress, cache.get("segment_scores")
            )
            self._log_usage(progress, "Transcript AI", usage_before)
            if scores_changed:
                _write(job_dir / "segment_scores.json", segment_scores)
            progress.stage("content", "done")
        else:
            # Saved speech scores stay on disk for when speech is turned back on.
            segment_scores = []
            progress.stage("content", "skipped", "Speech is turned off")
        cache["segment_scores"] = segment_scores

        # Priorities that change the vision prompt; their scores are redone when these change.
        vision_avoid = vision_avoid_text(config.avoid, config.avoid_custom)
        connection_focus = visual_focus_priorities(config.content_priorities)
        if vision_avoid:
            connection_focus = connection_focus + [f"avoid:{vision_avoid}"]
        if _stored_visual_focus(cache.get("visual_connection_focus")) != connection_focus:
            cache["visual"] = {}
        visual = dict(cache.get("visual") or {})
        ollama_ok = bool(self.ollama)
        vision_ok = ollama_ok and self.vision_enabled
        vision_ready = vision_ok and self.settings.enable_visual_analysis
        pending_scenes = [
            scene
            for scene in scenes
            if str(scene["scene_id"]) not in visual
            or (vision_ready and score_source.needs_ai(visual[str(scene["scene_id"])]))
        ]
        if pending_scenes or not visual:
            progress.stage("visual", "running")
            if ollama_ok and not vision_ok:
                progress.log(
                    f"Vision model '{self.settings.ollama_vision_model}' is not available. Using deterministic visual scoring.",
                    "WARNING",
                )
            elif not ollama_ok:
                progress.log("Ollama is unavailable. Using deterministic visual scoring.", "WARNING")
            shortlist = None
            if vision_ready and str(getattr(self.settings, "vision_scope", "shortlist")) != "all":
                shortlist = shortlist_scenes(
                    scenes,
                    segment_scores,
                    _highlight_seconds(config, source_duration),
                    quality,
                    visual_signal=signals,
                    match=match,
                )
            if vision_ready and config.avoid_mode == "never":
                # Scenes that will be left out anyway are not worth the vision model's time.
                left_out = _never_scenes(match, config)
                if left_out:
                    every = {int(scene["scene_id"]) for scene in scenes}
                    shortlist = (every if shortlist is None else set(shortlist)) - left_out
                    progress.log(f"Leave out: {len(left_out)} scene(s) skipped by the vision model")
            ai_scene_count = sum(
                1
                for scene in pending_scenes
                if float(scene.get("duration") or 0) >= self.settings.ollama_vision_min_scene_duration
                and (shortlist is None or int(scene["scene_id"]) in shortlist)
            )
            if visual and pending_scenes and ai_scene_count:
                progress.log(f"Re-running visual AI for {ai_scene_count} scene(s) that used fallback scores")
            if shortlist is not None:
                progress.log(
                    f"Visual AI: {len(shortlist)} of {len(scenes)} scenes could make the highlight; "
                    "the others use quick motion and picture-quality scoring"
                )
            vision_batch = self.settings.ollama_vision_batch_size
            if self.ollama is not None:
                vision_batch = fitting_vision_batch(
                    vision_batch,
                    self.ollama.role_options("vision")["num_ctx"],
                    self.settings.ollama_vision_num_predict,
                )
            batch_count = max(1, (ai_scene_count + vision_batch - 1) // max(1, vision_batch))
            progress.log(
                f"Visual AI: analyzing {ai_scene_count} scene(s) in up to {batch_count} batched call(s) "
                f"({self.settings.ollama_vision_max_pixels}px frames, model {self.settings.ollama_vision_model})"
            )
            progress.stage("visual", "running", "Preparing batched visual analysis", 0, batch_count)
            self._check_cancel()

            def _visual_batch_progress(batch_no: int, total: int, size: int) -> None:
                self._check_cancel()
                progress.log(f"Visual AI: batch {batch_no}/{total} ({size} scene{'s' if size != 1 else ''})")
                progress.stage(
                    "visual",
                    "running",
                    f"Visual batch {batch_no}/{total}",
                    batch_no,
                    total,
                )

            usage_before = self._usage()
            results, errors, fallback_reasons = analyze_scenes_visual(
                self.ollama,
                self.settings.ollama_vision_model,
                pending_scenes,
                frame_map,
                source_duration,
                visual_analysis_focus(config.custom_focus, config.content_priorities),
                enabled=self.settings.enable_visual_analysis and vision_ok,
                settings=self.settings,
                on_batch_start=_visual_batch_progress,
                shortlist=shortlist,
                avoid=vision_avoid,
            )
            for scene in pending_scenes:
                scene_id = int(scene["scene_id"])
                if scene_id in results:
                    visual[str(scene_id)] = score_source.tag(
                        results[scene_id].model_dump(), fallback_reasons.get(scene_id)
                    )
            for message in errors:
                progress.log(message, "WARNING")
            self._log_usage(progress, "Visual AI", usage_before)
            progress.stage("visual", "done", f"Analyzed {len(results)} scene(s)", batch_count, batch_count)
            _write(job_dir / "visual_analysis.json", visual)
            progress.log("Visual analysis completed")
        cache["visual"] = visual
        cache["visual_connection_focus"] = connection_focus
        _write(job_dir / "visual_connection_focus.json", connection_focus)

        visual_by_scene = _with_image_match({int(k): v for k, v in visual.items()}, match)
        candidates = generate_candidates(
            transcript,
            scenes,
            visual_by_scene,
            segment_scores,
            silences,
            source_duration,
            config,
            self.scorer,
            _candidate_limit(self.settings.maximum_candidates, source_duration),
            scene_quality=quality,
            energy=energy,
        )
        progress.log(f"{len(candidates)} candidates generated")
        picture_focus = picture_focus_texts(config.custom_focus, config.content_priorities)
        if picture_focus and match:
            matched = _apply_visual_focus(candidates, scenes, match)
            if matched:
                progress.log(f"Focus: {matched} moment(s) look like \"{'; '.join(picture_focus)[:80]}\"")
        if config.custom_focus.strip():
            boosted = apply_focus(candidates, config.custom_focus, self.embedder)
            if boosted:
                progress.log(f"Focus: {boosted} moment(s) match \"{config.custom_focus.strip()[:60]}\"")
            elif not self.embedder.available:
                progress.log(
                    f"Focus matching needs the embedding model '{self.settings.ollama_embed_model}'. "
                    "Install it in Ollama for better results; the focus still guides the AI prompts."
                )
        if config.avoid or config.avoid_custom:
            leave_out.annotate(
                candidates,
                config.avoid,
                config.avoid_custom,
                scenes,
                visual_by_scene,
                match,
                source_duration,
                self.embedder if config.avoid_custom else None,
            )
        cache["candidates"] = candidates
        if not plan:
            return cache

        progress.stage("repetition", "running")
        self.repetition.embedding_fn = repetition_embedding_fn(self.embedder)
        similarity_fn = None
        if self.settings.enable_repetition_analysis and candidates:
            similarity_fn = self.repetition.as_similarity_fn(candidates)
        # Kept before repeats are removed, so re-planning with a lighter setting can use them.
        _write(job_dir / "candidates_all.json", candidates)
        cache["candidates_all"] = candidates
        candidates = _leave_out(candidates, config, progress)
        if config.remove_repetition and self.settings.enable_repetition_analysis:
            before = len(candidates)
            threshold = self.repetition.threshold_for(config.repetition_mode)
            candidates = self.repetition.deduplicate(candidates, threshold=threshold)
            progress.log(f"{before - len(candidates)} duplicate candidates removed")
        elif config.allow_repeated_content:
            progress.log("Repeated content allowed; unique moments will still be preferred first.")
        _write(job_dir / "candidates.json", candidates)
        cache["candidates"] = candidates
        progress.stage("repetition", "done")

        progress.stage("planning", "running")
        plan = self.optimizer.select(
            candidates,
            min_duration=config.min_duration,
            max_duration=config.max_duration,
            must_include=config.must_include,
            must_exclude=config.must_exclude,
            order_mode=config.order_mode,
            source_duration=source_duration,
            similarity_fn=similarity_fn,
            max_sequence_duration=config.max_sequence_duration,
            min_sequence_duration=config.min_sequence_duration,
            sequence_duration_mode=config.sequence_duration_mode,
            allow_repeated_content=bool(config.allow_repeated_content),
            max_repeated_duration=config.max_repeated_duration,
            repeat_threshold=self.repetition.threshold_for(config.repetition_mode),
            content_priorities=config.content_priorities,
            length_mode=config.length_mode,
            auto_length_strict=config.auto_length_strict,
            style=config.style,
        )
        plan_data = plan.model_dump()
        _write(job_dir / "clip_plan.json", plan_data)
        cache["clip_plan"] = plan_data
        progress.log("Final plan generated")
        progress.stage("planning", "done")
        if not self.settings.debug_analysis:
            _cleanup_frames(frames_dir)
        return cache

    def analyze_sources(
        self,
        job_dir: Path,
        sources: list[dict],
        config: HighlightConfig,
        progress: PipelineProgress,
        cache: dict,
    ) -> dict:
        if not sources:
            raise ValueError("No source videos")
        all_candidates: list[dict] = []
        total_duration = 0.0
        multi = len(sources) > 1
        combined_meta: dict = cache.get("metadata") or {}
        for index, source in enumerate(sources):
            label = source.get("filename") or f"video {index + 1}"
            if multi:
                progress.log(f"Analyzing {label} ({index + 1}/{len(sources)})")
            work = job_dir / "work" / str(source.get("id") or f"v{index}") if multi else job_dir
            work.mkdir(parents=True, exist_ok=True)
            (work / "source").mkdir(exist_ok=True)
            if multi:
                from app.workers.worker import load_cache

                piece_cache = load_cache(work)
            else:
                piece_cache = dict(cache)
            piece = self.analyze(work, working_path(source), config, progress, piece_cache, plan=False)
            tagged = tag_candidates(piece.get("candidates") or [], source)
            all_candidates.extend(tagged)
            duration = float((piece.get("metadata") or {}).get("duration") or 0)
            total_duration += duration
            source["duration"] = duration
            source["width"] = int((piece.get("metadata") or {}).get("width") or 0)
            source["height"] = int((piece.get("metadata") or {}).get("height") or 0)
            source["fps"] = float((piece.get("metadata") or {}).get("fps") or 0)
            progress.log(f"{label}: {len(tagged)} candidate moments")

        best = max(sources, key=lambda item: int(item.get("width") or 0) * int(item.get("height") or 0))
        combined_meta = {
            **combined_meta,
            "duration": total_duration,
            "filename": ", ".join(str(s.get("filename") or "") for s in sources),
            "width": int(best.get("width") or 0),
            "height": int(best.get("height") or 0),
            "fps": float(best.get("fps") or 0),
            "source_count": len(sources),
        }
        _write(job_dir / "metadata.json", combined_meta)
        cache["metadata"] = combined_meta
        cache["sources"] = sources
        save_sources(job_dir, sources)

        progress.stage("repetition", "running")
        self.repetition.embedding_fn = repetition_embedding_fn(self.embedder)
        similarity_fn = None
        candidates = all_candidates
        if self.settings.enable_repetition_analysis and candidates:
            similarity_fn = self.repetition.as_similarity_fn(candidates)
        # Kept before repeats are removed, so re-planning with a lighter setting can use them.
        _write(job_dir / "candidates_all.json", candidates)
        cache["candidates_all"] = candidates
        candidates = _leave_out(candidates, config, progress)
        if config.remove_repetition and self.settings.enable_repetition_analysis:
            before = len(candidates)
            threshold = self.repetition.threshold_for(config.repetition_mode)
            candidates = self.repetition.deduplicate(candidates, threshold=threshold)
            progress.log(f"{before - len(candidates)} duplicate moments removed across all videos")
        _write(job_dir / "candidates.json", candidates)
        cache["candidates"] = candidates
        progress.stage("repetition", "done")

        progress.stage("planning", "running")
        plan = self.optimizer.select(
            candidates,
            min_duration=config.min_duration,
            max_duration=config.max_duration,
            must_include=config.must_include,
            must_exclude=config.must_exclude,
            order_mode=config.order_mode,
            source_duration=total_duration,
            multi_source=multi,
            similarity_fn=similarity_fn,
            max_sequence_duration=config.max_sequence_duration,
            min_sequence_duration=config.min_sequence_duration,
            sequence_duration_mode=config.sequence_duration_mode,
            allow_repeated_content=bool(config.allow_repeated_content),
            max_repeated_duration=config.max_repeated_duration,
            repeat_threshold=self.repetition.threshold_for(config.repetition_mode),
            content_priorities=config.content_priorities,
            length_mode=config.length_mode,
            auto_length_strict=config.auto_length_strict,
            style=config.style,
        )
        used = {c.source_id for c in plan.clips if c.source_id}
        if used and len(sources) > 1:
            skipped = [s.get("filename") for s in sources if s.get("id") not in used]
            if skipped:
                progress.log(
                    "Skipped videos with no unique high-value moments: "
                    + ", ".join(str(n) for n in skipped if n)
                )
        plan_data = plan.model_dump()
        _write(job_dir / "clip_plan.json", plan_data)
        cache["clip_plan"] = plan_data
        progress.log(f"Combined plan uses {len(plan.clips)} clips from {len(used) or 1} video(s)")
        progress.stage("planning", "done")
        summary = progress.timing_summary()
        if summary:
            progress.log(f"Stage timings: {summary}")
            _write(job_dir / "analysis_timings.json", progress.timings)
        return cache

    def _scene_quality(self, job_dir: Path, scenes: list[dict], frame_map: dict, cache: dict) -> dict[int, dict]:
        """Picture quality per scene from its sampled frames; cached with the analysis."""
        cached = cache.get("scene_quality")
        if isinstance(cached, dict) and cached:
            return {int(k): v for k, v in cached.items()}
        quality: dict[int, dict] = {}
        try:
            for scene in scenes:
                paths = [entry["image_path"] for entry in frame_map.get(str(scene["scene_id"]), [])]
                if paths:
                    quality[int(scene["scene_id"])] = scene_quality(paths)
        except Exception:  # noqa: BLE001 - quality is an extra signal, never a requirement
            return {}
        if quality:
            _write(job_dir / "scene_quality.json", {str(k): v for k, v in quality.items()})
            cache["scene_quality"] = {str(k): v for k, v in quality.items()}
        return quality

    def _image_match(
        self,
        job_dir: Path,
        scenes: list[dict],
        frame_map: dict,
        config: HighlightConfig,
        cache: dict,
        progress: PipelineProgress,
    ) -> dict[int, dict]:
        """SigLIP interest, focus match, and lookalikes per scene; cached with the focus text."""
        if not getattr(self.settings, "image_matching", True):
            return {}
        texts = picture_focus_texts(config.custom_focus, config.content_priorities)
        # The cache key is the plain focus text when there is only that (as in older versions).
        focus = " | ".join(texts)
        avoid_texts = picture_avoid_texts(config.avoid, config.avoid_custom)
        avoid_key = sorted(f"{key}:{'|'.join(value[0])}" for key, value in avoid_texts.items())
        cached = cache.get("image_match")
        if (
            isinstance(cached, dict)
            and cached.get("focus") == focus
            and (cached.get("avoid") or []) == avoid_key
            and cached.get("scenes")
        ):
            return {int(k): v for k, v in cached["scenes"].items()}
        frames = {}
        for scene in scenes:
            entries = frame_map.get(str(scene["scene_id"])) or []
            if entries:
                frames[int(scene["scene_id"])] = entries[len(entries) // 2]["image_path"]
        if not frames:
            return {}
        matcher = get_matcher(int(getattr(self.settings, "ollama_num_thread", 0) or 0))
        if matcher is None:
            return {}
        started = time.perf_counter()
        try:
            result = score_scenes(matcher, frames, texts, avoid_texts)
        except Exception as exc:  # noqa: BLE001 - a ranking aid, never a requirement
            progress.log(f"Image matching skipped ({exc})", "WARNING")
            return {}
        duplicates = sum(1 for item in result.values() if item.get("duplicate_of") is not None)
        progress.log(
            f"Image matching ({matcher.device}): {len(result)} scenes in {time.perf_counter() - started:.1f}s"
            + (f", {duplicates} look like an earlier scene" if duplicates else "")
        )
        payload = {"focus": focus, "avoid": avoid_key, "scenes": {str(k): v for k, v in result.items()}}
        _write(job_dir / "image_match.json", payload)
        cache["image_match"] = payload
        return result

    def _audio_energy(self, job_dir: Path, wav: Path, cache: dict) -> list[float]:
        """Loudness curve of the extracted audio; cached with the analysis."""
        cached = cache.get("audio_energy")
        if isinstance(cached, list) and cached:
            return cached
        energy = audio_energy(wav) if wav.exists() else []
        if energy:
            _write(job_dir / "audio_energy.json", energy)
            cache["audio_energy"] = energy
        return energy

    def _extract_frames(
        self,
        source: Path,
        scenes: list[dict],
        frames_dir: Path,
        source_duration: float,
        density: str,
        progress: PipelineProgress,
    ) -> dict[str, list[dict]]:
        """Sample frames for every scene. Each frame is a separate short FFmpeg call that is
        mostly process start-up and seeking, so a few run at once. The result keeps scene and
        timestamp order regardless of which call finishes first."""
        planned: list[tuple[str, float, Path]] = []
        for scene in scenes:
            model = Scene.model_validate(scene)
            for stamp in sample_timestamps(model, source_duration=source_duration, density=density):
                out = frames_dir / f"scene_{model.scene_id}_{int(stamp * 100)}.jpg"
                planned.append((str(model.scene_id), stamp, out))

        def extract(stamp: float, out: Path) -> str | None:
            if out.exists():
                return None
            try:
                self.runner.extract_frame(source, out, stamp, duration=source_duration)
            except CancelledError:
                # A stopped command can leave a partial image that a later run would reuse.
                out.unlink(missing_ok=True)
                raise
            except Exception as ext:  # noqa: BLE001
                return f"Frame extract failed at {stamp:.2f}s: {ext}"
            return None

        # Close timestamps can round to the same file name; grab each file once.
        unique: dict[Path, float] = {}
        for _, stamp, out in planned:
            unique.setdefault(out, stamp)
        workers = frame_extract_workers(len(unique))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="frames") as pool:
                outcomes = list(pool.map(lambda item: extract(item[1], item[0]), unique.items()))
        else:
            outcomes = [extract(stamp, out) for out, stamp in unique.items()]
        self._check_cancel()
        failures = dict(zip(unique, outcomes))

        frame_map: dict[str, list[dict]] = {str(Scene.model_validate(scene).scene_id): [] for scene in scenes}
        reported: set[Path] = set()
        for scene_id, stamp, out in planned:
            failure = failures.get(out)
            if failure:
                if out not in reported:
                    reported.add(out)
                    progress.log(failure, "WARNING")
                continue
            if out.exists() and out.stat().st_size > 0:
                frame_map[scene_id].append({"timestamp": stamp, "image_path": str(out)})
        return frame_map

    def _transcribe(self, wav: Path, progress: PipelineProgress) -> tuple[dict, bool]:
        """Transcript data, and whether it is worth caching. A failed run is not cached, so the
        next analysis tries again instead of reusing an empty transcript."""
        try:
            transcript, info = transcribe_detailed(
                wav,
                self.settings.whisper_model,
                self.settings.whisper_device,
                cpu_threads=self.settings.whisper_cpu_threads,
                num_workers=self.settings.whisper_num_workers,
            )
        except Exception as exc:  # noqa: BLE001
            transcript, info = Transcript(), {"ok": False, "error": str(exc)}
        if not info.get("ok"):
            progress.log(
                f"Transcription failed: {info.get('error') or 'unknown error'}. "
                "Continuing without speech; the next analysis will try again.",
                "WARNING",
            )
            return Transcript().model_dump(), False
        if info.get("warning"):
            progress.log(str(info["warning"]), "WARNING")
        nseg = len(transcript.segments)
        progress.log(f"Transcription completed ({nseg} segment{'s' if nseg != 1 else ''})")
        return transcript.model_dump(), True

    def _score_transcript(
        self,
        transcript: Transcript,
        config: HighlightConfig,
        progress: PipelineProgress,
        existing: list[dict] | None = None,
    ) -> tuple[list[dict], bool]:
        """Score every transcript segment, reusing cached scores.

        Cached fallback scores are redone when the text model is ready. Returns the scores and
        whether they differ from ``existing``."""
        segments = transcript.segments
        if not segments:
            return [], existing != []
        cached = existing if isinstance(existing, list) and len(existing) == len(segments) else None
        text_ready = self.ollama is not None and self.text_enabled
        results: list[dict | None] = list(cached) if cached is not None else [None] * len(segments)
        pending = [
            index
            for index, item in enumerate(results)
            if item is None or (text_ready and score_source.needs_ai(item))
        ]
        if not pending:
            return results, False
        if not text_ready:
            if self.ollama is None:
                progress.log("Text model unavailable; using heuristic transcript scores.", "WARNING")
            for index in pending:
                results[index] = _heuristic_segment(segments[index], score_source.AI_UNAVAILABLE)
            return results, True
        if cached is not None:
            progress.log(f"Re-scoring {len(pending)} transcript segment(s) that used fallback scores")

        batches = plan_text_batches(
            [segments[index].text for index in pending],
            max_items=self.settings.ollama_text_batch_size,
            fixed_tokens=estimate_tokens(transcript_batch_prompt_fast(
                    config.style, config.custom_focus, [], speech_avoid_text(config.avoid, config.avoid_custom)
                )),
            configured_predict=self.settings.ollama_text_num_predict,
            per_item_output=_TEXT_TOKENS_PER_SEGMENT,
            num_ctx=self.ollama.role_options("text")["num_ctx"],
        )
        batches = [[pending[position] for position in batch] for batch in batches]
        unreachable = False
        for batch_no, indexes in enumerate(batches, start=1):
            self._check_cancel()
            batch = [segments[index] for index in indexes]
            progress.log(
                f"Scoring transcript batch {batch_no}/{len(batches)} "
                f"({len(batch)} segment{'s' if len(batch) != 1 else ''})"
            )
            progress.stage(
                "content",
                "running",
                f"Scoring speech {batch_no}/{len(batches)}",
                batch_no,
                len(batches),
            )
            if unreachable:
                scored = [_heuristic_segment(seg, score_source.AI_FAILED) for seg in batch]
            else:
                try:
                    scored = self._score_text_batch(batch, config)
                except CancelledError:
                    raise
                except OllamaConnectionError as exc:
                    unreachable = True
                    progress.log(
                        f"Ollama stopped responding ({exc}); using heuristic scores for the rest of the transcript.",
                        "WARNING",
                    )
                    scored = [_heuristic_segment(seg, score_source.AI_FAILED) for seg in batch]
                except Exception as exc:  # noqa: BLE001
                    progress.log(f"Transcript batch scoring failed ({exc}); using fallback.", "WARNING")
                    scored = [_heuristic_segment(seg, score_source.AI_FAILED) for seg in batch]
            for index, item in zip(indexes, scored):
                results[index] = item
        return results, True

    def _score_text_batch(self, batch: list, config: HighlightConfig) -> list[dict]:
        """AI scores for one batch. A reply cut off by the token limit is retried as two smaller batches."""
        self._check_cancel()
        payload = [{"start": s.start, "end": s.end, "text": s.text} for s in batch]
        options = self.ollama.role_options("text")
        options["num_predict"] = predict_budget(
            self.settings.ollama_text_num_predict, len(batch), _TEXT_TOKENS_PER_SEGMENT, options["num_ctx"]
        )
        try:
            data = self.ollama.chat(
                self.settings.ollama_text_model,
                transcript_batch_prompt_fast(
                    config.style, config.custom_focus, payload, speech_avoid_text(config.avoid, config.avoid_custom)
                ),
                schema=TRANSCRIPT_BATCH_SCHEMA_FAST,
                options=options,
                schema_first=False,
            )
        except OllamaTruncatedError:
            if len(batch) <= 1:
                raise
            middle = len(batch) // 2
            return self._score_text_batch(batch[:middle], config) + self._score_text_batch(batch[middle:], config)
        scored = []
        for seg, raw in zip(batch, align_segment_scores(data.get("segments"), len(batch))):
            if raw is None:
                scored.append(_heuristic_segment(seg, score_source.AI_FAILED))
                continue
            score = _expand_segment_score(raw, seg)
            scored.append(
                score_source.tag(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "score": score.model_dump(),
                        "reason": self.scorer.reason_for(score),
                    }
                )
            )
        return scored

    def _usage(self) -> dict | None:
        snapshot = getattr(self.ollama, "usage_snapshot", None)
        return snapshot() if callable(snapshot) else None

    def _log_usage(self, progress: PipelineProgress, label: str, before: dict | None) -> None:
        """Report the model work of one stage, so slow stages can be traced to prompt size,
        reply length, or model loading."""
        after = self._usage()
        if before is None or after is None:
            return
        usage = usage_delta(before, after)
        text = describe_usage(usage)
        if not text:
            return
        key = "vision" if label.startswith("Visual") else "text"
        total = progress.usage.setdefault(key, {})
        for name, value in usage.items():
            total[name] = total.get(name, 0) + value
        progress.log(f"{label}: {text}")
        if usage["near_context_limit"]:
            progress.log(
                f"{label}: {int(usage['near_context_limit'])} call(s) filled the model's context window. "
                "A larger context size in Settings avoids cut-off prompts and replies.",
                "WARNING",
            )

    def _check_cancel(self) -> None:
        if self.runner.cancel_event.is_set():
            raise CancelledError("Job cancelled")


def plan_from_cache(cache: dict, config: HighlightConfig, settings: Settings) -> dict:
    """A new plan from the cached candidates under changed options (the review page's adjust
    buttons). Scores are recomputed with the same adjustments as the first analysis."""
    # The list from before repeat removal, when there is one, so a lighter repeat setting can
    # bring back moments that a stricter one removed.
    candidates = [dict(c) for c in (cache.get("candidates_all") or cache.get("candidates") or [])]
    populate_candidate_audio_metrics(
        candidates,
        Transcript.model_validate(cache.get("transcript") or {}),
        cache.get("audio_energy"),
    )
    scorer = HighlightScorer()
    for cand in candidates:
        metrics = CandidateScore.model_validate(cand.get("metrics") or {})
        score = rescore_candidate(cand, scorer, config)
        visual_focus = cand.get("visual_focus_match")
        if visual_focus is not None and float(visual_focus) > _VISUAL_FOCUS_FROM:
            strength = (float(visual_focus) - _VISUAL_FOCUS_FROM) / (1 - _VISUAL_FOCUS_FROM)
            score = min(1.0, score + _VISUAL_FOCUS_BOOST * strength)
        score = min(1.0, score + FOCUS_BOOST * float(cand.get("focus_match") or 0))
        cand["score"] = round(score, 4)
        if not cand.get("score_fixed"):
            cand["reason"] = scorer.reason_for(metrics, cand.get("reason") or "Meaningful moment")
    candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
    candidates = _leave_out(candidates, config, None)
    embedding_fn = repetition_embedding_fn(_plan_embedder(settings))
    if config.remove_repetition:
        analyzer = RepetitionAnalyzer(embedding_fn)
        candidates = analyzer.deduplicate(candidates, threshold=analyzer.threshold_for(config.repetition_mode))
    analyzer = RepetitionAnalyzer(embedding_fn)
    similarity_fn = analyzer.as_similarity_fn(candidates) if candidates else None
    optimizer = ClipOptimizer()
    meta = cache.get("metadata") or {}
    plan = optimizer.select(
        candidates,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        must_include=config.must_include,
        must_exclude=config.must_exclude,
        order_mode=config.order_mode,
        source_duration=float(meta.get("duration") or 0),
        multi_source=int(meta.get("source_count") or 1) > 1,
        similarity_fn=similarity_fn,
        max_sequence_duration=config.max_sequence_duration,
        min_sequence_duration=config.min_sequence_duration,
        sequence_duration_mode=config.sequence_duration_mode,
        allow_repeated_content=bool(config.allow_repeated_content),
        max_repeated_duration=config.max_repeated_duration,
        repeat_threshold=analyzer.threshold_for(config.repetition_mode),
        content_priorities=config.content_priorities,
        length_mode=config.length_mode,
        auto_length_strict=config.auto_length_strict,
        style=config.style,
    )
    return plan.model_dump()


def _expand_segment_score(raw: dict, seg) -> CandidateScore:
    text = seg.text.strip()
    lower = text.lower()
    words = text.split()
    filler = 0.85 if lower in {"um", "uh", "you know", "like", "yeah", "okay", "ok"} or len(words) < 3 else 0.05
    intro = 0.75 if any(w in lower for w in ("welcome", "hello", "hi everyone", "thanks for watching")) else 0.0
    hook = 0.65 if "?" in text or text.isupper() else 0.25
    emotional = min(1.0, 0.25 + float(raw.get("importance") or 0) * 0.55)
    return CandidateScore.model_validate(
        {
            **raw,
            "hook_strength": raw.get("hook_strength", hook),
            "filler_score": raw.get("filler_score", filler),
            "repetition_score": raw.get("repetition_score", 0.12),
            "context_dependency": raw.get("context_dependency", intro),
            "emotional_intensity": raw.get("emotional_intensity", emotional),
        }
    )


def align_segment_scores(items, count: int) -> list[dict | None]:
    """Match reply rows to the batch's segments by their "i" index.

    Falls back to reply order only when the reply has exactly one row per segment. Segments
    without a trustworthy row get None."""
    rows = [item for item in (items or []) if isinstance(item, dict)]
    by_index: dict[int, dict] | None = {}
    for row in rows:
        try:
            index = int(row.get("i"))
        except (TypeError, ValueError):
            by_index = None
            break
        if not 0 <= index < count or index in by_index:
            by_index = None
            break
        by_index[index] = row
    if rows and by_index is not None:
        return [by_index.get(index) for index in range(count)]
    if len(rows) == count:
        return rows
    return [None] * count


def _heuristic_segment(seg, reason: str = score_source.AI_UNAVAILABLE) -> dict:
    text = seg.text.strip()
    lower = text.lower()
    filler = 0.8 if lower in {"um", "uh", "you know", "like"} or len(text.split()) < 3 else 0.05
    intro = 0.7 if any(w in lower for w in ("welcome", "hello", "hi everyone", "thanks for")) else 0.0
    score = CandidateScore(
        importance=0.55 if len(text.split()) > 8 else 0.3,
        novelty=0.4,
        information_density=min(1.0, len(text.split()) / 18),
        hook_strength=0.6 if "?" in text or text.isupper() else 0.2,
        filler_score=filler,
        repetition_score=0.1,
        context_dependency=intro,
        emotional_intensity=0.3,
    )
    return score_source.tag(
        {
            "start": seg.start,
            "end": seg.end,
            "text": text,
            "score": score.model_dump(),
            "reason": "Useful information" if score.importance > 0.4 else "Low-value speech",
        },
        reason,
    )


# Prompt tokens per transcript line besides its text: index, time range, and chat template share.
_TEXT_LINE_OVERHEAD_TOKENS = 14
# Token counts are estimated, so batches aim below the real context size.
_CONTEXT_SAFETY = 0.85


def estimate_tokens(text: str) -> int:
    """Conservative token estimate without a tokenizer. Latin text averages about 4 characters
    per token; other scripts are often one token per character, so they are counted that way."""
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return int(ascii_chars / 3.5) + (len(text) - ascii_chars) + 1


def plan_text_batches(
    texts: list[str],
    *,
    max_items: int,
    fixed_tokens: int,
    configured_predict: int,
    per_item_output: int,
    num_ctx: int,
) -> list[list[int]]:
    """Split segments into batches whose prompt and reply both fit the context window.

    A fixed segment count overflows a small context when segments are long: Ollama then drops
    part of the prompt or cuts the reply off. Returns positions into ``texts``, in order."""
    limit = int(num_ctx * _CONTEXT_SAFETY)
    reply_room = max(configured_predict, num_ctx // 2)
    batches: list[list[int]] = []
    current: list[int] = []
    prompt_tokens = fixed_tokens
    for position, text in enumerate(texts):
        cost = estimate_tokens(text) + _TEXT_LINE_OVERHEAD_TOKENS
        count = len(current) + 1
        needed = prompt_tokens + cost + predict_budget(configured_predict, count, per_item_output, num_ctx)
        too_big = count > max(1, max_items) or needed > limit or per_item_output * count + 64 > reply_room
        if current and too_big:
            batches.append(current)
            current, prompt_tokens = [], fixed_tokens
        current.append(position)
        prompt_tokens += cost
    if current:
        batches.append(current)
    return batches


def _visual_signals(
    scenes: list[dict],
    frame_map: dict,
    quality: dict[int, dict],
    energy: list[float],
    match: dict[int, dict],
) -> dict[int, float]:
    """A quick 0-1 "worth a look" score per scene from picture matching, motion, sound, and
    picture quality, used to rank scenes when there is no speech to go by."""
    if not scenes:
        return {}
    levels = np.asarray(energy, dtype=np.float32) if energy else None
    mean = float(levels.mean()) if levels is not None and levels.size else 0.0
    std = float(levels.std()) if levels is not None and levels.size else 0.0
    signals: dict[int, float] = {}
    for scene in scenes:
        scene_id = int(scene["scene_id"])
        paths = [Path(entry["image_path"]) for entry in frame_map.get(str(scene_id), [])]
        parts: list[tuple[float, float]] = [(motion_score(paths) if len(paths) > 1 else 0.3, 0.3)]
        if levels is not None and levels.size and std > 0:
            first = max(0, int(float(scene["start"]) / ENERGY_HOP))
            last = max(first + 1, int(float(scene["end"]) / ENERGY_HOP))
            window = levels[first:last]
            if window.size:
                parts.append((float(np.clip(0.5 + (float(window.mean()) - mean) / (3 * std), 0.0, 1.0)), 0.2))
        picture = (quality.get(scene_id) or {}).get("quality")
        if picture is not None:
            parts.append((float(picture), 0.15))
        interest = (match.get(scene_id) or {}).get("interest")
        if interest is not None:
            parts.append((float(interest), 0.45))
        weight = sum(w for _, w in parts)
        signals[scene_id] = round(sum(v * w for v, w in parts) / weight, 3)
    return signals


def _with_image_match(visual_by_scene: dict[int, dict], match: dict[int, dict]) -> dict[int, dict]:
    """Better quick visual scores for scenes the vision model did not analyze. Scores from the
    vision model are kept as they are: picture matching cannot judge a scene."""
    if not match:
        return visual_by_scene
    blended = {}
    for scene_id, entry in visual_by_scene.items():
        interest = (match.get(scene_id) or {}).get("interest")
        if interest is None or entry.get("score_source") == score_source.AI:
            blended[scene_id] = entry
            continue
        quick = float(entry.get("visual_importance", 0.4))
        blended[scene_id] = {**entry, "visual_importance": round(0.5 * quick + 0.5 * (0.25 + 0.65 * float(interest)), 3)}
    return blended


def _leave_out(candidates: list[dict], config: HighlightConfig, progress: "PipelineProgress | None") -> list[dict]:
    """Lower or remove the moments that show what the user chose to leave out."""
    if not (config.avoid or config.avoid_custom):
        return candidates
    kept, removed, fell_back = leave_out.apply(candidates, config.avoid, config.avoid_custom, config.avoid_mode)
    if progress is not None:
        lowered = sum(1 for c in kept if c.get("avoid_factor") is not None)
        if fell_back:
            progress.log(
                "Leave out: every moment matched, so they were ranked lower instead of removed.", "WARNING"
            )
        elif removed or lowered:
            progress.log(f"Leave out: {removed} moment(s) removed, {lowered} ranked lower")
    return kept


def _never_scenes(match: dict[int, dict], config: HighlightConfig) -> set[int]:
    """Scenes whose picture clearly shows something to leave out (fast picture matching)."""
    keys = [key for key in config.avoid if key in {"text", "screens", "empty"}]
    if leave_out.custom_terms(config.avoid_custom):
        keys.append("custom")
    if not keys or not match:
        return set()
    return {
        int(scene_id)
        for scene_id, row in match.items()
        if max(float((row.get("avoid") or {}).get(key) or 0) for key in keys) >= leave_out.NEVER_FROM
    }


def _stored_visual_focus(value) -> list[str]:
    """The priorities the cached visual scores were made for. Older versions stored only
    whether Connection was on."""
    if isinstance(value, bool):
        return ["connection"] if value else []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


# Most a strong visual focus match adds to a moment's score, and where matching starts to count.
_VISUAL_FOCUS_BOOST = 0.1
_VISUAL_FOCUS_FROM = 0.5


def _apply_visual_focus(candidates: list[dict], scenes: list[dict], match: dict[int, dict]) -> int:
    """Raise moments whose picture matches the focus text. Returns how many were raised."""
    raised = 0
    for candidate in candidates:
        start = float(candidate["start"])
        scene = next((s for s in scenes if float(s["start"]) <= start < float(s["end"])), None)
        focus = (match.get(int(scene["scene_id"])) or {}).get("focus") if scene else None
        if focus is None or focus <= _VISUAL_FOCUS_FROM:
            continue
        strength = (float(focus) - _VISUAL_FOCUS_FROM) / (1 - _VISUAL_FOCUS_FROM)
        candidate["score"] = round(min(1.0, float(candidate.get("score", 0)) + _VISUAL_FOCUS_BOOST * strength), 4)
        candidate["visual_focus_match"] = round(float(focus), 3)
        raised += 1
    if raised:
        candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
    return raised


def _plan_embedder(settings: Settings) -> Embedder | None:
    """Embeddings for re-planning outside a job, when Ollama answers quickly."""
    if not str(getattr(settings, "ollama_embed_model", "") or "").strip():
        return None
    try:
        from app.services.ollama_runtime import get_ollama_client

        client = get_ollama_client(settings)
        if not client.reachable():
            return None
        return Embedder(client, settings.ollama_embed_model)
    except Exception:  # noqa: BLE001 - re-planning works without embeddings
        return None


def _highlight_seconds(config: HighlightConfig, source_duration: float) -> float:
    """Roughly how long the highlight will be, for sizing the vision shortlist."""
    if getattr(config, "length_mode", "range") == "auto":
        return max(float(config.min_duration), source_duration * 0.25)
    return float(config.max_duration)


def _candidate_limit(configured: int, source_duration: float) -> int:
    scaled = int(source_duration / _SECONDS_PER_CANDIDATE)
    return max(int(configured), min(_MAX_CANDIDATES_CEILING, scaled))


def frame_extract_workers(frame_count: int) -> int:
    """Concurrent FFmpeg frame grabs: half the logical CPUs, at most 4, never more than needed."""
    cpus = os.cpu_count() or 2
    return max(1, min(4, cpus // 2, frame_count))


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _cleanup_frames(frames_dir: Path) -> None:
    if not frames_dir.exists():
        return
    for file in frames_dir.glob("*.jpg"):
        try:
            file.unlink()
        except OSError:
            pass
