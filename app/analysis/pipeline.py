from __future__ import annotations

import json
import time
from pathlib import Path

from app.ai.ollama import OllamaClient, OllamaConnectionError, OllamaTruncatedError, predict_budget
from app.ai.prompts import (
    TRANSCRIPT_BATCH_SCHEMA_FAST,
    transcript_batch_prompt_fast,
)
from app.analysis import score_source
from app.analysis.audio import detect_silence, extract_audio, transcribe_detailed
from app.analysis.candidates import generate_candidates
from app.analysis.optimizer import ClipOptimizer
from app.analysis.repetition import RepetitionAnalyzer
from app.analysis.scenes import detect_scenes
from app.analysis.scorer import HighlightScorer
from app.analysis.visual import analyze_scenes_visual, sample_timestamps
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
    ("visual", "Visual AI analysis"),
    ("content", "Finding important moments"),
    ("repetition", "Removing repetition"),
    ("planning", "Planning highlight"),
]

# Upper bound on reply tokens per segment in a batched transcript-scoring reply.
_TEXT_TOKENS_PER_SEGMENT = 40


class PipelineProgress:
    def __init__(self, emit):
        self.emit = emit
        self.started = time.time()

    def log(self, message: str, level: str = "INFO") -> None:
        self.emit("log", {"level": level, "message": message, "ts": time.time()})

    def stage(self, name: str, status: str, detail: str = "", index: int = 0, total: int = 1) -> None:
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
        if self.settings.enable_whisper and not wav.exists():
            progress.stage("audio", "running")
            try:
                extract_audio(self.runner, source, wav)
                progress.log("Audio extracted")
            except Exception as exc:  # noqa: BLE001
                progress.log(f"Audio extraction failed: {exc}", "WARNING")
            progress.stage("audio", "done")
        else:
            progress.stage("audio", "done", "cached")

        transcript_data = cache.get("transcript")
        if transcript_data is None:
            progress.stage("transcript", "running")
            transcribed = True
            if wav.exists() and self.settings.enable_whisper:
                transcript_data, transcribed = self._transcribe(wav, progress)
            else:
                transcript_data = Transcript().model_dump()
                progress.log("No transcript available; continuing with visual/scene analysis", "WARNING")
            if transcribed:
                _write(job_dir / "transcript.json", transcript_data)
            progress.stage("transcript", "done", "" if transcribed else "Transcription failed")
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
            frame_map = {}
            use_lite_frames = self.settings.enable_visual_analysis and bool(self.ollama) and self.vision_enabled
            frame_density = "lite" if use_lite_frames else "auto"
            for scene in scenes:
                model = Scene.model_validate(scene)
                paths = []
                for stamp in sample_timestamps(model, source_duration=source_duration, density=frame_density):
                    out = frames_dir / f"scene_{model.scene_id}_{int(stamp * 100)}.jpg"
                    if not out.exists():
                        try:
                            self.runner.extract_frame(source, out, stamp, duration=source_duration)
                        except Exception as ext:  # noqa: BLE001
                            progress.log(f"Frame extract failed at {stamp:.2f}s: {ext}", "WARNING")
                            continue
                    if out.exists() and out.stat().st_size > 0:
                        paths.append({"timestamp": stamp, "image_path": str(out)})
                frame_map[str(model.scene_id)] = paths
            _write(job_dir / "frames.json", frame_map)
            progress.log(f"Representative frames sampled ({sum(len(v) for v in frame_map.values())} frames)")
            progress.stage("frames", "done")
        cache["frames"] = frame_map

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
            if visual and pending_scenes:
                progress.log(f"Re-running visual AI for {len(pending_scenes)} scene(s) that used fallback scores")
            ai_scene_count = sum(
                1
                for scene in pending_scenes
                if float(scene.get("duration") or 0) >= self.settings.ollama_vision_min_scene_duration
            )
            batch_count = max(
                1,
                (ai_scene_count + self.settings.ollama_vision_batch_size - 1)
                // self.settings.ollama_vision_batch_size,
            )
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

            results, errors, fallback_reasons = analyze_scenes_visual(
                self.ollama,
                self.settings.ollama_vision_model,
                pending_scenes,
                frame_map,
                source_duration,
                config.custom_focus,
                enabled=self.settings.enable_visual_analysis and vision_ok,
                settings=self.settings,
                on_batch_start=_visual_batch_progress,
            )
            for scene in pending_scenes:
                scene_id = int(scene["scene_id"])
                if scene_id in results:
                    visual[str(scene_id)] = score_source.tag(
                        results[scene_id].model_dump(), fallback_reasons.get(scene_id)
                    )
            for message in errors:
                progress.log(message, "WARNING")
            progress.stage("visual", "done", f"Analyzed {len(results)} scene(s)", batch_count, batch_count)
            _write(job_dir / "visual_analysis.json", visual)
            progress.log("Visual analysis completed")
        cache["visual"] = visual

        progress.stage("content", "running")
        segment_scores, scores_changed = self._score_transcript(
            transcript, config, progress, cache.get("segment_scores")
        )
        if scores_changed:
            _write(job_dir / "segment_scores.json", segment_scores)
        cache["segment_scores"] = segment_scores

        visual_by_scene = {int(k): v for k, v in visual.items()}
        candidates = generate_candidates(
            transcript,
            scenes,
            visual_by_scene,
            segment_scores,
            silences,
            source_duration,
            config,
            self.scorer,
            self.settings.maximum_candidates,
        )
        progress.log(f"{len(candidates)} candidates generated")
        progress.stage("content", "done")
        cache["candidates"] = candidates
        if not plan:
            return cache

        progress.stage("repetition", "running")
        similarity_fn = None
        if self.settings.enable_repetition_analysis and candidates:
            similarity_fn = self.repetition.as_similarity_fn(candidates)
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
        similarity_fn = None
        candidates = all_candidates
        if self.settings.enable_repetition_analysis and candidates:
            similarity_fn = self.repetition.as_similarity_fn(candidates)
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
            similarity_fn=similarity_fn,
            max_sequence_duration=config.max_sequence_duration,
            min_sequence_duration=config.min_sequence_duration,
            sequence_duration_mode=config.sequence_duration_mode,
            allow_repeated_content=bool(config.allow_repeated_content),
            max_repeated_duration=config.max_repeated_duration,
            repeat_threshold=self.repetition.threshold_for(config.repetition_mode),
            content_priorities=config.content_priorities,
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
        return cache

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

        batch_size = max(1, self.settings.ollama_text_batch_size)
        batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
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
                transcript_batch_prompt_fast(config.style, config.custom_focus, payload),
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

    def _check_cancel(self) -> None:
        if self.runner.cancel_event.is_set():
            raise CancelledError("Job cancelled")


def plan_from_cache(cache: dict, config: HighlightConfig, settings: Settings) -> dict:
    candidates = list(cache.get("candidates") or [])
    scorer = HighlightScorer()
    for cand in candidates:
        metrics = CandidateScore.model_validate(cand.get("metrics") or {})
        cand["score"] = scorer.finalize(
            metrics,
            style=config.style,
            priorities=config.content_priorities,
            priority_weights=config.priority_weights,
        )
        cand["reason"] = scorer.reason_for(metrics, cand.get("reason") or "Meaningful moment")
    if config.remove_repetition:
        analyzer = RepetitionAnalyzer()
        candidates = analyzer.deduplicate(candidates, threshold=analyzer.threshold_for(config.repetition_mode))
    analyzer = RepetitionAnalyzer()
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
        similarity_fn=similarity_fn,
        max_sequence_duration=config.max_sequence_duration,
        min_sequence_duration=config.min_sequence_duration,
        sequence_duration_mode=config.sequence_duration_mode,
        allow_repeated_content=bool(config.allow_repeated_content),
        max_repeated_duration=config.max_repeated_duration,
        repeat_threshold=analyzer.threshold_for(config.repetition_mode),
        content_priorities=config.content_priorities,
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
