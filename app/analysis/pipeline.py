from __future__ import annotations

import hashlib
import os
import threading
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.ai.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaTruncatedError,
    describe_usage,
    is_memory_error,
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
from app.analysis import music as music_beat
from app.analysis import score_source, speech_check
from app.analysis.audio import detect_silence, extract_audio, release_whisper_models, transcribe_detailed
from app.analysis import avoid as leave_out
from app.analysis.avoid import picture_avoid_texts, speech_avoid_text, vision_avoid_text
from app.analysis.candidates import generate_candidates, populate_candidate_audio_metrics, rescore_candidate
from app.analysis.optimizer import ClipOptimizer
from app.analysis.progress import STAGES, PipelineProgress  # noqa: F401 - STAGES is re-exported
from app.analysis.repetition import RepetitionAnalyzer
from app.analysis.scenes import detect_scenes
from app.analysis.scorer import HighlightScorer
from app.analysis.semantic import FOCUS_BOOST, Embedder, apply_focus, repetition_embedding_fn
from app.analysis.signals import ENERGY_HOP, audio_energy, scene_quality
from app.analysis.image_match import get_matcher, release_matcher, score_scenes
from app.analysis import audio_events
from app.analysis.director import CONTENT_FILE, DIRECTOR_FILE, apply_director, content_entry, run_director
from app.brief.engines.base import BriefEngineError, BriefPaused
from app.brief.schema import BRIEF_DIR
from app.brief.service import apply_brief, clear_failure, make_brief, planning_config, read_failure, save_failure
from app.analysis.visual import (
    vision_reply_tokens,
    analyze_scenes_visual,
    fitting_vision_batch,
    motion_score,
    sample_timestamps,
    shortlist_scenes,
    vision_scenes_per_image,
)
from app.config import Settings
from app.models.schemas import CandidateScore, ClipPlan, ClipPlanItem, HighlightConfig, Scene, Transcript
from app.services.run_plan import RunPlan
from app.video.ffmpeg_runner import CancelledError, FFmpegRunner, report_progress
from app.utils.timefmt import format_duration
from app.utils.atomic import atomic_write_json, part_path, read_json, replace_file
from app.video.processor import VideoProcessor
from app.video.sources import save_sources, tag_candidates, working_path



# Model work per step and what the analysis ran with, for the processing details page.
AI_USAGE_FILE = "ai_usage.json"
RUN_INFO_FILE = "run_info.json"

# Small copy of the picture that scenes and frames are read from (see _analysis_proxy).
ANALYSIS_PROXY = "analysis_proxy.mp4"
_PROXY_SHORT_SIDE = 540
_SLOW_CODECS = {"av1", "hevc", "h265", "vp9"}
# The vision shortlist is sized for a highlight of at most this share of the video.
_SHORTLIST_SOURCE_SHARE = 0.4

# Candidate moments kept for planning: the configured number, or more for long videos so
# their later parts are not crowded out (one per ~8 s), up to this ceiling.
_MAX_CANDIDATES_CEILING = 200
_SECONDS_PER_CANDIDATE = 8.0

# Upper bound on reply tokens per segment in a batched transcript-scoring reply. qwen3:4b
# pretty-prints the JSON and was measured at 42-46 tokens per segment; a lower budget cuts
# full batches off, and each cut-off reply is generated again as two halves.
_TEXT_TOKENS_PER_SEGMENT = 48

# Progress of the slow steps, saved while they run so a pause, crash, or power cut loses at
# most one batch. Each is removed when its step finishes; nothing else reads them.
TRANSCRIPT_PARTIAL = "transcript.partial.json"
# The English translation of unreliable speech, saved as it is written (see _translate_speech).
TRANSLATION_PARTIAL = "transcript_translation.partial.json"
SEGMENT_SCORES_PARTIAL = "segment_scores.partial.json"
VISUAL_PARTIAL = "visual_analysis.partial.json"
PARTIAL_FILES = (TRANSCRIPT_PARTIAL, TRANSLATION_PARTIAL, SEGMENT_SCORES_PARTIAL, VISUAL_PARTIAL)

# Sound labels per window and the sound vectors behind them (prompt mode).
SOUND_EVENTS_FILE = "sound_events.json"
SOUND_VECTORS_FILE = "sound_vectors.npy"
# A scene whose picture matches the request at least this well (0-1, relative to the video) is
# always given to the vision model in prompt mode.
_PROMPT_MATCH_SHORTLIST = 0.7


class VideoAnalyzer:
    def __init__(
        self,
        settings: Settings,
        runner: FFmpegRunner,
        ollama: OllamaClient | None,
        vision_enabled: bool = True,
        text_enabled: bool = True,
        run_plan: RunPlan | None = None,
    ):
        self.settings = settings
        # Memory decisions for this job only (see app.services.run_plan).
        self.run_plan = run_plan or RunPlan()
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
        if meta and "source_count" in meta and "has_audio" not in meta:
            # Older versions saved the shortened multi-video summary over a single video's
            # metadata; probe again so sound, codecs, and rotation are known.
            try:
                meta = {**meta, **self.runner.probe(source).model_dump()}
                _write(job_dir / "metadata.json", meta)
            except CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the summary still has duration and size
                pass
        if not meta:
            progress.stage("metadata", "running")
            meta = self.runner.probe(source).model_dump()
            _write(job_dir / "metadata.json", meta)
            progress.log(f"Video loaded ({meta.get('width')}x{meta.get('height')}, {meta.get('duration'):.1f}s)")
            progress.stage("metadata", "done")
        cache["metadata"] = meta
        source_duration = float(meta.get("duration") or 0)
        progress.source_duration(source_duration)

        proxy: dict[str, Path] = {}

        def analysis_video(stage: str) -> Path:
            """The video that scenes and frames are read from: a small copy of a large source.
            Only scene detection (which decodes every frame) makes the copy; a few frames alone
            are read faster from the original than a copy could be made."""
            if "path" not in proxy:
                existing = job_dir / ANALYSIS_PROXY
                if stage != "scenes" and not (existing.is_file() and existing.stat().st_size > 1000):
                    return source
                proxy["path"] = self._analysis_proxy(job_dir, source, meta, progress, stage)
            return proxy["path"]

        scenes = cache.get("scenes")
        if not scenes:
            progress.stage("scenes", "running")
            detected = detect_scenes(
                analysis_video("scenes"),
                self.settings.scene_threshold,
                self.runner,
                on_progress=progress.share("scenes", "time", source_duration),
            )
            scenes = [s.model_dump() for s in detected]
            _write(job_dir / "scenes.json", scenes)
            progress.log(f"{len(scenes)} scenes detected")
            progress.stage("scenes", "done")
        cache["scenes"] = scenes

        wav = job_dir / "source" / "audio.wav"
        # Speech can be turned off for the whole app or for one highlight. Audio is still
        # extracted when the video has sound: loudness and pauses come from it.
        want_speech = bool(self.settings.enable_whisper) and bool(getattr(config, "use_speech", True))
        self._plan_progress(progress, cache, config, meta, wav, want_speech)
        if not wav.exists() and meta.get("has_audio", True):
            progress.stage("audio", "running")
            # Written under another name and renamed when complete: a cancel or a power cut
            # mid-file must never leave a partial file that is reused as finished audio.
            pending_wav = part_path(wav)
            try:
                with report_progress(self.runner, source_duration, progress.share("audio", "time", source_duration)):
                    extract_audio(self.runner, source, pending_wav)
                replace_file(pending_wav, wav)
                progress.log("Audio extracted")
            except CancelledError:
                pending_wav.unlink(missing_ok=True)
                raise
            except Exception as exc:  # noqa: BLE001
                pending_wav.unlink(missing_ok=True)
                progress.log(f"Audio extraction failed: {exc}", "WARNING")
            progress.stage("audio", "done")
        elif not wav.exists():
            progress.stage("audio", "skipped", "No sound in this video")
        else:
            progress.stage("audio", "done", "cached")

        transcript_data = cache.get("transcript")
        wanted_language = str(getattr(config, "speech_language", "") or "")
        if (
            want_speech
            and isinstance(transcript_data, dict)
            and wanted_language
            and str(transcript_data.get("language") or "") != wanted_language
        ):
            # Made in another language than the one now chosen: transcribed again.
            progress.log(f"The spoken language is now set to '{wanted_language}': transcribing again")
            transcript_data = None
            _forget_speech(job_dir, cache)
        if not want_speech:
            # Not written to disk, so turning speech back on transcribes (or reuses) as usual.
            transcript_data = Transcript().model_dump()
            progress.stage("transcript", "skipped", "Speech is turned off")
            progress.log("Speech is turned off: moments are chosen from pictures and sound")
        elif transcript_data is None:
            progress.stage("transcript", "running")
            transcribed = True
            if wav.exists():
                if self.run_plan.free_gpu_for_speech:
                    self._unload_models(None)
                transcript_data, transcribed = self._transcribe(
                    wav, progress, job_dir / TRANSCRIPT_PARTIAL, duration=source_duration, language=wanted_language
                )
                if self.run_plan.one_at_a_time or self.run_plan.free_gpu_for_speech:
                    release_whisper_models()
            else:
                # Not cached: if the audio can be extracted on a later run, it is transcribed then.
                transcribed = False
                transcript_data = Transcript().model_dump()
                progress.log("No transcript available; continuing with visual/scene analysis", "WARNING")
            if transcribed:
                _write(job_dir / "transcript.json", transcript_data)
                (job_dir / TRANSCRIPT_PARTIAL).unlink(missing_ok=True)
            progress.stage(
                "transcript", "done", "" if transcribed or not wav.exists() else "Transcription failed"
            )
        cache["transcript"] = transcript_data
        transcript = Transcript.model_validate(transcript_data)
        speech = speech_check.check(transcript) if want_speech else {"reliable": True}
        if not speech.get("reliable", True) and self._can_translate(transcript, wav):
            translated = self._translate_speech(wav, transcript, progress, job_dir, source_duration)
            if translated is False:
                # Not readable either: remembered, so later analyses do not translate again.
                transcript = transcript.model_copy(update={"translation_unreadable": True})
                transcript_data = transcript.model_dump()
                _write(job_dir / "transcript.json", transcript_data)
                cache["transcript"] = transcript_data
            elif translated is not None:
                transcript = translated
                transcript_data = translated.model_dump()
                _write(job_dir / "transcript.json", transcript_data)
                # Scores and voices were read from the old segments.
                _forget_words(job_dir, cache)
                cache["transcript"] = transcript_data
                speech = speech_check.check(transcript)
        speech_reliable = bool(speech.get("reliable", True))
        cache["speech_reliable"] = speech_reliable
        if not speech_reliable:
            progress.log(
                f"The transcript looks unreliable ({speech.get('reason')}). Moments are chosen from pictures, "
                f"sound and speech timing instead of the words. For better results, "
                f"{speech_check.advice(speech, self.settings.whisper_model)}.",
                "WARNING",
            )

        silences = cache.get("silences")
        if silences is None:
            progress.stage("silence", "running")
            target = wav if wav.exists() else source
            with report_progress(self.runner, source_duration, progress.share("silence", "time", source_duration)):
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
            frame_map = self._extract_frames(
                analysis_video("frames"), scenes, frames_dir, source_duration, frame_density, progress
            )
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

        want_visuals = bool(getattr(config, "use_visuals", True))
        match = self._image_match(job_dir, scenes, frame_map, config, cache, progress) if want_visuals else {}
        if self.run_plan.one_at_a_time:
            release_matcher()
        signals = _visual_signals(scenes, frame_map, quality, energy, match)
        if config.mode == "prompt" and want_speech:
            cache["sound_events"] = self._sound_events(job_dir, source, meta, config, cache, progress)

        if want_speech and not speech_reliable:
            # Scoring garbled words gives random scores; the timings are still used.
            segment_scores = []
            progress.stage("content", "skipped", "The transcript is unreliable")
        elif want_speech:
            progress.stage("content", "running")
            usage_before = self._usage()
            segment_scores, scores_changed = self._score_transcript(
                transcript, config, progress, cache.get("segment_scores"), job_dir / SEGMENT_SCORES_PARTIAL
            )
            self._log_usage(progress, "Transcript AI", usage_before)
            if scores_changed:
                _write(job_dir / "segment_scores.json", segment_scores)
            (job_dir / SEGMENT_SCORES_PARTIAL).unlink(missing_ok=True)
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
        visual_partial = job_dir / VISUAL_PARTIAL
        # Scenes the vision model scored before a pause or an interruption.
        saved_scenes = _restore_partial_visual(visual, read_json(visual_partial), connection_focus, scenes)
        if saved_scenes:
            progress.log(f"Visual AI: {len(saved_scenes)} scene(s) were analyzed before the interruption")
        ollama_ok = bool(self.ollama)
        vision_ok = ollama_ok and self.vision_enabled and want_visuals
        vision_ready = vision_ok and self.settings.enable_visual_analysis
        # Prompt mode: the AI editor needs to know what each analyzed scene shows.
        describe = vision_ready and config.mode == "prompt"
        pending_scenes = [
            scene
            for scene in scenes
            if str(scene["scene_id"]) not in visual
            or (vision_ready and score_source.needs_ai(visual[str(scene["scene_id"])]))
            or (describe and _needs_caption(visual[str(scene["scene_id"])]))
        ]
        if pending_scenes or not visual:
            progress.stage("visual", "running")
            if not want_visuals:
                progress.log("Video analysis is off: scenes use quick motion and picture-quality scoring only")
            elif ollama_ok and not vision_ok:
                progress.log(
                    f"Vision model '{self.settings.ollama_vision_model}' is not available. Using deterministic visual scoring.",
                    "WARNING",
                )
            elif not ollama_ok:
                progress.log("Ollama is unavailable. Using deterministic visual scoring.", "WARNING")
            shortlist = None
            every_scene = str(getattr(self.settings, "vision_scope", "shortlist")) == "all" or config.analysis_depth == "best"
            if vision_ready and not every_scene:
                shortlist = shortlist_scenes(
                    scenes,
                    segment_scores,
                    _highlight_seconds(config, source_duration),
                    quality,
                    visual_signal=signals,
                    match=match,
                )
                if shortlist is not None and config.mode == "prompt":
                    # Scenes that look like what the user asked for are always worth a look.
                    wanted = {sid for sid, item in match.items() if float(item.get("focus") or 0) >= _PROMPT_MATCH_SHORTLIST}
                    if wanted - set(shortlist):
                        progress.log(f"Visual AI: {len(wanted - set(shortlist))} more scene(s) look like what you asked for")
                    shortlist = set(shortlist) | wanted
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
            per_image = 1
            if self.ollama is not None:
                vision_ctx = self.ollama.role_options("vision")["num_ctx"]
                per_image = vision_scenes_per_image(self.settings, vision_ctx, describe)
                vision_batch = fitting_vision_batch(
                    vision_batch,
                    vision_ctx,
                    self.settings.ollama_vision_num_predict,
                    vision_reply_tokens(describe),
                    per_image,
                ) * per_image
            batch_count = max(1, (ai_scene_count + vision_batch - 1) // max(1, vision_batch))
            # For the time estimates, and for the speed this run records (auto_tune.record_run).
            progress.speeds["vision_scenes_per_image"] = per_image
            tiling = f", {per_image} scenes per picture" if per_image > 1 else ""
            progress.log(
                f"Visual AI: analyzing {ai_scene_count} scene(s) in up to {batch_count} batched call(s) "
                f"({self.settings.ollama_vision_max_pixels}px frames{tiling}, model {self.settings.ollama_vision_model})"
            )
            ai_scenes = ai_scene_count if vision_ready else 0
            progress.expect("visual", progress.vision_seconds(ai_scenes, per_image) if ai_scenes else 2.0)
            # Scenes handed to the vision model so far (every batch that has started).
            sent = {"scenes": 0}
            if ai_scenes:
                progress.step("visual", 0, ai_scenes, "scenes")
            self._check_cancel()

            def _visual_batch_progress(batch_no: int, total: int, size: int) -> None:
                self._check_cancel()
                progress.log(f"Visual AI: batch {batch_no}/{total} ({size} scene{'s' if size != 1 else ''})")
                if ai_scenes:
                    progress.step("visual", min(sent["scenes"], ai_scenes), ai_scenes, "scenes")
                sent["scenes"] += size

            if self.run_plan.one_at_a_time and ai_scene_count:
                # The speech model has finished; its memory goes to the video model.
                self._unload_models(self.settings.ollama_vision_model)

            def _save_visual_progress(done: dict, reasons: dict) -> None:
                if ai_scenes:
                    progress.step("visual", min(sent["scenes"], ai_scenes), ai_scenes, "scenes")
                added = False
                for scene_id, result in done.items():
                    if scene_id not in reasons and str(scene_id) not in saved_scenes:
                        saved_scenes[str(scene_id)] = score_source.tag(result.model_dump())
                        added = True
                if added:
                    atomic_write_json(visual_partial, {"focus": connection_focus, "scenes": saved_scenes})

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
                on_memory_error=lambda: self._relieve_memory(
                    self.settings.ollama_vision_model, progress, "The video model"
                ),
                on_batch_done=_save_visual_progress,
                describe=describe,
            )
            for scene in pending_scenes:
                scene_id = int(scene["scene_id"])
                existing = visual.get(str(scene_id))
                if (
                    fallback_reasons.get(scene_id)
                    and isinstance(existing, dict)
                    and existing.get("score_source") == score_source.AI
                ):
                    # A scene waiting only for its description keeps its AI score when it
                    # was not looked at again this time.
                    continue
                if scene_id in results:
                    visual[str(scene_id)] = score_source.tag(
                        results[scene_id].model_dump(), fallback_reasons.get(scene_id)
                    )
            for message in errors:
                progress.log(message, "WARNING")
            self._log_usage(progress, "Visual AI", usage_before)
            progress.stage("visual", "done", f"Analyzed {len(results)} scene(s)")
            _write(job_dir / "visual_analysis.json", visual)
            progress.log("Visual analysis completed")
        elif saved_scenes:
            # Every scene was scored before the interruption: keep those scores with the analysis.
            _write(job_dir / "visual_analysis.json", visual)
        cache["visual"] = visual
        cache["visual_connection_focus"] = connection_focus
        _write(job_dir / "visual_connection_focus.json", connection_focus)
        visual_partial.unlink(missing_ok=True)

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
        picture_focus = _picture_focus(config)
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
        plan = self.optimizer.plan(
            candidates,
            config,
            source_duration=source_duration,
            similarity_fn=similarity_fn,
            repeat_threshold=self.repetition.threshold_for(config.repetition_mode),
        )
        plan_data = plan.model_dump()
        _write(job_dir / "clip_plan.json", plan_data)
        cache["clip_plan"] = plan_data
        progress.log("Final plan generated")
        progress.stage("planning", "done")
        if not self.settings.debug_analysis:
            _cleanup_frames(frames_dir)
            (job_dir / ANALYSIS_PROXY).unlink(missing_ok=True)
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
        content: dict[str, dict] = {}
        total_duration = 0.0
        multi = len(sources) > 1
        combined_meta: dict = cache.get("metadata") or {}
        piece_metas: list[dict] = []
        speech_reliable: list[bool] = []
        progress.configure_sources(
            [_file_size(working_path(source)) for source in sources],
            [source.get("duration") for source in sources],
        )
        for index, source in enumerate(sources):
            progress.begin_source(index)
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
            piece_metas.append(dict(piece.get("metadata") or {}))
            speech_reliable.append(bool(piece.get("speech_reliable", True)))
            if not multi:
                # The analysis ran on a copy; the processing details (run_info.json) and later
                # steps read these from the job's own cache (a translated transcript also drops
                # the scores and voices read from the old one).
                for key in ("scenes", "transcript", "visual", "segment_scores", "speakers"):
                    if key in piece:
                        cache[key] = piece[key]
                    else:
                        cache.pop(key, None)
            content[str(source.get("id") or f"v{index}")] = content_entry(
                piece.get("scenes") or [],
                piece.get("visual") or {},
                piece.get("sound_events"),
                speech_reliable=piece.get("speech_reliable", True),
            )
            tagged = tag_candidates(piece.get("candidates") or [], source)
            all_candidates.extend(tagged)
            duration = float((piece.get("metadata") or {}).get("duration") or 0)
            total_duration += duration
            source["duration"] = duration
            source["width"] = int((piece.get("metadata") or {}).get("width") or 0)
            source["height"] = int((piece.get("metadata") or {}).get("height") or 0)
            source["fps"] = float((piece.get("metadata") or {}).get("fps") or 0)
            progress.log(f"{label}: {len(tagged)} candidate moments")

        # Unreliable when the words of any video could not be used.
        cache["speech_reliable"] = all(speech_reliable)
        best = max(sources, key=lambda item: int(item.get("width") or 0) * int(item.get("height") or 0))
        if not multi:
            # One video: keep its full probe (sound, codecs, rotation) under the summary.
            combined_meta = piece_metas[0] if piece_metas else combined_meta
        else:
            combined_meta = {**combined_meta, "has_audio": any(m.get("has_audio") for m in piece_metas)}
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
        _write(job_dir / CONTENT_FILE, content)
        cache["content_index"] = content
        if config.mode == "prompt":
            director_plan = self._direct(job_dir, config, candidates, content, cache, progress)
            candidates = apply_director(candidates, director_plan)
        if config.video_brief:
            brief = self._brief(job_dir, config, candidates, content, cache, progress, sources)
            candidates = apply_brief(candidates, brief)

        progress.stage("planning", "running")
        plan = None
        if config.music_led:
            from app.video.audio_mix import find_music

            track = find_music(job_dir)
            if track is None:
                progress.log("Music-led editing needs a music file; add one on the Create or Review page.", "WARNING")
            analysis = music_beat.load_or_analyze(job_dir, track, self.runner, progress.log)
            cache["music_analysis"] = analysis
            if analysis:
                plan = music_led_plan(
                    candidates,
                    planning_config(config),
                    analysis,
                    self.optimizer,
                    source_duration=total_duration,
                    source_lengths=_source_lengths(sources, total_duration),
                    multi_source=multi,
                    similarity_fn=similarity_fn,
                    repeat_threshold=self.repetition.threshold_for(config.repetition_mode),
                )
                if plan is not None:
                    progress.log(plan.notes)
        if plan is None:
            plan = self.optimizer.plan(
                candidates,
                planning_config(config),
                source_duration=total_duration,
                multi_source=multi,
                similarity_fn=similarity_fn,
                repeat_threshold=self.repetition.threshold_for(config.repetition_mode),
            )
        used = {c.source_id for c in plan.clips if c.source_id}
        if used and len(sources) > 1:
            skipped = [s.get("filename") for s in sources if s.get("id") not in used]
            if skipped:
                progress.log(
                    "Skipped videos with no unique high-value moments: "
                    + ", ".join(str(n) for n in skipped if n)
                )
        if not self.settings.debug_analysis:
            # Frames and scenes are cached; the small analysis copies are made again if needed.
            for index, source in enumerate(sources):
                work = job_dir / "work" / str(source.get("id") or f"v{index}") if multi else job_dir
                (work / ANALYSIS_PROXY).unlink(missing_ok=True)
        plan_data = plan.model_dump()
        _write(job_dir / "clip_plan.json", plan_data)
        cache["clip_plan"] = plan_data
        progress.log(f"Combined plan uses {len(plan.clips)} clips from {len(used) or 1} video(s)")
        progress.stage("planning", "done")
        summary = progress.timing_summary()
        if summary:
            progress.log(f"Stage timings: {summary}")
            _write(job_dir / "analysis_timings.json", progress.timings)
        usage_by_step = getattr(progress, "usage_by_step", {})
        if usage_by_step:
            _write(job_dir / AI_USAGE_FILE, usage_by_step)
        _write(job_dir / RUN_INFO_FILE, self._run_info(config, cache, sources))
        return cache

    def _run_info(self, config: HighlightConfig, cache: dict, sources: list[dict]) -> dict:
        """What this analysis ran with and what it found, for the processing details page."""
        settings = self.settings
        transcript = cache.get("transcript") or {}
        visual = cache.get("visual") or {}
        ai_scenes = sum(
            1 for entry in visual.values() if isinstance(entry, dict) and entry.get("score_source") == score_source.AI
        )
        try:
            import psutil

            memory_gb = round(psutil.virtual_memory().total / 1024**3, 1)
        except Exception:  # noqa: BLE001 - optional detail
            memory_gb = None
        return {
            "finished_at": time.time(),
            "computer": {"cpu_threads": os.cpu_count() or 0, "memory_gb": memory_gb},
            "models": {
                "speech": settings.whisper_model if bool(getattr(config, "use_speech", True)) and settings.enable_whisper else "",
                "speech_device": settings.whisper_device,
                "vision": settings.ollama_vision_model
                if settings.enable_visual_analysis and bool(getattr(config, "use_visuals", True))
                else "",
                "text": settings.ollama_text_model,
                "editor": (
                    ("claude:" + str(getattr(settings, "claude_model", "")))
                    if director_engine(config, settings) == "claude"
                    else director_model(settings)
                )
                if config.mode == "prompt"
                else "",
            },
            "options": {
                "mode": config.mode,
                "vision_scope": "all" if config.analysis_depth == "best" else str(getattr(settings, "vision_scope", "shortlist")),
                "editor_review": bool(getattr(settings, "director_review", False)) and config.mode == "prompt",
                "analysis_copy": bool(getattr(self, "_proxy_used", False)),
                "music_led": bool(getattr(config, "music_led", False)),
            },
            "found": {
                "videos": len(sources),
                "scenes": len(cache.get("scenes") or []) if len(sources) == 1 else None,
                "scenes_ai": ai_scenes if len(sources) == 1 else None,
                "segments": len(transcript.get("segments") or []) if len(sources) == 1 else None,
                "language": transcript.get("language") if len(sources) == 1 else None,
                "language_probability": transcript.get("language_probability") if len(sources) == 1 else None,
                "speech_reliable": cache.get("speech_reliable", True),
                "speech_translated": bool(transcript.get("translated")) if len(sources) == 1 else None,
                "candidates": len(cache.get("candidates") or []),
                "music_bpm": (cache.get("music_analysis") or {}).get("bpm"),
            },
        }

    def _plan_progress(
        self, progress: PipelineProgress, cache: dict, config: HighlightConfig, meta: dict, wav: Path, want_speech: bool
    ) -> None:
        """Tell the progress which steps will not run for this video (finished earlier, or off)
        and which AI steps only use quick scoring, so the bar moves by the work that is left."""
        skip = []
        if wav.exists() or not meta.get("has_audio", True):
            skip.append("audio")
        if not want_speech or cache.get("transcript") is not None:
            skip.append("transcript")
        if cache.get("silences") is not None:
            skip.append("silence")
        if config.mode != "prompt" or not want_speech:
            skip.append("sounds")
        if not want_speech:
            skip.append("content")
        if config.mode != "prompt":
            skip.append("director")
        if not config.video_brief:
            skip.append("brief")
        progress.skip(*skip)
        if self.ollama is None or not self.text_enabled:
            progress.expect("content", 2.0)
        vision = (
            self.ollama is not None
            and self.vision_enabled
            and self.settings.enable_visual_analysis
            and bool(getattr(config, "use_visuals", True))
        )
        if not vision:
            progress.expect("visual", 2.0)

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
        texts = _picture_focus(config)
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
        progress.stage("frames", "running", "Matching pictures")
        try:
            result = score_scenes(
                matcher,
                frames,
                texts,
                avoid_texts,
                on_progress=lambda done, total: progress.step("frames", done, total, "frames", "Matching pictures"),
            )
        except Exception as exc:  # noqa: BLE001 - a ranking aid, never a requirement
            progress.log(f"Image matching skipped ({exc})", "WARNING")
            progress.stage("frames", "done")
            return {}
        progress.stage("frames", "done")
        duplicates = sum(1 for item in result.values() if item.get("duplicate_of") is not None)
        progress.log(
            f"Image matching ({matcher.device}): {len(result)} scenes in {time.perf_counter() - started:.1f}s"
            + (f", {duplicates} look like an earlier scene" if duplicates else "")
        )
        payload = {"focus": focus, "avoid": avoid_key, "scenes": {str(k): v for k, v in result.items()}}
        _write(job_dir / "image_match.json", payload)
        cache["image_match"] = payload
        return result

    def _sound_events(
        self,
        job_dir: Path,
        source: Path,
        meta: dict,
        config: HighlightConfig,
        cache: dict,
        progress: PipelineProgress,
    ) -> dict:
        """What can be heard in each 5 s window (sound labels) and how well it matches the sounds
        the user asked for; cached. The sound vectors are kept, so new wanted sounds are matched
        without another pass over the audio."""
        if not getattr(self.settings, "sound_matching", True) or not meta.get("has_audio", True):
            return {}
        cues = list(config.requirements.audio_cues) if config.requirements else []
        cue_key = " | ".join(cues)
        cached = cache.get("sound_events")
        vectors_path = job_dir / SOUND_VECTORS_FILE
        if isinstance(cached, dict) and cached.get("version") == audio_events.VERSION and cached.get("cue_key") == cue_key:
            progress.stage("sounds", "done", "cached")
            return cached
        matcher = audio_events.get_sound_matcher(int(getattr(self.settings, "ollama_num_thread", 0) or 0))
        if matcher is None:
            progress.stage("sounds", "skipped", "Sound model not downloaded")
            progress.log("Sound labels are off: download the sound model in Settings for better sound understanding.")
            return {}
        progress.stage("sounds", "running")
        started = time.perf_counter()
        try:
            if isinstance(cached, dict) and cached.get("version") == audio_events.VERSION and vectors_path.is_file():
                # Only the wanted sounds changed: match them against the saved vectors.
                starts = [float(item["start"]) for item in cached.get("windows") or []]
                vectors = np.load(vectors_path).astype(np.float32)
                labels = [item.get("labels") or {} for item in cached.get("windows") or []]
            else:
                starts, vectors = self._embed_sounds(
                    job_dir, source, matcher, progress, float(meta.get("duration") or 0)
                )
                labels = audio_events.label_windows(matcher, vectors)
                if len(starts):
                    np.save(vectors_path, vectors.astype(np.float16))
            match = audio_events.match_texts(matcher, vectors, cues)
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - sounds are an extra signal, never a requirement
            progress.log(f"Sound labels skipped ({exc})", "WARNING")
            progress.stage("sounds", "skipped", "Could not read the sound")
            return {}
        finally:
            if self.run_plan.one_at_a_time:
                audio_events.release_sound_matcher()
        windows = []
        for index, start in enumerate(starts):
            item = {"start": round(float(start), 3), "labels": labels[index] if index < len(labels) else {}}
            if match:
                item["cue"] = match[index]
            windows.append(item)
        events = {
            "version": audio_events.VERSION,
            "window": audio_events.WINDOW_SECONDS,
            "hop": audio_events.HOP_SECONDS,
            "cue_key": cue_key,
            "windows": windows,
        }
        _write(job_dir / SOUND_EVENTS_FILE, events)
        heard = sorted({key for item in windows for key in item["labels"] if key not in {"speech", "silence"}})
        progress.log(
            f"Sound labels ({matcher.device}): {len(windows)} windows in {time.perf_counter() - started:.1f}s"
            + (f"; heard {', '.join(heard[:8])}" if heard else "")
        )
        progress.stage("sounds", "done")
        return events

    def _embed_sounds(
        self,
        job_dir: Path,
        source: Path,
        matcher,
        progress: PipelineProgress | None = None,
        duration: float = 0.0,
    ) -> tuple[list[float], np.ndarray]:
        """Sound vectors from a 48 kHz copy of the sound, removed afterwards. Copying the sound
        counts as the first fifth of the step, listening to it as the rest."""
        wav48 = job_dir / "source" / "audio48k.wav"
        pending = part_path(wav48)

        def copy_share(fraction: float) -> None:
            if progress is not None:
                progress.step("sounds", 20 * fraction, 100)

        def listen_share(done: int, total: int) -> None:
            if progress is not None and total:
                progress.step("sounds", 20 + 80 * done / total, 100)

        try:
            with report_progress(self.runner, duration, copy_share):
                self.runner.extract_audio(source, pending, sample_rate=audio_events.SAMPLE_RATE)
            replace_file(pending, wav48)
            return audio_events.embed_recording(matcher, wav48, self._check_cancel, on_progress=listen_share)
        finally:
            pending.unlink(missing_ok=True)
            if not self.settings.debug_analysis:
                wav48.unlink(missing_ok=True)

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

    def _analysis_proxy(self, job_dir: Path, source: Path, meta: dict, progress: PipelineProgress, stage: str) -> Path:
        """A small copy of a large or slow-to-decode source for scene detection and frames
        (ANALYSIS_PROXY), or the source itself when it is small already or the copy fails.
        Times are unchanged: every frame keeps its timestamp."""
        target = job_dir / ANALYSIS_PROXY
        if target.is_file() and target.stat().st_size > 1000:
            self._proxy_used = True
            return target
        if not _needs_proxy(meta):
            return source
        duration = float(meta.get("duration") or 0)
        pending = part_path(target)
        progress.log(
            f"Making a small copy of the {meta.get('width')}x{meta.get('height')} video for analysis "
            "(the original is used for the final video)"
        )
        try:
            report = progress.share(stage, "time", duration, "Preparing the video for analysis")
            with report_progress(self.runner, duration, report):
                self.runner.run(VideoProcessor.analysis_proxy_command(source, pending, _PROXY_SHORT_SIDE))
            replace_file(pending, target)
            self._proxy_used = True
        except CancelledError:
            pending.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001 - analysis still works on the original
            pending.unlink(missing_ok=True)
            progress.log(f"The analysis copy could not be made ({str(exc)[:160]}); analyzing the original", "WARNING")
            return source
        return target

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
            # Renamed into place when complete: a stopped command or a power cut can leave a
            # partial image, which a later run must not reuse.
            pending = part_path(out)
            try:
                self.runner.extract_frame(source, pending, stamp, duration=source_duration)
                if pending.exists():
                    replace_file(pending, out)
            except CancelledError:
                pending.unlink(missing_ok=True)
                raise
            except Exception as ext:  # noqa: BLE001
                pending.unlink(missing_ok=True)
                return f"Frame extract failed at {stamp:.2f}s: {ext}"
            return None

        # Close timestamps can round to the same file name; grab each file once.
        unique: dict[Path, float] = {}
        for _, stamp, out in planned:
            unique.setdefault(out, stamp)
        finished = {"count": 0}
        counter = threading.Lock()

        def extract_counted(stamp: float, out: Path) -> str | None:
            try:
                return extract(stamp, out)
            finally:
                with counter:
                    finished["count"] += 1
                    done = finished["count"]
                progress.step("frames", done, len(unique), "frames")

        workers = frame_extract_workers(len(unique))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="frames") as pool:
                outcomes = list(pool.map(lambda item: extract_counted(item[1], item[0]), unique.items()))
        else:
            outcomes = [extract_counted(stamp, out) for out, stamp in unique.items()]
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

    def _transcribe(
        self,
        wav: Path,
        progress: PipelineProgress,
        partial: Path | None = None,
        duration: float = 0.0,
        language: str = "",
        task: str = "transcribe",
    ) -> tuple[dict, bool]:
        """Transcript data, and whether it is worth caching. A failed run is not cached, so the
        next analysis tries again instead of reusing an empty transcript. Finished segments are
        saved to ``partial`` as they come, and a later run continues after them."""
        extra: dict = {"check_cancel": self._check_cancel}
        if language:
            extra["language"] = language
        if task != "transcribe":
            extra["task"] = task
        if duration > 0:
            extra["on_position"] = lambda seconds: progress.step("transcript", seconds, duration, "time")
        if self.run_plan.whisper_gpu_compute:
            extra["gpu_compute"] = self.run_plan.whisper_gpu_compute
        if partial is not None:
            resume = read_json(partial)
            if isinstance(resume, dict) and language and resume.get("language") not in (None, language):
                # Saved while transcribing in another language: not continued.
                resume = None
            if isinstance(resume, dict) and resume.get("segments"):
                progress.log(
                    f"Continuing transcription from {format_duration(float(resume.get('until') or 0))} "
                    "(saved before the interruption)"
                )
                extra["resume"] = resume
            extra["on_progress"] = lambda state: atomic_write_json(partial, state)
        try:
            transcript, info = transcribe_detailed(
                wav,
                self.settings.whisper_model,
                self.settings.whisper_device,
                cpu_threads=self.settings.whisper_cpu_threads,
                num_workers=self.settings.whisper_num_workers,
                batch_size=self.settings.whisper_batch_size,
                **extra,
            )
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            transcript, info = Transcript(), {"ok": False, "error": str(exc)}
        if not info.get("ok"):
            if task == "translate":
                progress.log(
                    f"Translation failed: {info.get('error') or 'unknown error'}. "
                    "The original transcript is kept; the next analysis will try again.",
                    "WARNING",
                )
            else:
                progress.log(
                    f"Transcription failed: {info.get('error') or 'unknown error'}. "
                    "Continuing without speech; the next analysis will try again.",
                    "WARNING",
                )
            return Transcript().model_dump(), False
        if info.get("warning"):
            progress.log(str(info["warning"]), "WARNING")
        nseg = len(transcript.segments)
        heard = ""
        if transcript.language:
            heard = f", language {transcript.language}"
            if transcript.language_probability is not None:
                heard += f" ({transcript.language_probability:.0%} sure)"
        done = "Translation" if task == "translate" else "Transcription"
        progress.log(f"{done} completed ({nseg} segment{'s' if nseg != 1 else ''}{heard})")
        if (
            task == "transcribe"
            and not language
            and transcript.language_probability is not None
            and transcript.language_probability < speech_check.UNSURE_PROBABILITY
        ):
            progress.log(
                "Whisper was unsure which language is spoken. If the transcript is wrong, choose the spoken "
                "language on the Create page.",
                "WARNING",
            )
        return transcript.model_dump(), True

    def _can_translate(self, transcript: Transcript, wav: Path) -> bool:
        """An unreliable transcript can be written down again as an English translation: when
        the setting is on, it is not a translation already, and the audio is still there."""
        return (
            bool(getattr(self.settings, "whisper_translate_unreliable", False))
            and not transcript.translated
            and not transcript.translation_unreadable
            and bool(transcript.segments)
            and str(transcript.language or "").lower() != "en"
            and wav.exists()
        )

    def _translate_speech(
        self, wav: Path, transcript: Transcript, progress: PipelineProgress, job_dir: Path, duration: float
    ) -> Transcript | None | bool:
        """Whisper's English translation of speech it could not write down in its own language.
        None when it failed (tried again next time) and False when it is not readable either; the
        transcript is then kept and its words are not used, as before. The spoken language stays
        in ``language``."""
        spoken = str(transcript.language or "")
        progress.stage("transcript", "running", "Translating to English")
        progress.log(
            f"The transcript looks unreliable{f' ({spoken})' if spoken else ''}: writing the speech down again "
            "as an English translation, so its meaning can still be used to choose moments"
        )
        if self.run_plan.free_gpu_for_speech:
            self._unload_models(None)
        partial = job_dir / TRANSLATION_PARTIAL
        data, ok = self._transcribe(wav, progress, partial, duration=duration, language=spoken, task="translate")
        if self.run_plan.one_at_a_time or self.run_plan.free_gpu_for_speech:
            release_whisper_models()
        partial.unlink(missing_ok=True)
        if not ok:
            progress.stage("transcript", "done", "Translation failed")
            return None
        translated = Transcript.model_validate({**data, "language": spoken or data.get("language"), "translated": True})
        check = speech_check.check(translated)
        if not translated.segments or not check.get("reliable", True):
            reason = f" ({check.get('reason')})" if check.get("reason") else ""
            progress.log(
                f"The English translation is not readable either{reason}; the original transcript is kept.",
                "WARNING",
            )
            progress.stage("transcript", "done", "Translation not readable")
            return False
        progress.log(
            "Using the English translation of the speech. Subtitles made from it are in English."
        )
        progress.stage("transcript", "done", "Translated to English")
        return translated

    def _score_transcript(
        self,
        transcript: Transcript,
        config: HighlightConfig,
        progress: PipelineProgress,
        existing: list[dict] | None = None,
        partial: Path | None = None,
    ) -> tuple[list[dict], bool]:
        """Score every transcript segment, reusing cached scores.

        Cached fallback scores are redone when the text model is ready. Scores are saved to
        ``partial`` after every batch, and a later run reuses them. Returns the scores and
        whether they differ from ``existing``."""
        segments = transcript.segments
        if not segments:
            return [], existing != []
        cached = existing if isinstance(existing, list) and len(existing) == len(segments) else None
        text_ready = self.ollama is not None and self.text_enabled
        results: list[dict | None] = list(cached) if cached is not None else [None] * len(segments)
        key = _segments_key(segments) if partial is not None else ""
        restored = _restore_partial_scores(results, read_json(partial) if partial is not None else None, key)
        if restored:
            progress.log(f"Transcript AI: {restored} segment(s) were scored before the interruption")
        pending = [
            index
            for index, item in enumerate(results)
            if item is None or (text_ready and score_source.needs_ai(item))
        ]
        if not pending:
            return results, bool(restored)
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
        out_of_memory = False
        progress.expect("content", progress.text_seconds(len(pending)))
        scored_count = 0
        progress.step("content", 0, len(pending), "segments")
        for batch_no, indexes in enumerate(batches, start=1):
            self._check_cancel()
            batch = [segments[index] for index in indexes]
            progress.log(
                f"Scoring transcript batch {batch_no}/{len(batches)} "
                f"({len(batch)} segment{'s' if len(batch) != 1 else ''})"
            )
            if unreachable or out_of_memory:
                scored = [_heuristic_segment(seg, score_source.AI_FAILED) for seg in batch]
            else:
                try:
                    scored = self._score_text_batch(batch, config, progress)
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
                    if is_memory_error(str(exc)):
                        # Every later request would fail the same way; a later analysis redoes these.
                        out_of_memory = True
                        progress.log(
                            "The speech model ran out of memory; the rest of the transcript uses quick scoring "
                            "and is analyzed again next time. Close other programs or choose a smaller model.",
                            "WARNING",
                        )
                    else:
                        progress.log(f"Transcript batch scoring failed ({exc}); using fallback.", "WARNING")
                    scored = [_heuristic_segment(seg, score_source.AI_FAILED) for seg in batch]
            for index, item in zip(indexes, scored):
                results[index] = item
            if partial is not None:
                atomic_write_json(partial, {"key": key, "scores": results})
            scored_count += len(indexes)
            progress.step("content", scored_count, len(pending), "segments")
        return results, True

    def refine(self, job_dir: Path, config: HighlightConfig, progress: PipelineProgress, cache: dict) -> dict:
        """Plan again from the finished analysis with the AI editor, for changes asked for on
        the Review page. Nothing is analyzed again."""
        if not cache.get("candidates_all") and not cache.get("candidates"):
            raise ValueError("This highlight has no analysis to plan from. Start the analysis again.")
        progress.only("director", "brief", "planning") if config.video_brief else progress.only("director", "planning")
        pool, _note, _embedding = planning_pool(cache, config, self.settings)
        previous = [
            str(clip.get("candidate_id"))
            for clip in (cache.get("clip_plan") or {}).get("clips") or []
            if clip.get("candidate_id") and clip.get("kept", True) is not False
        ]
        plan = self._direct(job_dir, config, pool, cache.get("content_index") or {}, cache, progress, previous=previous)
        if config.video_brief:
            self._brief(
                job_dir,
                config,
                apply_director(pool, plan),
                cache.get("content_index") or {},
                cache,
                progress,
                cache.get("sources") or [],
            )
        progress.stage("planning", "running")
        plan_data = plan_from_cache(cache, config, self.settings)
        _write(job_dir / "clip_plan.json", plan_data)
        cache["clip_plan"] = plan_data
        progress.log(f"New plan: {len(plan_data.get('clips') or [])} clip(s)")
        progress.stage("planning", "done")
        return cache

    def _direct(
        self,
        job_dir: Path,
        config: HighlightConfig,
        candidates: list[dict],
        content: dict,
        cache: dict,
        progress: PipelineProgress,
        previous: list[str] | None = None,
    ) -> dict | None:
        """The AI editor's choice of moments for prompt mode (saved as director_plan.json), or
        None when it is unavailable: the plan is then made from the scores alone."""
        engine = director_engine(config, self.settings)
        claude = None
        progress.stage("director", "running", "Claude is choosing the moments" if engine == "claude" else "")
        if engine == "claude":
            from app.brief.engines.claude_cli import ClaudeCliEngine

            claude = ClaudeCliEngine(self.settings, self.runner)
        elif self.ollama is None or not self.text_enabled:
            progress.log("The AI editor needs the text model; the moments are chosen from their scores.", "WARNING")
            progress.stage("director", "skipped", "Text model not available")
            return None
        model = director_model(self.settings)
        if self.run_plan.one_at_a_time and claude is None:
            self._unload_models(model)
        usage_before = self._usage()
        started = time.perf_counter()
        claude_cwd = job_dir / BRIEF_DIR
        claude_cwd.mkdir(parents=True, exist_ok=True)
        try:
            plan = run_director(
                self.ollama,
                model,
                self.settings,
                config,
                candidates,
                content,
                previous=previous,
                cached=cache.get("director_plan"),
                check_cancel=self._check_cancel,
                log=progress.log,
                on_part=lambda done, total: progress.step("director", done, total, "requests"),
                engine=engine,
                claude=claude,
                claude_cwd=claude_cwd,
            )
        except BriefEngineError as exc:
            # The user chose Claude: the job pauses with the reason instead of switching engines.
            save_failure(job_dir, str(exc), exc.reason, stage="director")
            progress.log(f"Claude could not choose the moments: {exc}", "WARNING")
            raise BriefPaused(str(exc), exc.reason) from exc
        if (read_failure(job_dir) or {}).get("stage") == "director":
            clear_failure(job_dir)
        self._log_usage(progress, "AI editor", usage_before)
        if plan is None:
            progress.log("The AI editor could not make a plan; the moments are chosen from their scores.", "WARNING")
            progress.stage("director", "skipped", "No usable reply")
            return None
        _write(job_dir / DIRECTOR_FILE, plan)
        cache["director_plan"] = plan
        progress.log(
            f"AI editor chose {len(plan['clips'])} moment(s) from {plan.get('considered', len(candidates))} "
            f"in {time.perf_counter() - started:.1f}s: {plan.get('understanding') or ''}".rstrip(": ")
        )
        progress.stage("director", "done")
        return plan

    def _brief(
        self,
        job_dir: Path,
        config: HighlightConfig,
        candidates: list[dict],
        content: dict,
        cache: dict,
        progress: PipelineProgress,
        sources: list[dict],
    ):
        """The video brief (brief.json) with the engine the user chose. The local engine always
        gives one (rules when Ollama cannot); when Claude cannot write it, the job pauses with
        the reason (BriefPaused) and the user decides: retry, or use the local engine."""
        claude = config.brief_engine == "claude"
        progress.stage("brief", "running", "Claude is writing the brief" if claude else "")
        if not claude and self.run_plan.one_at_a_time and self.ollama is not None:
            from app.brief.engines.local import text_model

            self._unload_models(text_model(self.settings))
        usage_before = self._usage()
        try:
            brief = make_brief(
                job_dir,
                config,
                candidates,
                content,
                settings=self.settings,
                ollama=self.ollama if self.text_enabled else None,
                runner=self.runner,
                sources=sources,
                video_name=", ".join(str(s.get("filename") or "") for s in sources if s.get("filename")),
                cached=cache.get("brief"),
                director_plan=cache.get("director_plan") if config.mode == "prompt" else None,
                log=progress.log,
                check_cancel=self._check_cancel,
            )
        except BriefEngineError as exc:
            save_failure(job_dir, str(exc), exc.reason)
            raise BriefPaused(str(exc), exc.reason) from exc
        if not claude:
            self._log_usage(progress, "Brief", usage_before)
        cache["brief"] = brief.model_dump()
        progress.stage("brief", "done")
        return brief

    def _unload_models(self, keep: str | None) -> None:
        """Ask Ollama to free loaded models now (all but ``keep``). Best effort."""
        unload = getattr(self.ollama, "unload_others", None)
        if callable(unload):
            try:
                unload(keep)
            except Exception:  # noqa: BLE001 - freeing memory early is an optimization
                pass

    def _relieve_memory(self, keep: str, progress: PipelineProgress, what: str) -> None:
        """After ``what`` ran out of memory: free everything except the model in use."""
        release_whisper_models()
        release_matcher()
        self._unload_models(keep)
        message = f"{what} ran out of memory. Other models were unloaded and the request is retried in smaller parts."
        if message not in progress.adjustments:
            progress.adjustments.append(message)
            progress.log(message, "WARNING")

    def _score_text_batch(
        self,
        batch: list,
        config: HighlightConfig,
        progress: PipelineProgress | None = None,
        retried: bool = False,
    ) -> list[dict]:
        """AI scores for one batch. A reply cut off by the token limit is retried as two smaller
        batches; so is a request that ran out of memory, after freeing the other models."""
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
            return self._score_text_batch(batch[:middle], config, progress) + self._score_text_batch(
                batch[middle:], config, progress
            )
        except OllamaError as exc:
            if progress is None or retried or not is_memory_error(str(exc)):
                raise
            self._relieve_memory(self.settings.ollama_text_model, progress, "The speech model")
            if len(batch) <= 1:
                return self._score_text_batch(batch, config, progress, retried=True)
            middle = len(batch) // 2
            return self._score_text_batch(batch[:middle], config, progress, retried=True) + self._score_text_batch(
                batch[middle:], config, progress, retried=True
            )
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
        step = getattr(progress, "usage_by_step", {}).setdefault(label, {})
        for name, value in usage.items():
            total[name] = total.get(name, 0) + value
            step[name] = step.get(name, 0) + value
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


def _rescored_candidates(cache: dict, config: HighlightConfig, embedding_fn=None) -> list[dict]:
    """Cached candidates scored for the current style and priorities with the same adjustments
    as the first analysis, left-out moments and (when the config asks for it) repeats removed."""
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
    if config.remove_repetition:
        analyzer = RepetitionAnalyzer(embedding_fn)
        candidates = analyzer.deduplicate(candidates, threshold=analyzer.threshold_for(config.repetition_mode))
    return candidates


def planning_pool(cache: dict, config: HighlightConfig, settings: Settings) -> tuple[list[dict], str, object]:
    """The cached candidates as the planner sees them for ``config``, a note about the chosen
    speaker, and the embedding function used for repeats."""
    embedding_fn = repetition_embedding_fn(_plan_embedder(settings))
    speaker_note = ""
    if config.focus_speaker and cache.get("speakers"):
        # Before repeats are removed: otherwise this speaker's moment could be dropped as a
        # "repeat" of another speaker's similar words.
        from app.analysis.speakers import filter_candidates

        pool = list(cache.get("candidates_all") or cache.get("candidates") or [])
        only = filter_candidates(pool, cache["speakers"], config.focus_speaker)
        if only:
            cache = {**cache, "candidates": only, "candidates_all": only}
        else:
            speaker_note = "No strong moments where that speaker talks most; showing everyone."
    return _rescored_candidates(cache, config, embedding_fn), speaker_note, embedding_fn


def plan_from_cache(cache: dict, config: HighlightConfig, settings: Settings) -> dict:
    candidates, speaker_note, embedding_fn = planning_pool(cache, config, settings)
    if config.mode == "prompt":
        candidates = apply_director(candidates, cache.get("director_plan"))
    if config.video_brief:
        candidates = apply_brief(candidates, cache.get("brief"))
    config = planning_config(config)
    analyzer = RepetitionAnalyzer(embedding_fn)
    similarity_fn = analyzer.as_similarity_fn(candidates) if candidates else None
    optimizer = ClipOptimizer()
    meta = cache.get("metadata") or {}
    source_duration = float(meta.get("duration") or 0)
    multi = int(meta.get("source_count") or 1) > 1
    plan = None
    analysis = cache.get("music_analysis")
    if config.music_led and isinstance(analysis, dict) and analysis.get("beats"):
        plan = music_led_plan(
            candidates,
            config,
            analysis,
            optimizer,
            source_duration=source_duration,
            source_lengths=_source_lengths(cache.get("sources") or [], source_duration),
            multi_source=multi,
            similarity_fn=similarity_fn,
            repeat_threshold=analyzer.threshold_for(config.repetition_mode),
        )
    if plan is None:
        plan = optimizer.plan(
            candidates,
            config,
            source_duration=source_duration,
            multi_source=multi,
            similarity_fn=similarity_fn,
            repeat_threshold=analyzer.threshold_for(config.repetition_mode),
        )
    data = plan.model_dump()
    if speaker_note:
        data["notes"] = f"{speaker_note} {data.get('notes') or ''}".strip()
    return data


def music_led_plan(
    candidates: list[dict],
    config: HighlightConfig,
    analysis: dict,
    optimizer: ClipOptimizer,
    *,
    source_duration: float,
    source_lengths: dict[str, float],
    multi_source: bool,
    similarity_fn=None,
    repeat_threshold: float | None = None,
) -> ClipPlan | None:
    """The clip plan cut to the music (see app.analysis.music), or None when the music gives no
    usable beat or no moment fits (the usual plan is made instead)."""
    plan_slots = music_beat.slots(analysis, music_beat.target_seconds(analysis, config, source_duration))
    if not plan_slots:
        return None
    extra = {} if repeat_threshold is None else {"repeat_threshold": repeat_threshold}
    plan = optimizer.plan(
        music_beat.bias_candidates(candidates, analysis),
        music_beat.planning_config(config, plan_slots),
        source_duration=source_duration,
        multi_source=multi_source,
        similarity_fn=similarity_fn,
        **extra,
    )
    by_id = {str(c.get("id")): c for c in candidates}
    fitted, used = music_beat.fit_clips([c.model_dump() for c in plan.clips], plan_slots, by_id, source_lengths)
    if not fitted:
        return None
    clips = [ClipPlanItem.model_validate(c) for c in fitted]
    length = used[-1][1] - used[0][0]
    name = str(analysis.get("name") or "the music")
    notes = f"Cut to the beat of {name}: {len(clips)} clips at {float(analysis.get('bpm') or 0):.0f} beats per minute, {length:.1f}s."
    return ClipPlan(
        target_duration=round(length, 3),
        estimated_duration=round(sum(c.duration for c in clips), 3),
        clips=clips,
        notes=notes,
        order_mode=plan.order_mode,
    )


def _source_lengths(sources: list[dict], total: float) -> dict[str, float]:
    """Length of each source video by id ("" = the only one), for fitting clips to the beat."""
    lengths = {str(s.get("id") or ""): float(s.get("duration") or 0) for s in sources or [] if s.get("duration")}
    if len(sources or []) <= 1:
        lengths[""] = float(total or 0)
        for key in list(lengths):
            lengths[key] = lengths[key] or float(total or 0)
    return lengths


def alternative_moments(
    cache: dict,
    config: HighlightConfig,
    taken: list[dict],
    *,
    single_source: bool = False,
    limit: int = 12,
) -> list[dict]:
    """
    Strong moments the plan left out, best first, so the user can add one by hand.

    Uses word-based repetition only: no model call, so the Review page gets the list at once.
    """
    candidates = _rescored_candidates(cache, config)
    items = ClipOptimizer().alternatives(
        candidates,
        taken,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        must_exclude=config.must_exclude,
        max_sequence_duration=config.max_sequence_duration,
        min_sequence_duration=config.min_sequence_duration,
        sequence_duration_mode=config.sequence_duration_mode,
        content_priorities=config.content_priorities,
        length_mode=config.length_mode,
        single_source=single_source,
        limit=limit,
    )
    # The browser plays sources by id; it never needs the path on disk.
    return [item.model_dump(exclude={"source_path", "thumbnail"}) for item in items]


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
    """Roughly how long the highlight will be, for sizing the vision shortlist. A range aims
    for its middle; neither can be more than a share of the video (a highlight that long would
    hardly be shortened, and the vision model would look at almost every scene)."""
    limit = source_duration * _SHORTLIST_SOURCE_SHARE if source_duration > 0 else float("inf")
    if getattr(config, "length_mode", "range") == "auto":
        return max(float(config.min_duration), min(source_duration * 0.25, limit))
    middle = (float(config.min_duration) + float(config.max_duration)) / 2
    count = max(1, int(getattr(config, "shorts_count", 0) or 0))
    return max(float(config.min_duration), min(middle * count, limit))


def _needs_proxy(meta: dict) -> bool:
    """Whether scenes and frames should be read from a small copy: a picture larger than 720p,
    or one in a codec that is slow to decode on a CPU."""
    width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
    if width <= 0 or height <= 0 or float(meta.get("duration") or 0) <= 0:
        return False
    short_side = min(width, height)
    codec = str(meta.get("video_codec") or "").lower()
    return short_side > 720 or (codec in _SLOW_CODECS and short_side > _PROXY_SHORT_SIDE)


def _candidate_limit(configured: int, source_duration: float) -> int:
    scaled = int(source_duration / _SECONDS_PER_CANDIDATE)
    return max(int(configured), min(_MAX_CANDIDATES_CEILING, scaled))


def frame_extract_workers(frame_count: int) -> int:
    """Concurrent FFmpeg frame grabs: half the logical CPUs, at most 4, never more than needed."""
    cpus = os.cpu_count() or 2
    return max(1, min(4, cpus // 2, frame_count))


def director_engine(config: HighlightConfig, settings: Settings) -> str:
    """Who is the AI editor for this job: its own choice, or the one in Settings."""
    chosen = str(getattr(config, "director_engine", "") or "") or str(getattr(settings, "director_engine", "local") or "local")
    return "claude" if chosen == "claude" else "local"


def director_model(settings: Settings) -> str:
    """The model of the AI editor: its own setting, or the text model."""
    return str(getattr(settings, "ollama_director_model", "") or "").strip() or settings.ollama_text_model


def _needs_caption(entry) -> bool:
    """A scene the vision model scored without describing it (analyzed before prompt mode)."""
    return isinstance(entry, dict) and entry.get("score_source") == score_source.AI and not entry.get("caption")


def _picture_focus(config: HighlightConfig) -> list[str]:
    """What picture matching compares scenes with: the focus, the priorities, and in prompt mode
    what the wanted moments look like."""
    texts = picture_focus_texts(config.custom_focus, config.content_priorities)
    if config.requirements:
        texts += [cue for cue in config.requirements.visual_cues if cue not in texts]
    return texts


def _segments_key(segments) -> str:
    """Identifies a transcript, so saved scores are only reused for the transcript they belong to."""
    digest = hashlib.sha1()
    for seg in segments:
        digest.update(f"{float(seg.start):.3f}|{float(seg.end):.3f}|{seg.text}\n".encode("utf-8"))
    return digest.hexdigest()


def _restore_partial_scores(results: list, saved, key: str) -> int:
    """Fill ``results`` with segment scores saved by an interrupted run; returns how many."""
    if not key or not isinstance(saved, dict) or saved.get("key") != key:
        return 0
    scores = saved.get("scores")
    if not isinstance(scores, list) or len(scores) != len(results):
        return 0
    restored = 0
    for index, item in enumerate(scores):
        if not isinstance(item, dict) or score_source.needs_ai(item):
            continue
        current = results[index]
        if current is None or score_source.needs_ai(current):
            results[index] = item
            restored += 1
    return restored


def _restore_partial_visual(visual: dict, saved, focus: list[str], scenes: list[dict]) -> dict[str, dict]:
    """Vision-model scores saved by an interrupted run with the same focus. They are added to
    ``visual`` and returned, so the run keeps saving on top of them."""
    if not isinstance(saved, dict) or saved.get("focus") != focus or not isinstance(saved.get("scenes"), dict):
        return {}
    known = {str(scene["scene_id"]) for scene in scenes}
    restored = {
        scene_id: entry
        for scene_id, entry in saved["scenes"].items()
        if scene_id in known and isinstance(entry, dict) and not score_source.needs_ai(entry)
    }
    visual.update(restored)
    return restored


def _file_size(path) -> float:
    try:
        return float(Path(path).stat().st_size)
    except (OSError, TypeError, ValueError):
        return 0.0


def _write(path: Path, data) -> None:
    atomic_write_json(path, data, indent=2)


def _forget_speech(job_dir: Path, cache: dict) -> None:
    """Remove the transcript and everything read from its words, so they are made again."""
    for name in ("transcript.json", TRANSCRIPT_PARTIAL, TRANSLATION_PARTIAL):
        (job_dir / name).unlink(missing_ok=True)
    cache.pop("transcript", None)
    _forget_words(job_dir, cache)


def _forget_words(job_dir: Path, cache: dict) -> None:
    """Remove what was read from the transcript's segments (scores, voices), so it is made again
    from a new transcript."""
    for name in ("segment_scores.json", SEGMENT_SCORES_PARTIAL, "speakers.json"):
        (job_dir / name).unlink(missing_ok=True)
    for key in ("segment_scores", "speakers"):
        cache.pop(key, None)


def _cleanup_frames(frames_dir: Path) -> None:
    if not frames_dir.exists():
        return
    for file in frames_dir.glob("*.jpg"):
        try:
            file.unlink()
        except OSError:
            pass
