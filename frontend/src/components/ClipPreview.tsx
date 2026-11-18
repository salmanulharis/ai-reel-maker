import { forwardRef, useCallback, useEffect, useRef, useState } from "react";
import { fill, messages } from "../i18n";
import { Spinner } from "./Feedback";
import {
  FINE_STEP_SECONDS,
  clipSourceId,
  formatPrecise,
  nextKept,
  previewUrl,
  reachedEnd,
  trimTo,
} from "../services/clipPreview";
import type { Clip, JobSource } from "../types";

const copy = messages.review.preview;

type Mode = "idle" | "clip" | "all";

export interface ClipPreviewState {
  /** Attach to the <video>. A callback, so listeners are added whenever the player mounts
   *  (the page first shows a loader without it). */
  videoRef: (element: HTMLVideoElement | null) => void;
  /** Length of the loaded video in seconds, when known. */
  duration?: number;
  /** Index of the clip the player is set to, or -1. */
  selected: number;
  mode: Mode;
  /** Player position in seconds of the loaded source. */
  time: number;
  /** Source id of the video in the player, or "" before anything is loaded. */
  loadedSource: string;
  loading: boolean;
  failed: boolean;
  /** Metadata of the loaded video is available, so seeking and trimming work. */
  ready: boolean;
  playClip: (index: number) => void;
  playAll: () => void;
  pause: () => void;
  step: (seconds: number) => void;
  /** Seek to an edited start or end so the new cut point is visible (only while paused). */
  showEdge: (index: number, edge: "start" | "end", value: number) => void;
  /** The selected clip moved to another place in the list; keep it selected there. */
  retarget: (index: number) => void;
  clear: () => void;
}

/**
 * Plays clips straight from the source video, so a cut can be checked without creating the
 * highlight. One player is shared by all clips; "play all" runs the kept clips in highlight
 * order. Nothing is rendered or uploaded.
 */
export function useClipPreview(jobId: string, clips: Clip[], sources: JobSource[]): ClipPreviewState {
  const videoNode = useRef<HTMLVideoElement | null>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const videoRef = useCallback((element: HTMLVideoElement | null) => {
    videoNode.current = element;
    setVideoEl(element);
  }, []);
  const [duration, setDuration] = useState<number | undefined>(undefined);
  const [selected, setSelected] = useState(-1);
  const [mode, setMode] = useState<Mode>("idle");
  const [time, setTime] = useState(0);
  const [loadedSource, setLoadedSource] = useState("");
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  const [ready, setReady] = useState(false);

  // The playback loop and event handlers read the latest values through refs.
  const clipsRef = useRef(clips);
  clipsRef.current = clips;
  const sourcesRef = useRef(sources);
  sourcesRef.current = sources;
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const modeRef = useRef<Mode>(mode);
  modeRef.current = mode;
  /** Raised while the player switches clips itself, so its own pause is not read as the user's. */
  const switching = useRef(false);
  /** Each load gets a number; a slower, older load that finishes later is ignored. */
  const loadToken = useRef(0);
  const lastTime = useRef(0);

  const setModeNow = useCallback((next: Mode) => {
    modeRef.current = next;
    setMode(next);
  }, []);

  const selectNow = useCallback((index: number) => {
    selectedRef.current = index;
    setSelected(index);
  }, []);

  /**
   * Load the video a clip comes from (if it is not already loaded) and seek to `at`.
   * "stale": a newer request replaced this one, which must then leave the state alone.
   */
  const loadAndSeek = useCallback(
    async (clip: Clip, at: number): Promise<"ok" | "failed" | "stale"> => {
      const video = videoNode.current;
      if (!video) return "failed";
      const url = previewUrl(jobId, clip, sourcesRef.current);
      const token = ++loadToken.current;
      if (video.getAttribute("src") !== url || video.readyState < HTMLMediaElement.HAVE_METADATA) {
        setReady(false);
        setFailed(false);
        setLoading(true);
        const loaded = await new Promise<boolean>((resolve) => {
          const done = (ok: boolean) => {
            video.removeEventListener("loadedmetadata", onLoaded);
            video.removeEventListener("error", onError);
            resolve(ok);
          };
          const onLoaded = () => done(true);
          const onError = () => done(false);
          video.addEventListener("loadedmetadata", onLoaded);
          video.addEventListener("error", onError);
          // Also reloads the same address when an earlier attempt failed or did not finish.
          if (video.getAttribute("src") !== url) video.setAttribute("src", url);
          video.load();
        });
        if (token !== loadToken.current) return "stale";
        setLoading(false);
        if (!loaded) {
          setFailed(true);
          return "failed";
        }
        setLoadedSource(clipSourceId(clip, sourcesRef.current));
        setReady(true);
      } else {
        setLoadedSource(clipSourceId(clip, sourcesRef.current));
        setReady(true);
      }
      video.currentTime = at;
      lastTime.current = at;
      setTime(at);
      return "ok";
    },
    [jobId],
  );

  const startPlayback = useCallback(
    async (index: number, nextMode: Mode) => {
      const clip = clipsRef.current[index];
      const video = videoNode.current;
      if (!clip || !video) return;
      switching.current = true;
      video.pause();
      selectNow(index);
      setModeNow(nextMode);
      const result = await loadAndSeek(clip, clip.start);
      // A newer request (another clip, or the preview being cleared) owns the player now.
      if (result === "stale") return;
      switching.current = false;
      if (result === "failed") {
        setModeNow("idle");
        return;
      }
      if (selectedRef.current !== index || modeRef.current !== nextMode) return;
      try {
        await video.play();
      } catch {
        // The browser refused to start playback (for example, no user gesture). Stay paused.
        setModeNow("idle");
      }
    },
    [loadAndSeek, selectNow, setModeNow],
  );

  const playClip = useCallback((index: number) => void startPlayback(index, "clip"), [startPlayback]);

  const playAll = useCallback(() => {
    const first = nextKept(clipsRef.current, -1);
    if (first >= 0) void startPlayback(first, "all");
  }, [startPlayback]);

  const pause = useCallback(() => {
    setModeNow("idle");
    videoNode.current?.pause();
  }, [setModeNow]);

  const step = useCallback(
    (seconds: number) => {
      const video = videoNode.current;
      if (!video || !ready) return;
      pause();
      const limit = Number.isFinite(video.duration) ? video.duration : Infinity;
      const at = Math.min(limit, Math.max(0, video.currentTime + seconds));
      video.currentTime = at;
      lastTime.current = at;
      setTime(at);
    },
    [pause, ready],
  );

  const showEdge = useCallback(
    (index: number, edge: "start" | "end", value: number) => {
      const video = videoNode.current;
      const clip = clipsRef.current[index];
      if (!video || !clip || !ready || index !== selectedRef.current || modeRef.current !== "idle") return;
      if (clipSourceId(clip, sourcesRef.current) !== loadedSource || !Number.isFinite(value)) return;
      // The last frame before the end is what the viewer sees last.
      const at = edge === "start" ? value : Math.max(0, value - 0.05);
      const limit = Number.isFinite(video.duration) ? video.duration : Infinity;
      video.currentTime = Math.min(limit, Math.max(0, at));
      lastTime.current = video.currentTime;
      setTime(video.currentTime);
    },
    [loadedSource, ready],
  );

  const clear = useCallback(() => {
    loadToken.current += 1;
    switching.current = false;
    videoNode.current?.pause();
    setModeNow("idle");
    selectNow(-1);
    setLoading(false);
  }, [selectNow, setModeNow]);

  // Stop at the end of a clip, or move on to the next kept clip. A frame-by-frame check
  // stops much closer to the cut than the browser's few-times-a-second time updates.
  useEffect(() => {
    if (mode === "idle") return;
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const video = videoNode.current;
      if (!video || video.paused || switching.current) return;
      const clip = clipsRef.current[selectedRef.current];
      if (!clip) {
        pause();
        return;
      }
      const now = video.currentTime;
      const before = lastTime.current;
      lastTime.current = now;
      // Only playback that runs into the end counts; a jump with the seek bar does not.
      const crossed = before < clip.end && reachedEnd(now, clip) && now - before < 1;
      if (!crossed) return;
      if (modeRef.current === "clip") {
        pause();
        return;
      }
      const next = nextKept(clipsRef.current, selectedRef.current);
      if (next < 0) {
        pause();
        return;
      }
      const nextClip = clipsRef.current[next];
      if (previewUrl(jobId, nextClip, sourcesRef.current) === video.getAttribute("src")) {
        selectNow(next);
        video.currentTime = nextClip.start;
        lastTime.current = nextClip.start;
      } else {
        void startPlayback(next, "all");
      }
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [mode, jobId, pause, selectNow, startPlayback]);

  // Keep the shown time and state in step with the player, including its own controls.
  useEffect(() => {
    const video = videoEl;
    if (!video) return;
    const onTime = () => setTime(video.currentTime);
    const onDuration = () => setDuration(Number.isFinite(video.duration) ? video.duration : undefined);
    // Starting the player's own play button inside a clip stops at that clip's end, as
    // "Play clip" does. Playing on from beyond the end is not stopped (see the crossing check).
    const onPlay = () => {
      if (switching.current || modeRef.current !== "idle" || selectedRef.current < 0) return;
      lastTime.current = video.currentTime;
      setModeNow("clip");
    };
    const onPause = () => {
      if (!switching.current) setModeNow("idle");
    };
    const onWaiting = () => setLoading(true);
    const onPlaying = () => setLoading(false);
    const onError = () => {
      if (!video.getAttribute("src")) return;
      setFailed(true);
      setLoading(false);
      setModeNow("idle");
    };
    video.addEventListener("timeupdate", onTime);
    video.addEventListener("seeked", onTime);
    video.addEventListener("durationchange", onDuration);
    video.addEventListener("emptied", onDuration);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("waiting", onWaiting);
    video.addEventListener("playing", onPlaying);
    video.addEventListener("canplay", onPlaying);
    video.addEventListener("error", onError);
    return () => {
      video.removeEventListener("timeupdate", onTime);
      video.removeEventListener("seeked", onTime);
      video.removeEventListener("durationchange", onDuration);
      video.removeEventListener("emptied", onDuration);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("waiting", onWaiting);
      video.removeEventListener("playing", onPlaying);
      video.removeEventListener("canplay", onPlaying);
      video.removeEventListener("error", onError);
    };
  }, [videoEl, setModeNow]);

  // A new set of clips (for example after choosing new clips) may not have this index.
  useEffect(() => {
    if (selected >= clips.length) clear();
  }, [clips.length, selected, clear]);

  return {
    videoRef,
    duration,
    selected,
    mode,
    time,
    loadedSource,
    loading,
    failed,
    ready,
    playClip,
    playAll,
    pause,
    step,
    showEdge,
    retarget: selectNow,
    clear,
  };
}

interface PanelProps {
  preview: ClipPreviewState;
  clips: Clip[];
  sources: JobSource[];
  /** Save a new start or end for a clip. */
  onTrim: (index: number, edge: "start" | "end", value: number) => void;
}

/** The shared player with controls for the selected clip. */
export const ClipPreviewPanel = forwardRef<HTMLElement, PanelProps>(function ClipPreviewPanel(
  { preview, clips, sources, onTrim },
  ref,
) {
  const { videoRef, duration, selected, mode, time, loading, failed, ready } = preview;
  const clip = selected >= 0 ? clips[selected] : undefined;
  const hasKept = nextKept(clips, -1) >= 0;
  const label = clip ? fill(messages.review.clipNumber, { n: selected + 1 }) : "";
  const sameSource = !!clip && clipSourceId(clip, sources) === preview.loadedSource;
  const canEdit = !!clip && clip.kept !== false && ready && !failed && sameSource;
  const newStart = canEdit && clip ? trimTo("start", time, clip, duration) : null;
  const newEnd = canEdit && clip ? trimTo("end", time, clip, duration) : null;
  const playing = mode !== "idle";

  let status = copy.idle;
  if (failed) status = copy.unsupported;
  else if (mode === "all" && clip) {
    const order = clips.filter((c) => c.kept !== false).indexOf(clip) + 1;
    const total = clips.filter((c) => c.kept !== false).length;
    status = fill(copy.playingAll, { clip: label, n: order, total });
  } else if (mode === "clip" && clip) status = fill(copy.playingClip, { clip: label });
  else if (clip) status = fill(copy.paused, { clip: label });

  return (
    <section className="panel preview-panel" ref={ref} aria-labelledby="clip-preview-title">
      <h3 id="clip-preview-title" className="sub-title">
        {copy.title}
      </h3>
      <div className={`preview-frame ${clip ? "" : "empty"}`}>
        <video
          ref={videoRef}
          className="preview"
          controls={!!clip && !failed}
          preload="metadata"
          playsInline
          aria-label={clip ? fill(copy.videoLabel, { clip: label }) : copy.title}
          hidden={!clip || failed}
        />
        {!clip && <p className="preview-placeholder">{copy.empty}</p>}
        {loading && !failed && (
          <span className="preview-loading">
            <Spinner label={copy.loading} />
          </span>
        )}
      </div>
      <p className="preview-status" role="status" aria-live="polite">
        {status}
      </p>
      {clip && !failed && (
        <p className="preview-time">
          <span>{copy.position}</span> <strong>{formatPrecise(time)}</strong>
          <span className="muted">
            {" "}
            · {fill(copy.clipRange, { start: formatPrecise(clip.start), end: formatPrecise(clip.end) })}
          </span>
        </p>
      )}
      <div className="btn-row preview-controls">
        {playing ? (
          <button type="button" className="btn sm primary" onClick={preview.pause}>
            {copy.pause}
          </button>
        ) : (
          <button
            type="button"
            className="btn sm primary"
            disabled={!clip || failed}
            onClick={() => preview.playClip(selected)}
          >
            {clip ? fill(copy.playClip, { clip: label }) : copy.playSelected}
          </button>
        )}
        <button type="button" className="btn sm" disabled={!hasKept || failed} onClick={preview.playAll}>
          {copy.playAll}
        </button>
      </div>
      <div className="btn-row preview-controls">
        <button
          type="button"
          className="btn sm ghost"
          disabled={!clip || !ready || failed}
          aria-label={copy.backLabel}
          onClick={() => preview.step(-FINE_STEP_SECONDS)}
        >
          {copy.back}
        </button>
        <button
          type="button"
          className="btn sm ghost"
          disabled={!clip || !ready || failed}
          aria-label={copy.forwardLabel}
          onClick={() => preview.step(FINE_STEP_SECONDS)}
        >
          {copy.forward}
        </button>
        <button
          type="button"
          className="btn sm"
          disabled={newStart === null}
          onClick={() => newStart !== null && onTrim(selected, "start", newStart)}
        >
          {copy.setStart}
        </button>
        <button
          type="button"
          className="btn sm"
          disabled={newEnd === null}
          onClick={() => newEnd !== null && onTrim(selected, "end", newEnd)}
        >
          {copy.setEnd}
        </button>
      </div>
      {clip && clip.kept === false && <p className="field-hint">{copy.removedHint}</p>}
      {canEdit && newStart === null && newEnd !== null && <p className="field-hint">{copy.startTooLate}</p>}
      {canEdit && newEnd === null && newStart !== null && <p className="field-hint">{copy.endTooEarly}</p>}
      <p className="field-hint">{copy.hint}</p>
    </section>
  );
});
