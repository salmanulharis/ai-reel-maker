import type { Clip, JobSource } from "../types";

/** Shortest clip the preview controls will create when setting a start or end point. */
export const MIN_CLIP_SECONDS = 0.2;

/** Step used by the fine back and forward buttons. */
export const FINE_STEP_SECONDS = 0.1;

/** Times saved from the player are rounded to hundredths, like the other clip times. */
export function roundTime(seconds: number): number {
  return Math.round(seconds * 100) / 100;
}

/** "1:05.3" or "1:02:05.3": precise enough to place a cut. */
export function formatPrecise(seconds?: number): string {
  const tenths = Math.max(0, Math.round((seconds || 0) * 10));
  const whole = Math.floor(tenths / 10);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = String(whole % 60).padStart(2, "0");
  const tail = `${s}.${tenths % 10}`;
  return h ? `${h}:${String(m).padStart(2, "0")}:${tail}` : `${m}:${tail}`;
}

/** The source a clip comes from; clips without one belong to the first video. */
export function clipSourceId(clip: Clip, sources: JobSource[]): string {
  return clip.source_id || sources[0]?.id || "v0";
}

/** Address of the video the preview plays for a clip. With one source this is the same
 *  address the Analysis page uses (the cropped working copy when a crop is set). */
export function previewUrl(jobId: string, clip: Clip, sources: JobSource[]): string {
  if (sources.length > 1) {
    return `/api/media/${encodeURIComponent(jobId)}/source/${encodeURIComponent(clipSourceId(clip, sources))}`;
  }
  return `/api/media/${encodeURIComponent(jobId)}/source`;
}

/** Index of the next kept clip after `index` in highlight order, or -1 when there is none.
 *  Pass -1 to get the first kept clip. */
export function nextKept(clips: Clip[], index: number): number {
  for (let i = Math.max(-1, index) + 1; i < clips.length; i += 1) {
    if (clips[i].kept !== false) return i;
  }
  return -1;
}

/**
 * The new start or end when it is set to the player's time, or null when that would leave
 * a clip shorter than MIN_CLIP_SECONDS (or the time is not a usable number).
 * `limit` is the video length when known; an end is never placed after it.
 */
export function trimTo(edge: "start" | "end", time: number, clip: Clip, limit?: number): number | null {
  if (!Number.isFinite(time) || time < 0) return null;
  const at = roundTime(limit && limit > 0 ? Math.min(time, limit) : time);
  if (edge === "start") return at <= clip.end - MIN_CLIP_SECONDS ? at : null;
  return at >= clip.start + MIN_CLIP_SECONDS ? at : null;
}

/** True when the playhead has reached the end of the clip being previewed. A small
 *  tolerance keeps a frame that lands just before the end from slipping past it. */
export function reachedEnd(time: number, clip: Clip): boolean {
  return time >= clip.end - 0.02;
}
