import type { Clip, JobSource, Moment } from "../types";

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
export function clipSourceId(clip: Clip | Moment, sources: JobSource[]): string {
  return clip.source_id || sources[0]?.id || "v0";
}

/** Address of the video the preview plays for a clip. With one source this is the same
 *  address the Analysis page uses (the cropped working copy when a crop is set). */
export function previewUrl(jobId: string, clip: Clip | Moment, sources: JobSource[]): string {
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
export function reachedEnd(time: number, clip: Clip | Moment): boolean {
  return time >= clip.end - 0.02;
}

/** Stable identity for a suggested moment (it has no saved id). */
export function momentKey(moment: Moment, sources: JobSource[]): string {
  return `${clipSourceId(moment, sources)}:${moment.start}-${moment.end}`;
}

/** True when the moment shares any time with a clip from the same video (kept or removed). */
export function overlapsClips(moment: Moment, clips: Clip[], sources: JobSource[]): boolean {
  const source = clipSourceId(moment, sources);
  return clips.some(
    (clip) => clipSourceId(clip, sources) === source && moment.start < clip.end && clip.start < moment.end,
  );
}

/**
 * Where an added moment goes in the clip list. In time order it goes among the clips by
 * video and start time; for other orders it goes last, where the user can move it.
 */
export function insertIndex(clips: Clip[], moment: Moment, sources: JobSource[], chronological: boolean): number {
  if (!chronological) return clips.length;
  const order = (item: Clip | Moment) => Math.max(0, sources.findIndex((s) => s.id === clipSourceId(item, sources)));
  const after = clips.findIndex(
    (clip) => order(clip) > order(moment) || (order(clip) === order(moment) && clip.start > moment.start),
  );
  return after < 0 ? clips.length : after;
}

/** The short a clip belongs to (0 for a single highlight). */
export function shortIndex(item: Clip | Moment): number {
  const value = Number((item as Clip).short_index || 0);
  return Number.isFinite(value) && value > 0 ? Math.trunc(value) : 0;
}

/** The short whose clips are closest in time to a moment, in the same video (0 when none). */
export function nearestShort(clips: Clip[], moment: Moment, sources: JobSource[]): number {
  const middle = (moment.start + moment.end) / 2;
  const source = clipSourceId(moment, sources);
  let best = clips.length ? shortIndex(clips[0]) : 0;
  let bestDistance = Infinity;
  for (const clip of clips) {
    if (clipSourceId(clip, sources) !== source) continue;
    const distance = Math.abs((clip.start + clip.end) / 2 - middle);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = shortIndex(clip);
    }
  }
  return best;
}

/** Where a moment goes in a short: among that short's clips by time, or after the last one. */
export function insertIndexInShort(clips: Clip[], moment: Moment, short: number): number {
  let last = -1;
  for (let i = 0; i < clips.length; i += 1) {
    if (shortIndex(clips[i]) !== short) continue;
    if (clips[i].start > moment.start) return i;
    last = i;
  }
  return last >= 0 ? last + 1 : clips.length;
}

/** A kept clip made from a suggested moment. */
export function momentToClip(moment: Moment, short = 0): Clip {
  return {
    start: moment.start,
    end: moment.end,
    score: moment.score,
    reason: moment.reason,
    duration_reason: moment.duration_reason,
    transcript: moment.transcript,
    source_id: moment.source_id,
    source_name: moment.source_name,
    kept: true,
    ...(short > 0 ? { short_index: short } : {}),
  };
}
