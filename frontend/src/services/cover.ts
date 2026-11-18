import type { Clip } from "../types";

/** Address of a frame of the finished highlight; `width` asks for a small preview. */
export function coverUrl(jobId: string, seconds: number, width?: number): string {
  const params = new URLSearchParams({ t: seconds.toFixed(2) });
  if (width) params.set("width", String(width));
  return `/api/jobs/${encodeURIComponent(jobId)}/cover?${params}`;
}

/**
 * Suggested cover frames, as seconds into the finished highlight: a frame a little into each
 * of the best-scoring kept clips (kept clips play back to back in list order), earliest first.
 */
export function coverSuggestions(clips: Clip[], count = 4): number[] {
  let offset = 0;
  const spots: Array<{ time: number; score: number }> = [];
  for (const clip of clips) {
    const length = clip.end - clip.start;
    if (clip.kept === false || !(length > 0)) continue;
    spots.push({ time: offset + Math.min(length * 0.4, Math.max(0, length - 0.1)), score: clip.score || 0 });
    offset += length;
  }
  return spots
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(0, count))
    .map((spot) => Math.round(spot.time * 10) / 10)
    .sort((a, b) => a - b);
}
