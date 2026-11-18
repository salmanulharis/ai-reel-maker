import type { Clip } from "../types";

export interface Strength {
  /** 0-100, an estimate from the AI scores, not a prediction of views. */
  score: number;
  opening: number;
  moments: number;
  length: number;
}

/**
 * How strong a highlight or short looks, from what the app already knows: how strong its
 * opening clip is, how strong its clips are on average, and how well it fits the length goal.
 */
export function strength(clips: Clip[], minLength: number, maxLength: number): Strength | null {
  const kept = clips.filter((clip) => clip.kept !== false && clip.end > clip.start);
  if (!kept.length) return null;
  const opening = Math.round((kept[0].score || 0) * 100);
  const moments = Math.round((kept.reduce((sum, clip) => sum + (clip.score || 0), 0) / kept.length) * 100);
  const total = kept.reduce((sum, clip) => sum + (clip.end - clip.start), 0);
  let length = 100;
  if (minLength > 0 && total < minLength * 0.9) length = Math.round((100 * total) / minLength);
  else if (maxLength > 0 && total > maxLength * 1.05) length = Math.round((100 * maxLength) / total);
  const score = Math.round(0.35 * opening + 0.45 * moments + 0.2 * length);
  return { score: Math.max(0, Math.min(100, score)), opening, moments, length };
}
