import type { HighlightForm } from "../types";

/** Choices that belong to one video (crop, time ranges); a preset never stores or changes them. */
export const VIDEO_SPECIFIC: ReadonlyArray<keyof HighlightForm> = [
  "crop",
  "cropEnabled",
  "must_include",
  "must_exclude",
  "music_file_name",
];

export interface Preset {
  name: string;
  form: Partial<HighlightForm>;
  saved_at?: number;
}

/** The part of the form a preset keeps. */
export function presetPayload(form: HighlightForm): Partial<HighlightForm> {
  const copy: Partial<HighlightForm> = { ...form };
  for (const key of VIDEO_SPECIFIC) delete copy[key];
  return copy;
}

/** Settings that may be empty (null) and otherwise hold a number. */
const NULLABLE_NUMBERS: ReadonlyArray<string> = ["min_sequence_duration"];

function sameKind(key: string, current: unknown, value: unknown): boolean {
  if (NULLABLE_NUMBERS.includes(key)) return value === null || typeof value === "number";
  if (current === null || value === null) return false;
  if (Array.isArray(current)) return Array.isArray(value);
  return typeof current === typeof value && !Array.isArray(value);
}

/**
 * The form with a preset's choices applied. Only known settings whose type matches are
 * taken, so a preset saved by an older version (or edited by hand) cannot break the form;
 * anything missing keeps its current value, and the video's crop and time ranges stay.
 */
export function applyPreset(current: HighlightForm, saved: Partial<HighlightForm>): HighlightForm {
  const next: HighlightForm = { ...current };
  const target = next as unknown as Record<string, unknown>;
  for (const [key, value] of Object.entries(saved || {})) {
    if (!(key in current) || (VIDEO_SPECIFIC as ReadonlyArray<string>).includes(key)) continue;
    const existing = (current as unknown as Record<string, unknown>)[key];
    if (value === undefined || !sameKind(key, existing, value)) continue;
    if (typeof value === "number" && !Number.isFinite(value)) continue;
    if (Array.isArray(value)) target[key] = value.filter((item) => typeof item === "string");
    else if (value && typeof value === "object") target[key] = { ...(value as object) };
    else target[key] = value;
  }
  return next;
}

/** An existing preset with this name (names are matched without case), if any. */
export function findPreset(presets: Preset[], name: string): Preset | undefined {
  const wanted = name.trim().replace(/\s+/g, " ").toLocaleLowerCase();
  return presets.find((preset) => preset.name.toLocaleLowerCase() === wanted);
}
