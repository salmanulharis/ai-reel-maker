import { messages } from "../i18n";
import type { HighlightForm, Job, OllamaStatus, OptimizationPlan } from "../types";

// Short server messages that are worth rewording for people.
const SERVER_MESSAGES: Record<string, string> = {
  "file is required": "Choose a video file first.",
  "no output": "This highlight has no finished video yet.",
  "invalid model name": "That model name is not valid.",
};

/** A message people can act on, instead of an HTTP status or a raw server error. */
export function friendlyError(status: number, message?: string): string {
  if (status === 404) return messages.errors.notFound;
  if (status === 413) return messages.errors.tooLarge;
  const text = (message || "").trim();
  if (!text) return messages.errors.server;
  const known = SERVER_MESSAGES[text.toLowerCase()];
  if (known) return known;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    // fetch only rejects when the server cannot be reached at all.
    throw new Error(messages.errors.offline);
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(friendlyError(response.status, (data as { error?: string }).error));
  }
  return data as T;
}

export function parseRanges(text: string): [number, number][] {
  return text
    .split(/[,\n]/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [a, b] = part.split(/–|-|to/i).map((s) => s.trim());
      return [toSeconds(a), toSeconds(b)] as [number, number];
    })
    .filter((pair) => Number.isFinite(pair[0]) && Number.isFinite(pair[1]));
}

export function toSeconds(value: string): number {
  if (!value) return NaN;
  if (/^\d+(\.\d+)?$/.test(value)) return Number(value);
  const parts = value.split(":").map(Number);
  if (parts.some(Number.isNaN)) return NaN;
  while (parts.length < 3) parts.unshift(0);
  const [h, m, s] = parts;
  return h * 3600 + m * 60 + s;
}

export function formatTime(seconds?: number): string {
  const total = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Processing time in words: "12.4 seconds", "2 min 15 sec", "1 hr 5 min".
 *  live: whole seconds under a minute ("12 sec"), so a ticking counter does not flicker. */
export function formatElapsed(seconds?: number | null, live = false): string {
  const value = Math.max(0, Number(seconds) || 0);
  if (value < 60) return live ? `${Math.floor(value)} sec` : `${value.toFixed(1)} seconds`;
  const total = Math.floor(value);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return minutes ? `${hours} hr ${minutes} min` : `${hours} hr`;
  return secs ? `${minutes} min ${secs} sec` : `${minutes} min`;
}

export function formatBytes(bytes?: number): string {
  const value = bytes || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}

function finiteOr(value: number, fallback: number) {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export function configPayload(form: HighlightForm) {
  // A length field can be empty while it is being edited; never send that to the server.
  const max = finiteOr(form.max_duration, finiteOr(form.min_duration, 30));
  return {
    min_duration: finiteOr(form.min_duration, Math.min(20, max)),
    max_duration: max,
    length_mode: form.social_short ? "range" : form.length_mode,
    use_speech: form.use_speech,
    style: form.style,
    custom_focus: form.custom_focus,
    preserve_context: form.preserve_context,
    remove_repetition: form.remove_repetition,
    allow_repeated_content: form.allow_repeated_content,
    max_sequence_duration: form.max_sequence_duration,
    min_sequence_duration: form.min_sequence_duration,
    sequence_duration_mode: form.sequence_duration_mode,
    content_priorities: form.content_priorities,
    priority_weights: form.priority_weights,
    max_repeated_duration: form.max_repeated_duration,
    repetition_mode: form.repetition_mode,
    order_mode: form.order_mode,
    social_short: form.social_short,
    aspect_ratio: form.aspect_ratio,
    subtitles: form.subtitles,
    transition: form.transition,
    frame_rate: form.frame_rate,
    smooth_motion: form.smooth_motion,
    codec: form.codec,
    quality: form.quality,
    resolution: form.resolution,
    must_include: parseRanges(form.must_include),
    must_exclude: parseRanges(form.must_exclude),
    crop: form.cropEnabled ? form.crop : null,
  };
}

export const api = {
  jobs: () => http<{ jobs: Job[] }>("/api/jobs"),
  job: (id: string) => http<Job>(`/api/jobs/${id}`),
  createJob: async (files: File | File[], form: HighlightForm) => {
    const body = new FormData();
    const list = Array.isArray(files) ? files : [files];
    for (const file of list) body.append("file", file);
    body.append("config", JSON.stringify(configPayload(form)));
    return http<Job>("/api/jobs", { method: "POST", body });
  },
  analyze: (id: string, form: HighlightForm) =>
    http<Job>(`/api/jobs/${id}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configPayload(form)),
    }),
  render: (id: string, clips?: unknown) =>
    http<Job>(`/api/jobs/${id}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clips }),
    }),
  cancel: (id: string) => http<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  retry: (id: string) => http<Job>(`/api/jobs/${id}/retry`, { method: "POST" }),
  regenerate: (id: string, preset: string) =>
    http<{ job: Job; plan: unknown }>(`/api/jobs/${id}/regenerate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset }),
    }),
  clips: (id: string, clips: unknown) =>
    http(`/api/jobs/${id}/clips`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clips }),
    }),
  logs: (id: string) => http<{ logs: Array<{ level: string; message: string; ts?: number }> }>(`/api/jobs/${id}/logs`),
  analysis: (id: string) => http<Record<string, unknown>>(`/api/jobs/${id}/analysis`),
  timeline: (id: string) => http<Record<string, unknown>>(`/api/jobs/${id}/timeline`),
  deleteJob: (id: string) => http(`/api/jobs/${id}`, { method: "DELETE" }),
  openFolder: (id: string) => http(`/api/jobs/${id}/open-folder`, { method: "POST" }),
  ollama: () => http<OllamaStatus>("/api/system/ollama/status"),
  info: (opts?: { refresh?: boolean; encoders?: boolean }) => {
    const params = new URLSearchParams();
    if (opts?.refresh) params.set("refresh", "1");
    if (opts?.encoders) params.set("encoders", "1");
    const query = params.toString();
    return http<Record<string, unknown>>(`/api/system/info${query ? `?${query}` : ""}`);
  },
  pullModel: (model: string) =>
    http("/api/system/ollama/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    }),
  pullModels: (models: string[]) =>
    http("/api/system/ollama/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models }),
    }),
  applyRecommended: () => http<Record<string, unknown>>("/api/system/ai/apply-recommended", { method: "POST" }),
  optimization: (refresh = false) =>
    http<OptimizationPlan>(`/api/system/ai/optimization${refresh ? "?refresh=1" : ""}`),
  setOptimization: (enabled: boolean) =>
    http<OptimizationPlan>("/api/system/ai/optimization", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    }),
  undoOptimization: () => http<OptimizationPlan>("/api/system/ai/optimization/undo", { method: "POST" }),
  startAiTest: () => http("/api/system/ai/test", { method: "POST" }),
  downloadImageModel: () => http("/api/system/image-model/download", { method: "POST" }),
  downloadWhisper: (model: string) =>
    http("/api/system/whisper/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    }),
  settings: () => http<{ settings: Record<string, unknown>; storage: Record<string, number | string> }>("/api/settings"),
  saveSettings: (payload: Record<string, unknown>) =>
    http("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  clearTmp: () => http("/api/settings/storage/clear-tmp", { method: "POST" }),
  clearCache: () => http("/api/settings/storage/clear-cache", { method: "POST" }),
  deleteCompleted: () => http("/api/settings/storage/delete-completed", { method: "POST" }),
};
