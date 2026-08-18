import type { HighlightForm, Job, OllamaStatus } from "../types";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error((data as { error?: string }).error || response.statusText);
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

export function formatBytes(bytes?: number): string {
  const value = bytes || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}

export function configPayload(form: HighlightForm) {
  return {
    min_duration: form.min_duration,
    max_duration: form.max_duration,
    style: form.style,
    custom_focus: form.custom_focus,
    preserve_context: form.preserve_context,
    remove_repetition: form.remove_repetition,
    allow_repeated_content: form.allow_repeated_content,
    max_sequence_duration: form.max_sequence_duration,
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
  ollama: () => http<OllamaStatus>("/api/system/ollama/status"),
  info: () => http<Record<string, unknown>>("/api/system/info"),
  pullModel: (model: string) =>
    http("/api/system/ollama/pull", {
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
