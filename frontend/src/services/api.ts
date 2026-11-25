import { messages } from "../i18n";
import type {
  JobInsights,
  LogEntry,
  AiPlanInfo,
  BriefInfo,
  EngineStatus,
  HighlightForm,
  InterpretResult,
  Job,
  Moment,
  MeasureStatus,
  OllamaStatus,
  NarrationState,
  OptimizationGoal,
  OptimizationPlan,
  VideoBrief,
  VideoToolsStatus,
} from "../types";
import type { Preset } from "./presets";
import { coverUrl } from "./cover";

// Short server messages that are worth rewording for people.
const SERVER_MESSAGES: Record<string, string> = {
  "file is required": "Choose a video file first.",
  "no output": "This highlight has no finished video yet.",
  "invalid model name": "That model name is not valid.",
  "no clips to export": "Keep at least one clip to download a timeline.",
  "source video missing": "The original video file is missing, so a timeline can't be made.",
  "preset name is required": "Give the preset a name.",
  "preset not found": "That preset no longer exists.",
  "could not read a frame": "That frame couldn't be read from the video. Try another moment.",
  "no clips to describe": "Keep at least one clip to get suggestions.",
  "unsupported music format": "Choose an MP3, WAV, M4A, AAC, OGG, FLAC or Opus file.",
  "music file is too large": "That music file is too large (at most 200 MB).",
  "unsupported image format": "Choose a PNG, JPG or WebP image.",
  "image is too large": "That image is too large (at most 5 MB).",
  "choose a folder to watch": "Type the folder to watch.",
  "that folder does not exist": "That folder doesn't exist on this computer.",
  "that preset does not exist": "That preset no longer exists.",
};

/** Suggested text for posting the highlight. "simple" when made without the AI model. */
export interface PostText {
  titles: string[];
  description: string;
  hashtags: string[];
  source: "ai" | "simple";
  note?: string;
}

/** Who speaks in a job's videos (from GET /jobs/:id/speakers). */
export interface SpeakerData {
  speakers: Array<{ id: string; letter: string; source_id: string; seconds: number; sample: { start: number; end: number } | null }>;
  segments: Array<{ source_id: string; start: number; end: number; speaker: string }>;
  note?: string;
}

/** Upload progress callback and a way to stop the upload. */
export interface UploadOptions {
  onProgress?: (loaded: number, total: number) => void;
  signal?: AbortSignal;
}

/**
 * POST a form with upload progress. fetch cannot report how much of a request body was sent,
 * so this uses XMLHttpRequest. A stopped upload rejects with an AbortError.
 */
function sendForm<T>(path: string, body: FormData, options: UploadOptions = {}): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const { onProgress, signal } = options;
    if (signal?.aborted) {
      reject(new DOMException("Upload cancelled", "AbortError"));
      return;
    }
    xhr.open("POST", path);
    if (onProgress) {
      xhr.upload.onprogress = (event) => onProgress(event.loaded, event.lengthComputable ? event.total : 0);
    }
    xhr.onload = () => {
      let data: unknown = {};
      try {
        data = JSON.parse(xhr.responseText || "{}");
      } catch {
        data = {};
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data as T);
      else reject(new Error(friendlyError(xhr.status, (data as { error?: string }).error)));
    };
    // Like fetch: a network failure means the server could not be reached at all.
    xhr.onerror = () => reject(new Error(messages.errors.offline));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(body);
  });
}

/** Send one file to the API as a form upload. */
function upload<T>(path: string, file: File, options?: UploadOptions): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  return sendForm<T>(path, form, options);
}

/** Download a file from the API, with a readable error instead of a JSON page when it fails. */
async function download(path: string, fallbackName: string): Promise<string> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch {
    throw new Error(messages.errors.offline);
  }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(friendlyError(response.status, (data as { error?: string }).error));
  }
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition);
  const name = match ? decodeURIComponent(match[1]) : fallbackName;
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Give the browser time to start saving before the address is released.
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
  return name;
}

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

/** Ranges in the form the time fields use ("01:20-02:05, 05:00-05:40"). */
export function formatRanges(ranges: unknown): string {
  if (!Array.isArray(ranges)) return "";
  return ranges
    .filter((pair): pair is [number, number] => Array.isArray(pair) && pair.length === 2)
    .map(([a, b]) => `${formatTime(a)}-${formatTime(b)}`)
    .join(", ");
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

/** Leave-out kinds detected from what is said (app/analysis/avoid.py SPEECH_ITEMS). */
export const SPEECH_AVOID = ["filler", "sponsor", "swearing", "talking_heads"];

export function configPayload(form: HighlightForm) {
  // A length field can be empty while it is being edited; never send that to the server.
  const max = finiteOr(form.max_duration, finiteOr(form.min_duration, 30));
  const prompt = form.mode === "prompt";
  return {
    mode: form.mode,
    prompt: prompt ? form.prompt.trim() : "",
    requirements: prompt ? form.requirements : null,
    use_visuals: form.use_visuals,
    analysis_depth: form.analysis_depth,
    render_after_analysis: form.render_after_analysis,
    min_duration: finiteOr(form.min_duration, Math.min(20, max)),
    max_duration: max,
    length_mode: form.social_short ? "range" : form.length_mode,
    auto_length_strict: !form.social_short && form.length_mode === "auto" && Boolean(form.auto_length_strict),
    use_speech: form.use_speech,
    speech_language: form.use_speech ? form.speech_language || "" : "",
    style: form.style,
    // The focus text box only shows for the custom style; a hidden focus must not steer the AI.
    // In prompt mode the focus comes from the description.
    custom_focus: form.style === "custom" || prompt ? form.custom_focus : "",
    preserve_context: form.preserve_context,
    remove_repetition: form.remove_repetition,
    allow_repeated_content: form.allow_repeated_content,
    max_sequence_duration: form.max_sequence_duration,
    min_sequence_duration: form.min_sequence_duration,
    sequence_duration_mode: form.sequence_duration_mode,
    content_priorities: form.content_priorities,
    // Sliders go from 1 (less) to 5 (more) and show 3 until moved; the server takes a share
    // from 0.2 to 1.
    priority_weights: Object.fromEntries(
      form.content_priorities.map((key) => [key, Math.min(5, Math.max(1, form.priority_weights[key] ?? 3)) / 5])
    ),
    // Speech kinds need a transcript; the server drops them too when speech is off.
    avoid: form.use_speech ? form.avoid : form.avoid.filter((key) => !SPEECH_AVOID.includes(key)),
    avoid_custom: form.avoid_custom,
    avoid_mode: form.avoid_mode,
    max_repeated_duration: form.max_repeated_duration,
    repetition_mode: form.repetition_mode,
    order_mode: form.order_mode,
    social_short: form.social_short,
    aspect_ratio: form.aspect_ratio,
    subtitles: form.subtitles,
    caption_style: form.caption_style,
    tighten: form.use_speech ? form.tighten : "off",
    reframe: form.reframe,
    shorts_count: form.shorts_count >= 2 ? form.shorts_count : 0,
    level_audio: form.level_audio,
    subtitle_language: form.use_speech ? form.subtitle_language : "",
    punch_in: form.punch_in,
    progress_bar: form.progress_bar,
    logo_position: form.logo_position,
    logo_size: form.logo_size,
    transition: form.transition,
    audio_focus: form.audio_focus,
    frame_rate: form.frame_rate,
    // The smoothing option only shows when a frame rate is chosen.
    smooth_motion: form.frame_rate !== "source" && form.smooth_motion,
    codec: form.codec,
    quality: form.quality,
    resolution: form.resolution,
    must_include: parseRanges(form.must_include),
    must_exclude: parseRanges(form.must_exclude),
    crop: form.cropEnabled ? form.crop : null,
    video_brief: form.video_brief,
    brief_engine: form.brief_engine,
    narration: form.video_brief && form.narration,
    music_led: Boolean(form.music_led),
    music_led_sound: form.music_led_sound || "lower",
    // The music carries a music-led video, so it starts loud; the slider on Review changes it.
    ...(form.music_led ? { music_volume: 90 } : {}),
  };
}

export const api = {
  jobs: () => http<{ jobs: Job[] }>("/api/jobs"),
  job: (id: string) => http<Job>(`/api/jobs/${id}`),
  createJob: async (files: File | File[], form: HighlightForm, options?: UploadOptions) => {
    const body = new FormData();
    const list = Array.isArray(files) ? files : [files];
    for (const file of list) body.append("file", file);
    body.append("config", JSON.stringify(configPayload(form)));
    return sendForm<Job>("/api/jobs", body, options);
  },
  analyze: (id: string, form: HighlightForm) =>
    http<Job>(`/api/jobs/${id}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configPayload(form)),
    }),
  /** One highlight per video of a job that has not started (the job is replaced). */
  split: (id: string, form: HighlightForm) =>
    http<{ jobs: Job[] }>(`/api/jobs/${id}/split`, {
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
  pause: (id: string) => http<Job>(`/api/jobs/${id}/pause`, { method: "POST" }),
  interpret: (prompt: string, jobId: string | null, useSpeech: boolean) =>
    http<InterpretResult>("/api/prompt/interpret", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, job_id: jobId, use_speech: useSpeech }),
    }),
  refine: (id: string, request: string) =>
    http<Job>(`/api/jobs/${id}/refine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request }),
    }),
  aiPlan: (id: string) => http<AiPlanInfo>(`/api/jobs/${id}/ai-plan`),
  resume: (id: string) => http<Job>(`/api/jobs/${id}/resume`, { method: "POST" }),
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
  logs: (id: string) => http<{ logs: LogEntry[] }>(`/api/jobs/${id}/logs`),
  insights: (id: string) => http<JobInsights>(`/api/jobs/${id}/insights`),
  analysis: (id: string) => http<Record<string, unknown>>(`/api/jobs/${id}/analysis`),
  timeline: (id: string) => http<Record<string, unknown>>(`/api/jobs/${id}/timeline`),
  /** Save a full-size frame of the finished highlight as a JPEG. Returns the saved file name. */
  downloadCover: (id: string, seconds: number, short?: number) =>
    download(`${coverUrl(id, seconds, undefined, short)}&download=1`, "highlight-cover.jpg"),
  postText: (id: string, short?: number) =>
    http<{ suggestion: PostText | null }>(`/api/jobs/${encodeURIComponent(id)}/post-text${short ? `?short=${short}` : ""}`),
  makePostText: (id: string, short?: number) =>
    http<{ suggestion: PostText }>(`/api/jobs/${encodeURIComponent(id)}/post-text${short ? `?short=${short}` : ""}`, {
      method: "POST",
    }),
  music: (id: string) => http<{ name: string | null }>(`/api/jobs/${encodeURIComponent(id)}/music`),
  uploadMusic: (id: string, file: File, options?: UploadOptions) =>
    upload<{ name: string; replanned?: boolean; notes?: string }>(`/api/jobs/${encodeURIComponent(id)}/music`, file, options),
  deleteMusic: (id: string) =>
    http<{ name: null; replanned?: boolean; notes?: string }>(`/api/jobs/${encodeURIComponent(id)}/music`, { method: "DELETE" }),
  renderOptions: (
    id: string,
    options: { music_volume?: number; music_duck?: boolean; narration?: boolean; music_led_sound?: "lower" | "mute" },
  ) =>
    http(`/api/jobs/${encodeURIComponent(id)}/render-options`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    }),
  /** The parts of each clip kept after pauses are cut (for the Review preview). */
  tightenSpans: (id: string, clips: Array<{ start: number; end: number; source_id?: string }>) =>
    http<{ mode: string; spans: Array<Array<[number, number]>> }>(`/api/jobs/${encodeURIComponent(id)}/tighten-spans`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clips }),
    }),
  uploadLogo: (file: File, options?: UploadOptions) => upload<{ logo: boolean }>("/api/brand/logo", file, options),
  deleteLogo: () => http("/api/brand/logo", { method: "DELETE" }),
  watchFolder: () => http<{ enabled: boolean; folder: string; preset: string }>("/api/watch-folder"),
  saveWatchFolder: (config: { enabled: boolean; folder: string; preset: string }) =>
    http<{ enabled: boolean; folder: string; preset: string }>("/api/watch-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  speakers: (id: string) => http<SpeakerData>(`/api/jobs/${encodeURIComponent(id)}/speakers`),
  /** Pick new clips with changed choices (for example one speaker only). */
  regenerateWith: (id: string, changes: Record<string, unknown>) =>
    http<{ job: Job; plan: unknown }>(`/api/jobs/${id}/regenerate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(changes),
    }),
  presets: () => http<{ presets: Preset[] }>("/api/presets"),
  savePreset: (name: string, form: Partial<HighlightForm>) =>
    http<{ presets: Preset[] }>("/api/presets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, form }),
    }),
  deletePreset: (name: string) =>
    http<{ presets: Preset[] }>(`/api/presets?name=${encodeURIComponent(name)}`, { method: "DELETE" }),
  /** Save the kept clips as timelines for video editors (zip). Returns the saved file name. */
  exportForEditors: (id: string) =>
    download(`/api/jobs/${encodeURIComponent(id)}/export/editor`, "highlight-for-editors.zip"),
  /** Strong moments the current clips leave out, best first. */
  moments: (id: string, limit?: number) =>
    http<{ moments: Moment[]; available: boolean }>(
      `/api/jobs/${id}/moments${limit ? `?limit=${encodeURIComponent(limit)}` : ""}`,
    ),
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
  setOptimizationGoal: (goal: OptimizationGoal) =>
    http<OptimizationPlan>("/api/system/ai/optimization", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal }),
    }),
  pinSetting: (key: string, pinned: boolean) =>
    http<OptimizationPlan>("/api/system/ai/optimization", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pin: { key, pinned } }),
    }),
  undoOptimization: () => http<OptimizationPlan>("/api/system/ai/optimization/undo", { method: "POST" }),
  startMeasure: () => http<MeasureStatus>("/api/system/ai/measure", { method: "POST" }),
  startAiTest: () => http("/api/system/ai/test", { method: "POST" }),
  downloadImageModel: () => http("/api/system/image-model/download", { method: "POST" }),
  downloadSoundModel: () => http("/api/system/sound-model/download", { method: "POST" }),
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
  // Narrated videos
  briefEngines: (refresh = false) =>
    http<{ engines: EngineStatus[]; default: string }>(`/api/system/brief-engines${refresh ? "?refresh=1" : ""}`),
  testClaude: () =>
    http<{ ok: boolean; seconds: number; model: string | null; error: string | null }>("/api/system/brief-engines/claude/test", {
      method: "POST",
    }),
  videoTools: () => http<VideoToolsStatus>("/api/system/video-tools"),
  downloadVoiceModel: () => http("/api/system/voice-model/download", { method: "POST" }),
  prepareVideoTools: () => http("/api/system/video-tools/prepare", { method: "POST" }),
  jobBrief: (id: string) => http<BriefInfo>(`/api/jobs/${encodeURIComponent(id)}/brief`),
  saveBrief: (
    id: string,
    changes: { title?: string; hook?: string; call_to_action?: string; beats?: Array<{ id: string; narration?: string; on_screen_text?: string }> }
  ) =>
    http<{ brief: VideoBrief; warnings: string[] }>(`/api/jobs/${encodeURIComponent(id)}/brief`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(changes),
    }),
  switchBriefEngine: (id: string, engine: "local" | "claude", resume = false) =>
    http<Job>(`/api/jobs/${encodeURIComponent(id)}/brief-engine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ engine, resume }),
    }),
  speakNarration: (id: string) => http<NarrationState>(`/api/jobs/${encodeURIComponent(id)}/narration`, { method: "POST" }),
  previewSpec: (id: string) =>
    http<{ spec: Record<string, unknown> | null; narration: NarrationState }>(`/api/jobs/${encodeURIComponent(id)}/preview-spec`),
  clearTmp: () => http("/api/settings/storage/clear-tmp", { method: "POST" }),
  clearCache: () => http("/api/settings/storage/clear-cache", { method: "POST" }),
  deleteCompleted: () => http("/api/settings/storage/delete-completed", { method: "POST" }),
};
