export type JobStatus =
  | "uploaded"
  | "queued"
  | "analyzing"
  | "planning"
  | "rendering"
  | "completed"
  | "failed"
  | "cancelled"
  | "paused";


export interface VideoMeta {
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;
  bitrate?: number;
  video_codec?: string;
  audio_codec?: string;
  filename?: string;
  size_bytes?: number;
  has_audio?: boolean;
}

export interface Clip {
  id?: number | string;
  start: number;
  end: number;
  score?: number;
  reason?: string;
  duration_reason?: string;
  transcript?: string;
  kept?: boolean;
  source_id?: string;
  source_name?: string;
  /** Which short the clip belongs to when the job makes several shorts (0-based). */
  short_index?: number;
  /** The part the AI editor gave the clip in prompt mode (hook, key, ending, ...). */
  role?: string;
}

/** What the user asked for in prompt mode, as understood from the description. */
export interface PromptRequirements {
  summary: string;
  keep: string[];
  remove: string[];
  moments: string[];
  visual_cues: string[];
  audio_cues: string[];
}

export type RequirementKey = "keep" | "remove" | "moments" | "visual_cues" | "audio_cues";

/** POST /api/prompt/interpret: settings and requirements read from a description. */
export interface InterpretResult {
  summary: string;
  settings: Record<string, unknown>;
  changes: Array<{ field: string; value: unknown; source: "ai" | "rules" | "default" }>;
  requirements: PromptRequirements;
  source: "ai" | "rules";
  warning?: string | null;
}

/** GET /api/jobs/:id/ai-plan: what the AI understood and chose. */
export interface AiPlanInfo {
  mode: "manual" | "prompt";
  prompt: string;
  requirements: PromptRequirements | null;
  refine_notes: string[];
  plan: { understanding: string; left_out: string; order: string; chosen: number; considered: number } | null;
}

/** A strong moment the current clips leave out (from GET /jobs/:id/moments). */
export interface Moment {
  start: number;
  end: number;
  score?: number;
  reason?: string;
  duration_reason?: string;
  transcript?: string;
  source_id?: string;
  source_name?: string;
  candidate_id?: string | null;
}

export interface JobSource {
  id: string;
  index?: number;
  filename: string;
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;
}

export interface Stage {
  name: string;
  status: string;
  detail: string;
  /** Finished time spent in this stage, summed over its runs. */
  seconds?: number;
  /** Set while the stage runs: live time is seconds + (now - started_at). */
  started_at?: string | null;
  /** How far the current run is: done of total in unit ("time" = seconds of video, "" = percent). */
  done?: number;
  total?: number;
  unit?: string;
}

export interface JobTiming {
  /** Total of finished attempts (analysis and rendering), in seconds. */
  total_seconds: number;
  analysis_seconds: number;
  render_seconds: number;
  /** Length of the last finished attempt. */
  attempt_seconds: number;
  /** Start of the attempt in progress, or null when nothing runs. */
  attempt_started_at: string | null;
  attempt_phase: string;
  attempt_outcome: string;
  /** Server clock when this was sent, to count live time without trusting the local clock. */
  server_time: string;
}

export interface Job {
  id: string;
  status: JobStatus;
  original_filename: string;
  analysis_status: string;
  render_status: string;
  error?: string | null;
  style: string;
  min_duration: number;
  max_duration: number;
  progress: number;
  /** Seconds left for the running step list, or null while it is not known yet. */
  eta_seconds?: number | null;
  /** Source video being analyzed (0-based) and how many there are. */
  source_index?: number;
  source_count?: number;
  last_stage: string;
  processing_seconds: number;
  timing?: JobTiming;
  metadata: VideoMeta;
  config: Record<string, unknown>;
  stages: Stage[];
  clips: Clip[];
  sources?: JobSource[];
  /** Why a narrated video's brief paused the job (Claude could not write it). */
  brief_failure?: { message: string; reason: string; stage?: "brief" | "director" } | null;
  outputs: Array<{
    id: number;
    kind?: string;
    duration: number;
    size_bytes: number;
    codec: string;
    width: number;
    height: number;
    version_label: string;
  }>;
  created_at?: string;
}

export interface CropBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface HighlightForm {
  /** "prompt": describe what you want and let the AI set things up; "manual": set it yourself. */
  mode: "manual" | "prompt";
  prompt: string;
  requirements: PromptRequirements | null;
  /** Look at the picture (vision model, picture matching, descriptions). */
  use_visuals: boolean;
  analysis_depth: "balanced" | "best";
  /** Create the video as soon as the analysis is done, without stopping at Review. */
  render_after_analysis: boolean;
  min_duration: number;
  max_duration: number;
  length_mode: "range" | "auto";
  /** Automatic length, keeping only moments close to the very best ("as short as possible"). */
  auto_length_strict: boolean;
  use_speech: boolean;
  /** The language spoken in the video as a Whisper code ("ml", "en"); "" = detect it. */
  speech_language: string;
  style: string;
  cropEnabled: boolean;
  crop: CropBox;
  custom_focus: string;
  preserve_context: string;
  remove_repetition: boolean;
  allow_repeated_content: boolean;
  max_sequence_duration: number;
  min_sequence_duration: number | null;
  sequence_duration_mode: "auto" | "custom";
  content_priorities: string[];
  priority_weights: Record<string, number>;
  avoid: string[];
  avoid_custom: string;
  avoid_mode: "avoid" | "never";
  max_repeated_duration: number;
  repetition_mode: string;
  order_mode: string;
  social_short: boolean;
  aspect_ratio: string;
  subtitles: string;
  caption_style: "plain" | "clean" | "bold" | "boxed";
  tighten: "off" | "pauses" | "fillers";
  reframe: "center" | "subject";
  /** 0: one highlight; 2-10: that many separate shorts. */
  shorts_count: number;
  level_audio: boolean;
  subtitle_language: string;
  punch_in: boolean;
  progress_bar: boolean;
  logo_position: "off" | "top-left" | "top-right" | "bottom-left" | "bottom-right";
  logo_size: number;
  transition: string;
  audio_focus: "balanced" | "natural" | "dialogue";
  frame_rate: string;
  smooth_motion: boolean;
  codec: string;
  quality: string;
  resolution: string;
  must_include: string;
  must_exclude: string;
  /** A narrated video: a brief (title, hook, beats) is written and drawn on the video. */
  video_brief: boolean;
  /** Who writes the brief; chosen by the user, never switched automatically. */
  brief_engine: "local" | "claude";
  /** Speak the brief's narration over the video (needs video_brief). */
  narration: boolean;
  /** Cut the video to the beat of the user's music (uploaded to the job before analysis). */
  music_led: boolean;
  /** The video's own sound under the music: lowered or muted. */
  music_led_sound: "lower" | "mute";
  /** Name of the music uploaded for music-led editing (page state only, not sent). */
  music_file_name: string;
}

export interface LogEntry {
  level: string;
  message: string;
  ts?: number;
}

/** GET /api/jobs/:id/insights: where the processing time went (app/services/insights.py). */
export interface JobInsights {
  job: { id: string; filename: string; status: string; created_at: string; mode: string };
  video: { duration: number; width: number; height: number; codec: string; fps: number; size_bytes: number; sources: number };
  result: { clips: number; outputs: Array<{ duration: number | null; size_bytes: number | null; kind: string }> };
  totals: { total_seconds: number; analysis_seconds: number; render_seconds: number; ratio: number | null };
  steps: Array<{
    name: string;
    label: string;
    phase: "analysis" | "render";
    seconds: number;
    share: number;
    status: string;
    detail: string;
    done: number;
    total: number;
    unit: string;
  }>;
  slowest: string[];
  render: {
    seconds: number;
    clips: number;
    videos: number;
    output_seconds: number;
    encoder?: { codec?: string; preset?: string; crf?: number };
    parts: Array<{ name: string; label: string; seconds: number }>;
  } | null;
  ai: Array<{
    step: string;
    calls: number;
    prompt_tokens: number;
    prompt_seconds: number;
    reply_tokens: number;
    reply_seconds: number;
    tokens_per_second: number | null;
    seconds_per_call: number;
    load_seconds: number;
    near_context_limit: number;
  }>;
  run: {
    computer: { cpu_threads: number; memory_gb: number | null };
    models: { speech: string; speech_device: string; vision: string; text: string; editor: string };
    options: { mode: string; vision_scope: string; editor_review: boolean; analysis_copy: boolean; music_led?: boolean };
    found: {
      videos: number;
      scenes: number | null;
      scenes_ai: number | null;
      segments: number | null;
      language: string | null;
      language_probability: number | null;
      speech_reliable: boolean;
      candidates: number;
      music_bpm?: number | null;
    };
  } | null;
  bottlenecks: Array<{ name: string; label: string; seconds: number; share: number; why: string; tips: string[] }>;
  warnings: Array<{ level: string; message: string; count: number; ts?: number }>;
  has_log: boolean;
}

/** GET /api/system/brief-engines: whether each brief engine is ready. */
export interface EngineStatus {
  name: "local" | "claude";
  label: string;
  ready: boolean;
  problems: string[];
  notes: string[];
  details: Record<string, unknown>;
}

export interface BriefClip {
  id: string;
  source_id: string;
  start: number;
  end: number;
}

export interface BriefBeat {
  id: string;
  purpose: string;
  narration: string;
  on_screen_text: string;
  /** About how many narration words fit the beat's moments. */
  max_words: number;
  clips: BriefClip[];
}

/** A narrated video's brief (brief.json). */
export interface VideoBrief {
  engine: string;
  model: string;
  title: string;
  hook: string;
  tone: string;
  music_mood: string;
  call_to_action: string;
  beats: BriefBeat[];
  /** What the checks fixed. */
  checks: string[];
  edited: boolean;
}

export interface NarrationState {
  state: "idle" | "running" | "done" | "error";
  done: number;
  total: number;
  error?: string | null;
}

/** GET /api/jobs/:id/brief */
export interface BriefInfo {
  brief: VideoBrief | null;
  failure: { message: string; reason: string } | null;
  narration: NarrationState;
  /** Beat ids whose narration is already spoken with the current voice. */
  spoken: string[];
}

/** GET /api/system/video-tools */
export interface VideoToolsStatus {
  voice: TransferFields & {
    state: string;
    files_done: number;
    files_total: number;
    error?: string | null;
    installed: boolean;
    size_mb: number;
    runtime: boolean;
    voices: Record<string, string>;
  };
  remotion: {
    node: string | null;
    node_ok: boolean;
    installed: boolean;
    bundle: boolean;
    browser: boolean;
    ready: boolean;
    prepare: { state: string; step: string; error?: string | null; log: string[] };
  };
}

export interface ModelSpec {
  name: string;
  kind: string;
  disk_gb: number;
  min_vram_gb: number;
  min_ram_gb: number;
  summary: string;
  speed: string;
}

export interface Placement {
  measured?: boolean;
  mode?: string | null;
  label?: string;
  model?: string | null;
  gpu_fraction?: number | null;
  size?: number | null;
  size_vram?: number | null;
}

export interface RuntimeStatus {
  mode?: string;
  label?: string;
  measured?: boolean;
  hint?: string;
  placement?: Placement;
}

/** Byte progress shared by every model download. */
export interface TransferFields {
  bytes_done?: number;
  /** 0 when the size is not known. */
  bytes_total?: number;
  /** Bytes per second; 0 until steady. */
  speed?: number;
  /** Seconds left, or null while not known. */
  eta?: number | null;
  phase?: string;
}

export interface PullStatus extends TransferFields {
  state?: string;
  status?: string;
  completed?: number;
  total?: number;
  error?: string | null;
}

/** A job update sent over the socket while it runs (job_progress). */
export interface ProgressEvent {
  job_id: string;
  stage: string;
  status: string;
  detail?: string;
  step_done?: number;
  step_total?: number;
  step_unit?: string;
  overall: number;
  eta?: number | null;
  source_index?: number;
  source_count?: number;
}

export interface ProbeResult {
  ok?: boolean;
  skipped?: boolean;
  model?: string;
  error?: string | null;
  warning?: string | null;
  message?: string;
  text?: string;
  elapsed_sec?: number;
  load_sec?: number | null;
  tokens_per_sec?: number | null;
  eval_count?: number;
  device?: string;
  compute_type?: string | null;
  fallback?: string | null;
  audio_sec?: number;
  realtime_factor?: number;
  placement?: Placement;
}

export interface AiTestResult {
  ok?: boolean;
  error?: string | null;
  warning?: string | null;
  text?: ProbeResult | null;
  vision?: ProbeResult | null;
  whisper?: ProbeResult | null;
  runtime?: RuntimeStatus;
}

export interface OllamaStatus {
  connected: boolean;
  version?: string | null;
  vision_model: string;
  text_model: string;
  vision_present: boolean;
  text_present: boolean;
  /** False when the vision model cannot read images; null when Ollama does not say. */
  vision_capable?: boolean | null;
  /** Installed, able to read images, and supported by this Ollama version. */
  vision_ready?: boolean;
  warnings?: string[];
  models: string[];
  model_details?: Array<{ name: string; size?: number }>;
  error?: string | null;
  base_url?: string;
  pull?: Record<string, PullStatus>;
  runtime?: RuntimeStatus;
  test?: { state?: string; result?: AiTestResult | null };
  whisper_download?: TransferFields & { state?: string; model?: string | null; error?: string | null };
  image_model?: ImageModelStatus;
  sound_model?: ImageModelStatus;
  measure?: MeasureStatus;
}

export interface ImageModelStatus extends TransferFields {
  state: string;
  files_done: number;
  files_total: number;
  error?: string | null;
  installed: boolean;
  size_mb: number;
  runtime: boolean;
}

export interface OptimizationItem {
  key: string;
  label: string;
  current: string | number | null;
  recommended: string | number | null;
  changed: boolean;
  reason: string;
  /** Chosen by hand while automatic optimization is on; kept as it is. */
  pinned?: boolean;
  /** The recommended model is not downloaded yet; the current one stays in use. */
  needs_download?: boolean;
  /** False when automatic optimization leaves this setting alone (Ollama on another computer). */
  managed?: boolean;
}

export type OptimizationGoal = "quality" | "balanced" | "fast";

export interface MeasureStatus {
  state: "idle" | "running" | "done" | "error";
  step?: "text" | "vision" | "whisper" | null;
  error?: string | null;
  result?: { ts: number } | null;
}

export interface TimeEstimate {
  video_seconds: number;
  seconds: number;
  parts: Partial<Record<"speech_to_text" | "speech_understanding" | "video_understanding", number>>;
  /** False when a part that is turned on has not been measured yet. */
  complete: boolean;
}

export interface PerformanceRun {
  ts: number;
  video_seconds: number;
  total_seconds: number;
  visual_enabled: boolean;
  vision?: { calls: number; seconds_per_call: number; tokens_per_sec: number } | null;
  text?: { calls: number; seconds_per_call: number; tokens_per_sec: number } | null;
}

export interface OptimizationPlan {
  enabled: boolean;
  goal: OptimizationGoal;
  hardware: {
    text: string;
    details?: string[];
    system?: string;
    ram_available_gb?: number | null;
    disk_free_gb?: number | null;
    uses_gpu: boolean;
  };
  items: OptimizationItem[];
  changes: OptimizationItem[];
  tradeoffs: string[];
  warnings: string[];
  notes: string[];
  can_undo: boolean;
  undo_saved_at?: number | null;
  history: PerformanceRun[];
  tips: string[];
  estimate?: TimeEstimate | null;
  measurement?: MeasureStatus;
}
