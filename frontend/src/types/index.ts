export type JobStatus =
  | "uploaded"
  | "queued"
  | "analyzing"
  | "planning"
  | "rendering"
  | "completed"
  | "failed"
  | "cancelled";

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
}

export interface JobSource {
  id: string;
  index?: number;
  filename: string;
  duration?: number;
  width?: number;
  height?: number;
}

export interface Stage {
  name: string;
  status: string;
  detail: string;
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
  last_stage: string;
  processing_seconds: number;
  metadata: VideoMeta;
  config: Record<string, unknown>;
  stages: Stage[];
  clips: Clip[];
  sources?: JobSource[];
  outputs: Array<{
    id: number;
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
  min_duration: number;
  max_duration: number;
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
  max_repeated_duration: number;
  repetition_mode: string;
  order_mode: string;
  social_short: boolean;
  aspect_ratio: string;
  subtitles: string;
  transition: string;
  codec: string;
  quality: string;
  resolution: string;
  must_include: string;
  must_exclude: string;
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

export interface PullStatus {
  state?: string;
  status?: string;
  completed?: number;
  total?: number;
  error?: string | null;
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
  models: string[];
  model_details?: Array<{ name: string; size?: number }>;
  error?: string | null;
  base_url?: string;
  pull?: Record<string, PullStatus>;
  runtime?: RuntimeStatus;
  test?: { state?: string; result?: AiTestResult | null };
  whisper_download?: { state?: string; model?: string | null; error?: string | null };
}
