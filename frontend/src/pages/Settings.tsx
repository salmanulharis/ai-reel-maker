import { useEffect, useState } from "react";
import { api, formatBytes } from "../services/api";
import { messages } from "../i18n";

const VISION = ["qwen3-vl:2b", "qwen3-vl:4b", "qwen3-vl:8b", "qwen3-vl:30b"];
const TEXT = ["qwen3:4b", "qwen3:8b", "qwen3:30b"];

export default function Settings() {
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [storage, setStorage] = useState<Record<string, number | string>>({});
  const [info, setInfo] = useState<Record<string, unknown> | null>(null);
  const [ollama, setOllama] = useState<Record<string, unknown> | null>(null);
  const [saved, setSaved] = useState("");

  const load = async () => {
    const s = await api.settings();
    setSettings(s.settings);
    setStorage(s.storage);
    setInfo(await api.info());
    setOllama(await api.ollama());
  };

  useEffect(() => {
    load();
  }, []);

  function set(key: string, value: unknown) {
    setSettings((cur) => ({ ...cur, [key]: value }));
  }

  async function save() {
    await api.saveSettings({
      ollama_base_url: settings.ollama_base_url,
      ollama_vision_model: settings.ollama_vision_model,
      ollama_text_model: settings.ollama_text_model,
      ollama_timeout: settings.ollama_timeout,
      ollama_max_retries: settings.ollama_max_retries,
      whisper_model: settings.whisper_model,
      scene_threshold: settings.scene_threshold,
      maximum_candidates: settings.maximum_candidates,
      silence_threshold_db: settings.silence_threshold_db,
      output_video_codec: settings.output_video_codec,
      output_crf: settings.output_crf,
      output_audio_bitrate: settings.output_audio_bitrate,
      output_format: settings.output_format,
      debug_analysis: settings.debug_analysis,
    });
    setSaved("Saved");
    setTimeout(() => setSaved(""), 1500);
  }

  const hw = (info?.hardware || {}) as Record<string, unknown>;
  const checks = (info?.checks || {}) as Record<string, boolean>;

  return (
    <div>
      <p className="privacy" style={{ marginTop: 0 }}>
        {messages.privacy}
      </p>
      <div className="panel">
        <h3>{messages.settings.hardware}</h3>
        <div className="meta-grid">
          <div>
            <span>CPU</span>
            {String(hw.cpu || "—")}
          </div>
          <div>
            <span>RAM</span>
            {hw.ram_gb ? `${hw.ram_gb} GB` : "—"}
          </div>
          <div>
            <span>GPU</span>
            {String(hw.gpu || "Not detected")}
          </div>
          <div>
            <span>VRAM</span>
            {hw.vram_gb ? `${hw.vram_gb} GB` : "—"}
          </div>
          <div>
            <span>CUDA</span>
            {hw.cuda ? "Available" : "No"}
          </div>
          <div>
            <span>Acceleration</span>
            {String(hw.acceleration || "CPU fallback")}
          </div>
        </div>
        <p style={{ color: "var(--muted)" }}>
          Recommended Ollama model: {String(hw.recommended_vision_model || "qwen3-vl:8b")}
        </p>
        <p style={{ color: "var(--warn)", fontSize: 12 }}>
          Larger vision models need substantially more VRAM. qwen3-vl:8b is about 6.1 GB; 30B-class
          models can require 20 GB+ VRAM or heavy offload.
        </p>
        <div className="checks">
          {Object.entries(checks).map(([k, ok]) => (
            <div key={k} className={`check ${ok ? "ok" : "bad"}`}>
              {ok ? "✓" : "✗"} {k.replace("_", " ")}
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <h3>{messages.settings.ai}</h3>
        <p className="recommend">
          Best models for this app: vision <strong>qwen3-vl:8b</strong> (~6.1 GB) and text{" "}
          <strong>qwen3:8b</strong>. Use <strong>qwen3-vl:4b</strong> if VRAM is under ~8 GB.
        </p>
        <div className="field">
          <label>Ollama URL</label>
          <input value={String(settings.ollama_base_url || "")} onChange={(e) => set("ollama_base_url", e.target.value)} />
        </div>
        <div className="grid-2">
          <div className="field">
            <label>Vision model</label>
            <input
              list="vision-models"
              value={String(settings.ollama_vision_model || "")}
              onChange={(e) => set("ollama_vision_model", e.target.value)}
            />
            <datalist id="vision-models">
              {VISION.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
          <div className="field">
            <label>Text model</label>
            <input
              list="text-models"
              value={String(settings.ollama_text_model || "")}
              onChange={(e) => set("ollama_text_model", e.target.value)}
            />
            <datalist id="text-models">
              {TEXT.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
        </div>
        {ollama && !ollama.vision_present && (
          <p>
            {String(settings.ollama_vision_model)} not installed{" "}
            <button className="btn" onClick={() => api.pullModel(String(settings.ollama_vision_model))}>
              Install model
            </button>
          </p>
        )}
        {ollama && !ollama.text_present && (
          <p>
            {String(settings.ollama_text_model)} not installed{" "}
            <button className="btn" onClick={() => api.pullModel(String(settings.ollama_text_model))}>
              Install model
            </button>
          </p>
        )}
      </div>

      <div className="panel">
        <h3>{messages.settings.video}</h3>
        <div className="grid-2">
          <div className="field">
            <label>Whisper model</label>
            <select value={String(settings.whisper_model || "small")} onChange={(e) => set("whisper_model", e.target.value)}>
              <option>tiny</option>
              <option>base</option>
              <option>small</option>
              <option>medium</option>
              <option>large-v3</option>
            </select>
          </div>
          <div className="field">
            <label>Scene threshold</label>
            <input
              type="number"
              value={Number(settings.scene_threshold || 27)}
              onChange={(e) => set("scene_threshold", Number(e.target.value))}
            />
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>{messages.settings.output}</h3>
        <div className="grid-2">
          <div className="field">
            <label>Codec</label>
            <select
              value={String(settings.output_video_codec || "auto")}
              onChange={(e) => set("output_video_codec", e.target.value)}
            >
              <option value="auto">Auto / source-compatible H.264</option>
              <option value="h264">H.264</option>
              <option value="h265">H.265</option>
              <option value="vp9">VP9</option>
              <option value="av1">AV1</option>
            </select>
          </div>
          <div className="field">
            <label>CRF</label>
            <input type="number" value={Number(settings.output_crf || 18)} onChange={(e) => set("output_crf", Number(e.target.value))} />
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>{messages.settings.storage}</h3>
        <div className="meta-grid">
          <div>
            <span>Jobs</span>
            {formatBytes(Number(storage.jobs_bytes || 0))}
          </div>
          <div>
            <span>Temporary files</span>
            {formatBytes(Number(storage.tmp_bytes || 0))}
          </div>
          <div>
            <span>Cached analysis</span>
            {formatBytes(Number(storage.cache_bytes || 0))}
          </div>
          <div>
            <span>Completed videos</span>
            {formatBytes(Number(storage.outputs_bytes || 0))}
          </div>
        </div>
        <div className="btn-row">
          <button className="btn" onClick={() => api.clearTmp().then(load)}>
            Clear temporary files
          </button>
          <button className="btn" onClick={() => api.clearCache().then(load)}>
            Clear cache
          </button>
          <button className="btn danger" onClick={() => api.deleteCompleted().then(load)}>
            Delete all completed jobs
          </button>
        </div>
      </div>

      <div className="btn-row">
        <button className="btn primary" onClick={save}>
          Save settings
        </button>
        {saved && <span>{saved}</span>}
      </div>
    </div>
  );
}
