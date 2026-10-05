import { useEffect, useState } from "react";
import { api, formatBytes } from "../services/api";
import { messages } from "../i18n";
import type { AiTestResult, ModelSpec, OllamaStatus, PullStatus, RuntimeStatus } from "../types";

type Info = {
  hardware?: Record<string, unknown>;
  recommendation?: Record<string, unknown>;
  catalog?: { vision?: ModelSpec[]; text?: ModelSpec[]; whisper?: ModelSpec[] };
  runtime?: RuntimeStatus;
  checks?: Record<string, boolean>;
  whisper_cache?: Record<string, boolean>;
  profile?: { mode?: string; auto_summary?: string };
  ollama?: { model_details?: Array<{ name: string; size?: number }> };
};

const copy = messages.settings;

export default function Settings() {
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [loaded, setLoaded] = useState<Record<string, unknown>>({});
  const [storage, setStorage] = useState<Record<string, number | string>>({});
  const [info, setInfo] = useState<Info | null>(null);
  const [ollama, setOllama] = useState<OllamaStatus | null>(null);
  const [saved, setSaved] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");

  const load = async (refresh = false) => {
    const [systemSettings, systemInfo, status] = await Promise.all([
      api.settings(),
      api.info({ refresh, encoders: refresh }),
      api.ollama(),
    ]);
    setSettings(systemSettings.settings);
    setLoaded(systemSettings.settings);
    setStorage(systemSettings.storage);
    setInfo(systemInfo as Info);
    setOllama(status);
  };

  useEffect(() => {
    load().catch((err) => setError((err as Error).message));
  }, []);

  const pullBusy = Object.values(ollama?.pull || {}).some((item) => item?.state === "running");
  const testBusy = ollama?.test?.state === "running" || busy === "test";
  const whisperBusy = ollama?.whisper_download?.state === "running";
  const whisperState = ollama?.whisper_download?.state;

  useEffect(() => {
    if (!pullBusy && !testBusy && !whisperBusy) return;
    const id = setInterval(() => {
      api.ollama().then(setOllama).catch(() => undefined);
    }, 1000);
    return () => clearInterval(id);
  }, [pullBusy, testBusy, whisperBusy]);

  useEffect(() => {
    if (whisperState !== "done") return;
    api.info().then((systemInfo) => setInfo(systemInfo as Info)).catch(() => undefined);
  }, [whisperState]);

  function set(key: string, value: unknown) {
    setSettings((cur) => ({ ...cur, [key]: value }));
  }

  async function saveAdvanced() {
    const inferenceKeys = [
      "ollama_vision_model",
      "ollama_text_model",
      "whisper_model",
      "whisper_device",
      "ollama_vision_batch_size",
      "ollama_vision_num_ctx",
      "ollama_text_num_ctx",
      "ollama_num_thread",
      "whisper_cpu_threads",
    ];
    const payload: Record<string, unknown> = {
      ollama_base_url: settings.ollama_base_url,
      ollama_timeout: settings.ollama_timeout,
      ollama_max_retries: settings.ollama_max_retries,
      scene_threshold: settings.scene_threshold,
      output_video_codec: settings.output_video_codec,
      output_crf: settings.output_crf,
      enable_visual_analysis: settings.enable_visual_analysis,
    };
    for (const key of inferenceKeys) {
      if (String(settings[key] ?? "") !== String(loaded[key] ?? "")) payload[key] = settings[key];
    }
    setError("");
    try {
      await api.saveSettings(payload);
      setSaved(copy.saved);
      setTimeout(() => setSaved(""), 1500);
      await load();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function useModel(key: string, name: string) {
    setError("");
    setBusy(name);
    try {
      await api.saveSettings({ [key]: name });
      await load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function install(names: string[]) {
    setError("");
    try {
      await api.pullModels(names);
      const status = await api.ollama();
      setOllama(status);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function downloadWhisper(name: string) {
    setError("");
    try {
      await api.downloadWhisper(name);
      setOllama(await api.ollama());
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function applyRecommended() {
    setError("");
    setBusy("apply");
    try {
      await api.applyRecommended();
      await load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function setCustom() {
    setError("");
    try {
      await api.saveSettings({ ai_config_mode: "custom" });
      await load();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function runTest() {
    setError("");
    setBusy("test");
    try {
      await api.startAiTest();
      setOllama(await api.ollama());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  const hw = info?.hardware || {};
  const recommendation = info?.recommendation || {};
  const test = ollama?.test?.result;
  const runtime = test?.runtime || info?.runtime || ollama?.runtime;
  const checks = info?.checks || {};
  const mode = String(info?.profile?.mode || settings.ai_config_mode || "auto");
  const visionName = String(recommendation.vision_model || "");
  const textName = String(recommendation.text_model || "");
  const missingRecommended = [visionName, textName].filter((name) => name && !modelPresent(name, ollama));

  return (
    <div>
      <p className="privacy" style={{ marginTop: 0 }}>
        {messages.privacy}
      </p>

      <div className="panel">
        <h3>{copy.hardware}</h3>
        <div className="status-strip">
          <span className={`mode-badge ${runtime?.mode === "gpu" ? "gpu" : runtime?.mode === "split" ? "split" : "cpu"}`}>
            {runtime?.label || copy.hardware}
          </span>
          <span className="muted">{mode === "auto" ? copy.automatic : copy.custom}</span>
          <span className="muted">{runtime?.measured ? copy.measured : copy.detected}</span>
        </div>
        {runtime?.hint && <p className="hint">{runtime.hint}</p>}
        <div className="meta-grid">
          <Stat label="CPU" value={String(hw.cpu || "—")} />
          <Stat label={copy.cores} value={hw.cpu_cores ? String(hw.cpu_cores) : "—"} />
          <Stat label="RAM" value={hw.ram_gb ? `${hw.ram_gb} GB` : "—"} />
          <Stat label={copy.freeRam} value={hw.ram_available_gb ? `${hw.ram_available_gb} GB` : "—"} />
          <Stat label="GPU" value={String(hw.gpu || "—")} />
          <Stat label="VRAM" value={hw.vram_gb ? `${hw.vram_gb} GB` : "—"} />
          <Stat label={copy.freeVram} value={hw.vram_free_gb ? `${hw.vram_free_gb} GB` : "—"} />
          <Stat label={copy.driver} value={String(hw.driver || "—")} />
          <Stat label={copy.freeDisk} value={hw.disk_free_gb ? `${hw.disk_free_gb} GB` : "—"} />
          <Stat label={copy.encode} value={String(hw.encode_acceleration || "—")} />
        </div>
        {hw.vram_note ? <p className="hint">{String(hw.vram_note)}</p> : null}
        <div className="checks">
          {Object.entries(checks).map(([key, ok]) => (
            <div key={key} className={`check ${ok ? "ok" : "bad"}`}>
              {ok ? "✓" : "✗"} {key.replace(/_/g, " ")}
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <h3>{copy.ai}</h3>
        {info?.profile?.auto_summary && mode === "auto" && <p className="recommend">{info.profile.auto_summary}</p>}
        {Array.isArray(recommendation.notes) &&
          recommendation.notes.map((note) => (
            <p key={String(note)} className="hint">
              {String(note)}
            </p>
          ))}
        <div className="btn-row stack-buttons">
          <button className="btn primary" disabled={busy === "apply"} onClick={applyRecommended}>
            {copy.useRecommended}
          </button>
          <button className={`btn ${mode === "custom" ? "primary" : ""}`} onClick={setCustom}>
            {copy.custom}
          </button>
          <button className="btn" disabled={testBusy} onClick={runTest}>
            {testBusy ? copy.testing : copy.test}
          </button>
          <button className="btn" disabled={!missingRecommended.length || pullBusy} onClick={() => install(missingRecommended)}>
            {copy.installRecommended}
          </button>
          <button className="btn" onClick={() => load(true).catch((err) => setError((err as Error).message))}>
            {copy.refresh}
          </button>
        </div>
        <TestResult result={test} />
        <ModelGroup
          title={copy.vision}
          models={info?.catalog?.vision || []}
          active={String(settings.ollama_vision_model || "")}
          recommended={visionName}
          ollama={ollama}
          busy={busy}
          onUse={(name) => useModel("ollama_vision_model", name)}
          onInstall={(name) => install([name])}
        />
        <ModelGroup
          title={copy.text}
          models={info?.catalog?.text || []}
          active={String(settings.ollama_text_model || "")}
          recommended={textName}
          ollama={ollama}
          busy={busy}
          onUse={(name) => useModel("ollama_text_model", name)}
          onInstall={(name) => install([name])}
        />
        <h4 className="group-title">{copy.whisper}</h4>
        <div className="model-list">
          {(info?.catalog?.whisper || []).map((model) => {
            const cached = Boolean(info?.whisper_cache?.[model.name]);
            const active = String(settings.whisper_model || "") === model.name;
            const downloading = whisperBusy && ollama?.whisper_download?.model === model.name;
            return (
              <article key={model.name} className={`model-card ${active ? "active" : ""}`}>
                <header>
                  <strong>{model.name}</strong>
                  <span className="tag-row">
                    {model.name === recommendation.whisper_model && <span className="tag">{copy.recommended}</span>}
                    <span className={`tag ${cached ? "ok" : ""}`}>{cached ? copy.installed : copy.missing}</span>
                  </span>
                </header>
                <p>{model.summary}</p>
                <p className="hint">
                  {copy.disk.replace("{size}", String(model.disk_gb))} · {model.speed}
                </p>
                <div className="btn-row stack-buttons">
                  <button className="btn" disabled={active || busy === model.name} onClick={() => useModel("whisper_model", model.name)}>
                    {copy.useModel}
                  </button>
                  <button className="btn" disabled={cached || downloading} onClick={() => downloadWhisper(model.name)}>
                    {downloading ? copy.downloading : copy.download}
                  </button>
                </div>
                {downloading && ollama?.whisper_download?.error && <p className="error-text">{ollama.whisper_download.error}</p>}
              </article>
            );
          })}
        </div>

        <details className="help">
          <summary>{copy.help}</summary>
          <p>{copy.helpIntro}</p>
          <p>{copy.helpCpu}</p>
          <p>{copy.helpGpu}</p>
          <p>{copy.helpAmd}</p>
          <p>{copy.helpTrouble}</p>
        </details>
      </div>

      <details className="panel advanced">
        <summary>{copy.advanced}</summary>
        <div className="field">
          <label title={copy.ollamaUrlHint}>{copy.ollamaUrl}</label>
          <input value={String(settings.ollama_base_url || "")} onChange={(e) => set("ollama_base_url", e.target.value)} />
        </div>
        <div className="grid-2">
          <NumberField label={copy.timeout} hint={copy.timeoutHint} value={settings.ollama_timeout} onChange={(value) => set("ollama_timeout", value)} />
          <NumberField label={copy.retries} hint={copy.retriesHint} value={settings.ollama_max_retries} onChange={(value) => set("ollama_max_retries", value)} />
          <NumberField label={copy.batch} hint={copy.batchHint} value={settings.ollama_vision_batch_size} onChange={(value) => set("ollama_vision_batch_size", value)} />
          <NumberField label={copy.visionCtx} hint={copy.visionCtxHint} value={settings.ollama_vision_num_ctx} onChange={(value) => set("ollama_vision_num_ctx", value)} />
          <NumberField label={copy.textCtx} hint={copy.textCtxHint} value={settings.ollama_text_num_ctx} onChange={(value) => set("ollama_text_num_ctx", value)} />
          <NumberField label={copy.threads} hint={copy.threadsHint} value={settings.ollama_num_thread} onChange={(value) => set("ollama_num_thread", value)} />
          <div className="field">
            <label title={copy.whisperDeviceHint}>{copy.whisperDevice}</label>
            <select value={String(settings.whisper_device || "auto")} onChange={(e) => set("whisper_device", e.target.value)}>
              <option value="auto">auto</option>
              <option value="cpu">cpu</option>
              <option value="cuda">cuda</option>
            </select>
          </div>
          <NumberField label={copy.scene} value={settings.scene_threshold} onChange={(value) => set("scene_threshold", value)} />
        </div>
        <label className="check-row">
          <input
            type="checkbox"
            checked={Boolean(settings.enable_visual_analysis)}
            onChange={(e) => set("enable_visual_analysis", e.target.checked)}
          />
          <span title={copy.visualHint}>{copy.visual}</span>
        </label>
        <div className="grid-2">
          <div className="field">
            <label>{copy.codec}</label>
            <select value={String(settings.output_video_codec || "auto")} onChange={(e) => set("output_video_codec", e.target.value)}>
              <option value="auto">Auto / source-compatible H.264</option>
              <option value="h264">H.264</option>
              <option value="h265">H.265</option>
              <option value="vp9">VP9</option>
              <option value="av1">AV1</option>
            </select>
          </div>
          <NumberField label={copy.crf} value={settings.output_crf} onChange={(value) => set("output_crf", value)} />
        </div>
        <div className="btn-row stack-buttons">
          <button className="btn primary" onClick={saveAdvanced}>
            {copy.save}
          </button>
          {saved && <span>{saved}</span>}
        </div>
      </details>

      <div className="panel">
        <h3>{copy.storage}</h3>
        <div className="meta-grid">
          <Stat label="Jobs" value={formatBytes(Number(storage.jobs_bytes || 0))} />
          <Stat label="Temporary files" value={formatBytes(Number(storage.tmp_bytes || 0))} />
          <Stat label="Cached analysis" value={formatBytes(Number(storage.cache_bytes || 0))} />
          <Stat label="Completed videos" value={formatBytes(Number(storage.outputs_bytes || 0))} />
        </div>
        <div className="btn-row stack-buttons">
          <button className="btn" onClick={() => api.clearTmp().then(() => load())}>
            Clear temporary files
          </button>
          <button className="btn" onClick={() => api.clearCache().then(() => load())}>
            Clear cache
          </button>
          <button className="btn danger" onClick={() => api.deleteCompleted().then(() => load())}>
            Delete all completed jobs
          </button>
        </div>
      </div>
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      {value}
    </div>
  );
}

function NumberField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: unknown;
  onChange: (value: number) => void;
}) {
  return (
    <div className="field">
      <label title={hint}>{label}</label>
      <input type="number" value={Number(value || 0)} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

function ModelGroup({
  title,
  models,
  active,
  recommended,
  ollama,
  busy,
  onUse,
  onInstall,
}: {
  title: string;
  models: ModelSpec[];
  active: string;
  recommended: string;
  ollama: OllamaStatus | null;
  busy: string;
  onUse: (name: string) => void;
  onInstall: (name: string) => void;
}) {
  return (
    <>
      <h4 className="group-title">{title}</h4>
      <div className="model-list">
        {models.map((model) => {
          const present = modelPresent(model.name, ollama);
          const pull = ollama?.pull?.[model.name];
          const running = pull?.state === "running";
          return (
            <article key={model.name} className={`model-card ${active === model.name ? "active" : ""}`}>
              <header>
                <strong>{model.name}</strong>
                <span className="tag-row">
                  {model.name === recommended && <span className="tag">{copy.recommended}</span>}
                  <span className={`tag ${present ? "ok" : ""}`}>{present ? copy.installed : copy.missing}</span>
                </span>
              </header>
              <p>{model.summary}</p>
              <p className="hint">
                {copy.disk.replace("{size}", String(model.disk_gb))} · {model.min_vram_gb} GB VRAM · {model.speed}
              </p>
              <div className="btn-row stack-buttons">
                <button className="btn" disabled={active === model.name || busy === model.name} onClick={() => onUse(model.name)}>
                  {copy.useModel}
                </button>
                <button className="btn" disabled={present || running} onClick={() => onInstall(model.name)}>
                  {running ? copy.installing : copy.install}
                </button>
              </div>
              {running && <PullBar pull={pull} />}
              {pull?.state === "error" && pull.error && <p className="error-text">{pull.error}</p>}
            </article>
          );
        })}
      </div>
    </>
  );
}

function PullBar({ pull }: { pull?: PullStatus }) {
  const total = Number(pull?.total || 0);
  const completed = Number(pull?.completed || 0);
  const pct = total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : 0;
  return (
    <div>
      <div className="progress-line" aria-hidden="true">
        <span style={{ width: `${pct}%` }} />
      </div>
      <p className="hint">{pull?.status || copy.installing}{total > 0 ? ` · ${pct}%` : ""}</p>
    </div>
  );
}

function TestResult({ result }: { result?: AiTestResult | null }) {
  if (!result) return null;
  return (
    <div className="test-result">
      {result.error && <p className="error-text">{result.error}</p>}
      {result.warning && <p className="hint">{result.warning}</p>}
      <div className="meta-grid">
        <Probe label="Text" probe={result.text} />
        <Probe label="Vision" probe={result.vision} />
        <Probe label="Whisper" probe={result.whisper} />
      </div>
    </div>
  );
}

function Probe({ label, probe }: { label: string; probe?: AiTestResult["text"] }) {
  if (!probe) return <Stat label={label} value="—" />;
  if (probe.skipped) return <Stat label={label} value={probe.message || "Skipped"} />;
  if (!probe.ok) return <Stat label={label} value={probe.error || "Failed"} />;
  if (probe.tokens_per_sec != null) {
    const place = probe.placement?.label ? ` · ${probe.placement.label}` : "";
    const memory = formatPlacementMemory(probe.placement);
    return <Stat label={label} value={`${probe.tokens_per_sec} tok/s · ${probe.elapsed_sec}s${place}${memory}`} />;
  }
  const device = probe.device ? `${probe.device}${probe.compute_type ? ` ${probe.compute_type}` : ""}` : "done";
  const speed = probe.realtime_factor ? ` · ${probe.realtime_factor}x` : "";
  return <Stat label={label} value={`${device}${speed}${probe.fallback ? " · CPU fallback" : ""}`} />;
}

function formatPlacementMemory(placement?: { size?: number | null; size_vram?: number | null }) {
  if (!placement?.size) return "";
  const total = formatBytes(placement.size);
  const vram = formatBytes(placement.size_vram || 0);
  return ` · ${vram} VRAM / ${total}`;
}

function modelPresent(name: string, ollama: OllamaStatus | null) {
  const models = ollama?.models || [];
  return models.some((item) => item === name || item.startsWith(`${name}:`));
}
