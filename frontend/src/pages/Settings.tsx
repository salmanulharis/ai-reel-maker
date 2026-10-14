import { useEffect, useId, useState, type ReactNode } from "react";
import { api, formatBytes } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, InlineStatus, PageLoader, Spinner } from "../components/Feedback";
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

// Model choices are saved only when changed, so an untouched form never switches to custom mode.
const INFERENCE_KEYS = [
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
const ADVANCED_KEYS = [
  "ollama_base_url",
  "ollama_timeout",
  "ollama_max_retries",
  "scene_threshold",
  "output_video_codec",
  "output_crf",
  ...INFERENCE_KEYS,
];

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

  function flashSaved(key: string) {
    setSaved(key);
    setTimeout(() => setSaved((current) => (current === key ? "" : current)), 2000);
  }

  /** Run a button's request with a busy state and shared error handling. */
  async function act(key: string, call: () => Promise<unknown>) {
    setError("");
    setBusy(key);
    try {
      await call();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  const changed = (key: string) => String(settings[key] ?? "") !== String(loaded[key] ?? "");
  const unsaved = ADVANCED_KEYS.some(changed);

  async function saveAdvanced() {
    const payload: Record<string, unknown> = {
      ollama_base_url: settings.ollama_base_url,
      ollama_timeout: settings.ollama_timeout,
      ollama_max_retries: settings.ollama_max_retries,
      scene_threshold: settings.scene_threshold,
      output_video_codec: settings.output_video_codec,
      output_crf: settings.output_crf,
    };
    for (const key of INFERENCE_KEYS) {
      if (changed(key)) payload[key] = settings[key];
    }
    await act("save", async () => {
      await api.saveSettings(payload);
      await load();
      flashSaved("advanced");
    });
  }

  /** Switches apply at once: there is no separate save step to forget. */
  async function toggleVisual(enabled: boolean) {
    set("enable_visual_analysis", enabled);
    await act("visual", async () => {
      try {
        await api.saveSettings({ enable_visual_analysis: enabled });
        setLoaded((cur) => ({ ...cur, enable_visual_analysis: enabled }));
        flashSaved("visual");
      } catch (err) {
        set("enable_visual_analysis", !enabled);
        throw err;
      }
    });
  }

  async function useModel(key: string, name: string) {
    await act(name, async () => {
      await api.saveSettings({ [key]: name });
      await load();
    });
  }

  async function install(names: string[]) {
    setError("");
    try {
      await api.pullModels(names);
      setOllama(await api.ollama());
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

  const hw = info?.hardware || {};
  const recommendation = info?.recommendation || {};
  const test = ollama?.test?.result;
  const runtime = test?.runtime || info?.runtime || ollama?.runtime;
  const checks = info?.checks || {};
  const mode = String(info?.profile?.mode || settings.ai_config_mode || "auto");
  const visionName = String(recommendation.vision_model || "");
  const textName = String(recommendation.text_model || "");
  const missingRecommended = [visionName, textName].filter((name) => name && !modelPresent(name, ollama));
  const gb = (value: unknown) => (value ? `${value} GB` : "—");

  if (!info && !error) return <PageLoader />;

  return (
    <div>
      {error && <Alert>{error}</Alert>}

      <section className="panel" aria-labelledby="ai-title">
        <h3 id="ai-title">{copy.ai}</h3>
        <p className="panel-intro">{copy.aiIntro}</p>
        <div className="status-strip">
          <span className={`mode-badge ${runtime?.mode === "gpu" ? "gpu" : runtime?.mode === "split" ? "split" : "cpu"}`}>
            {runtime?.label || copy.hardware}
          </span>
          <span className="muted">{mode === "auto" ? copy.automatic : copy.customActive}</span>
          <span className="muted">{runtime?.measured ? copy.measured : copy.detected}</span>
        </div>
        {runtime?.hint && <p className="hint">{runtime.hint}</p>}
        {info?.profile?.auto_summary && mode === "auto" && <p className="recommend">{info.profile.auto_summary}</p>}
        {Array.isArray(recommendation.notes) &&
          recommendation.notes.map((note) => (
            <p key={String(note)} className="hint">
              {String(note)}
            </p>
          ))}
        {(ollama?.warnings || []).map((warning) => (
          <Alert key={warning}>{warning}</Alert>
        ))}

        <div className="btn-row settings-actions">
          {missingRecommended.length > 0 && (
            <button
              className="btn primary"
              disabled={pullBusy}
              aria-busy={pullBusy}
              onClick={() => install(missingRecommended)}
            >
              {pullBusy && <Spinner />}
              {copy.installRecommended}
            </button>
          )}
          <button
            className={`btn ${mode === "auto" ? "" : "primary"}`}
            disabled={busy === "apply" || mode === "auto"}
            aria-busy={busy === "apply"}
            onClick={() =>
              act("apply", async () => {
                await api.applyRecommended();
                await load();
              })
            }
          >
            {busy === "apply" && <Spinner />}
            {copy.useRecommended}
          </button>
          {mode === "auto" && (
            <button
              className="btn"
              onClick={() =>
                act("custom", async () => {
                  await api.saveSettings({ ai_config_mode: "custom" });
                  await load();
                })
              }
            >
              {copy.custom}
            </button>
          )}
          <button className="btn" disabled={testBusy} aria-busy={testBusy} aria-describedby="test-hint" onClick={() =>
            act("test", async () => {
              await api.startAiTest();
              setOllama(await api.ollama());
            })
          }>
            {testBusy && <Spinner />}
            {testBusy ? copy.testing : copy.test}
          </button>
          <span className="muted small" id="test-hint">
            {copy.testHint}
          </span>
        </div>
        <TestResult result={test} />

        <div className="switch-row">
          <label className="check">
            <input
              type="checkbox"
              checked={Boolean(settings.enable_visual_analysis)}
              disabled={busy === "visual"}
              aria-describedby="visual-hint"
              onChange={(e) => void toggleVisual(e.target.checked)}
            />
            <span>
              <strong>{copy.visual}</strong>
              <span className="field-hint block" id="visual-hint">
                {copy.visualHint}
              </span>
            </span>
          </label>
          {saved === "visual" && <InlineStatus ok>{copy.saved}</InlineStatus>}
        </div>

        <ModelGroup
          title={copy.vision}
          hint={copy.visionHint}
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
          hint={copy.textHint}
          models={info?.catalog?.text || []}
          active={String(settings.ollama_text_model || "")}
          recommended={textName}
          ollama={ollama}
          busy={busy}
          onUse={(name) => useModel("ollama_text_model", name)}
          onInstall={(name) => install([name])}
        />
        <h4 className="group-title">{copy.whisper}</h4>
        <p className="hint group-hint">{copy.whisperHint}</p>
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
                    {active && <span className="tag accent">{copy.inUse}</span>}
                    {model.name === recommendation.whisper_model && <span className="tag accent">{copy.recommended}</span>}
                    <span className={`tag ${cached ? "ok" : ""}`}>{cached ? copy.installed : copy.missing}</span>
                  </span>
                </header>
                <p>{model.summary}</p>
                <p className="hint">
                  {fill(copy.disk, { size: model.disk_gb })} · {model.speed}
                </p>
                <div className="btn-row">
                  <button
                    className="btn sm"
                    disabled={active || busy === model.name}
                    aria-busy={busy === model.name}
                    onClick={() => useModel("whisper_model", model.name)}
                  >
                    {busy === model.name && <Spinner />}
                    {active ? copy.inUse : copy.useModel}
                  </button>
                  {!cached && (
                    <button className="btn sm" disabled={downloading} aria-busy={downloading} onClick={() => downloadWhisper(model.name)}>
                      {downloading && <Spinner />}
                      {downloading ? copy.downloading : copy.download}
                    </button>
                  )}
                </div>
                {ollama?.whisper_download?.model === model.name && ollama?.whisper_download?.error && (
                  <p className="error-text">{ollama.whisper_download.error}</p>
                )}
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
      </section>

      <section className="panel" aria-labelledby="hw-title">
        <h3 id="hw-title">{copy.hardware}</h3>
        <p className="panel-intro">{copy.hardwareHint}</p>
        <div className="meta-grid">
          <Stat label={copy.cpu} value={String(hw.cpu || "—")} />
          <Stat label={copy.cores} value={hw.cpu_cores ? String(hw.cpu_cores) : "—"} />
          <Stat label={copy.ram} value={gb(hw.ram_gb)} />
          <Stat label={copy.freeRam} value={gb(hw.ram_available_gb)} />
          <Stat label={copy.gpu} value={String(hw.gpu || "—")} />
          <Stat label={copy.vramTotal} value={gb(hw.vram_gb)} />
          <Stat label={copy.freeVram} value={gb(hw.vram_free_gb)} />
          <Stat label={copy.driver} value={String(hw.driver || "—")} />
          <Stat label={copy.freeDisk} value={gb(hw.disk_free_gb)} />
          <Stat label={copy.encode} value={String(hw.encode_acceleration || "—")} />
        </div>
        {hw.vram_note ? <p className="hint">{String(hw.vram_note)}</p> : null}
        {Object.keys(checks).length > 0 && (
          <>
            <h4 className="group-title">{copy.checksTitle}</h4>
            <ul className="checks">
              {Object.entries(checks).map(([key, ok]) => (
                <li key={key} className={`check-item ${ok ? "ok" : "bad"}`}>
                  <span aria-hidden="true">{ok ? "✓" : "✗"}</span>
                  {copy.checks[key as keyof typeof copy.checks] || key.replace(/_/g, " ")}
                  <span className="sr-only">{ok ? ": ready" : ": not ready"}</span>
                </li>
              ))}
            </ul>
          </>
        )}
        <div className="btn-row panel-actions">
          <button
            className="btn"
            disabled={busy === "refresh"}
            aria-busy={busy === "refresh"}
            onClick={() => act("refresh", () => load(true))}
          >
            {busy === "refresh" && <Spinner />}
            {copy.refresh}
          </button>
        </div>
      </section>

      <details className="panel advanced" open={unsaved || undefined}>
        <summary>{copy.advanced}</summary>
        <p className="panel-intro">{copy.advancedHint}</p>

        <h4 className="group-title">{copy.sectionAi}</h4>
        <TextField
          label={copy.ollamaUrl}
          hint={copy.ollamaUrlHint}
          value={String(settings.ollama_base_url || "")}
          onChange={(value) => set("ollama_base_url", value)}
        />
        <div className="grid-2">
          <NumberField label={copy.timeout} hint={copy.timeoutHint} value={settings.ollama_timeout} onChange={(value) => set("ollama_timeout", value)} />
          <NumberField label={copy.retries} hint={copy.retriesHint} value={settings.ollama_max_retries} onChange={(value) => set("ollama_max_retries", value)} />
        </div>

        <h4 className="group-title">{copy.sectionSpeed}</h4>
        <div className="grid-2">
          <NumberField label={copy.batch} hint={copy.batchHint} value={settings.ollama_vision_batch_size} onChange={(value) => set("ollama_vision_batch_size", value)} />
          <NumberField label={copy.threads} hint={copy.threadsHint} value={settings.ollama_num_thread} onChange={(value) => set("ollama_num_thread", value)} />
          <NumberField label={copy.visionCtx} hint={copy.visionCtxHint} value={settings.ollama_vision_num_ctx} onChange={(value) => set("ollama_vision_num_ctx", value)} />
          <NumberField label={copy.textCtx} hint={copy.textCtxHint} value={settings.ollama_text_num_ctx} onChange={(value) => set("ollama_text_num_ctx", value)} />
          <Field label={copy.whisperDevice} hint={copy.whisperDeviceHint}>
            {(id, hintId) => (
              <select id={id} aria-describedby={hintId} value={String(settings.whisper_device || "auto")} onChange={(e) => set("whisper_device", e.target.value)}>
                <option value="auto">{copy.deviceAuto}</option>
                <option value="cpu">{copy.deviceCpu}</option>
                <option value="cuda">{copy.deviceCuda}</option>
              </select>
            )}
          </Field>
        </div>

        <h4 className="group-title">{copy.sectionVideo}</h4>
        <div className="grid-2">
          <NumberField label={copy.scene} hint={copy.sceneHint} value={settings.scene_threshold} onChange={(value) => set("scene_threshold", value)} />
          <NumberField label={copy.crf} hint={copy.crfHint} value={settings.output_crf} onChange={(value) => set("output_crf", value)} />
          <Field label={copy.codec} hint={copy.codecHint}>
            {(id, hintId) => (
              <select id={id} aria-describedby={hintId} value={String(settings.output_video_codec || "auto")} onChange={(e) => set("output_video_codec", e.target.value)}>
                <option value="auto">{copy.codecAuto}</option>
                <option value="h264">H.264</option>
                <option value="h265">H.265 (HEVC)</option>
                <option value="vp9">VP9</option>
                <option value="av1">AV1</option>
              </select>
            )}
          </Field>
        </div>

        <div className="btn-row panel-actions">
          <button className="btn primary" disabled={busy === "save" || !unsaved} aria-busy={busy === "save"} onClick={saveAdvanced}>
            {busy === "save" && <Spinner />}
            {copy.save}
          </button>
          {unsaved && busy !== "save" && <span className="muted">{copy.unsaved}</span>}
          {saved === "advanced" && <InlineStatus ok>{copy.saved}</InlineStatus>}
        </div>
      </details>

      <section className="panel" aria-labelledby="storage-title">
        <h3 id="storage-title">{copy.storage}</h3>
        <p className="panel-intro">{copy.storageHint}</p>
        <div className="meta-grid">
          <Stat label={copy.storageJobs} value={formatBytes(Number(storage.jobs_bytes || 0))} />
          <Stat label={copy.storageOutputs} value={formatBytes(Number(storage.outputs_bytes || 0))} />
          <Stat label={copy.storageCache} value={formatBytes(Number(storage.cache_bytes || 0))} />
          <Stat label={copy.storageTmp} value={formatBytes(Number(storage.tmp_bytes || 0))} />
        </div>
        <div className="storage-actions">
          <StorageAction
            label={copy.clearTmp}
            hint={copy.clearTmpHint}
            busy={busy === "tmp"}
            onClick={() => act("tmp", () => api.clearTmp().then(() => load()))}
          />
          <StorageAction
            label={copy.clearCache}
            hint={copy.clearCacheHint}
            busy={busy === "cache"}
            onClick={() => {
              if (!window.confirm(copy.confirmClearCache)) return;
              void act("cache", () => api.clearCache().then(() => load()));
            }}
          />
          <StorageAction
            label={copy.deleteCompleted}
            hint={copy.deleteCompletedHint}
            danger
            busy={busy === "completed"}
            onClick={() => {
              if (!window.confirm(messages.common.confirmDeleteCompleted)) return;
              void act("completed", () => api.deleteCompleted().then(() => load()));
            }}
          />
        </div>
      </section>
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

function StorageAction({
  label,
  hint,
  busy,
  danger = false,
  onClick,
}: {
  label: string;
  hint: string;
  busy: boolean;
  danger?: boolean;
  onClick: () => void;
}) {
  const id = useId();
  return (
    <div className="storage-action">
      <button className={`btn ${danger ? "danger" : ""}`} disabled={busy} aria-busy={busy} aria-describedby={id} onClick={onClick}>
        {busy && <Spinner />}
        {label}
      </button>
      <span className="hint" id={id}>
        {hint}
      </span>
    </div>
  );
}

/** A labelled field with a visible hint, both linked to the control for screen readers. */
function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: (id: string, hintId?: string) => ReactNode;
}) {
  const id = useId();
  const hintId = hint ? `${id}-hint` : undefined;
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children(id, hintId)}
      {hint && (
        <p className="field-hint" id={hintId}>
          {hint}
        </p>
      )}
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
    <Field label={label} hint={hint}>
      {(id, hintId) => (
        <input id={id} type="number" aria-describedby={hintId} value={Number(value || 0)} onChange={(e) => onChange(Number(e.target.value))} />
      )}
    </Field>
  );
}

function TextField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <Field label={label} hint={hint}>
      {(id, hintId) => (
        <input id={id} aria-describedby={hintId} spellCheck={false} value={value} onChange={(e) => onChange(e.target.value)} />
      )}
    </Field>
  );
}

function ModelGroup({
  title,
  hint,
  models,
  active,
  recommended,
  ollama,
  busy,
  onUse,
  onInstall,
}: {
  title: string;
  hint: string;
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
      <p className="hint group-hint">{hint}</p>
      <div className="model-list">
        {models.map((model) => {
          const present = modelPresent(model.name, ollama);
          const pull = ollama?.pull?.[model.name];
          const running = pull?.state === "running";
          const inUse = active === model.name;
          return (
            <article key={model.name} className={`model-card ${inUse ? "active" : ""}`}>
              <header>
                <strong>{model.name}</strong>
                <span className="tag-row">
                  {inUse && <span className="tag accent">{copy.inUse}</span>}
                  {model.name === recommended && <span className="tag accent">{copy.recommended}</span>}
                  <span className={`tag ${present ? "ok" : ""}`}>{present ? copy.installed : copy.missing}</span>
                </span>
              </header>
              <p>{model.summary}</p>
              <p className="hint">
                {fill(copy.disk, { size: model.disk_gb })} · {fill(copy.vram, { size: model.min_vram_gb })} · {model.speed}
              </p>
              <div className="btn-row">
                <button
                  className="btn sm"
                  disabled={inUse || busy === model.name}
                  aria-busy={busy === model.name}
                  onClick={() => onUse(model.name)}
                >
                  {busy === model.name && <Spinner />}
                  {inUse ? copy.inUse : copy.useModel}
                </button>
                {!present && (
                  <button className="btn sm" disabled={running} aria-busy={running} onClick={() => onInstall(model.name)}>
                    {running && <Spinner />}
                    {running ? copy.installing : copy.install}
                  </button>
                )}
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
      <div
        className="progress-line"
        role="progressbar"
        aria-label={copy.installing}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={total > 0 ? pct : undefined}
      >
        <span style={{ width: `${pct}%` }} />
      </div>
      <p className="hint">
        {copy.installing}
        {total > 0 ? ` ${pct}% · ${formatBytes(completed)} / ${formatBytes(total)}` : ""}
      </p>
    </div>
  );
}

function TestResult({ result }: { result?: AiTestResult | null }) {
  if (!result) return null;
  const probe = copy.probe;
  return (
    <div className="test-result" role="status">
      {result.error && <Alert>{result.error}</Alert>}
      {result.warning && <p className="hint">{result.warning}</p>}
      <div className="meta-grid">
        <Probe label={probe.vision} probe={result.vision} />
        <Probe label={probe.text} probe={result.text} />
        <Probe label={probe.whisper} probe={result.whisper} />
      </div>
    </div>
  );
}

function Probe({ label, probe }: { label: string; probe?: AiTestResult["text"] }) {
  const text = copy.probe;
  if (!probe) return <Stat label={label} value="—" />;
  if (probe.skipped) return <Stat label={label} value={probe.message || text.skipped} />;
  if (!probe.ok) return <Stat label={label} value={`✗ ${probe.error || text.failed}`} />;
  if (probe.tokens_per_sec != null) {
    const parts = [
      `✓ ${fill(text.speed, { n: probe.tokens_per_sec })}`,
      fill(text.time, { n: probe.elapsed_sec ?? "—" }),
      probe.placement?.label,
    ].filter(Boolean);
    return <Stat label={label} value={parts.join(" · ")} />;
  }
  const parts = [
    "✓",
    probe.device ? (probe.device === "cuda" ? "GPU" : "CPU") : "",
    probe.realtime_factor ? fill(text.realtime, { n: probe.realtime_factor }) : "",
    probe.fallback ? text.cpuFallback : "",
  ].filter(Boolean);
  return <Stat label={label} value={parts.join(" ")} />;
}

function modelPresent(name: string, ollama: OllamaStatus | null) {
  const models = ollama?.models || [];
  return models.some((item) => item === name || item.startsWith(`${name}:`));
}
