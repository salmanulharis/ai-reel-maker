import { useEffect, useId, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";
import { api, formatBytes } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, InlineStatus, PageLoader, Spinner } from "../components/Feedback";
import { DocLink } from "./Docs";
import type { DocArticleId } from "../docs/content";
import type { AiTestResult, ModelSpec, OllamaStatus, OptimizationItem, OptimizationPlan, PullStatus, RuntimeStatus } from "../types";

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

type SectionId = keyof typeof copy.sections;

// Everyday sections first; tuning and connection details are grouped under "Advanced".
const NAV: Array<{ group: string; items: SectionId[] }> = [
  { group: copy.navGeneral, items: ["optimize", "ai", "speech", "video", "storage", "system"] },
  { group: copy.navAdvanced, items: ["performance", "connection"] },
];
const SECTIONS = NAV.flatMap((entry) => entry.items);

// Fields saved together with "Save changes", by the section that shows them. Everything else
// on the page (model choices, switches, storage actions) applies right away.
const SECTION_FIELDS: Partial<Record<SectionId, string[]>> = {
  speech: ["whisper_device"],
  video: ["scene_threshold", "output_crf", "output_video_codec"],
  performance: ["ollama_vision_batch_size", "ollama_num_thread", "ollama_vision_num_ctx", "ollama_text_num_ctx"],
  connection: ["ollama_base_url", "ollama_timeout", "ollama_max_retries"],
};
const FORM_KEYS = Object.values(SECTION_FIELDS).flat();
const DEFAULT_SECTION: SectionId = "optimize";
// The help article behind each section, linked under its introduction.
const SECTION_HELP: Partial<Record<SectionId, DocArticleId>> = {
  optimize: "cpu-speed",
  ai: "choose-models",
  speech: "setup-whisper",
  system: "hardware-detection",
  performance: "threads-memory",
  connection: "remote-gpu",
};
const opt = copy.optimize;
// Model choices are saved only when changed, so an untouched form never switches to custom mode.
const INFERENCE_KEYS = new Set([
  "whisper_device",
  "ollama_vision_batch_size",
  "ollama_vision_num_ctx",
  "ollama_text_num_ctx",
  "ollama_num_thread",
]);

function isSection(value: string | null): value is SectionId {
  return Boolean(value && (SECTIONS as string[]).includes(value));
}

export default function Settings() {
  const [params, setParams] = useSearchParams();
  const section: SectionId = isSection(params.get("section")) ? (params.get("section") as SectionId) : DEFAULT_SECTION;
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [loaded, setLoaded] = useState<Record<string, unknown>>({});
  const [storage, setStorage] = useState<Record<string, number | string>>({});
  const [info, setInfo] = useState<Info | null>(null);
  const [ollama, setOllama] = useState<OllamaStatus | null>(null);
  const [plan, setPlan] = useState<OptimizationPlan | null>(null);
  const [reviewing, setReviewing] = useState(false);
  const [saved, setSaved] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");

  const load = async (refresh = false) => {
    const [systemSettings, systemInfo, status, optimization] = await Promise.all([
      api.settings(),
      api.info({ refresh, encoders: refresh }),
      api.ollama(),
      api.optimization(refresh).catch(() => null),
    ]);
    setPlan(optimization);
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
  const imageBusy = ollama?.image_model?.state === "running";
  const whisperState = ollama?.whisper_download?.state;

  useEffect(() => {
    if (!pullBusy && !testBusy && !whisperBusy && !imageBusy) return;
    const id = setInterval(() => {
      api.ollama().then(setOllama).catch(() => undefined);
    }, 1000);
    return () => clearInterval(id);
  }, [pullBusy, testBusy, whisperBusy, imageBusy]);

  useEffect(() => {
    if (whisperState !== "done") return;
    api.info().then((systemInfo) => setInfo(systemInfo as Info)).catch(() => undefined);
  }, [whisperState]);

  const changed = (key: string) => String(settings[key] ?? "") !== String(loaded[key] ?? "");
  const dirtySections = new Set(SECTIONS.filter((id) => (SECTION_FIELDS[id] || []).some(changed)));
  const unsaved = dirtySections.size > 0;

  // Reloading or closing the tab would lose unsaved edits; let the browser ask first.
  useEffect(() => {
    if (!unsaved) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [unsaved]);

  function go(next: SectionId) {
    setParams(next === DEFAULT_SECTION ? {} : { section: next }, { replace: false });
  }

  function set(key: string, value: unknown) {
    setSettings((cur) => ({ ...cur, [key]: value }));
  }

  function flashSaved(key: string) {
    setSaved(key);
    setTimeout(() => setSaved((current) => (current === key ? "" : current)), 2500);
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

  async function saveChanges() {
    const payload: Record<string, unknown> = {};
    for (const key of FORM_KEYS) {
      if (!INFERENCE_KEYS.has(key) || changed(key)) payload[key] = settings[key];
    }
    await act("save", async () => {
      await api.saveSettings(payload);
      await load();
      flashSaved("form");
    });
  }

  function discardChanges() {
    setSettings((cur) => {
      const next = { ...cur };
      for (const key of FORM_KEYS) next[key] = loaded[key];
      return next;
    });
  }

  /** Turning optimization on shows what will change first; turning it off keeps every value. */
  function toggleOptimization(enabled: boolean) {
    if (enabled && plan && plan.changes.length > 0) {
      setReviewing(true);
      return;
    }
    void setOptimization(enabled);
  }

  async function setOptimization(enabled: boolean) {
    setReviewing(false);
    await act("optimize", async () => {
      setPlan(await api.setOptimization(enabled));
      await load();
      flashSaved("optimize");
    });
  }

  async function undoOptimization() {
    if (!window.confirm(opt.confirmUndo)) return;
    await act("undo", async () => {
      setPlan(await api.undoOptimization());
      await load();
      flashSaved("optimize");
    });
  }

  /** Save one choice right away, restoring the previous value if saving fails. */
  async function setInstant(key: string, value: unknown, flash: string) {
    const previous = settings[key];
    set(key, value);
    await act(flash, async () => {
      try {
        await api.saveSettings({ [key]: value });
        setLoaded((cur) => ({ ...cur, [key]: value }));
        flashSaved(flash);
      } catch (err) {
        set(key, previous);
        throw err;
      }
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
  const embedModel = String(settings.ollama_embed_model || "");
  const embedInstalled = Boolean(embedModel) && modelPresent(embedModel, ollama);
  const embedPull = embedModel ? ollama?.pull?.[embedModel] : undefined;
  const imageModel = ollama?.image_model;
  const imageDownloading = imageModel?.state === "running";
  const gb = (value: unknown) => (value ? `${value} GB` : "—");

  if (!info && !error) return <PageLoader />;

  const enabled = Boolean(plan?.enabled);
  const lastRun = plan?.history?.[0];
  const views: Record<SectionId, ReactNode> = {
    optimize: !plan ? (
      <p className="hint">{messages.common.loading}</p>
    ) : (
      <>
        <div className="switch-row">
          <label className="check">
            <input
              type="checkbox"
              checked={enabled || reviewing}
              disabled={busy === "optimize" || busy === "undo"}
              aria-describedby="optimize-hint"
              onChange={(e) => toggleOptimization(e.target.checked)}
            />
            <span>
              <strong>{opt.switch}</strong>
              <span className="field-hint block" id="optimize-hint">
                {opt.switchHint}
              </span>
            </span>
          </label>
          {busy === "optimize" && <InlineStatus busy>{messages.common.working}</InlineStatus>}
          {saved === "optimize" && <InlineStatus ok>{copy.saved}</InlineStatus>}
        </div>

        {reviewing && (
          <div className="review-box reveal" role="region" aria-label={opt.reviewTitle}>
            <h4 className="group-title">{opt.reviewTitle}</h4>
            <p className="hint group-hint">{opt.reviewHint}</p>
            <ChangeList items={plan.changes} />
            <div className="btn-row panel-actions">
              <button className="btn primary" onClick={() => void setOptimization(true)}>
                {opt.confirm}
              </button>
              <button className="btn ghost" onClick={() => setReviewing(false)}>
                {opt.cancel}
              </button>
            </div>
          </div>
        )}

        <h4 className="group-title">{opt.detected}</h4>
        <p className="hardware-line">
          {plan.hardware.text}
          {plan.hardware.system ? ` · ${plan.hardware.system}` : ""}
          <button
            className="btn sm ghost"
            disabled={busy === "refresh"}
            aria-busy={busy === "refresh"}
            onClick={() => act("refresh", () => load(true))}
          >
            {busy === "refresh" && <Spinner />}
            {opt.checkAgain}
          </button>
        </p>

        {plan.warnings.length > 0 && (
          <>
            <h4 className="group-title">{opt.warnings}</h4>
            {plan.warnings.map((warning) => (
              <Alert key={warning}>{warning}</Alert>
            ))}
          </>
        )}

        {enabled ? (
          <>
            <h4 className="group-title">{opt.chosen}</h4>
            <ChangeList items={plan.items} showCurrentOnly />
          </>
        ) : (
          !reviewing && (
            <>
              <p className="hint">{opt.offNote}</p>
              {plan.changes.length > 0 ? (
                <>
                  <h4 className="group-title">{opt.recommendedTitle}</h4>
                  <p className="hint group-hint">{opt.recommendedHint}</p>
                  <ChangeList items={plan.changes} />
                  <div className="btn-row panel-actions">
                    <button className="btn" onClick={() => toggleOptimization(true)}>
                      {opt.apply}
                    </button>
                  </div>
                </>
              ) : (
                <p className="hint">{opt.upToDate}</p>
              )}
            </>
          )
        )}

        {(plan.tradeoffs.length > 0 || plan.notes.length > 0) && (
          <>
            <h4 className="group-title">{opt.tradeoffs}</h4>
            <ul className="plain-list">
              {[...plan.tradeoffs, ...plan.notes].map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </>
        )}

        <h4 className="group-title">{opt.tips}</h4>
        {lastRun ? (
          <>
            <p className="hint group-hint">
              {fill(opt.lastRun, { video: formatDuration(lastRun.video_seconds), time: formatDuration(lastRun.total_seconds) })}{" "}
              {opt.tipsHint}
            </p>
            {plan.tips.length > 0 && (
              <ul className="plain-list">
                {plan.tips.map((tip) => (
                  <li key={tip}>{tip}</li>
                ))}
              </ul>
            )}
          </>
        ) : (
          <p className="hint">{opt.noHistory}</p>
        )}

        {plan.can_undo && (
          <div className="storage-action panel-actions">
            <button className="btn" disabled={busy === "undo"} aria-busy={busy === "undo"} aria-describedby="undo-hint" onClick={undoOptimization}>
              {busy === "undo" && <Spinner />}
              {opt.undo}
            </button>
            <span className="hint" id="undo-hint">
              {fill(opt.undoHint, { when: plan.undo_saved_at ? new Date(plan.undo_saved_at * 1000).toLocaleString() : "" })}
            </span>
          </div>
        )}
      </>
    ),

    ai: (
      <>
        <div className="status-strip">
          <span className={`mode-badge ${runtime?.mode === "gpu" ? "gpu" : runtime?.mode === "split" ? "split" : "cpu"}`}>
            {runtime?.label || copy.hardware}
          </span>
          <span className="muted">{mode === "auto" ? copy.automatic : copy.customActive}</span>
          <span className="muted">{runtime?.measured ? copy.measured : copy.detected}</span>
        </div>
        {runtime?.hint && <p className="hint">{runtime.hint}</p>}
        <ManagedNote enabled={mode === "auto"} onOpen={() => go("optimize")} />
        {(ollama?.warnings || []).map((warning) => (
          <Alert key={warning}>{warning}</Alert>
        ))}

        <div className="btn-row settings-actions">
          {missingRecommended.length > 0 && (
            <button className="btn primary" disabled={pullBusy} aria-busy={pullBusy} onClick={() => install(missingRecommended)}>
              {pullBusy && <Spinner />}
              {copy.installRecommended}
            </button>
          )}
          <button
            className="btn"
            disabled={testBusy}
            aria-busy={testBusy}
            aria-describedby="test-hint"
            onClick={() =>
              act("test", async () => {
                await api.startAiTest();
                setOllama(await api.ollama());
              })
            }
          >
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

        {Boolean(settings.enable_visual_analysis) && (
          <fieldset className="choice-group">
            <legend>{copy.visionScope}</legend>
            {(["shortlist", "all"] as const).map((scope) => (
              <label key={scope} className="check">
                <input
                  type="radio"
                  name="vision-scope"
                  value={scope}
                  checked={String(settings.vision_scope || "shortlist") === scope}
                  disabled={busy === "scope"}
                  aria-describedby={`scope-${scope}-hint`}
                  onChange={() => void setInstant("vision_scope", scope, "scope")}
                />
                <span>
                  {scope === "shortlist" ? copy.visionScopeShortlist : copy.visionScopeAll}
                  <span className="field-hint block" id={`scope-${scope}-hint`}>
                    {scope === "shortlist" ? copy.visionScopeShortlistHint : copy.visionScopeAllHint}
                  </span>
                </span>
              </label>
            ))}
            {saved === "scope" && <InlineStatus ok>{copy.saved}</InlineStatus>}
          </fieldset>
        )}

        <div className="embed-row">
          <div>
            <strong>{copy.embedTitle}</strong>
            <p className="field-hint">{copy.embedHint}</p>
          </div>
          <div className="btn-row">
            <span className={`tag ${embedInstalled ? "ok" : ""}`}>
              {fill(embedInstalled ? copy.embedReady : copy.embedMissing, { model: embedModel })}
            </span>
            {!embedInstalled && embedModel && (
              <button
                className="btn sm"
                disabled={embedPull?.state === "running"}
                aria-busy={embedPull?.state === "running"}
                onClick={() => install([embedModel])}
              >
                {embedPull?.state === "running" && <Spinner />}
                {embedPull?.state === "running" ? copy.installing : copy.install}
              </button>
            )}
          </div>
          {embedPull?.state === "running" && <PullBar pull={embedPull} />}
          {embedPull?.state === "error" && embedPull.error && <p className="error-text">{embedPull.error}</p>}
        </div>

        <div className="embed-row">
          <div>
            <label className="check">
              <input
                type="checkbox"
                checked={settings.image_matching !== false}
                disabled={busy === "image"}
                aria-describedby="image-matching-hint"
                onChange={(e) => void setInstant("image_matching", e.target.checked, "image")}
              />
              <strong>{copy.imageMatching}</strong>
            </label>
            <p className="field-hint" id="image-matching-hint">
              {copy.imageMatchingHint}
            </p>
          </div>
          <div className="btn-row">
            {saved === "image" && <InlineStatus ok>{copy.saved}</InlineStatus>}
            <span className={`tag ${imageModel?.installed ? "ok" : ""}`}>
              {imageModel?.installed
                ? copy.imageModelReady
                : imageDownloading
                  ? fill(copy.imageModelDownloading, { done: imageModel?.files_done ?? 0, total: imageModel?.files_total ?? 3 })
                  : fill(copy.imageModelMissing, { size: imageModel?.size_mb ?? 780 })}
            </span>
            {!imageModel?.installed && (
              <button
                className="btn sm"
                disabled={imageDownloading || imageModel?.runtime === false}
                aria-busy={imageDownloading}
                onClick={() =>
                  act("image-download", async () => {
                    await api.downloadImageModel();
                    setOllama(await api.ollama());
                  })
                }
              >
                {imageDownloading && <Spinner />}
                {imageDownloading ? copy.downloading : copy.download}
              </button>
            )}
          </div>
          {imageModel?.runtime === false && <p className="hint">{copy.imageModelNoRuntime}</p>}
          {imageModel?.state === "error" && imageModel.error && <p className="error-text">{imageModel.error}</p>}
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
      </>
    ),

    speech: (
      <>
        <div className="switch-row speech-switch">
          <label className="check">
            <input
              type="checkbox"
              checked={settings.enable_whisper !== false}
              disabled={busy === "speech"}
              aria-describedby="use-speech-hint"
              onChange={(e) => void setInstant("enable_whisper", e.target.checked, "speech")}
            />
            <span>
              <strong>{copy.useSpeech}</strong>
              <span className="field-hint block" id="use-speech-hint">
                {copy.useSpeechHint}
              </span>
            </span>
          </label>
          {saved === "speech" && <InlineStatus ok>{copy.saved}</InlineStatus>}
        </div>
        <ManagedNote enabled={mode === "auto"} onOpen={() => go("optimize")} />
        <div className="form-grid">
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
        <h4 className="group-title">{copy.whisper}</h4>
        <p className="hint group-hint">
          {copy.whisperHint} <span className="instant-note">{copy.instantNote}</span>
        </p>
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
      </>
    ),

    video: (
      <div className="form-grid">
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
        <NumberField label={copy.crf} hint={copy.crfHint} min={0} max={51} value={settings.output_crf} onChange={(value) => set("output_crf", value)} />
        <NumberField label={copy.scene} hint={copy.sceneHint} min={1} max={100} value={settings.scene_threshold} onChange={(value) => set("scene_threshold", value)} />
      </div>
    ),

    storage: (
      <>
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
      </>
    ),

    system: (
      <>
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
          <button className="btn" disabled={busy === "refresh"} aria-busy={busy === "refresh"} onClick={() => act("refresh", () => load(true))}>
            {busy === "refresh" && <Spinner />}
            {copy.refresh}
          </button>
        </div>
        <details className="help">
          <summary>{copy.help}</summary>
          <p>{copy.helpIntro}</p>
          <p>{copy.helpCpu}</p>
          <p>{copy.helpGpu}</p>
          <p>{copy.helpAmd}</p>
          <p>{copy.helpTrouble}</p>
        </details>
      </>
    ),

    performance: (
      <>
      <ManagedNote enabled={mode === "auto"} onOpen={() => go("optimize")} />
      <div className="form-grid">
        <NumberField label={copy.batch} hint={copy.batchHint} min={1} max={16} value={settings.ollama_vision_batch_size} onChange={(value) => set("ollama_vision_batch_size", value)} />
        <NumberField label={copy.threads} hint={copy.threadsHint} min={1} max={64} value={settings.ollama_num_thread} onChange={(value) => set("ollama_num_thread", value)} />
        <NumberField label={copy.visionCtx} hint={copy.visionCtxHint} min={1024} step={1024} value={settings.ollama_vision_num_ctx} onChange={(value) => set("ollama_vision_num_ctx", value)} />
        <NumberField label={copy.textCtx} hint={copy.textCtxHint} min={1024} step={1024} value={settings.ollama_text_num_ctx} onChange={(value) => set("ollama_text_num_ctx", value)} />
      </div>
      </>
    ),

    connection: (
      <>
        <TextField
          label={copy.ollamaUrl}
          hint={copy.ollamaUrlHint}
          value={String(settings.ollama_base_url || "")}
          onChange={(value) => set("ollama_base_url", value)}
        />
        <div className="form-grid">
          <NumberField label={copy.timeout} hint={copy.timeoutHint} min={10} value={settings.ollama_timeout} onChange={(value) => set("ollama_timeout", value)} />
          <NumberField label={copy.retries} hint={copy.retriesHint} min={0} max={10} value={settings.ollama_max_retries} onChange={(value) => set("ollama_max_retries", value)} />
        </div>
      </>
    ),
  };

  return (
    <div className="settings-layout">
      <nav className="settings-nav" aria-label={copy.navLabel}>
        {NAV.map((entry) => (
          <div key={entry.group} className="settings-nav-group">
            <p className="settings-nav-heading" aria-hidden="true">
              {entry.group}
            </p>
            <ul>
              {entry.items.map((id) => (
                <li key={id}>
                  <button
                    type="button"
                    className={`settings-nav-item ${section === id ? "active" : ""}`}
                    aria-current={section === id ? "page" : undefined}
                    onClick={() => go(id)}
                  >
                    {copy.sections[id]}
                    {dirtySections.has(id) && (
                      <span className="dirty-dot" role="img" aria-label={copy.unsavedDot} />
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      <div className="settings-body">
        {error && <Alert>{error}</Alert>}
        <section className="panel settings-section" aria-labelledby="settings-section-title" key={section}>
          <h3 id="settings-section-title" className="sub-title settings-title">
            {copy.sections[section]}
          </h3>
          <p className="panel-intro">
            {copy.sectionIntros[section]}
            {SECTION_HELP[section] && (
              <>
                {" "}
                <DocLink article={SECTION_HELP[section] as DocArticleId} />
              </>
            )}
          </p>
          {views[section]}
        </section>

        {(unsaved || saved === "form") && (
          <div className="action-bar save-bar" role="region" aria-label={copy.saveChanges}>
            <div className="btn-row">
              {unsaved ? (
                <>
                  <button className="btn primary" disabled={busy === "save"} aria-busy={busy === "save"} onClick={saveChanges}>
                    {busy === "save" && <Spinner />}
                    {copy.saveChanges}
                  </button>
                  <button className="btn ghost" disabled={busy === "save"} onClick={discardChanges}>
                    {copy.discard}
                  </button>
                  <span className="muted" role="status">
                    {copy.saveBar}
                  </span>
                </>
              ) : (
                <InlineStatus ok>{copy.savedChanges}</InlineStatus>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ManagedNote({ enabled, onOpen }: { enabled: boolean; onOpen: () => void }) {
  if (!enabled) return null;
  return (
    <p className="managed-note">
      {opt.managedNote}{" "}
      <button type="button" className="link-button" onClick={onOpen}>
        {copy.sections.optimize}
      </button>
    </p>
  );
}

const DEVICE_LABELS: Record<string, string> = {
  auto: copy.deviceAuto,
  cpu: copy.deviceCpu,
  cuda: copy.deviceCuda,
};

function showValue(item: OptimizationItem, value: OptimizationItem["current"]) {
  if (value === null || value === undefined || value === "") return "—";
  if (item.key === "whisper_device") return DEVICE_LABELS[String(value)] || String(value);
  return String(value);
}

/** Settings with their values and the reason for each. */
function ChangeList({ items, showCurrentOnly = false }: { items: OptimizationItem[]; showCurrentOnly?: boolean }) {
  return (
    <ul className="change-list">
      {items.map((item) => (
        <li key={item.key}>
          <div className="change-head">
            <span className="change-label">{item.label}</span>
            {showCurrentOnly ? (
              <strong>{showValue(item, item.current)}</strong>
            ) : (
              <span className="change-values">
                <span className="muted" aria-label={opt.current}>
                  {showValue(item, item.current)}
                </span>
                <span aria-hidden="true"> → </span>
                <strong aria-label={opt.recommended}>{showValue(item, item.recommended)}</strong>
              </span>
            )}
          </div>
          {item.reason && <p className="hint">{item.reason}</p>}
        </li>
      ))}
    </ul>
  );
}

function formatDuration(seconds: number) {
  const total = Math.max(0, Math.round(seconds || 0));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (!minutes) return `${rest} s`;
  return rest ? `${minutes} min ${rest} s` : `${minutes} min`;
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
  min,
  max,
  step,
}: {
  label: string;
  hint?: string;
  value: unknown;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <Field label={label} hint={hint}>
      {(id, hintId) => (
        <input
          id={id}
          type="number"
          inputMode="numeric"
          min={min}
          max={max}
          step={step}
          aria-describedby={hintId}
          value={Number(value || 0)}
          onChange={(e) => onChange(Number(e.target.value))}
        />
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
