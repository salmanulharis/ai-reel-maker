import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { UploadZone } from "../components/UploadZone";
import { HighlightConfigPanel, defaultForm, formProblem } from "../components/HighlightConfig";
import { VideoCropPreview } from "../components/VideoCrop";
import { PresetBar } from "../components/PresetBar";
import { PromptPanel, understoodForm } from "../components/PromptPanel";
import { NarratedVideoPanel } from "../components/NarratedVideoPanel";
import { Alert, InlineStatus, Spinner } from "../components/Feedback";
import { TransferProgress } from "../components/Progress";
import { useUpload } from "../hooks/useUpload";
import { api, formatBytes, formatTime } from "../services/api";
import { isAbort } from "../services/transfer";
import { fill, messages } from "../i18n";
import type { HighlightForm, InterpretResult, Job } from "../types";

const MAX_FILES = 8;
const copy = messages.create;
const MODE_KEY = "create-mode";

/** The way of working chosen last time in this browser ("prompt" the first time). */
function startForm(): HighlightForm {
  const form = defaultForm();
  try {
    if (localStorage.getItem(MODE_KEY) === "manual") form.mode = "manual";
  } catch {
    // Storage can be unavailable (private window); the default is fine.
  }
  return form;
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [preview, setPreview] = useState<string>("");
  const [job, setJob] = useState<Job | null>(null);
  const [form, setForm] = useState<HighlightForm>(startForm);
  const [understood, setUnderstood] = useState<{ prompt: string; result: InterpretResult } | null>(null);
  const [busy, setBusy] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const upload = useUpload();
  const [aiNotice, setAiNotice] = useState(false);
  // With several videos: one highlight from all of them, or one highlight per video.
  const [separate, setSeparate] = useState(false);

  useEffect(() => {
    if (sessionStorage.getItem("ai-setup-dismissed")) return;
    api
      .ollama()
      .then((status) => {
        const visionReady = status.vision_ready ?? status.vision_present;
        const missing = !status.connected || !visionReady || !status.text_present;
        setAiNotice(missing);
      })
      .catch(() => setAiNotice(true));
  }, []);

  const previewFile = files[previewIndex] || files[0] || null;

  useEffect(() => {
    if (!previewFile) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(previewFile);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [previewFile]);

  async function syncJob(next: File[]) {
    setFiles(next);
    setPreviewIndex((i) => Math.max(0, Math.min(i, next.length - 1)));
    if (!next.length) {
      setJob(null);
      return;
    }
    setBusy(true);
    setError("");
    try {
      if (job) await api.deleteJob(job.id).catch(() => undefined);
      const size = next.reduce((sum, file) => sum + file.size, 0);
      const created = await upload.run(size, (options) => api.createJob(next, form, options));
      setJob(created);
    } catch (err) {
      if (isAbort(err)) {
        // Nothing was saved: start again from an empty list.
        setFiles([]);
        setJob(null);
        setError(messages.transfer.uploadCancelled);
      } else {
        setError((err as Error).message);
      }
    } finally {
      setBusy(false);
    }
  }

  function addFiles(incoming: File[]) {
    const merged = [...files];
    for (const file of incoming) {
      if (merged.length >= MAX_FILES) break;
      if (!merged.some((item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified)) {
        merged.push(file);
      }
    }
    void syncJob(merged);
  }

  function removeFile(index: number) {
    void syncJob(files.filter((_, i) => i !== index));
  }

  async function useSample() {
    setError("");
    setBusy(true);
    try {
      const response = await fetch("/api/media/sample").catch(() => null);
      if (!response) throw new Error(messages.errors.offline);
      if (!response.ok) throw new Error(messages.errors.sample);
      const blob = await response.blob();
      addFiles([new File([blob], "sample.mp4", { type: "video/mp4" })]);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  function chooseMode(mode: HighlightForm["mode"]) {
    setForm({ ...form, mode });
    try {
      localStorage.setItem(MODE_KEY, mode);
    } catch {
      // Only a convenience.
    }
  }

  async function analyze() {
    if (!job) return;
    setBusy(true);
    setStarting(true);
    setError("");
    try {
      let ready = form;
      if (form.mode === "prompt") {
        // A description that was not read yet (or changed since) is read now, so starting
        // straight away still gets the settings it asks for.
        const read = await understoodForm(form, job.id, understood);
        ready = read.form;
        if (read.result) {
          setForm(ready);
          setUnderstood({ prompt: form.prompt.trim(), result: read.result });
        }
      }
      if (separate && files.length > 1) {
        const result = await api.split(job.id, ready);
        navigate(`/history?queued=${result.jobs.length}`);
        return;
      }
      await api.analyze(job.id, ready);
      navigate(`/jobs/${job.id}/analysis`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
      setStarting(false);
    }
  }

  const meta = job?.metadata || {};
  const sources = job?.sources || [];
  const problem = formProblem(form);
  // Tell people why the main button is disabled instead of leaving them guessing.
  const blocker = !files.length ? copy.needVideo : busy && !starting ? copy.uploading : problem;

  return (
    <div>
      {aiNotice && (
        <div className="notice" role="status">
          <span>
            {messages.localAi.notice} <Link to="/settings?section=ai">{messages.localAi.action}</Link>
          </span>
          <button
            className="btn sm"
            type="button"
            onClick={() => {
              sessionStorage.setItem("ai-setup-dismissed", "1");
              setAiNotice(false);
            }}
          >
            {messages.localAi.dismiss}
          </button>
        </div>
      )}

      <section aria-labelledby="step-1">
        <h3 className="step-title" id="step-1">
          {copy.step1}
        </h3>
        <UploadZone onFiles={addFiles} files={files} onRemove={removeFile} />
        <div className="btn-row upload-actions">
          {!files.length && (
            <button className="btn ghost" type="button" disabled={busy} onClick={useSample}>
              {messages.upload.sample}
            </button>
          )}
          {busy && !starting && !upload.status && <InlineStatus busy>{messages.common.preparing}</InlineStatus>}
        </div>
        {upload.status && (
          <div className="upload-progress">
            <TransferProgress status={upload.status} label={messages.transfer.uploadLabel} />
            <button className="btn sm ghost" type="button" onClick={upload.cancel}>
              {messages.transfer.cancelUpload}
            </button>
          </div>
        )}
        {preview && (
          <>
            {files.length > 1 && (
              <div className="field">
                <label htmlFor="preview-video">{messages.config.cropSource}</label>
                <select id="preview-video" value={String(previewIndex)} onChange={(e) => setPreviewIndex(Number(e.target.value))}>
                  {files.map((file, i) => (
                    <option key={`${file.name}-${i}`} value={i}>
                      {fill(messages.upload.video, { n: i + 1 })}: {file.name}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <VideoCropPreview
              src={preview}
              enabled={form.cropEnabled}
              crop={form.crop}
              sourceWidth={sources[previewIndex]?.width || meta.width}
              sourceHeight={sources[previewIndex]?.height || meta.height}
              onToggle={(cropEnabled) => setForm({ ...form, cropEnabled })}
              onChange={(crop) => setForm({ ...form, crop })}
            />
          </>
        )}
        {files.length > 1 && (
          <fieldset className="batch-choice">
            <legend className="sub-title">{copy.batchLegend}</legend>
            <label className="check">
              <input type="radio" name="batch" checked={!separate} onChange={() => setSeparate(false)} />
              <span>{fill(copy.batchCombine, { n: files.length })}</span>
            </label>
            <label className="check">
              <input type="radio" name="batch" checked={separate} onChange={() => setSeparate(true)} />
              <span>{fill(copy.batchSeparate, { n: files.length })}</span>
            </label>
            {separate && (
              <p className="field-hint">{form.cropEnabled ? copy.batchHintCrop : copy.batchHint}</p>
            )}
          </fieldset>
        )}
        {job && (
          <div className="meta-grid" aria-label={copy.details}>
            {(sources.length || files.length) > 1 && (
              <div>
                <span>{copy.videos}</span>
                {sources.length || files.length}
              </div>
            )}
            <div>
              <span>{copy.duration}</span>
              {meta.duration ? formatTime(meta.duration) : "—"}
            </div>
            <div>
              <span>{copy.resolution}</span>
              {meta.width && meta.height ? `${meta.width}×${meta.height}` : "—"}
            </div>
            <div>
              <span>{copy.size}</span>
              {formatBytes(meta.size_bytes || files.reduce((sum, file) => sum + file.size, 0))}
            </div>
          </div>
        )}
      </section>

      <section aria-labelledby="step-2">
        <h3 className="step-title" id="step-2">
          {copy.step2}
        </h3>
        <div className="chips mode-switch" role="radiogroup" aria-label={messages.prompt.modeLabel}>
          <button
            type="button"
            role="radio"
            aria-checked={form.mode === "prompt"}
            className={`chip ${form.mode === "prompt" ? "active" : ""}`}
            onClick={() => chooseMode("prompt")}
          >
            {messages.prompt.modePrompt}
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={form.mode === "manual"}
            className={`chip ${form.mode === "manual" ? "active" : ""}`}
            onClick={() => chooseMode("manual")}
          >
            {messages.prompt.modeManual}
          </button>
        </div>
        {form.mode === "prompt" ? (
          <>
            <PromptPanel form={form} setForm={setForm} jobId={job?.id || null} understood={understood} setUnderstood={setUnderstood} />
            <details className="advanced all-settings">
              <summary>{messages.prompt.allSettings}</summary>
              <p className="field-hint">{messages.prompt.allSettingsHint}</p>
              <PresetBar form={form} setForm={setForm} />
              <HighlightConfigPanel
                form={form}
                setForm={setForm}
                sourceFps={sources[previewIndex]?.fps || sources[0]?.fps || meta.fps}
              />
            </details>
          </>
        ) : (
          <>
            <PresetBar form={form} setForm={setForm} />
            <HighlightConfigPanel
              form={form}
              setForm={setForm}
              sourceFps={sources[previewIndex]?.fps || sources[0]?.fps || meta.fps}
            />
          </>
        )}
        <NarratedVideoPanel form={form} setForm={setForm} />
      </section>

      {error && <Alert>{error}</Alert>}
      <div className="action-bar">
        <div className="btn-row">
          <button
            className="btn primary lg"
            disabled={!job || busy || Boolean(problem)}
            aria-busy={starting}
            aria-describedby={blocker ? "start-blocker" : undefined}
            onClick={analyze}
          >
            {starting && <Spinner />}
            {starting
              ? messages.config.starting
              : files.length > 1
                ? fill(separate ? messages.config.analyzeEach : messages.config.analyzeMany, { n: files.length })
                : messages.config.analyze}
          </button>
          {blocker && (
            <span className="muted" id="start-blocker">
              {blocker}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
