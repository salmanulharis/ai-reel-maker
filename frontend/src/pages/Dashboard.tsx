import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { UploadZone } from "../components/UploadZone";
import { HighlightConfigPanel, defaultForm } from "../components/HighlightConfig";
import { VideoCropPreview } from "../components/VideoCrop";
import { Alert, InlineStatus, Spinner } from "../components/Feedback";
import { api, formatBytes, formatTime } from "../services/api";
import { messages } from "../i18n";
import type { HighlightForm, Job } from "../types";

const MAX_FILES = 8;

export default function Dashboard() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [preview, setPreview] = useState<string>("");
  const [job, setJob] = useState<Job | null>(null);
  const [form, setForm] = useState<HighlightForm>(defaultForm());
  const [busy, setBusy] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [aiNotice, setAiNotice] = useState(false);

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
      const created = await api.createJob(next, form);
      setJob(created);
    } catch (err) {
      setError((err as Error).message);
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
      const response = await fetch("/api/media/sample");
      if (!response.ok) throw new Error("Sample video is not available");
      const blob = await response.blob();
      addFiles([new File([blob], "sample.mp4", { type: "video/mp4" })]);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  async function analyze() {
    if (!job) return;
    setBusy(true);
    setStarting(true);
    setError("");
    try {
      await api.analyze(job.id, form);
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

  return (
    <div>
      {aiNotice && (
        <div className="notice" role="status">
          <span>
            {messages.localAi.notice}{" "}
            <Link to="/settings">{messages.localAi.action}</Link>
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
      <UploadZone onFiles={addFiles} files={files} onRemove={removeFile} />
      <div className="btn-row" style={{ marginTop: 10, marginBottom: 12 }}>
        <button className="btn ghost" type="button" disabled={busy || files.length >= MAX_FILES} onClick={useSample}>
          {messages.upload.sample}
        </button>
        {busy && !starting && <InlineStatus busy>{messages.common.preparing}</InlineStatus>}
      </div>
      {preview && (
        <>
          {files.length > 1 && (
            <label className="field">
              {messages.config.cropSource}
              <select value={String(previewIndex)} onChange={(e) => setPreviewIndex(Number(e.target.value))}>
                {files.map((file, i) => (
                  <option key={`${file.name}-${i}`} value={i}>
                    Video {i + 1}: {file.name}
                  </option>
                ))}
              </select>
            </label>
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
      {job && (
        <div className="meta-grid">
          <div>
            <span>Filename</span>
            {job.original_filename}
          </div>
          <div>
            <span>Videos</span>
            {sources.length || files.length}
          </div>
          <div>
            <span>File size</span>
            {formatBytes(meta.size_bytes || files.reduce((sum, file) => sum + file.size, 0))}
          </div>
          <div>
            <span>Duration</span>
            {formatTime(meta.duration)}
          </div>
          <div>
            <span>Resolution</span>
            {meta.width && meta.height ? `${meta.width}×${meta.height}` : "—"}
          </div>
          <div>
            <span>FPS</span>
            {meta.fps || "—"}
          </div>
          <div>
            <span>Video codec</span>
            {meta.video_codec || "—"}
          </div>
          <div>
            <span>Audio codec</span>
            {meta.audio_codec || "—"}
          </div>
        </div>
      )}
      <HighlightConfigPanel form={form} setForm={setForm} />
      {error && <Alert>{error}</Alert>}
      <div className="action-bar">
        <div className="btn-row">
          <button className="btn primary" disabled={!job || busy} aria-busy={starting} onClick={analyze}>
            {starting && <Spinner />}
            {files.length > 1 ? messages.config.analyzeMany : messages.config.analyze}
          </button>
        </div>
      </div>
    </div>
  );
}
