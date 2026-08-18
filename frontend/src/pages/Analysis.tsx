import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { io } from "socket.io-client";
import { api, formatTime } from "../services/api";
import { messages } from "../i18n";
import type { Job, JobSource } from "../types";

const LABELS: Record<string, string> = {
  metadata: "Video metadata",
  scenes: "Scene detection",
  audio: "Audio extraction",
  transcript: "Transcription",
  silence: "Silence detection",
  frames: "Keyframe extraction",
  visual: "Visual AI analysis",
  content: "Finding important moments",
  repetition: "Removing repetition",
  planning: "Planning highlight",
  rendering: "Rendering",
};

function mark(status: string) {
  if (status === "done") return { cls: "done", ch: "✓" };
  if (status === "running") return { cls: "run", ch: "●" };
  return { cls: "", ch: "○" };
}

function SourcePreview({ jobId, sources }: { jobId: string; sources: JobSource[] }) {
  const [current, setCurrent] = useState(sources[0]?.id || "v0");
  return (
    <div>
      <label className="field">
        {messages.review.source}
        <select value={current} onChange={(e) => setCurrent(e.target.value)}>
          {sources.map((source, i) => (
            <option key={source.id} value={source.id}>
              Video {i + 1}: {source.filename}
            </option>
          ))}
        </select>
      </label>
      <video className="preview" src={`/api/media/${jobId}/source/${current}`} controls />
    </div>
  );
}

export default function Analysis() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<Array<{ level: string; message: string }>>([]);
  const [openLog, setOpenLog] = useState(true);
  const queued = useRef(false);

  useEffect(() => {
    if (!id) return;
    const load = () =>
      api.job(id).then((j) => {
        setJob(j);
        if (j.status === "uploaded" && !queued.current) {
          queued.current = true;
          api.retry(id).then((next) => setJob(next)).catch(() => {
            queued.current = false;
          });
        }
        if (j.status === "planning" && j.analysis_status === "completed") {
          navigate(`/jobs/${id}/review`);
        }
        if (j.status === "completed") navigate(`/jobs/${id}/result`);
      });
    load();
    api.logs(id).then((r) => setLogs(r.logs || []));
    const socket = io("/", { transports: ["websocket", "polling"] });
    socket.on("job_progress", (payload: { job_id: string }) => {
      if (payload.job_id === id) load();
    });
    socket.on("job_log", (payload: { job_id: string; level: string; message: string }) => {
      if (payload.job_id === id) setLogs((cur) => [...cur, payload]);
    });
    socket.on("job_complete", (payload: { job_id: string; phase?: string }) => {
      if (payload.job_id === id) {
        if (payload.phase === "render") navigate(`/jobs/${id}/result`);
        else navigate(`/jobs/${id}/review`);
      }
    });
    const timer = setInterval(load, 2500);
    return () => {
      socket.close();
      clearInterval(timer);
    };
  }, [id, navigate]);

  if (!job || !id) return <p>Loading…</p>;
  const sources = job.sources || [];
  const previewSrc =
    sources.length > 1 ? `/api/media/${id}/source/${sources[0].id}` : `/api/media/${id}/source`;

  return (
    <div>
      {sources.length > 1 ? (
        <SourcePreview jobId={id} sources={sources} />
      ) : (
        <video className="preview" src={previewSrc} controls />
      )}
      <div className="panel">
        <h3>{messages.progress.title}</h3>
        <p>
          {messages.progress.overall}: {Math.round(job.progress)}% · {messages.progress.elapsed}:{" "}
          {formatTime(job.processing_seconds)} · {messages.progress.current}: {job.last_stage || job.status}
        </p>
        <ul className="stage-list">
          {(job.stages || []).map((stage) => {
            const m = mark(stage.status);
            return (
              <li key={stage.name}>
                <span className={`stage-mark ${m.cls}`}>{m.ch}</span>
                <div>
                  <strong>{LABELS[stage.name] || stage.name}</strong>
                  {stage.detail && <div style={{ color: "var(--muted)" }}>{stage.detail}</div>}
                </div>
              </li>
            );
          })}
        </ul>
        <div className="btn-row" style={{ marginTop: 12 }}>
          <button className="btn danger" onClick={() => api.cancel(id)}>
            {messages.progress.cancel}
          </button>
          {job.status === "failed" && (
            <button className="btn" onClick={() => api.retry(id)}>
              Retry
            </button>
          )}
        </div>
        {job.error && <p style={{ color: "var(--err)" }}>{job.error}</p>}
      </div>
      <details open={openLog} onToggle={(e) => setOpenLog((e.target as HTMLDetailsElement).open)}>
        <summary>{messages.progress.logs}</summary>
        <div className="log">
          {logs.map((line, i) => (
            <div key={i} className={line.level}>
              [{line.level}] {line.message}
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
