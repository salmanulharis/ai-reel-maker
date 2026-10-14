import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { io } from "socket.io-client";
import { api, formatElapsed } from "../services/api";
import { useElapsed } from "../hooks/useElapsed";
import { fill, messages } from "../i18n";
import { Alert, CheckIcon, PageLoader, Spinner } from "../components/Feedback";
import type { Job, JobSource } from "../types";

const copy = messages.progress;
const LABELS: Record<string, string> = copy.stages;

function StageMark({ status }: { status: string }) {
  if (status === "done") {
    return (
      <span className="stage-mark done">
        <CheckIcon size={12} />
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className="stage-mark run">
        <Spinner />
      </span>
    );
  }
  if (status === "error" || status === "failed") return <span className="stage-mark error">✕</span>;
  if (status === "skipped" || status === "cancelled") return <span className="stage-mark skipped">–</span>;
  return <span className="stage-mark" />;
}

const STATUS_TEXT: Record<string, string> = {
  done: "done",
  running: "in progress",
  error: "failed",
  failed: "failed",
  skipped: "skipped",
  cancelled: "stopped",
};

function SourcePreview({ jobId, sources }: { jobId: string; sources: JobSource[] }) {
  const [current, setCurrent] = useState(sources[0]?.id || "v0");
  return (
    <div>
      <div className="field">
        <label htmlFor="source-preview">{messages.review.source}</label>
        <select id="source-preview" value={current} onChange={(e) => setCurrent(e.target.value)}>
          {sources.map((source, i) => (
            <option key={source.id} value={source.id}>
              {fill(messages.upload.video, { n: i + 1 })}: {source.filename}
            </option>
          ))}
        </select>
      </div>
      <video className="preview" src={`/api/media/${jobId}/source/${current}`} controls />
    </div>
  );
}

export default function Analysis() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<Array<{ level: string; message: string }>>([]);
  const [openLog, setOpenLog] = useState(false);
  const [action, setAction] = useState<"" | "cancel" | "retry">("");
  const [actionError, setActionError] = useState("");
  const elapsed = useElapsed(job?.timing);
  const queued = useRef(false);
  const logRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  // Follow new log lines unless the user has scrolled up to read earlier ones.
  useEffect(() => {
    const box = logRef.current;
    if (box && stickToBottom.current) box.scrollTop = box.scrollHeight;
  }, [logs, openLog]);

  async function run(kind: "cancel" | "retry", call: () => Promise<Job>) {
    setAction(kind);
    setActionError("");
    try {
      setJob(await call());
    } catch (err) {
      setActionError((err as Error).message);
    } finally {
      setAction("");
    }
  }

  useEffect(() => {
    if (!id) return;
    const load = () =>
      api
        .job(id)
        .then((j) => {
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
        })
        .catch((err) => setActionError((err as Error).message));
    load();
    api.logs(id).then((r) => setLogs(r.logs || [])).catch(() => undefined);
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

  if (!job || !id) return actionError ? <Alert>{actionError}</Alert> : <PageLoader />;
  const sources = job.sources || [];
  const previewSrc =
    sources.length > 1 ? `/api/media/${id}/source/${sources[0].id}` : `/api/media/${id}/source`;
  const percent = Math.max(0, Math.min(100, Math.round(job.progress || 0)));
  const failed = job.status === "failed";
  const cancelled = job.status === "cancelled";
  const stopped = failed || cancelled;
  const rendering = job.status === "rendering" || job.render_status === "running";
  const waiting = job.status === "queued" && !job.last_stage;
  const current = waiting ? copy.waiting : job.last_stage ? LABELS[job.last_stage] || job.last_stage : "";

  return (
    <div>
      <div className="panel">
        <h3>{rendering ? copy.rendering : copy.title}</h3>
        {!stopped && (
          <>
            <p className="progress-summary" aria-live="polite">
              <strong>{percent}%</strong>
              <span>
                {copy.elapsed}: <span className="tabular">{formatElapsed(elapsed.attempt, elapsed.running)}</span>
              </span>
              {elapsed.total - elapsed.attempt >= 1 && (
                <span>
                  {copy.totalElapsed}: <span className="tabular">{formatElapsed(elapsed.total, elapsed.running)}</span>
                </span>
              )}
              {current && (
                <span>
                  {copy.current}: {current}
                </span>
              )}
            </p>
            <div
              className="progress-line lg"
              role="progressbar"
              aria-label={copy.overall}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={percent}
            >
              <span style={{ width: `${percent}%` }} />
            </div>
          </>
        )}
        {failed && (
          <Alert>
            <strong>
              {copy.failed}
              {elapsed.attempt > 0 ? ` ${fill(copy.endedAfter, { time: formatElapsed(elapsed.attempt) })}` : ""}.
            </strong>{" "}
            {copy.failedHint}
            {job.error && <span className="alert-detail">{job.error}</span>}
          </Alert>
        )}
        {cancelled && !failed && (
          <Alert kind="info">
            <strong>
              {copy.cancelled}
              {!elapsed.running && elapsed.attempt > 0 ? ` ${fill(copy.endedAfter, { time: formatElapsed(elapsed.attempt) })}` : ""}.
            </strong>{" "}
            {copy.cancelledHint}
          </Alert>
        )}
        <ul className="stage-list">
          {(job.stages || []).map((stage) => {
            const running = stage.status === "running" && Boolean(stage.started_at);
            const seconds = elapsed.stageSeconds(stage.seconds, running ? stage.started_at : null);
            return (
              <li key={stage.name} className={stage.status || "pending"}>
                <StageMark status={stage.status} />
                <div className="stage-body">
                  <strong>{LABELS[stage.name] || stage.name}</strong>
                  <span className="sr-only"> — {STATUS_TEXT[stage.status] || "waiting"}</span>
                  {stage.detail && <div className="stage-detail">{stage.detail}</div>}
                </div>
                {seconds >= 0.05 && (
                  <span className="stage-time tabular" title={fill(copy.stageTime, { time: formatElapsed(seconds) })}>
                    {formatElapsed(seconds, running)}
                  </span>
                )}
              </li>
            );
          })}
        </ul>
        <div className="btn-row panel-actions">
          {stopped ? (
            <button
              className="btn primary"
              disabled={action !== ""}
              aria-busy={action === "retry"}
              onClick={() => run("retry", () => api.retry(id))}
            >
              {action === "retry" && <Spinner />}
              {messages.common.retry}
            </button>
          ) : (
            <button
              className="btn danger"
              disabled={action !== ""}
              aria-busy={action === "cancel"}
              onClick={() => {
                if (!window.confirm(copy.confirmCancel)) return;
                void run("cancel", () => api.cancel(id));
              }}
            >
              {action === "cancel" && <Spinner />}
              {action === "cancel" ? messages.common.cancelling : copy.cancel}
            </button>
          )}
        </div>
        {actionError && <Alert>{actionError}</Alert>}
      </div>

      {sources.length > 1 ? (
        <SourcePreview jobId={id} sources={sources} />
      ) : (
        <video className="preview" src={previewSrc} controls />
      )}

      <details
        className="log-details"
        open={openLog}
        onToggle={(e) => setOpenLog((e.target as HTMLDetailsElement).open)}
      >
        <summary>{copy.logs}</summary>
        <p className="field-hint">{copy.logsHint}</p>
        <div
          className="log"
          ref={logRef}
          tabIndex={0}
          aria-label={copy.logs}
          onScroll={(e) => {
            const box = e.currentTarget;
            stickToBottom.current = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
          }}
        >
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
