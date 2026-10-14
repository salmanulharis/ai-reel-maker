import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, formatTime } from "../services/api";
import { messages } from "../i18n";
import { Alert, HistoryIcon, PageLoader, Spinner } from "../components/Feedback";
import type { Job } from "../types";

const copy = messages.history;
const ACTIVE = ["queued", "analyzing", "rendering"];

function statusTone(status: string) {
  if (status === "completed") return "ok";
  if (status === "failed") return "err";
  if (status === "cancelled") return "warn";
  if (status === "planning") return "accent";
  if (ACTIVE.includes(status)) return "run";
  return "";
}

function statusLabel(status: string) {
  return copy.statuses[status as keyof typeof copy.statuses] || status;
}

function styleLabel(style: string) {
  return messages.styles[style as keyof typeof messages.styles] || style || "—";
}

function formatDate(value?: string) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export default function JobHistory() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [deleting, setDeleting] = useState("");
  const [error, setError] = useState("");

  const load = () =>
    api
      .jobs()
      .then((r) => setJobs(r.jobs || []))
      .catch((err) => setError((err as Error).message))
      .finally(() => setLoaded(true));

  useEffect(() => {
    load();
  }, []);

  function href(job: Job) {
    if (job.status === "completed") return `/jobs/${job.id}/result`;
    if (job.analysis_status === "completed" && !ACTIVE.includes(job.status)) return `/jobs/${job.id}/review`;
    // Failed and stopped jobs open the progress page, where they can be started again.
    if ([...ACTIVE, "planning", "failed", "cancelled"].includes(job.status)) return `/jobs/${job.id}/analysis`;
    return "/";
  }

  async function remove(job: Job) {
    if (!window.confirm(copy.confirmDelete.replace("{name}", job.original_filename))) return;
    setDeleting(job.id);
    setError("");
    try {
      await api.deleteJob(job.id);
      await load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setDeleting("");
    }
  }

  if (!loaded) return <PageLoader />;

  if (!jobs.length) {
    return (
      <>
        {error && <Alert>{error}</Alert>}
        <div className="empty-state">
          <HistoryIcon size={28} />
          <h3>{copy.empty}</h3>
          <p>{copy.emptyHint}</p>
          <Link className="btn primary" to="/">
            {copy.start}
          </Link>
        </div>
      </>
    );
  }

  return (
    <div>
      {error && <Alert>{error}</Alert>}
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">{copy.file}</th>
              <th scope="col" className="hide-sm">
                {copy.created}
              </th>
              <th scope="col">{copy.original}</th>
              <th scope="col">{copy.highlight}</th>
              <th scope="col" className="hide-sm">
                {copy.style}
              </th>
              <th scope="col">{copy.status}</th>
              <th scope="col">
                <span className="sr-only">{copy.actions}</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id}>
                <td className="file-cell" title={job.original_filename}>
                  <Link to={href(job)}>{job.original_filename}</Link>
                </td>
                <td className="hide-sm nowrap">{formatDate(job.created_at)}</td>
                <td>{job.metadata?.duration ? formatTime(job.metadata.duration) : "—"}</td>
                <td>{job.outputs?.[0]?.duration ? formatTime(job.outputs[0].duration) : "—"}</td>
                <td className="hide-sm">{styleLabel(job.style)}</td>
                <td>
                  <span className={`status-pill ${statusTone(job.status)}`}>{statusLabel(job.status)}</span>
                </td>
                <td>
                  <div className="row-actions">
                    <Link className="btn sm" to={href(job)} aria-label={`${copy.open} ${job.original_filename}`}>
                      {copy.open}
                    </Link>
                    {job.outputs?.length ? (
                      <a
                        className="btn sm"
                        href={`/api/jobs/${job.id}/download`}
                        download
                        aria-label={`${copy.download} ${job.original_filename}`}
                      >
                        {copy.download}
                      </a>
                    ) : null}
                    <button
                      className="btn sm danger"
                      disabled={deleting !== ""}
                      aria-busy={deleting === job.id}
                      aria-label={`${copy.delete} ${job.original_filename}`}
                      onClick={() => remove(job)}
                    >
                      {deleting === job.id && <Spinner />}
                      {deleting === job.id ? copy.deleting : copy.delete}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
