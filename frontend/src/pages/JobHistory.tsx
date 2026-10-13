import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, formatTime } from "../services/api";
import { messages } from "../i18n";
import { Alert, HistoryIcon, PageLoader, Spinner } from "../components/Feedback";
import type { Job } from "../types";

const copy = messages.history;

function statusTone(status: string) {
  if (status === "completed") return "ok";
  if (status === "failed") return "err";
  if (status === "cancelled") return "warn";
  if (["queued", "analyzing", "rendering", "planning"].includes(status)) return "run";
  return "";
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
    if (job.analysis_status === "completed") return `/jobs/${job.id}/review`;
    if (["queued", "analyzing", "rendering", "planning"].includes(job.status)) return `/jobs/${job.id}/analysis`;
    return "/";
  }

  async function remove(job: Job) {
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
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">{copy.file}</th>
              <th scope="col">{copy.original}</th>
              <th scope="col">{copy.highlight}</th>
              <th scope="col">{copy.style}</th>
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
                  {job.original_filename}
                </td>
                <td>{formatTime(job.metadata?.duration)}</td>
                <td>{formatTime(job.outputs?.[0]?.duration)}</td>
                <td>{job.style}</td>
                <td>
                  <span className={`status-pill ${statusTone(job.status)}`}>{job.status}</span>
                </td>
                <td>
                  <div className="row-actions">
                    <Link className="btn sm" to={href(job)}>
                      {copy.open}
                    </Link>
                    {job.outputs?.length ? (
                      <a className="btn sm" href={`/api/jobs/${job.id}/download`}>
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
                      {copy.delete}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {error && <Alert>{error}</Alert>}
    </div>
  );
}
