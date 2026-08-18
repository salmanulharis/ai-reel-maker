import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, formatTime } from "../services/api";
import type { Job } from "../types";

export default function JobHistory() {
  const [jobs, setJobs] = useState<Job[]>([]);

  const load = () => api.jobs().then((r) => setJobs(r.jobs || []));
  useEffect(() => {
    load();
  }, []);

  function href(job: Job) {
    if (job.status === "completed") return `/jobs/${job.id}/result`;
    if (job.analysis_status === "completed") return `/jobs/${job.id}/review`;
    if (["queued", "analyzing", "rendering", "planning"].includes(job.status)) return `/jobs/${job.id}/analysis`;
    return "/";
  }

  return (
    <div>
      <table className="table">
        <thead>
          <tr>
            <th>File</th>
            <th>Original</th>
            <th>Highlight</th>
            <th>Style</th>
            <th>Status</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td>{job.original_filename}</td>
              <td>{formatTime(job.metadata?.duration)}</td>
              <td>{formatTime(job.outputs?.[0]?.duration)}</td>
              <td>{job.style}</td>
              <td>{job.status}</td>
              <td>
                <Link to={href(job)}>Open</Link>{" "}
                {job.outputs?.length ? <a href={`/api/jobs/${job.id}/download`}>Download</a> : null}{" "}
                <button
                  className="btn"
                  onClick={async () => {
                    await api.deleteJob(job.id);
                    load();
                  }}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!jobs.length && <p style={{ color: "var(--muted)" }}>No jobs yet.</p>}
    </div>
  );
}
