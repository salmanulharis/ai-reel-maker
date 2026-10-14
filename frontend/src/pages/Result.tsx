import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, formatBytes, formatTime } from "../services/api";
import { messages } from "../i18n";
import { Alert, CheckIcon, InlineStatus, PageLoader, Spinner } from "../components/Feedback";
import type { Job } from "../types";

const copy = messages.result;

export default function Result() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [details, setDetails] = useState(false);
  const [folder, setFolder] = useState<"" | "busy" | "done">("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!id) return;
    api
      .job(id)
      .then(setJob)
      .catch((err) => setError((err as Error).message));
  }, [id]);

  async function openFolder() {
    if (!id) return;
    setFolder("busy");
    setError("");
    try {
      await api.openFolder(id);
      setFolder("done");
      setTimeout(() => setFolder(""), 2500);
    } catch (err) {
      setError((err as Error).message);
      setFolder("");
    }
  }

  if (!job || !id) return error ? <Alert>{error}</Alert> : <PageLoader />;
  const original = job.metadata?.duration || 0;
  const output = job.outputs?.[0];
  const highlight = output?.duration || 0;
  const reduction = original && highlight ? `${Math.round((1 - highlight / original) * 100)}%` : "—";
  const clips = job.clips || [];
  const multi = (job.sources || []).length > 1;

  return (
    <div>
      <h3 className="result-title">
        <span className="check-icon">
          <CheckIcon size={16} />
        </span>
        {copy.ready}
      </h3>
      <video className="preview" src={`/api/media/${id}/output`} controls />
      <div className="btn-row result-actions">
        <a className="btn primary lg" href={`/api/jobs/${id}/download`} download>
          {copy.download}
        </a>
        <button className="btn" disabled={folder === "busy"} aria-busy={folder === "busy"} onClick={openFolder}>
          {folder === "busy" && <Spinner />}
          {copy.openFolder}
        </button>
        <button className="btn" onClick={() => navigate(`/jobs/${id}/review`)}>
          {copy.edit}
        </button>
        <button className="btn ghost" onClick={() => navigate("/")}>
          {copy.another}
        </button>
        {folder === "done" && <InlineStatus ok>{copy.openedFolder}</InlineStatus>}
      </div>
      {error && <Alert>{error}</Alert>}

      <div className="meta-grid">
        <div>
          <span>{copy.original}</span>
          {formatTime(original)}
        </div>
        <div>
          <span>{copy.highlight}</span>
          {formatTime(highlight)}
        </div>
        <div>
          <span>{copy.shorter}</span>
          {reduction}
        </div>
        <div>
          <span>{copy.clips}</span>
          {clips.length}
        </div>
        <div>
          <span>{copy.processing}</span>
          {formatTime(job.processing_seconds)}
        </div>
        <div>
          <span>{copy.output}</span>
          {output ? `${output.width}×${output.height}` : "—"}
        </div>
        <div>
          <span>{copy.size}</span>
          {formatBytes(output?.size_bytes)}
        </div>
      </div>

      {clips.length > 0 && (
        <>
          <button
            className="btn"
            aria-expanded={details}
            aria-controls="clip-details"
            onClick={() => setDetails((v) => !v)}
          >
            {details ? copy.hideAnalysis : copy.analysis}
          </button>
          {details && (
            <div className="table-wrap reveal" id="clip-details">
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col">#</th>
                    <th scope="col">{copy.clipTime}</th>
                    {multi && <th scope="col">{messages.review.source}</th>}
                    <th scope="col">{copy.clipReason}</th>
                    <th scope="col">{copy.clipScore}</th>
                  </tr>
                </thead>
                <tbody>
                  {clips.map((clip, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td className="nowrap">
                        {formatTime(clip.start)}–{formatTime(clip.end)} ({(clip.end - clip.start).toFixed(1)}s)
                      </td>
                      {multi && <td>{clip.source_name || clip.source_id}</td>}
                      <td>{clip.reason || "—"}</td>
                      <td>{Math.round((clip.score || 0) * 100)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      <h3 className="subheading">{copy.compare}</h3>
      {multi && (
        <div className="source-used">
          <p>{copy.sources}</p>
          <ul>
            {(job.sources || []).map((source) => {
              const used = clips.some((clip) => clip.source_id === source.id);
              return (
                <li key={source.id}>
                  {source.filename}
                  {used ? "" : ` (${copy.notUsed})`}
                </li>
              );
            })}
          </ul>
        </div>
      )}
      <div className="compare">
        <div>
          <p>{multi ? copy.sources : copy.original}</p>
          <video src={`/api/media/${id}/source`} controls preload="metadata" />
        </div>
        <div>
          <p>{copy.highlight}</p>
          <video src={`/api/media/${id}/output`} controls preload="metadata" />
        </div>
      </div>
    </div>
  );
}
