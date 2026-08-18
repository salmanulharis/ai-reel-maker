import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, formatBytes, formatTime } from "../services/api";
import { messages } from "../i18n";
import type { Job } from "../types";

export default function Result() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [analysis, setAnalysis] = useState(false);

  useEffect(() => {
    if (!id) return;
    api.job(id).then(setJob);
  }, [id]);

  if (!job || !id) return <p>Loading…</p>;
  const original = job.metadata?.duration || 0;
  const output = job.outputs?.[0];
  const highlight = output?.duration || 0;
  const reduction = original ? ((1 - highlight / original) * 100).toFixed(1) : "—";

  return (
    <div>
      <h3>{messages.result.ready}</h3>
      <video className="preview" src={`/api/media/${id}/output`} controls />
      <div className="meta-grid">
        <div>
          <span>Original</span>
          {formatTime(original)}
        </div>
        <div>
          <span>Highlight</span>
          {formatTime(highlight)}
        </div>
        <div>
          <span>Reduction</span>
          {reduction}%
        </div>
        <div>
          <span>Clips</span>
          {job.clips?.length || 0}
        </div>
        <div>
          <span>Processing</span>
          {formatTime(job.processing_seconds)}
        </div>
        <div>
          <span>Output</span>
          {output ? `${output.width}×${output.height} ${output.codec}` : "—"}
        </div>
        <div>
          <span>File size</span>
          {formatBytes(output?.size_bytes)}
        </div>
      </div>
      <div className="btn-row">
        <a className="btn primary" href={`/api/jobs/${id}/download`}>
          {messages.result.download}
        </a>
        <button
          className="btn"
          onClick={() => fetch(`/api/jobs/${id}/open-folder`, { method: "POST" })}
        >
          {messages.result.openFolder}
        </button>
        <button className="btn" onClick={() => navigate(`/jobs/${id}/review`)}>
          {messages.result.edit}
        </button>
        <button className="btn" onClick={() => setAnalysis((v) => !v)}>
          {messages.result.analysis}
        </button>
        <button className="btn" onClick={() => navigate("/")}>
          {messages.result.another}
        </button>
      </div>
      <h3 style={{ marginTop: 24 }}>{messages.result.compare}</h3>
      {(job.sources || []).length > 1 && (
        <div className="source-used">
          <p>{messages.result.sources}</p>
          <ul>
            {(job.sources || []).map((source) => {
              const used = (job.clips || []).some((clip) => clip.source_id === source.id);
              return (
                <li key={source.id}>
                  {source.filename}
                  {used ? "" : " (not used)"}
                </li>
              );
            })}
          </ul>
        </div>
      )}
      <div className="compare">
        <div>
          <p>{(job.sources || []).length > 1 ? messages.result.sources : "Original"}</p>
          <video src={`/api/media/${id}/source`} controls />
        </div>
        <div>
          <p>Highlight</p>
          <video src={`/api/media/${id}/output`} controls />
        </div>
      </div>
      {analysis && (
        <pre className="log" style={{ marginTop: 16 }}>
          {JSON.stringify(job.clips, null, 2)}
        </pre>
      )}
    </div>
  );
}
