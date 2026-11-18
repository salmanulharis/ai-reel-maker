import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, formatBytes, formatElapsed, formatTime } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, CheckIcon, InlineStatus, PageLoader, Spinner } from "../components/Feedback";
import { EditorExport } from "../components/EditorExport";
import { CoverPicker } from "../components/CoverPicker";
import { PostTextPanel } from "../components/PostTextPanel";
import { strength } from "../services/strength";
import type { Job } from "../types";

const copy = messages.result;

export default function Result() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [details, setDetails] = useState(false);
  const [folder, setFolder] = useState<"" | "busy" | "done">("");
  const [error, setError] = useState("");
  const player = useRef<HTMLVideoElement>(null);
  // With several shorts: one player per short, and which short the cover and text are for.
  const shortPlayers = useRef<Record<number, HTMLVideoElement | null>>({});
  const [activeShort, setActiveShort] = useState(0);

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
  const shorts = (job.outputs || [])
    .filter((item) => item.kind === "short")
    .map((item) => ({ ...item, number: Number(String(item.version_label).replace(/\D/g, "")) || 0 }))
    .filter((item) => item.number > 0)
    .sort((a, b) => a.number - b.number);
  const shortsMode = shorts.length > 0;
  const output = shortsMode ? undefined : job.outputs?.find((item) => item.kind !== "short") || job.outputs?.[0];
  const highlight = shortsMode ? shorts.reduce((sum, item) => sum + (item.duration || 0), 0) : output?.duration || 0;
  const current = shorts.find((item) => item.number === activeShort) || shorts[0];
  const subtitleChoice = (job.config as { subtitles?: string } | undefined)?.subtitles || "none";
  const hasSubtitles = subtitleChoice !== "none";
  const reduction = original && highlight ? `${Math.round((1 - highlight / original) * 100)}%` : "—";
  const clips = job.clips || [];
  const timing = job.timing;
  const multi = (job.sources || []).length > 1;

  return (
    <div>
      <h3 className="result-title">
        <span className="check-icon">
          <CheckIcon size={16} />
        </span>
        {shortsMode ? fill(copy.readyShorts, { n: shorts.length }) : copy.ready}
      </h3>
      {shortsMode ? (
        <div className="shorts-grid">
          {shorts.map((item) => (
            <article key={item.number} className="short-card" aria-label={fill(copy.shortLabel, { n: item.number })}>
              <video
                ref={(el) => {
                  shortPlayers.current[item.number] = el;
                }}
                className="preview"
                src={`/api/media/${id}/output?short=${item.number}`}
                preload="metadata"
                controls
              />
              <div className="short-card-foot">
                <strong>{fill(copy.shortLabel, { n: item.number })}</strong>
                <span className="muted">
                  {formatTime(item.duration)} · {formatBytes(item.size_bytes)}
                </span>
                {(() => {
                  const value = strength(
                    clips.filter((clip) => (clip.short_index || 0) === item.number - 1),
                    job.min_duration,
                    job.max_duration,
                  );
                  return value ? (
                    <span
                      className="tag accent"
                      title={fill(messages.review.strengthHint, { opening: value.opening, moments: value.moments, length: value.length })}
                    >
                      {fill(messages.review.strengthTag, { n: value.score })}
                    </span>
                  ) : null;
                })()}
                {hasSubtitles && (
                  <a className="btn sm ghost" href={`/api/media/${id}/srt?short=${item.number}`} download>
                    {copy.subtitlesFile}
                  </a>
                )}
                <a
                  className="btn sm primary"
                  href={`/api/jobs/${id}/download?short=${item.number}`}
                  download
                  aria-label={fill(copy.downloadShort, { n: item.number })}
                >
                  {copy.downloadOne}
                </a>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <video ref={player} className="preview" src={`/api/media/${id}/output`} controls />
      )}
      <div className="btn-row result-actions">
        {!shortsMode && (
          <a className="btn primary lg" href={`/api/jobs/${id}/download`} download>
            {copy.download}
          </a>
        )}
        <EditorExport jobId={id} disabled={clips.every((c) => c.kept === false)} />
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

      {shortsMode && current && (
        <div className="field short-picker">
          <label htmlFor="active-short">{copy.toolsFor}</label>
          <select id="active-short" value={current.number} onChange={(e) => setActiveShort(Number(e.target.value))}>
            {shorts.map((item) => (
              <option key={item.number} value={item.number}>
                {fill(copy.shortLabel, { n: item.number })}
              </option>
            ))}
          </select>
        </div>
      )}
      {shortsMode && current ? (
        <CoverPicker
          key={`cover-${current.number}`}
          jobId={id}
          short={current.number}
          clips={clips.filter((clip) => (clip.short_index || 0) === current.number - 1)}
          getPlayerTime={() => shortPlayers.current[current.number]?.currentTime}
        />
      ) : (
        output && <CoverPicker jobId={id} clips={clips} getPlayerTime={() => player.current?.currentTime} />
      )}
      <PostTextPanel jobId={id} short={shortsMode ? current?.number : undefined} />

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
        <div title={timing ? fill(copy.processingSplit, { analysis: formatElapsed(timing.analysis_seconds), render: formatElapsed(timing.render_seconds) }) : undefined}>
          <span>{copy.processing}</span>
          {formatElapsed(timing?.total_seconds ?? job.processing_seconds)}
          {timing && timing.render_seconds > 0 && (
            <small className="stat-note">
              {fill(copy.processingSplit, {
                analysis: formatElapsed(timing.analysis_seconds),
                render: formatElapsed(timing.render_seconds),
              })}
            </small>
          )}
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
