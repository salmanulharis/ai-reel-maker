import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, formatTime } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, FilmIcon, InlineStatus, PageLoader, Spinner } from "../components/Feedback";
import type { Clip, Job, JobSource } from "../types";

const copy = messages.review;
const REGEN = Object.entries(copy.regen);

function sourceLabel(clip: Clip, fallback = "Video 1") {
  return clip.source_name || clip.source_id || fallback;
}

export default function Review() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [duration, setDuration] = useState(0);
  const [pending, setPending] = useState("");
  const [error, setError] = useState("");

  async function refresh() {
    if (!id) return;
    const data = await api.job(id);
    setJob(data);
    setClips(data.clips || []);
    setDuration(data.metadata?.duration || 0);
    if (data.status === "completed") navigate(`/jobs/${id}/result`);
    if (["queued", "analyzing", "rendering"].includes(data.status)) {
      navigate(`/jobs/${id}/analysis`);
    }
  }

  useEffect(() => {
    refresh().catch((err) => setError((err as Error).message));
  }, [id]);

  /** Show the change at once and save it in the background; say so if saving fails. */
  function save(copyOfClips: Clip[]) {
    setClips(copyOfClips);
    if (!id) return;
    api
      .clips(id, copyOfClips)
      .then(() => setError((current) => (current === messages.errors.save ? "" : current)))
      .catch(() => setError(messages.errors.save));
  }

  function patch(index: number, next: Partial<Clip>) {
    save(clips.map((c, i) => (i === index ? { ...c, ...next } : c)));
  }

  function move(index: number, direction: -1 | 1) {
    const next = index + direction;
    if (next < 0 || next >= clips.length) return;
    const copyOfClips = [...clips];
    [copyOfClips[index], copyOfClips[next]] = [copyOfClips[next], copyOfClips[index]];
    save(copyOfClips);
  }

  async function generate() {
    if (!id) return;
    setPending("render");
    setError("");
    try {
      await api.render(id, clips.filter((c) => c.kept !== false));
      navigate(`/jobs/${id}/analysis`);
    } catch (err) {
      setError((err as Error).message);
      setPending("");
    }
  }

  async function regenerate(preset: string) {
    if (!id) return;
    setPending(preset);
    setError("");
    try {
      const result = await api.regenerate(id, preset);
      if (result.job) setJob(result.job);
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPending("");
    }
  }

  if (!job || !id) return error ? <Alert>{error}</Alert> : <PageLoader />;
  const kept = clips.filter((c) => c.kept !== false);
  const estimated = kept.reduce((s, c) => s + (c.end - c.start), 0);
  const over = estimated > job.max_duration;
  const sources: JobSource[] =
    job.sources && job.sources.length
      ? job.sources
      : [{ id: "v0", filename: job.original_filename, duration }];
  const multi = sources.length > 1;

  const adjust = (
    <div className="panel">
      <h3 className="sub-title">{copy.adjust}</h3>
      <div className="chips" role="group" aria-label={copy.adjust}>
        {REGEN.map(([key, label]) => (
          <button
            key={key}
            type="button"
            className="chip"
            disabled={pending !== ""}
            aria-busy={pending === key}
            onClick={() => regenerate(key)}
          >
            {pending === key && <Spinner />}
            {label}
          </button>
        ))}
      </div>
      <p className="field-hint">{copy.adjustHint}</p>
      {pending && pending !== "render" && <InlineStatus busy>{copy.regenerating}</InlineStatus>}
    </div>
  );

  if (!clips.length) {
    return (
      <div>
        <div className="empty-state">
          <FilmIcon size={28} />
          <h3>{copy.empty}</h3>
          <p>{copy.emptyHint}</p>
          <Link className="btn" to="/">
            {messages.result.another}
          </Link>
        </div>
        {adjust}
        {error && <Alert>{error}</Alert>}
      </div>
    );
  }

  return (
    <div>
      <div className="summary-bar">
        <span>
          {copy.target} <strong>{job.min_duration === job.max_duration ? `${job.max_duration}s` : `${job.min_duration}–${job.max_duration}s`}</strong>
        </span>
        <span className={over ? "over" : ""}>
          {copy.estimated} <strong>{estimated.toFixed(1)}s</strong>
        </span>
        <span>
          {copy.clips} <strong>{kept.length}/{clips.length}</strong>
        </span>
        {multi && (
          <span>
            {copy.sources} <strong>{sources.length}</strong>
          </span>
        )}
      </div>
      {over && <p className="field-hint warn">{copy.overTarget}</p>}

      {sources.map((source) => {
        const sourceClips = clips.filter((clip) => (clip.source_id || sources[0].id) === source.id);
        const sourceDuration = source.duration || duration;
        return (
          <div key={source.id}>
            <p className="section-label">{copy.sourceTimeline.replace("{name}", source.filename || source.id)}</p>
            <div className="timeline" aria-hidden="true">
              {sourceClips.map((clip, i) => (
                <div
                  key={`${source.id}-${i}`}
                  className="clip-block"
                  title={`${formatTime(clip.start)}–${formatTime(clip.end)} (${(clip.end - clip.start).toFixed(1)}s)`}
                  style={{
                    left: `${sourceDuration ? (clip.start / sourceDuration) * 100 : 0}%`,
                    width: `${sourceDuration ? ((clip.end - clip.start) / sourceDuration) * 100 : 0}%`,
                    opacity: clip.kept === false ? 0.25 : 0.9,
                  }}
                />
              ))}
            </div>
          </div>
        );
      })}
      <p className="section-label">{copy.highlightOrder}</p>
      <div className="sequence-timeline" aria-hidden="true">
        {kept.map((clip, i) => {
          const len = Math.max(0, clip.end - clip.start);
          return (
            <div
              key={i}
              className="sequence-block"
              title={`${sourceLabel(clip)} · ${clip.duration_reason || `${len.toFixed(1)}s`}`}
              style={{ flexGrow: Math.max(len, 0.4), flexBasis: 0 }}
            >
              {len.toFixed(1)}s
              {multi && <small>{sourceLabel(clip)}</small>}
            </div>
          );
        })}
      </div>

      {clips.map((clip, i) => {
        const removed = clip.kept === false;
        const label = fill(copy.clipNumber, { n: i + 1 });
        return (
          <article className={`clip-card ${removed ? "removed" : ""}`} key={i} aria-label={label}>
            <header>
              <strong>
                {label} · {formatTime(clip.start)}–{formatTime(clip.end)} · {(clip.end - clip.start).toFixed(1)}s
              </strong>
              <span className="tag-row">
                {removed && <span className="tag">{copy.removed}</span>}
                <span className="score" title={copy.scoreHint}>
                  {fill(copy.score, { n: Math.round((clip.score || 0) * 100) })}
                </span>
              </span>
            </header>
            {multi && (
              <p className="source-badge">
                {copy.source}: {sourceLabel(clip)}
              </p>
            )}
            {clip.reason && <p>{clip.reason}</p>}
            {clip.duration_reason && (
              <p className="duration-reason">
                <span>{copy.whyLength}</span>
                {clip.duration_reason}
              </p>
            )}
            {clip.transcript && <p className="clip-transcript">“{clip.transcript}”</p>}
            <div className="grid-2 clip-times">
              <div className="field">
                <label htmlFor={`clip-${i}-start`}>{copy.start}</label>
                <input
                  id={`clip-${i}-start`}
                  type="number"
                  step="0.1"
                  min={0}
                  value={clip.start}
                  onChange={(e) => patch(i, { start: Number(e.target.value) })}
                />
              </div>
              <div className="field">
                <label htmlFor={`clip-${i}-end`}>{copy.end}</label>
                <input
                  id={`clip-${i}-end`}
                  type="number"
                  step="0.1"
                  min={0}
                  value={clip.end}
                  onChange={(e) => patch(i, { end: Number(e.target.value) })}
                />
              </div>
            </div>
            <div className="btn-row clip-actions">
              <button
                className={`btn sm ${removed ? "primary" : ""}`}
                aria-label={`${removed ? copy.keep : copy.remove} ${label}`}
                onClick={() => patch(i, { kept: removed })}
              >
                {removed ? copy.keep : copy.remove}
              </button>
              <button className="btn sm" disabled={removed} onClick={() => patch(i, { start: Math.max(0, clip.start - 0.5) })}>
                {copy.extend}
              </button>
              <button
                className="btn sm"
                disabled={removed}
                onClick={() => patch(i, { end: Math.max(clip.start + 0.4, clip.end - 0.4) })}
              >
                {copy.trim}
              </button>
              <span className="btn-row push-right">
                <button
                  className="btn sm ghost"
                  disabled={i === 0}
                  aria-label={`${copy.moveUp}: ${label}`}
                  onClick={() => move(i, -1)}
                >
                  ↑ {copy.moveUp}
                </button>
                <button
                  className="btn sm ghost"
                  disabled={i === clips.length - 1}
                  aria-label={`${copy.moveDown}: ${label}`}
                  onClick={() => move(i, 1)}
                >
                  ↓ {copy.moveDown}
                </button>
              </span>
            </div>
          </article>
        );
      })}

      {adjust}
      {error && <Alert>{error}</Alert>}
      <div className="action-bar">
        <div className="btn-row">
          <button
            className="btn primary lg"
            disabled={pending !== "" || kept.length === 0}
            aria-busy={pending === "render"}
            onClick={generate}
          >
            {pending === "render" && <Spinner />}
            {pending === "render" ? copy.generating : copy.generate}
          </button>
          <span className="muted">
            {kept.length}/{clips.length} · {estimated.toFixed(1)}s
          </span>
        </div>
      </div>
    </div>
  );
}
