import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, formatTime } from "../services/api";
import { messages } from "../i18n";
import type { Clip, Job, JobSource } from "../types";

const REGEN = [
  ["more_concise", "More concise"],
  ["more_exciting", "More exciting"],
  ["more_informative", "More informative"],
  ["more_visual", "More visual"],
  ["less_repetitive", "Less repetitive"],
  ["focus_on_speech", "Focus on speech"],
  ["focus_on_action", "Focus on action"],
];

function sourceLabel(clip: Clip, fallback = "Video 1") {
  return clip.source_name || clip.source_id || fallback;
}

export default function Review() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [duration, setDuration] = useState(0);

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
    refresh();
  }, [id]);

  function patch(index: number, next: Partial<Clip>) {
    const copy = clips.map((c, i) => (i === index ? { ...c, ...next } : c));
    setClips(copy);
    if (id) api.clips(id, copy);
  }

  function move(index: number, direction: -1 | 1) {
    const next = index + direction;
    if (next < 0 || next >= clips.length) return;
    const copy = [...clips];
    [copy[index], copy[next]] = [copy[next], copy[index]];
    setClips(copy);
    if (id) api.clips(id, copy);
  }

  async function generate() {
    if (!id) return;
    await api.render(id, clips.filter((c) => c.kept !== false));
    navigate(`/jobs/${id}/analysis`);
  }

  if (!job || !id) return <p>Loading…</p>;
  const estimated = clips.filter((c) => c.kept !== false).reduce((s, c) => s + (c.end - c.start), 0);
  const kept = clips.filter((c) => c.kept !== false);
  const sources: JobSource[] =
    job.sources && job.sources.length
      ? job.sources
      : [{ id: "v0", filename: job.original_filename, duration }];
  const multi = sources.length > 1;

  return (
    <div>
      <p>
        Target {job.min_duration}–{job.max_duration}s · Estimated {estimated.toFixed(1)}s · {clips.length} clips
        {multi ? ` · ${sources.length} source videos` : ""}
      </p>
      {sources.map((source) => {
        const sourceClips = clips.filter((clip) => (clip.source_id || sources[0].id) === source.id);
        const sourceDuration = source.duration || duration;
        return (
          <div key={source.id}>
            <p style={{ color: "var(--muted)" }}>
              {messages.review.sourceTimeline.replace("{name}", source.filename || source.id)}
            </p>
            <div className="timeline" aria-label={`${source.filename} timeline`}>
              {sourceClips.map((clip, i) => (
                <div
                  key={`${source.id}-${i}`}
                  className="clip-block"
                  title={`${sourceLabel(clip)} ${formatTime(clip.start)}–${formatTime(clip.end)} (${(clip.end - clip.start).toFixed(1)}s)`}
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
      <p style={{ color: "var(--muted)" }}>{messages.review.highlightOrder}</p>
      <div className="sequence-timeline" aria-label="Highlight sequence durations">
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
      {clips.map((clip, i) => (
        <article className="clip-card" key={i}>
          <header>
            <strong>
              {formatTime(clip.start)}–{formatTime(clip.end)} · {(clip.end - clip.start).toFixed(1)}s
            </strong>
            <span className="score">{Math.round((clip.score || 0) * 100)}%</span>
          </header>
          <p className="source-badge">
            {messages.review.source}: {sourceLabel(clip)}
          </p>
          <p>{clip.reason}</p>
          {clip.duration_reason && (
            <p className="duration-reason">
              <span>{messages.review.whyLength}</span>
              {clip.duration_reason}
            </p>
          )}
          {clip.transcript && <p style={{ color: "var(--muted)" }}>{clip.transcript}</p>}
          <div className="grid-2">
            <label>
              Start
              <input
                type="number"
                step="0.1"
                value={clip.start}
                onChange={(e) => patch(i, { start: Number(e.target.value) })}
              />
            </label>
            <label>
              End
              <input
                type="number"
                step="0.1"
                value={clip.end}
                onChange={(e) => patch(i, { end: Number(e.target.value) })}
              />
            </label>
          </div>
          <div className="btn-row" style={{ marginTop: 8 }}>
            <button className="btn" onClick={() => patch(i, { kept: clip.kept === false })}>
              {clip.kept === false ? messages.review.keep : messages.review.remove}
            </button>
            <button className="btn" onClick={() => patch(i, { start: Math.max(0, clip.start - 0.5) })}>
              {messages.review.extend}
            </button>
            <button className="btn" onClick={() => patch(i, { end: Math.max(clip.start + 0.4, clip.end - 0.4) })}>
              {messages.review.trim}
            </button>
            <button className="btn" disabled={i === 0} onClick={() => move(i, -1)}>
              {messages.review.moveUp}
            </button>
            <button className="btn" disabled={i === clips.length - 1} onClick={() => move(i, 1)}>
              {messages.review.moveDown}
            </button>
          </div>
        </article>
      ))}
      <div className="btn-row">
        <button className="btn primary" onClick={generate}>
          {messages.review.generate}
        </button>
        {REGEN.map(([key, label]) => (
          <button
            key={key}
            className="btn"
            onClick={async () => {
              const result = await api.regenerate(id, key);
              if (result.job) setJob(result.job);
              await refresh();
            }}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
