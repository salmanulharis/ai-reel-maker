import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, formatElapsed, formatTime } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, FilmIcon, InlineStatus, PageLoader, PlayIcon, Spinner } from "../components/Feedback";
import { ClipPreviewPanel, useClipPreview } from "../components/ClipPreview";
import { MoreMoments } from "../components/MoreMoments";
import { EditorExport } from "../components/EditorExport";
import { insertIndex, insertIndexInShort, momentToClip, nearestShort, shortIndex } from "../services/clipPreview";
import type { Clip, Job, JobSource, Moment } from "../types";

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
  const sources: JobSource[] =
    job?.sources && job.sources.length
      ? job.sources
      : [{ id: "v0", filename: job?.original_filename || "", duration }];
  const preview = useClipPreview(id || "", clips, sources);
  const previewPanel = useRef<HTMLElement>(null);
  /** Goes up when a new selection replaces the clips, so "More moments" reloads. */
  const [planVersion, setPlanVersion] = useState(0);

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

  /** Change a start or end, and show the new cut point in the preview when it is paused there. */
  function setEdge(index: number, edge: "start" | "end", value: number) {
    patch(index, { [edge]: value });
    preview.showEdge(index, edge, value);
  }

  function move(index: number, direction: -1 | 1) {
    const next = index + direction;
    if (next < 0 || next >= clips.length) return;
    if (shortIndex(clips[next]) !== shortIndex(clips[index])) return;
    const copyOfClips = [...clips];
    [copyOfClips[index], copyOfClips[next]] = [copyOfClips[next], copyOfClips[index]];
    // The preview follows the clip it was showing to its new place.
    if (preview.selected === index) preview.retarget(next);
    else if (preview.selected === next) preview.retarget(index);
    save(copyOfClips);
  }

  function previewClip(index: number) {
    preview.playClip(index);
    showPreviewPanel();
  }

  function previewMoment(moment: Moment, key: string) {
    preview.playMoment(moment, key);
    showPreviewPanel();
  }

  /** Add a suggested moment as a kept clip. Returns its clip number. */
  function addMoment(moment: Moment): number {
    const order = (job?.config as { order_mode?: string } | undefined)?.order_mode;
    const short = isShorts() ? nearestShort(clips, moment, sources) : 0;
    const index = isShorts()
      ? insertIndexInShort(clips, moment, short)
      : insertIndex(clips, moment, sources, !order || order === "chronological");
    // Clips after the new one move down a place; the preview stays on the clip it showed.
    if (preview.selected >= index) preview.retarget(preview.selected + 1);
    save([...clips.slice(0, index), momentToClip(moment, short), ...clips.slice(index)]);
    return index + 1;
  }

  /** The job makes several separate shorts (clips are grouped by short). */
  function isShorts() {
    const count = Number((job?.config as { shorts_count?: number } | undefined)?.shorts_count || 0);
    return count >= 2 || clips.some((clip) => shortIndex(clip) > 0);
  }

  /** Bring the player into view when it is off screen (it is above the clips on narrow screens). */
  function showPreviewPanel() {
    const panel = previewPanel.current;
    if (!panel) return;
    const box = panel.getBoundingClientRect();
    if (box.top >= 0 && box.bottom <= window.innerHeight) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    panel.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
  }

  async function generate() {
    if (!id) return;
    preview.pause();
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
    preview.clear();
    setPending(preset);
    setError("");
    try {
      const result = await api.regenerate(id, preset);
      if (result.job) setJob(result.job);
      await refresh();
      setPlanVersion((value) => value + 1);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPending("");
    }
  }

  if (!job || !id) return error ? <Alert>{error}</Alert> : <PageLoader />;
  const kept = clips.filter((c) => c.kept !== false);
  const estimated = kept.reduce((s, c) => s + (c.end - c.start), 0);
  const autoLength = (job.config as { length_mode?: string } | undefined)?.length_mode === "auto";
  const shortsMode = isShorts();
  // Each short has its own length goal; one highlight is checked as a whole.
  const over = !shortsMode && !autoLength && estimated > job.max_duration;
  const multi = sources.length > 1;
  const shortNumbers = shortsMode ? [...new Set(clips.map(shortIndex))].sort((a, b) => a - b) : [0];

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
          {copy.target}{" "}
          <strong>
            {autoLength
              ? fill(copy.targetAuto, { n: job.min_duration })
              : job.min_duration === job.max_duration
                ? `${job.max_duration}s`
                : `${job.min_duration}–${job.max_duration}s`}
          </strong>
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
        {shortsMode && (
          <span>
            {copy.shorts} <strong>{shortNumbers.length}</strong>
          </span>
        )}
        {(job.timing?.analysis_seconds ?? 0) > 0 && (
          <span>
            {copy.analysisTook} <strong>{formatElapsed(job.timing?.analysis_seconds)}</strong>
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
              {(preview.selected >= 0 || preview.moment) && preview.loadedSource === source.id && sourceDuration > 0 && (
                <div
                  className="playhead"
                  style={{ left: `${Math.min(100, Math.max(0, (preview.time / sourceDuration) * 100))}%` }}
                />
              )}
            </div>
          </div>
        );
      })}
      {!shortsMode && <p className="section-label">{copy.highlightOrder}</p>}
      {!shortsMode && <div className="sequence-timeline" aria-hidden="true">
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
      </div>}

      <div className="review-body">
      <ClipPreviewPanel
        ref={previewPanel}
        preview={preview}
        clips={clips}
        sources={sources}
        onTrim={setEdge}
      />
      <div className="clip-groups">
      {shortNumbers.map((short) => {
        const items = clips.map((clip, i) => ({ clip, i })).filter(({ clip }) => !shortsMode || shortIndex(clip) === short);
        const keptItems = items.filter(({ clip }) => clip.kept !== false);
        const length = keptItems.reduce((sum, { clip }) => sum + (clip.end - clip.start), 0);
        const shortOver = shortsMode && length > job.max_duration;
        return (
          <section
            key={short}
            className={shortsMode ? "clip-group" : undefined}
            aria-label={shortsMode ? fill(copy.shortLabel, { n: short + 1 }) : undefined}
          >
            {shortsMode && (
              <h3 className="short-title">
                {fill(copy.shortTitle, { n: short + 1 })}
                <span className={`muted ${shortOver ? "over" : ""}`}>
                  {fill(copy.shortSummary, { length: length.toFixed(1), kept: keptItems.length, total: items.length })}
                </span>
              </h3>
            )}
            {shortOver && <p className="field-hint warn">{copy.shortOver}</p>}
            <div className="clip-grid">
      {items.map(({ clip, i }) => {
        const removed = clip.kept === false;
        const label = fill(copy.clipNumber, { n: i + 1 });
        const inPreview = preview.selected === i;
        return (
          <article
            className={`clip-card ${removed ? "removed" : ""} ${inPreview ? "selected" : ""}`}
            key={i}
            aria-label={label}
            aria-current={inPreview ? "true" : undefined}
          >
            <header>
              <strong>
                {label} · {formatTime(clip.start)}–{formatTime(clip.end)} · {(clip.end - clip.start).toFixed(1)}s
              </strong>
              <span className="tag-row">
                {inPreview && <span className="tag accent">{copy.previewing}</span>}
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
            <div className="form-grid clip-times">
              <div className="field">
                <label htmlFor={`clip-${i}-start`}>{copy.start}</label>
                <input
                  id={`clip-${i}-start`}
                  type="number"
                  step="0.1"
                  min={0}
                  value={clip.start}
                  onChange={(e) => setEdge(i, "start", Number(e.target.value))}
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
                  onChange={(e) => setEdge(i, "end", Number(e.target.value))}
                />
              </div>
            </div>
            <div className="btn-row clip-actions">
              <button
                type="button"
                className="btn sm"
                aria-label={fill(copy.previewClipLabel, { clip: label })}
                onClick={() => previewClip(i)}
              >
                <PlayIcon size={14} />
                {copy.previewClip}
              </button>
              <button
                className={`btn sm ${removed ? "primary" : ""}`}
                aria-label={`${removed ? copy.keep : copy.remove} ${label}`}
                onClick={() => patch(i, { kept: removed })}
              >
                {removed ? copy.keep : copy.remove}
              </button>
              <button className="btn sm" disabled={removed} onClick={() => setEdge(i, "start", Math.max(0, clip.start - 0.5))}>
                {copy.extend}
              </button>
              <button
                className="btn sm"
                disabled={removed}
                onClick={() => setEdge(i, "end", Math.max(clip.start + 0.4, clip.end - 0.4))}
              >
                {copy.trim}
              </button>
              <span className="btn-row push-right">
                <button
                  className="btn sm ghost"
                  disabled={i === 0 || shortIndex(clips[i - 1]) !== shortIndex(clip)}
                  aria-label={`${copy.moveUp}: ${label}`}
                  onClick={() => move(i, -1)}
                >
                  ↑ {copy.moveUp}
                </button>
                <button
                  className="btn sm ghost"
                  disabled={i === clips.length - 1 || shortIndex(clips[i + 1]) !== shortIndex(clip)}
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
            </div>
          </section>
        );
      })}
      </div>
      </div>

      <MoreMoments
        jobId={id}
        clips={clips}
        sources={sources}
        version={planVersion}
        preview={preview}
        onPreview={previewMoment}
        onAdd={addMoment}
      />
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
          <EditorExport
            jobId={id}
            disabled={pending !== "" || kept.length === 0}
            // Clip changes save in the background; make sure the latest are saved first.
            beforeExport={() => api.clips(id, clips)}
          />
          <span className="muted">
            {kept.length}/{clips.length} · {estimated.toFixed(1)}s
          </span>
        </div>
      </div>
    </div>
  );
}
