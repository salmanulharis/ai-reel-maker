import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, InlineStatus, PlayIcon, PlusIcon } from "./Feedback";
import { formatPrecise, momentKey, overlapsClips } from "../services/clipPreview";
import type { ClipPreviewState } from "./ClipPreview";
import type { Clip, JobSource, Moment } from "../types";

const copy = messages.review.more;

/** Asked for from the server; more than shown, so some remain after clips are added. */
const FETCH_LIMIT = 24;
/** Shown at a time, best first. */
const SHOW_LIMIT = 8;

interface Props {
  jobId: string;
  clips: Clip[];
  sources: JobSource[];
  /** Changes when the clips are replaced by a new selection; an open list then reloads. */
  version: number;
  preview: ClipPreviewState;
  /** Play the moment in the shared preview. */
  onPreview: (moment: Moment, key: string) => void;
  /** Add the moment to the clips. Returns the clip number it got. */
  onAdd: (moment: Moment) => number;
}

/**
 * "More moments": strong moments the AI left out, so one can be watched and added by hand
 * instead of choosing a whole new selection. Loaded only when opened.
 */
export function MoreMoments({ jobId, clips, sources, version, preview, onPreview, onAdd }: Props) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [moments, setMoments] = useState<Moment[]>([]);
  const [available, setAvailable] = useState(true);
  const [error, setError] = useState("");
  const [added, setAdded] = useState("");
  const request = useRef(0);
  const multi = sources.length > 1;

  const load = useCallback(async () => {
    const token = ++request.current;
    setStatus("loading");
    setError("");
    try {
      const data = await api.moments(jobId, FETCH_LIMIT);
      if (token !== request.current) return;
      setMoments(data.moments || []);
      setAvailable(data.available !== false);
      setStatus("ready");
    } catch (err) {
      if (token !== request.current) return;
      setError((err as Error).message);
      setStatus("error");
    }
  }, [jobId]);

  useEffect(() => {
    if (open) void load();
  }, [open, version, load]);

  useEffect(() => {
    setAdded("");
  }, [version]);

  // Moments that now overlap a clip (for example one just added) drop out of the list.
  const visible = moments.filter((m) => !overlapsClips(m, clips, sources)).slice(0, SHOW_LIMIT);

  function add(moment: Moment) {
    const inPreview = preview.momentKey === momentKey(moment, sources);
    const n = onAdd(moment);
    // The player keeps showing it, now as the clip it became, so it can be trimmed at once.
    if (inPreview) preview.retarget(n - 1);
    setAdded(fill(copy.added, { n, start: formatPrecise(moment.start) }));
  }

  return (
    <section className="panel more-moments" aria-labelledby="more-moments-title">
      <div className="more-moments-head">
        <div>
          <h3 id="more-moments-title" className="sub-title">
            {copy.title}
          </h3>
          <p className="field-hint">{copy.hint}</p>
        </div>
        <button
          type="button"
          className="btn sm"
          aria-expanded={open}
          aria-controls="more-moments-body"
          onClick={() => setOpen((value) => !value)}
        >
          {open ? copy.hide : copy.show}
        </button>
      </div>
      {open && (
        <div id="more-moments-body" className="more-moments-body">
          {status === "loading" && <InlineStatus busy>{copy.loading}</InlineStatus>}
          {status === "error" && (
            <Alert>
              {fill(copy.error, { error })}{" "}
              <button type="button" className="btn sm" onClick={() => void load()}>
                {copy.retry}
              </button>
            </Alert>
          )}
          {status === "ready" && !available && <p className="field-hint">{copy.unavailable}</p>}
          {status === "ready" && available && visible.length === 0 && (
            <p className="field-hint">{copy.empty}</p>
          )}
          {added && <InlineStatus ok>{added}</InlineStatus>}
          {status === "ready" && visible.length > 0 && (
            <ul className="moment-list">
              {visible.map((moment) => {
                const key = momentKey(moment, sources);
                const range = fill(copy.range, { start: formatPrecise(moment.start), end: formatPrecise(moment.end) });
                const inPreview = preview.momentKey === key;
                return (
                  <li key={key} className={`moment ${inPreview ? "selected" : ""}`}>
                    <header>
                      <strong>
                        {range} · {(moment.end - moment.start).toFixed(1)}s
                      </strong>
                      <span className="tag-row">
                        {inPreview && <span className="tag accent">{messages.review.previewing}</span>}
                        <span className="score" title={messages.review.scoreHint}>
                          {fill(messages.review.score, { n: Math.round((moment.score || 0) * 100) })}
                        </span>
                      </span>
                    </header>
                    {multi && (
                      <p className="source-badge">
                        {messages.review.source}: {moment.source_name || moment.source_id}
                      </p>
                    )}
                    {moment.reason && <p>{moment.reason}</p>}
                    {moment.transcript && <p className="clip-transcript">“{moment.transcript}”</p>}
                    <div className="btn-row">
                      <button
                        type="button"
                        className="btn sm"
                        aria-label={fill(copy.previewLabel, { range })}
                        onClick={() => onPreview(moment, key)}
                      >
                        <PlayIcon size={14} />
                        {messages.review.previewClip}
                      </button>
                      <button
                        type="button"
                        className="btn sm primary"
                        aria-label={fill(copy.addLabel, { range })}
                        onClick={() => add(moment)}
                      >
                        <PlusIcon size={14} />
                        {copy.add}
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
