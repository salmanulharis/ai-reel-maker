import { Suspense, lazy, useEffect, useRef, useState } from "react";
import { Alert, InlineStatus, Spinner } from "./Feedback";
import { api, formatTime } from "../services/api";
import { fill, messages } from "../i18n";
import type { BriefBeat, NarrationState, VideoBrief } from "../types";

const ReelPreview = lazy(() => import("./ReelPreview"));
const copy = messages.narrated;

function words(text: string) {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

/**
 * Review page: the narrated video's brief. The words can be edited; speaking the narration only
 * voices lines that changed (the same cache the render uses), and the preview plays the kept
 * clips with the voice, texts and captions in the browser.
 */
export function BriefPanel({
  jobId,
  narrationOn,
  clipsKey,
  beforePreview,
}: {
  jobId: string;
  narrationOn: boolean;
  clipsKey: string;
  /** Saves the clips as they are on the page, so the preview plays them. */
  beforePreview?: () => Promise<unknown>;
}) {
  const [brief, setBrief] = useState<VideoBrief | null>(null);
  const [draft, setDraft] = useState<VideoBrief | null>(null);
  const [spoken, setSpoken] = useState<string[]>([]);
  const [narration, setNarration] = useState<NarrationState | null>(null);
  const [voiceOn, setVoiceOn] = useState(narrationOn);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [saved, setSaved] = useState(false);
  const [spec, setSpec] = useState<Record<string, unknown> | null | undefined>(undefined);
  const [loaded, setLoaded] = useState(false);
  const poll = useRef<number | null>(null);

  useEffect(() => {
    let stop = false;
    api
      .jobBrief(jobId)
      .then((info) => {
        if (stop) return;
        setBrief(info.brief);
        setDraft(info.brief);
        setSpoken(info.spoken);
        setNarration(info.narration);
        setLoaded(true);
        if (info.narration.state === "running") watch();
      })
      .catch((err) => {
        if (stop) return;
        setError(String((err as Error).message || err));
        setLoaded(true);
      });
    return () => {
      stop = true;
      if (poll.current) window.clearInterval(poll.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  // A preview shown for older clips is out of date once clips change.
  useEffect(() => setSpec(undefined), [clipsKey]);

  function watch() {
    if (poll.current) window.clearInterval(poll.current);
    poll.current = window.setInterval(async () => {
      try {
        const info = await api.jobBrief(jobId);
        setNarration(info.narration);
        setSpoken(info.spoken);
        if (info.narration.state !== "running" && poll.current) {
          window.clearInterval(poll.current);
          poll.current = null;
          // The preview is opened again with the new voice.
          setSpec(undefined);
        }
      } catch {
        // The next poll tries again.
      }
    }, 1500);
  }

  if (!loaded) return null;
  if (error && !brief) return <Alert>{error}</Alert>;
  if (!brief || !draft) {
    return (
      <section className="panel" aria-labelledby="brief-title">
        <h3 id="brief-title" className="sub-title">
          {copy.briefTitle}
        </h3>
        <p className="muted">{copy.noBrief}</p>
      </section>
    );
  }

  const dirty = JSON.stringify(draft) !== JSON.stringify(brief);
  const unspoken = draft.beats.filter((beat) => beat.narration.trim() && !spoken.includes(beat.id)).length;

  function setBeat(id: string, patch: Partial<BriefBeat>) {
    setDraft((current) =>
      current ? { ...current, beats: current.beats.map((beat) => (beat.id === id ? { ...beat, ...patch } : beat)) } : current
    );
    setSaved(false);
  }

  async function save() {
    if (!draft) return;
    setBusy("save");
    setError("");
    try {
      const result = await api.saveBrief(jobId, {
        title: draft.title,
        hook: draft.hook,
        call_to_action: draft.call_to_action,
        beats: draft.beats.map((beat) => ({ id: beat.id, narration: beat.narration, on_screen_text: beat.on_screen_text })),
      });
      setBrief(result.brief);
      setDraft(result.brief);
      setWarnings(result.warnings);
      setSaved(true);
      setSpec(undefined);
    } catch (err) {
      setError(String((err as Error).message || err));
    } finally {
      setBusy("");
    }
  }

  async function speak() {
    setError("");
    try {
      if (dirty) await save();
      const state = await api.speakNarration(jobId);
      setNarration(state);
      if (state.state === "running") watch();
    } catch (err) {
      setError(String((err as Error).message || err));
    }
  }

  async function loadPreview() {
    setBusy("preview");
    try {
      if (beforePreview) await beforePreview();
      setSpec((await api.previewSpec(jobId)).spec);
    } catch (err) {
      setError(String((err as Error).message || err));
    } finally {
      setBusy("");
    }
  }

  async function toggleVoice(on: boolean) {
    setVoiceOn(on);
    try {
      await api.renderOptions(jobId, { narration: on });
      setSpec(undefined);
    } catch (err) {
      setVoiceOn(!on);
      setError(String((err as Error).message || err));
    }
  }

  const engine = brief.engine === "claude" ? copy.engineNameClaude : copy.engineNameLocal;
  const speaking = narration?.state === "running";

  return (
    <section className="panel brief-panel" aria-labelledby="brief-title">
      <h3 id="brief-title" className="sub-title">
        {copy.briefTitle}
      </h3>
      <p className="panel-intro">
        {copy.briefIntro} <span className="muted">{fill(copy.writtenBy, { engine, model: brief.model || "rules" })}</span>
      </p>
      {error && <Alert>{error}</Alert>}

      <div className="form-grid">
        {(["title", "hook", "call_to_action"] as const).map((field) => (
          <label key={field} className="field">
            <span className="field-label">{field === "title" ? copy.title : field === "hook" ? copy.hook : copy.cta}</span>
            <input
              type="text"
              value={draft[field]}
              maxLength={field === "title" ? 80 : field === "hook" ? 160 : 100}
              onChange={(e) => {
                setDraft({ ...draft, [field]: e.target.value });
                setSaved(false);
              }}
            />
          </label>
        ))}
      </div>

      <ol className="brief-beats">
        {draft.beats.map((beat, index) => {
          const count = words(beat.narration);
          const seconds = beat.clips.reduce((sum, clip) => sum + Math.max(0, clip.end - clip.start), 0);
          return (
            <li key={beat.id} className="brief-beat">
              <p className="brief-beat-head">
                <strong>{fill(copy.beat, { n: index + 1 })}</strong> <span className="tag">{beat.purpose}</span>{" "}
                <span className="muted">
                  {beat.clips.map((clip) => `${formatTime(clip.start)}–${formatTime(clip.end)}`).join(", ")} ({seconds.toFixed(0)}s)
                </span>
              </p>
              <label className="field">
                <span className="field-label">{copy.narration}</span>
                <textarea
                  rows={2}
                  maxLength={400}
                  value={beat.narration}
                  disabled={beat.max_words === 0 && !beat.narration}
                  placeholder={beat.max_words === 0 ? copy.noNarration : ""}
                  onChange={(e) => setBeat(beat.id, { narration: e.target.value })}
                />
                {beat.max_words > 0 && (
                  <span className={`field-hint ${count > beat.max_words ? "error-text" : ""}`}>
                    {fill(count > beat.max_words ? copy.wordsOver : copy.words, { n: count, max: beat.max_words })}
                  </span>
                )}
              </label>
              <label className="field">
                <span className="field-label">{copy.onScreen}</span>
                <input
                  type="text"
                  maxLength={60}
                  value={beat.on_screen_text}
                  onChange={(e) => setBeat(beat.id, { on_screen_text: e.target.value })}
                />
              </label>
            </li>
          );
        })}
      </ol>

      {warnings.map((warning) => (
        <p key={warning} className="field-hint error-text">
          {warning}
        </p>
      ))}
      {brief.checks.length > 0 && (
        <details className="advanced">
          <summary>{copy.checks}</summary>
          <ul>
            {brief.checks.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </details>
      )}

      <div className="btn-row">
        <button className="btn" type="button" disabled={!dirty || busy === "save"} aria-busy={busy === "save"} onClick={save}>
          {busy === "save" && <Spinner />}
          {busy === "save" ? copy.saving : copy.save}
        </button>
        {saved && !dirty && <InlineStatus ok>{copy.saved}</InlineStatus>}
      </div>

      <label className="check">
        <input type="checkbox" checked={voiceOn} onChange={(e) => void toggleVoice(e.target.checked)} />
        <span>
          <strong>{copy.voiceInVideo}</strong>
          <span className="field-hint block">{copy.voiceHint}</span>
        </span>
      </label>
      {voiceOn && (
        <div className="btn-row">
          <button className="btn sm" type="button" disabled={speaking || (unspoken === 0 && !dirty)} onClick={speak}>
            {speaking && <Spinner />}
            {spoken.length ? copy.speakAgain : copy.speak}
          </button>
          {speaking && narration && (
            <InlineStatus busy>{fill(copy.speaking, { done: narration.done, total: narration.total })}</InlineStatus>
          )}
          {!speaking && unspoken === 0 && spoken.length > 0 && <InlineStatus ok>{copy.spoken}</InlineStatus>}
          {narration?.state === "error" && narration.error && <span className="error-text">{narration.error}</span>}
        </div>
      )}

      <h4 className="sub-title">{copy.preview}</h4>
      <p className="field-hint">{copy.previewHint}</p>
      {spec === undefined ? (
        <button className="btn sm" type="button" disabled={busy === "preview"} aria-busy={busy === "preview"} onClick={loadPreview}>
          {busy === "preview" && <Spinner />}
          {copy.showPreview}
        </button>
      ) : spec === null ? (
        <p className="muted">{copy.noPreview}</p>
      ) : (
        <Suspense fallback={<InlineStatus busy>{copy.loadingPreview}</InlineStatus>}>
          <ReelPreview spec={spec} />
        </Suspense>
      )}
    </section>
  );
}
