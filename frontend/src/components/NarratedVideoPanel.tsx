import { useEffect, useState } from "react";
import { InlineStatus, Spinner } from "./Feedback";
import { api } from "../services/api";
import { messages } from "../i18n";
import type { EngineStatus, HighlightForm } from "../types";

const copy = messages.narrated;

/**
 * Create page: make a narrated video, choose who writes its brief, and the voice-over.
 * The engine is the user's choice: it is checked on request and never switched automatically.
 */
export function NarratedVideoPanel({ form, setForm }: { form: HighlightForm; setForm: (next: HighlightForm) => void }) {
  const [engines, setEngines] = useState<EngineStatus[] | null>(null);
  const [checking, setChecking] = useState(false);
  const [voiceReady, setVoiceReady] = useState<boolean | null>(null);
  const [defaults, setDefaults] = useState<{ engine?: string; narration?: boolean } | null>(null);

  // The Settings choices are the starting point when a narrated video is switched on.
  useEffect(() => {
    let stop = false;
    api
      .settings()
      .then(({ settings }) => {
        if (!stop) setDefaults({ engine: String(settings.brief_engine || "local"), narration: settings.narration_default !== false });
      })
      .catch(() => !stop && setDefaults({}));
    return () => {
      stop = true;
    };
  }, []);

  useEffect(() => {
    if (!form.video_brief) return;
    let stop = false;
    api
      .videoTools()
      .then((tools) => !stop && setVoiceReady(Boolean(tools.voice.installed && tools.voice.runtime)))
      .catch(() => !stop && setVoiceReady(null));
    return () => {
      stop = true;
    };
  }, [form.video_brief]);

  function toggle(on: boolean) {
    if (!on) {
      setForm({ ...form, video_brief: false });
      return;
    }
    setForm({
      ...form,
      video_brief: true,
      brief_engine: defaults?.engine === "claude" ? "claude" : form.brief_engine,
      narration: defaults?.narration ?? form.narration,
    });
  }

  async function check() {
    setChecking(true);
    try {
      setEngines((await api.briefEngines(true)).engines);
    } catch {
      setEngines([]);
    } finally {
      setChecking(false);
    }
  }

  const status = (name: "local" | "claude") => engines?.find((item) => item.name === name);

  return (
    <fieldset className="config-section">
      <legend>{copy.section}</legend>
      <label className="check">
        <input type="checkbox" checked={form.video_brief} onChange={(e) => toggle(e.target.checked)} />
        <span>
          <strong>{copy.enable}</strong>
          <span className="field-hint block">{copy.enableHint}</span>
        </span>
      </label>
      {form.video_brief && (
        <>
          <p className="field-label" id="brief-engine-label">
            {copy.engine}
          </p>
          <div role="radiogroup" aria-labelledby="brief-engine-label">
            {(["local", "claude"] as const).map((name) => {
              const item = status(name);
              return (
                <label className="check" key={name}>
                  <input
                    type="radio"
                    name="brief-engine"
                    checked={form.brief_engine === name}
                    onChange={() => setForm({ ...form, brief_engine: name })}
                  />
                  <span>
                    <strong>{name === "local" ? copy.engineLocal : copy.engineClaude}</strong>{" "}
                    <span className={`tag ${item ? (item.ready ? "ok" : "warn") : ""}`}>
                      {item ? (item.ready ? copy.ready : copy.notReady) : copy.notChecked}
                    </span>
                    <span className="field-hint block">{name === "local" ? copy.engineLocalHint : copy.engineClaudeHint}</span>
                    {item?.problems.map((problem) => (
                      <span className="field-hint block error-text" key={problem}>
                        {problem}
                      </span>
                    ))}
                    {item?.ready &&
                      item.notes.map((note) => (
                        <span className="field-hint block" key={note}>
                          {note}
                        </span>
                      ))}
                  </span>
                </label>
              );
            })}
          </div>
          <div className="btn-row">
            <button type="button" className="btn sm" disabled={checking} aria-busy={checking} onClick={check}>
              {checking && <Spinner />}
              {checking ? copy.checking : copy.check}
            </button>
            {checking && <InlineStatus busy>{copy.checking}</InlineStatus>}
          </div>
          <p className="field-hint">{copy.chooseHint}</p>

          <label className="check">
            <input type="checkbox" checked={form.narration} onChange={(e) => setForm({ ...form, narration: e.target.checked })} />
            <span>
              <strong>{copy.voice}</strong>
              <span className="field-hint block">{copy.voiceHint}</span>
              {form.narration && voiceReady === false && <span className="field-hint block error-text">{copy.voiceMissing}</span>}
            </span>
          </label>
          {form.subtitles !== "burned" && <p className="field-hint">{copy.captionsHint}</p>}
        </>
      )}
    </fieldset>
  );
}
