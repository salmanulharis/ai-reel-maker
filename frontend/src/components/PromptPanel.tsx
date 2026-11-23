import { useState } from "react";
import { Alert, InlineStatus, Spinner } from "./Feedback";
import { api, formatRanges } from "../services/api";
import { fill, messages } from "../i18n";
import type { HighlightForm, InterpretResult, PromptRequirements, RequirementKey } from "../types";

const copy = messages.prompt;
const config = messages.config;
const GROUPS: RequirementKey[] = ["keep", "remove", "moments", "visual_cues", "audio_cues"];
const RANGE_FIELDS = new Set(["must_include", "must_exclude"]);
const EMPTY: PromptRequirements = { summary: "", keep: [], remove: [], moments: [], visual_cues: [], audio_cues: [] };

const OPTION_LABELS: Record<string, Record<string, string>> = {
  aspect_ratio: {
    original: config.aspectOriginal,
    "16:9": config.aspect169,
    "9:16": config.aspect916,
    "1:1": config.aspect11,
    "4:5": config.aspect45,
  },
  subtitles: { none: config.subtitlesNone, srt: config.subtitlesFile, burned: config.subtitlesBurned, auto: config.subtitlesFile },
  caption_style: { plain: config.captionPlain, clean: config.captionClean, bold: config.captionBold, boxed: config.captionBoxed },
  tighten: { off: config.tightenOff, pauses: config.tightenPauses, fillers: config.tightenFillers },
  transition: { cut: config.transitionCut, fade: config.transitionFade, auto: config.transitionAuto },
  order_mode: {
    chronological: config.orderChronological,
    narrative: config.orderNarrative,
    hook: config.orderHook,
    ai: config.orderAi,
  },
  reframe: { subject: config.reframeSubject, center: config.reframeCenter },
  style: messages.styles as Record<string, string>,
};

const FIELD_LABELS: Record<string, string> = {
  aspect_ratio: config.aspect,
  subtitles: config.subtitles,
  caption_style: config.captionStyle,
  tighten: config.tighten,
  transition: config.transition,
  order_mode: config.ordering,
  reframe: config.reframe,
  style: config.style,
  level_audio: config.levelAudio,
  content_priorities: config.priorities,
  avoid: config.avoid,
  avoid_custom: config.avoidCustom,
  custom_focus: copy.focus,
  shorts_count: copy.shorts,
  must_include: copy.keepTimes,
  must_exclude: copy.removeTimes,
};

type Change = { field: string; label: string; value: string; source: string };

function describe(result: InterpretResult): Change[] {
  const settings = result.settings;
  const changes: Change[] = [];
  const sourceOf = (field: string) => result.changes.find((c) => c.field === field)?.source || "ai";
  if (settings.length_mode === "auto") {
    changes.push({ field: "length", label: copy.length, value: copy.lengthAuto, source: sourceOf("length_mode") });
  } else if (settings.min_duration !== undefined || settings.max_duration !== undefined) {
    changes.push({
      field: "length",
      label: copy.length,
      value: fill(copy.lengthRange, { min: Number(settings.min_duration ?? 0), max: Number(settings.max_duration ?? 0) }),
      source: sourceOf("min_duration"),
    });
  }
  for (const [field, raw] of Object.entries(settings)) {
    if (["min_duration", "max_duration", "length_mode"].includes(field) || !(field in FIELD_LABELS)) continue;
    let value: string;
    if (RANGE_FIELDS.has(field)) value = formatRanges(raw);
    else if (field === "content_priorities") value = (raw as string[]).map((k) => (messages.priorities as Record<string, string>)[k] || k).join(", ");
    else if (field === "avoid")
      value = (raw as string[])
        .map((k) => (config.avoidItems as Record<string, { label: string }>)[k]?.label || k)
        .join(", ");
    else if (field === "shorts_count") value = String(raw);
    else if (typeof raw === "boolean") value = raw ? copy.on : copy.off;
    else value = OPTION_LABELS[field]?.[String(raw)] || String(raw);
    changes.push({ field, label: FIELD_LABELS[field], value, source: sourceOf(field) });
  }
  return changes;
}

/** The form with the settings read from the description applied (time ranges as text). */
function applySettings(form: HighlightForm, result: InterpretResult): HighlightForm {
  const next: HighlightForm = { ...form, requirements: result.requirements };
  const target = next as unknown as Record<string, unknown>;
  for (const [field, value] of Object.entries(result.settings)) {
    if (!(field in form)) continue;
    target[field] = RANGE_FIELDS.has(field) ? formatRanges(value) : value;
  }
  return next;
}

export function PromptPanel({
  form,
  setForm,
  jobId,
  understood,
  setUnderstood,
}: {
  form: HighlightForm;
  setForm: (next: HighlightForm) => void;
  jobId: string | null;
  /** The last reading of the description, or null before the first one. */
  understood: { prompt: string; result: InterpretResult } | null;
  setUnderstood: (next: { prompt: string; result: InterpretResult } | null) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [adding, setAdding] = useState<Record<string, string>>({});
  const stale = understood !== null && understood.prompt !== form.prompt.trim();
  const requirements = form.requirements || EMPTY;

  async function understand() {
    setBusy(true);
    setError("");
    try {
      const result = await api.interpret(form.prompt.trim(), jobId, form.use_speech);
      setForm(applySettings(form, result));
      setUnderstood({ prompt: form.prompt.trim(), result });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function setGroup(key: RequirementKey, items: string[]) {
    setForm({ ...form, requirements: { ...requirements, [key]: items } });
  }

  function addTo(key: RequirementKey) {
    const text = (adding[key] || "").trim();
    if (!text || requirements[key].includes(text)) return;
    setGroup(key, [...requirements[key], text].slice(0, 8));
    setAdding({ ...adding, [key]: "" });
  }

  return (
    <div className="panel config prompt-panel">
      <div className="field">
        <label htmlFor="prompt-text">{copy.label}</label>
        <textarea
          id="prompt-text"
          rows={4}
          maxLength={2000}
          value={form.prompt}
          placeholder={copy.placeholder}
          aria-describedby="prompt-hint"
          onChange={(e) => setForm({ ...form, prompt: e.target.value })}
        />
        <p className="field-hint" id="prompt-hint">
          {copy.hint}
        </p>
      </div>

      <fieldset className="config-section">
        <legend>{copy.analysis}</legend>
        <label className="check">
          <input
            type="checkbox"
            checked={form.use_visuals}
            disabled={form.use_visuals && !form.use_speech}
            onChange={(e) => setForm({ ...form, use_visuals: e.target.checked })}
          />
          <span>
            <strong>{copy.video}</strong>
            <span className="field-hint block">{copy.videoHint}</span>
          </span>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={form.use_speech}
            disabled={form.use_speech && !form.use_visuals}
            onChange={(e) =>
              setForm({ ...form, use_speech: e.target.checked, ...(e.target.checked ? {} : { subtitles: "none", tighten: "off" as const }) })
            }
          />
          <span>
            <strong>{copy.audio}</strong>
            <span className="field-hint block">{copy.audioHint}</span>
          </span>
        </label>
        <p className="field-hint">{copy.bothHint}</p>
      </fieldset>

      <fieldset className="config-section">
        <legend>{copy.depth}</legend>
        <label className="check">
          <input
            type="radio"
            name="analysis-depth"
            checked={form.analysis_depth === "balanced"}
            onChange={() => setForm({ ...form, analysis_depth: "balanced" })}
          />
          <span>
            <strong>{copy.depthBalanced}</strong>
            <span className="field-hint block">{copy.depthBalancedHint}</span>
          </span>
        </label>
        <label className="check">
          <input
            type="radio"
            name="analysis-depth"
            checked={form.analysis_depth === "best"}
            onChange={() => setForm({ ...form, analysis_depth: "best" })}
          />
          <span>
            <strong>{copy.depthBest}</strong>
            <span className="field-hint block">{copy.depthBestHint}</span>
          </span>
        </label>
      </fieldset>

      <label className="check">
        <input
          type="checkbox"
          checked={form.render_after_analysis}
          onChange={(e) => setForm({ ...form, render_after_analysis: e.target.checked })}
        />
        <span>
          <strong>{copy.renderNow}</strong>
          <span className="field-hint block">{copy.renderNowHint}</span>
        </span>
      </label>

      <div className="btn-row">
        <button className="btn" type="button" disabled={busy || !form.prompt.trim()} aria-busy={busy} onClick={understand}>
          {busy && <Spinner />}
          {busy ? copy.understanding : understood ? copy.understandAgain : copy.understand}
        </button>
        {busy && <InlineStatus busy>{copy.understandingHint}</InlineStatus>}
        {stale && !busy && <span className="muted">{copy.stale}</span>}
      </div>
      {error && <Alert>{error}</Alert>}

      {understood && (
        <div className="prompt-preview" aria-live="polite">
          <h4 className="sub-title">{copy.understood}</h4>
          <p>{understood.result.summary}</p>
          {understood.result.warning && <Alert kind="info">{understood.result.warning}</Alert>}
          {describe(understood.result).length > 0 && (
            <>
              <h4 className="sub-title">{copy.settingsChosen}</h4>
              <ul className="prompt-changes">
                {describe(understood.result).map((change) => (
                  <li key={change.field}>
                    <span className="muted">{change.label}:</span> {change.value}{" "}
                    <span className={`tag ${change.source === "default" ? "" : "ok"}`}>
                      {change.source === "default" ? copy.sourceDefault : copy.sourceYou}
                    </span>
                  </li>
                ))}
              </ul>
              <p className="field-hint">{copy.adjustHint}</p>
            </>
          )}
          <h4 className="sub-title">{copy.requirements}</h4>
          {GROUPS.map((key) => (
            <div className="field requirement-group" key={key}>
              <span className="field-label">{copy.groups[key]}</span>
              <div className="chips" role="list" aria-label={copy.groups[key]}>
                {requirements[key].map((item) => (
                  <span className="chip active" role="listitem" key={item}>
                    {item}
                    <button
                      type="button"
                      className="chip-remove"
                      aria-label={fill(copy.removeItem, { item })}
                      onClick={() => setGroup(key, requirements[key].filter((other) => other !== item))}
                    >
                      ×
                    </button>
                  </span>
                ))}
                {requirements[key].length === 0 && <span className="muted">{copy.none}</span>}
              </div>
              <div className="inline-add">
                <input
                  type="text"
                  maxLength={160}
                  value={adding[key] || ""}
                  placeholder={copy.addPlaceholder}
                  aria-label={fill(copy.addTo, { group: copy.groups[key] })}
                  onChange={(e) => setAdding({ ...adding, [key]: e.target.value })}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addTo(key);
                    }
                  }}
                />
                <button type="button" className="btn sm" disabled={!(adding[key] || "").trim()} onClick={() => addTo(key)}>
                  {copy.add}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Read the description (when it was not read yet, or changed) and return the form to start with. */
export async function understoodForm(
  form: HighlightForm,
  jobId: string | null,
  understood: { prompt: string } | null,
): Promise<{ form: HighlightForm; result: InterpretResult | null }> {
  if (understood && understood.prompt === form.prompt.trim()) return { form, result: null };
  const result = await api.interpret(form.prompt.trim(), jobId, form.use_speech);
  return { form: applySettings(form, result), result };
}
