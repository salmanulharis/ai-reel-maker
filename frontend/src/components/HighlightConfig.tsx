import { useState } from "react";
import { fill, messages } from "../i18n";
import type { HighlightForm } from "../types";

const PRESETS = [
  [10, 10],
  [15, 15],
  [20, 20],
  [20, 30],
  [30, 30],
  [45, 45],
  [60, 60],
  [90, 90],
];

const STYLES = ["auto", "informative", "exciting", "social", "educational", "custom"] as const;
const PRIORITIES = ["balanced", "action", "speech", "magic", "visual", "information"] as const;

const copy = messages.config;

/** A reason the form cannot be submitted yet, or "" when it is fine. */
export function formProblem(form: HighlightForm): string {
  // With "No limit" the longest length is not used, so only the shortest has to be valid.
  const lengths = form.length_mode === "auto" ? [form.min_duration] : [form.min_duration, form.max_duration];
  if (lengths.some((value) => !Number.isFinite(value) || value < 1)) return copy.lengthMissing;
  return "";
}

const FRAME_RATES = ["24", "25", "30", "50", "60"] as const;
const FRAME_RATE_LABELS: Record<(typeof FRAME_RATES)[number], string> = {
  "24": copy.frameRate24,
  "25": copy.frameRate25,
  "30": copy.frameRate30,
  "50": copy.frameRate50,
  "60": copy.frameRate60,
};

function showRate(fps: number) {
  return Number.isInteger(Math.round(fps * 100) / 100) ? String(Math.round(fps)) : fps.toFixed(2);
}

/** What happens to the motion for this choice, mirroring the server's conversion plan. */
export function frameRateHint(choice: string, sourceFps?: number): string {
  const fps = sourceFps && sourceFps > 0 ? sourceFps : 0;
  if (choice === "source" || !choice) {
    return fps ? fill(copy.frameRateHintSourceKnown, { fps: showRate(fps) }) : copy.frameRateHintSource;
  }
  const target = Number(choice);
  if (!fps) return fill(copy.frameRateHintUnknown, { target });
  const values = { fps: showRate(fps), target };
  if (Math.abs(fps - target) < 0.01) return fill(copy.frameRateHintSame, values);
  if (target > fps) return fill(copy.frameRateHintUp, values);
  const ratio = fps / target;
  return fill(Math.abs(ratio - Math.round(ratio)) < 0.01 ? copy.frameRateHintEven : copy.frameRateHintUneven, values);
}

/** A number input value: empty stays empty while typing instead of jumping to 0. */
function numberValue(value: number) {
  return Number.isFinite(value) ? value : "";
}

export function HighlightConfigPanel({
  form,
  setForm,
  sourceFps,
}: {
  form: HighlightForm;
  setForm: (next: HighlightForm) => void;
  /** Frame rate of the uploaded video, when known, to explain what a choice will do. */
  sourceFps?: number;
}) {
  const update = (patch: Partial<HighlightForm>) => setForm({ ...form, ...patch });
  const autoLength = form.length_mode === "auto" && !form.social_short;
  const isPreset = !autoLength && PRESETS.some(([a, b]) => a === form.min_duration && b === form.max_duration);
  const [customOpen, setCustomOpen] = useState(false);
  const showCustom = customOpen || !isPreset;
  const swapped = !autoLength && form.min_duration > form.max_duration;
  const autoClipLength = form.sequence_duration_mode === "auto";
  const clipCap = autoLength ? fill(copy.capSeconds, { n: 30 }) : copy.capHighlight;

  return (
    <div className="panel config">
      <fieldset className="config-section">
        <legend>{copy.target}</legend>
        <p className="field-hint top">{copy.targetHint}</p>
        <div className="chips" role="group" aria-label={copy.target}>
          {PRESETS.map(([a, b]) => {
            const active = !customOpen && !autoLength && form.min_duration === a && form.max_duration === b;
            return (
              <button
                key={`${a}-${b}`}
                className={`chip ${active ? "active" : ""}`}
                aria-pressed={active}
                onClick={() => {
                  setCustomOpen(false);
                  update({ min_duration: a, max_duration: b, length_mode: "range" });
                }}
                type="button"
              >
                {a === b ? `${a}s` : `${a}–${b}s`}
              </button>
            );
          })}
          <button
            className={`chip ${showCustom ? "active" : ""}`}
            aria-pressed={showCustom}
            aria-expanded={showCustom}
            aria-controls="custom-length"
            type="button"
            onClick={() => setCustomOpen(true)}
          >
            {copy.custom}
          </button>
        </div>
        {showCustom && (
          <div className="form-grid reveal" id="custom-length">
            <div className="field">
              <label htmlFor="min">{copy.min}</label>
              <input
                id="min"
                type="number"
                min={1}
                inputMode="numeric"
                value={numberValue(form.min_duration)}
                onChange={(e) => update({ min_duration: e.target.value === "" ? NaN : Number(e.target.value) })}
              />
            </div>
            <div className="field">
              <label htmlFor="max">{copy.max}</label>
              <input
                id="max"
                type="number"
                min={1}
                inputMode="numeric"
                disabled={autoLength}
                aria-describedby="no-limit-hint"
                value={numberValue(form.max_duration)}
                onChange={(e) => update({ max_duration: e.target.value === "" ? NaN : Number(e.target.value) })}
              />
              <label className="check">
                <input
                  type="checkbox"
                  checked={autoLength}
                  disabled={form.social_short}
                  onChange={(e) => update({ length_mode: e.target.checked ? "auto" : "range" })}
                />
                <span>{copy.noLimit}</span>
              </label>
            </div>
            <p className="field-hint span-2" id="no-limit-hint">
              {form.social_short ? copy.noLimitSocial : autoLength ? copy.noLimitHint : ""}
            </p>
            {swapped && <p className="field-hint warn span-2">{copy.rangeSwap}</p>}
          </div>
        )}
      </fieldset>

      <fieldset className="config-section">
        <legend>{copy.style}</legend>
        <div className="chips" role="group" aria-label={copy.style}>
          {STYLES.map((style) => (
            <button
              key={style}
              type="button"
              className={`chip ${form.style === style ? "active" : ""}`}
              aria-pressed={form.style === style}
              onClick={() => update({ style })}
            >
              {messages.styles[style]}
            </button>
          ))}
        </div>
        <p className="field-hint" aria-live="polite">
          {copy.styleHints[form.style as keyof typeof copy.styleHints] || ""}
        </p>
        {form.style === "custom" && (
          <div className="field reveal">
            <label htmlFor="focus">{copy.customFocus}</label>
            <textarea
              id="focus"
              value={form.custom_focus}
              onChange={(e) => update({ custom_focus: e.target.value })}
              placeholder={copy.customFocusPlaceholder}
            />
          </div>
        )}
      </fieldset>

      <fieldset className="config-section">
        <legend>{copy.priorities}</legend>
        <div className="chips" role="group" aria-label={copy.priorities}>
          {PRIORITIES.map((key) => (
            <button
              key={key}
              type="button"
              className={`chip ${form.content_priorities.includes(key) ? "active" : ""}`}
              aria-pressed={form.content_priorities.includes(key)}
              onClick={() => {
                if (key === "balanced") {
                  update({ content_priorities: ["balanced"] });
                  return;
                }
                const withoutBalanced = form.content_priorities.filter((item) => item !== "balanced");
                const next = withoutBalanced.includes(key)
                  ? withoutBalanced.filter((item) => item !== key)
                  : [...withoutBalanced, key];
                update({ content_priorities: next.length ? next : ["balanced"] });
              }}
            >
              {messages.priorities[key]}
            </button>
          ))}
        </div>
        <p className="field-hint">{copy.prioritiesHint}</p>
        {form.content_priorities[0] !== "balanced" && form.content_priorities.length > 1 && (
          <div className="priority-weights reveal">
            {form.content_priorities.map((key) => {
              const name = messages.priorities[key as keyof typeof messages.priorities];
              const value = form.priority_weights[key] ?? 3;
              return (
                <label key={key}>
                  <span>{name}</span>
                  <span className="weight-input">
                    <small aria-hidden="true">{copy.weightLess}</small>
                    <input
                      type="range"
                      min={1}
                      max={5}
                      step={1}
                      value={value}
                      aria-valuetext={`${value} of 5`}
                      onChange={(e) =>
                        update({
                          priority_weights: { ...form.priority_weights, [key]: Number(e.target.value) },
                        })
                      }
                    />
                    <small aria-hidden="true">{copy.weightMore}</small>
                  </span>
                </label>
              );
            })}
          </div>
        )}
      </fieldset>

      <fieldset className="config-section">
        <legend>{copy.format}</legend>
        <label className="check">
          <input
            type="checkbox"
            checked={form.social_short}
            aria-describedby="social-hint"
            onChange={(e) =>
              update({
                social_short: e.target.checked,
                length_mode: e.target.checked ? "range" : form.length_mode,
                aspect_ratio: e.target.checked ? "9:16" : form.aspect_ratio,
                min_duration: e.target.checked ? Math.min(form.min_duration, 15) : form.min_duration,
                max_duration: e.target.checked ? Math.min(Math.max(form.max_duration, 15), 60) : form.max_duration,
              })
            }
          />
          <span>
            {copy.socialShort}
            <span className="field-hint block" id="social-hint">
              {copy.socialShortHint}
            </span>
          </span>
        </label>
        <div className="form-grid">
          <div className="field">
            <label htmlFor="aspect">{copy.aspect}</label>
            <select id="aspect" value={form.aspect_ratio} onChange={(e) => update({ aspect_ratio: e.target.value })}>
              <option value="original">{copy.aspectOriginal}</option>
              <option value="16:9">{copy.aspect169}</option>
              <option value="9:16">{copy.aspect916}</option>
              <option value="1:1">{copy.aspect11}</option>
              <option value="4:5">{copy.aspect45}</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="subtitles">{copy.subtitles}</label>
            <select id="subtitles" value={form.subtitles === "auto" ? "srt" : form.subtitles} onChange={(e) => update({ subtitles: e.target.value })}>
              <option value="none">{copy.subtitlesNone}</option>
              <option value="srt">{copy.subtitlesFile}</option>
              <option value="burned">{copy.subtitlesBurned}</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="transition">{copy.transition}</label>
            <select id="transition" value={form.transition} onChange={(e) => update({ transition: e.target.value })}>
              <option value="cut">{copy.transitionCut}</option>
              <option value="fade">{copy.transitionFade}</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="frameRate">{copy.frameRate}</label>
            <select
              id="frameRate"
              aria-describedby="frameRate-hint"
              value={form.frame_rate}
              onChange={(e) => update({ frame_rate: e.target.value })}
            >
              <option value="source">{copy.frameRateSource}</option>
              {FRAME_RATES.map((rate) => (
                <option key={rate} value={rate}>
                  {FRAME_RATE_LABELS[rate]}
                </option>
              ))}
            </select>
            <p className="field-hint" id="frameRate-hint" aria-live="polite">
              {frameRateHint(form.frame_rate, sourceFps)}
            </p>
          </div>
        </div>
        {form.frame_rate !== "source" && (
          <label className="check reveal">
            <input
              type="checkbox"
              checked={form.smooth_motion}
              aria-describedby="smooth-hint"
              onChange={(e) => update({ smooth_motion: e.target.checked })}
            />
            <span>
              {copy.smoothMotion}
              <span className="field-hint block" id="smooth-hint">
                {copy.smoothMotionHint}
              </span>
            </span>
          </label>
        )}
      </fieldset>

      <details className="advanced">
        <summary>{copy.advanced}</summary>

        <div className="config-section">
          <h4 className="sub-title">{copy.clipLength}</h4>
          {/* Empty means Auto. A longest value switches to fixed limits; Auto (with or without a
              shortest value) lets each clip's content decide its length. */}
          <div className="form-grid">
            <div className="field">
              <label htmlFor="seqMin">{copy.minSequence}</label>
              <input
                id="seqMin"
                type="number"
                min={0.5}
                step={0.5}
                inputMode="decimal"
                placeholder={copy.auto}
                aria-describedby="seqMin-hint"
                value={form.min_sequence_duration ?? ""}
                onChange={(e) => {
                  const raw = e.target.value.trim();
                  const value = Number(raw);
                  update({ min_sequence_duration: raw && Number.isFinite(value) ? value : null });
                }}
              />
              <p className="field-hint" id="seqMin-hint">
                {copy.shortestClipHint}
              </p>
            </div>
            <div className="field">
              <label htmlFor="seqMax">{copy.maxSequence}</label>
              <input
                id="seqMax"
                type="number"
                min={1}
                step={0.5}
                inputMode="decimal"
                placeholder={copy.auto}
                aria-describedby="seqMax-hint"
                value={autoClipLength ? "" : numberValue(form.max_sequence_duration)}
                onChange={(e) => {
                  const raw = e.target.value.trim();
                  const value = Number(raw);
                  if (!raw || !Number.isFinite(value)) {
                    update({ sequence_duration_mode: "auto" });
                    return;
                  }
                  update({ sequence_duration_mode: "custom", max_sequence_duration: value });
                }}
              />
              <p className="field-hint" id="seqMax-hint">
                {autoClipLength ? fill(copy.longestClipAuto, { cap: clipCap }) : copy.longestClipSet}
              </p>
            </div>
          </div>
        </div>

        <div className="config-section">
          <h4 className="sub-title">{copy.content}</h4>
          <div className="form-grid">
            <div className="field">
              <label htmlFor="context">{copy.preserve}</label>
              <select
                id="context"
                aria-describedby="context-hint"
                value={form.preserve_context}
                onChange={(e) => update({ preserve_context: e.target.value })}
              >
                <option value="low">{copy.preserveLow}</option>
                <option value="balanced">{copy.preserveBalanced}</option>
                <option value="high">{copy.preserveHigh}</option>
              </select>
              <p className="field-hint" id="context-hint">
                {copy.preserveHint}
              </p>
            </div>
            <div className="field">
              <label htmlFor="ordering">{copy.ordering}</label>
              <select
                id="ordering"
                value={form.order_mode === "ai" ? "chronological" : form.order_mode}
                onChange={(e) => update({ order_mode: e.target.value })}
              >
                <option value="chronological">{copy.orderChronological}</option>
                <option value="narrative">{copy.orderNarrative}</option>
              </select>
            </div>
            <div className="field">
              <label htmlFor="repetition">{copy.repetitionSensitivity}</label>
              <select id="repetition" value={form.repetition_mode} onChange={(e) => update({ repetition_mode: e.target.value })}>
                <option value="strict">{copy.repetitionStrict}</option>
                <option value="balanced">{copy.repetitionBalanced}</option>
                <option value="light">{copy.repetitionLight}</option>
              </select>
            </div>
          </div>
          <label className="check">
            <input
              type="checkbox"
              checked={form.allow_repeated_content}
              aria-describedby="repeat-hint"
              onChange={(e) =>
                update({
                  allow_repeated_content: e.target.checked,
                  remove_repetition: !e.target.checked,
                })
              }
            />
            <span>
              {copy.allowRepeat}
              <span className="field-hint block" id="repeat-hint">
                {copy.allowRepeatHint}
              </span>
            </span>
          </label>
          {form.allow_repeated_content && (
            <div className="form-grid reveal">
              <div className="field">
                <label htmlFor="repMax">{copy.maxRepeat}</label>
                <input
                  id="repMax"
                  type="number"
                  min={0}
                  step={1}
                  aria-describedby="repMax-hint"
                  value={form.max_repeated_duration}
                  onChange={(e) => update({ max_repeated_duration: Number(e.target.value) })}
                />
                <p className="field-hint" id="repMax-hint">
                  {copy.maxRepeatHint}
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="config-section">
          <div className="form-grid">
            <div className="field">
              <label htmlFor="mustInclude">{copy.mustInclude}</label>
              <input
                id="mustInclude"
                aria-describedby="range-hint"
                value={form.must_include}
                onChange={(e) => update({ must_include: e.target.value })}
                placeholder="00:32-00:38"
              />
            </div>
            <div className="field">
              <label htmlFor="mustExclude">{copy.mustExclude}</label>
              <input
                id="mustExclude"
                aria-describedby="range-hint"
                value={form.must_exclude}
                onChange={(e) => update({ must_exclude: e.target.value })}
                placeholder="01:24-01:55"
              />
            </div>
          </div>
          <p className="field-hint" id="range-hint">
            {copy.rangeHint}
          </p>
        </div>
      </details>
    </div>
  );
}

export const defaultForm = (): HighlightForm => ({
  min_duration: 20,
  max_duration: 30,
  length_mode: "range",
  style: "auto",
  custom_focus: "",
  preserve_context: "balanced",
  remove_repetition: true,
  allow_repeated_content: false,
  max_sequence_duration: 5,
  min_sequence_duration: null,
  sequence_duration_mode: "auto",
  content_priorities: ["balanced"],
  priority_weights: { action: 3, speech: 3, magic: 3, visual: 3, information: 3 },
  max_repeated_duration: 15,
  repetition_mode: "balanced",
  order_mode: "chronological",
  social_short: false,
  aspect_ratio: "original",
  subtitles: "none",
  transition: "cut",
  frame_rate: "source",
  smooth_motion: false,
  codec: "auto",
  quality: "source",
  resolution: "original",
  must_include: "",
  must_exclude: "",
  cropEnabled: false,
  crop: { x: 0.1, y: 0.1, width: 0.8, height: 0.8 },
});
