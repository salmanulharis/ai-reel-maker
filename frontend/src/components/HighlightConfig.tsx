import { messages } from "../i18n";
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

export function HighlightConfigPanel({
  form,
  setForm,
}: {
  form: HighlightForm;
  setForm: (next: HighlightForm) => void;
}) {
  const update = (patch: Partial<HighlightForm>) => setForm({ ...form, ...patch });
  const customRange = !PRESETS.some(([a, b]) => a === form.min_duration && b === form.max_duration);

  return (
    <div className="panel">
      <h3>{messages.config.target}</h3>
      <div className="chips" role="group" aria-label={messages.config.target}>
        {PRESETS.map(([a, b]) => (
          <button
            key={`${a}-${b}`}
            className={`chip ${form.min_duration === a && form.max_duration === b ? "active" : ""}`}
            onClick={() => update({ min_duration: a, max_duration: b })}
            type="button"
          >
            {a === b ? `${a}s` : `${a}–${b}s`}
          </button>
        ))}
        <button className={`chip ${customRange ? "active" : ""}`} type="button" onClick={() => update({})}>
          {messages.config.custom}
        </button>
      </div>
      <div className="grid-2" style={{ marginTop: 12 }}>
        <div className="field">
          <label htmlFor="min">{messages.config.min}</label>
          <input
            id="min"
            type="number"
            min={3}
            value={form.min_duration}
            onChange={(e) => update({ min_duration: Number(e.target.value) })}
          />
        </div>
        <div className="field">
          <label htmlFor="max">{messages.config.max}</label>
          <input
            id="max"
            type="number"
            min={3}
            value={form.max_duration}
            onChange={(e) => update({ max_duration: Number(e.target.value) })}
          />
        </div>
      </div>

      <h3 style={{ marginTop: 18 }}>{messages.config.style}</h3>
      <div className="chips">
        {STYLES.map((style) => (
          <button
            key={style}
            type="button"
            className={`chip ${form.style === style ? "active" : ""}`}
            onClick={() => update({ style })}
          >
            {messages.styles[style]}
          </button>
        ))}
      </div>
      {form.style === "custom" && (
        <div className="field" style={{ marginTop: 12 }}>
          <label htmlFor="focus">{messages.config.customFocus}</label>
          <textarea
            id="focus"
            value={form.custom_focus}
            onChange={(e) => update({ custom_focus: e.target.value })}
            placeholder="Focus on the product demo. Avoid introductions and repetitive explanations."
          />
        </div>
      )}

      <div className="grid-2" style={{ marginTop: 12 }}>
        <div className="field">
          <label htmlFor="context">{messages.config.preserve}</label>
          <select
            id="context"
            value={form.preserve_context}
            onChange={(e) => update({ preserve_context: e.target.value })}
          >
            <option value="low">Low — aggressive</option>
            <option value="balanced">Balanced</option>
            <option value="high">High — keep meaning</option>
          </select>
        </div>
      </div>

      <h3 style={{ marginTop: 18 }}>{messages.config.sequenceMode}</h3>
      <div className="chips" role="group" aria-label={messages.config.sequenceMode}>
        <button
          type="button"
          className={`chip ${form.sequence_duration_mode === "auto" ? "active" : ""}`}
          onClick={() => update({ sequence_duration_mode: "auto" })}
        >
          {messages.config.sequenceAuto}
        </button>
        <button
          type="button"
          className={`chip ${form.sequence_duration_mode === "custom" ? "active" : ""}`}
          onClick={() => update({ sequence_duration_mode: "custom" })}
        >
          {messages.config.sequenceCustom}
        </button>
      </div>
      {form.sequence_duration_mode === "auto" ? (
        <p className="field-hint">{messages.config.sequenceAutoHint}</p>
      ) : (
        <div className="field" style={{ marginTop: 12 }}>
          <label htmlFor="seqMax">{messages.config.maxSequence}</label>
          <input
            id="seqMax"
            type="number"
            min={1}
            step={1}
            value={form.max_sequence_duration}
            onChange={(e) => update({ max_sequence_duration: Number(e.target.value) })}
          />
          <p className="field-hint">{messages.config.maxSequenceHint}</p>
        </div>
      )}

      <h3 style={{ marginTop: 18 }}>{messages.config.priorities}</h3>
      <div className="chips" role="group" aria-label={messages.config.priorities}>
        {PRIORITIES.map((key) => (
          <button
            key={key}
            type="button"
            className={`chip ${form.content_priorities.includes(key) ? "active" : ""}`}
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
      <p className="field-hint">{messages.config.prioritiesHint}</p>
      {form.content_priorities[0] !== "balanced" && form.content_priorities.length > 1 && (
        <div className="priority-weights">
          {form.content_priorities.map((key) => (
            <label key={key}>
              {messages.priorities[key as keyof typeof messages.priorities]}
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                value={form.priority_weights[key] ?? 3}
                onChange={(e) =>
                  update({
                    priority_weights: { ...form.priority_weights, [key]: Number(e.target.value) },
                  })
                }
              />
            </label>
          ))}
        </div>
      )}

      <label className="check">
        <input
          type="checkbox"
          checked={form.allow_repeated_content}
          onChange={(e) =>
            update({
              allow_repeated_content: e.target.checked,
              remove_repetition: !e.target.checked,
            })
          }
        />
        {messages.config.allowRepeat}
      </label>
      <p className="field-hint">{messages.config.allowRepeatHint}</p>

      <div className="field" style={{ marginTop: 12 }}>
        <label htmlFor="repMax">{messages.config.maxRepeat}</label>
        <input
          id="repMax"
          type="number"
          min={0}
          step={1}
          disabled={!form.allow_repeated_content}
          value={form.max_repeated_duration}
          onChange={(e) => update({ max_repeated_duration: Number(e.target.value) })}
        />
        <p className="field-hint">{messages.config.maxRepeatHint}</p>
      </div>

      <label className="check">
        <input
          type="checkbox"
          checked={form.social_short}
          onChange={(e) =>
            update({
              social_short: e.target.checked,
              aspect_ratio: e.target.checked ? "9:16" : form.aspect_ratio,
              min_duration: e.target.checked ? Math.min(form.min_duration, 15) : form.min_duration,
              max_duration: e.target.checked ? Math.min(Math.max(form.max_duration, 15), 60) : form.max_duration,
            })
          }
        />
        {messages.config.socialShort} (9:16, 15–60s, strong hook)
      </label>

      <details className="advanced">
        <summary>{messages.config.advanced}</summary>
        <div className="grid-2">
          <div className="field">
            <label>{messages.config.ordering}</label>
            <select value={form.order_mode} onChange={(e) => update({ order_mode: e.target.value })}>
              <option value="chronological">Chronological</option>
              <option value="narrative">Narrative</option>
              <option value="ai">AI Auto</option>
            </select>
          </div>
          <div className="field">
            <label>{messages.config.aspect}</label>
            <select value={form.aspect_ratio} onChange={(e) => update({ aspect_ratio: e.target.value })}>
              <option value="original">Original</option>
              <option value="16:9">16:9</option>
              <option value="9:16">9:16</option>
              <option value="1:1">1:1</option>
              <option value="4:5">4:5</option>
            </select>
          </div>
          <div className="field">
            <label>{messages.config.subtitles}</label>
            <select value={form.subtitles} onChange={(e) => update({ subtitles: e.target.value })}>
              <option value="none">No subtitles</option>
              <option value="auto">Auto subtitles (SRT)</option>
              <option value="burned">Burned-in subtitles</option>
              <option value="srt">Export SRT</option>
            </select>
          </div>
          <div className="field">
            <label>Transition</label>
            <select value={form.transition} onChange={(e) => update({ transition: e.target.value })}>
              <option value="cut">Hard cut</option>
              <option value="fade">Short fade</option>
            </select>
          </div>
          <div className="field">
            <label>{messages.config.repetitionSensitivity}</label>
            <select
              value={form.repetition_mode}
              onChange={(e) => update({ repetition_mode: e.target.value })}
            >
              <option value="strict">Strict</option>
              <option value="balanced">Balanced</option>
              <option value="light">Light</option>
            </select>
          </div>
        </div>
        <div className="field">
          <label>{messages.config.mustInclude} (00:32–00:38)</label>
          <input
            value={form.must_include}
            onChange={(e) => update({ must_include: e.target.value })}
            placeholder="00:32-00:38"
          />
        </div>
        <div className="field">
          <label>{messages.config.mustExclude}</label>
          <input
            value={form.must_exclude}
            onChange={(e) => update({ must_exclude: e.target.value })}
            placeholder="01:24-01:55"
          />
        </div>
      </details>
    </div>
  );
}

export const defaultForm = (): HighlightForm => ({
  min_duration: 20,
  max_duration: 30,
  style: "auto",
  custom_focus: "",
  preserve_context: "balanced",
  remove_repetition: true,
  allow_repeated_content: false,
  max_sequence_duration: 5,
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
  codec: "auto",
  quality: "source",
  resolution: "original",
  must_include: "",
  must_exclude: "",
  cropEnabled: false,
  crop: { x: 0.1, y: 0.1, width: 0.8, height: 0.8 },
});
