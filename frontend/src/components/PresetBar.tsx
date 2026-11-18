import { useEffect, useRef, useState, type FormEvent } from "react";
import { api } from "../services/api";
import { applyPreset, findPreset, presetPayload, type Preset } from "../services/presets";
import { fill, messages } from "../i18n";
import { Alert, InlineStatus, Spinner } from "./Feedback";
import type { HighlightForm } from "../types";

const copy = messages.presets;
const MAX_NAME = 60;

/** "My presets": apply, save and delete named sets of Create-page choices. */
export function PresetBar({ form, setForm }: { form: HighlightForm; setForm: (next: HighlightForm) => void }) {
  const [presets, setPresets] = useState<Preset[] | null>(null);
  const [selected, setSelected] = useState("");
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const nameInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .presets()
      .then((data) => setPresets(data.presets || []))
      .catch(() => setPresets([]));
  }, []);

  useEffect(() => {
    if (naming) nameInput.current?.focus();
  }, [naming]);

  function apply(value: string) {
    setSelected(value);
    setError("");
    const preset = presets?.find((item) => item.name === value);
    if (!preset) return;
    setForm(applyPreset(form, preset.form));
    setStatus(fill(copy.applied, { name: preset.name }));
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    const clean = name.trim().replace(/\s+/g, " ");
    if (!clean) {
      setError(copy.nameMissing);
      return;
    }
    const existing = findPreset(presets || [], clean);
    if (existing && !window.confirm(fill(copy.confirmReplace, { name: existing.name }))) return;
    setBusy(true);
    setError("");
    try {
      const data = await api.savePreset(clean, presetPayload(form));
      setPresets(data.presets);
      setSelected(findPreset(data.presets, clean)?.name || "");
      setNaming(false);
      setName("");
      setStatus(fill(copy.saved, { name: clean }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!selected || !window.confirm(fill(copy.confirmDelete, { name: selected }))) return;
    setBusy(true);
    setError("");
    try {
      const data = await api.deletePreset(selected);
      setPresets(data.presets);
      setStatus(fill(copy.deleted, { name: selected }));
      setSelected("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (presets === null) return null;

  return (
    <div className="preset-bar">
      <div className="preset-row">
        {presets.length > 0 ? (
          <div className="field">
            <label htmlFor="preset-select">{copy.label}</label>
            <select id="preset-select" value={selected} disabled={busy} onChange={(e) => apply(e.target.value)}>
              <option value="">{copy.choose}</option>
              {presets.map((preset) => (
                <option key={preset.name} value={preset.name}>
                  {preset.name}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <p className="field-hint">{copy.empty}</p>
        )}
        <div className="btn-row">
          {!naming && (
            <button
              type="button"
              className="btn sm"
              disabled={busy}
              onClick={() => {
                setName(selected);
                setNaming(true);
                setStatus("");
              }}
            >
              {copy.save}
            </button>
          )}
          {selected && !naming && (
            <button type="button" className="btn sm ghost" disabled={busy} onClick={remove}>
              {copy.delete}
            </button>
          )}
        </div>
      </div>
      {naming && (
        <form className="preset-save" onSubmit={save}>
          <div className="field">
            <label htmlFor="preset-name">{copy.nameLabel}</label>
            <input
              id="preset-name"
              ref={nameInput}
              value={name}
              maxLength={MAX_NAME}
              placeholder={copy.namePlaceholder}
              aria-describedby="preset-name-hint"
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") setNaming(false);
              }}
            />
            <p id="preset-name-hint" className="field-hint">
              {copy.nameHint}
            </p>
          </div>
          <div className="btn-row">
            <button type="submit" className="btn sm primary" disabled={busy} aria-busy={busy}>
              {busy && <Spinner />}
              {copy.confirmSave}
            </button>
            <button type="button" className="btn sm ghost" disabled={busy} onClick={() => setNaming(false)}>
              {copy.cancel}
            </button>
          </div>
        </form>
      )}
      {status && !error && <InlineStatus ok>{status}</InlineStatus>}
      {error && <Alert>{error}</Alert>}
    </div>
  );
}
