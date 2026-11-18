import { useEffect, useState, type FormEvent } from "react";
import { api } from "../services/api";
import type { Preset } from "../services/presets";
import { messages } from "../i18n";
import { Alert, InlineStatus, Spinner } from "./Feedback";

const copy = messages.watch;

/** Settings for the watch folder: new videos there become highlights automatically. */
export function WatchFolderPanel() {
  const [enabled, setEnabled] = useState(false);
  const [folder, setFolder] = useState("");
  const [preset, setPreset] = useState("");
  const [presets, setPresets] = useState<Preset[]>([]);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .watchFolder()
      .then((data) => {
        setEnabled(data.enabled);
        setFolder(data.folder);
        setPreset(data.preset);
      })
      .catch(() => undefined);
    api
      .presets()
      .then((data) => setPresets(data.presets || []))
      .catch(() => undefined);
  }, []);

  async function save(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setSaved(false);
    setError("");
    try {
      const data = await api.saveWatchFolder({ enabled, folder, preset });
      setFolder(data.folder);
      setSaved(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="panel watch-panel" onSubmit={save} aria-labelledby="watch-title">
      <h3 id="watch-title" className="sub-title">
        {copy.title}
      </h3>
      <p className="field-hint">{copy.hint}</p>
      <label className="check">
        <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
        <span>{copy.enable}</span>
      </label>
      <div className="form-grid">
        <div className="field">
          <label htmlFor="watch-folder">{copy.folder}</label>
          <input
            id="watch-folder"
            value={folder}
            placeholder={copy.folderPlaceholder}
            onChange={(e) => setFolder(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="watch-preset">{copy.preset}</label>
          <select id="watch-preset" value={preset} onChange={(e) => setPreset(e.target.value)}>
            <option value="">{copy.presetDefault}</option>
            {presets.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      <p className="field-hint">{copy.output}</p>
      <div className="btn-row">
        <button type="submit" className="btn sm primary" disabled={busy} aria-busy={busy}>
          {busy && <Spinner />}
          {copy.save}
        </button>
        {saved && <InlineStatus ok>{copy.saved}</InlineStatus>}
      </div>
      {error && <Alert>{error}</Alert>}
    </form>
  );
}
