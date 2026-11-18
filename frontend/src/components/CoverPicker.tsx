import { useMemo, useState } from "react";
import { api } from "../services/api";
import { coverSuggestions, coverUrl } from "../services/cover";
import { formatPrecise } from "../services/clipPreview";
import { fill, messages } from "../i18n";
import { Alert, InlineStatus, Spinner } from "./Feedback";
import type { Clip } from "../types";

const copy = messages.cover;
const PREVIEW_WIDTH = 320;

interface Props {
  jobId: string;
  clips: Clip[];
  /** Where the highlight player is, in seconds (undefined when unknown). */
  getPlayerTime: () => number | undefined;
}

/** Choose a frame of the finished highlight and save it as a cover (thumbnail) image. */
export function CoverPicker({ jobId, clips, getPlayerTime }: Props) {
  const suggestions = useMemo(() => coverSuggestions(clips), [clips]);
  const [fromPlayer, setFromPlayer] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(suggestions[0] ?? null);
  const [broken, setBroken] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState("");
  const [error, setError] = useState("");

  const options = [
    ...(fromPlayer !== null ? [fromPlayer] : []),
    ...suggestions.filter((time) => time !== fromPlayer),
  ].filter((time) => !broken.includes(time));

  function choose(time: number) {
    setSelected(time);
    setSaved("");
    setError("");
  }

  function usePlayer() {
    const time = getPlayerTime();
    if (time === undefined || !Number.isFinite(time)) return;
    const rounded = Math.round(time * 10) / 10;
    setFromPlayer(rounded);
    setBroken((list) => list.filter((item) => item !== rounded));
    choose(rounded);
  }

  async function save() {
    if (selected === null) return;
    setBusy(true);
    setError("");
    try {
      setSaved(await api.downloadCover(jobId, selected));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel cover-picker" aria-labelledby="cover-title">
      <h3 id="cover-title" className="sub-title">
        {copy.title}
      </h3>
      <p className="field-hint">{copy.hint}</p>
      {options.length > 0 ? (
        <div className="cover-grid" role="group" aria-label={copy.frames}>
          {options.map((time) => {
            const active = selected === time;
            const label = time === fromPlayer ? fill(copy.fromPlayer, { time: formatPrecise(time) }) : formatPrecise(time);
            return (
              <button
                key={time}
                type="button"
                className={`cover-option ${active ? "selected" : ""}`}
                aria-pressed={active}
                aria-label={fill(copy.frameAt, { time: formatPrecise(time) })}
                onClick={() => choose(time)}
              >
                <img
                  src={coverUrl(jobId, time, PREVIEW_WIDTH)}
                  alt=""
                  loading="lazy"
                  onError={() => setBroken((list) => [...list, time])}
                />
                <span>{label}</span>
              </button>
            );
          })}
        </div>
      ) : (
        <p className="field-hint">{copy.empty}</p>
      )}
      <div className="btn-row cover-actions">
        <button type="button" className="btn sm" onClick={usePlayer}>
          {copy.usePlayer}
        </button>
        <button
          type="button"
          className="btn sm primary"
          disabled={selected === null || busy || broken.includes(selected)}
          aria-busy={busy}
          onClick={save}
        >
          {busy && <Spinner />}
          {copy.download}
        </button>
        {saved && <InlineStatus ok>{fill(copy.saved, { name: saved })}</InlineStatus>}
        {error && <Alert>{error}</Alert>}
      </div>
    </section>
  );
}
