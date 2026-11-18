import { useEffect, useRef, useState } from "react";
import { api } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, InlineStatus, Spinner } from "./Feedback";

const copy = messages.review.music;

interface Props {
  jobId: string;
  volume: number;
  duck: boolean;
}

/** Background music for the video: a file of your own, its volume, and lowering it under speech. */
export function MusicPanel({ jobId, volume: initialVolume, duck: initialDuck }: Props) {
  const [name, setName] = useState<string | null>(null);
  const [volume, setVolume] = useState(initialVolume);
  const [duck, setDuck] = useState(initialDuck);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const timer = useRef<number>();

  useEffect(() => {
    api
      .music(jobId)
      .then((data) => setName(data.name))
      .catch(() => undefined);
  }, [jobId]);

  function saveOptions(next: { music_volume?: number; music_duck?: boolean }) {
    setSaved(false);
    window.clearTimeout(timer.current);
    // Saved shortly after the last change, so dragging the slider sends one request.
    timer.current = window.setTimeout(() => {
      api
        .renderOptions(jobId, next)
        .then(() => setSaved(true))
        .catch((err) => setError((err as Error).message));
    }, 400);
  }

  async function upload(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      setName((await api.uploadMusic(jobId, file)).name);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  async function remove() {
    setBusy(true);
    setError("");
    try {
      await api.deleteMusic(jobId);
      setName(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel music-panel" aria-labelledby="music-title">
      <h3 id="music-title" className="sub-title">
        {copy.title}
      </h3>
      <p className="field-hint">{name ? fill(copy.current, { name }) : copy.hint}</p>
      <div className="btn-row">
        <label className="btn sm">
          {busy && <Spinner />}
          {name ? copy.replace : copy.add}
          <input
            ref={input}
            type="file"
            accept="audio/*,.mp3,.wav,.m4a,.aac,.ogg,.flac,.opus"
            className="sr-only"
            disabled={busy}
            onChange={(e) => void upload(e.target.files?.[0])}
          />
        </label>
        {name && (
          <button type="button" className="btn sm ghost" disabled={busy} onClick={remove}>
            {copy.remove}
          </button>
        )}
      </div>
      {name && (
        <div className="music-options">
          <label htmlFor="music-volume">{fill(copy.volume, { n: Math.round(volume) })}</label>
          <input
            id="music-volume"
            type="range"
            min={5}
            max={100}
            step={5}
            value={volume}
            onChange={(e) => {
              const value = Number(e.target.value);
              setVolume(value);
              saveOptions({ music_volume: value });
            }}
          />
          <label className="check">
            <input
              type="checkbox"
              checked={duck}
              onChange={(e) => {
                setDuck(e.target.checked);
                saveOptions({ music_duck: e.target.checked });
              }}
            />
            <span>{copy.duck}</span>
          </label>
          {saved && <InlineStatus ok>{copy.saved}</InlineStatus>}
        </div>
      )}
      {error && <Alert>{error}</Alert>}
    </section>
  );
}
