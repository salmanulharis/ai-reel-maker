import { useEffect, useRef, useState } from "react";
import { api } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, Spinner } from "./Feedback";
import { TransferProgress } from "./Progress";
import { useUpload } from "../hooks/useUpload";
import { isAbort } from "../services/transfer";
import type { HighlightForm } from "../types";

const copy = messages.musicLed;

/**
 * Create page: cut the video to the user's music. The music is uploaded to the job before the
 * analysis, which reads its beat; the plan then follows the music's length and pace.
 */
export function MusicLedPanel({
  form,
  setForm,
  jobId,
}: {
  form: HighlightForm;
  setForm: (next: HighlightForm) => void;
  jobId: string | null;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const sending = useUpload();
  const formRef = useRef(form);
  formRef.current = form;

  // The video was replaced (a new job): the music it has, if any.
  useEffect(() => {
    if (!jobId) return;
    let stop = false;
    api
      .music(jobId)
      .then((data) => {
        if (!stop && (data.name || "") !== formRef.current.music_file_name) {
          setForm({ ...formRef.current, music_file_name: data.name || "" });
        }
      })
      .catch(() => undefined);
    return () => {
      stop = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  async function upload(file: File | undefined) {
    if (!file || !jobId) return;
    setBusy(true);
    setError("");
    try {
      const result = await sending.run(file.size, (options) => api.uploadMusic(jobId, file, options));
      setForm({ ...formRef.current, music_file_name: result.name });
    } catch (err) {
      if (!isAbort(err)) setError((err as Error).message);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  return (
    <section className="panel" aria-labelledby="music-led-title">
      <label className="check">
        <input
          type="checkbox"
          checked={form.music_led}
          aria-describedby="music-led-hint"
          onChange={(e) => setForm({ ...form, music_led: e.target.checked })}
        />
        <strong id="music-led-title">{copy.title}</strong>
      </label>
      <p className="field-hint" id="music-led-hint">
        {copy.hint}
      </p>
      {form.music_led && (
        <div className="music-led-body">
          <p className="field-hint">
            {form.music_file_name ? fill(copy.current, { name: form.music_file_name }) : jobId ? copy.needMusic : copy.addVideoFirst}
          </p>
          <div className="btn-row">
            <label className={`btn sm${!jobId ? " disabled" : ""}`} aria-disabled={!jobId}>
              {busy && <Spinner />}
              {form.music_file_name ? copy.replace : copy.add}
              <input
                ref={input}
                type="file"
                accept="audio/*,.mp3,.wav,.m4a,.aac,.ogg,.flac,.opus"
                className="sr-only"
                disabled={busy || !jobId}
                onChange={(e) => void upload(e.target.files?.[0])}
              />
            </label>
            {sending.status && (
              <button type="button" className="btn sm ghost" onClick={sending.cancel}>
                {messages.transfer.cancelUpload}
              </button>
            )}
          </div>
          {sending.status && <TransferProgress status={sending.status} label={messages.transfer.uploadLabel} />}
          <fieldset className="field">
            <legend>{copy.sound}</legend>
            {(["lower", "mute"] as const).map((choice) => (
              <label key={choice} className="check">
                <input
                  type="radio"
                  name="music-led-sound"
                  checked={form.music_led_sound === choice}
                  onChange={() => setForm({ ...form, music_led_sound: choice })}
                />
                <span>{choice === "mute" ? copy.soundMute : copy.soundLower}</span>
              </label>
            ))}
          </fieldset>
          <p className="field-hint">{copy.note}</p>
        </div>
      )}
      {error && <Alert>{error}</Alert>}
    </section>
  );
}
