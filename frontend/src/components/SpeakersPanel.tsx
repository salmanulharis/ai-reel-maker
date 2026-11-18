import { useState } from "react";
import { api, type SpeakerData } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, InlineStatus, PlayIcon, Spinner } from "./Feedback";
import type { JobSource, Moment } from "../types";

const copy = messages.review.speakers;

interface Props {
  jobId: string;
  sources: JobSource[];
  data: SpeakerData | null;
  focus: string;
  busy: boolean;
  onLoaded: (data: SpeakerData) => void;
  onFocus: (speakerId: string) => void;
  onPreview: (moment: Moment, key: string) => void;
}

/** "Speaker A" (or "Video 2 · Speaker A" with several videos). */
export function speakerLabel(id: string, sources: JobSource[]): string {
  const [sourceId, letter] = id.split(":");
  const name = fill(copy.speaker, { letter: letter || id });
  if (sources.length < 2) return name;
  const index = sources.findIndex((source) => source.id === sourceId);
  return `${fill(messages.upload.video, { n: index + 1 })} · ${name}`;
}

/** Who speaks: find the speakers, hear a sample of each, and choose moments of one of them. */
export function SpeakersPanel({ jobId, sources, data, focus, busy, onLoaded, onFocus, onPreview }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function find() {
    setLoading(true);
    setError("");
    try {
      onLoaded(await api.speakers(jobId));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const total = (data?.speakers || []).reduce((sum, s) => sum + s.seconds, 0) || 1;

  return (
    <section className="panel speakers-panel" aria-labelledby="speakers-title">
      <h3 id="speakers-title" className="sub-title">
        {copy.title}
      </h3>
      <p className="field-hint">{copy.hint}</p>
      {!data && (
        <button type="button" className="btn sm" disabled={loading} aria-busy={loading} onClick={find}>
          {loading && <Spinner />}
          {loading ? copy.finding : copy.find}
        </button>
      )}
      {data && data.speakers.length === 0 && <p className="field-hint">{copy.none}</p>}
      {data && data.speakers.length > 0 && (
        <>
          <ul className="speaker-list">
            {data.speakers.map((speaker) => {
              const label = speakerLabel(speaker.id, sources);
              const share = Math.round((100 * speaker.seconds) / total);
              return (
                <li key={speaker.id} className={focus === speaker.id ? "selected" : ""}>
                  <strong>{label}</strong>
                  <span className="muted">{fill(copy.share, { n: share })}</span>
                  {speaker.sample && (
                    <button
                      type="button"
                      className="btn sm ghost"
                      aria-label={fill(copy.sampleLabel, { name: label })}
                      onClick={() =>
                        onPreview(
                          { ...speaker.sample!, source_id: speaker.source_id, reason: label },
                          `speaker-${speaker.id}`,
                        )
                      }
                    >
                      <PlayIcon size={14} />
                      {copy.sample}
                    </button>
                  )}
                  <button
                    type="button"
                    className={`btn sm ${focus === speaker.id ? "primary" : ""}`}
                    disabled={busy || focus === speaker.id}
                    onClick={() => onFocus(speaker.id)}
                  >
                    {focus === speaker.id ? copy.focused : copy.focus}
                  </button>
                </li>
              );
            })}
          </ul>
          {focus && (
            <button type="button" className="btn sm" disabled={busy} onClick={() => onFocus("")}>
              {copy.all}
            </button>
          )}
          {data.speakers.length === 1 && <p className="field-hint">{copy.one}</p>}
          {busy && <InlineStatus busy>{copy.choosing}</InlineStatus>}
        </>
      )}
      {error && <Alert>{error}</Alert>}
    </section>
  );
}
