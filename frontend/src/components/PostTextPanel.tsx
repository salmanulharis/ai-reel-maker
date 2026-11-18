import { useEffect, useState } from "react";
import { api, type PostText } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, InlineStatus, Spinner } from "./Feedback";

const copy = messages.postText;

/** Suggested titles, description and hashtags for posting the highlight, with copy buttons. */
export function PostTextPanel({ jobId }: { jobId: string }) {
  const [suggestion, setSuggestion] = useState<PostText | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState("");

  useEffect(() => {
    api
      .postText(jobId)
      .then((data) => setSuggestion(data.suggestion))
      .catch(() => undefined);
  }, [jobId]);

  async function make() {
    setBusy(true);
    setError("");
    setCopied("");
    try {
      setSuggestion((await api.makePostText(jobId)).suggestion);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function copyText(key: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(key);
    } catch {
      setError(copy.copyFailed);
    }
  }

  const hashtags = suggestion?.hashtags.map((tag) => `#${tag}`).join(" ") || "";

  return (
    <section className="panel post-text" aria-labelledby="post-text-title">
      <h3 id="post-text-title" className="sub-title">
        {copy.title}
      </h3>
      <p className="field-hint">{copy.hint}</p>
      <div className="btn-row">
        <button type="button" className="btn sm" disabled={busy} aria-busy={busy} onClick={make}>
          {busy && <Spinner />}
          {suggestion ? copy.again : copy.make}
        </button>
        {busy && <span className="muted">{copy.busy}</span>}
      </div>
      {error && <Alert>{error}</Alert>}
      {suggestion && (
        <div className="post-text-body">
          {suggestion.source === "simple" && (
            <p className="field-hint">{fill(copy.simple, { note: suggestion.note || "" }).trim()}</p>
          )}
          {suggestion.titles.length > 0 && (
            <>
              <p className="section-label">{copy.titles}</p>
              <ul className="post-text-list">
                {suggestion.titles.map((title, i) => (
                  <li key={title}>
                    <span>{title}</span>
                    <button
                      type="button"
                      className="btn sm ghost"
                      aria-label={fill(copy.copyTitle, { n: i + 1 })}
                      onClick={() => copyText(`title-${i}`, title)}
                    >
                      {copied === `title-${i}` ? copy.copied : copy.copy}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
          {suggestion.description && (
            <>
              <p className="section-label">{copy.description}</p>
              <div className="post-text-item">
                <p>{suggestion.description}</p>
                <button
                  type="button"
                  className="btn sm ghost"
                  aria-label={copy.copyDescription}
                  onClick={() => copyText("description", suggestion.description)}
                >
                  {copied === "description" ? copy.copied : copy.copy}
                </button>
              </div>
            </>
          )}
          {hashtags && (
            <>
              <p className="section-label">{copy.hashtags}</p>
              <div className="post-text-item">
                <p>{hashtags}</p>
                <button
                  type="button"
                  className="btn sm ghost"
                  aria-label={copy.copyHashtags}
                  onClick={() => copyText("hashtags", hashtags)}
                >
                  {copied === "hashtags" ? copy.copied : copy.copy}
                </button>
              </div>
            </>
          )}
          {copied && (
            <span className="sr-only">
              <InlineStatus ok>{copy.copied}</InlineStatus>
            </span>
          )}
        </div>
      )}
    </section>
  );
}
