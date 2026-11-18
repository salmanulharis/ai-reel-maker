import { useState } from "react";
import { api } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, InlineStatus, Spinner } from "./Feedback";

const copy = messages.editorExport;

interface Props {
  jobId: string;
  disabled?: boolean;
  /** Runs first, for example to save clip changes that are still being saved. */
  beforeExport?: () => Promise<unknown>;
  className?: string;
}

/** "Download for video editors": the kept clips as a Premiere / Resolve / Final Cut timeline. */
export function EditorExport({ jobId, disabled, beforeExport, className = "btn" }: Props) {
  const [state, setState] = useState<"" | "busy" | "done">("");
  const [saved, setSaved] = useState("");
  const [error, setError] = useState("");

  async function run() {
    setState("busy");
    setError("");
    try {
      if (beforeExport) await beforeExport();
      setSaved(await api.exportForEditors(jobId));
      setState("done");
    } catch (err) {
      setError((err as Error).message);
      setState("");
    }
  }

  return (
    <>
      <button
        type="button"
        className={className}
        disabled={disabled || state === "busy"}
        aria-busy={state === "busy"}
        aria-describedby="editor-export-hint"
        title={copy.hint}
        onClick={run}
      >
        {state === "busy" && <Spinner />}
        {state === "busy" ? copy.busy : copy.button}
      </button>
      <span id="editor-export-hint" className="sr-only">
        {copy.hint}
      </span>
      {state === "done" && <InlineStatus ok>{fill(copy.done, { name: saved })}</InlineStatus>}
      {error && <Alert>{error}</Alert>}
    </>
  );
}
