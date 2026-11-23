import { useEffect, useState } from "react";
import { Alert, Spinner } from "./Feedback";
import { api } from "../services/api";
import { messages } from "../i18n";
import type { AiPlanInfo } from "../types";

const copy = messages.prompt;

/**
 * What the AI understood and chose (prompt mode), and a box to ask for a change. A change plans
 * the highlight again with the AI editor without analyzing the video again; on a highlight made
 * with manual options it becomes the description.
 */
export function AiPlanPanel({
  jobId,
  version,
  disabled,
  onRefining,
}: {
  jobId: string;
  /** Changes when the plan changes, to load the summary again. */
  version: number;
  disabled: boolean;
  onRefining: () => void;
}) {
  const [info, setInfo] = useState<AiPlanInfo | null>(null);
  const [change, setChange] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.aiPlan(jobId).then(setInfo).catch(() => setInfo(null));
  }, [jobId, version]);

  async function refine() {
    setBusy(true);
    setError("");
    try {
      await api.refine(jobId, change.trim());
      onRefining();
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  const prompt = info?.mode === "prompt";
  const plan = info?.plan;
  return (
    <div className="panel ai-plan">
      <h3 className="sub-title">{prompt ? copy.planTitle : copy.askTitle}</h3>
      {prompt && info?.prompt && (
        <p>
          <span className="muted">{copy.yourRequest}</span> “{info.prompt}”
        </p>
      )}
      {plan?.understanding && (
        <p>
          <span className="muted">{copy.planSays}</span> {plan.understanding}
        </p>
      )}
      {plan?.left_out && (
        <p>
          <span className="muted">{copy.leftOut}</span> {plan.left_out}
        </p>
      )}
      {prompt && !plan && <p className="field-hint">{copy.noPlan}</p>}
      {(info?.refine_notes.length ?? 0) > 0 && (
        <>
          <span className="field-label">{copy.changesSoFar}</span>
          <ol className="refine-notes">
            {info?.refine_notes.map((note, index) => <li key={`${index}-${note}`}>{note}</li>)}
          </ol>
        </>
      )}
      <div className="field">
        <label htmlFor="ai-change">{copy.askLabel}</label>
        <textarea
          id="ai-change"
          rows={2}
          maxLength={500}
          value={change}
          placeholder={copy.askPlaceholder}
          aria-describedby="ai-change-hint"
          onChange={(e) => setChange(e.target.value)}
        />
        <p className="field-hint" id="ai-change-hint">
          {prompt ? copy.askHint : copy.askHintManual}
        </p>
      </div>
      <div className="btn-row">
        <button className="btn" type="button" disabled={disabled || busy || !change.trim()} aria-busy={busy} onClick={refine}>
          {busy && <Spinner />}
          {copy.askButton}
        </button>
      </div>
      {error && <Alert>{error}</Alert>}
    </div>
  );
}
