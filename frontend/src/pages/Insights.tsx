import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, formatBytes, formatElapsed, formatTime } from "../services/api";
import { fill, messages } from "../i18n";
import { Alert, PageLoader } from "../components/Feedback";
import type { JobInsights, LogEntry } from "../types";

const copy = messages.insights;
const STAGE_LABELS: Record<string, string> = messages.progress.stages;
const UNITS: Record<string, string> = messages.progress.units;

function stepLabel(name: string, fallback: string): string {
  return STAGE_LABELS[name] || fallback;
}

function percent(share: number): string {
  if (share <= 0) return "0%";
  return share < 0.01 ? "<1%" : `${Math.round(share * 100)}%`;
}

function number(value: number): string {
  return Math.round(value).toLocaleString();
}

/** Processing details of one highlight: where the time went, and why. */
export default function Insights() {
  const { id } = useParams();
  const [data, setData] = useState<JobInsights | null>(null);
  const [error, setError] = useState("");
  const [logs, setLogs] = useState<LogEntry[] | null>(null);
  const [logFilter, setLogFilter] = useState<"all" | "warnings">("all");

  useEffect(() => {
    if (!id) return;
    api
      .insights(id)
      .then(setData)
      .catch((err) => setError((err as Error).message));
  }, [id]);

  function loadLogs() {
    if (!id || logs) return;
    api
      .logs(id)
      .then((result) => setLogs(result.logs || []))
      .catch((err) => setError((err as Error).message));
  }

  if (!id) return null;
  if (!data) return error ? <Alert>{error}</Alert> : <PageLoader />;

  const { totals, video, steps, render, ai, run, bottlenecks, warnings } = data;
  const longest = Math.max(1, ...steps.map((step) => step.seconds));
  const slowest = new Set(data.slowest.slice(0, 1));
  const back = data.job.status === "completed" ? `/jobs/${id}/result` : `/jobs/${id}/analysis`;
  const running = ["queued", "analyzing", "rendering"].includes(data.job.status);
  const shownLogs = (logs || []).filter((entry) => logFilter === "all" || entry.level !== "INFO");
  const output = data.result.outputs.reduce((sum, item) => sum + (Number(item.duration) || 0), 0);

  return (
    <div className="insights">
      <div className="insights-head">
        <div>
          <h3 className="insights-file" title={data.job.filename}>
            {data.job.filename}
          </h3>
          <p className="field-hint">{copy.intro}</p>
        </div>
        <Link className="btn" to={back}>
          {data.job.status === "completed" ? copy.backToVideo : copy.backToProgress}
        </Link>
      </div>
      {error && <Alert>{error}</Alert>}
      {running && <Alert kind="info">{copy.stillRunning}</Alert>}

      <div className="meta-grid insights-summary">
        <div>
          <span>{copy.total}</span>
          <strong className="tabular">{formatElapsed(totals.total_seconds)}</strong>
        </div>
        <div>
          <span>{copy.analysis}</span>
          <span className="value tabular">{formatElapsed(totals.analysis_seconds)}</span>
        </div>
        <div>
          <span>{copy.render}</span>
          <span className="value tabular">{totals.render_seconds > 0 ? formatElapsed(totals.render_seconds) : "—"}</span>
        </div>
        <div title={copy.ratioHint}>
          <span>{copy.ratio}</span>
          <span className="value tabular">{totals.ratio ? fill(copy.ratioValue, { ratio: totals.ratio.toFixed(1) }) : "—"}</span>
        </div>
        <div>
          <span>{copy.video}</span>
          <span className="value">
            {formatTime(video.duration)}
            {video.width ? ` · ${video.width}×${video.height}` : ""}
            {video.codec ? ` · ${video.codec.toUpperCase()}` : ""}
          </span>
        </div>
        <div>
          <span>{copy.result}</span>
          <span className="value">
            {output > 0 ? `${formatTime(output)} · ` : ""}
            {fill(copy.clips, { n: data.result.clips })}
          </span>
        </div>
      </div>

      <section className="panel" aria-labelledby="time-title">
        <h3 id="time-title">{copy.whereTime}</h3>
        <p className="panel-intro">{copy.whereTimeHint}</p>
        {steps.length === 0 ? (
          <p className="field-hint">{copy.noSteps}</p>
        ) : (
          <ol className="time-bars">
            {steps.map((step) => {
              const label = stepLabel(step.name, step.label);
              const count =
                step.total > 0 && step.unit && step.unit !== "time"
                  ? fill(messages.progress.stepCount, {
                      done: number(Math.min(step.done, step.total)),
                      total: number(step.total),
                      unit: UNITS[step.unit] || step.unit,
                    })
                  : "";
              const top = slowest.has(step.name);
              return (
                <li key={step.name} className={top ? "slowest" : undefined}>
                  <div className="time-bar-label">
                    <span>{label}</span>
                    {top && <span className="status-pill accent">{copy.slowest}</span>}
                    {step.status === "error" && <span className="status-pill err">{copy.failedStep}</span>}
                    {count && <span className="field-hint">{count}</span>}
                  </div>
                  <div
                    className="time-bar-track"
                    role="img"
                    aria-label={fill(copy.barLabel, { step: label, time: formatElapsed(step.seconds), share: percent(step.share) })}
                    title={`${label}: ${formatElapsed(step.seconds)} (${percent(step.share)})`}
                  >
                    <span className="time-bar" style={{ width: `${Math.max(0.5, (step.seconds / longest) * 100)}%` }} />
                  </div>
                  <div className="time-bar-value tabular">
                    {formatElapsed(step.seconds)} <span className="field-hint">{percent(step.share)}</span>
                  </div>
                  {step.name === "rendering" && render?.parts && render.parts.length > 0 && (
                    <ul className="time-bar-parts">
                      {render.parts.map((part) => (
                        <li key={part.name}>
                          <span>{copy.renderParts[part.name as keyof typeof copy.renderParts] || part.label}</span>
                          <span className="tabular">{formatElapsed(part.seconds)}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              );
            })}
          </ol>
        )}
      </section>

      <section className="panel" aria-labelledby="slow-title">
        <h3 id="slow-title">{copy.slowedDown}</h3>
        {bottlenecks.length === 0 ? (
          <p className="field-hint">{copy.noBottlenecks}</p>
        ) : (
          <ul className="bottlenecks">
            {bottlenecks.map((item) => (
              <li key={item.name}>
                <div className="bottleneck-head">
                  <strong>{STAGE_LABELS[item.name] || item.label}</strong>
                  {item.seconds > 0 && (
                    <span className="field-hint tabular">
                      {fill(copy.bottleneckTime, { time: formatElapsed(item.seconds), share: percent(item.share) })}
                    </span>
                  )}
                </div>
                <p>{item.why}</p>
                {item.tips.length > 0 && (
                  <>
                    <p className="bottleneck-tips-title">{copy.whatHelps}</p>
                    <ul className="bottleneck-tips">
                      {item.tips.map((tip) => (
                        <li key={tip}>{tip}</li>
                      ))}
                    </ul>
                  </>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      {ai.length > 0 && (
        <section className="panel" aria-labelledby="ai-title">
          <h3 id="ai-title">{copy.aiWork}</h3>
          <p className="panel-intro">{copy.aiWorkHint}</p>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">{copy.aiStep}</th>
                  <th scope="col">{copy.aiCalls}</th>
                  <th scope="col">{copy.aiRead}</th>
                  <th scope="col">{copy.aiWrote}</th>
                  <th scope="col">{copy.aiSpeed}</th>
                  <th scope="col" className="hide-sm">
                    {copy.aiPerCall}
                  </th>
                </tr>
              </thead>
              <tbody>
                {ai.map((row) => (
                  <tr key={row.step}>
                    <td>{row.step}</td>
                    <td className="tabular">{number(row.calls)}</td>
                    <td className="tabular">
                      {fill(copy.tokensIn, { tokens: number(row.prompt_tokens), time: formatElapsed(row.prompt_seconds) })}
                    </td>
                    <td className="tabular">
                      {fill(copy.tokensIn, { tokens: number(row.reply_tokens), time: formatElapsed(row.reply_seconds) })}
                    </td>
                    <td className="tabular">
                      {row.tokens_per_second ? fill(copy.tokensPerSecond, { n: row.tokens_per_second.toFixed(1) }) : "—"}
                    </td>
                    <td className="tabular hide-sm">{formatElapsed(row.seconds_per_call)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {(run || render) && (
        <section className="panel" aria-labelledby="ran-title">
          <h3 id="ran-title">{copy.ranWith}</h3>
          <dl className="facts">
            {run?.models.speech && (
              <>
                <dt>{copy.speechModel}</dt>
                <dd>
                  {run.models.speech}
                  {run.found.language
                    ? ` · ${fill(copy.language, {
                        language: run.found.language,
                        sure: run.found.language_probability != null ? ` (${Math.round(run.found.language_probability * 100)}%)` : "",
                      })}`
                    : ""}
                  {run.found.speech_reliable === false && <span className="status-pill accent">{copy.unreliable}</span>}
                </dd>
              </>
            )}
            {run?.models.vision && (
              <>
                <dt>{copy.visionModel}</dt>
                <dd>
                  {run.models.vision}
                  {run.found.scenes != null && run.found.scenes_ai != null
                    ? ` · ${fill(copy.scenesLooked, { ai: run.found.scenes_ai, scenes: run.found.scenes })}`
                    : ""}
                </dd>
              </>
            )}
            {run?.models.text && (
              <>
                <dt>{copy.textModel}</dt>
                <dd>{run.models.text}</dd>
              </>
            )}
            {run?.models.editor && (
              <>
                <dt>{copy.editorModel}</dt>
                <dd>
                  {run.models.editor}
                  {run.options.editor_review ? ` · ${copy.reviewOn}` : ""}
                </dd>
              </>
            )}
            {run && (
              <>
                {run.options.music_led && (
                  <>
                    <dt>{copy.musicLed}</dt>
                    <dd>{run.found.music_bpm ? fill(copy.musicBpm, { bpm: Math.round(run.found.music_bpm) }) : copy.musicMissing}</dd>
                  </>
                )}
                <dt>{copy.analysisCopy}</dt>
                <dd>{run.options.analysis_copy ? copy.yes : copy.no}</dd>
                <dt>{copy.moments}</dt>
                <dd>{number(run.found.candidates)}</dd>
                <dt>{copy.computer}</dt>
                <dd>
                  {fill(copy.computerValue, {
                    threads: run.computer.cpu_threads,
                    memory: run.computer.memory_gb != null ? `${run.computer.memory_gb} GB` : "—",
                  })}
                </dd>
              </>
            )}
            {render?.encoder?.codec && (
              <>
                <dt>{copy.encoder}</dt>
                <dd>
                  {[render.encoder.codec, render.encoder.preset, render.encoder.crf ? `CRF ${render.encoder.crf}` : ""]
                    .filter(Boolean)
                    .join(" · ")}
                </dd>
              </>
            )}
            {video.size_bytes > 0 && (
              <>
                <dt>{copy.sourceSize}</dt>
                <dd>{formatBytes(video.size_bytes)}</dd>
              </>
            )}
          </dl>
          {!run && <p className="field-hint">{copy.olderJob}</p>}
        </section>
      )}

      <section className="panel" aria-labelledby="warn-title">
        <h3 id="warn-title">{fill(copy.warnings, { n: warnings.length })}</h3>
        {warnings.length === 0 ? (
          <p className="field-hint">{copy.noWarnings}</p>
        ) : (
          <ul className="warning-list">
            {warnings.map((item) => (
              <li key={item.message} className={item.level === "ERROR" ? "error" : undefined}>
                {item.message}
                {item.count > 1 && <span className="field-hint"> {fill(copy.warningCount, { n: item.count })}</span>}
              </li>
            ))}
          </ul>
        )}
      </section>

      {data.has_log && (
        <details className="panel log-details" onToggle={(e) => (e.currentTarget as HTMLDetailsElement).open && loadLogs()}>
          <summary>{copy.log}</summary>
          <div className="btn-row log-tools">
            <div className="chips" role="group" aria-label={copy.logFilter}>
              <button type="button" className={`chip${logFilter === "all" ? " active" : ""}`} aria-pressed={logFilter === "all"} onClick={() => setLogFilter("all")}>
                {copy.logAll}
              </button>
              <button
                type="button"
                className={`chip${logFilter === "warnings" ? " active" : ""}`}
                aria-pressed={logFilter === "warnings"}
                onClick={() => setLogFilter("warnings")}
              >
                {copy.logWarnings}
              </button>
            </div>
            <a className="btn sm" href={`/api/jobs/${id}/logs?download=1`} download>
              {copy.downloadLog}
            </a>
          </div>
          {logs === null ? (
            <PageLoader />
          ) : (
            <div className="log log-tall" tabIndex={0} aria-label={copy.log}>
              {shownLogs.length === 0
                ? copy.logEmpty
                : shownLogs.map((entry, index) => (
                    <div key={index} className={entry.level}>
                      {entry.ts ? `${new Date(entry.ts * 1000).toLocaleTimeString()} ` : ""}
                      {entry.level === "INFO" ? "" : `[${entry.level}] `}
                      {entry.message}
                    </div>
                  ))}
            </div>
          )}
          <p className="field-hint">{copy.logNote}</p>
        </details>
      )}
    </div>
  );
}
