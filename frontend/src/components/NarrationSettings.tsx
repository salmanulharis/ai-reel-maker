import { useEffect, useRef, useState } from "react";
import { Alert, InlineStatus, Spinner } from "./Feedback";
import { TransferProgress } from "./Progress";
import { api } from "../services/api";
import { fill, messages } from "../i18n";
import type { EngineStatus, VideoToolsStatus } from "../types";

const copy = messages.narrated;

/** A number field that saves when it loses focus and follows the saved value. */
function SavedNumber({
  value,
  min,
  max,
  step,
  onSave,
}: {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onSave: (value: number) => Promise<void>;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  return (
    <input
      type="number"
      min={min}
      max={max}
      step={step}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        const number = Number(text);
        if (text.trim() !== "" && Number.isFinite(number) && number !== value) void onSave(number);
        else setText(String(value));
      }}
    />
  );
}

/**
 * Settings → Narrated videos: the brief engine chosen for new videos (with a readiness check
 * and a Claude test), the narration voice (download, voice, speed), and the Remotion tools.
 * Each choice is saved at once through ``save``; downloads start only from their buttons.
 */
export function NarrationSettings({
  settings,
  save,
}: {
  settings: Record<string, unknown>;
  save: (key: string, value: unknown, flash: string) => Promise<void>;
}) {
  const [tools, setTools] = useState<VideoToolsStatus | null>(null);
  const [engines, setEngines] = useState<EngineStatus[] | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [test, setTest] = useState<{ ok: boolean; seconds: number; model: string | null; error: string | null } | null>(null);
  const timer = useRef<number | null>(null);

  async function refresh() {
    try {
      const next = await api.videoTools();
      setTools(next);
      const working = next.voice.state === "running" || next.remotion.prepare.state === "running";
      if (working && !timer.current) {
        timer.current = window.setInterval(() => void refresh(), 1500);
      } else if (!working && timer.current) {
        window.clearInterval(timer.current);
        timer.current = null;
      }
    } catch (err) {
      setError(String((err as Error).message || err));
    }
  }

  useEffect(() => {
    void refresh();
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function run(key: string, call: () => Promise<unknown>) {
    setBusy(key);
    setError("");
    try {
      await call();
    } catch (err) {
      setError(String((err as Error).message || err));
    } finally {
      setBusy("");
    }
  }

  const voice = tools?.voice;
  const remotion = tools?.remotion;
  const voices = voice?.voices || {};
  const engine = String(settings.brief_engine || "local");
  const claude = engines?.find((item) => item.name === "claude");

  return (
    <>
      {error && <Alert>{error}</Alert>}

      <fieldset className="config-section">
        <legend>{copy.settingsEngine}</legend>
        {(["local", "claude"] as const).map((name) => (
          <label className="check" key={name}>
            <input type="radio" name="settings-brief-engine" checked={engine === name} onChange={() => void save("brief_engine", name, "brief")} />
            <span>
              <strong>{name === "local" ? copy.engineLocal : copy.engineClaude}</strong>
              <span className="field-hint block">{name === "local" ? copy.engineLocalHint : copy.engineClaudeHint}</span>
            </span>
          </label>
        ))}
        <div className="btn-row">
          <button
            className="btn sm"
            type="button"
            disabled={busy === "check"}
            aria-busy={busy === "check"}
            onClick={() => run("check", async () => setEngines((await api.briefEngines(true)).engines))}
          >
            {busy === "check" && <Spinner />}
            {busy === "check" ? copy.checking : copy.check}
          </button>
          <button
            className="btn sm"
            type="button"
            disabled={busy === "test"}
            aria-busy={busy === "test"}
            onClick={() => run("test", async () => setTest(await api.testClaude()))}
          >
            {busy === "test" && <Spinner />}
            {busy === "test" ? copy.testingClaude : copy.testClaude}
          </button>
        </div>
        {engines?.map((item) => (
          <p key={item.name} className="field-hint">
            <strong>{item.label}:</strong> {item.ready ? copy.ready : copy.notReady}
            {[...item.problems, ...(item.ready ? item.notes : [])].map((line) => (
              <span key={line} className={`block ${item.ready ? "" : "error-text"}`}>
                {line}
              </span>
            ))}
          </p>
        ))}
        {test && (
          <p className={test.ok ? "field-hint" : "field-hint error-text"} role="status">
            {test.ok ? fill(copy.claudeOk, { seconds: test.seconds, model: test.model || String(settings.claude_model || "") }) : test.error}
          </p>
        )}
        {claude?.details?.version ? <p className="field-hint">Claude Code {String(claude.details.version)}</p> : null}

        <div className="form-grid">
          <label className="field">
            <span className="field-label">{copy.claudeModel}</span>
            <select value={String(settings.claude_model || "opus")} onChange={(e) => void save("claude_model", e.target.value, "claude")}>
              <option value="opus">Opus</option>
              <option value="sonnet">Sonnet</option>
              <option value="haiku">Haiku</option>
            </select>
            <span className="field-hint">{copy.claudeModelHint}</span>
          </label>
          <label className="field">
            <span className="field-label">{copy.claudeEffort}</span>
            <select value={String(settings.claude_effort || "medium")} onChange={(e) => void save("claude_effort", e.target.value, "claude")}>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
            <span className="field-hint">{copy.claudeEffortHint}</span>
          </label>
          <label className="field">
            <span className="field-label">{copy.claudeTimeout}</span>
            <SavedNumber min={30} max={3600} value={Number(settings.claude_timeout || 300)} onSave={(value) => save("claude_timeout", value, "claude")} />
            <span className="field-hint">{copy.claudeTimeoutHint}</span>
          </label>
          <label className="field">
            <span className="field-label">{copy.claudeBudget}</span>
            <SavedNumber min={0.05} max={20} step={0.05} value={Number(settings.claude_max_budget_usd || 1)} onSave={(value) => save("claude_max_budget_usd", value, "claude")} />
            <span className="field-hint">{copy.claudeBudgetHint}</span>
          </label>
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.claude_send_frames !== false}
            onChange={(e) => void save("claude_send_frames", e.target.checked, "claude")}
          />
          <span>
            <strong>{copy.sendFrames}</strong>
            <span className="field-hint block">{copy.sendFramesHint}</span>
          </span>
        </label>
      </fieldset>

      <fieldset className="config-section">
        <legend>{copy.voiceModel}</legend>
        <p className="field-hint">{fill(copy.voiceModelHint, { size: voice?.size_mb ?? 338 })}</p>
        {voice && !voice.runtime ? (
          <p className="error-text">{copy.voiceNoRuntime}</p>
        ) : (
          <div className="btn-row">
            <span className={`tag ${voice?.installed ? "ok" : ""}`}>{voice?.installed ? copy.voiceReady : copy.toolsMissing}</span>
            {voice && !voice.installed && (
              <button
                className="btn sm"
                type="button"
                disabled={voice.state === "running"}
                aria-busy={voice.state === "running"}
                onClick={() => run("voice", async () => {
                  await api.downloadVoiceModel();
                  await refresh();
                })}
              >
                {voice.state === "running" && <Spinner />}
                {voice.state === "running" ? copy.voiceDownloading : copy.voiceDownload}
              </button>
            )}
          </div>
        )}
        {voice?.state === "running" && <TransferProgress status={voice} label={copy.voiceDownloading} />}
        {voice?.state === "error" && voice.error && <p className="error-text">{voice.error}</p>}
        <div className="form-grid">
          <label className="field">
            <span className="field-label">{copy.voiceName}</span>
            <select value={String(settings.tts_voice || "af_heart")} onChange={(e) => void save("tts_voice", e.target.value, "voice")}>
              {Object.entries(voices).map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">{copy.voiceSpeed}</span>
            <SavedNumber min={0.6} max={1.5} step={0.05} value={Number(settings.tts_speed || 1)} onSave={(value) => save("tts_speed", value, "voice")} />
            <span className="field-hint">{copy.voiceSpeedHint}</span>
          </label>
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.narration_default !== false}
            onChange={(e) => void save("narration_default", e.target.checked, "voice")}
          />
          <span>
            <strong>{copy.narrationDefault}</strong>
          </span>
        </label>
      </fieldset>

      <fieldset className="config-section">
        <legend>{copy.tools}</legend>
        <label className="field">
          <span className="field-label">{copy.renderEngine}</span>
          <select value={String(settings.render_engine || "remotion")} onChange={(e) => void save("render_engine", e.target.value, "tools")}>
            <option value="remotion">{copy.renderRemotion}</option>
            <option value="ffmpeg">{copy.renderFfmpeg}</option>
          </select>
          <span className="field-hint">{copy.renderEngineHint}</span>
        </label>
        <p className="field-hint">{copy.toolsHint}</p>
        {remotion && !remotion.node_ok && <p className="error-text">{copy.nodeMissing}</p>}
        <div className="btn-row">
          <span className={`tag ${remotion?.ready ? "ok" : ""}`}>{remotion?.ready ? copy.toolsReady : copy.toolsMissing}</span>
          {remotion && !remotion.ready && (
            <button
              className="btn sm"
              type="button"
              disabled={!remotion.node_ok || remotion.prepare.state === "running"}
              aria-busy={remotion.prepare.state === "running"}
              onClick={() => run("prepare", async () => {
                await api.prepareVideoTools();
                await refresh();
              })}
            >
              {remotion.prepare.state === "running" && <Spinner />}
              {copy.prepare}
            </button>
          )}
          {remotion?.prepare.state === "running" && <InlineStatus busy>{fill(copy.preparing, { step: remotion.prepare.step })}</InlineStatus>}
        </div>
        {remotion?.prepare.state === "error" && remotion.prepare.error && <p className="error-text">{remotion.prepare.error}</p>}
        {remotion?.prepare.state === "running" && remotion.prepare.log.length > 0 && (
          <pre className="log-tail">{remotion.prepare.log.slice(-4).join("\n")}</pre>
        )}
        <label className="field">
          <span className="field-label">{copy.concurrency}</span>
          <SavedNumber min={0} max={16} value={Number(settings.remotion_concurrency || 0)} onSave={(value) => save("remotion_concurrency", value, "tools")} />
          <span className="field-hint">{copy.concurrencyHint}</span>
        </label>
      </fieldset>
    </>
  );
}
