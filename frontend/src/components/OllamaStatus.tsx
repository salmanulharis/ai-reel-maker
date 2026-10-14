import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api";
import { messages } from "../i18n";
import type { OllamaStatus as Status } from "../types";

const copy = messages.aiStatus;

/** Header chip: is local AI usable right now? Links to Settings, where it can be fixed. */
export function OllamaStatus() {
  const [status, setStatus] = useState<Status | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .ollama()
        .then((s) => alive && setStatus(s))
        .catch(() => alive && setStatus(null))
        .finally(() => alive && setLoaded(true));
    load();
    const id = setInterval(load, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const connected = Boolean(status?.connected);
  const runtime = status?.runtime;
  const visionReady = status?.vision_ready ?? status?.vision_present;
  const modelsReady = Boolean(visionReady && status?.text_present);
  const state = !loaded ? "checking" : !connected ? "offline" : modelsReady ? "ready" : "partial";
  const dot = { checking: "", offline: "err", ready: "ok", partial: "warn" }[state];
  const label = { checking: copy.checking, offline: copy.offline, ready: copy.ready, partial: copy.partial }[state];
  const hint = { checking: "", offline: copy.offlineHint, ready: copy.readyHint, partial: copy.partialHint }[state];
  const accel = connected && runtime ? (runtime.mode === "gpu" ? "GPU" : runtime.mode === "split" ? "GPU + CPU" : "CPU") : "";
  const details = [
    hint,
    status ? `${status.vision_model} ${visionReady ? "✓" : "✗"} · ${status.text_model} ${status.text_present ? "✓" : "✗"}` : "",
    ...(status?.warnings || []),
  ]
    .filter(Boolean)
    .join("\n");

  return (
    <Link className="ollama-chip" to="/settings" title={details || undefined} aria-label={`${label}. ${hint}`.trim()}>
      <span className={`status-dot ${dot}`} aria-hidden="true" />
      {label}
      {accel && <span className="chip-accel">{accel}</span>}
    </Link>
  );
}
