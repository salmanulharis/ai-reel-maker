import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { OllamaStatus as Status } from "../types";

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
  const accel = !connected ? "" : runtime?.mode === "gpu" ? "GPU" : runtime?.mode === "split" ? "Partial GPU" : runtime ? "CPU" : "";
  const visionReady = status?.vision_ready ?? status?.vision_present;
  const modelsReady = Boolean(visionReady && status?.text_present);
  const dot = !loaded ? "" : !connected ? "err" : modelsReady ? "ok" : "warn";
  const title = [...(status?.warnings || []), runtime?.hint, runtime?.label, status?.error, status?.base_url]
    .filter(Boolean)
    .join(" · ");
  return (
    <div className="ollama-chip" title={title || "Ollama"}>
      <span className={`status-dot ${dot}`} aria-hidden="true" />
      Ollama {!loaded ? "…" : connected ? "Connected" : "Offline"}
      {accel && <span className="chip-accel">{accel}</span>}
      {status && (
        <span className="chip-models">
          {status.vision_model} {visionReady ? "✓" : "✗"} · {status.text_model}{" "}
          {status.text_present ? "✓" : "✗"}
        </span>
      )}
    </div>
  );
}
