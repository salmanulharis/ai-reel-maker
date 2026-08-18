import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { OllamaStatus as Status } from "../types";

export function OllamaStatus() {
  const [status, setStatus] = useState<Status | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api.ollama().then((s) => alive && setStatus(s)).catch(() => alive && setStatus(null));
    load();
    const id = setInterval(load, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const connected = Boolean(status?.connected);
  return (
    <div className="ollama-chip" title={status?.error || status?.base_url || "Ollama"}>
      <span className={`status-dot ${connected ? "ok" : "err"}`} />
      Ollama {connected ? "Connected" : "Offline"}
      {status && (
        <span style={{ color: "var(--muted)" }}>
          {status.vision_model} {status.vision_present ? "✓" : "✗"} · {status.text_model}{" "}
          {status.text_present ? "✓" : "✗"}
        </span>
      )}
    </div>
  );
}
