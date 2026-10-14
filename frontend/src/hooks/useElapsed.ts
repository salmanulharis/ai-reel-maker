import { useEffect, useMemo, useState } from "react";
import type { JobTiming } from "../types";

/** Seconds between a server timestamp and "now" on the server's clock. */
function secondsSince(iso: string | null | undefined, offsetMs: number, nowMs: number): number {
  if (!iso) return 0;
  const start = Date.parse(iso);
  if (Number.isNaN(start)) return 0;
  return Math.max(0, (nowMs + offsetMs - start) / 1000);
}

/**
 * Live processing time for a job.
 *
 * The server sends when the current attempt started and its own current time. The difference
 * between that and the browser clock is applied to every tick, so a wrong or drifting local
 * clock does not change the result. One 1-second ticker runs only while an attempt is running.
 */
export function useElapsed(timing?: JobTiming | null) {
  const running = Boolean(timing?.attempt_started_at);
  const offsetMs = useMemo(() => {
    const server = timing?.server_time ? Date.parse(timing.server_time) : NaN;
    return Number.isNaN(server) ? 0 : server - Date.now();
    // Recompute only when the server sends a new timestamp.
  }, [timing?.server_time]);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    setNow(Date.now());
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [running, timing?.attempt_started_at]);

  const attempt = running ? secondsSince(timing?.attempt_started_at, offsetMs, now) : timing?.attempt_seconds ?? 0;
  const total = (timing?.total_seconds ?? 0) + (running ? attempt : 0);
  return {
    running,
    /** Current attempt while running; the last attempt's length otherwise. */
    attempt,
    /** Every attempt so far, including the running one. */
    total,
    /** Live time of a stage: finished runs plus the run in progress. */
    stageSeconds: (seconds?: number, startedAt?: string | null) =>
      (seconds ?? 0) + (startedAt ? secondsSince(startedAt, offsetMs, now) : 0),
  };
}
