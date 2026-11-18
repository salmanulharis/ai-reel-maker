import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { io } from "socket.io-client";
import { api } from "../services/api";
import { fill, messages } from "../i18n";
import { chime, inBackground, readPrefs, showNotification } from "../services/notify";

const copy = messages.notify;

type Outcome = "review" | "result" | "failed";

/** All open tabs get the same event within moments; a later event (a retry) is a new one. */
const CLAIM_WINDOW_MS = 60_000;

/**
 * One tab handles each finished job: with several tabs open, the first to claim the event
 * notifies. Old claims are removed so storage does not grow.
 */
function claim(tag: string): boolean {
  try {
    const now = Date.now();
    const key = `notify-claim:${tag}`;
    if (now - Number(localStorage.getItem(key) || 0) < CLAIM_WINDOW_MS) return false;
    localStorage.setItem(key, String(now));
    for (let i = localStorage.length - 1; i >= 0; i -= 1) {
      const name = localStorage.key(i);
      if (name?.startsWith("notify-claim:") && now - Number(localStorage.getItem(name)) > CLAIM_WINDOW_MS) {
        localStorage.removeItem(name);
      }
    }
    return true;
  } catch {
    return true;
  }
}

/**
 * Listens for finished jobs on every page. While the app is in the background it marks the tab
 * title (returned as a prefix such as "✓ "), and, when chosen, shows a notification and plays a
 * chime. The mark clears when the app is in front again.
 */
export function useJobNotifications(): string {
  const navigate = useNavigate();
  const [badge, setBadge] = useState("");

  useEffect(() => {
    const socket = io("/", { transports: ["websocket", "polling"] });

    const handle = async (jobId: string, outcome: Outcome) => {
      if (!jobId || !inBackground()) return;
      setBadge(outcome === "failed" ? copy.badgeFailed : copy.badgeDone);
      const tag = `job-${jobId}-${outcome}`;
      if (!claim(tag)) return;
      const prefs = readPrefs();
      if (prefs.sound) chime(outcome === "failed" ? "failed" : "done");
      if (!prefs.notify) return;
      let name = "";
      try {
        name = (await api.job(jobId)).original_filename || "";
      } catch {
        // The name is only a nicety.
      }
      const title = outcome === "result" ? copy.resultTitle : outcome === "review" ? copy.reviewTitle : copy.failedTitle;
      const body = name ? fill(copy.body, { name }) : copy.bodyGeneric;
      const page = outcome === "result" ? "result" : outcome === "review" ? "review" : "analysis";
      showNotification(title, body, tag, () => navigate(`/jobs/${jobId}/${page}`));
    };

    socket.on("job_complete", (payload: { job_id: string; phase?: string }) => {
      void handle(payload.job_id, payload.phase === "render" ? "result" : "review");
    });
    socket.on("job_failed", (payload: { job_id: string; error?: string }) => {
      // Stopping a job is the person's own action; there is nothing to tell them.
      if (payload.error === "cancelled") return;
      void handle(payload.job_id, "failed");
    });

    const clear = () => {
      if (!inBackground()) setBadge("");
    };
    document.addEventListener("visibilitychange", clear);
    window.addEventListener("focus", clear);
    return () => {
      socket.close();
      document.removeEventListener("visibilitychange", clear);
      window.removeEventListener("focus", clear);
    };
  }, [navigate]);

  return badge;
}
