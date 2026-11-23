import { formatBytes, formatElapsed } from "../services/api";
import { fill, messages } from "../i18n";
import type { TransferFields } from "../types";

const copy = messages.transfer;

/**
 * A progress bar. With a number (0-100) it shows that share; without one it shows activity
 * only (the amount of work is not known yet). The moving bar stays still for people who
 * ask for reduced motion (see styles.css).
 */
export function ProgressBar({
  value,
  label,
  size,
}: {
  value?: number | null;
  label: string;
  size?: "sm" | "lg";
}) {
  const known = typeof value === "number" && Number.isFinite(value);
  const percent = known ? Math.max(0, Math.min(100, value)) : 0;
  return (
    <div
      className={["progress-line", size, known ? "" : "indeterminate"].filter(Boolean).join(" ")}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={known ? Math.round(percent) : undefined}
    >
      <span style={known ? { width: `${percent}%` } : undefined} />
    </div>
  );
}

/** "about 3 min left": whole seconds below two minutes, whole minutes above. */
export function formatTimeLeft(seconds: number): string {
  const value = Math.max(1, seconds);
  const rounded = value >= 120 ? Math.round(value / 60) * 60 : Math.ceil(value);
  return formatElapsed(rounded, true);
}

const PHASES: Record<string, string> = {
  starting: copy.starting,
  verifying: copy.verifying,
  finishing: copy.finishing,
  uploading: copy.uploading,
};

/** Progress of a download or upload: bar, share, bytes, speed and time left. */
export function TransferProgress({ status, label = copy.label }: { status?: TransferFields | null; label?: string }) {
  const total = Number(status?.bytes_total || 0);
  const done = Math.min(Number(status?.bytes_done || 0), total || Number.MAX_SAFE_INTEGER);
  const phase = status?.phase || "downloading";
  // Checking and finishing happen after every byte arrived: the bar stays full.
  const settled = phase === "verifying" || phase === "finishing";
  const percent = settled ? 100 : total > 0 ? (done / total) * 100 : null;
  const parts = [PHASES[phase] || copy.downloading];
  if (percent !== null && !settled) parts.push(`${Math.floor(percent)}%`);
  if (total > 0) parts.push(fill(copy.of, { done: formatBytes(done), total: formatBytes(total) }));
  else if (done > 0) parts.push(formatBytes(done));
  if (!settled && status?.speed) parts.push(fill(copy.speed, { speed: formatBytes(Math.round(status.speed)) }));
  if (!settled && typeof status?.eta === "number") parts.push(fill(copy.timeLeft, { time: formatTimeLeft(status.eta) }));
  return (
    <div className="transfer">
      <ProgressBar value={percent} label={label} />
      <p className="hint tabular">{parts.join(" · ")}</p>
    </div>
  );
}
