import type { TransferFields } from "../types";

/** Share of a new speed sample in the smoothed speed (same as the server's downloads). */
const SMOOTHING = 0.3;
/** A speed is shown only after this long, so the first burst does not promise a quick finish. */
const STEADY_MS = 1500;

/** Smoothed speed and time left for a transfer measured in the browser (uploads). */
export class RateMeter {
  private readonly begun = performance.now();
  private last: { at: number; done: number } | null = null;
  private speed = 0;

  fields(done: number, total: number, phase = "uploading"): TransferFields {
    const now = performance.now();
    if (!this.last) {
      this.last = { at: now, done };
    } else if (now - this.last.at >= 200 && done >= this.last.done) {
      const sample = ((done - this.last.done) * 1000) / (now - this.last.at);
      this.speed = this.speed <= 0 ? sample : this.speed + SMOOTHING * (sample - this.speed);
      this.last = { at: now, done };
    }
    const steady = now - this.begun >= STEADY_MS && this.speed > 0;
    return {
      bytes_done: done,
      bytes_total: total,
      speed: steady ? this.speed : 0,
      eta: steady && total > 0 && done < total ? (total - done) / this.speed : null,
      phase: total > 0 && done >= total ? "finishing" : phase,
    };
  }
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
