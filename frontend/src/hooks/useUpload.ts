import { useCallback, useEffect, useRef, useState } from "react";
import type { UploadOptions } from "../services/api";
import { RateMeter } from "../services/transfer";
import type { TransferFields } from "../types";

/** Redraws at most this often while bytes are being sent. */
const REDRAW_MS = 150;

/**
 * Progress of one upload at a time: `run(size, send)` starts `send` with progress and cancel
 * options, `status` is what to show (null when nothing is uploading), `cancel()` stops it.
 * An upload still running when the component goes away is stopped.
 */
export function useUpload() {
  const [status, setStatus] = useState<TransferFields | null>(null);
  const current = useRef<AbortController | null>(null);

  useEffect(() => () => current.current?.abort(), []);

  const run = useCallback(async <T>(size: number, send: (options: UploadOptions) => Promise<T>): Promise<T> => {
    current.current?.abort();
    const controller = new AbortController();
    current.current = controller;
    const meter = new RateMeter();
    let drawn = 0;
    setStatus(meter.fields(0, size));
    try {
      return await send({
        signal: controller.signal,
        onProgress: (loaded, total) => {
          const fields = meter.fields(loaded, total || size);
          const now = performance.now();
          if (now - drawn >= REDRAW_MS || fields.phase === "finishing") {
            drawn = now;
            setStatus(fields);
          }
        },
      });
    } finally {
      if (current.current === controller) {
        current.current = null;
        setStatus(null);
      }
    }
  }, []);

  const cancel = useCallback(() => current.current?.abort(), []);

  return { status, run, cancel };
}
