import { useCallback, useEffect, useState } from "react";
import { api } from "../services/api";
import type { Job } from "../types";

export function useJob(id?: string, intervalMs = 2500) {
  const [job, setJob] = useState<Job | null>(null);

  const refresh = useCallback(() => {
    if (!id) return;
    api.job(id).then(setJob).catch(() => undefined);
  }, [id]);

  useEffect(() => {
    refresh();
    if (!id) return;
    const timer = setInterval(refresh, intervalMs);
    return () => clearInterval(timer);
  }, [id, intervalMs, refresh]);

  return { job, refresh, setJob };
}
