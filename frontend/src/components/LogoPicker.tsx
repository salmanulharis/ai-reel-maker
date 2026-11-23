import { useEffect, useRef, useState } from "react";
import { api } from "../services/api";
import { messages } from "../i18n";
import { Alert, Spinner } from "./Feedback";
import { TransferProgress } from "./Progress";
import { useUpload } from "../hooks/useUpload";

const copy = messages.config;

/** The saved logo (one for all videos): add, replace or remove it. */
export function LogoPicker() {
  const [version, setVersion] = useState(0);
  const [has, setHas] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const sending = useUpload();

  useEffect(() => {
    fetch(`/api/brand/logo?v=${version}`, { method: "GET" })
      .then((response) => setHas(response.ok))
      .catch(() => setHas(false));
  }, [version]);

  async function upload(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      await sending.run(file.size, (options) => api.uploadLogo(file, options));
      setVersion((v) => v + 1);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  async function remove() {
    setBusy(true);
    setError("");
    try {
      await api.deleteLogo();
      setVersion((v) => v + 1);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="logo-picker">
      {has && <img src={`/api/brand/logo?v=${version}`} alt={copy.logoCurrent} />}
      <div className="btn-row">
        <label className="btn sm">
          {busy && <Spinner />}
          {has ? copy.logoReplace : copy.logoAdd}
          <input
            ref={input}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="sr-only"
            disabled={busy}
            onChange={(e) => void upload(e.target.files?.[0])}
          />
        </label>
        {has && (
          <button type="button" className="btn sm ghost" disabled={busy} onClick={remove}>
            {copy.logoRemove}
          </button>
        )}
      </div>
      {sending.status && <TransferProgress status={sending.status} label={messages.transfer.uploadLabel} />}
      {error && <Alert>{error}</Alert>}
    </div>
  );
}
