import { useEffect, useState } from "react";
import { messages } from "../i18n";
import {
  chime,
  onPrefsChange,
  permission,
  readPrefs,
  requestPermission,
  writePrefs,
  type NotifyPermission,
  type NotifyPrefs,
} from "../services/notify";

const copy = messages.notify;

/** "When it's done" choices, shown where people wait for a job. Saved in this browser. */
export function NotifySettings() {
  const [prefs, setPrefs] = useState<NotifyPrefs>(readPrefs);
  const [perm, setPerm] = useState<NotifyPermission>(permission);

  useEffect(() => onPrefsChange(() => setPrefs(readPrefs())), []);

  function save(next: NotifyPrefs) {
    setPrefs(next);
    writePrefs(next);
  }

  async function toggleNotify(on: boolean) {
    if (on) {
      const result = await requestPermission();
      setPerm(result);
      if (result !== "granted") {
        save({ ...prefs, notify: false });
        return;
      }
    }
    save({ ...prefs, notify: on });
  }

  function toggleSound(on: boolean) {
    save({ ...prefs, sound: on });
    // Let people hear what they chose (this also lets the browser play it later).
    if (on) chime("done");
  }

  const blocked = perm === "denied" || perm === "unsupported" || perm === "insecure";
  const hint =
    perm === "denied"
      ? copy.blocked
      : perm === "insecure"
        ? copy.insecure
        : perm === "unsupported"
          ? copy.unsupported
          : copy.hint;

  return (
    <fieldset className="notify-settings">
      <legend className="sub-title">{copy.legend}</legend>
      <div className="notify-options">
        <label className="check">
          <input
            type="checkbox"
            checked={prefs.notify && perm === "granted"}
            disabled={blocked}
            aria-describedby="notify-hint"
            onChange={(e) => void toggleNotify(e.target.checked)}
          />
          <span>{copy.notify}</span>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={prefs.sound}
            aria-describedby="notify-hint"
            onChange={(e) => toggleSound(e.target.checked)}
          />
          <span>{copy.sound}</span>
        </label>
      </div>
      <p id="notify-hint" className="field-hint">
        {hint}
      </p>
    </fieldset>
  );
}
