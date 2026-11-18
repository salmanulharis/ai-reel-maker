/**
 * "Tell me when it's done": a system notification and an optional chime when a job finishes
 * while the app is in the background. Choices are kept in this browser only.
 */

export interface NotifyPrefs {
  /** Show a system notification. Needs the browser's permission. */
  notify: boolean;
  /** Play a short chime. */
  sound: boolean;
}

const STORAGE_KEY = "notify-when-done";
const DEFAULTS: NotifyPrefs = { notify: false, sound: false };

export function readPrefs(): NotifyPrefs {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const data = JSON.parse(raw) as Partial<NotifyPrefs>;
    return { notify: data.notify === true, sound: data.sound === true };
  } catch {
    return { ...DEFAULTS };
  }
}

export function writePrefs(prefs: NotifyPrefs): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // Private windows may refuse storage; the choice then lasts until the page is closed.
  }
  window.dispatchEvent(new Event(STORAGE_KEY));
}

/** Follow changes made on another page of the app (same tab) or in another tab. */
export function onPrefsChange(listener: () => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) listener();
  };
  window.addEventListener(STORAGE_KEY, listener);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(STORAGE_KEY, listener);
    window.removeEventListener("storage", onStorage);
  };
}

export type NotifyPermission = "granted" | "denied" | "default" | "unsupported" | "insecure";

/**
 * Whether system notifications can be shown. Browsers only allow them on a secure page,
 * which includes http://localhost but not an address such as http://192.168.1.20:5000.
 */
export function permission(): NotifyPermission {
  if (typeof window === "undefined" || !("Notification" in window)) return "unsupported";
  if (!window.isSecureContext) return "insecure";
  return window.Notification.permission as NotifyPermission;
}

export async function requestPermission(): Promise<NotifyPermission> {
  const current = permission();
  if (current !== "default") return current;
  try {
    return (await window.Notification.requestPermission()) as NotifyPermission;
  } catch {
    return permission();
  }
}

/** True when the person is probably not looking at the app. */
export function inBackground(): boolean {
  return document.visibilityState === "hidden" || !document.hasFocus();
}

/** A soft two-note chime made in the browser (no sound file). Silent if audio is blocked. */
export function chime(kind: "done" | "failed"): void {
  try {
    const Context = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Context) return;
    const context = new Context();
    const notes = kind === "done" ? [660, 880] : [440, 330];
    notes.forEach((frequency, index) => {
      const start = context.currentTime + index * 0.18;
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.2, start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.35);
      oscillator.connect(gain).connect(context.destination);
      oscillator.start(start);
      oscillator.stop(start + 0.4);
    });
    window.setTimeout(() => void context.close().catch(() => undefined), 1000);
  } catch {
    // Audio is optional.
  }
}

/** Show a system notification; clicking it brings the app forward and runs `onClick`. */
export function showNotification(title: string, body: string, tag: string, onClick: () => void): void {
  if (permission() !== "granted") return;
  try {
    const note = new window.Notification(title, { body, tag });
    note.onclick = () => {
      window.focus();
      onClick();
      note.close();
    };
  } catch {
    // Some browsers only allow notifications from a service worker; the tab title still changes.
  }
}
