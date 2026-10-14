import en from "./en.json";

const dictionaries: Record<string, typeof en> = { en };
let locale = "en";

export function t(path: string): string {
  const dict = dictionaries[locale] || en;
  return path.split(".").reduce((acc: unknown, key) => {
    if (acc && typeof acc === "object" && key in (acc as object)) {
      return (acc as Record<string, unknown>)[key];
    }
    return path;
  }, dict) as string;
}

export function setLocale(next: string) {
  locale = next;
}

export const messages = en;

/** Fill {placeholders} in a message, e.g. fill("Video {n}", { n: 2 }). */
export function fill(text: string, values: Record<string, string | number>): string {
  return text.replace(/\{(\w+)\}/g, (match, key: string) => (key in values ? String(values[key]) : match));
}
