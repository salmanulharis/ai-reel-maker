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
