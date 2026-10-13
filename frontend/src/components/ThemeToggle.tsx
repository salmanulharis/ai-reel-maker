import { useEffect, useState } from "react";
import { applyTheme, readTheme, type Theme } from "../theme";
import { t } from "../i18n";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() =>
    typeof document !== "undefined" ? (document.documentElement.getAttribute("data-theme") as Theme) || readTheme() : "dark"
  );

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const next: Theme = theme === "dark" ? "light" : "dark";
  const label = theme === "dark" ? t("theme.light") : t("theme.dark");

  return (
    <button
      type="button"
      className="btn theme-toggle"
      onClick={() => setTheme(next)}
      aria-label={t("theme.switchTo").replace("{mode}", label)}
      aria-pressed={theme === "light"}
      title={t("theme.switchTo").replace("{mode}", label)}
    >
      <span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span>
      <span className="theme-label">{label}</span>
    </button>
  );
}
