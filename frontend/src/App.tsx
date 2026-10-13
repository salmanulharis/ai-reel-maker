import { useEffect, useRef } from "react";
import { Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { messages } from "./i18n";
import { OllamaStatus } from "./components/OllamaStatus";
import { ThemeToggle } from "./components/ThemeToggle";
import { FilmIcon, HistoryIcon, HomeIcon, SettingsIcon } from "./components/Feedback";
import Dashboard from "./pages/Dashboard";
import Analysis from "./pages/Analysis";
import Review from "./pages/Review";
import Result from "./pages/Result";
import Settings from "./pages/Settings";
import History from "./pages/JobHistory";

const titles: Record<string, [string, string]> = {
  "/": [messages.appName, messages.subtitle],
  "/history": [messages.history.title, messages.history.subtitle],
  "/settings": ["Settings", "Local processing, models, and storage"],
};

export default function App() {
  const location = useLocation();
  const mainRef = useRef<HTMLElement>(null);
  const key = location.pathname.startsWith("/jobs") ? "/jobs" : location.pathname;
  const [title, subtitle] =
    key === "/jobs"
      ? [messages.job.title, messages.job.subtitle]
      : titles[location.pathname] || [messages.appName, messages.subtitle];

  useEffect(() => {
    document.title = title === messages.appName ? messages.appName : `${title} · ${messages.appName}`;
  }, [title]);

  // A new page starts at the top, like a full page load would.
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <div className="app">
      <a
        className="skip-link"
        href="#main-content"
        onClick={(event) => {
          event.preventDefault();
          mainRef.current?.focus();
        }}
      >
        {messages.common.skipToContent}
      </a>
      <aside className="sidebar">
        <div className="brand">
          <h1>
            <span className="brand-mark">
              <FilmIcon size={14} />
            </span>
            {messages.appName}
          </h1>
          <p>{messages.subtitle}</p>
        </div>
        <nav className="nav" aria-label="Main">
          <NavLink to="/" end>
            <HomeIcon />
            {messages.nav.dashboard}
          </NavLink>
          <NavLink to="/history">
            <HistoryIcon />
            {messages.nav.history}
          </NavLink>
          <NavLink to="/settings">
            <SettingsIcon />
            {messages.nav.settings}
          </NavLink>
        </nav>
        <p className="privacy">{messages.privacy}</p>
      </aside>
      <div className="main">
        <header className="header">
          <div>
            <h2>{title}</h2>
            <p className="sub">{subtitle}</p>
          </div>
          <div className="header-actions">
            <ThemeToggle />
            <OllamaStatus />
            <NavLink className="btn header-link" to="/history">
              {messages.nav.queue}
            </NavLink>
            <NavLink className="btn header-link" to="/settings">
              {messages.nav.settings}
            </NavLink>
          </div>
        </header>
        <main className="content" id="main-content" ref={mainRef} tabIndex={-1}>
          <div className="page" key={location.pathname}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/jobs/:id/analysis" element={<Analysis />} />
              <Route path="/jobs/:id/review" element={<Review />} />
              <Route path="/jobs/:id/result" element={<Result />} />
              <Route path="/history" element={<History />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}
