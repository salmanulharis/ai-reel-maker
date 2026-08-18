import { Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { messages } from "./i18n";
import { OllamaStatus } from "./components/OllamaStatus";
import { ThemeToggle } from "./components/ThemeToggle";
import Dashboard from "./pages/Dashboard";
import Analysis from "./pages/Analysis";
import Review from "./pages/Review";
import Result from "./pages/Result";
import Settings from "./pages/Settings";
import History from "./pages/JobHistory";

const titles: Record<string, [string, string]> = {
  "/": [messages.appName, messages.subtitle],
  "/history": ["History", "Previous highlight jobs"],
  "/settings": ["Settings", "Local processing, models, and storage"],
};

export default function App() {
  const location = useLocation();
  const key = location.pathname.startsWith("/jobs") ? "/jobs" : location.pathname;
  const [title, subtitle] =
    key === "/jobs"
      ? ["Job", "Analysis, review, and export"]
      : titles[location.pathname] || [messages.appName, messages.subtitle];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>{messages.appName}</h1>
          <p>{messages.subtitle}</p>
        </div>
        <nav className="nav">
          <NavLink to="/" end>
            {messages.nav.dashboard}
          </NavLink>
          <NavLink to="/history">{messages.nav.history}</NavLink>
          <NavLink to="/settings">{messages.nav.settings}</NavLink>
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
            <NavLink className="btn" to="/history">
              {messages.nav.queue}
            </NavLink>
            <NavLink className="btn" to="/settings">
              {messages.nav.settings}
            </NavLink>
          </div>
        </header>
        <div className="content">
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
      </div>
    </div>
  );
}
