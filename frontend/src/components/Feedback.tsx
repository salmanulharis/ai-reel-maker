import type { ReactNode } from "react";
import { messages } from "../i18n";

export function Spinner({ large = false, label }: { large?: boolean; label?: string }) {
  return (
    <span className={`spinner${large ? " lg" : ""}`} role={label ? "status" : undefined} aria-label={label} aria-hidden={label ? undefined : true} />
  );
}

/** Full-width loading state for a page whose data has not arrived yet. */
export function PageLoader({ label = messages.common.loading }: { label?: string }) {
  return (
    <div className="page-loader" role="status" aria-live="polite">
      <Spinner large />
      <span>{label}</span>
    </div>
  );
}

export function Alert({
  kind = "error",
  children,
}: {
  kind?: "error" | "success" | "info";
  children: ReactNode;
}) {
  return (
    <div className={`alert ${kind}`} role={kind === "error" ? "alert" : "status"}>
      <span className="alert-icon" aria-hidden="true">
        {kind === "error" ? <AlertIcon /> : kind === "success" ? <CheckIcon /> : <InfoIcon />}
      </span>
      <div>{children}</div>
    </div>
  );
}

/** Small "working…" or "saved" text next to a control. Announced politely. */
export function InlineStatus({ busy, ok, children }: { busy?: boolean; ok?: boolean; children: ReactNode }) {
  return (
    <span className={`inline-status${ok ? " ok" : ""}`} role="status" aria-live="polite">
      {busy && <Spinner />}
      {ok && <CheckIcon size={14} />}
      {children}
    </span>
  );
}

type IconProps = { size?: number };

function Svg({ size = 16, children }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

export const CheckIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M20 6 9 17l-5-5" />
  </Svg>
);
export const AlertIcon = (props: IconProps) => (
  <Svg {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 8v5M12 16h.01" />
  </Svg>
);
export const InfoIcon = (props: IconProps) => (
  <Svg {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5M12 8h.01" />
  </Svg>
);
export const UploadIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M12 16V4M7 9l5-5 5 5" />
    <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
  </Svg>
);
export const PlusIcon = (props: IconProps) => (
  <Svg {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 8v8M8 12h8" />
  </Svg>
);
export const HistoryIcon = (props: IconProps) => (
  <Svg {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </Svg>
);
export const SettingsIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" />
    <circle cx="16" cy="6" r="2" />
    <circle cx="10" cy="12" r="2" />
    <circle cx="18" cy="18" r="2" />
  </Svg>
);
export const PlayIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M7 4.5v15l12-7.5z" />
  </Svg>
);
export const FilmIcon = (props: IconProps) => (
  <Svg {...props}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M7 4v16M17 4v16M3 9h4M17 9h4M3 15h4M17 15h4" />
  </Svg>
);
