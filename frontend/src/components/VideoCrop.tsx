import { useEffect, useRef, useState } from "react";
import { messages } from "../i18n";
import type { CropBox } from "../types";

const DEFAULT_CROP: CropBox = { x: 0.1, y: 0.1, width: 0.8, height: 0.8 };
const MIN = 0.08;

type Handle = "move" | "n" | "s" | "e" | "w" | "nw" | "ne" | "sw" | "se";

function clampCrop(next: CropBox): CropBox {
  const width = Math.min(Math.max(next.width, MIN), 1);
  const height = Math.min(Math.max(next.height, MIN), 1);
  const x = Math.min(Math.max(next.x, 0), 1 - width);
  const y = Math.min(Math.max(next.y, 0), 1 - height);
  return { x, y, width, height };
}

function contentRect(video: HTMLVideoElement, wrap: HTMLElement) {
  const vr = video.getBoundingClientRect();
  const wr = wrap.getBoundingClientRect();
  const vw = video.videoWidth || 16;
  const vh = video.videoHeight || 9;
  const scale = Math.min(vr.width / vw, vr.height / vh);
  const width = vw * scale;
  const height = vh * scale;
  return {
    left: vr.left - wr.left + (vr.width - width) / 2,
    top: vr.top - wr.top + (vr.height - height) / 2,
    width,
    height,
  };
}

export function VideoCropPreview({
  src,
  enabled,
  crop,
  sourceWidth,
  sourceHeight,
  onToggle,
  onChange,
}: {
  src: string;
  enabled: boolean;
  crop: CropBox;
  sourceWidth?: number;
  sourceHeight?: number;
  onToggle: (on: boolean) => void;
  onChange: (crop: CropBox) => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const drag = useRef<{ handle: Handle; startX: number; startY: number; crop: CropBox } | null>(null);
  const [stage, setStage] = useState({ left: 0, top: 0, width: 0, height: 0 });
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);

  function layout() {
    const video = videoRef.current;
    const wrap = wrapRef.current;
    if (!video || !wrap) return;
    setStage(contentRect(video, wrap));
  }

  useEffect(() => {
    layout();
    const wrap = wrapRef.current;
    if (!wrap) return;
    const observer = new ResizeObserver(layout);
    observer.observe(wrap);
    window.addEventListener("resize", layout);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", layout);
    };
  }, [src, enabled]);

  useEffect(() => {
    function onMove(event: PointerEvent) {
      const current = drag.current;
      if (!current || stage.width < 1 || stage.height < 1) return;
      const dx = (event.clientX - current.startX) / stage.width;
      const dy = (event.clientY - current.startY) / stage.height;
      const box = { ...current.crop };
      const handle = current.handle;
      if (handle === "move" || handle.includes("w")) {
        const nextX = handle === "move" ? box.x + dx : Math.min(box.x + dx, box.x + box.width - MIN);
        if (handle !== "move") box.width += box.x - nextX;
        box.x = nextX;
      }
      if (handle === "move" || handle.includes("n")) {
        const nextY = handle === "move" ? box.y + dy : Math.min(box.y + dy, box.y + box.height - MIN);
        if (handle !== "move") box.height += box.y - nextY;
        box.y = nextY;
      }
      if (handle.includes("e")) box.width += dx;
      if (handle.includes("s")) box.height += dy;
      onChange(clampCrop(box));
    }
    function onUp() {
      drag.current = null;
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [onChange, stage.height, stage.width]);

  const pixelW = sourceWidth ? Math.round(crop.width * sourceWidth) : 0;
  const pixelH = sourceHeight ? Math.round(crop.height * sourceHeight) : 0;

  return (
    <div>
      <div className="crop-preview" ref={wrapRef}>
        <video
          ref={videoRef}
          className="preview"
          src={src}
          controls={!enabled}
          onLoadedMetadata={() => {
            layout();
            setDuration(videoRef.current?.duration || 0);
          }}
          onTimeUpdate={() => setTime(videoRef.current?.currentTime || 0)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
        {enabled && stage.width > 0 && (
          <div
            className="crop-stage"
            style={{ left: stage.left, top: stage.top, width: stage.width, height: stage.height }}
          >
            <div
              className="crop-box"
              style={{
                left: `${crop.x * 100}%`,
                top: `${crop.y * 100}%`,
                width: `${crop.width * 100}%`,
                height: `${crop.height * 100}%`,
              }}
              onPointerDown={(event) => {
                event.preventDefault();
                drag.current = { handle: "move", startX: event.clientX, startY: event.clientY, crop };
              }}
            >
              {(["nw", "n", "ne", "e", "se", "s", "sw", "w"] as Handle[]).map((handle) => (
                <span
                  key={handle}
                  className={`crop-handle crop-handle-${handle}`}
                  onPointerDown={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    drag.current = { handle, startX: event.clientX, startY: event.clientY, crop };
                  }}
                />
              ))}
            </div>
          </div>
        )}
      </div>
      {enabled && (
        <div className="crop-transport">
          <button
            className="btn"
            type="button"
            onClick={() => {
              if (!videoRef.current) return;
              if (playing) videoRef.current.pause();
              else void videoRef.current.play();
            }}
          >
            {playing ? messages.config.cropPause : messages.config.cropPlay}
          </button>
          <input
            type="range"
            aria-label="Video position"
            min={0}
            max={duration || 0}
            step={0.1}
            value={Math.min(time, duration || 0)}
            onChange={(event) => {
              const next = Number(event.target.value);
              if (videoRef.current) videoRef.current.currentTime = next;
              setTime(next);
            }}
          />
        </div>
      )}
      <div className="crop-toolbar">
        <label className="check">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(event) => {
              if (event.target.checked && crop.width >= 0.99 && crop.height >= 0.99) {
                onChange(DEFAULT_CROP);
              }
              onToggle(event.target.checked);
            }}
          />
          {messages.config.crop}
        </label>
        {enabled && (
          <>
            <button className="btn" type="button" onClick={() => onChange(DEFAULT_CROP)}>
              {messages.config.cropReset}
            </button>
            <span className="crop-size">
              {pixelW && pixelH
                ? messages.config.cropSize
                    .replace("{width}", String(pixelW))
                    .replace("{height}", String(pixelH))
                : messages.config.cropHint}
            </span>
          </>
        )}
      </div>
      {enabled && <p className="hint">{messages.config.cropHint}</p>}
    </div>
  );
}

export { DEFAULT_CROP };
