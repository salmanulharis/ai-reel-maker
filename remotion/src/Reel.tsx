// The narrated video: the picture (one finished video, or source clips in the preview), with the
// title, beat texts, captions and closing line drawn over it. Fonts are the system's (nothing is
// downloaded while rendering).
import React from "react";
import {
  AbsoluteFill,
  Audio,
  Easing,
  interpolate,
  OffthreadVideo,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import type { ReelProps, RenderSpec } from "./spec";

const FONT = "'Segoe UI', Inter, 'Helvetica Neue', Arial, sans-serif";

export const Reel: React.FC<ReelProps> = ({ spec, audio = false }) => {
  const { fps } = useVideoConfig();
  const voiced = spec.narration.length > 0;
  return (
    <AbsoluteFill style={{ backgroundColor: "black", fontFamily: FONT }}>
      <Picture spec={spec} audio={audio} sourceVolume={voiced ? 0.35 : 1} />
      {spec.texts.map((item, index) => {
        const frames = Math.max(1, Math.round((item.end - item.start) * fps));
        return (
          <Sequence key={`text-${index}`} from={Math.round(item.start * fps)} durationInFrames={frames} layout="none">
            {item.kind === "beat" ? (
              <BeatText text={item.text} spec={spec} frames={frames} />
            ) : (
              <TitleText text={item.text} spec={spec} kind={item.kind} frames={frames} />
            )}
          </Sequence>
        );
      })}
      {spec.caption_style !== "none" &&
        spec.captions.map((caption, index) => (
          <Sequence
            key={`caption-${index}`}
            from={Math.round(caption.start * fps)}
            durationInFrames={Math.max(1, Math.round((caption.end - caption.start) * fps))}
            layout="none"
          >
            <Caption caption={caption} spec={spec} />
          </Sequence>
        ))}
      {audio &&
        spec.narration.map((line, index) => (
          <Sequence key={`voice-${index}`} from={Math.round(line.start * fps)} layout="none">
            <Audio src={line.src} />
          </Sequence>
        ))}
    </AbsoluteFill>
  );
};

const Picture: React.FC<{ spec: RenderSpec; audio: boolean; sourceVolume: number }> = ({ spec, audio, sourceVolume }) => {
  const { fps } = useVideoConfig();
  const cover: React.CSSProperties = { width: "100%", height: "100%", objectFit: "cover" };
  if (spec.video.kind === "base") {
    return spec.video.src ? <OffthreadVideo src={spec.video.src} muted style={cover} /> : null;
  }
  return (
    <>
      {spec.video.clips.map((clip, index) => (
        <Sequence
          key={`clip-${index}`}
          from={Math.round(clip.start * fps)}
          durationInFrames={Math.max(1, Math.round((clip.end - clip.start) * fps))}
        >
          <OffthreadVideo
            src={clip.src}
            startFrom={Math.round(clip.source_start * fps)}
            muted={!audio}
            volume={sourceVolume}
            style={cover}
          />
        </Sequence>
      ))}
    </>
  );
};

function useEnter(durationFrames: number) {
  const frame = useCurrentFrame();
  const fade = Math.min(8, Math.floor(durationFrames / 3));
  const opacity = interpolate(frame, [0, fade, durationFrames - fade, durationFrames], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const rise = interpolate(frame, [0, fade + 4], [24, 0], { extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  return { opacity, rise };
}

const TitleText: React.FC<{ text: string; spec: RenderSpec; kind: "title" | "cta"; frames: number }> = ({
  text,
  spec,
  kind,
  frames,
}) => {
  const frame = useCurrentFrame();
  const { opacity, rise } = useEnter(frames);
  const short = Math.min(spec.width, spec.height);
  const scale = interpolate(frame, [0, 12], [0.92, 1], { extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  return (
    <AbsoluteFill
      style={{
        justifyContent: kind === "title" ? "center" : "flex-end",
        alignItems: "center",
        paddingBottom: kind === "cta" ? spec.height * 0.16 : 0,
      }}
    >
      <div
        style={{
          opacity,
          transform: `translateY(${rise}px) scale(${scale})`,
          maxWidth: "86%",
          textAlign: "center",
          color: "white",
          fontWeight: 800,
          fontSize: short * (kind === "title" ? 0.085 : 0.06),
          lineHeight: 1.15,
          textShadow: "0 4px 24px rgba(0,0,0,0.65), 0 2px 4px rgba(0,0,0,0.8)",
        }}
      >
        {text}
        <div style={{ margin: `${short * 0.02}px auto 0`, width: short * 0.12, height: short * 0.008, background: spec.accent, borderRadius: 4 }} />
      </div>
    </AbsoluteFill>
  );
};

const BeatText: React.FC<{ text: string; spec: RenderSpec; frames: number }> = ({ text, spec, frames }) => {
  const { opacity, rise } = useEnter(frames);
  const short = Math.min(spec.width, spec.height);
  return (
    <AbsoluteFill style={{ justifyContent: "flex-start", alignItems: "center", paddingTop: spec.height * 0.1 }}>
      <div
        style={{
          opacity,
          transform: `translateY(${-rise}px)`,
          background: "rgba(0,0,0,0.62)",
          color: "white",
          borderLeft: `${short * 0.01}px solid ${spec.accent}`,
          padding: `${short * 0.018}px ${short * 0.03}px`,
          borderRadius: short * 0.012,
          fontSize: short * 0.05,
          fontWeight: 700,
          maxWidth: "84%",
          textAlign: "center",
        }}
      >
        {text}
      </div>
    </AbsoluteFill>
  );
};

const Caption: React.FC<{ caption: RenderSpec["captions"][number]; spec: RenderSpec }> = ({ caption, spec }) => {
  const frame = useCurrentFrame();
  const now = caption.start + frame / spec.fps;
  const short = Math.min(spec.width, spec.height);
  const portrait = spec.height > spec.width;
  const style = spec.caption_style;
  const upper = style === "bold";
  const size = short * (style === "bold" ? 0.072 : 0.058);
  const words = caption.words.length ? caption.words : [{ text: caption.text, start: caption.start, end: caption.end }];
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: spec.height * (portrait ? 0.2 : 0.08) }}>
      <div
        style={{
          maxWidth: "88%",
          textAlign: "center",
          fontSize: size,
          fontWeight: style === "plain" ? 600 : 800,
          lineHeight: 1.2,
          color: "white",
          padding: style === "boxed" ? `${size * 0.18}px ${size * 0.4}px` : 0,
          background: style === "boxed" ? "rgba(24,24,24,0.92)" : "transparent",
          borderRadius: style === "boxed" ? size * 0.2 : 0,
          textShadow: style === "boxed" ? "none" : "0 0 6px rgba(0,0,0,0.9), 0 3px 3px rgba(0,0,0,0.9)",
        }}
      >
        {words.map((word, index) => {
          const active = style !== "plain" && caption.words.length > 0 && now >= word.start && now < (words[index + 1]?.start ?? caption.end);
          return (
            <span
              key={index}
              style={{
                color: active ? spec.accent : "white",
                display: "inline-block",
                transform: active && style === "bold" ? "scale(1.12)" : undefined,
                marginRight: size * 0.25,
              }}
            >
              {upper ? word.text.toUpperCase() : word.text}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
