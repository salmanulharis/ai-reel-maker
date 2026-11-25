// The narrated video, played in the browser with Remotion's Player and the same composition the
// renderer draws (remotion/src). Loaded only when a preview is opened (React.lazy).
import { useMemo } from "react";
import { Player } from "@remotion/player";
import { Reel } from "@reel/Reel";
import { durationInFrames, parseSpec } from "@reel/spec";

export default function ReelPreview({ spec }: { spec: Record<string, unknown> }) {
  const parsed = useMemo(() => {
    try {
      return { spec: parseSpec(spec), error: "" };
    } catch (error) {
      return { spec: null, error: String((error as Error).message || error) };
    }
  }, [spec]);
  if (!parsed.spec) return <p className="error-text">{parsed.error}</p>;
  const portrait = parsed.spec.height > parsed.spec.width;
  return (
    <Player
      component={Reel}
      inputProps={{ spec: parsed.spec, audio: true }}
      durationInFrames={durationInFrames(parsed.spec)}
      fps={parsed.spec.fps}
      compositionWidth={parsed.spec.width}
      compositionHeight={parsed.spec.height}
      controls
      acknowledgeRemotionLicense
      style={{ width: "100%", maxWidth: portrait ? 360 : 720, aspectRatio: `${parsed.spec.width} / ${parsed.spec.height}`, borderRadius: 8 }}
    />
  );
}
