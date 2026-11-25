import React from "react";
import { Composition, type CalculateMetadataFunction } from "remotion";
import { Reel } from "./Reel";
import { EMPTY_SPEC, durationInFrames, parseSpec, type ReelProps } from "./spec";

// The size, frame rate and length come from the spec (checked here, with a clear error).
const calculateMetadata: CalculateMetadataFunction<ReelProps> = ({ props }) => {
  const spec = parseSpec(props.spec);
  return {
    props: { ...props, spec },
    durationInFrames: durationInFrames(spec),
    fps: spec.fps,
    width: spec.width,
    height: spec.height,
  };
};

const defaultProps: ReelProps = { spec: EMPTY_SPEC, audio: false };

export const RemotionRoot: React.FC = () => (
  <Composition
    id="Reel"
    component={Reel}
    defaultProps={defaultProps}
    durationInFrames={durationInFrames(EMPTY_SPEC)}
    fps={EMPTY_SPEC.fps}
    width={EMPTY_SPEC.width}
    height={EMPTY_SPEC.height}
    calculateMetadata={calculateMetadata}
  />
);
