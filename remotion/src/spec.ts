// The timeline of a narrated video, written by app/compose/render_spec.py (RenderSpec).
// Kept in step with spec.schema.json, which a Python test regenerates and compares.
import { z } from "zod";

const word = z.object({ text: z.string(), start: z.number(), end: z.number() });

export const specSchema = z.object({
  version: z.number().int(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  fps: z.number().positive(),
  duration: z.number().nonnegative(),
  video: z.object({
    kind: z.enum(["base", "clips"]),
    src: z.string().default(""),
    clips: z
      .array(z.object({ src: z.string(), start: z.number(), end: z.number(), source_start: z.number() }))
      .default([]),
  }),
  texts: z
    .array(z.object({ kind: z.enum(["title", "beat", "cta"]), text: z.string(), start: z.number(), end: z.number() }))
    .default([]),
  captions: z
    .array(z.object({ start: z.number(), end: z.number(), text: z.string(), words: z.array(word).default([]) }))
    .default([]),
  caption_style: z.enum(["none", "plain", "clean", "bold", "boxed"]).default("none"),
  narration: z
    .array(z.object({ beat: z.string(), src: z.string(), start: z.number(), duration: z.number() }))
    .default([]),
  accent: z.string().default("#FFD700"),
});

export type RenderSpec = z.infer<typeof specSchema>;

/** Props of the Reel composition. ``audio`` plays the source sound and narration (the preview);
 * rendering is silent because FFmpeg mixes the sound afterwards. */
export interface ReelProps {
  spec: RenderSpec;
  audio?: boolean;
  [key: string]: unknown;
}

export function parseSpec(value: unknown): RenderSpec {
  const result = specSchema.safeParse(value);
  if (!result.success) {
    const issue = result.error.issues[0];
    throw new Error(`Invalid render spec at ${issue.path.join(".") || "(root)"}: ${issue.message}`);
  }
  return result.data;
}

export function durationInFrames(spec: RenderSpec): number {
  return Math.max(1, Math.round(spec.duration * spec.fps));
}

export const EMPTY_SPEC: RenderSpec = {
  version: 1,
  width: 1080,
  height: 1920,
  fps: 30,
  duration: 3,
  video: { kind: "clips", src: "", clips: [] },
  texts: [{ kind: "title", text: "Preview", start: 0, end: 2.5 }],
  captions: [],
  caption_style: "none",
  narration: [],
  accent: "#FFD700",
};
