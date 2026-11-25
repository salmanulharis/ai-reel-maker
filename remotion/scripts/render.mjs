// Draw one chunk of a narrated video (no sound): node scripts/render.mjs <job.json>
// job.json: {bundle, composition, props (the render spec), frameRange: [first, last], output,
// concurrency, crf}. Prints JSON lines: {"type":"progress","value":0..1}, then
// {"type":"done"} or {"type":"error","message":...}.
import { readFileSync } from "node:fs";
import { renderMedia, selectComposition } from "@remotion/renderer";

const job = JSON.parse(readFileSync(process.argv[2], "utf8"));
const inputProps = { spec: job.props, audio: false };
let last = -1;
try {
  const composition = await selectComposition({ serveUrl: job.bundle, id: job.composition, inputProps });
  await renderMedia({
    composition,
    serveUrl: job.bundle,
    codec: "h264",
    crf: job.crf,
    outputLocation: job.output,
    inputProps,
    frameRange: job.frameRange,
    concurrency: job.concurrency,
    muted: true,
    onProgress: ({ progress }) => {
      const value = Math.floor(progress * 100) / 100;
      if (value !== last) {
        last = value;
        console.log(JSON.stringify({ type: "progress", value }));
      }
    },
  });
  console.log(JSON.stringify({ type: "done" }));
  process.exit(0);
} catch (error) {
  console.log(JSON.stringify({ type: "error", message: String(error?.message || error).slice(0, 2000) }));
  process.exit(1);
}
