// Build the composition once: node scripts/bundle.mjs <entry> <out dir>
// Prints JSON lines: {"type":"progress","value":0..1}, then {"type":"done"} or {"type":"error"}.
import { bundle } from "@remotion/bundler";

const [entryPoint, outDir] = process.argv.slice(2);
let last = -1;
try {
  await bundle({
    entryPoint,
    outDir,
    onProgress: (percent) => {
      const value = Math.floor(percent) / 100;
      if (value !== last) {
        last = value;
        console.log(JSON.stringify({ type: "progress", value }));
      }
    },
  });
  console.log(JSON.stringify({ type: "done" }));
  process.exit(0);
} catch (error) {
  console.log(JSON.stringify({ type: "error", message: String(error?.message || error) }));
  process.exit(1);
}
