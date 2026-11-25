// Download the headless browser Remotion draws with (one time): node scripts/browser.mjs
import { ensureBrowser } from "@remotion/renderer";

try {
  await ensureBrowser({
    onBrowserDownload: () => {
      console.log(JSON.stringify({ type: "progress", value: 0, message: "Downloading the browser" }));
      return {
        version: null,
        onProgress: ({ percent }) => console.log(JSON.stringify({ type: "progress", value: percent })),
      };
    },
  });
  console.log(JSON.stringify({ type: "done" }));
  process.exit(0);
} catch (error) {
  console.log(JSON.stringify({ type: "error", message: String(error?.message || error) }));
  process.exit(1);
}
