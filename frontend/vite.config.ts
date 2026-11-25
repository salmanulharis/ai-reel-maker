import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The narrated-video composition lives in ../remotion/src (shared with the renderer). Its
// imports of react, remotion and zod resolve to this app's copies, so the preview and the
// render use the same code without a second React.
const reel = fileURLToPath(new URL("../remotion/src", import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@reel": reel },
    dedupe: ["react", "react-dom", "remotion", "zod"],
  },
  server: {
    port: 5173,
    fs: { allow: [".", reel] },
    proxy: {
      "/api": "http://127.0.0.1:5000",
      "/socket.io": {
        target: "http://127.0.0.1:5000",
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
