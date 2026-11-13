/// <reference types="vitest/config" />
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

// Frontend build config. No frontend framework -- DOM/CSS/Canvas only, per
// game_implementation_plan.md Phase 7 stack decision. Three.js (added in
// Phase C) is a plain runtime dependency, not a framework choice.
export default defineConfig({
  // No explicit `root`: Vite defaults it to this config file's directory.
  server: {
    // Explicit IPv4 loopback: matches serve_transport.py's 127.0.0.1 default
    // so "the backend" and "the frontend" mean the same host consistently.
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
  build: {
    target: "es2022",
    sourcemap: true,
    rollupOptions: {
      // Two independent dev-harness pages for now (index.html: Phase B
      // camera/transport; scene.html: Phase C environment/map). Vite's dev
      // server serves any .html file automatically; production builds need
      // every entry listed explicitly or they're silently dropped.
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        scene: fileURLToPath(new URL("./scene.html", import.meta.url)),
      },
    },
  },
  test: {
    environment: "jsdom",
    include: ["tests/unit/**/*.test.ts"],
    reporters: "default",
  },
});
