/// <reference types="vitest" />
import { defineConfig } from "vite";

// base './' so the build works from a GitHub Pages sub-path (user.github.io/repo/).
export default defineConfig({
  base: "./",
  build: { target: "es2022", outDir: "dist" },
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
});
