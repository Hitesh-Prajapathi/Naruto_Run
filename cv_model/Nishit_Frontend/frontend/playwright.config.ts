import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 90_000,
  // These are all GPU-bound WebGL tests sharing one headless GPU, and the
  // scene is not cheap to build (skinned character + retarget bake + a few
  // hundred trees + shadow maps). Running them 8-way parallel made every
  // test slower and pushed model loading past its timeouts; two at a time is
  // both faster overall and far more stable.
  fullyParallel: true,
  workers: 2,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: false,
    timeout: 30_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
