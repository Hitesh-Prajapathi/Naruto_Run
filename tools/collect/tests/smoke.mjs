// Headless Chromium smoke test using Chrome's fake camera. Run: npm run build && npm run smoke
// It drives the real UI through a SHORT protocol (?fast), records real MediaRecorder clips from the
// fake device, downloads the zip and validates session.json against the schema.
// It does NOT cover Safari, Android Chrome, real cameras, or the full-length protocol.
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { chromium } from "playwright";
import { unzipSync } from "fflate";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";

const PORT = 5199;
const schema = JSON.parse(readFileSync(new URL("../../../schemas/collection_session_v1.schema.json", import.meta.url), "utf8"));
const server = spawn("npx", ["vite", "preview", "--port", String(PORT), "--strictPort"], { stdio: "ignore" });
const kill = () => server.kill();
process.on("exit", kill);

async function waitForServer() {
  for (let i = 0; i < 50; i++) {
    try { if ((await fetch(`http://localhost:${PORT}/`)).ok) return; } catch {}
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error("preview server did not start");
}

let failed = false;
let pageRef;
const ok = (name, cond, extra = "") => { console.log(`${cond ? "PASS" : "FAIL"} ${name} ${extra}`); if (!cond) failed = true; };

try {
  await waitForServer();
  const browser = await chromium.launch({ args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
  const ctx = await browser.newContext({ permissions: ["camera"], acceptDownloads: true });
  const page = await ctx.newPage();
  pageRef = page;
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  await page.goto(`http://localhost:${PORT}/?fast`);

  await page.check("#consent");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await page.getByRole("button", { name: "Continue to camera check" }).click();
  await page.getByRole("button", { name: "Looks good, start recording" }).click({ timeout: 20000 });

  for (let guard = 0; guard < 40; guard++) {
    const done = await page.locator("text=All done").count();
    if (done) break;
    const btn = page.locator("button:not([disabled])", { hasText: /\(Space\)/ });
    await btn.first().waitFor({ timeout: 30000 }).catch(() => {});
    if (await page.locator("text=All done").count()) break;
    await page.keyboard.press("Space");
    await page.waitForTimeout(300);
  }
  await page.waitForSelector("text=All done", { timeout: 30000 });

  const [dl] = await Promise.all([page.waitForEvent("download"), page.click("text=Download session")]);
  const files = unzipSync(new Uint8Array(readFileSync(await dl.path())));
  const session = JSON.parse(new TextDecoder().decode(files["session.json"]));
  const ajv = new Ajv2020({ allErrors: true, strict: true, strictRequired: false });
  addFormats(ajv);
  const validate = ajv.compile(schema);
  ok("session.json validates against schema", validate(session), JSON.stringify(validate.errors));
  ok("6 clips (2 seals x 2 takes + 1 combo + 1 free)", session.clips.length === 6, `got ${session.clips.length}`);
  ok("every clip has a non-trivial file", session.clips.every((c) => (files[c.file]?.length ?? 0) > 1000));
  ok("mime is webm (Chromium)", session.clips.every((c) => c.mime.startsWith("video/webm")), session.clips[0]?.mime);
  ok("session complete + mirrored_display", session.session.status === "complete" && session.device.mirrored_display === true);
  const seal = session.clips[0];
  const o = seal.phase_offsets_ms;
  console.log("first clip:", seal.clip_id, JSON.stringify(o), `duration=${seal.duration_ms}ms fps=${seal.fps_measured} ${seal.width}x${seal.height}`);
  ok("phase offsets near nominal (form~1000, hold~1400, hold_end~1900, end~2200; +-120 ms)",
    [[o.form_start, 1000], [o.hold_start, 1400], [o.hold_end, 1900], [o.release_end, 2200]].every(([a, b]) => Math.abs(a - b) <= 120));
  ok("duration >= release_end", seal.duration_ms >= o.release_end);
  await browser.close();
} catch (e) {
  console.error(e);
  if (pageRef) console.error("BODY:", (await pageRef.locator("body").innerText().catch(() => "?")).slice(0, 600));
  failed = true;
} finally {
  kill();
}
process.exit(failed ? 1 : 0);
