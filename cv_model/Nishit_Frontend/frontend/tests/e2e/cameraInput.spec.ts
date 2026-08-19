import { expect, test, type Page } from "@playwright/test";

/**
 * Feature Brief 05 in a real browser -- the startup gate, the camera UI, and
 * the non-regression checks.
 *
 * No camera and no human: pose frames are injected through `__cvDebug`,
 * which is the same entry point the live camera uses (§9.1). What that
 * cannot cover is the camera hardware itself and the real latency, both of
 * which need the manual protocol in §9.7.
 */

interface CvDebug {
  enabled: () => boolean;
  gateState: () => string;
  gateBlocking: () => boolean;
  trackingStatus: () => string;
  previewVisible: () => boolean;
  laneEvents: () => number;
  jumpEvents: () => number;
  injectSample: (sample: unknown) => void;
  simulateCameraReady: () => void;
  simulateCameraStatus: (status: string) => void;
  chooseKeyboard: () => void;
  recalibrate: () => void;
}

interface BossDebug {
  distance: () => number;
  playerLane: () => number;
}

/**
 * Call a zero-argument method on `window.__cvDebug`.
 *
 * By name, not by callback: Playwright serialises the function to the page,
 * so a closure like `(d) => d.gateState()` arrives with `d` undefined. The
 * function has to reach for `window` itself.
 */
function cvCall<T>(page: Page, method: keyof CvDebug): Promise<T> {
  return page.evaluate(
    (name) => (window as unknown as { __cvDebug: Record<string, () => unknown> }).__cvDebug[name]!(),
    method as string,
  ) as Promise<T>;
}

function cvStatus(page: Page, status: string): Promise<void> {
  return page.evaluate(
    (value) => (window as unknown as { __cvDebug: CvDebug }).__cvDebug.simulateCameraStatus(value),
    status,
  );
}

function distance(page: Page): Promise<number> {
  return page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.distance());
}

function playerLane(page: Page): Promise<number> {
  return page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.playerLane());
}

/**
 * `?cv=1` by default: the flag ships OFF until the camera path is signed
 * off, so the camera tests opt in explicitly and the R-01 case below opts
 * out just as explicitly. Neither relies on what the default happens to be.
 */
async function open(page: Page, query = "?cv=1"): Promise<void> {
  // `domcontentloaded`, not the default `load`: the page keeps fetching the
  // 24 MB boss model in the background, so waiting for `load` means every
  // navigation waits on an asset none of these tests need -- which timed
  // the whole file out. `__cvDebug` below is the real readiness signal.
  await page.goto(`/scene.html${query}`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => Boolean((window as unknown as { __cvDebug?: CvDebug }).__cvDebug),
    undefined,
    { timeout: 60000 },
  );
}

/** Build a synthetic pose frame in *pipeline* space, as the model emits it. */
function poseFrame(options: {
  /** Shoulder tilt as a sine, pipeline space: positive = player's own left. */
  leanX?: number;
  /** Shoulder midpoint y. Smaller = higher up the frame. */
  riseY?: number;
  /** Shoulder width in frame heights. */
  bodyScale?: number;
  present?: boolean;
  label?: string;
  emitted?: string | null;
  timestampMs: number;
}): unknown {
  const {
    leanX = 0,
    riseY = 0.4,
    bodyScale = 0.3,
    present = true,
    label = "idle",
    emitted = null,
    timestampMs,
  } = options;
  const aspect = 640 / 360;
  const landmarks: Array<[number, number]> = [];
  if (present) {
    // Upper body only: two shoulders, rolled by the requested sine. No hips —
    // the adapter does not read them, and supplying them would let this
    // fixture pass while the real chest-up framing failed.
    const dy = leanX * bodyScale;
    const dx = Math.sqrt(Math.max(bodyScale ** 2 - dy ** 2, 0)) / aspect;
    for (let i = 0; i < 33; i += 1) landmarks.push([0.5, riseY]);
    landmarks[11] = [0.5 + dx / 2, riseY + dy / 2];
    landmarks[12] = [0.5 - dx / 2, riseY - dy / 2];
  }
  // Mirrors what poseAdapter.toPoseSample produces, since __cvDebug takes a
  // finished PoseSample (the adapter has already run inside the app for the
  // live path; here the test supplies the post-adapter shape directly).
  return {
    timestampMs,
    personPresent: present,
    stableLabel: label === "bending_left" ? "bending_right" : label === "bending_right" ? "bending_left" : label,
    emittedMovement:
      emitted === "bending_left" ? "bending_right" : emitted === "bending_right" ? "bending_left" : emitted,
    leanX: -leanX,
    jumpHeight: 0,
    lateralVelocity: 0,
    displayLandmarks: landmarks.map(([x, y]) => [1 - x, y]),
    riseY,
    bodyScale,
  };
}

/** Feed frames at 30fps, letting the render loop run between batches. */
async function feed(
  page: Page,
  count: number,
  options: Parameters<typeof poseFrame>[0] extends infer T
    ? Omit<T & object, "timestampMs">
    : never,
  startMs = Date.now(),
): Promise<number> {
  for (let i = 0; i < count; i += 1) {
    const frame = poseFrame({ ...options, timestampMs: startMs + i * 33 });
    await page.evaluate(
      (payload) => (window as unknown as { __cvDebug: CvDebug }).__cvDebug.injectSample(payload),
      frame,
    );
    if (i % 5 === 4) await page.waitForTimeout(16);
  }
  return startMs + count * 33;
}

test.describe("§0.2 / R-01: with the flag off nothing changes", () => {
  test("requests no camera, shows no new UI, and plays normally", async ({ page }) => {
    const requested: string[] = [];
    await page.addInitScript(() => {
      const media = navigator.mediaDevices;
      if (media) {
        const original = media.getUserMedia.bind(media);
        media.getUserMedia = ((constraints: MediaStreamConstraints) => {
          (window as unknown as { __gum: number }).__gum =
            ((window as unknown as { __gum?: number }).__gum ?? 0) + 1;
          return original(constraints);
        }) as typeof media.getUserMedia;
      }
    });
    await open(page, "?cv=0");

    expect(await cvCall(page, "enabled")).toBe(false);
    expect(await cvCall(page, "gateBlocking")).toBe(false);
    await expect(page.locator("#startup-gate")).toBeHidden();
    await expect(page.locator("#camera-hud")).toBeHidden();
    expect(await page.evaluate(() => (window as unknown as { __gum?: number }).__gum ?? 0)).toBe(0);
    void requested;

    // The world is running, exactly as in the approved build.
    const before = await distance(page);
    await page.waitForTimeout(600);
    expect(
      await distance(page),
    ).toBeGreaterThan(before);
  });
});

test.describe("§4: the startup gate", () => {
  test("blocks the run until the player is detected and calibrated (I-01)", async ({ page }) => {
    test.setTimeout(150_000);
    await open(page);

    // The gate is up and the world is frozen.
    await expect(page.locator("#startup-gate")).toBeVisible();
    // ...and actually *painted*. Playwright's toBeVisible ignores opacity,
    // and the panel reuses the Game Over classes, whose reveal rule is
    // ID-scoped -- it rendered as a blank screen until that was fixed.
    expect(
      await page.locator("#startup-gate .go-panel").evaluate((el) => Number(getComputedStyle(el).opacity)),
    ).toBeGreaterThan(0.9);
    await page.screenshot({ path: "test-results/startup-gate.png" });
    expect(await cvCall(page, "gateBlocking")).toBe(true);
    const frozen = await distance(page);
    await page.waitForTimeout(500);
    expect(
      await distance(page),
    ).toBeCloseTo(frozen, 1);

    await cvCall(page, "simulateCameraReady");
    await expect.poll(() => cvCall(page, "gateState"), { timeout: 10000 }).toBe("PLAYER_DETECTION");

    // Sustained detection, then calibration, then ready.
    let now = Date.now();
    for (let batch = 0; batch < 12; batch += 1) {
      now = await feed(page, 10, {}, now);
      await page.waitForTimeout(120);
      const state = await cvCall(page, "gateState");
      if (state === "RUNNING") break;
    }
    await expect.poll(() => cvCall(page, "gateState"), { timeout: 30000 }).toBe("RUNNING");
    await expect(page.locator("#startup-gate")).toBeHidden();

    // And now the world moves.
    const started = await distance(page);
    await page.waitForTimeout(600);
    expect(
      await distance(page),
    ).toBeGreaterThan(started);
  });

  test("offers a playable keyboard path when the camera is refused (I-02)", async ({ page }) => {
    await open(page);
    await cvStatus(page, "camera_denied");
    await expect(page.locator("#startup-gate .gate-title")).toContainText("Camera blocked");
    await expect(page.locator("#startup-gate .gate-keyboard")).toBeVisible();

    await page.locator("#startup-gate .gate-keyboard").click();
    await expect(page.locator("#startup-gate")).toBeHidden();
    expect(await cvCall(page, "gateBlocking")).toBe(false);

    // Keyboard is fully operational.
    const lane = await playerLane(page);
    await page.keyboard.press(lane > 0 ? "ArrowLeft" : "ArrowRight");
    await page.waitForTimeout(400);
    expect(
      await playerLane(page),
    ).not.toBe(lane);
  });

  test("names the service failure distinctly (I-03)", async ({ page }) => {
    await open(page);
    await cvStatus(page, "service_unavailable");
    await expect(page.locator("#startup-gate .gate-title")).toContainText("Detection service not running");
    await expect(page.locator("#startup-gate .gate-keyboard")).toBeVisible();
  });
});

/**
 * These five share one page. None of them changes gate state irreversibly,
 * and a cold load costs ~25s here -- eight of them timed the file out. The
 * gate tests above still get their own page, because the gate is one-shot
 * per load and cannot be rewound.
 */
test.describe("§7: camera UI", () => {
  test.describe.configure({ mode: "serial" });
  let shared: Page;

  test.beforeAll(async ({ browser }) => {
    shared = await browser.newPage();
    await open(shared);
  });
  test.afterAll(async () => {
    await shared?.close();
  });

  test("👁 preview sits bottom-right and overlaps nothing (§0.1)", async () => {
    const page = shared;
    await expect(page.locator("#camera-hud")).toBeVisible();
    const preview = page.locator(".cam-preview");
    await expect(preview).toBeVisible();

    const box = (await preview.boundingBox())!;
    const viewport = page.viewportSize()!;
    // Bottom-right quadrant.
    expect(box.x).toBeGreaterThan(viewport.width * 0.5);
    expect(box.y).toBeGreaterThan(viewport.height * 0.5);

    // It must not overlap any reserved region (§0.1).
    for (const selector of ["#game-hud .hud-primary", ".scene-controls-hint", "#combat-hud"]) {
      const other = page.locator(selector).first();
      if (!(await other.isVisible().catch(() => false))) continue;
      const otherBox = await other.boundingBox();
      if (!otherBox) continue;
      const overlaps =
        box.x < otherBox.x + otherBox.width &&
        box.x + box.width > otherBox.x &&
        box.y < otherBox.y + otherBox.height &&
        box.y + box.height > otherBox.y;
      expect(overlaps, `camera preview overlaps ${selector}`).toBe(false);
    }

    await page.screenshot({ path: "test-results/camera-hud.png" });
  });

  test("preview is mirrored, and toggles with C (§5.1, §7.1)", async () => {
    const page = shared;
    const transform = await page
      .locator(".cam-video")
      .evaluate((el) => getComputedStyle(el).transform);
    // scaleX(-1) -> matrix(-1, 0, 0, 1, 0, 0)
    expect(transform.startsWith("matrix(-1")).toBe(true);

    expect(await cvCall(page, "previewVisible")).toBe(true);
    await page.keyboard.press("KeyC");
    expect(await cvCall(page, "previewVisible")).toBe(false);
    await expect(page.locator(".cam-preview")).toBeHidden();
    await page.keyboard.press("KeyC");
    await expect(page.locator(".cam-preview")).toBeVisible();
  });

  test("§7.2 tracking indicator is present and starts as no-signal", async () => {
    const page = shared;
    await expect(page.locator(".cam-indicator")).toBeVisible();
    await expect(page.locator(".cam-indicator")).toContainText("camera off");
  });

  test("§7.3 debug overlay lists lean, thresholds, return-to-neutral, jump", async () => {
    const page = shared;
    await page.keyboard.press("Backquote");
    const panel = page.locator("#scene-debug-hud");
    await expect(panel).toBeVisible();
    // The panel repaints on a frame-counter tick, and this machine renders
    // at ~6fps headless, so give it more than the 5s default before
    // concluding the overlay is missing content.
    await expect(panel).toContainText("body input", { timeout: 20000 });
    for (const label of [
      "lean",
      "lean state",
      "returned to neutral",
      "jump rise",
      "jump velocity",
      "jump refractory",
      "baseline",
      "event rate",
    ]) {
      await expect(panel).toContainText(label);
    }
  });
});
