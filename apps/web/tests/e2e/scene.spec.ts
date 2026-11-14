import { expect, test, type Page } from "@playwright/test";

/**
 * Real-browser verification of scene.html -- everything tests/unit/*.test.ts
 * can't cover because it needs an actual WebGL context (jsdom has none).
 *
 * Note on the debug panel: it now starts hidden and is toggled with `H`
 * (naruto_run_agent_brief.md §I -- the player-facing HUD owns the screen,
 * the dev panel stays available but out of the way). Tests that assert on
 * engine internals therefore open it first via `openDebugPanel`.
 */

/** Opens the debug panel and waits for it to start reporting. */
async function openDebugPanel(page: Page) {
  const hud = page.locator("#scene-debug-hud");
  // Wait for main() to have run (keydown listener attached, scene built)
  // rather than a fixed sleep -- a keypress sent before then is simply lost.
  await expect(page.locator("#game-hud")).toContainText("Distance", { timeout: 20000 });
  await page.keyboard.press("KeyH");
  await expect(hud).toContainText("fps", { timeout: 10000 });
  return hud;
}

test.describe("scene.html", () => {
  test("renders without console errors and keeps the render loop running", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => consoleErrors.push(error.message));

    await page.goto("/scene.html");
    await expect(page.locator("#scene-canvas")).toBeVisible();

    const distance = page.locator("#game-hud [data-distance]");
    await expect(distance).toContainText("m", { timeout: 20000 });

    // Assert the loop is *advancing*, not a frame-rate number. Under N
    // parallel Playwright workers sharing one headless GPU the measured FPS
    // says nothing useful about real performance (it lands in the teens here
    // purely from contention), so pinning a threshold would be a flaky test
    // masquerading as a benchmark. Distance only increases when frames are
    // being simulated, which is the property this test actually cares about.
    const readDistance = async () =>
      Number((await distance.innerText()).replace(/[^\d]/g, ""));

    const first = await readDistance();
    await page.waitForTimeout(1200);
    const second = await readDistance();
    expect(second, "the render/simulation loop should keep advancing").toBeGreaterThan(first);

    expect(consoleErrors).toEqual([]);
  });

  test("the player HUD shows distance and lives", async ({ page }) => {
    await page.goto("/scene.html");
    const gameHud = page.locator("#game-hud");
    await expect(gameHud).toContainText("Distance", { timeout: 20000 });
    await expect(gameHud).toContainText("Speed");
    // Three life pips, all full at the start.
    await expect(page.locator("#game-hud [data-lives]")).toContainText("●●●");
  });

  test("the debug panel is hidden by default and toggles with H", async ({ page }) => {
    await page.goto("/scene.html");
    await expect(page.locator("#game-hud")).toContainText("Distance", { timeout: 20000 });

    const hud = page.locator("#scene-debug-hud");
    await expect(hud).toBeHidden();

    await page.keyboard.press("KeyH");
    await expect(hud).toBeVisible();

    await page.keyboard.press("KeyH");
    await expect(hud).toBeHidden();
  });

  test("keyboard lane change moves the player", async ({ page }) => {
    await page.goto("/scene.html");
    const hud = await openDebugPanel(page);

    await page.keyboard.press("KeyD"); // move right
    await expect(hud).toContainText(/lane\s*\n?\s*2/, { timeout: 5000 });
  });

  test("jump makes the character briefly airborne", async ({ page }) => {
    await page.goto("/scene.html");
    const hud = await openDebugPanel(page);

    await page.keyboard.press("Space");

    // Poll generously rather than sleeping a fixed time: sceneRoot clamps
    // each frame's delta to at most 1/15s (so one real stall can't teleport
    // anything), which means under heavy parallel-worker CPU contention the
    // in-game clock runs slower than wall-clock. The jump's own timing
    // correctness is covered precisely in tests/unit/laneController.test.ts;
    // this only asserts the state is reachable in a real browser.
    await expect(hud).toContainText(/airborne\s*\n?\s*yes/, { timeout: 10000 });
    await expect(hud).toContainText(/airborne\s*\n?\s*no/, { timeout: 12000 });
  });

  test("obstacle pool stops growing after the first lap (allocation gate)", async ({ page }) => {
    test.setTimeout(90_000); // this test waits 30s across two checkpoints
    await page.goto("/scene.html");
    const hud = await openDebugPanel(page);

    const readConstructed = async () => {
      const text = await hud.innerText();
      return Number(text.match(/obstacles constructed\s*\n?\s*(\d+)/)?.[1] ?? -1);
    };

    // Since Feature Brief 02 added the Game Over screen, an idle player now
    // dies after three obstacle hits and the world freezes -- this test used
    // to rely on the run continuing forever. Dismissing the panel keeps the
    // run going so the pool can still be observed over a long stretch (and
    // incidentally exercises restart repeatedly).
    const dismissGameOverIfShown = async () => {
      if (await page.locator("#game-over").isVisible()) {
        await page.keyboard.press("Enter");
        await expect(page.locator("#game-over")).toBeHidden();
        // The panel steals nothing, but the debug panel state survives a
        // restart, so no need to re-open it.
      }
    };

    const waitDismissing = async (totalMs: number) => {
      const step = 1_000;
      for (let elapsed = 0; elapsed < totalMs; elapsed += step) {
        await page.waitForTimeout(step);
        await dismissGameOverIfShown();
      }
    };

    // SEGMENT_1 is 25s of authored obstacles; give it time to fully drain
    // and loop at least once via ObstacleSpawner.reset().
    await waitDismissing(15_000);
    const midRun = await readConstructed();
    expect(midRun).toBeGreaterThan(0);

    await waitDismissing(15_000);
    // The pool may legitimately grow a little across restarts if a restart
    // lands while several obstacles are live, so assert it is *bounded*
    // rather than frozen -- the gate is "no per-spawn allocation", not
    // "never constructs another mesh".
    const later = await readConstructed();
    expect(later).toBeGreaterThanOrEqual(midRun);
    expect(later).toBeLessThanOrEqual(midRun + 4);
  });

  test("screenshot for visual review", async ({ page }) => {
    await page.goto("/scene.html");
    await expect(page.locator("#game-hud")).toContainText("Distance", { timeout: 20000 });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: "test-results/scene-screenshot.png" });
  });
});

type NarutoDebug = { isAnimating: () => boolean; clipTime: () => number };

/**
 * Regression cover for the single biggest bug this overhaul fixed: the
 * character rendering correctly but frozen in its bind T-pose because the
 * animation was never actually driven. That failure is silent -- no console
 * error, nothing missing from the scene -- so it needs an explicit assertion
 * that the clip's playback head is moving.
 */
test.describe("character run animation", () => {
  test("a run clip is loaded and its playback head advances", async ({ page }) => {
    await page.goto("/scene.html");

    // The hook only appears once the character model has finished loading.
    await page.waitForFunction(
      () => Boolean((window as unknown as { __narutoDebug?: NarutoDebug }).__narutoDebug),
      undefined,
      { timeout: 20000 },
    );

    const isAnimating = await page.evaluate(
      () => (window as unknown as { __narutoDebug: NarutoDebug }).__narutoDebug.isAnimating(),
    );
    expect(isAnimating, "a retargeted run clip should be playing").toBe(true);

    const readClipTime = () =>
      page.evaluate(() => (window as unknown as { __narutoDebug: NarutoDebug }).__narutoDebug.clipTime());

    const first = await readClipTime();
    await page.waitForTimeout(700);
    const second = await readClipTime();

    // The clip loops (~0.56s), so "advanced" means "changed", not ">".
    expect(second, "the animation mixer must actually be ticking").not.toBe(first);
  });
});
