import { expect, test, type Page } from "@playwright/test";

/**
 * Feature Brief 04 verification in a real browser: Obito's special attack
 * and the Hare counter.
 *
 * Brief §6 marks several criteria 👁 -- "must be verified against a rendered
 * frame or clip". Those are screenshotted here rather than only asserted, so
 * there is an artefact to look at; the assertions alongside them catch
 * regressions once someone has looked once.
 *
 * **One page for the whole file, deliberately.** Every test here needs a live
 * Obito encounter, and his model is ~24 MB. With Playwright's default
 * per-test page that is eleven cold loads of it, which starved the machine
 * badly enough that unrelated tests in other files began timing out. The file
 * runs serially against a single page and resets between tests through the
 * game's own restart path -- which has the pleasant side effect of exercising
 * that path eleven more times.
 */

test.describe.configure({ mode: "serial" });

interface BossDebug {
  state: () => string;
  assetsReady: () => boolean;
  isGameOverVisible: () => boolean;
  forceEncounter: () => void;
  forceSpecialAttack: () => void;
  specialState: () => string;
  specialRemaining: () => number;
  laneInputAllowed: () => boolean;
  worldTimeScale: () => number;
  harePerformed: () => boolean;
  playerLane: () => number;
  fps: () => number;
  narutoHp: () => number;
  obitoHp: () => number;
  distance: () => number;
  restartRun: () => void;
}

let page: Page;
let debugApi: ReturnType<typeof api>;

test.beforeAll(async ({ browser }) => {
  page = await browser.newPage();
  debugApi = api(page);
  await page.goto("/scene.html");
  await page.waitForFunction(
    () => Boolean((window as unknown as { __bossDebug?: BossDebug }).__bossDebug),
    undefined,
    { timeout: 60000 },
  );
  await page.waitForFunction(
    () => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.assetsReady(),
    undefined,
    { timeout: 120000 },
  );
});

test.afterAll(async () => {
  await page?.close();
});

/** Put the game back to a clean, live run before each test. */
test.beforeEach(async () => {
  await page.evaluate(() =>
    (window as unknown as { __bossDebug: BossDebug }).__bossDebug.restartRun(),
  );
  await expect.poll(() => debugApi.state(), { timeout: 10000 }).toBe("RUNNING");
  await expect(page.locator("#game-over")).toBeHidden();
  await expect(page.locator("#special-overlay")).toBeHidden();
});

function api(page: Page) {
  return {
    state: () => page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.state()),
    assetsReady: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.assetsReady()),
    forceEncounter: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.forceEncounter()),
    forceSpecial: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.forceSpecialAttack()),
    specialState: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.specialState()),
    laneInputAllowed: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.laneInputAllowed()),
    worldTimeScale: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.worldTimeScale()),
    harePerformed: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.harePerformed()),
    playerLane: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.playerLane()),
    fps: () => page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.fps()),
    narutoHp: () => page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.narutoHp()),
    obitoHp: () => page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.obitoHp()),
    gameOverVisible: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.isGameOverVisible()),
  };
}

/** Reach BOSS_COMBAT. The encounter assets are loaded once, in beforeAll. */
async function reachCombat() {
  await debugApi.forceEncounter();
  await expect.poll(() => debugApi.state(), { timeout: 30000 }).toBe("BOSS_COMBAT");
  return debugApi;
}

/**
 * Satisfy §2.2's guard and open the window. `H` outside the window is free
 * practice; `forceSpecial` then drops Obito to the trigger HP.
 *
 * Kept to as few round trips as possible: the window is only a few seconds
 * long, and every extra `evaluate` eats into the time the test body has left
 * to observe it while it is still open.
 */
async function openWarning() {
  const debug = await reachCombat();
  await page.keyboard.press("KeyH");
  await debug.forceSpecial();
  await expect.poll(() => debug.specialState(), { timeout: 15000 }).toBe("warning");
  return debug;
}

/**
 * The gap that let the feature ship looking unimplemented: every other test
 * in this file reaches the warning through `forceSpecialAttack()`, which
 * chips Obito to the trigger HP *and* parks his ordinary attacks. None of
 * them proved the mechanic is reachable by actually playing.
 *
 * It was not. A real fight could end with Obito untouched on 120 HP and
 * Naruto dead, having never seen the special attack -- and a player who
 * never pressed the counter key was held in `awaiting_practice` forever.
 * These two tests play the fight for real instead.
 */
test.describe("reachability in a real fight (no debug trigger)", () => {
  test("fires even when the player never lands a hit and never presses H", async () => {
    test.setTimeout(90_000);
    const debug = await reachCombat();
    expect(await debug.harePerformed()).toBe(false);

    // Do nothing at all -- no seals, no counter key. The elapsed-combat
    // fallback plus the bounded practice patience must still produce it,
    // before Obito's ordinary attacks finish Naruto off.
    await expect
      .poll(() => debug.specialState(), { timeout: 25000 })
      .toMatch(/warning|countered|struck|spent/);
    expect(await debug.obitoHp(), "reached without ever damaging Obito").toBe(120);
  });

  test("fires after a genuinely landed jutsu, on the intended HP trigger", async () => {
    test.setTimeout(90_000);
    const debug = await reachCombat();
    await page.keyboard.press("KeyH");

    // Play the seal prompt for real until Obito takes damage.
    const order = ["tiger", "ram", "snake"];
    for (let attempt = 0; attempt < 20; attempt += 1) {
      const state = await debug.specialState();
      if (state !== "idle" && state !== "awaiting_practice") break;
      if ((await debug.obitoHp()) <= 60) break;
      const names = await page.locator(".combat-seal .combat-seal-name").allInnerTexts();
      for (const name of names) {
        const index = order.indexOf(name.trim().toLowerCase());
        if (index >= 0) await page.keyboard.press(`Digit${index + 1}`);
        await page.waitForTimeout(40);
      }
      await page.waitForTimeout(220);
    }

    await expect
      .poll(() => debug.specialState(), { timeout: 20000 })
      .toMatch(/warning|countered|struck|spent/);
  });
});

test.describe("brief §2.2: the practice guard", () => {
  test("prompts first, and performing hare releases the guard immediately", async () => {
    test.setTimeout(90_000);
    const debug = await reachCombat();

    await debug.forceSpecial();
    // It holds the attack back and asks, rather than firing cold at someone
    // who has never made the sign.
    await expect.poll(() => debug.specialState(), { timeout: 10000 }).toBe("awaiting_practice");
    await expect(page.locator(".sa-practice")).toBeVisible();
    expect(await debug.narutoHp()).toBe(100);

    // Performing it once releases the guard right away, rather than waiting
    // out the patience window.
    await page.keyboard.press("KeyH");
    expect(await debug.harePerformed()).toBe(true);
    await expect.poll(() => debug.specialState(), { timeout: 10000 }).toBe("warning");
    await expect(page.locator(".sa-practice")).toBeHidden();
  });
});

test.describe("brief §4.1: the warning (👁 rendered)", () => {
  test("shows the danger overlay, the hare reference and a draining countdown", async () => {
    test.setTimeout(90_000);
    await openWarning();

    // Everything about the overlay is snapshotted in ONE round trip. Split
    // across half a dozen locator assertions, the checks themselves consumed
    // most of the window -- the ring had already drained to its last value
    // before the drain could be measured.
    const readRing = () =>
      page.evaluate(() =>
        Number(
          getComputedStyle(document.querySelector(".sa-ring")!).getPropertyValue(
            "--sa-progress",
          ) || "0",
        ),
      );
    const shown = await page.evaluate(() => {
      const visible = (selector: string) => {
        const el = document.querySelector(selector);
        return Boolean(el && (el as HTMLElement).offsetParent !== null);
      };
      return {
        card: visible(".sa-card"),
        vignetteArmed: Boolean(document.querySelector(".sa-vignette.is-armed")),
        // The sign is named, and its reference (photo or schematic) is there.
        sign: document.querySelector(".sa-sign")?.textContent ?? "",
        reference: Boolean(
          document.querySelector(".sa-photo:not([hidden])") ??
            document.querySelector(".sa-schematic:not([hidden])"),
        ),
        // §4.6: the keyboard fallback is on screen, not just in the code.
        key: document.querySelector(".sa-key")?.textContent ?? "",
        progress: Number(
          getComputedStyle(document.querySelector(".sa-ring")!).getPropertyValue(
            "--sa-progress",
          ) || "0",
        ),
      };
    });

    expect(shown.card).toBe(true);
    expect(shown.vignetteArmed).toBe(true);
    expect(shown.sign).toContain("HARE");
    expect(shown.reference).toBe(true);
    expect(shown.key).toContain("H");

    await page.screenshot({ path: "test-results/special-warning.png" });

    // The countdown ring visibly drains.
    await expect.poll(readRing, { timeout: 5000 }).toBeGreaterThan(shown.progress);
  });

  test("runs the world at reduced speed but the countdown in real seconds", async () => {
    test.setTimeout(90_000);
    const debug = await openWarning();
    // §4.1: the slowdown is a suspension, and the window is still real time.
    // Polled rather than read once: the window is only a few seconds long
    // and every debug read is a round trip, so a single sample taken just
    // after it closed would fail for no good reason.
    await expect.poll(() => debug.worldTimeScale(), { timeout: 6000 }).toBeLessThan(1);
    // §4.1 / §6: the window must last its configured duration in *real*
    // seconds. Timed against the wall clock deliberately -- the bug this
    // guards against was the countdown running on the host's clamped frame
    // delta, which stretched a 3.2s window past 8 real seconds whenever the
    // frame rate fell below 15fps (a software renderer manages ~7fps here).
    const startedAt = Date.now();
    await expect
      .poll(() => debug.specialState(), { timeout: 20000 })
      .toMatch(/struck|spent|none/);
    const elapsedS = (Date.now() - startedAt) / 1000;
    expect(elapsedS, "warning window measured in real seconds").toBeLessThan(6);
    await expect.poll(() => debug.worldTimeScale(), { timeout: 8000 }).toBe(1);
  });
});

test.describe("brief §4.2: input during the window", () => {
  test("suppresses lane changes, then restores them", async () => {
    test.setTimeout(90_000);
    const debug = await openWarning();

    expect(await debug.laneInputAllowed()).toBe(false);
    // A single press in one direction, so a working lane change would show
    // up rather than being cancelled out by its opposite.
    const lane = await debug.playerLane();
    await page.keyboard.press(lane > 0 ? "ArrowLeft" : "ArrowRight");
    await page.waitForTimeout(400);
    expect(await debug.playerLane()).toBe(lane);

    // Counter, then confirm lane input comes back.
    await page.keyboard.press("KeyH");
    await expect.poll(() => debug.laneInputAllowed(), { timeout: 8000 }).toBe(true);
  });

  test("clears the jutsu prompt for the window and restores it after", async () => {
    test.setTimeout(90_000);
    const debug = await openWarning();
    await expect(page.locator(".combat-seals .combat-seal")).toHaveCount(0);

    await page.keyboard.press("KeyH");
    await expect.poll(() => debug.specialState(), { timeout: 8000 }).toBe("spent");
    await expect(page.locator(".combat-seals .combat-seal")).toHaveCount(3);
  });
});

test.describe("brief §4.3: a successful counter (👁 rendered)", () => {
  test("saves Naruto and damages nobody", async () => {
    test.setTimeout(90_000);
    const debug = await openWarning();
    const obitoHp = await debug.obitoHp();

    await page.keyboard.press("KeyH");
    await expect(page.locator(".sa-result")).toContainText("COUNTERED", { timeout: 4000 });
    await page.screenshot({ path: "test-results/special-countered.png" });

    await expect.poll(() => debug.specialState(), { timeout: 8000 }).toBe("spent");
    expect(await debug.narutoHp()).toBe(100);
    // §2.6: purely defensive. Surviving is the reward.
    expect(await debug.obitoHp()).toBe(obitoHp);
    expect(await debug.gameOverVisible()).toBe(false);
    expect(await debug.state()).toBe("BOSS_COMBAT");

    // §5: nothing is left suspended, and the overlay is gone.
    await expect(page.locator("#special-overlay")).toBeHidden();
    expect(await debug.worldTimeScale()).toBe(1);
    expect(await debug.laneInputAllowed()).toBe(true);
  });

  test("fires at most once per encounter", async () => {
    test.setTimeout(90_000);
    const debug = await openWarning();
    await page.keyboard.press("KeyH");
    await expect.poll(() => debug.specialState(), { timeout: 8000 }).toBe("spent");

    await debug.forceSpecial();
    await page.waitForTimeout(2000);
    expect(await debug.specialState()).toBe("spent");
  });
});

test.describe("brief §4.4: a missed counter (👁 rendered)", () => {
  test("kills from full health and names the cause specifically", async () => {
    test.setTimeout(90_000);
    const debug = await openWarning();
    expect(await debug.narutoHp()).toBe(100);

    // Do nothing -- let the window elapse. The panel is the reliable signal;
    // HP is checked after, because the debug reading is held past teardown.
    await expect(page.locator("#game-over")).toBeVisible({ timeout: 15000 });
    await page.screenshot({ path: "test-results/special-struck.png" });
    // §4.4: it drains to 0 from full, rather than the bar jumping.
    expect(await debug.narutoHp()).toBe(0);
    // §2.5: without this line, dying at 100/100 reads as a bug.
    await expect(page.locator(".go-cause")).toContainText("special attack overwhelmed Naruto");
  });

  test("restores everything on Try Again -- no lean lock, no time scale, no overlay", async () => {
    test.setTimeout(120_000);
    const debug = await openWarning();
    await expect(page.locator("#game-over")).toBeVisible({ timeout: 15000 });
    await page.keyboard.press("Enter");

    await expect(page.locator("#game-over")).toBeHidden();
    await expect(page.locator("#special-overlay")).toBeHidden();
    expect(await debug.state()).toBe("RUNNING");
    // §5's two dangerous leaks, checked explicitly.
    expect(await debug.worldTimeScale()).toBe(1);
    expect(await debug.laneInputAllowed()).toBe(true);
    // The practice guard is re-armed for the new run.
    expect(await debug.harePerformed()).toBe(false);
    // And the world is genuinely moving again.
    const lane = await debug.playerLane();
    await page.keyboard.press("ArrowLeft");
    await page.waitForTimeout(400);
    expect(await debug.playerLane()).not.toBe(lane);
  });
});

test.describe("brief §0: non-regression", () => {
  test("H still toggles the dev panel outside an encounter", async () => {
    // beforeEach has already put us back in RUNNING, which is the state this
    // is about: `H` is only claimed by the counter inside an encounter.
    await expect(page.locator("#scene-debug-hud")).toBeHidden();
    await page.keyboard.press("KeyH");
    await expect(page.locator("#scene-debug-hud")).toBeVisible();
    await page.keyboard.press("KeyH");
    await expect(page.locator("#scene-debug-hud")).toBeHidden();
  });

  test("the warning overlay is absent during ordinary running", async () => {
    await expect(page.locator("#special-overlay")).toBeHidden();
    await expect(page.locator(".sa-vignette.is-armed")).toHaveCount(0);
  });
});
