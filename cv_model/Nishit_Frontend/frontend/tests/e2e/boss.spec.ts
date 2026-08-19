import { expect, test, type Page } from "@playwright/test";

/**
 * Feature Brief 02 verification: the Obito encounter and the Game Over
 * screen, in a real browser.
 *
 * Encounters trigger on distance, which takes a while at normal speed, so
 * these tests drive the exposed debug hook to jump the run forward rather
 * than waiting ~20s of wall-clock per encounter.
 */

interface BossDebug {
  state: () => string;
  assetsReady: () => boolean;
  encountersWon: () => number;
  isGameOverVisible: () => boolean;
  forceEncounter: () => void;
  forceGameOver: (cause: "obstacles" | "obito") => void;
  narutoHp: () => number;
  obitoHp: () => number;
  distance: () => number;
}

async function waitForGame(page: Page) {
  await page.goto("/scene.html");
  await page.waitForFunction(
    () => Boolean((window as unknown as { __bossDebug?: BossDebug }).__bossDebug),
    undefined,
    { timeout: 30000 },
  );
}

/**
 * Obito's model is ~24 MB, so encounter assets land well after the running
 * game is interactive. Any test that starts an encounter must wait for them
 * first -- otherwise the encounter is (correctly) postponed and the test is
 * asserting against a state that hasn't been reached yet.
 */
async function waitForBossAssets(page: Page) {
  await page.waitForFunction(
    () => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.assetsReady(),
    undefined,
    { timeout: 60000 },
  );
  // Waiting that long for a 24 MB model means an idle player can take three
  // obstacle hits and hit Game Over in the meantime -- and forceEncounter
  // (correctly) refuses to start from GAME_OVER. Clear it so the encounter
  // tests start from a live run.
  await dismissGameOverIfShown(page);
}

async function dismissGameOverIfShown(page: Page) {
  if (await page.locator("#game-over").isVisible()) {
    await page.keyboard.press("Enter");
    await expect(page.locator("#game-over")).toBeHidden();
  }
}

function boss(page: Page) {
  return {
    state: () => page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.state()),
    won: () => page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.encountersWon()),
    gameOverVisible: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.isGameOverVisible()),
    forceEncounter: () =>
      page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.forceEncounter()),
    forceGameOver: (cause: "obstacles" | "obito") =>
      page.evaluate(
        (c) => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.forceGameOver(c),
        cause,
      ),
    narutoHp: () => page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.narutoHp()),
    obitoHp: () => page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.obitoHp()),
    distance: () => page.evaluate(() => (window as unknown as { __bossDebug: BossDebug }).__bossDebug.distance()),
  };
}

test.describe("Game Over screen (brief §4)", () => {
  test("appears after the third obstacle hit, with the right cause", async ({ page }) => {
    await waitForGame(page);
    const api = boss(page);

    await api.forceGameOver("obstacles");
    // Brief §4.2: a ~700ms beat between the killing blow and the panel.
    await expect(page.locator("#game-over")).toBeVisible({ timeout: 5000 });
    await expect(page.locator(".go-cause")).toContainText("couldn't dodge the obstacles");
    await expect(page.locator(".go-title")).toHaveText("Try Again");
    expect(await api.state()).toBe("GAME_OVER");
  });

  test("shows the Obito cause line when a duel is lost", async ({ page }) => {
    await waitForGame(page);
    await boss(page).forceGameOver("obito");
    await expect(page.locator(".go-cause")).toContainText("Obito defeated you", { timeout: 5000 });
  });

  test("freezes the world: distance stops advancing", async ({ page }) => {
    await waitForGame(page);
    const api = boss(page);
    await api.forceGameOver("obstacles");
    await expect(page.locator("#game-over")).toBeVisible({ timeout: 5000 });

    const first = await api.distance();
    await page.waitForTimeout(700);
    expect(await api.distance()).toBe(first);
  });

  test("Try Again restarts cleanly via mouse", async ({ page }) => {
    await waitForGame(page);
    const api = boss(page);
    await api.forceGameOver("obstacles");
    await expect(page.locator("#game-over")).toBeVisible({ timeout: 5000 });

    await page.locator(".go-button").click();

    await expect(page.locator("#game-over")).toBeHidden();
    expect(await api.state()).toBe("RUNNING");
    // Lives restored and the world moving again.
    await expect(page.locator("#game-hud [data-lives]")).toContainText("●●●");
    const d1 = await api.distance();
    await page.waitForTimeout(600);
    expect(await api.distance()).toBeGreaterThan(d1);
  });

  test("Try Again also works from the keyboard (brief §4.5)", async ({ page }) => {
    await waitForGame(page);
    const api = boss(page);
    await api.forceGameOver("obstacles");
    await expect(page.locator("#game-over")).toBeVisible({ timeout: 5000 });

    await page.keyboard.press("Enter");

    await expect(page.locator("#game-over")).toBeHidden();
    expect(await api.state()).toBe("RUNNING");
  });

  test("ten consecutive restarts leave the game running with no drift", async ({ page }) => {
    // Brief §4.5 warns the classic restart bug is an orphaned timer from the
    // previous run firing during the new one.
    test.setTimeout(90_000);
    await waitForGame(page);
    const api = boss(page);

    for (let i = 0; i < 10; i += 1) {
      await api.forceGameOver("obstacles");
      await expect(page.locator("#game-over")).toBeVisible({ timeout: 5000 });
      await page.keyboard.press("Enter");
      await expect(page.locator("#game-over")).toBeHidden();
      expect(await api.state()).toBe("RUNNING");
    }

    expect(await api.won()).toBe(0);
    const d1 = await api.distance();
    await page.waitForTimeout(600);
    expect(await api.distance()).toBeGreaterThan(d1);
  });
});

test.describe("Obito encounter (brief §5)", () => {
  test("runs intro -> combat with both health bars at full HP", async ({ page }) => {
    await waitForGame(page);
    const api = boss(page);
    await waitForBossAssets(page);

    await api.forceEncounter();
    await expect.poll(() => api.state(), { timeout: 15000 }).toBe("BOSS_INTRO");

    // HP resets to 100/120 at the start of each encounter (§2.3 + §2.1).
    expect(await api.narutoHp()).toBe(100);
    expect(await api.obitoHp()).toBe(120);

    await expect.poll(() => api.state(), { timeout: 20000 }).toBe("BOSS_COMBAT");
    await expect(page.locator("#combat-hud")).toBeVisible();
    await expect(page.locator(".combat-seals .combat-seal")).toHaveCount(3);
  });

  test("suspends the world scroll during combat", async ({ page }) => {
    await waitForGame(page);
    const api = boss(page);
    await waitForBossAssets(page);
    await api.forceEncounter();
    await expect.poll(() => api.state(), { timeout: 25000 }).toBe("BOSS_COMBAT");

    const first = await api.distance();
    await page.waitForTimeout(800);
    // Scroll is stopped in BOSS_COMBAT, so distance must not advance.
    expect(await api.distance()).toBeCloseTo(first, 1);
  });

  test("performing the seal sequence damages Obito", async ({ page }) => {
    test.setTimeout(60_000);
    await waitForGame(page);
    const api = boss(page);
    await waitForBossAssets(page);
    await api.forceEncounter();
    await expect.poll(() => api.state(), { timeout: 25000 }).toBe("BOSS_COMBAT");

    const startHp = await api.obitoHp();

    // Keyboard fallback (brief §6): read the prompt and press the matching
    // digit for each seal, so this exercises the real sequence logic.
    for (let attempt = 0; attempt < 12; attempt += 1) {
      if ((await api.obitoHp()) < startHp) break;
      const names = await page.locator(".combat-seal .combat-seal-name").allInnerTexts();
      const order = ["tiger", "ram", "snake"];
      for (const name of names) {
        const index = order.indexOf(name.trim().toLowerCase());
        if (index >= 0) await page.keyboard.press(`Digit${index + 1}`);
        await page.waitForTimeout(60);
      }
      await page.waitForTimeout(400);
    }

    expect(await api.obitoHp(), "a completed seal sequence should land a hit").toBeLessThan(startHp);
  });
});
