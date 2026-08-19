/**
 * Combat overlay -- the seal-sequence prompt and encounter banner
 * (Feature Brief 02 §5.3).
 *
 * Shows the required sequence, which seal the player is currently on, and a
 * countdown for the attempt window, so a failed attack is always
 * self-explanatory rather than mysterious.
 *
 * Only mounted during BOSS_COMBAT; hidden (and inert) otherwise.
 */

import { SEAL_GLYPHS, type SealName } from "../config/bossConfig";

export interface CombatHudState {
  sequence: readonly SealName[];
  matchedCount: number;
  /** 0..1 through the attempt window; drives the timer bar. */
  windowProgress: number;
  phase: "idle" | "active" | "cooldown";
  /** Set briefly after a failure so the player sees *why* it reset. */
  failureReason: "wrong_seal" | "timeout" | null;
}

export class CombatHud {
  private readonly sealsEl: HTMLElement;
  private readonly timerEl: HTMLElement;
  private readonly statusEl: HTMLElement;
  private readonly bannerEl: HTMLElement;
  private lastSignature = "";

  constructor(private readonly root: HTMLElement) {
    root.innerHTML = `
      <div class="combat-banner" data-banner hidden></div>
      <div class="combat-seals-wrap">
        <div class="combat-status" data-status></div>
        <div class="combat-seals" data-seals></div>
        <div class="combat-timer"><span data-timer></span></div>
      </div>
    `;
    this.sealsEl = root.querySelector("[data-seals]")!;
    this.timerEl = root.querySelector("[data-timer]")!;
    this.statusEl = root.querySelector("[data-status]")!;
    this.bannerEl = root.querySelector("[data-banner]")!;
    root.hidden = true;
  }

  setVisible(visible: boolean): void {
    this.root.hidden = !visible;
  }

  /** Short centred flourish, e.g. "OBITO APPEARS" / "ENCOUNTER WON". */
  showBanner(text: string): void {
    this.bannerEl.textContent = text;
    this.bannerEl.hidden = false;
    this.bannerEl.classList.remove("combat-banner-in");
    void this.bannerEl.offsetWidth; // restart the animation
    this.bannerEl.classList.add("combat-banner-in");
  }

  hideBanner(): void {
    this.bannerEl.hidden = true;
  }

  render(state: CombatHudState): void {
    // The seal row only changes on a new sequence or a matched seal, so
    // rebuilding its DOM every frame would be needless churn.
    const signature = `${state.sequence.join(",")}|${state.matchedCount}|${state.phase}|${state.failureReason ?? ""}`;
    if (signature !== this.lastSignature) {
      this.lastSignature = signature;
      this.sealsEl.innerHTML = state.sequence
        .map((seal, index) => {
          const done = index < state.matchedCount;
          const current = index === state.matchedCount && state.phase === "active";
          const cls = ["combat-seal", done ? "is-done" : "", current ? "is-current" : ""]
            .filter(Boolean)
            .join(" ");
          return `<span class="${cls}"><span class="combat-seal-glyph">${SEAL_GLYPHS[seal]}</span><span class="combat-seal-name">${seal}</span></span>`;
        })
        .join("");

      if (state.phase === "cooldown") {
        this.statusEl.textContent =
          state.failureReason === "timeout" ? "Too slow — resetting" : state.failureReason === "wrong_seal" ? "Wrong seal — resetting" : "Jutsu released!";
        this.statusEl.dataset["tone"] = state.failureReason ? "bad" : "good";
      } else {
        this.statusEl.textContent = "Perform the seals";
        this.statusEl.dataset["tone"] = "neutral";
      }
    }

    const remaining = Math.max(0, 1 - state.windowProgress);
    this.timerEl.style.transform = `scaleX(${state.phase === "active" ? remaining : 0})`;
  }
}
