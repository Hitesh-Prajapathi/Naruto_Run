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

import {
  ATTACK_CATALOG,
  SEAL_GLYPHS,
  type AttackDefinition,
  type SealName,
} from "../config/bossConfig";

export interface CombatHudState {
  sequence: readonly SealName[];
  matchedCount: number;
  /** 0..1 through the attempt window; drives the timer bar. */
  windowProgress: number;
  phase: "idle" | "active" | "cooldown";
  /** Set briefly after a failure so the player sees *why* it reset. */
  failureReason: "wrong_seal" | "timeout" | null;
  attackProgress?: Record<string, number>;
  lastRecognizedSeal?: SealName | null;
  lastTriggeredAttack?: AttackDefinition | null;
}

export class CombatHud {
  private readonly catalogEl: HTMLElement;
  private readonly sealIndicatorEl: HTMLElement;
  private readonly timerEl: HTMLElement;
  private readonly statusEl: HTMLElement;
  private readonly bannerEl: HTMLElement;
  private lastSignature = "";

  constructor(private readonly root: HTMLElement) {
    root.innerHTML = `
      <div class="combat-banner" data-banner hidden></div>
      
      <!-- Side Jutsu Catalog list -->
      <div class="combat-jutsu-catalog" data-catalog></div>

      <!-- Center live seal indicator and prompt -->
      <div class="combat-seals-wrap">
        <div class="combat-status" data-status></div>
        <div class="combat-recent-seal" data-recent-seal hidden></div>
        <div class="combat-seals" data-seals></div>
        <div class="combat-timer"><span data-timer></span></div>
      </div>
    `;
    this.catalogEl = root.querySelector("[data-catalog]")!;
    this.sealIndicatorEl = root.querySelector("[data-recent-seal]")!;
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
    const progressMap = state.attackProgress ?? {};
    const signature = `${state.sequence.join(",")}|${state.matchedCount}|${state.phase}|${state.failureReason ?? ""}|${state.lastRecognizedSeal ?? ""}|${JSON.stringify(progressMap)}`;

    if (signature !== this.lastSignature) {
      this.lastSignature = signature;

      // Render the 5 Jutsu in the Attack Catalog on the side
      this.catalogEl.innerHTML = `
        <div class="jutsu-catalog-header">⚡ JUTSU ATTACKS</div>
        <div class="jutsu-list">
          ${ATTACK_CATALOG.map((attack) => {
            const currentMatch = progressMap[attack.id] ?? 0;
            const isCompleted = currentMatch >= attack.seals.length;
            const inProgress = currentMatch > 0 && !isCompleted;
            const cardCls = [
              "jutsu-card",
              isCompleted ? "is-ready" : "",
              inProgress ? "in-progress" : "",
            ]
              .filter(Boolean)
              .join(" ");

            return `
              <div class="${cardCls}" style="--jutsu-color: ${attack.color};">
                <div class="jutsu-card-top">
                  <span class="jutsu-icon">${attack.icon}</span>
                  <span class="jutsu-name">${attack.name}</span>
                  <span class="jutsu-key">[${attack.keyLabel}]</span>
                </div>
                <div class="jutsu-seals-row">
                  ${attack.seals
                    .map((seal, sIdx) => {
                      const matched = sIdx < currentMatch;
                      const sealCls = ["jutsu-seal-badge", matched ? "is-matched" : ""]
                        .filter(Boolean)
                        .join(" ");
                      return `
                        <span class="${sealCls}">
                          <span class="glyph">${SEAL_GLYPHS[seal] ?? ""}</span>
                          <span class="name">${seal}</span>
                        </span>
                      `;
                    })
                    .join("")}
                </div>
              </div>
            `;
          }).join("")}
        </div>
      `;

      // Render recent seal indicator if present
      if (state.lastRecognizedSeal) {
        const glyph = SEAL_GLYPHS[state.lastRecognizedSeal] ?? "";
        this.sealIndicatorEl.innerHTML = `
          <span class="recent-seal-label">RECOGNIZED:</span>
          <span class="recent-seal-badge">
            <span class="glyph">${glyph}</span>
            <span class="name">${state.lastRecognizedSeal.toUpperCase()}</span>
          </span>
        `;
        this.sealIndicatorEl.hidden = false;
      } else {
        this.sealIndicatorEl.hidden = true;
      }

      // Status text
      if (state.phase === "cooldown") {
        this.statusEl.textContent =
          state.failureReason === "timeout"
            ? "Too slow — resetting"
            : state.failureReason === "wrong_seal"
              ? "Wrong seal — resetting"
              : "Jutsu released!";
        this.statusEl.dataset["tone"] = state.failureReason ? "bad" : "good";
      } else {
        this.statusEl.textContent = "Perform hand seals to execute Jutsu";
        this.statusEl.dataset["tone"] = "neutral";
      }
    }

    const remaining = Math.max(0, 1 - state.windowProgress);
    this.timerEl.style.transform = `scaleX(${state.phase === "active" ? remaining : 0})`;
  }
}
