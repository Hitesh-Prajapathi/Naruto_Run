/**
 * Player-facing HUD -- naruto_run_agent_brief.md §I/§P2.4. Replaces the
 * debug panel in the primary slot (that panel still exists, behind a toggle,
 * because it stays useful for tuning).
 *
 * Also owns the damage flash: a red screen-edge vignette pulse, one of the
 * three-plus required collision feedback channels (the others -- hit-stop
 * and camera shake -- live in sceneMain and chaseCamera, since they affect
 * simulation and view rather than DOM).
 */

export interface GameHudState {
  distanceMetres: number;
  speed: number;
  lives: number;
  maxLives: number;
  /** Camera/CV status text, e.g. "keyboard" until Phase E wires real input. */
  inputStatus: string;
}

const FLASH_DURATION_S = 0.45;

export class GameHud {
  private readonly distanceEl: HTMLElement;
  private readonly speedEl: HTMLElement;
  private readonly livesEl: HTMLElement;
  private readonly statusEl: HTMLElement;
  private readonly flashEl: HTMLElement;
  private flashRemaining = 0;

  constructor(private readonly root: HTMLElement) {
    root.innerHTML = `
      <div class="hud-row hud-primary">
        <div class="hud-stat"><span class="hud-label">Distance</span><span class="hud-value" data-distance>0 m</span></div>
        <div class="hud-stat"><span class="hud-label">Speed</span><span class="hud-value" data-speed>0.0x</span></div>
      </div>
      <div class="hud-row hud-secondary">
        <div class="hud-lives" data-lives aria-label="Lives remaining"></div>
        <div class="hud-input-status" data-status>--</div>
      </div>
      <div class="hud-flash" data-flash aria-hidden="true"></div>
    `;
    this.distanceEl = root.querySelector("[data-distance]")!;
    this.speedEl = root.querySelector("[data-speed]")!;
    this.livesEl = root.querySelector("[data-lives]")!;
    this.statusEl = root.querySelector("[data-status]")!;
    this.flashEl = root.querySelector("[data-flash]")!;
  }

  /** Trigger the red damage flash. */
  flash(): void {
    this.flashRemaining = FLASH_DURATION_S;
  }

  render(state: GameHudState, dt: number): void {
    this.distanceEl.textContent = `${Math.floor(state.distanceMetres)} m`;
    this.speedEl.textContent = `${state.speed.toFixed(1)}x`;

    // Rebuild the pip row only when the count actually changes -- this runs
    // every frame and innerHTML churn would be needless GC pressure.
    const pips = "●".repeat(state.lives) + "○".repeat(Math.max(0, state.maxLives - state.lives));
    if (this.livesEl.textContent !== pips) {
      this.livesEl.textContent = pips;
      this.livesEl.classList.toggle("hud-lives-critical", state.lives <= 1);
    }
    if (this.statusEl.textContent !== state.inputStatus) {
      this.statusEl.textContent = state.inputStatus;
    }

    if (this.flashRemaining > 0) {
      this.flashRemaining = Math.max(0, this.flashRemaining - dt);
      const t = this.flashRemaining / FLASH_DURATION_S;
      this.flashEl.style.opacity = String(t * t);
    } else if (this.flashEl.style.opacity !== "0") {
      this.flashEl.style.opacity = "0";
    }
  }

  setVisible(visible: boolean): void {
    this.root.hidden = !visible;
  }
}
