/**
 * Game Over / "Try Again" panel -- Feature Brief 02 §4.
 *
 * Styling notes (brief §4.3 requires the palette be *sampled from the
 * existing environment*, not invented). The values below are taken from the
 * live scene modules:
 *   #6b4423  fallen-log brown        (gameConfig OBSTACLE_TYPES.log.color)
 *   #4a3122 / #5b3a21  trunk browns  (environmentProps trunkMaterials)
 *   #3f5c2a  grass                   (jungleTrack GRASS_COLOR)
 *   #2f7d3a  canopy green            (environmentProps canopyMaterials)
 *   #c9a227  log accent / leaf yellow(gameConfig OBSTACLE_TYPES.log.accent)
 *   #ff8a3d  the existing NarutoCV accent orange (reused for the button)
 *
 * The panel deliberately does not unload the scene: the frozen, blurred 3D
 * world stays visible behind it so the screen reads as part of the game.
 */

export type DeathCause = "obstacles" | "obito" | "obito_special";

export interface GameOverStats {
  distanceM: number;
  obstaclesCleared: number;
  encountersWon: number;
  totalEncounters: number;
}

const CAUSE_TEXT: Record<DeathCause, string> = {
  obstacles: "Naruto couldn't dodge the obstacles.",
  obito: "Obito defeated you.",
  // Feature Brief 04 §2.5: the special attack kills regardless of HP, so the
  // health bar can read 100/100 at the moment of death. Without a specific
  // line saying why, that looks like a bug rather than the intended surprise.
  obito_special: "Obito's special attack overwhelmed Naruto.",
};

export class GameOverPanel {
  private readonly panel: HTMLElement;
  private readonly causeEl: HTMLElement;
  private readonly distanceEl: HTMLElement;
  private readonly clearedEl: HTMLElement;
  private readonly encountersEl: HTMLElement;
  private readonly tryAgainButton: HTMLButtonElement;
  private onTryAgain: (() => void) | null = null;
  private visible = false;

  constructor(private readonly root: HTMLElement) {
    root.innerHTML = `
      <div class="go-backdrop"></div>
      <div class="go-panel" role="dialog" aria-modal="true" aria-labelledby="go-title">
        <div class="go-leaves" aria-hidden="true"></div>
        <h2 class="go-title" id="go-title">Try Again</h2>
        <p class="go-cause" data-cause></p>
        <dl class="go-stats">
          <div><dt>Distance</dt><dd data-distance>0 m</dd></div>
          <div><dt>Obstacles cleared</dt><dd data-cleared>0</dd></div>
          <div><dt>Encounters won</dt><dd data-encounters>0/2</dd></div>
        </dl>
        <button class="go-button" type="button" data-try-again>Try Again</button>
        <p class="go-hint">Press <kbd>Enter</kbd> or <kbd>Space</kbd></p>
      </div>
    `;
    this.panel = root.querySelector(".go-panel")!;
    this.causeEl = root.querySelector("[data-cause]")!;
    this.distanceEl = root.querySelector("[data-distance]")!;
    this.clearedEl = root.querySelector("[data-cleared]")!;
    this.encountersEl = root.querySelector("[data-encounters]")!;
    this.tryAgainButton = root.querySelector("[data-try-again]")!;

    this.tryAgainButton.addEventListener("click", () => this.fireTryAgain());
    root.hidden = true;

    // Keyboard activation (brief §4.5): a player standing in front of a
    // webcam shouldn't have to walk back to the mouse. Bound on window so it
    // works without the button being focused.
    window.addEventListener("keydown", this.handleKey);
  }

  private readonly handleKey = (event: KeyboardEvent): void => {
    if (!this.visible) return;
    if (event.code === "Enter" || event.code === "NumpadEnter" || event.code === "Space") {
      event.preventDefault();
      this.fireTryAgain();
    }
  };

  private fireTryAgain(): void {
    if (!this.visible) return;
    const callback = this.onTryAgain;
    // Hide first so a double-press can't fire two restarts.
    this.hide();
    callback?.();
  }

  setOnTryAgain(callback: () => void): void {
    this.onTryAgain = callback;
  }

  get isVisible(): boolean {
    return this.visible;
  }

  show(cause: DeathCause, stats: GameOverStats): void {
    this.causeEl.textContent = CAUSE_TEXT[cause];
    this.distanceEl.textContent = `${Math.floor(stats.distanceM)} m`;
    this.clearedEl.textContent = String(stats.obstaclesCleared);
    this.encountersEl.textContent = `${stats.encountersWon}/${stats.totalEncounters}`;

    this.root.hidden = false;
    this.visible = true;
    // Force a reflow so the transition runs from the initial state rather
    // than being collapsed into the same frame as the unhide.
    void this.panel.offsetWidth;
    this.root.classList.add("go-visible");
    this.tryAgainButton.focus({ preventScroll: true });
  }

  hide(): void {
    this.visible = false;
    this.root.classList.remove("go-visible");
    this.root.hidden = true;
  }

  dispose(): void {
    window.removeEventListener("keydown", this.handleKey);
  }
}
