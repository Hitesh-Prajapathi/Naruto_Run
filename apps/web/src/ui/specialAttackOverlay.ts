/**
 * The screen-space half of the special-attack warning -- Feature Brief 04
 * §4.1 and §4.6.
 *
 * The world-space marker (specialWarningMarker.ts) sits above Naruto's head,
 * but the brief is emphatic that the warning must land "even if the player is
 * looking at their own hands rather than at Naruto". So this adds the
 * full-screen reinforcement: a pulsing danger vignette around the edges of
 * the viewport, plus a large, high-contrast card showing the sign to make and
 * the key that stands in for it.
 *
 * On the reference image (§4.1): there is no `hare` photo in the repo -- the
 * CV training set is not checked in -- so the card draws a schematic of the
 * sign and swaps in a real photo automatically if one is ever dropped at
 * HARE_REFERENCE_IMAGE_URL. That keeps the requirement satisfied now without
 * inventing an asset, and makes upgrading it a file copy.
 *
 * The card also carries the §2.2 practice prompt, which is the *only* thing
 * that teaches the player this sign exists before it can kill them.
 */

import { COUNTER_KEY_LABEL, HARE_REFERENCE_IMAGE_URL } from "../config/specialAttackConfig";

/**
 * A schematic of the hare seal: hands clasped, index and middle fingers of
 * both hands extended straight up.
 *
 * It is paired with a written description below it, deliberately. A drawing
 * this small can be misread -- the first version looked like a candle -- and
 * §4.1's requirement is that the player "must not have to remember which
 * sign `hare` is". A sentence cannot be misread, so the words carry the
 * meaning and the picture only speeds up recognition.
 */
const HARE_SCHEMATIC = `
<svg class="sa-hands" viewBox="0 0 120 120" aria-hidden="true">
  <g fill="none" stroke="currentColor" stroke-width="7"
     stroke-linecap="round" stroke-linejoin="round">
    <!-- the two pairs of raised fingers -->
    <path d="M48 60 V20" /><path d="M62 60 V20" />
    <path d="M76 60 V26" /><path d="M90 60 V26" />
    <!-- clasped hands: two interlocking blocks -->
    <path d="M34 62 h56 a8 8 0 0 1 8 8 v10 a8 8 0 0 1 -8 8 h-56 a8 8 0 0 1 -8 -8 v-10 a8 8 0 0 1 8 -8 z" />
    <!-- thumbs folded across the front -->
    <path d="M38 88 v10 a6 6 0 0 0 6 6 h32 a6 6 0 0 0 6 -6 v-10" />
  </g>
</svg>`;

export class SpecialAttackOverlay {
  private readonly vignette: HTMLElement;
  private readonly card: HTMLElement;
  private readonly ring: HTMLElement;
  private readonly practice: HTMLElement;
  private readonly result: HTMLElement;
  private lastRingStep = -1;

  constructor(private readonly root: HTMLElement) {
    root.innerHTML = `
      <div class="sa-vignette" data-vignette></div>
      <div class="sa-card" data-card hidden>
        <div class="sa-ring" data-ring>
          <div class="sa-ring-inner">
            <img class="sa-photo" data-photo alt="" hidden />
            <div class="sa-schematic" data-schematic>${HARE_SCHEMATIC}</div>
          </div>
        </div>
        <div class="sa-copy">
          <div class="sa-title">SPECIAL ATTACK INCOMING</div>
          <div class="sa-sign">HARE</div>
          <div class="sa-how">Clasp both hands — index and middle fingers straight up</div>
          <div class="sa-key">Press <kbd>${COUNTER_KEY_LABEL}</kbd> to counter</div>
        </div>
      </div>
      <div class="sa-practice" data-practice hidden>
        <span class="sa-practice-mark">!</span>
        <span>
          <b>Obito is preparing something.</b>
          Press <kbd>${COUNTER_KEY_LABEL}</kbd> once to ready the <b>HARE</b> counter.
        </span>
      </div>
      <div class="sa-result" data-result hidden></div>
    `;
    this.vignette = root.querySelector("[data-vignette]")!;
    this.card = root.querySelector("[data-card]")!;
    this.ring = root.querySelector("[data-ring]")!;
    this.practice = root.querySelector("[data-practice]")!;
    this.result = root.querySelector("[data-result]")!;
    root.hidden = true;

    // Use the real reference photo if someone has added one; otherwise the
    // schematic stays. Failing to load is the expected path today, so it
    // must be silent rather than a console error.
    const photo = root.querySelector<HTMLImageElement>("[data-photo]")!;
    const schematic = root.querySelector<HTMLElement>("[data-schematic]")!;
    photo.addEventListener("load", () => {
      photo.hidden = false;
      schematic.hidden = true;
    });
    photo.addEventListener("error", () => {
      photo.removeAttribute("src");
    });
    photo.src = HARE_REFERENCE_IMAGE_URL;
  }

  /** Open the warning: vignette, card and countdown all at once. */
  showWarning(): void {
    this.root.hidden = false;
    this.card.hidden = false;
    this.practice.hidden = true;
    this.result.hidden = true;
    this.vignette.classList.add("is-armed");
    this.lastRingStep = -1;
    this.setProgress(0);
  }

  /** @param progress 0..1 through the window, in real seconds. */
  setProgress(progress: number): void {
    // The ring is a conic-gradient driven by a custom property; stepping it
    // avoids a style write on every frame for a change nobody can see.
    const step = Math.round(Math.max(0, Math.min(1, progress)) * 120);
    if (step === this.lastRingStep) return;
    this.lastRingStep = step;
    this.ring.style.setProperty("--sa-progress", String(step / 120));
  }

  /** §2.2: shown until the player has tried `hare` at least once. */
  showPracticePrompt(): void {
    this.root.hidden = false;
    this.practice.hidden = false;
  }

  hidePracticePrompt(): void {
    this.practice.hidden = true;
    if (this.card.hidden && this.result.hidden) this.root.hidden = true;
  }

  /** Brief flourish on resolution, then everything comes down. */
  showResult(outcome: "countered" | "struck"): void {
    this.root.hidden = false;
    this.card.hidden = true;
    this.vignette.classList.remove("is-armed");
    this.result.hidden = false;
    this.result.textContent = outcome === "countered" ? "COUNTERED!" : "OVERWHELMED";
    this.result.dataset["outcome"] = outcome;
    // Restart the animation on a repeat.
    this.result.classList.remove("sa-result-in");
    void this.result.offsetWidth;
    this.result.classList.add("sa-result-in");
  }

  /**
   * Full teardown. Part of the single exit path required by brief §5 -- it
   * is safe to call from any state, including states where nothing is shown.
   */
  hide(): void {
    this.card.hidden = true;
    this.practice.hidden = true;
    this.result.hidden = true;
    this.result.classList.remove("sa-result-in");
    this.vignette.classList.remove("is-armed");
    this.root.hidden = true;
  }
}
