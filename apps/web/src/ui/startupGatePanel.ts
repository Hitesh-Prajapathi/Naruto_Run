/**
 * The startup gate's on-screen panels -- Feature Brief 05 §4.
 *
 * §4 asks for "the existing jungle/scroll UI language established by the
 * Game Over panel. Reuse those colours, frames and typography — do not
 * introduce a new visual style." So this shares the Game Over panel's
 * framing (`.go-panel`, `.go-backdrop`, `.go-leaves`) and its exact type and
 * button styling.
 *
 * The *text* elements deliberately carry their own class names
 * (`.gate-title`, `.gate-body`, `.gate-action`) even though they are styled
 * by the same CSS rules. Wearing `.go-title` / `.go-cause` / `.go-button` as
 * well made `page.locator(".go-cause")` in the already-approved boss tests
 * match two elements and fail on strict mode -- a new panel must not change
 * what an existing selector resolves to.
 */

export interface GatePanelView {
  title: string;
  body: string;
  /** 0..1, or -1 for no progress bar. */
  progress: number;
  action: string | null;
  showKeyboardEscape: boolean;
  blocking: boolean;
}

export class StartupGatePanel {
  private readonly panel: HTMLElement;
  private readonly titleEl: HTMLElement;
  private readonly bodyEl: HTMLElement;
  private readonly progressEl: HTMLElement;
  private readonly progressFill: HTMLElement;
  private readonly actionButton: HTMLButtonElement;
  private readonly keyboardButton: HTMLButtonElement;

  private onAction: (() => void) | null = null;
  private onKeyboard: (() => void) | null = null;
  private lastSignature = "";

  constructor(private readonly root: HTMLElement) {
    root.innerHTML = `
      <div class="go-backdrop"></div>
      <div class="go-panel gate-panel" role="dialog" aria-modal="true" aria-labelledby="gate-title">
        <div class="go-leaves" aria-hidden="true"></div>
        <h2 class="gate-title" id="gate-title" data-title>Play with your body</h2>
        <p class="gate-body" data-body></p>
        <div class="gate-progress" data-progress hidden><span data-progress-fill></span></div>
        <button class="gate-action" type="button" data-action hidden></button>
        <button class="gate-keyboard" type="button" data-keyboard hidden>Play with keyboard</button>
      </div>
    `;
    this.panel = root.querySelector(".go-panel")!;
    this.titleEl = root.querySelector("[data-title]")!;
    this.bodyEl = root.querySelector("[data-body]")!;
    this.progressEl = root.querySelector("[data-progress]")!;
    this.progressFill = root.querySelector("[data-progress-fill]")!;
    this.actionButton = root.querySelector("[data-action]")!;
    this.keyboardButton = root.querySelector("[data-keyboard]")!;

    this.actionButton.addEventListener("click", () => this.onAction?.());
    this.keyboardButton.addEventListener("click", () => this.onKeyboard?.());
    root.hidden = true;
  }

  setOnAction(callback: () => void): void {
    this.onAction = callback;
  }

  setOnKeyboard(callback: () => void): void {
    this.onKeyboard = callback;
  }

  render(view: GatePanelView): void {
    if (!view.blocking) {
      this.hide();
      return;
    }

    if (this.root.hidden) {
      this.root.hidden = false;
      void this.panel.offsetWidth; // let the transition run from its start
      this.root.classList.add("go-visible");
    }

    // Text changes rarely; the progress bar changes every frame. Splitting
    // them keeps the DOM writes to one style property most frames.
    const signature = `${view.title}|${view.body}|${view.action ?? ""}|${view.showKeyboardEscape}`;
    if (signature !== this.lastSignature) {
      this.lastSignature = signature;
      this.titleEl.textContent = view.title;
      this.bodyEl.textContent = view.body;
      this.actionButton.textContent = view.action ?? "";
      this.actionButton.hidden = view.action === null;
      this.keyboardButton.hidden = !view.showKeyboardEscape;
      if (view.action !== null) {
        this.actionButton.focus({ preventScroll: true });
      }
    }

    if (view.progress < 0) {
      this.progressEl.hidden = true;
    } else {
      this.progressEl.hidden = false;
      this.progressFill.style.transform = `scaleX(${Math.max(0, Math.min(1, view.progress))})`;
    }
  }

  hide(): void {
    if (this.root.hidden) return;
    this.root.classList.remove("go-visible");
    this.root.hidden = true;
    this.lastSignature = "";
  }
}
