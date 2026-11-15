/**
 * The input abstraction layer -- Feature Brief 05 §5.4.
 *
 * The game asks for *intent* ("lane-left", "jump"), never for "was A
 * pressed" or "is the torso leaning". Both providers push intents in here and
 * the host drains them once a frame, so a camera-driven lane change and a
 * keyboard one are literally the same call downstream — which is the brief's
 * test for whether the abstraction is in the right place (§5.4, T-16).
 *
 * This is the one sanctioned refactor of approved code. It is kept as small
 * as possible: `sceneMain` still owns the key listener and still calls
 * `controller.moveLane` / `controller.requestJump` — those calls now happen
 * once, in one place, fed by whichever provider produced the intent.
 *
 * **Keyboard wins on conflict** (§0.2). If both providers ask for a lane
 * change on the same frame the keyboard's is taken and the camera's is
 * dropped, rather than applying two steps.
 */

export type InputIntent =
  | { kind: "lane"; direction: -1 | 1 }
  | { kind: "jump" };

export type IntentSource = "keyboard" | "camera";

interface QueuedIntent {
  intent: InputIntent;
  source: IntentSource;
}

export class InputRouter {
  private queue: QueuedIntent[] = [];

  /** Push an intent. Called by the key handler and by the body input source. */
  push(intent: InputIntent, source: IntentSource): void {
    this.queue.push({ intent, source });
  }

  laneLeft(source: IntentSource): void {
    this.push({ kind: "lane", direction: -1 }, source);
  }

  laneRight(source: IntentSource): void {
    this.push({ kind: "lane", direction: 1 }, source);
  }

  jump(source: IntentSource): void {
    this.push({ kind: "jump" }, source);
  }

  /**
   * Take everything queued since the last call, resolved for conflicts.
   *
   * At most one lane intent and one jump intent survive a frame: two lane
   * steps in a single frame is never what the player meant, and it is the
   * shape a competing-input bug takes.
   */
  drain(): InputIntent[] {
    if (this.queue.length === 0) {
      return [];
    }
    const queued = this.queue;
    this.queue = [];

    const pick = (kind: InputIntent["kind"]): InputIntent | null => {
      const matching = queued.filter((entry) => entry.intent.kind === kind);
      if (matching.length === 0) return null;
      // §0.2: keyboard wins on conflict.
      const keyboard = matching.find((entry) => entry.source === "keyboard");
      return (keyboard ?? matching[0])!.intent;
    };

    const resolved: InputIntent[] = [];
    const lane = pick("lane");
    if (lane) resolved.push(lane);
    const jump = pick("jump");
    if (jump) resolved.push(jump);
    return resolved;
  }

  /** Drop anything queued -- used on restart so a stale intent from the
   * previous run cannot apply to the new one. */
  clear(): void {
    this.queue = [];
  }

  get pendingCount(): number {
    return this.queue.length;
  }
}
