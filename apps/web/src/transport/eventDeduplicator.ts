/**
 * At-most-once event consumption, keyed by `event_id`.
 *
 * game_implementation_plan.md invariant: "Repeated network messages with the
 * same event_id have no second effect." A reconnect, a server retry, or the
 * mock/real source both delivering the same tape entry must never double-fire
 * a game reaction (e.g. two fire bursts for one ATTACK_TRIGGERED).
 *
 * Bounded by a max-size ring so long sessions don't leak memory; the window
 * only needs to be larger than plausible reconnect/retry overlap, not the
 * whole session's event count.
 */

import type { PipelineEventV1 } from "./protocol";

export class EventDeduplicator {
  private readonly seen = new Set<string>();
  private readonly order: string[] = [];

  constructor(private readonly maxTracked: number = 512) {
    if (maxTracked < 1) {
      throw new RangeError("maxTracked must be positive");
    }
  }

  /** Returns true the first time this event_id is seen, false on every repeat. */
  admit(event: PipelineEventV1): boolean {
    if (this.seen.has(event.event_id)) {
      return false;
    }
    this.seen.add(event.event_id);
    this.order.push(event.event_id);
    if (this.order.length > this.maxTracked) {
      const oldest = this.order.shift();
      if (oldest !== undefined) {
        this.seen.delete(oldest);
      }
    }
    return true;
  }

  get trackedCount(): number {
    return this.order.length;
  }

  reset(): void {
    this.seen.clear();
    this.order.length = 0;
  }
}
