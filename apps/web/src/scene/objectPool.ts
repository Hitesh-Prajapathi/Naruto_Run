/**
 * Generic acquire/release object pool. Framework-agnostic on purpose (no
 * Three.js import) so it's usable for meshes, particles (Phase G), or
 * anything else that must not be constructed per frame -- and so its
 * recycling behavior can be unit-tested without a WebGL context.
 *
 * This is the mechanism behind game_implementation_plan.md's Phase-C gate:
 * "no allocation spikes over a 3-minute run." `factory` is only ever called
 * to grow the pool the first time more concurrent items are needed than
 * currently exist; steady-state traffic (spawn/despawn at a roughly constant
 * rate, which is exactly what a fixed obstacle-spacing rule produces) should
 * settle into calling it zero times after warmup.
 */

export class ObjectPool<T> {
  private readonly free: T[] = [];
  private readonly inUse = new Set<T>();
  private factoryCalls = 0;

  constructor(
    private readonly factory: () => T,
    private readonly onRelease: (item: T) => void,
    initialSize = 0,
  ) {
    for (let i = 0; i < initialSize; i += 1) {
      this.free.push(this.factory());
      this.factoryCalls += 1;
    }
  }

  acquire(): T {
    const item = this.free.pop();
    if (item !== undefined) {
      this.inUse.add(item);
      return item;
    }
    const created = this.factory();
    this.factoryCalls += 1;
    this.inUse.add(created);
    return created;
  }

  release(item: T): void {
    if (!this.inUse.delete(item)) {
      return; // not ours, or already released -- double-release is a no-op
    }
    this.onRelease(item);
    this.free.push(item);
  }

  releaseAll(): void {
    for (const item of [...this.inUse]) {
      this.release(item);
    }
  }

  get activeCount(): number {
    return this.inUse.size;
  }

  get freeCount(): number {
    return this.free.length;
  }

  /** Total objects ever constructed. A flat line after warmup is the signal
   * this pool is not allocating during steady-state play. */
  get totalConstructed(): number {
    return this.factoryCalls;
  }

  get active(): ReadonlySet<T> {
    return this.inUse;
  }
}
