import { describe, expect, it } from "vitest";
import { ObjectPool } from "../../src/scene/objectPool";

function makeCounterPool(initialSize = 0) {
  let created = 0;
  let resetCalls = 0;
  const pool = new ObjectPool<{ id: number }>(
    () => {
      created += 1;
      return { id: created };
    },
    () => {
      resetCalls += 1;
    },
    initialSize,
  );
  return { pool, createdCount: () => created, resetCalls: () => resetCalls };
}

describe("ObjectPool", () => {
  it("pre-allocates initialSize items without any being active", () => {
    const { pool, createdCount } = makeCounterPool(5);
    expect(createdCount()).toBe(5);
    expect(pool.freeCount).toBe(5);
    expect(pool.activeCount).toBe(0);
  });

  it("acquire reuses a free item instead of constructing a new one", () => {
    const { pool, createdCount } = makeCounterPool(1);
    const item = pool.acquire();
    expect(item.id).toBe(1);
    expect(createdCount()).toBe(1);
    expect(pool.activeCount).toBe(1);
    expect(pool.freeCount).toBe(0);
  });

  it("acquire constructs a new item when the pool is empty", () => {
    const { pool, createdCount } = makeCounterPool(0);
    const item = pool.acquire();
    expect(item.id).toBe(1);
    expect(createdCount()).toBe(1);
  });

  it("release returns an item to the free list and calls the reset callback", () => {
    const { pool, resetCalls } = makeCounterPool(0);
    const item = pool.acquire();

    pool.release(item);

    expect(resetCalls()).toBe(1);
    expect(pool.activeCount).toBe(0);
    expect(pool.freeCount).toBe(1);
  });

  it("a released item is reused by the next acquire without constructing a new one", () => {
    const { pool, createdCount } = makeCounterPool(0);
    const first = pool.acquire();
    pool.release(first);

    const second = pool.acquire();

    expect(second).toBe(first);
    expect(createdCount()).toBe(1);
  });

  it("double-releasing the same item is a no-op", () => {
    const { pool, resetCalls } = makeCounterPool(0);
    const item = pool.acquire();
    pool.release(item);
    pool.release(item);

    expect(resetCalls()).toBe(1);
    expect(pool.freeCount).toBe(1);
  });

  it("releasing an item this pool never produced is a no-op", () => {
    const { pool, resetCalls } = makeCounterPool(0);
    pool.release({ id: 999 });

    expect(resetCalls()).toBe(0);
    expect(pool.freeCount).toBe(0);
  });

  it("releaseAll returns every active item to the free list", () => {
    const { pool } = makeCounterPool(0);
    pool.acquire();
    pool.acquire();
    pool.acquire();

    pool.releaseAll();

    expect(pool.activeCount).toBe(0);
    expect(pool.freeCount).toBe(3);
  });

  it("stays at a constant totalConstructed across many acquire/release cycles at a fixed concurrency", () => {
    const { pool, createdCount } = makeCounterPool(0);

    // Simulate steady-state traffic: never more than 4 concurrently active.
    const active: Array<{ id: number }> = [];
    for (let tick = 0; tick < 5000; tick += 1) {
      if (active.length < 4) {
        active.push(pool.acquire());
      } else {
        const item = active.shift();
        if (item) pool.release(item);
      }
    }

    // Growth is bounded by the peak concurrency, not the number of ticks.
    expect(createdCount()).toBeLessThanOrEqual(4);
    expect(pool.totalConstructed).toBe(createdCount());
  });
});
