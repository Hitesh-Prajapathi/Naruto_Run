import { describe, expect, it, vi } from "vitest";
import { FramePublisher, type BlobLike, type CaptureSurface } from "../../src/camera/framePublisher";
import type { PipelineEventSource } from "../../src/transport/eventSource";

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function fakeSource(): PipelineEventSource & { sentFrames: Uint8Array[] } {
  const sentFrames: Uint8Array[] = [];
  return {
    connectionState: "open",
    connect: () => {},
    disconnect: () => {},
    sendFrame: (bytes: Uint8Array) => {
      sentFrames.push(bytes);
    },
    sendControl: () => {},
    onConnectionStateChange: () => () => {},
    onStateSnapshot: () => () => {},
    onEvent: () => () => {},
    onAck: () => () => {},
    onError: () => () => {},
    sentFrames,
  };
}

// A plain BlobLike rather than a real Blob: jsdom's Blob does not implement
// arrayBuffer(), and framePublisher only ever calls that one method.
function fakeBlob(byte: number): BlobLike {
  return { arrayBuffer: () => Promise.resolve(new Uint8Array([byte]).buffer) };
}

// HAVE_CURRENT_DATA per the HTMLMediaElement spec; framePublisher.ts inlines
// the same constant so it can accept a plain object instead of a real
// <video> element.
const HAVE_CURRENT_DATA = 2;
const HAVE_NOTHING = 0;

describe("FramePublisher", () => {
  it("sends an encoded frame on each tick while idle", async () => {
    vi.useFakeTimers();
    try {
      const surface: CaptureSurface = {
        width: 32,
        height: 16,
        captureFrame: vi.fn(),
        encodeJpeg: vi.fn().mockResolvedValue(fakeBlob(1)),
      };
      const source = fakeSource();
      const publisher = new FramePublisher({
        video: { readyState: HAVE_CURRENT_DATA },
        source,
        surface,
        targetFps: 10,
      });

      publisher.start();
      await vi.advanceTimersByTimeAsync(100);

      expect(source.sentFrames.length).toBeGreaterThanOrEqual(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("drops a tick instead of queuing when a capture is already in flight", async () => {
    vi.useFakeTimers();
    try {
      const gate = deferred<BlobLike | null>();
      const surface: CaptureSurface = {
        width: 32,
        height: 16,
        captureFrame: vi.fn(),
        encodeJpeg: vi.fn().mockReturnValue(gate.promise),
      };
      const source = fakeSource();
      const publisher = new FramePublisher({
        video: { readyState: HAVE_CURRENT_DATA },
        source,
        surface,
        targetFps: 10, // one tick every 100ms
      });

      publisher.start();
      await vi.advanceTimersByTimeAsync(350); // ~3 more ticks fire while the first is still pending

      expect(surface.captureFrame).toHaveBeenCalledTimes(1);
      expect(publisher.stats.droppedTicks).toBeGreaterThan(0);
      expect(source.sentFrames).toHaveLength(0);

      gate.resolve(fakeBlob(9));
      await vi.advanceTimersByTimeAsync(0);

      expect(source.sentFrames).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not capture while the video has no current data", async () => {
    vi.useFakeTimers();
    try {
      const surface: CaptureSurface = {
        width: 32,
        height: 16,
        captureFrame: vi.fn(),
        encodeJpeg: vi.fn(),
      };
      const source = fakeSource();
      const publisher = new FramePublisher({
        video: { readyState: HAVE_NOTHING },
        source,
        surface,
        targetFps: 10,
      });

      publisher.start();
      await vi.advanceTimersByTimeAsync(100);

      expect(surface.captureFrame).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  describe("round-trip backpressure", () => {
    // The regression: the publisher used to send at a fixed rate regardless
    // of whether the service was keeping up, so a service slower than the
    // capture rate accumulated an unbounded backlog. Observed in play as
    // sample ages past 2.5s and a "not detected" badge over a visible player.
    const build = (source: ReturnType<typeof fakeSource>, now: () => number) =>
      new FramePublisher({
        video: { readyState: HAVE_CURRENT_DATA },
        source,
        surface: {
          width: 32,
          height: 16,
          captureFrame: vi.fn(),
          encodeJpeg: vi.fn().mockResolvedValue(fakeBlob(1)),
        },
        targetFps: 20,
        now,
        responseTimeoutMs: 600,
        congestionMs: 250,
      });

    it("keeps sending freely while the service is keeping up", async () => {
      // The healthy case must NOT be throttled: the service keeps only the
      // newest frame and caps its own output rate, so oversending buys
      // fresher results. Measured 47ms against 79ms when self-clocked.
      vi.useFakeTimers();
      try {
        let clock = 0;
        const source = fakeSource();
        const publisher = build(source, () => clock);

        publisher.start();
        for (let i = 0; i < 10; i += 1) {
          clock += 50;
          await vi.advanceTimersByTimeAsync(50);
          publisher.noteResponse(40); // quick round trips
        }

        expect(publisher.stats.throttled).toBe(false);
        expect(source.sentFrames.length).toBeGreaterThanOrEqual(9);
      } finally {
        vi.useRealTimers();
      }
    });

    it("throttles once round trips get slow", async () => {
      vi.useFakeTimers();
      try {
        let clock = 0;
        const source = fakeSource();
        const publisher = build(source, () => clock);

        publisher.start();
        // Report congestion, then stop answering entirely.
        clock += 50;
        await vi.advanceTimersByTimeAsync(50);
        publisher.noteResponse(900);
        expect(publisher.stats.throttled).toBe(true);

        const sentBefore = source.sentFrames.length;
        for (let i = 0; i < 8; i += 1) {
          clock += 50;
          await vi.advanceTimersByTimeAsync(50);
        }

        // One more send at most, then it waits for the response.
        expect(source.sentFrames.length - sentBefore).toBeLessThanOrEqual(1);
        expect(publisher.stats.waitedTicks).toBeGreaterThan(4);
      } finally {
        vi.useRealTimers();
      }
    });

    it("sends only one frame until the service answers, once congested", async () => {
      vi.useFakeTimers();
      try {
        let clock = 0;
        const source = fakeSource();
        const publisher = build(source, () => clock);

        publisher.start();
        clock += 50;
        await vi.advanceTimersByTimeAsync(50);
        publisher.noteResponse(900); // congested
        source.sentFrames.length = 0;
        // Stay inside the 600ms response timeout, so the only thing that
        // could release a second send is a response -- and none is given.
        for (let i = 0; i < 10; i += 1) {
          clock += 50;
          await vi.advanceTimersByTimeAsync(50);
        }

        expect(source.sentFrames).toHaveLength(1);
        expect(publisher.stats.waitedTicks).toBeGreaterThan(5);
      } finally {
        vi.useRealTimers();
      }
    });

    it("sends the next frame as soon as a response arrives", async () => {
      vi.useFakeTimers();
      try {
        let clock = 0;
        const source = fakeSource();
        const publisher = build(source, () => clock);

        publisher.start();
        clock += 50;
        await vi.advanceTimersByTimeAsync(50);
        expect(source.sentFrames).toHaveLength(1);

        publisher.noteResponse(900);
        clock += 50;
        await vi.advanceTimersByTimeAsync(50);
        expect(source.sentFrames).toHaveLength(2);
      } finally {
        vi.useRealTimers();
      }
    });

    it("tracks the service rate rather than the capture rate", async () => {
      // A service taking 150ms per frame against a 20fps (50ms) capture tick.
      // Fixed-rate sending would emit 20 frames in a second; this should emit
      // roughly the six or seven the service can actually consume.
      vi.useFakeTimers();
      try {
        let clock = 0;
        const source = fakeSource();
        const publisher = build(source, () => clock);
        let nextResponseAt = Infinity;

        publisher.start();
        for (let i = 0; i < 20; i += 1) {
          clock += 50;
          await vi.advanceTimersByTimeAsync(50);
          if (clock >= nextResponseAt) {
            publisher.noteResponse(400); // slow enough to stay throttled
            nextResponseAt = Infinity;
          }
          if (source.sentFrames.length > 0 && nextResponseAt === Infinity) {
            nextResponseAt = clock + 150;
          }
        }

        // The point is the gap from the unthrottled rate: 20 ticks fired, and
        // fixed-rate sending would have emitted all 20. The exact figure
        // depends on how the simulated responses interleave with ticks, so
        // this asserts the regime rather than a precise count.
        expect(source.sentFrames.length).toBeGreaterThanOrEqual(4);
        expect(source.sentFrames.length).toBeLessThanOrEqual(12);
      } finally {
        vi.useRealTimers();
      }
    });

    it("recovers if a response never arrives", async () => {
      // A lost response must not stall input forever.
      vi.useFakeTimers();
      try {
        let clock = 0;
        const source = fakeSource();
        const publisher = build(source, () => clock);

        publisher.start();
        clock += 50;
        await vi.advanceTimersByTimeAsync(50);
        publisher.noteResponse(900); // congested, so it starts waiting
        expect(source.sentFrames).toHaveLength(1);

        for (let i = 0; i < 15; i += 1) {
          clock += 50;
          await vi.advanceTimersByTimeAsync(50);
        }

        expect(source.sentFrames.length).toBeGreaterThan(1);
        expect(publisher.stats.timedOutResponses).toBeGreaterThan(0);
      } finally {
        vi.useRealTimers();
      }
    });
  });

  it("stops sending after stop()", async () => {
    vi.useFakeTimers();
    try {
      const surface: CaptureSurface = {
        width: 32,
        height: 16,
        captureFrame: vi.fn(),
        encodeJpeg: vi.fn().mockResolvedValue(fakeBlob(1)),
      };
      const source = fakeSource();
      const publisher = new FramePublisher({
        video: { readyState: HAVE_CURRENT_DATA },
        source,
        surface,
        targetFps: 10,
      });

      publisher.start();
      await vi.advanceTimersByTimeAsync(100);
      const sentSoFar = source.sentFrames.length;

      publisher.stop();
      await vi.advanceTimersByTimeAsync(500);

      expect(source.sentFrames.length).toBe(sentSoFar);
      expect(publisher.isRunning).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});
