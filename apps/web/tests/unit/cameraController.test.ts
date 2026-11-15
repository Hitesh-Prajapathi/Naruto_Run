import { describe, expect, it, vi } from "vitest";
import { CameraController } from "../../src/camera/cameraController";

class FakeTrack {
  private endedHandler: (() => void) | null = null;
  stopped = false;

  addEventListener(type: string, handler: () => void): void {
    if (type === "ended") {
      this.endedHandler = handler;
    }
  }

  stop(): void {
    this.stopped = true;
  }

  triggerEnded(): void {
    this.endedHandler?.();
  }
}

class FakeStream {
  readonly track = new FakeTrack();
  getVideoTracks(): FakeTrack[] {
    return [this.track];
  }
  getTracks(): FakeTrack[] {
    return [this.track];
  }
}

class FakeVideo {
  srcObject: unknown = null;
  play(): Promise<void> {
    return Promise.resolve();
  }
}

function deniedError(): DOMException {
  return new DOMException("permission denied", "NotAllowedError");
}

describe("CameraController", () => {
  it("starts idle", () => {
    const controller = new CameraController({ video: new FakeVideo() as unknown as HTMLVideoElement });
    expect(controller.cameraState).toBe("idle");
  });

  it("goes idle -> requesting -> ready on a granted permission", async () => {
    const stream = new FakeStream();
    const mediaDevices = { getUserMedia: vi.fn().mockResolvedValue(stream) };
    const video = new FakeVideo();
    const controller = new CameraController({
      video: video as unknown as HTMLVideoElement,
      mediaDevices: mediaDevices as unknown as MediaDevices,
    });
    const states: string[] = [];
    controller.onStateChange((state) => states.push(state));

    await controller.start();

    expect(states).toEqual(["requesting", "ready"]);
    expect(video.srcObject).toBe(stream);
  });

  it("goes to denied on a NotAllowedError", async () => {
    const mediaDevices = { getUserMedia: vi.fn().mockRejectedValue(deniedError()) };
    const controller = new CameraController({
      video: new FakeVideo() as unknown as HTMLVideoElement,
      mediaDevices: mediaDevices as unknown as MediaDevices,
    });
    const states: string[] = [];
    controller.onStateChange((state) => states.push(state));

    await controller.start();

    expect(states).toEqual(["requesting", "denied"]);
    expect(controller.lastError).toContain("permission denied");
  });

  it("goes to error on any other failure", async () => {
    const mediaDevices = { getUserMedia: vi.fn().mockRejectedValue(new Error("device busy")) };
    const controller = new CameraController({
      video: new FakeVideo() as unknown as HTMLVideoElement,
      mediaDevices: mediaDevices as unknown as MediaDevices,
    });
    const states: string[] = [];
    controller.onStateChange((state) => states.push(state));

    await controller.start();

    expect(states).toEqual(["requesting", "error"]);
  });

  it("goes to ended when stop() is called after being ready", async () => {
    const stream = new FakeStream();
    const mediaDevices = { getUserMedia: vi.fn().mockResolvedValue(stream) };
    const video = new FakeVideo();
    const controller = new CameraController({
      video: video as unknown as HTMLVideoElement,
      mediaDevices: mediaDevices as unknown as MediaDevices,
    });
    await controller.start();

    controller.stop();

    expect(controller.cameraState).toBe("ended");
    expect(stream.track.stopped).toBe(true);
    expect(video.srcObject).toBeNull();
  });

  it("goes to ended when the video track ends on its own", async () => {
    const stream = new FakeStream();
    const mediaDevices = { getUserMedia: vi.fn().mockResolvedValue(stream) };
    const controller = new CameraController({
      video: new FakeVideo() as unknown as HTMLVideoElement,
      mediaDevices: mediaDevices as unknown as MediaDevices,
    });
    await controller.start();

    stream.track.triggerEnded();

    expect(controller.cameraState).toBe("ended");
  });

  it("goes to error when getUserMedia is unavailable", async () => {
    const controller = new CameraController({
      video: new FakeVideo() as unknown as HTMLVideoElement,
      mediaDevices: {} as unknown as MediaDevices,
    });

    await controller.start();

    expect(controller.cameraState).toBe("error");
  });

  it("ignores a second start() call while already requesting or ready", async () => {
    const stream = new FakeStream();
    const mediaDevices = { getUserMedia: vi.fn().mockResolvedValue(stream) };
    const controller = new CameraController({
      video: new FakeVideo() as unknown as HTMLVideoElement,
      mediaDevices: mediaDevices as unknown as MediaDevices,
    });

    await Promise.all([controller.start(), controller.start()]);

    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
  });
});
