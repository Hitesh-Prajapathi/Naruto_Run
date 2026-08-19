import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { updateForestProps } from "../../src/scene/forestProps";

describe("updateForestProps", () => {
  it("advances every prop's Z by deltaZ", () => {
    const a = new THREE.Group();
    a.position.set(1, 0, -10);
    const b = new THREE.Group();
    b.position.set(-2, 0, -30);

    updateForestProps([a, b], 5);

    expect(a.position.z).toBeCloseTo(-5, 5);
    expect(b.position.z).toBeCloseTo(-25, 5);
  });

  it("leaves X and Y untouched", () => {
    const prop = new THREE.Group();
    prop.position.set(3, 1.5, -10);

    updateForestProps([prop], 7);

    expect(prop.position.x).toBe(3);
    expect(prop.position.y).toBe(1.5);
  });

  it("is a no-op on an empty list", () => {
    expect(() => updateForestProps([], 5)).not.toThrow();
  });
});
