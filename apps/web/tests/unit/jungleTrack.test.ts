import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { JungleTrack } from "../../src/scene/jungleTrack";
import { addEnvironmentProps } from "../../src/scene/environmentProps";
import { BASE_RUN_SPEED_U_S, CAMERA_POSITION, GROUND_TILE_COUNT, GROUND_TILE_LENGTH } from "../../src/config/gameConfig";

describe("JungleTrack", () => {
  it("creates a dirt tile, a grass terrain plane, and an anchor per tile", () => {
    const scene = new THREE.Scene();
    new JungleTrack(scene);
    // 3 objects per tile: the dirt path mesh, the wide grass terrain plane
    // beneath/beside it, and the prop anchor.
    expect(scene.children.length).toBe(GROUND_TILE_COUNT * 3);
  });

  it("advances every tile's Z by the given delta, or by delta minus one span if it wraps", () => {
    // The frontmost tile is constructed already sitting exactly at the
    // recycle threshold (see the constructor's "start already at maximum
    // extent" comment), so any positive update immediately wraps it -- a
    // deliberate choice, not a bug, but it means this test can't assume
    // every tile moves by a uniform +delta.
    const scene = new THREE.Scene();
    const track = new JungleTrack(scene);
    const span = GROUND_TILE_COUNT * GROUND_TILE_LENGTH;
    const before = track.anchors.map((anchor) => anchor.position.z);

    track.update(3);

    const after = track.anchors.map((anchor) => anchor.position.z);
    for (let i = 0; i < before.length; i += 1) {
      const moved = after[i]! - before[i]!;
      const wrapped = Math.abs(moved - (3 - span)) < 1e-5;
      const unwrapped = Math.abs(moved - 3) < 1e-5;
      expect(wrapped || unwrapped).toBe(true);
    }
  });

  it("never leaves a gap or overlap between tiles over a long run", () => {
    const scene = new THREE.Scene();
    const track = new JungleTrack(scene);

    for (let i = 0; i < 2000; i += 1) {
      track.update(0.5);
      const sortedZ = track.anchors.map((a) => a.position.z).sort((a, b) => b - a);
      for (let j = 1; j < sortedZ.length; j += 1) {
        expect(sortedZ[j - 1]! - sortedZ[j]!).toBeCloseTo(GROUND_TILE_LENGTH, 5);
      }
    }
  });

  it("keeps tile and anchor Z in sync", () => {
    const scene = new THREE.Scene();
    const track = new JungleTrack(scene);
    track.update(17.3);
    for (let i = 0; i < track.anchors.length; i += 1) {
      expect(track.anchors[i]!.position.z).toBeCloseTo(track.tileMeshZ(i), 5);
    }
  });

  it("never leaves the camera with no ground beneath it, at any point across a long run", () => {
    // Regression test for a real bug: recycling one tile at a time creates a
    // "sawtooth" dip in the belt's near edge of up to one GROUND_TILE_LENGTH
    // every recycle cycle. With the wrong tile length/threshold this dip
    // reached back past the camera entirely -- ground visibly missing under
    // the player. Every tile's *near* edge (position.z + length/2, since
    // tiles are laid out toward -Z) must never leave a gap in front of the
    // camera at any simulated moment, not just at the sampled instants the
    // other tests happen to check.
    const scene = new THREE.Scene();
    const track = new JungleTrack(scene);
    const dt = 1 / 60;
    const deltaZ = BASE_RUN_SPEED_U_S * dt;
    const halfTile = GROUND_TILE_LENGTH / 2;

    for (let tick = 0; tick < 60 * 30; tick += 1) {
      track.update(deltaZ);
      const nearestEdge = Math.max(...track.anchors.map((a) => a.position.z + halfTile));
      expect(nearestEdge).toBeGreaterThanOrEqual(CAMERA_POSITION.z);
    }
  });
});

describe("addEnvironmentProps", () => {
  it("adds at least one prop to every tile anchor", () => {
    const scene = new THREE.Scene();
    const track = new JungleTrack(scene);

    addEnvironmentProps(track);

    for (const anchor of track.anchors) {
      expect(anchor.children.length).toBeGreaterThan(0);
    }
  });

  it("does not touch the ground tile meshes themselves", () => {
    const scene = new THREE.Scene();
    const track = new JungleTrack(scene);
    const meshChildCounts = scene.children
      .filter((child) => child instanceof THREE.Mesh)
      .map((mesh) => mesh.children.length);

    addEnvironmentProps(track);

    const afterMeshChildCounts = scene.children
      .filter((child) => child instanceof THREE.Mesh)
      .map((mesh) => mesh.children.length);
    expect(afterMeshChildCounts).toEqual(meshChildCounts);
  });
});
