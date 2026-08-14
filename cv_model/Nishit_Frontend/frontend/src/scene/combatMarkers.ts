/**
 * Two small in-world combat readouts: the telegraph lane marker and the
 * floating damage numbers -- Feature Brief 02 §5.3 and §5.5.
 *
 * The lane marker is the thing that makes the fight fair: it shows *where*
 * Obito's attack will land, held for the full telegraph window, filling as
 * the wind-up completes so the player can read how long they have left.
 *
 * Both are pooled/preallocated and hidden rather than created and destroyed,
 * for the same reason the VFX are (brief §5.7).
 */

import * as THREE from "three";
import { LANE_WIDTH, LANE_X, type LaneIndex } from "../config/gameConfig";
import { OBITO_COMBAT_Z } from "../config/bossConfig";

/** Pulsing translucent strip down the threatened lane. */
export class TelegraphMarker {
  private readonly mesh: THREE.Mesh;
  private readonly material: THREE.MeshBasicMaterial;
  private pulse = 0;

  constructor(scene: THREE.Scene) {
    // Spans from Obito to just behind the player, so the whole threatened
    // corridor is visible rather than a spot at one end.
    const length = Math.abs(OBITO_COMBAT_Z) + 4;
    const geometry = new THREE.PlaneGeometry(LANE_WIDTH * 0.86, length);
    geometry.rotateX(-Math.PI / 2);
    geometry.translate(0, 0, -length / 2 + 2);

    this.material = new THREE.MeshBasicMaterial({
      color: 0xff3b30,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.position.y = 0.03; // just above the path, avoids z-fighting
    this.mesh.renderOrder = 2;
    this.mesh.visible = false;
    scene.add(this.mesh);
  }

  /** @param lane lane under threat, or null to clear the marker */
  set(lane: LaneIndex | null, progress: number): void {
    if (lane === null) {
      this.mesh.visible = false;
      return;
    }
    this.mesh.visible = true;
    this.mesh.position.x = LANE_X[lane];
    // Brightens as the wind-up completes, so urgency is readable at a glance.
    this.material.opacity = 0.18 + 0.34 * progress + Math.sin(this.pulse) * 0.05;
  }

  update(dt: number): void {
    this.pulse += dt * 9;
  }

  clear(): void {
    this.mesh.visible = false;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.mesh.parent?.remove(this.mesh);
  }
}

const NUMBER_POOL = 6;
const NUMBER_LIFETIME_S = 0.9;

interface FloatingNumber {
  sprite: THREE.Sprite;
  material: THREE.SpriteMaterial;
  texture: THREE.CanvasTexture;
  canvas: HTMLCanvasElement;
  elapsed: number;
  origin: THREE.Vector3;
  active: boolean;
}

/** Pooled "-60" / "-20" numbers that rise and fade at the point of impact. */
export class DamageNumbers {
  private readonly pool: FloatingNumber[] = [];

  constructor(scene: THREE.Scene) {
    for (let i = 0; i < NUMBER_POOL; i += 1) {
      const canvas = document.createElement("canvas");
      canvas.width = 128;
      canvas.height = 64;
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(1.1, 0.55, 1);
      sprite.visible = false;
      sprite.renderOrder = 11;
      scene.add(sprite);
      this.pool.push({ sprite, material, texture, canvas, elapsed: 0, origin: new THREE.Vector3(), active: false });
    }
  }

  spawn(damage: number, position: THREE.Vector3, color: string): void {
    const entry = this.pool.find((candidate) => !candidate.active);
    if (!entry) return; // drop rather than allocate mid-fight

    const context = entry.canvas.getContext("2d");
    if (context) {
      context.clearRect(0, 0, entry.canvas.width, entry.canvas.height);
      context.font = "bold 44px system-ui, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.lineWidth = 6;
      context.strokeStyle = "rgba(0,0,0,0.85)";
      context.strokeText(`-${damage}`, 64, 32);
      context.fillStyle = color;
      context.fillText(`-${damage}`, 64, 32);
      entry.texture.needsUpdate = true;
    }

    entry.origin.copy(position);
    entry.elapsed = 0;
    entry.active = true;
    entry.sprite.visible = true;
    entry.sprite.position.copy(position);
    entry.material.opacity = 1;
  }

  update(dt: number): void {
    for (const entry of this.pool) {
      if (!entry.active) continue;
      entry.elapsed += dt;
      const t = entry.elapsed / NUMBER_LIFETIME_S;
      if (t >= 1) {
        entry.active = false;
        entry.sprite.visible = false;
        continue;
      }
      entry.sprite.position.set(entry.origin.x, entry.origin.y + t * 0.9, entry.origin.z);
      entry.material.opacity = 1 - t * t;
    }
  }

  clear(): void {
    for (const entry of this.pool) {
      entry.active = false;
      entry.sprite.visible = false;
    }
  }

  dispose(): void {
    for (const entry of this.pool) {
      entry.texture.dispose();
      entry.material.dispose();
      entry.sprite.parent?.remove(entry.sprite);
    }
    this.pool.length = 0;
  }
}
