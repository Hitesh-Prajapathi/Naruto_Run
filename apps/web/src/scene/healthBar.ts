/**
 * World-space, billboarded health bars above each fighter's head --
 * Feature Brief 02 §5.5.
 *
 * Drawn as canvas textures on camera-facing planes rather than DOM overlays,
 * so they sit in the 3D scene, occlude correctly, and scale with distance
 * the way the brief's "anchored above each character's head" implies.
 *
 * Two details the brief calls out and this implements:
 * - a lighter **chip layer** that trails the main bar, so the player can see
 *   how much damage the last hit actually did;
 * - a **dark backing plate**, because the bars have to stay readable against
 *   both bright sky and dark foliage.
 */

import * as THREE from "three";

const TEXTURE_W = 256;
const TEXTURE_H = 64;
const BAR_WORLD_WIDTH = 1.5;
const BAR_WORLD_HEIGHT = BAR_WORLD_WIDTH * (TEXTURE_H / TEXTURE_W);

/** How fast the main bar slides toward the true value (§5.5: ~300ms). */
const BAR_DRAIN_S = 0.3;
/** The chip layer lags further behind so the delta stays visible. */
const CHIP_DRAIN_S = 0.85;
const CHIP_DELAY_S = 0.25;

export interface HealthBarStyle {
  /** Main fill at full health. */
  fill: string;
  label: string;
}

export const NARUTO_BAR_STYLE: HealthBarStyle = { fill: "#ff8a3d", label: "NARUTO" };
export const OBITO_BAR_STYLE: HealthBarStyle = { fill: "#8e2b4a", label: "OBITO" };

/** Colour shift as HP drops (§5.5): full -> amber below 40% -> red below 20%. */
function fillForRatio(ratio: number, base: string): string {
  if (ratio < 0.2) return "#e03131";
  if (ratio < 0.4) return "#e8a33d";
  return base;
}

export class HealthBar {
  readonly sprite: THREE.Mesh;

  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D | null;
  private readonly texture: THREE.CanvasTexture;
  private readonly material: THREE.MeshBasicMaterial;

  private targetRatio = 1;
  private displayRatio = 1;
  private chipRatio = 1;
  private chipDelay = 0;
  private lastDrawn = -1;

  constructor(
    scene: THREE.Scene,
    private readonly style: HealthBarStyle,
    private readonly maxHp: number,
  ) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = TEXTURE_W;
    this.canvas.height = TEXTURE_H;
    this.context = this.canvas.getContext("2d");

    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.material = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      depthTest: false, // always readable, never buried in foliage
    });
    this.sprite = new THREE.Mesh(new THREE.PlaneGeometry(BAR_WORLD_WIDTH, BAR_WORLD_HEIGHT), this.material);
    this.sprite.renderOrder = 10;
    this.sprite.visible = false;
    scene.add(this.sprite);
    this.draw(1, 1, this.maxHp);
  }

  setVisible(visible: boolean): void {
    this.sprite.visible = visible;
  }

  /** Snap to full, for the start of an encounter (§2.3: HP resets). */
  reset(): void {
    this.targetRatio = 1;
    this.displayRatio = 1;
    this.chipRatio = 1;
    this.chipDelay = 0;
    this.draw(1, 1, this.maxHp);
  }

  setHp(hp: number): void {
    const ratio = this.maxHp > 0 ? Math.max(0, Math.min(1, hp / this.maxHp)) : 0;
    if (ratio < this.targetRatio) {
      this.chipDelay = CHIP_DELAY_S; // hold the chip so the loss registers
    }
    this.targetRatio = ratio;
  }

  /**
   * @param anchor world position of the character's feet
   * @param height how far above the feet to float the bar
   */
  update(dt: number, anchor: THREE.Vector3, height: number, camera: THREE.Camera): void {
    // Ease the main bar toward the true value; frame-rate independent so a
    // slow frame doesn't make the drain visibly jump.
    this.displayRatio = damp(this.displayRatio, this.targetRatio, BAR_DRAIN_S, dt);

    if (this.chipDelay > 0) {
      this.chipDelay -= dt;
    } else {
      this.chipRatio = damp(this.chipRatio, this.targetRatio, CHIP_DRAIN_S, dt);
    }
    // The chip can never sit below the main bar.
    this.chipRatio = Math.max(this.chipRatio, this.displayRatio);

    this.sprite.position.set(anchor.x, anchor.y + height + 0.35, anchor.z);
    this.sprite.quaternion.copy(camera.quaternion); // billboard

    this.draw(this.displayRatio, this.chipRatio, Math.round(this.targetRatio * this.maxHp));
  }

  private draw(ratio: number, chip: number, hp: number): void {
    const context = this.context;
    if (!context) return;

    // Redrawing a canvas texture forces a GPU upload, so skip it when
    // nothing perceptible changed -- this runs every frame for two bars.
    const signature = Math.round(ratio * 400) * 1000 + Math.round(chip * 400);
    if (signature === this.lastDrawn) return;
    this.lastDrawn = signature;

    const pad = 6;
    const barX = pad;
    const barY = 26;
    const barW = TEXTURE_W - pad * 2;
    const barH = 22;

    context.clearRect(0, 0, TEXTURE_W, TEXTURE_H);

    // Backing plate -- the readability guarantee against sky and foliage.
    context.fillStyle = "rgba(12, 14, 18, 0.82)";
    roundRect(context, barX - 4, barY - 4, barW + 8, barH + 8, 6);
    context.fill();

    // Empty track.
    context.fillStyle = "rgba(0, 0, 0, 0.55)";
    context.fillRect(barX, barY, barW, barH);

    // Chip (damage just taken), behind the main fill.
    context.fillStyle = "rgba(255, 235, 200, 0.55)";
    context.fillRect(barX, barY, barW * chip, barH);

    // Main fill.
    context.fillStyle = fillForRatio(ratio, this.style.fill);
    context.fillRect(barX, barY, barW * ratio, barH);

    // Outline.
    context.strokeStyle = "rgba(0, 0, 0, 0.9)";
    context.lineWidth = 2;
    context.strokeRect(barX, barY, barW, barH);

    // Label + numeric HP.
    context.font = "bold 16px system-ui, sans-serif";
    context.fillStyle = "#f4f4f5";
    context.textBaseline = "alphabetic";
    context.fillText(this.style.label, barX, barY - 8);
    context.textAlign = "right";
    context.fillText(String(Math.max(0, hp)), barX + barW, barY - 8);
    context.textAlign = "left";

    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.sprite.geometry.dispose();
    this.texture.dispose();
    this.material.dispose();
    this.sprite.parent?.remove(this.sprite);
  }
}

/** Frame-rate-independent exponential approach. */
function damp(current: number, target: number, timeConstant: number, dt: number): number {
  const alpha = 1 - Math.exp(-dt / Math.max(timeConstant, 1e-4));
  return current + (target - current) * alpha;
}

function roundRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + w, y, x + w, y + h, r);
  context.arcTo(x + w, y + h, x, y + h, r);
  context.arcTo(x, y + h, x, y, r);
  context.arcTo(x, y, x + w, y, r);
  context.closePath();
}
