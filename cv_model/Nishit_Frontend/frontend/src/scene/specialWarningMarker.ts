/**
 * The world-space half of the special-attack warning -- Feature Brief 04
 * §4.1: "warning above Naruto's head... clearly separated from his health
 * bar so the two don't compete."
 *
 * Same construction as HealthBar (canvas texture on a billboarded plane with
 * `depthTest: false`), deliberately, so it sits in the scene and tracks him
 * rather than floating in screen space. It is anchored a further 0.75 units
 * above the health bar, which is the separation the brief asks for.
 *
 * The palette is Obito's dark red, chosen to be categorically unlike the
 * HUD's orange -- see DANGER_COLOR in specialAttackConfig.
 */

import * as THREE from "three";
import {
  COUNTER_KEY_LABEL,
  COUNTER_SIGN,
  DANGER_COLOR,
  DANGER_COLOR_DEEP,
} from "../config/specialAttackConfig";

const TEXTURE_W = 512;
const TEXTURE_H = 192;
const WORLD_WIDTH = 2.2;
const WORLD_HEIGHT = WORLD_WIDTH * (TEXTURE_H / TEXTURE_W);
/**
 * Clearance above Naruto's health bar.
 *
 * Lifted well clear of head height rather than sitting just above the bar.
 * Obito stands 8 units down the track, so in the duel framing his head
 * projects to almost exactly the screen height of Naruto's — and this panel
 * draws with `depthTest: false`, so at the original 0.75 it painted straight
 * over him and hid the wind-up that §4.1 requires the player be able to
 * read off his body. This puts the banner above the horizon line, where
 * nothing in the fight competes with it.
 */
const EXTRA_LIFT = 2.8;

export class SpecialWarningMarker {
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D | null;
  private readonly texture: THREE.CanvasTexture;
  private readonly material: THREE.MeshBasicMaterial;
  private readonly mesh: THREE.Mesh;

  private pulse = 0;
  private lastDrawn = -1;

  constructor(scene: THREE.Scene) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = TEXTURE_W;
    this.canvas.height = TEXTURE_H;
    this.context = this.canvas.getContext("2d");

    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.material = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      depthTest: false, // must never be lost behind foliage
    });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(WORLD_WIDTH, WORLD_HEIGHT), this.material);
    // Above the health bars (renderOrder 10) and the damage numbers (11).
    this.mesh.renderOrder = 12;
    this.mesh.visible = false;
    scene.add(this.mesh);
    this.draw(0);
  }

  get isVisible(): boolean {
    return this.mesh.visible;
  }

  setVisible(visible: boolean): void {
    this.mesh.visible = visible;
  }

  /**
   * @param anchor world position of Naruto's feet
   * @param headHeight his height, i.e. where the health bar sits
   * @param progress 0..1 through the warning window
   */
  update(
    rawDt: number,
    anchor: THREE.Vector3,
    headHeight: number,
    progress: number,
    camera: THREE.Camera,
  ): void {
    if (!this.mesh.visible) return;
    this.pulse += rawDt * 7;
    // A gentle throb on top of the anchor so the eye is drawn to it.
    const bob = Math.sin(this.pulse) * 0.045;
    this.mesh.position.set(
      anchor.x,
      anchor.y + headHeight + EXTRA_LIFT + WORLD_HEIGHT / 2 + bob,
      anchor.z,
    );
    this.mesh.quaternion.copy(camera.quaternion);
    this.draw(progress);
  }

  clear(): void {
    this.mesh.visible = false;
  }

  private draw(progress: number): void {
    const context = this.context;
    if (!context) return;
    // Canvas uploads are not free; only redraw when the bar visibly moved.
    const signature = Math.round(progress * 240);
    if (signature === this.lastDrawn) return;
    this.lastDrawn = signature;

    context.clearRect(0, 0, TEXTURE_W, TEXTURE_H);

    // Backing plate in Obito's deep red, hard-edged rather than the rounded
    // brown of the friendly HUD -- it should not look like the same family.
    context.fillStyle = "rgba(20, 3, 10, 0.9)";
    context.fillRect(8, 8, TEXTURE_W - 16, TEXTURE_H - 16);
    context.strokeStyle = DANGER_COLOR;
    context.lineWidth = 5;
    context.strokeRect(8, 8, TEXTURE_W - 16, TEXTURE_H - 16);

    context.textAlign = "center";
    context.textBaseline = "middle";

    context.font = "bold 44px system-ui, sans-serif";
    context.fillStyle = DANGER_COLOR;
    context.fillText("SPECIAL ATTACK", TEXTURE_W / 2, 52);

    // The instruction, with both the sign name and the keyboard fallback --
    // brief §4.6 requires the key be on screen, not just in the code.
    //
    // No zodiac glyph here, unlike the combat HUD's seals: this is drawn to
    // a canvas, where the CJK fallback font is whatever the OS happens to
    // provide, and a glyph that renders as tofu on the one warning the
    // player must not misread is not worth the flavour.
    context.font = "bold 54px system-ui, sans-serif";
    context.fillStyle = "#ffffff";
    context.fillText(
      `${COUNTER_SIGN.toUpperCase()}   [${COUNTER_KEY_LABEL}]`,
      TEXTURE_W / 2,
      108,
    );

    // Countdown bar, draining left-to-right over the whole window.
    const barX = 28;
    const barW = TEXTURE_W - 56;
    const barY = 148;
    const barH = 20;
    context.fillStyle = DANGER_COLOR_DEEP;
    context.fillRect(barX, barY, barW, barH);
    context.fillStyle = DANGER_COLOR;
    context.fillRect(barX, barY, barW * Math.max(0, 1 - progress), barH);

    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.texture.dispose();
    this.material.dispose();
    this.mesh.parent?.remove(this.mesh);
  }
}
