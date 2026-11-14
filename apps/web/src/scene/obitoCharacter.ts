/**
 * The Obito boss -- Feature Brief 02 §5.
 *
 * **The model has no skeleton and no animation clips at all** (inventoried:
 * 15 static meshes, 0 bones, 0 clips). Unlike Naruto, whose 132-bone rig let
 * a run cycle be retargeted onto it, nothing here can be posed. Confirmed
 * with the project owner: animate the whole object by transform instead.
 *
 * So every "animation" below is a rigid-body move -- bob, lean, slide,
 * lunge, recoil, topple, dissolve. For a masked, cloaked character seen at
 * duel distance that reads acceptably; it would not for a close-up.
 *
 * All motion is driven from an explicit state + elapsed time rather than by
 * mutating transforms incrementally, so an interrupted move (a hit landing
 * mid-slide, or the encounter ending mid-lunge) can never leave him stuck
 * part-way. Every state resolves back to a known pose.
 */

import * as THREE from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { LANE_X, PLAYER_HEIGHT, type LaneIndex } from "../config/gameConfig";
import { OBITO_COMBAT_Z, OBITO_LANE_SLIDE_S } from "../config/bossConfig";

const MODEL_URL = "/models/obito/obito.fbx";
/** Slightly taller than Naruto so he reads as the threat. */
const OBITO_HEIGHT = PLAYER_HEIGHT * 1.08;

type Motion =
  | { kind: "idle" }
  | { kind: "sliding"; from: number; to: number; elapsed: number; duration: number }
  | { kind: "lunging"; elapsed: number; duration: number }
  | { kind: "recoiling"; elapsed: number; duration: number }
  /** Feature Brief 04 §4.1: a long, visible wind-up held for the whole
   * warning window, so the threat is readable from his body and not only
   * from the UI. */
  | { kind: "charging"; elapsed: number; duration: number }
  | { kind: "defeated"; elapsed: number; duration: number };

const LUNGE_S = 0.26;
const RECOIL_S = 0.34;

function easeOutCubic(t: number): number {
  const inv = 1 - t;
  return 1 - inv * inv * inv;
}

export class ObitoCharacter {
  /** Container the game positions; the model hangs under it with its feet
   * at the container's origin. */
  readonly root = new THREE.Group();

  private materials: THREE.Material[] = [];
  private lane: LaneIndex = 1;
  // Explicitly `number`: LANE_X is `as const`, so inference would otherwise
  // pin this to the literal type of whichever lane it was initialised from.
  private laneX: number = LANE_X[1];
  private motion: Motion = { kind: "idle" };
  private phase = Math.random() * Math.PI * 2;
  private loaded = false;
  private hitFlash = 0;
  /** 0..1 special-attack charge, drives the ominous glow (Brief 04 §4.1). */
  private charge = 0;

  static async load(): Promise<ObitoCharacter> {
    const obito = new ObitoCharacter();
    await obito.loadModel();
    return obito;
  }

  private async loadModel(): Promise<void> {
    const model = await new FBXLoader().loadAsync(MODEL_URL);
    stripEmbeddedLights(model);

    // Normalise scale/orientation the same way the player character is
    // handled, and clear any cached skinned bounds first -- see
    // playerCharacter.measureBounds for why that cache is a trap.
    model.rotation.y = Math.PI; // face +Z, back down the track toward Naruto

    const measured = measureBounds(model);
    const height = measured.max.y - measured.min.y;
    const scale = height > 0 ? OBITO_HEIGHT / height : 1;
    model.scale.setScalar(scale);

    const scaled = measureBounds(model);
    model.position.y -= scaled.min.y; // plant the feet on the container origin
    model.position.x -= (scaled.min.x + scaled.max.x) / 2; // centre horizontally

    applyObitoTextures(model, this.materials);

    this.root.add(model);
    this.root.position.set(this.laneX, 0, OBITO_COMBAT_Z);
    this.root.visible = false; // only shown during an encounter
    this.loaded = true;
  }

  get isLoaded(): boolean {
    return this.loaded;
  }

  get currentLane(): LaneIndex {
    return this.lane;
  }

  /** Head-height anchor for the health bar, in world space. */
  get headHeight(): number {
    return OBITO_HEIGHT;
  }

  setVisible(visible: boolean): void {
    this.root.visible = visible;
  }

  /** Place him instantly (used when an encounter begins). */
  placeAt(lane: LaneIndex, z: number): void {
    this.lane = lane;
    this.laneX = LANE_X[lane];
    this.motion = { kind: "idle" };
    this.root.position.set(this.laneX, 0, z);
    this.root.rotation.set(0, 0, 0);
    this.setOpacity(1);
  }

  setZ(z: number): void {
    this.root.position.z = z;
  }

  /** Slide to another lane -- his dodge, and his repositioning between
   * attacks. Duration matches the combat controller's own slide timing so
   * the visual and the logical position never disagree. */
  moveToLane(lane: LaneIndex): void {
    if (this.motion.kind === "defeated") return;
    this.lane = lane;
    this.motion = {
      kind: "sliding",
      from: this.laneX,
      to: LANE_X[lane],
      elapsed: 0,
      duration: OBITO_LANE_SLIDE_S,
    };
  }

  /** Forward lunge as he releases an attack. */
  playAttack(): void {
    if (this.motion.kind === "defeated") return;
    this.motion = { kind: "lunging", elapsed: 0, duration: LUNGE_S };
  }

  /** Knockback + flash when Naruto's attack connects. */
  playHit(): void {
    if (this.motion.kind === "defeated") return;
    this.motion = { kind: "recoiling", elapsed: 0, duration: RECOIL_S };
    this.hitFlash = 1;
  }

  /**
   * Feature Brief 04 §4.1: hold a charged wind-up for the full warning
   * window. Driven by an explicit duration rather than "until told to stop"
   * so it cannot outlive the window if a frame is dropped.
   */
  playSpecialWindup(durationS: number): void {
    if (this.motion.kind === "defeated") return;
    this.motion = { kind: "charging", elapsed: 0, duration: Math.max(durationS, 0.1) };
  }

  /** Cancel the wind-up on any resolution -- part of the single teardown. */
  endSpecialWindup(): void {
    if (this.motion.kind === "charging") {
      this.motion = { kind: "idle" };
    }
    this.charge = 0;
  }

  playDefeat(durationS: number): void {
    this.motion = { kind: "defeated", elapsed: 0, duration: durationS };
  }

  get isDefeatFinished(): boolean {
    return this.motion.kind === "defeated" && this.motion.elapsed >= this.motion.duration;
  }

  update(dt: number): void {
    if (!this.loaded) return;
    this.phase += dt;

    let x = this.laneX;
    let z = this.root.position.z;
    let y = 0;
    let tilt = 0;
    let roll = 0;
    let opacity = 1;

    const motion = this.motion;
    switch (motion.kind) {
      case "idle": {
        // Slow breathing bob + sway so he never looks like a frozen prop.
        y = Math.sin(this.phase * 1.6) * 0.035;
        roll = Math.sin(this.phase * 0.9) * 0.02;
        break;
      }
      case "sliding": {
        motion.elapsed += dt;
        const t = Math.min(1, motion.elapsed / motion.duration);
        x = motion.from + (motion.to - motion.from) * easeOutCubic(t);
        // Bank into the slide, easing out as he arrives.
        roll = -Math.sin(Math.PI * t) * 0.22 * Math.sign(motion.to - motion.from);
        y = Math.sin(this.phase * 1.6) * 0.02;
        if (t >= 1) {
          this.laneX = motion.to;
          x = motion.to;
          this.motion = { kind: "idle" };
        }
        break;
      }
      case "lunging": {
        motion.elapsed += dt;
        const t = Math.min(1, motion.elapsed / motion.duration);
        // Snap forward then settle back.
        const push = Math.sin(Math.PI * t);
        z += push * 0.5;
        tilt = push * 0.28;
        if (t >= 1) this.motion = { kind: "idle" };
        break;
      }
      case "recoiling": {
        motion.elapsed += dt;
        const t = Math.min(1, motion.elapsed / motion.duration);
        const shove = Math.sin(Math.PI * t);
        z -= shove * 0.55;
        tilt = -shove * 0.3;
        // Small shudder on top of the knockback.
        x += Math.sin(t * Math.PI * 8) * 0.05 * (1 - t);
        if (t >= 1) this.motion = { kind: "idle" };
        break;
      }
      case "charging": {
        motion.elapsed += dt;
        const t = Math.min(1, motion.elapsed / motion.duration);
        // Rear back and sink, tremor building as the release approaches, so
        // the wind-up escalates rather than looping (§4.1).
        tilt = -0.34 * easeOutCubic(t);
        y = -0.16 * easeOutCubic(t);
        z -= 0.35 * easeOutCubic(t);
        const tremor = 0.02 + 0.07 * t;
        x += Math.sin(this.phase * 34) * tremor;
        roll = Math.sin(this.phase * 27) * tremor * 0.7;
        this.charge = t;
        if (t >= 1) this.motion = { kind: "idle" };
        break;
      }
      case "defeated": {
        motion.elapsed += dt;
        const t = Math.min(1, motion.elapsed / motion.duration);
        // Topple backward and dissolve out.
        tilt = -easeOutCubic(t) * (Math.PI / 2.2);
        y = -easeOutCubic(t) * 0.15;
        opacity = 1 - easeOutCubic(Math.max(0, (t - 0.35) / 0.65));
        break;
      }
    }

    this.root.position.set(x, y, z);
    this.root.rotation.set(tilt, 0, roll);

    this.hitFlash = Math.max(0, this.hitFlash - dt * 3);
    if (motion.kind !== "charging") {
      this.charge = Math.max(0, this.charge - dt * 2.5);
    }
    this.applyVisuals(opacity, this.hitFlash);
  }

  private applyVisuals(opacity: number, flash: number): void {
    const transparent = opacity < 0.999;
    for (const material of this.materials) {
      material.transparent = transparent;
      material.opacity = opacity;
      material.depthWrite = !transparent;
      const emissive = (material as THREE.MeshPhongMaterial).emissive;
      if (emissive) {
        // Red on a hit; a deepening crimson-violet while charging the
        // special, so the threat is legible off his body (Brief 04 §4.1).
        emissive.setRGB(flash * 0.7 + this.charge * 0.5, 0, this.charge * 0.24);
      }
    }
  }

  private setOpacity(opacity: number): void {
    this.applyVisuals(opacity, 0);
  }

  /** Return to a clean, undamaged idle for the next encounter. */
  resetForEncounter(lane: LaneIndex, z: number): void {
    this.hitFlash = 0;
    this.charge = 0;
    this.placeAt(lane, z);
  }

  dispose(): void {
    this.root.traverse((object) => {
      const mesh = object as Partial<THREE.Mesh>;
      mesh.geometry?.dispose();
    });
    for (const material of this.materials) material.dispose();
    this.materials = [];
    this.root.clear();
    this.root.parent?.remove(this.root);
  }
}

/**
 * Bind the pack's textures and mark meshes for shadow casting.
 *
 * The FBX's own texture references do not resolve -- every material loads
 * with `map = none`, which left the suit, cloak and mask as flat white and
 * made Obito read as a featureless pale figure. The two PNGs that ship
 * alongside the model are therefore attached explicitly, matched by the
 * material names the file actually uses (verified by loading it and dumping
 * them): "Suit"/"Black suit"/"Red suit" take the suit texture, and the mask
 * material takes the mask texture.
 */
function applyObitoTextures(model: THREE.Object3D, collected: THREE.Material[]): void {
  const loader = new THREE.TextureLoader();
  const suit = loader.load("/models/obito/suit.png");
  suit.colorSpace = THREE.SRGBColorSpace;
  suit.flipY = false; // FBX UVs are authored with the opposite V convention
  const mask = loader.load("/models/obito/MaskTexture2.png");
  mask.colorSpace = THREE.SRGBColorSpace;
  mask.flipY = false;

  model.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.castShadow = true;
    mesh.receiveShadow = false;

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      if (!material) continue;
      collected.push(material);

      const named = material as THREE.MeshPhongMaterial;
      const name = (named.name || "").toLowerCase();
      if (/suit/.test(name)) {
        named.map = suit;
        named.color.set(0xffffff); // let the texture supply the colour
        named.needsUpdate = true;
      } else if (mesh.name.toLowerCase() === "mask" || name === "material") {
        named.map = mask;
        named.color.set(0xffffff);
        named.needsUpdate = true;
      }
    }
  });
}

/**
 * Remove lights the model ships with.
 *
 * `tobi_v9.fbx` contains a white PointLight at **intensity 10** left over
 * from whatever scene it was authored in. It is inert while Obito is hidden
 * (three skips invisible subtrees when gathering lights), so the running
 * game is unaffected -- but the instant an encounter made him visible it
 * blew the entire environment out to near-white. The scene owns its own
 * lighting (skyAndLighting.ts); a character must never bring its own.
 */
function stripEmbeddedLights(model: THREE.Object3D): void {
  const lights: THREE.Object3D[] = [];
  model.traverse((object) => {
    if ((object as THREE.Light).isLight) lights.push(object);
  });
  for (const light of lights) light.parent?.remove(light);
}

/** Same cached-bounds guard as playerCharacter: three caches a mesh's
 * boundingBox and never invalidates it when the object is transformed. */
function measureBounds(model: THREE.Object3D): THREE.Box3 {
  model.updateMatrixWorld(true);
  model.traverse((object) => {
    const mesh = object as unknown as {
      isMesh?: boolean;
      geometry?: { boundingBox: THREE.Box3 | null; boundingSphere: THREE.Sphere | null };
    };
    if (mesh.isMesh && mesh.geometry) {
      mesh.geometry.boundingBox = null;
      mesh.geometry.boundingSphere = null;
    }
  });
  return new THREE.Box3().setFromObject(model);
}
