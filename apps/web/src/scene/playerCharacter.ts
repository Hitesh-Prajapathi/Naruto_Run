/**
 * Loads the character, retargets a real run clip onto its skeleton, and
 * drives it with an AnimationMixer.
 *
 * Background (naruto_run_agent_brief.md §P0.1): the character model ships
 * with a skeleton but **no animation clips**, so an earlier pass posed a few
 * leg bones procedurally each frame. That produced barely-perceptible
 * motion, and the arms stayed in the bind T-pose entirely -- reading, fairly,
 * as "the animation is not playing at all". This replaces that approach
 * wholesale: a genuine 0.56s run cycle from a *different* rig
 * (Character_3D_Models/Running_Style/run.fbx) is retargeted onto this
 * character's skeleton via the bone map in characterRig.ts.
 *
 * Model facts, established by loading and inspecting the files rather than
 * assumed:
 * - Authored Z-up, feet at local z~0, ~1.72 units tall (real human scale).
 * - 3 SkinnedMeshes (body, gear, eyes) sharing ONE 132-bone skeleton -- the
 *   same Bone *instances*, so animating that skeleton once animates all
 *   three meshes together.
 * - The retargeted clip binds via `.bones[NAME].quaternion` track names,
 *   which resolve against a SkinnedMesh's skeleton -- hence the mixer is
 *   rooted at a SkinnedMesh, not at the model group.
 *
 * Cadence matching (brief §A) is the other half of "reads as running, not
 * flying": `setSpeed()` scales the clip's timeScale in proportion to world
 * scroll speed so the feet stay planted relative to the ground instead of
 * sliding.
 */

import * as THREE from "three";
import { ColladaLoader } from "three/examples/jsm/loaders/ColladaLoader.js";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { retargetRunClip } from "./retargetRunClip";
import {
  BASE_RUN_SPEED_U_S,
  CHARACTER_MODEL_URL,
  CHARACTER_RUN_CLIP_URL,
  JUMP_CROUCH_SQUASH,
  PLAYER_HEIGHT,
  RUN_CLIP_TIMESCALE_AT_BASE_SPEED,
} from "../config/gameConfig";
import { RUN_CLIP_BONE_MAP } from "./characterRig";

export type CharacterAnimationState = "run" | "jump";

/** Retarget samples the source clip at this rate; the source is 0.56s, so
 * 30fps gives ~17 keyframes, plenty for a smooth loop. */
const RETARGET_FPS = 30;

export class PlayerCharacter {
  /** Stable container the game positions; the model hangs underneath it, so
   * the container's origin is the character's feet regardless of whatever
   * offset the source mesh was authored with. */
  readonly root = new THREE.Group();

  private mixer: THREE.AnimationMixer | null = null;
  private runAction: THREE.AnimationAction | null = null;
  private state: CharacterAnimationState = "run";
  private loaded = false;
  /** The loaded model, held so setCrouch() can squash it independently of
   * the root container the game positions. */
  private model: THREE.Object3D | null = null;
  /** Uniform scale that makes the model PLAYER_HEIGHT tall; setCrouch()
   * multiplies this rather than overwriting it. */
  private baseScale = 1;

  static async load(): Promise<PlayerCharacter> {
    const character = new PlayerCharacter();
    await character.loadModel();
    return character;
  }

  private async loadModel(): Promise<void> {
    const [collada, runFbx] = await Promise.all([
      new ColladaLoader().loadAsync(CHARACTER_MODEL_URL),
      new FBXLoader().loadAsync(CHARACTER_RUN_CLIP_URL),
    ]);
    if (!collada) {
      throw new Error(`failed to load character model: ${CHARACTER_MODEL_URL}`);
    }
    const model = collada.scene;

    // Z-up -> Y-up, plus a 180 degree yaw so the character faces -Z (down
    // the track, away from the camera). Verified from a clean bind-pose
    // render; composed as explicit quaternions because Euler ordering makes
    // "set .rotation.x then .rotation.y" ambiguous to reason about by hand.
    const upAxisFix = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
    const yawFix = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
    model.quaternion.copy(yawFix.multiply(upAxisFix));

    const measured = measureBounds(model);
    const measuredHeight = measured.max.y - measured.min.y;
    this.baseScale = measuredHeight > 0 ? PLAYER_HEIGHT / measuredHeight : 1;
    model.scale.setScalar(this.baseScale);

    // Re-measure after scaling so the soles land exactly on the container's
    // y=0 -- the character must not float (brief §P0.2).
    const scaled = measureBounds(model);
    model.position.y -= scaled.min.y;

    prepareMaterials(model);
    this.model = model;
    this.root.add(model);

    const skinnedMesh = findSkinnedMesh(model);
    if (skinnedMesh) {
      this.setUpAnimation(skinnedMesh, runFbx);
    } else {
      console.warn("playerCharacter: no skinned mesh found; character will not animate");
    }

    this.loaded = true;
  }

  private setUpAnimation(target: THREE.SkinnedMesh, runFbx: THREE.Group): void {
    const sourceClip = runFbx.animations[0];
    const sourceMesh = findSkinnedMesh(runFbx);
    if (!sourceClip || !sourceMesh) {
      console.warn("playerCharacter: run clip missing from FBX; character will not animate");
      return;
    }

    // The source clip carries tracks for a couple of nodes that aren't part
    // of its own skeleton (a leftover "Bip01" root from whatever tool
    // exported it). They bind to nothing and log a warning per track, so
    // drop them up front -- they carry no motion we map anyway.
    const sourceBoneNames = new Set(sourceMesh.skeleton.bones.map((bone) => bone.name));
    const usableClip = sourceClip.clone();
    usableClip.tracks = usableClip.tracks.filter((track) =>
      sourceBoneNames.has(track.name.split(".")[0] ?? ""),
    );

    let retargeted: THREE.AnimationClip | null;
    try {
      retargeted = retargetRunClip(target, sourceMesh, usableClip, {
        boneMap: RUN_CLIP_BONE_MAP,
        fps: RETARGET_FPS,
        name: "run",
      });
    } catch (error) {
      console.error("playerCharacter: run-clip retarget failed", error);
      return;
    }

    // A retarget that silently produced nothing would leave the character in
    // its T-pose with no error -- exactly the failure the brief flagged. Fail
    // loudly instead of shipping a frozen character.
    if (!retargeted || retargeted.tracks.length === 0) {
      console.error("playerCharacter: retargeted run clip has no tracks; check the bone map");
      return;
    }

    this.mixer = new THREE.AnimationMixer(target);
    this.runAction = this.mixer.clipAction(retargeted);
    this.runAction.setLoop(THREE.LoopRepeat, Infinity);
    this.runAction.play();
    this.setSpeed(BASE_RUN_SPEED_U_S);
  }

  get isLoaded(): boolean {
    return this.loaded;
  }

  /** True once a real clip is actually playing -- lets callers (and tests)
   * distinguish "animating" from "loaded but frozen in bind pose". */
  get isAnimating(): boolean {
    return this.runAction !== null;
  }

  /**
   * Playback head of the run clip, in seconds. Exposed so an end-to-end test
   * can assert the animation is genuinely *advancing* rather than merely
   * loaded -- a mixer that exists but is never ticked leaves the character
   * frozen in its bind T-pose with no error anywhere, which is exactly the
   * failure this whole pass was written to fix.
   */
  get clipTime(): number {
    return this.runAction?.time ?? 0;
  }

  setAnimationState(state: CharacterAnimationState): void {
    this.state = state;
    if (!this.runAction) {
      return;
    }
    // The run cycle keeps playing underneath a jump rather than stopping --
    // the legs are mid-stride when the character leaves the ground, and
    // freezing them there looks worse than letting the cycle continue. It's
    // slowed rather than paused so the airborne pose still reads as motion.
    // (A dedicated jump clip would be better; the source FBX only carries a
    // run cycle, so this is the honest best available -- see the report.)
    this.runAction.timeScale = this.currentTimeScale * (state === "jump" ? 0.35 : 1);
  }

  get animationState(): CharacterAnimationState {
    return this.state;
  }

  private currentTimeScale = RUN_CLIP_TIMESCALE_AT_BASE_SPEED;

  /**
   * Match the run cycle's cadence to world scroll speed (brief §A). Without
   * this the feet slide against the ground whenever game speed differs from
   * whatever speed the clip was authored at, which reads as skating.
   */
  setSpeed(worldSpeedUnitsPerSecond: number): void {
    const ratio = BASE_RUN_SPEED_U_S > 0 ? worldSpeedUnitsPerSecond / BASE_RUN_SPEED_U_S : 1;
    this.currentTimeScale = RUN_CLIP_TIMESCALE_AT_BASE_SPEED * ratio;
    this.setAnimationState(this.state);
  }

  /**
   * Squash the body for the jump's anticipation crouch and landing
   * absorption (brief §D). Applied as a non-uniform scale on the model
   * *inside* the root container: because the model's origin sits at the
   * feet, scaling Y compresses the body downward while the soles stay
   * planted on y=0. The slight X/Z widening is the usual squash-and-stretch
   * volume cheat -- it reads as weight, not as the character shrinking.
   */
  setCrouch(amount: number): void {
    const clamped = Math.min(1, Math.max(0, amount));
    const squash = JUMP_CROUCH_SQUASH * clamped;
    const widen = 1 + squash * 0.5;
    this.model?.scale.set(this.baseScale * widen, this.baseScale * (1 - squash), this.baseScale * widen);
  }

  update(dt: number): void {
    this.mixer?.update(dt);
  }

  dispose(): void {
    this.mixer?.stopAllAction();
    this.mixer = null;
    this.runAction = null;
    this.root.traverse((object) => {
      const mesh = object as Partial<THREE.Mesh>;
      mesh.geometry?.dispose();
      const material = mesh.material;
      if (Array.isArray(material)) {
        for (const item of material) disposeMaterial(item);
      } else if (material) {
        disposeMaterial(material);
      }
    });
    this.root.clear();
  }
}

/**
 * World-space bounds of a model, safe to call after changing its transform.
 *
 * `Box3.setFromObject` prefers a SkinnedMesh's *cached* `boundingBox` over
 * recomputing from geometry, and nothing invalidates that cache when the
 * model is rotated or scaled. Measuring this character straight after the
 * Z-up -> Y-up fix therefore returned the stale pre-rotation box: a height
 * of 0.534 (the model's original *depth*) instead of the true 1.722, which
 * scaled the character 3.3x too large and left it lying on its side.
 * Clearing the cache first forces an honest recompute.
 */
function measureBounds(model: THREE.Object3D): THREE.Box3 {
  model.updateMatrixWorld(true);
  model.traverse((object) => {
    // Cast through `unknown` to a standalone shape: @types/three declares
    // these non-nullable, but three itself initialises them to null and
    // treats null as "not yet computed", which is exactly the reset we want.
    // (An intersection type won't do -- `Box3 & (Box3 | null)` is just Box3.)
    const skinned = object as unknown as {
      isSkinnedMesh?: boolean;
      boundingBox: THREE.Box3 | null;
      boundingSphere: THREE.Sphere | null;
    };
    if (skinned.isSkinnedMesh) {
      skinned.boundingBox = null;
      skinned.boundingSphere = null;
    }
  });
  return new THREE.Box3().setFromObject(model);
}

function findSkinnedMesh(root: THREE.Object3D): THREE.SkinnedMesh | null {
  let found: THREE.SkinnedMesh | null = null;
  root.traverse((object) => {
    if (!found && (object as THREE.SkinnedMesh).isSkinnedMesh) {
      found = object as THREE.SkinnedMesh;
    }
  });
  return found;
}

function disposeMaterial(material: THREE.Material): void {
  const withMaps = material as THREE.MeshStandardMaterial | THREE.MeshPhongMaterial;
  withMaps.map?.dispose();
  material.dispose();
}

function prepareMaterials(model: THREE.Object3D): void {
  model.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) {
      return;
    }
    mesh.castShadow = true;
    // The character never needs to receive its own shadow, and skipping it
    // keeps the shadow map budget for the ground.
    mesh.receiveShadow = false;

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const item of materials) {
      const material = item as THREE.MeshPhongMaterial;
      if (material.map) {
        material.map.colorSpace = THREE.SRGBColorSpace;
      }
      // The eye sub-mesh's texture carries its own alpha; without alphaTest
      // its transparent border renders as opaque black rectangles across the
      // face (the "stray black bars" in brief §P0.3).
      if (material.map && /eye/i.test(material.name)) {
        material.transparent = true;
        material.alphaTest = 0.5;
        material.side = THREE.DoubleSide;
      }
    }
  });
}
