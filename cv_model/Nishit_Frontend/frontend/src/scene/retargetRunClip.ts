/**
 * Direction-driven animation retargeting between two unrelated humanoid rigs.
 *
 * The problem this solves: the character model ships no animation, and the
 * only run cycle available (Character_3D_Models/Running_Style/run.fbx) is
 * authored for a completely different skeleton. Two standard approaches were
 * tried against the real files and measured, and both failed:
 *
 * 1. `SkeletonUtils.retargetClip` copies each source bone's *world
 *    orientation* onto the target bone. That assumes both rigs share
 *    bone-axis conventions. These don't, and the result was a crumpled
 *    character -- posed bounding box 0.87 x 0.99 x 1.28, i.e. wider and
 *    deeper than tall, with feet pointing sideways.
 *
 * 2. Rest-relative rotation transfer (delta from each bone's own bind
 *    orientation, re-anchored on the target's). This got the character
 *    upright, but the legs swung sideways instead of forward/back, because
 *    the two rigs' bone rest frames differ by a twist about the limb axis.
 *
 * The root cause is that the source's bind pose is not a usable reference at
 * all: measured, its hips->head axis sits ~52 degrees off vertical and its
 * right leg points nearly straight *backwards* (hips->foot = [-0.12, -0.24,
 * -0.96]). Any method defined relative to that bind pose inherits the mess.
 *
 * So this transfers **limb directions** instead of rotations. For each bone
 * in `LIMB_CHAINS`, it reads the world-space direction from that bone to its
 * child on the source at time t, and aims the corresponding target bone
 * along the same direction -- a minimal-arc rotation applied in world space,
 * then converted back to local against the parent's already-final transform.
 *
 * That makes the transfer immune to bind-pose differences, bone-axis
 * conventions, and limb-length differences all at once; the only thing it
 * needs is that both rigs stand in the same world orientation and face the
 * same way (both face -Z here). The known cost is that twist about a limb's
 * own axis is not transferred -- forearm/wrist roll is lost. For a run cycle
 * viewed from behind that is not perceptible, and it is a good trade for a
 * pose that is correct in every other respect.
 *
 * Baked once at load into a plain AnimationClip; playback afterwards is an
 * ordinary AnimationMixer.
 */

import * as THREE from "three";

/**
 * Aimable limb segments as [bone, child, weight] in the *target* rig's
 * naming; each is resolved to the source rig through the caller's bone map.
 * Ordered parents-first so a bone is aimed only after everything above it in
 * the hierarchy is final.
 *
 * `weight` scales how much of the source's direction is adopted (1 = match
 * it exactly, 0 = stay at bind), via a slerp from the identity rotation.
 * The spine chain is deliberately damped: measured, the source clip leans
 * its torso ~58 degrees forward, which on our character read as nearly
 * doubled-over -- posed bounding box deeper than it was tall. Damping to
 * ~0.2 lands the lean at roughly 10-12 degrees, which is both what a real
 * runner does and what the brief asks for, while still letting the torso
 * counter-rotate subtly with the stride instead of being locked rigid.
 */
export const LIMB_CHAINS: ReadonlyArray<readonly [string, string, number]> = [
  ["hip", "waist", 0.1],
  ["waist", "breast", 0.1],
  ["breast", "neck", 0.1],
  ["neck", "head", 0.1],
  ["L_hleg", "L_lleg", 1],
  ["L_lleg", "L_foot", 1],
  ["L_foot", "L_toe", 1],
  ["R_hleg", "R_lleg", 1],
  ["R_lleg", "R_foot", 1],
  ["R_foot", "R_toe", 1],
];

/**
 * The arms are synthesized rather than retargeted, because the source clip
 * does not actually animate them: measured across the full cycle, its upper
 * arms move only from [0.28, 0.18, -0.94] to [0.38, 0.06, -0.92] -- a wobble
 * of a few degrees -- and both arms point the *same* way instead of
 * counter-swinging. Transferring that gives a character running with two
 * stiff arms held out in front, which is exactly the "hands look stiff, not
 * like a normal game" complaint this work exists to fix.
 *
 * So arms are posed parametrically instead, phase-locked to the same cycle
 * as the legs: each arm hangs down and slightly out, elbow bent so the hand
 * carries forward at hip height, swinging fore/aft opposite its same-side
 * leg. `sideX` is -1 for the left arm and +1 for the right (this rig's left
 * arm rests along -X).
 */
const ARM_CHAINS = [
  { bone: "L_harm", child: "L_larm", sideX: -1, segment: "upper" },
  { bone: "L_larm", child: "L_hand", sideX: -1, segment: "fore" },
  { bone: "R_harm", child: "R_larm", sideX: 1, segment: "upper" },
  { bone: "R_larm", child: "R_hand", sideX: 1, segment: "fore" },
] as const;

/** Peak fore/aft arm swing, as a fraction of the limb direction vector. */
const ARM_SWING = 0.42;

function armDirection(
  target: THREE.Vector3,
  sideX: number,
  segment: "upper" | "fore",
  swing: number,
): THREE.Vector3 {
  // Left and right swing in opposition; each is also opposite its own leg,
  // which the caller arranges by choosing the sign of `swing`.
  const signedSwing = swing * sideX;
  if (segment === "upper") {
    // Hangs down, slightly away from the body, swinging fore/aft.
    return target.set(sideX * 0.26, -0.9, signedSwing * ARM_SWING).normalize();
  }
  // Elbow bent: the forearm carries the hand forward and inward.
  return target.set(sideX * 0.12, -0.34, -0.86 + signedSwing * 0.25).normalize();
}

export interface RetargetOptions {
  /** Target bone name -> source bone name. */
  boneMap: Record<string, string>;
  fps?: number;
  name?: string;
}

interface AimPair {
  targetBone: THREE.Bone;
  targetChild: THREE.Bone;
  sourceBone: THREE.Bone;
  sourceChild: THREE.Bone;
  weight: number;
}

const IDENTITY = new THREE.Quaternion();

function rootOf(object: THREE.Object3D): THREE.Object3D {
  let current: THREE.Object3D = object;
  while (current.parent) current = current.parent;
  return current;
}

function findBone(mesh: THREE.SkinnedMesh, name: string): THREE.Bone | undefined {
  return mesh.skeleton.bones.find((bone) => bone.name === name);
}

/**
 * Bake `clip` (authored for `sourceMesh`) onto `targetMesh`'s skeleton.
 * Returns null if no limb chain could be resolved on both rigs.
 */
export function retargetRunClip(
  targetMesh: THREE.SkinnedMesh,
  sourceMesh: THREE.SkinnedMesh,
  clip: THREE.AnimationClip,
  options: RetargetOptions,
): THREE.AnimationClip | null {
  const fps = options.fps ?? 30;
  const sourceRoot = rootOf(sourceMesh);
  const targetRoot = rootOf(targetMesh);

  const pairs: AimPair[] = [];
  for (const [targetName, targetChildName, weight] of LIMB_CHAINS) {
    const sourceName = options.boneMap[targetName];
    const sourceChildName = options.boneMap[targetChildName];
    if (!sourceName || !sourceChildName) continue;

    const targetBone = findBone(targetMesh, targetName);
    const targetChild = findBone(targetMesh, targetChildName);
    const sourceBone = findBone(sourceMesh, sourceName);
    const sourceChild = findBone(sourceMesh, sourceChildName);
    if (!targetBone || !targetChild || !sourceBone || !sourceChild) continue;

    pairs.push({ targetBone, targetChild, sourceBone, sourceChild, weight });
  }
  if (pairs.length === 0) {
    return null;
  }

  // Resolve the synthesized arm chains against the target rig.
  interface ArmPair {
    bone: THREE.Bone;
    child: THREE.Bone;
    sideX: number;
    segment: "upper" | "fore";
  }
  const armPairs: ArmPair[] = [];
  for (const chain of ARM_CHAINS) {
    const bone = findBone(targetMesh, chain.bone);
    const child = findBone(targetMesh, chain.child);
    if (!bone || !child) continue;
    armPairs.push({ bone, child, sideX: chain.sideX, segment: chain.segment });
  }

  // The left leg's fore/aft direction is the phase reference the arms swing
  // against, so the two can never drift out of sync no matter how the source
  // clip is timed.
  const leftLegPair = pairs.find((pair) => pair.targetBone.name === "L_hleg");

  // Bones that get keyframes: every aimed bone plus every synthesized arm.
  const animatedBones = [
    ...new Set([...pairs.map((pair) => pair.targetBone), ...armPairs.map((entry) => entry.bone)]),
  ];

  targetMesh.skeleton.pose();
  targetRoot.updateMatrixWorld(true);

  const frameCount = Math.max(2, Math.round(clip.duration * fps) + 1);
  const times: number[] = [];
  const values = new Map<THREE.Bone, number[]>();
  for (const bone of animatedBones) values.set(bone, []);

  const mixer = new THREE.AnimationMixer(sourceRoot);
  const action = mixer.clipAction(clip);
  action.play();

  const sourceFrom = new THREE.Vector3();
  const sourceTo = new THREE.Vector3();
  const targetFrom = new THREE.Vector3();
  const targetTo = new THREE.Vector3();
  const desiredDir = new THREE.Vector3();
  const currentDir = new THREE.Vector3();
  const aim = new THREE.Quaternion();
  const worldQuat = new THREE.Quaternion();
  const parentQuat = new THREE.Quaternion();

  for (let frame = 0; frame < frameCount; frame += 1) {
    const time = (frame / (frameCount - 1)) * clip.duration;
    times.push(time);

    mixer.setTime(time);
    sourceRoot.updateMatrixWorld(true);

    // Start each frame from the bind pose so aiming is not accumulated on
    // top of the previous frame's result.
    targetMesh.skeleton.pose();
    targetRoot.updateMatrixWorld(true);

    for (const pair of pairs) {
      sourceFrom.setFromMatrixPosition(pair.sourceBone.matrixWorld);
      sourceTo.setFromMatrixPosition(pair.sourceChild.matrixWorld);
      desiredDir.subVectors(sourceTo, sourceFrom);
      if (desiredDir.lengthSq() < 1e-12) continue;
      desiredDir.normalize();

      targetFrom.setFromMatrixPosition(pair.targetBone.matrixWorld);
      targetTo.setFromMatrixPosition(pair.targetChild.matrixWorld);
      currentDir.subVectors(targetTo, targetFrom);
      if (currentDir.lengthSq() < 1e-12) continue;
      currentDir.normalize();

      // Minimal-arc rotation taking the target limb onto the source limb,
      // scaled by the chain's weight (slerp from identity), then composed
      // onto the bone's existing world orientation.
      aim.setFromUnitVectors(currentDir, desiredDir);
      if (pair.weight < 1) {
        aim.slerp(IDENTITY, 1 - pair.weight);
      }
      pair.targetBone.getWorldQuaternion(worldQuat);
      worldQuat.premultiply(aim);

      if (pair.targetBone.parent) {
        pair.targetBone.parent.getWorldQuaternion(parentQuat);
        pair.targetBone.quaternion.copy(parentQuat.invert()).multiply(worldQuat);
      } else {
        pair.targetBone.quaternion.copy(worldQuat);
      }

      // Children read this bone's world transform on the next iteration.
      targetRoot.updateMatrixWorld(true);
    }

    // --- synthesized arms, phase-locked to the left leg -----------------
    // The left leg's Z direction leads the cycle; the left arm swings
    // against it (hence the negation), and the right mirrors via sideX.
    let legPhase = 0;
    if (leftLegPair) {
      sourceFrom.setFromMatrixPosition(leftLegPair.sourceBone.matrixWorld);
      sourceTo.setFromMatrixPosition(leftLegPair.sourceChild.matrixWorld);
      legPhase = -desiredDir.subVectors(sourceTo, sourceFrom).normalize().z;
    }
    for (const entry of armPairs) {
      armDirection(desiredDir, entry.sideX, entry.segment, legPhase);

      targetFrom.setFromMatrixPosition(entry.bone.matrixWorld);
      targetTo.setFromMatrixPosition(entry.child.matrixWorld);
      currentDir.subVectors(targetTo, targetFrom);
      if (currentDir.lengthSq() < 1e-12) continue;
      currentDir.normalize();

      aim.setFromUnitVectors(currentDir, desiredDir);
      entry.bone.getWorldQuaternion(worldQuat);
      worldQuat.premultiply(aim);
      if (entry.bone.parent) {
        entry.bone.parent.getWorldQuaternion(parentQuat);
        entry.bone.quaternion.copy(parentQuat.invert()).multiply(worldQuat);
      } else {
        entry.bone.quaternion.copy(worldQuat);
      }
      targetRoot.updateMatrixWorld(true);
    }

    for (const bone of animatedBones) {
      const track = values.get(bone)!;
      track.push(bone.quaternion.x, bone.quaternion.y, bone.quaternion.z, bone.quaternion.w);
    }
  }

  action.stop();
  mixer.uncacheClip(clip);

  targetMesh.skeleton.pose();
  targetRoot.updateMatrixWorld(true);

  // `.bones[NAME]` rather than a bare node name: this binding form resolves
  // through the SkinnedMesh's skeleton, so it works regardless of where the
  // bones sit in the scene graph relative to the mesh (in this Collada rig
  // they are not descendants of it). The playback mixer must therefore be
  // rooted at the SkinnedMesh.
  const tracks: THREE.KeyframeTrack[] = animatedBones.map(
    (bone) => new THREE.QuaternionKeyframeTrack(`.bones[${bone.name}].quaternion`, times, values.get(bone)!),
  );

  return new THREE.AnimationClip(options.name ?? "run", clip.duration, tracks);
}
