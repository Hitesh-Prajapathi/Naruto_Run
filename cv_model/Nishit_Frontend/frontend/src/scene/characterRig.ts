/**
 * Bone-name mapping between the two rigs this project has to reconcile.
 *
 * The character model (public/models/naruto/Naruto.dae) is a Wii game-asset
 * rip with a 132-bone skeleton using Japanese-game-rig naming ("harm"/"lleg"
 * = upper/lower limb). It ships **no animation clips at all**.
 *
 * The run cycle (public/models/naruto/run.fbx, from
 * Character_3D_Models/Running_Style) is a different character entirely -- a
 * generic 52-bone humanoid with Mixamo-ish naming -- but it *does* carry a
 * real 0.56s run clip.
 *
 * Neither rig's bone names overlap with the other's at all, which is
 * precisely the silent-failure mode called out in the brief: playing the
 * clip directly binds nothing and leaves the character in its bind T-pose
 * with no error logged. The fix is to retarget the clip through this map
 * (see scene/playerCharacter.ts), which SkeletonUtils.retargetClip uses to
 * resample source bone orientations onto target bones -- correctly handling
 * the two rigs' different rest poses, which a naive rename would not.
 *
 * Verified against both files: every pair below exists on both skeletons.
 *
 * Swapping in a different character means rewriting RUN_CLIP_BONE_MAP's
 * keys (target side) to that rig's names; the values (source side) only
 * change if the run clip itself is replaced.
 */

/** Target (character rig) bone name -> source (run clip rig) bone name. */
export const RUN_CLIP_BONE_MAP: Record<string, string> = {
  hip: "bone_Hips",
  waist: "bone_Spine",
  breast: "bone_Spine1",
  neck: "bone_Neck",
  head: "bone_Head",
  L_hleg: "bone_LeftLegUpper",
  L_lleg: "bone_LeftLeg",
  L_foot: "bone_LeftAnkle",
  L_toe: "bone_LeftToe",
  R_hleg: "bone_RightLegUpper",
  R_lleg: "bone_RightLeg",
  R_foot: "bone_RightAnkle",
  R_toe: "bone_RightToe",
  L_collar: "bone_LeftClav",
  L_harm: "bone_LeftArm",
  L_larm: "bone_LeftForeArm",
  L_hand: "bone_LeftHand",
  R_collar: "bone_RightClav",
  R_harm: "bone_RightArm",
  R_larm: "bone_RightForeArm",
  R_hand: "bone_RightHand",
};

// Note: the limb *chains* that drive retargeting (which bone aims at which
// child) live in scene/retargetRunClip.ts alongside the algorithm that uses
// them; this file is only the name correspondence between the two rigs.
