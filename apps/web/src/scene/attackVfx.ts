/**
 * Projectile VFX for both fighters -- Feature Brief 02 §5.7.
 *
 * Uses the `Attack` asset (a low-poly fireball: 2 static meshes, no
 * animation, and authored ~800 units across, so it needs normalising hard).
 * The same geometry serves both fighters, tinted warm for Naruto and cool
 * for Obito, per the brief.
 *
 * **Pooled and pre-warmed.** The brief is explicit that instantiating VFX
 * mid-combat drops frames on the first hit, and cloning a multi-mesh FBX is
 * exactly the kind of hitch that would land on the most important frame of
 * the fight. Every instance is built up front and reused.
 */

import * as THREE from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";

const MODEL_URL = "/models/attack/fireball.fbx";
/** Diameter in world units after normalising the oversized source. */
const PROJECTILE_SIZE = 0.7;
const POOL_SIZE = 6;

export type VfxOwner = "naruto" | "obito";

const TINTS: Record<VfxOwner, { color: number; emissive: number }> = {
  // Warm chakra orange for Naruto, matching the existing accent.
  naruto: { color: 0xff8a3d, emissive: 0xff5a10 },
  // Cool purple/dark red for Obito.
  obito: { color: 0x7a3b8f, emissive: 0x4a1030 },
};

interface ActiveShot {
  mesh: THREE.Object3D;
  from: THREE.Vector3;
  to: THREE.Vector3;
  elapsed: number;
  duration: number;
  /** Feature Brief 04: the special attack, which arcs and spins harder and
   * can be deflected mid-flight by a successful counter. */
  special?: boolean;
}

export class AttackVfx {
  private readonly pools: Record<VfxOwner, THREE.Object3D[]> = { naruto: [], obito: [] };
  private readonly active: ActiveShot[] = [];
  private readonly disposables: { geometries: THREE.BufferGeometry[]; materials: THREE.Material[] } = {
    geometries: [],
    materials: [],
  };
  private loaded = false;
  /** Feature Brief 04: reserved instance + tint for the special attack. */
  private specialMesh: THREE.Group | null = null;
  private specialMaterial: THREE.MeshBasicMaterial | null = null;

  static async load(scene: THREE.Scene): Promise<AttackVfx> {
    const vfx = new AttackVfx();
    await vfx.build(scene);
    return vfx;
  }

  private async build(scene: THREE.Scene): Promise<void> {
    const source = await new FBXLoader().loadAsync(MODEL_URL);

    // Collect the source geometry once, normalised to unit size and centred,
    // so pooled instances are cheap Mesh wrappers over shared geometry.
    const box = new THREE.Box3().setFromObject(source);
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    const largest = Math.max(size.x, size.y, size.z) || 1;
    const scale = PROJECTILE_SIZE / largest;

    const geometries: THREE.BufferGeometry[] = [];
    source.updateMatrixWorld(true);
    source.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (!mesh.isMesh || !mesh.geometry) return;
      const geometry = mesh.geometry.clone();
      geometry.applyMatrix4(mesh.matrixWorld);
      geometry.translate(-centre.x, -centre.y, -centre.z);
      geometry.scale(scale, scale, scale);
      geometries.push(geometry);
      this.disposables.geometries.push(geometry);
    });

    for (const owner of ["naruto", "obito"] as VfxOwner[]) {
      const tint = TINTS[owner];
      const material = new THREE.MeshBasicMaterial({
        color: tint.color,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      this.disposables.materials.push(material);

      for (let i = 0; i < POOL_SIZE; i += 1) {
        const group = new THREE.Group();
        for (const geometry of geometries) {
          group.add(new THREE.Mesh(geometry, material));
        }
        group.visible = false;
        scene.add(group);
        this.pools[owner].push(group);
      }
    }

    // --- Feature Brief 04: the special attack's own instance -------------
    // Given its own reserved mesh rather than drawn from Obito's pool. §4.5:
    // "a frame hitch when this fires would directly cost the player their
    // run" -- and a pool shared with the ordinary projectiles could in
    // principle be exhausted at exactly the wrong moment. Built here, at
    // load, so nothing is allocated when it fires.
    this.specialMaterial = new THREE.MeshBasicMaterial({
      color: 0xff1f4b,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.disposables.materials.push(this.specialMaterial);
    this.specialMesh = new THREE.Group();
    for (const geometry of geometries) {
      this.specialMesh.add(new THREE.Mesh(geometry, this.specialMaterial));
    }
    this.specialMesh.visible = false;
    scene.add(this.specialMesh);

    this.loaded = true;
  }

  get isLoaded(): boolean {
    return this.loaded;
  }

  /**
   * Launch a shot. Silently no-ops if the pool is exhausted rather than
   * allocating mid-combat -- a missing sixth simultaneous projectile is far
   * less noticeable than a frame hitch.
   */
  fire(owner: VfxOwner, from: THREE.Vector3, to: THREE.Vector3, durationS: number): void {
    const mesh = this.pools[owner].find((candidate) => !candidate.visible);
    if (!mesh) return;
    mesh.visible = true;
    mesh.position.copy(from);
    mesh.scale.setScalar(1);
    this.active.push({ mesh, from: from.clone(), to: to.clone(), elapsed: 0, duration: Math.max(durationS, 1e-3) });
  }

  /**
   * Obito's special attack -- Feature Brief 04 §4.5. Same asset, "scaled up
   * substantially and tinted to Obito's palette", on its reserved instance so
   * it can never be starved by ordinary projectiles in flight.
   */
  fireSpecial(from: THREE.Vector3, to: THREE.Vector3, durationS: number, scale: number): void {
    const mesh = this.specialMesh;
    if (!mesh) return;
    // Re-firing replaces the shot in flight rather than stacking.
    this.dropSpecial();
    mesh.visible = true;
    mesh.position.copy(from);
    mesh.scale.setScalar(scale);
    this.active.push({
      mesh,
      from: from.clone(),
      to: to.clone(),
      elapsed: 0,
      duration: Math.max(durationS, 1e-3),
      special: true,
    });
  }

  /**
   * A successful counter -- §4.3: "the attack visibly misses... the miss must
   * be legible; the player needs to see that they were saved."
   *
   * Rather than deleting the shot (which would read as the game swallowing
   * it), it is re-aimed from wherever it currently is to a point wide of
   * Naruto and past the camera, so it visibly veers off and streaks by.
   */
  deflectSpecial(sideSign: number): void {
    const shot = this.active.find((candidate) => candidate.special);
    if (!shot) return;
    shot.from.copy(shot.mesh.position);
    shot.to.set(
      shot.mesh.position.x + 6 * (sideSign >= 0 ? 1 : -1),
      shot.mesh.position.y + 2.5,
      shot.mesh.position.z + 14, // behind the camera
    );
    shot.elapsed = 0;
    shot.duration = 0.5;
  }

  /** Despawn the special shot specifically (resolution, encounter end). */
  dropSpecial(): void {
    for (let i = this.active.length - 1; i >= 0; i -= 1) {
      const shot = this.active[i]!;
      if (!shot.special) continue;
      shot.mesh.visible = false;
      this.active.splice(i, 1);
    }
  }

  update(dt: number): void {
    for (let i = this.active.length - 1; i >= 0; i -= 1) {
      const shot = this.active[i]!;
      shot.elapsed += dt;
      const t = Math.min(1, shot.elapsed / shot.duration);
      shot.mesh.position.lerpVectors(shot.from, shot.to, t);
      // Slight arc and spin so it reads as thrown rather than slid. The
      // special arcs higher and tumbles faster, to read as heavier.
      const heavy = shot.special === true;
      shot.mesh.position.y += Math.sin(Math.PI * t) * (heavy ? 0.6 : 0.25);
      shot.mesh.rotation.y += dt * (heavy ? 4 : 12);
      shot.mesh.rotation.x += dt * (heavy ? 3 : 7);
      if (t >= 1) {
        shot.mesh.visible = false;
        this.active.splice(i, 1);
      }
    }
  }

  /** Despawn everything -- required on encounter end and on restart (§5.7). */
  clear(): void {
    for (const shot of this.active) {
      shot.mesh.visible = false;
    }
    this.active.length = 0;
  }

  dispose(): void {
    this.clear();
    for (const owner of ["naruto", "obito"] as VfxOwner[]) {
      for (const mesh of this.pools[owner]) mesh.parent?.remove(mesh);
      this.pools[owner].length = 0;
    }
    this.specialMesh?.parent?.remove(this.specialMesh);
    this.specialMesh = null;
    this.specialMaterial = null;
    for (const geometry of this.disposables.geometries) geometry.dispose();
    for (const material of this.disposables.materials) material.dispose();
    this.disposables.geometries.length = 0;
    this.disposables.materials.length = 0;
  }
}
