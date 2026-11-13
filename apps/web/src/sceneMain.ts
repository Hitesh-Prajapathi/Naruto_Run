/**
 * Scene dev harness entry point (scene.html) -- now the actual game view
 * rather than a systems test, per naruto_run_agent_brief.md.
 *
 * Wires: the retargeted-run-clip character, the chase camera, contact
 * shadow, themed obstacles, collision feedback (hit-stop + shake + flash +
 * lives), and the player HUD. The old debug panel is still here behind the
 * `H` key -- the brief explicitly says keep it, don't delete it.
 */

import type * as THREE from "three";
import { assertLevelSegmentValid } from "./game/levelLinter";
import { aabbFromGroundCenter, checkPlayerCollisions } from "./game/collisionSystem";
import { ObstacleSpawner } from "./game/obstacleSpawner";
import { LaneController } from "./game/laneController";
import { HitFeedback, MAX_LIVES } from "./game/hitFeedback";
import { BASE_RUN_SPEED_U_S, PLAYER_COLLISION_SIZE } from "./config/gameConfig";
import { SEGMENT_1 } from "./config/levelDefinition";
import { SceneRoot } from "./scene/sceneRoot";
import { addSkyAndLighting, updateSunTarget } from "./scene/skyAndLighting";
import { ChaseCamera } from "./scene/chaseCamera";
import { CharacterShadow } from "./scene/characterShadow";
import { JungleTrack } from "./scene/jungleTrack";
import { addEnvironmentProps } from "./scene/environmentProps";
import { addForestProps, updateForestProps } from "./scene/forestProps";
import { PlayerCharacter } from "./scene/playerCharacter";
import { GameHud } from "./ui/gameHud";
import { SceneDebugHud } from "./dev/sceneDebugHud";
// --- Feature Brief 02 additions (boss encounter + Game Over) -----------
// Everything new lives behind BossSystem so this file only gains a
// construction, one update call, one key hook, and a few state checks.
import { BossSystem } from "./game/bossSystem";
// --- Feature Brief 05: input abstraction + camera -----------------------
import { InputRouter } from "./input/inputRouter";
import { DEFAULT_BODY_TUNING, loadBodyTuning, type BodyTuning } from "./input/bodyTuning";
import { BodyInputSource, GATE_NONE, type BodyGate } from "./input/bodyInputSource";
import { CameraPoseSource } from "./input/cameraPoseSource";
import type { PoseSample } from "./input/poseAdapter";
import { StartupGate } from "./game/startupGate";
import { StartupGatePanel } from "./ui/startupGatePanel";
import { CameraHud } from "./ui/cameraHud";
import { isCvInputEnabled } from "./config/cvInputConfig";
import { BASE_RUN_SPEED_U_S as RUN_SPEED } from "./config/gameConfig";

// Fail loudly in dev rather than silently playing a pattern that violates
// the CV-driven spacing rules (game_implementation_plan.md §1.1).
assertLevelSegmentValid(SEGMENT_1);

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`missing required element: ${selector}`);
  }
  return element;
}

/**
 * How long until the nearest active obstacle reaches the player, or null if
 * the path ahead is clear. Used only to pick a "clean moment" to start an
 * encounter (Feature Brief 02 §5.1) -- it reads the spawner, never mutates
 * it. Obstacles already behind the player are ignored.
 */
function secondsToNearestObstacle(spawner: ObstacleSpawner): number | null {
  let nearest: number | null = null;
  for (const obstacle of spawner.collidables) {
    // maxZ is the near edge; positive means it has passed the player.
    const distance = -obstacle.aabb.maxZ;
    if (distance < 0) continue;
    const seconds = distance / RUN_SPEED;
    if (nearest === null || seconds < nearest) nearest = seconds;
  }
  return nearest;
}

async function main(): Promise<void> {
  const canvas = requireElement<HTMLCanvasElement>("#scene-canvas");
  const debugContainer = requireElement<HTMLDivElement>("#scene-debug-hud");
  const gameHudContainer = requireElement<HTMLDivElement>("#game-hud");
  const loadingContainer = requireElement<HTMLDivElement>("#scene-loading");

  const sceneRoot = new SceneRoot(canvas);
  const lighting = addSkyAndLighting(sceneRoot.scene);
  const chaseCamera = new ChaseCamera(sceneRoot.camera);
  chaseCamera.snapTo(0);

  const track = new JungleTrack(sceneRoot.scene);
  addEnvironmentProps(track);
  // Fire-and-forget: real jungle set-dressing (see forestProps.ts), loaded
  // in the background rather than gating the loading screen on it.
  let forestProps: THREE.Group[] = [];
  addForestProps(sceneRoot.scene)
    .then((placed) => {
      forestProps = placed;
    })
    .catch((error: unknown) => {
      console.error("forest set-dressing failed to load:", error);
    });

  const spawner = new ObstacleSpawner(sceneRoot.scene, SEGMENT_1);
  const controller = new LaneController();
  // Feature Brief 05 §5.4. Keyboard is routed through this from the start,
  // before any camera exists, so the "game plays exactly as before"
  // checkpoint is verifiable on its own.
  const input = new InputRouter();
  const hits = new HitFeedback();
  const shadow = new CharacterShadow(sceneRoot.scene);
  const gameHud = new GameHud(gameHudContainer);
  const debugHud = new SceneDebugHud(debugContainer);
  debugContainer.hidden = true; // brief §I: dev panel behind a toggle

  // Render the environment immediately rather than blocking on the character
  // load -- the loading overlay covers the canvas until the model attaches.
  sceneRoot.start();

  const character = await PlayerCharacter.load();
  sceneRoot.scene.add(character.root);
  loadingContainer.hidden = true;
  if (!character.isAnimating) {
    // Loud, because a silently-frozen character in the bind T-pose is
    // exactly the bug this whole pass exists to fix.
    console.error("character loaded but no run clip is playing -- check the retarget bone map");
  }

  // Minimal hook for the end-to-end animation check (tests/e2e/scene.spec.ts).
  // Read-only: nothing in the game reads back from it.
  (window as unknown as { __narutoDebug?: unknown }).__narutoDebug = {
    isAnimating: () => character.isAnimating,
    clipTime: () => character.clipTime,
  };

  // Feature Brief 02 test surface (tests/e2e/boss.spec.ts). Read-only apart
  // from the two force* helpers, which let a test reach an encounter or a
  // loss without running out hundreds of metres of track first.
  (window as unknown as { __bossDebug?: unknown }).__bossDebug = {
    state: () => boss.debugState,
    assetsReady: () => boss.assetsReady,
    encountersWon: () => boss.encountersWonCount,
    isGameOverVisible: () => boss.isGameOverVisible,
    narutoHp: () => boss.debugNarutoHp,
    obitoHp: () => boss.debugObitoHp,
    distance: () => distanceMetres,
    // Feature Brief 04 (tests/e2e/specialAttack.spec.ts).
    specialState: () => boss.debugSpecialState,
    specialRemaining: () => boss.debugSpecialRemainingS,
    laneInputAllowed: () => boss.debugLaneInputAllowed,
    worldTimeScale: () => boss.debugWorldTimeScale,
    harePerformed: () => boss.debugHarePerformed,
    playerLane: () => controller.currentLane,
    forceSpecialAttack: () => boss.forceSpecialAttack(),
    // Brief 04 §6 asks for 60fps sustained through the warning window, so
    // the smoothed frame rate has to be observable from a test.
    fps: () => fps,
    // Restart without reloading the page. Obito's model is ~24 MB, so a
    // test file that navigates per test pays for it every time and starves
    // the machine; this lets one page serve a whole spec file.
    restartRun: () => boss.restart(),
    forceGameOver: (cause: "obstacles" | "obito") => boss.forceGameOver(cause),
    forceEncounter: () =>
      boss.forceEncounter({
        distanceM: distanceMetres,
        narutoLane: controller.currentLane,
        narutoX: controller.currentX,
        narutoFeetHeight: controller.feetHeight,
        isJumping: controller.jumpPhase !== "grounded",
        isChangingLane: controller.isTransitioningLane,
        secondsToNearestObstacle: null,
        livesRemaining: hits.remainingLives,
        camera: sceneRoot.camera,
      }),
  };

  /**
   * Feature Brief 05 test surface. §9.1's whole point is that pose events
   * can be injected instead of coming from a camera, so the gate, the
   * detectors and the game's response are all testable in CI. `injectSample`
   * is the same entry point the live camera uses -- it does not bypass any
   * of the processing, it only replaces the producer.
   */
  (window as unknown as { __cvDebug?: unknown }).__cvDebug = {
    enabled: () => cvEnabled,
    gateState: () => startupGate.current,
    gateBlocking: () => startupGate.isBlocking,
    trackingStatus: () => bodyInput.trackingStatus,
    cameraStatus: () => cameraSource?.currentStatus ?? "idle",
    baseline: () => bodyInput.currentBaseline,
    latency: () => cameraSource?.latencyBreakdown ?? null,
    previewVisible: () => cameraHud.isPreviewVisible,
    laneEvents: () => bodyInput.debug.laneEvents,
    jumpEvents: () => bodyInput.debug.jumpEvents,
    /** Push a synthetic pose frame, exactly as the camera would. */
    injectSample: (sample: PoseSample) => {
      latestPose = sample;
    },
    /** Pretend the camera came up, so the gate can be driven without one. */
    simulateCameraReady: () => {
      permissionRequested = true;
      simulatedStatus = "streaming";
      startupGate.requestPermission();
    },
    simulateCameraStatus: (status: string) => {
      permissionRequested = true;
      simulatedStatus = status as typeof simulatedStatus;
      startupGate.requestPermission();
    },
    chooseKeyboard: () => {
      cameraSource?.stop();
      cameraSource = null;
      startupGate.chooseKeyboard();
      gatePanel.hide();
    },
    recalibrate: () => startupGate.recalibrate(),
  };

  let overlappingObstacleIds = new Set<number>();
  let totalHits = 0;
  let distanceMetres = 0;
  let fps = 0;
  let hudFrameCounter = 0;

  // --- Feature Brief 02: boss + Game Over -----------------------------
  const boss = new BossSystem({
    scene: sceneRoot.scene,
    gameOverRoot: requireElement<HTMLDivElement>("#game-over"),
    combatHudRoot: requireElement<HTMLDivElement>("#combat-hud"),
    specialOverlayRoot: requireElement<HTMLDivElement>("#special-overlay"),
    // Try Again: reset every piece of run state this file owns. Explicit
    // rather than relying on GC, per brief §4.5's warning about leaks.
    onRestart: () => {
      spawner.reset();
      hits.reset();
      controller.reset();
      // An intent queued on the frame the player died must not apply to the
      // new run (Brief 05 §5.4; same class of leak as Brief 02 §4.5).
      input.clear();
      overlappingObstacleIds = new Set<number>();
      totalHits = 0;
      distanceMetres = 0;
      character.setAnimationState("run");
      character.setCrouch(0);
      chaseCamera.clearFramingOverride();
      chaseCamera.snapTo(0);
    },
  });
  // Encounter assets load in the background; running is never blocked on
  // them, and an encounter simply cannot start until they are ready.
  boss.loadAssets().catch((error: unknown) => {
    console.error("boss assets failed to load:", error);
  });

  // --- Feature Brief 05: camera input ---------------------------------
  // Everything below is behind `isCvInputEnabled()`. With it off, no camera
  // is requested, no socket opened, and no new element is shown (§0.2).
  const cvEnabled = isCvInputEnabled();
  let bodyTuning: BodyTuning = DEFAULT_BODY_TUNING;
  const bodyInput = new BodyInputSource(bodyTuning);
  const startupGate = new StartupGate(bodyTuning);
  const gatePanel = new StartupGatePanel(requireElement<HTMLDivElement>("#startup-gate"));
  const cameraHud = new CameraHud(requireElement<HTMLDivElement>("#camera-hud"));
  let cameraSource: CameraPoseSource | null = null;
  let latestPose: PoseSample | null = null;
  let lastPushedSampleMs = -1;
  let permissionRequested = false;
  let calibrating = false;
  /** Overrides the camera's real status, for the fixture-driven tests. */
  let simulatedStatus: import("./input/cameraPoseSource").CameraSourceStatus | null = null;

  if (cvEnabled) {
    cameraHud.activate();
    // Tuning is fetched, not bundled, so thresholds change without a rebuild.
    loadBodyTuning()
      .then((tuning) => {
        bodyTuning = tuning;
        bodyInput.setTuning(tuning);
        startupGate.setTuning(tuning);
      })
      .catch(() => undefined);

    cameraSource = new CameraPoseSource({
      video: cameraHud.video,
      onSample: (sample) => {
        latestPose = sample;
      },
    });

    gatePanel.setOnAction(() => {
      permissionRequested = true;
      startupGate.requestPermission();
      void cameraSource?.start();
    });
    gatePanel.setOnKeyboard(() => {
      // §4.1: never dead-end the player. The camera is abandoned entirely.
      cameraSource?.stop();
      cameraSource = null;
      startupGate.chooseKeyboard();
      gatePanel.hide();
    });
  } else {
    // Flag off: the gate never blocks and the game is the approved build.
    startupGate.forceRunning();
  }

  window.addEventListener("keydown", (event) => {
    if (event.repeat) {
      return; // one physical press = one lane step, not auto-repeat spam
    }
    // Feature Brief 02: give the boss system first refusal (seal keys 1/2/3,
    // intro skip), then gate lane/jump on what the current state allows.
    if (boss.handleKey(event.code)) {
      return;
    }
    // boss.capabilities, not boss.state.capabilities: the resolved view also
    // accounts for Feature Brief 04's warning-window lane suppression, which
    // the raw state table knows nothing about.
    const caps = boss.capabilities;

    // --- Feature Brief 05 §5.4: the input abstraction layer -------------
    // The key handler no longer touches the LaneController directly. It
    // states an *intent*; the frame loop applies it. Camera input pushes the
    // identical intents through the identical path, which is the brief's own
    // test for whether the abstraction sits in the right place (T-16).
    switch (event.code) {
      case "ArrowLeft":
      case "KeyA":
        if (caps.laneInput) input.laneLeft("keyboard");
        break;
      case "ArrowRight":
      case "KeyD":
        if (caps.laneInput) input.laneRight("keyboard");
        break;
      case "Space":
      case "ArrowUp":
      case "KeyW":
        event.preventDefault(); // stop the page from scrolling on Space
        if (caps.jumpInput) input.jump("keyboard");
        break;
      // `H` is the Hare counter during an encounter (Feature Brief 04 §4.6)
      // and the dev-panel toggle everywhere else -- boss.handleKey above
      // gets first refusal and only claims it in combat. Backquote is an
      // unconditional alias so the panel is still reachable mid-fight.
      case "KeyH":
      case "Backquote":
        debugContainer.hidden = !debugContainer.hidden;
        break;
      // Feature Brief 05 §4.4: a manual recalibrate, because players drift
      // and should not have to restart to fix it.
      case "KeyR":
        if (cvEnabled) {
          // Also tells the service to re-detect, so R is how you hand the
          // camera to a different person -- the pose model tracks one player
          // and will otherwise stay locked onto the previous one.
          cameraSource?.requestRedetect();
          startupGate.recalibrate();
        }
        break;
      // §7.1: the preview is toggleable.
      case "KeyC":
        if (cvEnabled) cameraHud.togglePreview();
        break;
      default:
        break;
    }
  });

  sceneRoot.onFrame((rawDt, _elapsed, realDt) => {
    fps = rawDt > 0 ? fps * 0.9 + (1 / rawDt) * 0.1 : fps;

    // Hit-stop freezes the world (dt -> 0) without freezing the feedback
    // timers themselves, which is why HitFeedback is updated on raw dt.
    hits.update(rawDt);

    // --- Feature Brief 02: state machine gate --------------------------
    // Ask the boss system what this frame is allowed to do. With both its
    // feature flags off it always answers "normal running", so the approved
    // game below is unchanged.
    const bossFrame = boss.update(rawDt, {
      distanceM: distanceMetres,
      narutoLane: controller.currentLane,
      narutoX: controller.currentX,
      narutoFeetHeight: controller.feetHeight,
      isJumping: controller.jumpPhase !== "grounded",
      isChangingLane: controller.isTransitioningLane,
      secondsToNearestObstacle: secondsToNearestObstacle(spawner),
      livesRemaining: hits.remainingLives,
      camera: sceneRoot.camera,
    // realDt is the unclamped wall-clock delta; only Feature Brief 04's
    // warning countdown uses it, so "2.5s" stays 2.5s even below 15fps.
    }, realDt);
    // Feature Brief 05 §4: the run does not begin until the gate opens.
    // Expressed the same way the boss encounter suspends the world -- a
    // factor the existing speed is multiplied by, and capabilities that are
    // simply false -- rather than a second freeze mechanism.
    const gateBlocking = cvEnabled && startupGate.isBlocking;
    const baseCaps = bossFrame.capabilities;
    const caps = gateBlocking
      ? { ...baseCaps, runningGameplay: false, obstacleSpawn: false, laneInput: false, jumpInput: false }
      : baseCaps;
    if (bossFrame.cameraShake > 0) chaseCamera.shake(bossFrame.cameraShake);
    boss.applyCameraFraming(chaseCamera);

    // --- Feature Brief 05: camera input ---------------------------------
    if (cvEnabled) {
      const nowMs = Date.now();
      cameraSource?.tick(nowMs, bodyTuning.staleEventMs);

      const gateView = startupGate.advance(rawDt * 1000, {
        status: simulatedStatus ?? cameraSource?.currentStatus ?? "idle",
        sample: latestPose,
        permissionRequested,
      });
      gatePanel.render(gateView);

      // Calibration is owned by the gate's timing but captured by the input
      // source, so the two are joined here rather than either owning both.
      if (gateView.state === "CALIBRATION" && !calibrating) {
        calibrating = true;
        bodyInput.beginCalibration();
      } else if (gateView.state !== "CALIBRATION" && calibrating) {
        calibrating = false;
        bodyInput.finishCalibration();
      }

      // §6: inference always runs; only *consumption* is gated. Feeding the
      // sample even while blocked keeps the detectors' history warm.
      if (latestPose) {
        const gate: BodyGate = startupGate.cameraActive
          ? { laneInput: caps.laneInput, jumpInput: caps.jumpInput }
          : GATE_NONE;
        // Only on a genuinely new observation. Re-feeding the same sample
        // every rendered frame made the detectors count render frames
        // instead of samples -- see BodyInputSource.tick.
        if (latestPose.timestampMs !== lastPushedSampleMs) {
          lastPushedSampleMs = latestPose.timestampMs;
          bodyInput.push(latestPose, gate, input, nowMs);
          cameraSource?.noteConsumed(nowMs);
        } else {
          bodyInput.tick(nowMs);
        }
      }
      cameraHud.render(latestPose, bodyInput.trackingStatus);
    }

    // --- Feature Brief 05 §5.4 -----------------------------------------
    // The one place input is applied. Whatever produced the intent -- a key
    // or a camera -- it lands here, so a camera-driven lane change is
    // indistinguishable from a keyboard one downstream. Drained before the
    // controller updates so an intent takes effect on the frame it arrived.
    for (const intent of input.drain()) {
      if (intent.kind === "lane" && caps.laneInput) {
        controller.moveLane(intent.direction);
      } else if (intent.kind === "jump" && caps.jumpInput) {
        controller.requestJump();
      }
    }

    const dt = rawDt * hits.timeScale;
    // Suspend-and-restore (brief §0 rule 3): the world's own speed is never
    // modified, only scaled by a factor that is exactly 1 outside encounters.
    const scrolledDt = dt * bossFrame.scrollFactor * (gateBlocking ? 0 : 1);

    const deltaZ = BASE_RUN_SPEED_U_S * scrolledDt;
    distanceMetres += deltaZ;

    track.update(deltaZ);
    updateForestProps(forestProps, deltaZ);
    // Obstacles keep moving off-screen while the world still has momentum,
    // but no new ones spawn once spawning is suspended.
    spawner.update(caps.obstacleSpawn ? scrolledDt : 0, deltaZ);
    controller.update(dt);

    character.setAnimationState(controller.isAirborne ? "jump" : "run");
    character.setCrouch(controller.crouchAmount);
    character.update(dt);
    character.root.position.set(controller.currentX, controller.feetHeight, 0);
    // A roll about +Z tilts the body toward -X, so negate to make a positive
    // leanRad (defined as "toward +X") actually lean right.
    character.root.rotation.z = -controller.leanRad;

    shadow.update(controller.currentX, 0, controller.feetHeight);
    updateSunTarget(lighting, controller.currentX);
    chaseCamera.update(rawDt, controller.currentX);

    // Loop the segment once fully drained, reusing the same pools -- see
    // obstacleSpawner.ts's reset() doc comment.
    if (caps.obstacleSpawn && spawner.isScheduleFinished && spawner.activeCount === 0) {
      spawner.reset();
    }

    // Obstacle collision is part of the running gameplay path, so it is
    // gated wholesale -- an obstacle drifting past during a cutscene must
    // not be able to hit the player (brief §5.1).
    if (caps.runningGameplay) {
      const playerAabb = aabbFromGroundCenter(
        { x: controller.currentX, y: controller.feetHeight, z: 0 },
        PLAYER_COLLISION_SIZE,
      );
      const contacts = checkPlayerCollisions(playerAabb, spawner.collidables);
      const currentIds = new Set(contacts.map((hit) => hit.obstacleId));
      for (const id of currentIds) {
        if (!overlappingObstacleIds.has(id)) {
          totalHits += 1; // edge-triggered: entering a collision, not every overlapping frame
          boss.notifyObstacleCleared();
          if (hits.registerHit()) {
            chaseCamera.shake();
            gameHud.flash();
          }
        }
      }
      overlappingObstacleIds = currentIds;
    }

    gameHud.render(
      {
        distanceMetres,
        speed: 1,
        lives: hits.remainingLives,
        maxLives: MAX_LIVES,
        inputStatus: "keyboard",
      },
      rawDt,
    );

    hudFrameCounter += 1;
    if (!debugContainer.hidden && hudFrameCounter % 6 === 0) {
      debugHud.render({
        fps,
        obstacleActive: spawner.activeCount,
        obstacleConstructed: spawner.totalConstructed,
        lane: controller.currentLane,
        airborne: controller.isAirborne,
        totalHits,
        segmentElapsedSeconds: spawner.elapsedSeconds,
        segmentDurationSeconds: SEGMENT_1.durationSeconds,
        // Feature Brief 05 §7.3: extends the existing panel, same toggle.
        body: cvEnabled ? bodyInput.debug : undefined,
        bodyTuning: cvEnabled ? bodyTuning : undefined,
        latencyMs: cvEnabled ? cameraSource?.latencyBreakdown.totalMs : undefined,
      });
    }
  });

  window.addEventListener("resize", () => sceneRoot.resize());
}

main().catch((error: unknown) => {
  console.error("scene dev harness failed to start:", error);
  const loadingContainer = document.querySelector<HTMLDivElement>("#scene-loading");
  if (loadingContainer) {
    loadingContainer.textContent = "Failed to load — see console for details.";
    loadingContainer.hidden = false;
  }
});
