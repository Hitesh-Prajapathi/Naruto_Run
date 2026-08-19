/**
 * Single entry point for everything Feature Brief 02 adds: the game state
 * machine, the Obito encounters, and the Game Over screen.
 *
 * This exists so `sceneMain.ts` -- the approved running game -- only gains a
 * construction, one `update()` call, one key handler, and a few state
 * checks, rather than being restructured (brief §0 rules 1 and 2). Every
 * new system lives behind this facade.
 *
 * The suspend-and-restore contract (brief §0 rule 3) is expressed as
 * `scrollFactor` and `capabilities`: the host multiplies its own, unchanged
 * scroll speed by the factor and asks the capabilities whether to run its
 * existing gameplay. When no encounter is active the factor is exactly 1 and
 * every capability is true, so the running game is bit-for-bit itself.
 *
 * There are deliberately **no timers here** (no setTimeout/setInterval).
 * Everything advances on the frame delta, so "an orphaned tween from the
 * previous run firing during the new one" -- the restart bug the brief warns
 * about -- cannot happen: resetting drops the state that drives them.
 */

import * as THREE from "three";
import {
  COMBAT_CAMERA_LOOK_AHEAD_Z,
  COMBAT_CAMERA_LOOK_AT_HEIGHT,
  COMBAT_CAMERA_POSITION,
  COMBAT_CAMERA_SHAKE,
  ENABLE_BOSS,
  ENABLE_GAME_OVER,
  ENCOUNTER_OBSTACLE_CLEARANCE_S,
  GAME_OVER_DELAY_S,
  OBITO_COMBAT_Z,
  SEAL_POOL,
  TOTAL_ENCOUNTERS,
  type SealName,
} from "../config/bossConfig";
import { LANE_X, PLAYER_HEIGHT, type LaneIndex } from "../config/gameConfig";
import { GameStateMachine, type StateCapabilities } from "./gameState";
import { EncounterScheduler } from "./encounterScheduler";
import { EncounterDirector, type EncounterHooks } from "./encounterDirector";
import { ObitoCharacter } from "../scene/obitoCharacter";
import { AttackVfx } from "../scene/attackVfx";
import { HealthBar, NARUTO_BAR_STYLE, OBITO_BAR_STYLE } from "../scene/healthBar";
import { DamageNumbers, TelegraphMarker } from "../scene/combatMarkers";
import { CombatHud } from "../ui/combatHud";
import { GameOverPanel, type DeathCause } from "../ui/gameOverPanel";
import { NARUTO_MAX_HP, OBITO_MAX_HP } from "../config/bossConfig";
// --- Feature Brief 04 additions (special attack + Hare counter) --------
import {
  COUNTER_KEY_CODE,
  ENABLE_SPECIAL_ATTACK,
  SPECIAL_CAMERA_SHAKE,
  SPECIAL_COUNTER_SHAKE,
  SPECIAL_HIT_STOP_S,
  SPECIAL_VFX_SCALE,
  SPECIAL_VFX_TRAVEL_S,
} from "../config/specialAttackConfig";
import { SpecialWarningMarker } from "../scene/specialWarningMarker";
import { SpecialAttackOverlay } from "../ui/specialAttackOverlay";

/** What the host tells the boss system about the current frame. */
export interface BossFrameContext {
  distanceM: number;
  narutoLane: LaneIndex;
  narutoX: number;
  narutoFeetHeight: number;
  isJumping: boolean;
  isChangingLane: boolean;
  /** Seconds until the nearest obstacle reaches the player, or null. */
  secondsToNearestObstacle: number | null;
  /** Obstacle lives left; 0 means the run is over. */
  livesRemaining: number;
  camera: THREE.Camera;
}

/** What the host should do this frame. */
export interface BossFrameResult {
  /** Multiplier for the host's own, unchanged world-scroll speed. */
  scrollFactor: number;
  capabilities: StateCapabilities;
  /** Extra hit-stop requested by combat, in seconds (0 most frames). */
  hitStopS: number;
  /** Camera shake magnitude requested this frame (0 most frames). */
  cameraShake: number;
}

export interface BossSystemDeps {
  scene: THREE.Scene;
  gameOverRoot: HTMLElement;
  combatHudRoot: HTMLElement;
  /** Feature Brief 04's full-screen warning layer. */
  specialOverlayRoot: HTMLElement;
  /** Called when the player asks for a fresh run; the host resets itself. */
  onRestart: () => void;
}

const RUNNING_RESULT: BossFrameResult = {
  scrollFactor: 1,
  capabilities: {
    runningGameplay: true,
    worldScroll: "on",
    obstacleSpawn: true,
    laneInput: true,
    jumpInput: true,
    gestureInput: false,
  },
  hitStopS: 0,
  cameraShake: 0,
};

export class BossSystem {
  readonly state = new GameStateMachine();
  private readonly scheduler = new EncounterScheduler();
  private director: EncounterDirector | null = null;

  private obito: ObitoCharacter | null = null;
  private vfx: AttackVfx | null = null;
  private narutoBar: HealthBar | null = null;
  private obitoBar: HealthBar | null = null;
  private telegraph: TelegraphMarker | null = null;
  private damageNumbers: DamageNumbers | null = null;
  private readonly combatHud: CombatHud;
  private readonly gameOverPanel: GameOverPanel;
  // --- Feature Brief 04 -------------------------------------------------
  private readonly specialOverlay: SpecialAttackOverlay;
  private specialMarker: SpecialWarningMarker | null = null;
  /**
   * §2.2/§5: whether the player has ever performed `hare` this run. Lives
   * here rather than on the encounter because it must persist between the
   * two encounters and reset only on Try Again.
   */
  private hareEverPerformed = false;
  /** Hit-stop/shake asked for between frames (i.e. from a key handler). */
  private carriedHitStop = 0;
  private carriedShake = 0;
  /** Naruto's X last frame, so hooks firing outside update() can aim. */
  private lastNarutoX = 0;
  private lastNarutoHp = NARUTO_MAX_HP;
  private lastObitoHp = OBITO_MAX_HP;

  private encountersWon = 0;
  private obstaclesCleared = 0;
  private pendingHitStop = 0;
  private pendingShake = 0;
  /** Counts down between the killing blow and the panel (brief §4.2). */
  private gameOverDelay = 0;
  private pendingCause: DeathCause | null = null;
  private readonly scratch = new THREE.Vector3();

  constructor(private readonly deps: BossSystemDeps) {
    this.combatHud = new CombatHud(deps.combatHudRoot);
    this.gameOverPanel = new GameOverPanel(deps.gameOverRoot);
    this.gameOverPanel.setOnTryAgain(() => this.restart());
    this.specialOverlay = new SpecialAttackOverlay(deps.specialOverlayRoot);
  }

  /** Load the encounter assets in the background; the running game does not
   * wait on them, and an encounter simply won't start until they're ready. */
  async loadAssets(): Promise<void> {
    if (!ENABLE_BOSS) return;
    const [obito, vfx] = await Promise.all([
      ObitoCharacter.load(),
      AttackVfx.load(this.deps.scene),
    ]);
    this.obito = obito;
    this.vfx = vfx;
    this.deps.scene.add(obito.root);
    this.narutoBar = new HealthBar(this.deps.scene, NARUTO_BAR_STYLE, NARUTO_MAX_HP);
    this.obitoBar = new HealthBar(this.deps.scene, OBITO_BAR_STYLE, OBITO_MAX_HP);
    this.telegraph = new TelegraphMarker(this.deps.scene);
    this.damageNumbers = new DamageNumbers(this.deps.scene);
    if (ENABLE_SPECIAL_ATTACK) {
      this.specialMarker = new SpecialWarningMarker(this.deps.scene);
    }
  }

  /**
   * What is allowed right now, including Feature Brief 04's warning-window
   * suppression.
   *
   * The host's *key handler* must read this rather than `state.capabilities`
   * directly: the raw table knows nothing about the special attack, so
   * asking it would let a lane change through during the window even though
   * the frame loop is correctly suppressing it. That bug was caught by the
   * end-to-end lane-suppression test, not by any unit test -- the two
   * capability sources only diverge once a real key is pressed.
   */
  get capabilities(): StateCapabilities {
    return this.resolveCapabilities();
  }

  get isGameOverVisible(): boolean {
    return this.gameOverPanel.isVisible;
  }

  get encountersWonCount(): number {
    return this.encountersWon;
  }

  /** Route a key. Returns true when the boss system consumed it. */
  handleKey(code: string): boolean {
    if (this.state.state === "GAME_OVER") {
      // Enter/Space are handled by the panel itself.
      return false;
    }
    if (!this.director) return false;

    if (code === "KeyK") {
      this.director.skipIntro();
      return true;
    }

    // --- Feature Brief 04 §2.1: context-gated priority for `hare` --------
    // One input, three documented meanings. The decision is made in exactly
    // one place, by asking whether the warning window is open:
    //
    //   window open  -> the counter, and nothing else. Consumed entirely.
    //   combat, open -> free practice today; `ikazuchi` once Brief 03's
    //                   jutsu catalog exists. Note it is deliberately NOT
    //                   prompt-gated (§2.2): the sign that saves the run has
    //                   to be muscle memory, so the player can spam it.
    //   otherwise    -> ignored, and falls through to the dev-panel toggle.
    if (ENABLE_SPECIAL_ATTACK && code === COUNTER_KEY_CODE) {
      if (this.director.submitCounter()) {
        return true; // counter accepted; nothing else may fire
      }
      if (this.state.state === "BOSS_COMBAT") {
        this.notifyHarePerformed();
        return true;
      }
      return false;
    }

    // Keyboard fallback for hand seals (brief 02 §6) -- combat must stay
    // testable without a camera, permanently, not just during development.
    const index = { Digit1: 0, Digit2: 1, Digit3: 2 }[code];
    if (index !== undefined && this.state.capabilities.gestureInput) {
      const seal = SEAL_POOL[index];
      if (seal) this.director.submitSeal(seal as SealName);
      return true;
    }
    return false;
  }

  /** Called by the host when an obstacle hit lands, for the stats line. */
  notifyObstacleCleared(): void {
    this.obstaclesCleared += 1;
  }

  /**
   * The player performed `hare` outside the warning window -- Brief 04 §2.2.
   *
   * This is the whole of "guarantee the player has performed `hare` at least
   * once before the first special attack fires". Until it has been called,
   * the special attack sits in `awaiting_practice` and cannot kill anyone.
   *
   * Public so the CV input layer can call it directly when it lands, rather
   * than having to synthesise a key event.
   */
  notifyHarePerformed(): void {
    if (this.hareEverPerformed) return;
    this.hareEverPerformed = true;
    this.specialOverlay.hidePracticePrompt();
    this.combatHud.showBanner("HARE READY");
  }

  // --- test/debug surface ------------------------------------------------
  // Read-only apart from the two `force*` helpers, which exist so the
  // end-to-end tests can reach an encounter or a loss without waiting out
  // hundreds of metres of real running. Nothing in the game reads these.

  /**
   * Whether the encounter assets have finished loading. Obito's model is
   * ~24 MB, so on a cold load a player can plausibly reach the first
   * milestone before it arrives. `maybeStartEncounter` deliberately returns
   * *before* consulting the scheduler in that case, so the trigger is not
   * consumed -- the encounter is postponed until the assets land rather
   * than being silently skipped for the rest of the run.
   */
  get assetsReady(): boolean {
    return this.obito !== null && this.vfx !== null;
  }

  get debugState(): string {
    return this.state.state;
  }

  /**
   * Last known HP, held after the encounter ends rather than snapping back
   * to full.
   *
   * These used to fall back to the max whenever `director` was null, which
   * meant a fatal blow read as "100 HP" the moment the encounter tore down
   * -- so the Game Over screen for a death always reported full health. That
   * is exactly backwards for the one case anyone would be inspecting.
   */
  get debugNarutoHp(): number {
    return this.director?.combat.naruto.hp ?? this.lastNarutoHp;
  }

  get debugObitoHp(): number {
    return this.director?.combat.obito.hp ?? this.lastObitoHp;
  }

  /** Jump straight to the next encounter, ignoring the distance milestone. */
  forceEncounter(context: BossFrameContext): void {
    if (!ENABLE_BOSS || this.state.state !== "RUNNING") return;
    if (!this.obito || !this.vfx) return;
    this.scheduler.forceNext();
    this.maybeStartEncounter(1, { ...context, distanceM: Number.MAX_SAFE_INTEGER });
  }

  forceGameOver(cause: DeathCause): void {
    this.enterGameOver(cause);
  }

  // --- Feature Brief 04 test/debug surface -------------------------------

  get debugSpecialState(): string {
    return this.director?.special?.state ?? "none";
  }

  get debugSpecialRemainingS(): number {
    return this.director?.special?.remainingS ?? 0;
  }

  get debugLaneInputAllowed(): boolean {
    return this.resolveCapabilities().laneInput;
  }

  get debugWorldTimeScale(): number {
    return this.director?.worldTimeScale ?? 1;
  }

  get debugHarePerformed(): boolean {
    return this.hareEverPerformed;
  }

  /** Drive Obito to the special attack's trigger HP without a real fight. */
  forceSpecialAttack(): void {
    const combat = this.director?.combat;
    if (!combat) return;
    // One hit of the real damage curve (60) leaves him at exactly the 60 HP
    // trigger, which is what a player landing their first jutsu produces.
    while (combat.obito.hp > 60) {
      combat.applyDebugObitoDamage(10);
    }
    // Park the ordinary attack loop so a test can watch the special attack
    // play out without Naruto being killed by the normal duel meanwhile.
    combat.setDebugOrdinaryAttacksPaused(true);
  }

  /**
   * @param rawDt the host's frame delta (clamped by SceneRoot).
   * @param realDt true wall-clock delta. Only the special attack's countdown
   *   uses it -- see EncounterDirector.update.
   */
  update(rawDt: number, context: BossFrameContext, realDt: number = rawDt): BossFrameResult {
    if (!ENABLE_BOSS && !ENABLE_GAME_OVER) {
      return RUNNING_RESULT;
    }

    // Feedback requested *between* frames -- the Hare counter arrives from a
    // key handler, not from update() -- has to survive into this frame.
    // Zeroing unconditionally here would silently swallow the one piece of
    // feedback confirming the player's save landed.
    this.pendingHitStop = this.carriedHitStop;
    this.pendingShake = this.carriedShake;
    this.carriedHitStop = 0;
    this.carriedShake = 0;

    this.updateGameOverDelay(rawDt);
    this.checkObstacleDefeat(context);

    if (this.state.state === "RUNNING" && ENABLE_BOSS) {
      this.maybeStartEncounter(rawDt, context);
    }

    if (this.state.isInEncounter && this.director) {
      this.updateEncounter(rawDt, context, realDt);
    }

    // Brief 04 §4.1: the world runs at 70% through the warning window. The
    // countdown above deliberately does not -- it is fed rawDt -- so "2.5s"
    // stays 2.5 real seconds. Outside a window this factor is exactly 1.
    const worldDt = rawDt * (this.director?.worldTimeScale ?? 1);

    this.telegraph?.update(worldDt);
    this.vfx?.update(worldDt);
    this.damageNumbers?.update(worldDt);
    this.updateHealthBars(rawDt, context);
    this.updateSpecialWarning(rawDt, context);

    return {
      scrollFactor: this.director ? this.director.scrollFactor : this.state.state === "GAME_OVER" ? 0 : 1,
      capabilities: this.resolveCapabilities(),
      hitStopS: this.pendingHitStop,
      cameraShake: this.pendingShake,
    };
  }

  /**
   * Brief 04 §4.2: lane changes are suppressed for the warning window.
   *
   * Layered on top of the state machine's table rather than added to it, so
   * the approved capability table is untouched and the suppression is a
   * derived property of the window being open -- it cannot persist into the
   * next run, because there is no flag to leave set.
   */
  private resolveCapabilities(): StateCapabilities {
    const base = this.state.capabilities;
    if (!this.director?.suppressLaneInput) return base;
    return { ...base, laneInput: false };
  }

  /** Drive the world-space warning above Naruto's head, and the §2.2
   * practice prompt that teaches the sign before it can matter. */
  private updateSpecialWarning(rawDt: number, context: BossFrameContext): void {
    const special = this.director?.special;
    if (!this.specialMarker) return;
    if (!special) {
      this.specialMarker.clear();
      return;
    }

    this.specialMarker.setVisible(special.isWarning);
    if (special.isWarning) {
      this.scratch.set(context.narutoX, context.narutoFeetHeight, 0);
      this.specialMarker.update(rawDt, this.scratch, PLAYER_HEIGHT, special.windowProgress, context.camera);
      this.specialOverlay.setProgress(special.windowProgress);
    }

    // Ask for practice as soon as combat starts, not only once the trigger
    // is met: the prompt is the only thing that teaches this sign exists,
    // and §2.2 is explicit that nobody should meet the mechanic for the
    // first time in the moment it can kill them.
    if (this.state.state === "BOSS_COMBAT" && !this.hareEverPerformed && !special.hasFired) {
      this.specialOverlay.showPracticePrompt();
    }
  }

  /** Blend the *existing* camera toward the duel framing (brief §5.6). */
  applyCameraFraming(camera: { setFramingOverride: (b: number, p: { x: number; y: number; z: number }, z: number, h: number) => void }): void {
    const blend = this.director?.cameraBlend ?? 0;
    camera.setFramingOverride(
      blend,
      COMBAT_CAMERA_POSITION,
      COMBAT_CAMERA_LOOK_AHEAD_Z,
      COMBAT_CAMERA_LOOK_AT_HEIGHT,
    );
  }

  private updateGameOverDelay(rawDt: number): void {
    if (this.gameOverDelay <= 0) return;
    this.gameOverDelay -= rawDt;
    if (this.gameOverDelay > 0 || !this.pendingCause) return;
    const cause = this.pendingCause;
    this.pendingCause = null;
    this.gameOverPanel.show(cause, {
      distanceM: this.lastDistanceM,
      obstaclesCleared: this.obstaclesCleared,
      encountersWon: this.encountersWon,
      totalEncounters: TOTAL_ENCOUNTERS,
    });
  }

  private lastDistanceM = 0;

  private checkObstacleDefeat(context: BossFrameContext): void {
    this.lastDistanceM = context.distanceM;
    this.lastNarutoX = context.narutoX;
    if (!ENABLE_GAME_OVER) return;
    if (context.livesRemaining > 0) return;
    if (this.state.state === "GAME_OVER") return;
    // Brief §2.2: obstacle lives and combat HP stay separate systems; this
    // path only ever fires from the existing life counter.
    this.enterGameOver("obstacles");
  }

  private maybeStartEncounter(rawDt: number, context: BossFrameContext): void {
    if (!this.obito || !this.vfx) return; // assets still loading
    const shouldStart = this.scheduler.update(rawDt, context.distanceM, {
      isJumping: context.isJumping,
      isChangingLane: context.isChangingLane,
      secondsToNearestObstacle: context.secondsToNearestObstacle,
      requiredClearanceS: ENCOUNTER_OBSTACLE_CLEARANCE_S,
    });
    if (!shouldStart) return;

    const index = this.scheduler.encountersTriggered - 1;
    this.director = new EncounterDirector(index, this.buildHooks());
    this.state.transitionTo("BOSS_INTRO");

    // HP resets to full for both fighters each encounter (brief §2.3).
    this.obito.resetForEncounter(1, context.distanceM > 0 ? -70 : -70);
    this.obito.setVisible(true);
    this.narutoBar?.reset();
    this.obitoBar?.reset();
    this.narutoBar?.setVisible(true);
    this.obitoBar?.setVisible(true);
    this.combatHud.setVisible(true);
    this.combatHud.showBanner("OBITO APPEARS");
  }

  private updateEncounter(rawDt: number, context: BossFrameContext, realDt: number): void {
    const director = this.director;
    if (!director) return;

    // rawDt for the director: its own timing, and the special attack's
    // countdown inside it, must run in real seconds (Brief 04 §4.1).
    director.update(rawDt, context.narutoLane, this.hareEverPerformed, realDt);
    this.obito?.setZ(director.obitoZ);
    // ...but Obito is part of the world, so he slows with it.
    this.obito?.update(rawDt * director.worldTimeScale);

    if (director.currentPhase === "combat" && this.state.state === "BOSS_INTRO") {
      this.state.transitionTo("BOSS_COMBAT");
    }

    this.lastNarutoHp = director.combat.naruto.hp;
    this.lastObitoHp = director.combat.obito.hp;
    this.narutoBar?.setHp(director.combat.naruto.hp);
    this.obitoBar?.setHp(director.combat.obito.hp);
    this.telegraph?.set(director.combat.telegraphedLane, director.combat.telegraphProgress);

    this.combatHud.render({
      sequence: director.seals.sequence,
      matchedCount: director.seals.matchedCount,
      windowProgress: director.seals.windowProgress,
      phase: director.seals.currentPhase,
      failureReason: director.failureReason,
    });

    if (!director.isComplete) return;

    if (director.result === "lost") {
      // Brief 04 §2.5: name the special attack specifically. The health bar
      // can read 100/100 at the moment of death, so a generic "Obito
      // defeated you" would read as a bug.
      const cause: DeathCause = director.lostToSpecialAttack ? "obito_special" : "obito";
      this.finishEncounter();
      this.state.transitionTo("GAME_OVER");
      this.enterGameOver(cause, /* alreadyTransitioned */ true);
      return;
    }

    // Victory: hand the world back exactly as it was (brief §5.8).
    this.encountersWon += 1;
    this.finishEncounter();
    this.state.transitionTo("BOSS_DEFEATED");
    this.state.transitionTo("RUNNING");
  }

  private finishEncounter(): void {
    // Brief 04 §5: the encounter ending is an exit path like any other, so
    // it runs the same single teardown. Done *before* the director is
    // dropped, so a window still open when the encounter ends (e.g. the
    // player dies to the special attack) cannot leave its UI on screen.
    this.tearDownSpecial();
    this.director = null;
    this.obito?.setVisible(false);
    this.narutoBar?.setVisible(false);
    this.obitoBar?.setVisible(false);
    this.telegraph?.clear();
    this.vfx?.clear();
    this.damageNumbers?.clear();
    this.combatHud.setVisible(false);
    this.combatHud.hideBanner();
  }

  private updateHealthBars(rawDt: number, context: BossFrameContext): void {
    if (!this.narutoBar || !this.obitoBar || !this.obito) return;
    if (!this.state.isInEncounter) return;

    this.scratch.set(context.narutoX, context.narutoFeetHeight, 0);
    this.narutoBar.update(rawDt, this.scratch, PLAYER_HEIGHT, context.camera);

    this.scratch.copy(this.obito.root.position);
    this.obitoBar.update(rawDt, this.scratch, this.obito.headHeight, context.camera);
  }

  private enterGameOver(cause: DeathCause, alreadyTransitioned = false): void {
    if (!ENABLE_GAME_OVER) return;
    if (!alreadyTransitioned && !this.state.transitionTo("GAME_OVER")) return;
    this.finishEncounter();
    // Brief §4.2: a beat between the killing blow and the panel, so the
    // player actually sees what killed them.
    this.gameOverDelay = GAME_OVER_DELAY_S;
    this.pendingCause = cause;
  }

  private buildHooks(): EncounterHooks {
    return {
      onObitoMoveLane: (lane) => this.obito?.moveToLane(lane),
      onObitoAttackPose: () => this.obito?.playAttack(),
      onObitoHit: () => {
        this.obito?.playHit();
        this.pendingShake = COMBAT_CAMERA_SHAKE;
      },
      onObitoDefeated: (durationS) => this.obito?.playDefeat(durationS),
      onTelegraph: (lane) => this.telegraph?.set(lane, this.director?.combat.telegraphProgress ?? 0),
      onFireVfx: (owner, fromLane, toLane, travelS) => {
        const fromY = owner === "obito" ? this.obito!.headHeight * 0.6 : PLAYER_HEIGHT * 0.6;
        const from = new THREE.Vector3(LANE_X[fromLane], fromY, owner === "obito" ? OBITO_COMBAT_Z : 0);
        const to = new THREE.Vector3(LANE_X[toLane], PLAYER_HEIGHT * 0.55, owner === "obito" ? 0 : OBITO_COMBAT_Z);
        this.vfx?.fire(owner, from, to, travelS);
      },
      onNarutoHit: () => {
        this.pendingShake = COMBAT_CAMERA_SHAKE;
      },
      onDamageNumber: (target, damage, lane) => {
        const z = target === "obito" ? OBITO_COMBAT_Z : 0;
        const height = target === "obito" ? this.obito!.headHeight : PLAYER_HEIGHT;
        this.damageNumbers?.spawn(
          damage,
          new THREE.Vector3(LANE_X[lane], height * 0.9, z),
          target === "obito" ? "#ffd6a5" : "#ff6b6b",
        );
      },
      onHitStop: (seconds) => {
        this.pendingHitStop = Math.max(this.pendingHitStop, seconds);
      },
      onBanner: (text) => this.combatHud.showBanner(text),

      // --- Feature Brief 04 ---------------------------------------------
      onSpecialPracticeRequired: () => {
        this.specialOverlay.showPracticePrompt();
      },
      onSpecialWarning: (durationS) => {
        this.specialOverlay.showWarning();
        this.specialMarker?.setVisible(true);
        // §4.2: "switch the prompt UI entirely to the counter. Do not make
        // them track two prompts at once." The seal sequence is already
        // cancelled, but leaving its panel on screen still reads as a live
        // instruction ("Perform the seals") competing with the warning.
        this.combatHud.setVisible(false);
        // §4.1: Obito holds a visible wind-up for the *whole* window, so the
        // threat is readable from his body and not only from the UI.
        this.obito?.playSpecialWindup(durationS);
        this.combatHud.showBanner("SPECIAL ATTACK");
      },
      onSpecialReleased: () => {
        this.obito?.playAttack();
        // §4.5: the existing Attack asset, scaled up hard and in Obito's
        // palette, on its own reserved instance so it cannot be starved.
        const from = new THREE.Vector3(
          this.obito?.root.position.x ?? 0,
          (this.obito?.headHeight ?? PLAYER_HEIGHT) * 0.6,
          OBITO_COMBAT_Z,
        );
        const to = new THREE.Vector3(this.lastNarutoX, PLAYER_HEIGHT * 0.55, 0);
        this.vfx?.fireSpecial(from, to, SPECIAL_VFX_TRAVEL_S, SPECIAL_VFX_SCALE);
      },
      onSpecialCountered: () => {
        // §4.3: the miss has to be *legible*. The shot is re-aimed wide and
        // streaks past rather than being deleted, which would read as the
        // game swallowing it.
        this.vfx?.deflectSpecial(this.lastNarutoX >= 0 ? 1 : -1);
        // Carried, not pending: this hook runs from the key handler.
        this.carriedShake = SPECIAL_COUNTER_SHAKE;
        this.carriedHitStop = SPECIAL_HIT_STOP_S * 0.5;
        this.specialOverlay.showResult("countered");
        this.specialMarker?.clear();
        this.obito?.endSpecialWindup();
      },
      onSpecialStruck: () => {
        this.pendingShake = SPECIAL_CAMERA_SHAKE;
        this.pendingHitStop = Math.max(this.pendingHitStop, SPECIAL_HIT_STOP_S);
        this.specialOverlay.showResult("struck");
        this.specialMarker?.clear();
        this.obito?.endSpecialWindup();
        this.vfx?.dropSpecial();
      },
      // §5: the single teardown. Every exit path -- success, failure,
      // encounter end and restart -- routes through tearDownSpecial().
      onSpecialTeardown: () => this.tearDownSpecial(),
    };
  }

  /**
   * The one restore path for everything the special attack suspends --
   * Brief 04 §5: "every restore must run through a single teardown function
   * that executes on *every* exit path... do not scatter restores across the
   * success and failure branches."
   *
   * Note what is *not* here: lean suppression and the world time scale. Both
   * are getters derived from the controller's state (see
   * specialAttackController.ts), so they cannot be left set. §5 names those
   * two as the most dangerous leaks; making them underivable from stale
   * state is stronger than remembering to clear them.
   */
  private tearDownSpecial(): void {
    this.specialOverlay.hide();
    this.specialMarker?.clear();
    this.obito?.endSpecialWindup();
    this.vfx?.dropSpecial();
    // Restore the jutsu prompt -- but only if the duel is actually going to
    // continue. On the defeat path the encounter is already over, and
    // showing the prompt here would flash it back on for a quarter second
    // over the death beat before finishEncounter() hides it again.
    if (this.director && this.director.result === null && this.state.isInEncounter) {
      this.combatHud.setVisible(true);
    }
  }

  /**
   * Full reset for Try Again (brief §4.5). Everything this system owns is
   * dropped or re-armed; the host resets its own run state via `onRestart`.
   */
  restart(): void {
    this.finishEncounter();
    this.scheduler.reset();
    this.state.reset();
    this.encountersWon = 0;
    this.obstaclesCleared = 0;
    this.gameOverDelay = 0;
    this.pendingCause = null;
    this.pendingHitStop = 0;
    this.pendingShake = 0;
    this.carriedHitStop = 0;
    this.carriedShake = 0;
    // Brief 04 §5: `hareEverPerformed` resets on restart, so a fresh run
    // re-teaches the counter before it can kill anyone. The teardown runs
    // here too -- restart is an exit path.
    this.hareEverPerformed = false;
    this.lastNarutoHp = NARUTO_MAX_HP;
    this.lastObitoHp = OBITO_MAX_HP;
    this.tearDownSpecial();
    this.gameOverPanel.hide();
    this.narutoBar?.reset();
    this.obitoBar?.reset();
    this.obito?.resetForEncounter(1, OBITO_COMBAT_Z);
    this.obito?.setVisible(false);
    this.deps.onRestart();
  }

  dispose(): void {
    this.specialMarker?.dispose();
    this.gameOverPanel.dispose();
    this.obito?.dispose();
    this.vfx?.dispose();
    this.narutoBar?.dispose();
    this.obitoBar?.dispose();
    this.telegraph?.dispose();
    this.damageNumbers?.dispose();
  }
}
