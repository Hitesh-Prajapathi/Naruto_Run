/**
 * Camera-input feature flag and connection settings -- Feature Brief 05
 * §0.2.
 *
 * Separate from `bodyTuning.ts` on purpose: that file holds thresholds the
 * owner will re-tune constantly and which load at runtime from JSON. These
 * are build-time wiring decisions, and the flag in particular must be a
 * compile-time constant so that with it off there is provably no camera
 * request, no permission prompt, no socket and no new UI.
 */

/**
 * Master switch. With this off the game is identical to the approved build:
 * `sceneMain` never constructs the camera source, so `getUserMedia` is never
 * called and no WebSocket is opened.
 *
 * Overridable per-session with `?cv=1` / `?cv=0` in the page URL, which is
 * how both the camera path and the flag-off non-regression check (R-01) are
 * exercised without a rebuild.
 *
 * **Default off, deliberately, until the camera path is signed off.**
 * Turning it on makes the startup gate block the run until a player is
 * detected — which is the intended end state (§4), but it also means anyone
 * without the recognition service running meets a gate instead of a game.
 * Defaulting to on broke every existing end-to-end test for exactly that
 * reason, and those tests were right to break: it is a change to how the
 * approved build starts. Flip this to `true` once §9.7's manual protocol has
 * been run on the demo machine; until then, play the camera build with
 * `?cv=1`.
 */
const DEFAULT_ENABLE_CV_INPUT = false;

function queryOverride(name: string): boolean | null {
  if (typeof window === "undefined") return null;
  const value = new URLSearchParams(window.location.search).get(name);
  if (value === "1") return true;
  if (value === "0") return false;
  return null;
}

export function isCvInputEnabled(): boolean {
  return queryOverride("cv") ?? DEFAULT_ENABLE_CV_INPUT;
}

/** Kept as a constant too, for code that cannot call a function (tests). */
export const ENABLE_CV_INPUT = DEFAULT_ENABLE_CV_INPUT;

/** Where the Python recognition service listens. Matches serve_transport.py. */
export const CV_TRANSPORT_URL = "ws://127.0.0.1:8765/ws";

/**
 * Capture settings.
 *
 * 640x360 at 20fps rather than 30: measured inference is ~13ms/frame
 * regardless of resolution (MediaPipe rescales internally, so sending
 * smaller frames buys nothing), which means the lever that actually reduces
 * load is *rate*, not size. 20fps still gives the 3-frame lean debounce a
 * 150ms window, comfortably inside the latency budget.
 */
export const CV_CAPTURE_WIDTH = 640;
export const CV_CAPTURE_HEIGHT = 360;
export const CV_TARGET_FPS = 20;
export const CV_JPEG_QUALITY = 0.7;

/** The command to start the service, shown verbatim when it is not running
 * (§4.2: "detection service not running" must be actionable). */
export const CV_SERVICE_COMMAND =
  ".\\cv_model\\Nishit_Frontend\\venv\\Scripts\\python.exe cv_model\\Nishit_Frontend\\serve_transport.py";
