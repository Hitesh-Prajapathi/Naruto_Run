import "./style.css";
import { CONSENT_HASH, CONSENT_TEXT, renderMarkdown } from "./content";
import { DEFAULT_CONFIG, generatePlan } from "./protocol";
import { browserMime, cameraWarnings, FpsMeter, openCamera, REQUESTED } from "./recorder";
import {
  newProtocolInfo,
  newSeed,
  newSessionId,
  newSubjectId,
  type DeviceInfo,
  type SessionRecord,
  type SubjectInfo,
} from "./session";
import { Store } from "./store";
import { collectBundle, uploadBundle, zipBundle } from "./export";
import { h, mount } from "./ui/dom";
import { runSession } from "./ui/run";

const root = document.getElementById("app")!;
const UPLOAD_URL = (import.meta.env.VITE_UPLOAD_URL as string | undefined) || "";
const UPLOAD_TOKEN = (import.meta.env.VITE_UPLOAD_TOKEN as string | undefined) || "";

const MST = ["#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96", "#a07e56", "#825c43", "#604134", "#3a312a", "#292420"];

let store: Store;
let stream: MediaStream | null = null;
let stopRun: (() => void) | null = null;
let rec: SessionRecord | null = null;
let pendingSubject: SubjectInfo | null = null;
let cameraInfo: { width: number; height: number; fps: number; label: string } | null = null;

async function boot() {
  if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
    mount(root, h("div", { class: "error" }, "This page needs a camera and a secure context (https:// or localhost). Open it over https."));
    return;
  }
  if (!browserMime()) {
    mount(root, h("div", { class: "error" }, "This browser cannot record video (no MediaRecorder support). Please use a recent Chrome, Edge, Safari or Firefox."));
    return;
  }
  try {
    store = await Store.open();
  } catch {
    mount(root, h("div", { class: "error" }, "Browser storage (IndexedDB) is unavailable. Private/incognito mode can cause this; use a normal window."));
    return;
  }
  const existing = await store.latestInProgress();
  if (existing) return screenResume(existing);
  screenConsent();
}

// ---------------------------------------------------------------- resume
function screenResume(r: SessionRecord) {
  mount(
    root,
    h("h1", {}, "Unfinished session found"),
    h("div", { class: "panel" },
      h("p", {}, "You have an unfinished recording session on this device."),
      h("p", {}, "Subject ID: ", h("span", { class: "big-id" }, r.subject.subject_id)),
      h("button", { onclick: () => ((rec = r), screenCamera()) }, "Resume it"),
      h("button", { class: "secondary", onclick: async () => {
        if (confirm("Delete the unfinished session and its saved clips from this device?")) {
          await store.deleteSession(r.session_id);
          screenConsent();
        }
      } }, "Discard and start over"),
    ),
  );
}

// ---------------------------------------------------------------- consent
function screenConsent() {
  const cb = h("input", { type: "checkbox", id: "consent" });
  const go = h("button", { disabled: true, onclick: () => screenForm() }, "Continue");
  cb.addEventListener("change", () => (go.disabled = !cb.checked));
  const body = h("div", { class: "panel consent" });
  body.innerHTML = renderMarkdown(CONSENT_TEXT);
  mount(
    root,
    body,
    h("div", { class: "panel" },
      h("label", { for: "consent", style: "display:flex;gap:10px;align-items:flex-start;font-weight:600" },
        cb, "I have read the above, I am 18 or older, and I agree to be recorded for this purpose."),
      go,
    ),
  );
}

// ---------------------------------------------------------------- metadata form
function selectField(id: string, label: string, opts: [string, string][], dflt?: string) {
  const sel = h("select", { id });
  for (const [v, t] of opts) sel.append(h("option", { value: v, selected: v === dflt }, t));
  return { el: h("div", {}, h("label", { for: id }, label), sel), sel };
}

function screenForm() {
  const suggested = newSubjectId();
  const idInput = h("input", { type: "text", id: "sid", value: suggested, readOnly: true });
  const reuse = h("input", { type: "checkbox", id: "reuse" });
  reuse.addEventListener("change", () => {
    idInput.readOnly = !reuse.checked;
    if (!reuse.checked) idInput.value = suggested;
  });
  const cam = selectField("cam", "Camera", [["laptop", "Laptop webcam"], ["external", "External webcam"], ["phone", "Phone"]]);
  const light = selectField("light", "Lighting", [["normal", "Normal"], ["low", "Low light"], ["backlit", "Backlit (window/lamp behind you)"]]);
  const bg = selectField("bg", "Background", [["plain", "Plain"], ["cluttered", "Cluttered"]]);
  const dist = selectField("dist", "Approx. distance from camera", [["0.5", "0.5 m"], ["1", "1 m"], ["2", "2 m"]], "1");
  const hand = selectField("hand", "Dominant hand", [["R", "Right"], ["L", "Left"]]);

  let monk: number | null = null;
  const sw = h("div", { class: "swatches" });
  const none = h("div", { class: "none-opt selected" }, "Prefer not to say");
  const pick = (v: number | null) => {
    monk = v;
    none.classList.toggle("selected", v === null);
    sw.querySelectorAll("label").forEach((l, i) => l.classList.toggle("selected", v === i + 1));
  };
  MST.forEach((c, i) => {
    const r = h("input", { type: "radio", name: "monk", value: String(i + 1), onchange: () => pick(i + 1) });
    sw.append(h("label", {}, r, h("span", { class: "sw", style: `background:${c}` }), String(i + 1)));
  });
  none.addEventListener("click", () => pick(null));

  const err = h("div");
  mount(
    root,
    h("h1", {}, "About your setup"),
    h("div", { class: "panel" },
      h("label", { for: "sid" }, "Your subject ID"),
      idInput,
      h("div", { class: "big-id" }, suggested),
      h("p", { class: "muted" }, "Write this down. It is the ONLY way to ask us to delete your data. Do not enter your name or email anywhere."),
      h("label", { style: "font-weight:400" }, reuse, " I already have a subject ID from an earlier session (type it above)"),
    ),
    h("div", { class: "panel" },
      h("div", { class: "row" }, cam.el, light.el, bg.el),
      h("div", { class: "row" }, dist.el, hand.el),
      h("label", {}, "Skin tone (optional, self-reported Monk Skin Tone scale)"),
      sw, none,
      h("p", { class: "muted" }, "This is only used to check the model works across skin tones. Skipping is fine."),
    ),
    err,
    h("button", { onclick: () => {
      const sid = idInput.value.trim();
      if (!/^[A-Za-z0-9_-]{4,64}$/.test(sid)) {
        err.replaceChildren(h("div", { class: "error" }, "Subject ID must be 4-64 letters, digits, - or _."));
        return;
      }
      pendingSubject = {
        subject_id: sid,
        camera_type: cam.sel.value as SubjectInfo["camera_type"],
        lighting: light.sel.value as SubjectInfo["lighting"],
        background: bg.sel.value as SubjectInfo["background"],
        distance_m: Number(dist.sel.value) as SubjectInfo["distance_m"],
        dominant_hand: hand.sel.value as "L" | "R",
        monk_skin_tone: monk,
        consent: { accepted: true, accepted_at: new Date().toISOString(), text_hash: CONSENT_HASH },
      };
      screenCamera();
    } }, "Continue to camera check"),
  );
}

// ---------------------------------------------------------------- camera check
async function screenCamera() {
  stopRun?.();
  const video = h("video", { class: "preview", autoplay: true, muted: true, playsInline: true });
  const status = h("div", { class: "panel" }, "Starting camera…");
  const go = h("button", { disabled: true }, "Looks good, start recording");
  mount(
    root,
    h("h1", {}, "Camera check"),
    h("p", {}, "Make sure both hands and your upper body are visible and well lit. The preview is mirrored like a mirror; the saved video is not mirrored."),
    video,
    status,
    go,
  );
  try {
    stream?.getTracks().forEach((t) => t.stop());
    stream = await openCamera();
  } catch (e) {
    status.replaceChildren(h("div", { class: "error" }, `Could not open the camera: ${(e as Error).name}. Allow camera access in the browser and reload.`));
    return;
  }
  video.srcObject = stream;
  await video.play().catch(() => {});
  const track = stream.getVideoTracks()[0];
  const settings = track.getSettings();
  const fallback = settings.frameRate ?? 0;
  const meter = new FpsMeter(video, fallback);
  meter.start();
  await new Promise((r) => setTimeout(r, 2500));
  const fps = meter.stop();
  const width = video.videoWidth || settings.width || 0;
  const height = video.videoHeight || settings.height || 0;
  cameraInfo = { width, height, fps, label: track.label };
  const w = cameraWarnings(width, height, fps);
  mount(
    status,
    h("div", {}, `Camera: ${width}×${height} at about ${fps} fps (asked for ${REQUESTED.width}×${REQUESTED.height} @ ${REQUESTED.fps}).`),
    w.lowFps ? h("div", { class: "warn" }, "Frame rate is below 24 fps. Close other apps/tabs using the camera, add light, or try another camera. You can still continue, but the data is less useful.") : null,
    w.lowRes ? h("div", { class: "warn" }, "Resolution is below 640×480. Hands may be too small to recognise. Move closer or try another camera.") : null,
  );
  go.disabled = false;
  go.onclick = () => void beginSession();
}

function deviceInfo(): DeviceInfo {
  const c = cameraInfo!;
  return {
    user_agent: navigator.userAgent,
    platform: navigator.platform || "",
    screen: { width: screen.width, height: screen.height, device_pixel_ratio: window.devicePixelRatio || 1 },
    camera_label: c.label || undefined,
    requested: { ...REQUESTED },
    achieved: { width: c.width, height: c.height, fps_measured: c.fps },
    mirrored_display: true,
  };
}

async function beginSession() {
  if (rec) {
    rec = { ...rec, device: deviceInfo() };
  } else {
    const seed = newSeed();
    rec = {
      session_id: newSessionId(),
      subject: pendingSubject!,
      device: deviceInfo(),
      protocol: newProtocolInfo(seed, DEFAULT_CONFIG),
      plan: generatePlan(seed, DEFAULT_CONFIG),
      started_at: new Date().toISOString(),
      ended_at: null,
      status: "in_progress",
    };
  }
  await store.saveSession(rec);
  try {
    void (navigator as Navigator & { wakeLock?: { request(t: string): Promise<unknown> } }).wakeLock?.request("screen");
  } catch { /* optional */ }
  const r = rec;
  stopRun = runSession({
    root, store, rec: r, cfg: r.protocol.config, stream: stream!, mime: browserMime()!,
    fallbackFps: cameraInfo?.fps ?? 30,
    onFinished: () => void screenFinish(r),
  });
}

// ---------------------------------------------------------------- finish
async function screenFinish(r: SessionRecord) {
  stopRun?.();
  stopRun = null;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  r.status = "complete";
  r.ended_at = new Date().toISOString();
  await store.saveSession(r);
  const bundle = await collectBundle(store, r);
  const total = bundle.clips.reduce((n, c) => n + c.data.byteLength, 0);
  const msg = h("div", { class: "panel" });
  const tokenInput = h("input", { type: "password", id: "tok", value: UPLOAD_TOKEN, placeholder: "upload token (given to you by the organiser)" });

  const download = h("button", { onclick: () => {
    const zip = zipBundle(bundle);
    const url = URL.createObjectURL(new Blob([zip as BlobPart], { type: "application/zip" }));
    const a = h("a", { href: url, download: `${r.subject.subject_id}_${r.session_id}.zip` });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  } }, "Download session (.zip)");

  const upBtn = h("button", { class: "secondary", onclick: async () => {
    upBtn.disabled = true;
    try {
      await uploadBundle(bundle, {
        baseUrl: UPLOAD_URL,
        token: tokenInput.value,
        onProgress: (n, t, what) => msg.replaceChildren(`Uploading ${n}/${t}: ${what}`),
      });
      msg.replaceChildren(h("div", { style: "color:var(--ok)" }, "Upload complete. Thank you!"));
    } catch (e) {
      msg.replaceChildren(h("div", { class: "error" }, `Upload failed: ${(e as Error).message}. Your clips are still saved on this device; try again or download the zip.`));
      upBtn.disabled = false;
    }
  } }, "Upload");

  mount(
    root,
    h("h1", {}, "All done. Thank you!"),
    h("div", { class: "panel" },
      h("p", {}, `${bundle.clips.length} clips saved (${(total / 1e6).toFixed(0)} MB).`),
      h("p", {}, "Your subject ID (keep it to request deletion):"),
      h("div", { class: "big-id" }, r.subject.subject_id),
    ),
    h("div", { class: "panel" },
      download,
      UPLOAD_URL ? h("div", {}, h("label", { for: "tok" }, "Upload token"), tokenInput, upBtn) : h("p", { class: "muted" }, "Send the downloaded zip to the organiser."),
      msg,
    ),
    h("div", { class: "panel" },
      h("button", { class: "secondary", onclick: async () => {
        if (confirm("Remove the recordings from this browser? Only do this after you have downloaded or uploaded them.")) {
          await store.deleteSession(r.session_id);
          rec = null;
          screenConsent();
        }
      } }, "Clear recordings from this device"),
    ),
  );
}

void boot();
