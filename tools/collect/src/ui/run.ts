/** The prompted-recording screen: ready -> running -> review, with retake and pause/resume. */
import { buildTimeline, clipEndMs, type ClipPlan, type ProtocolConfig, type TimelineStep } from "../protocol";
import { PhaseClock } from "../timing";
import { startRecording, type ActiveRecording } from "../recorder";
import { extForMime, type ClipMeta, type SessionRecord } from "../session";
import type { Store } from "../store";
import { SEAL_INFO } from "../content";
import { h, mount, rAF } from "./dom";

export interface RunDeps {
  root: HTMLElement;
  store: Store;
  rec: SessionRecord;
  cfg: ProtocolConfig;
  stream: MediaStream;
  mime: string;
  fallbackFps: number;
  onFinished: () => void;
}

type State = "ready" | "running" | "review" | "paused";

class Cancelled extends Error {}

export function runSession(d: RunDeps): () => void {
  const { root, store, rec, cfg } = d;
  const plan = rec.plan;
  const done = new Set<string>();
  const attempts = new Map<string, number>();
  let cursor = 0;
  let state: State = "ready";
  let pausedFrom: "ready" | "review" = "ready";
  let cancel: (() => void) | null = null;
  let active: ActiveRecording | null = null;
  let alive = true;
  let lastMetaMsg = "";

  // --- DOM skeleton (built once; updated in place) ---
  const video = h("video", { class: "mini", autoplay: true, muted: true, playsInline: true });
  video.srcObject = d.stream;
  const recDot = h("div", { class: "rec" }, h("i"), "REC");
  const progText = h("div", { class: "muted" });
  const barFill = h("div");
  const blockText = h("strong");
  const promptBox = h("div", { class: "prompt" });
  const cardBox = h("div", { class: "card" });
  const hint = h("div", { class: "muted" });
  const btnMain = h("button", { tabIndex: -1, onclick: () => primary() });
  const btnRetake = h("button", { class: "secondary", tabIndex: -1, onclick: () => retake() }, "Retake (R)");
  const btnPause = h("button", { class: "secondary", tabIndex: -1, onclick: () => togglePause() }, "Pause (P)");
  mount(
    root,
    h("div", { class: "panel" },
      h("div", { class: "hud" }, blockText, recDot),
      h("div", { class: "bar" }, barFill),
      progText,
    ),
    h("div", { class: "panel stage" },
      h("div", {}, promptBox, cardBox),
      h("div", {}, video, h("div", { class: "muted" }, "Your preview (mirrored). The saved video is not mirrored.")),
    ),
    h("div", { class: "panel" }, btnMain, btnRetake, btnPause, hint),
  );

  function renderCard(p: ClipPlan, seal?: string) {
    const s = seal ?? p.label ?? p.sequence?.[0];
    if (!s) {
      cardBox.replaceChildren(h("div", { class: "ph" }, "free motion"));
      return;
    }
    const info = SEAL_INFO[s];
    const img = h("img", { src: `${import.meta.env.BASE_URL}seals/${s}.jpg`, alt: `${s} seal reference` });
    const ph = h("div", { class: "ph" }, s);
    const box = h("div", {});
    img.onerror = () => box.replaceChildren(ph);
    box.append(img);
    cardBox.replaceChildren(box, h("div", { class: "desc" }, h("strong", {}, s.toUpperCase() + ": "), info?.description ?? ""));
    if (info?.confusable_with?.length) cardBox.append(h("div", { class: "conf" }, `Easily confused with: ${info.confusable_with.join(", ")}. Make it distinct.`));
  }

  function setPrompt(word: string, sub: string, cls = "") {
    promptBox.className = `prompt ${cls}`;
    promptBox.replaceChildren(h("div", { class: "word" }, word), h("div", { class: "sub" }, sub));
  }

  function refreshChrome() {
    const p = plan[cursor];
    const nDone = done.size;
    blockText.textContent = p ? { seal: "Single seals", combo: "Combos", free: "Free motion" }[p.block] : "Finished";
    progText.textContent = `Clip ${Math.min(cursor + 1, plan.length)} of ${plan.length} · ${nDone} saved${lastMetaMsg ? " · " + lastMetaMsg : ""}`;
    barFill.style.width = `${(nDone / plan.length) * 100}%`;
    recDot.classList.toggle("on", state === "running");
    btnRetake.disabled = !(state === "review");
    btnPause.textContent = state === "paused" ? "Resume (P)" : "Pause (P)";
    btnMain.disabled = state === "running";
    if (state === "ready") {
      btnMain.textContent = "Start (Space)";
      hint.textContent = "Look at the reference below, get both hands in view, then press Space. Recording starts with a 3-2-1 countdown.";
    } else if (state === "review") {
      btnMain.textContent = cursor + 1 >= plan.length ? "Finish (Space)" : "Next (Space)";
      hint.textContent = "Saved. Space = next prompt, R = retake this clip.";
    } else if (state === "paused") {
      btnMain.textContent = "Resume (Space)";
      hint.textContent = "Paused. Nothing is recording. The clip in progress (if any) was discarded and will be redone.";
    } else hint.textContent = "Recording. Pausing now discards this clip.";
  }

  function showReady() {
    state = "ready";
    const p = plan[cursor];
    renderCard(p);
    if (p.block === "seal") setPrompt(p.label!.toUpperCase(), "Take " + p.take);
    else if (p.block === "combo") setPrompt(p.sequence!.join(" → ").toUpperCase(), "Do these seals in order");
    else setPrompt("FREE MOTION", "2 minutes: move your hands naturally. Do NOT make seals.");
    refreshChrome();
  }

  function sleepUntil(t: number): Promise<void> {
    return new Promise((res, rej) => {
      const id = window.setTimeout(res, Math.max(0, t - performance.now()));
      cancel = () => {
        clearTimeout(id);
        rej(new Cancelled());
      };
    });
  }

  function applyStep(p: ClipPlan, s: TimelineStep) {
    switch (s.phase) {
      case "countdown":
        setPrompt(String(s.n), "Get ready", "phase-countdown");
        break;
      case "form": {
        const total = p.sequence?.length ?? 1;
        renderCard(p, s.seal);
        setPrompt("FORM", `${s.seal!.toUpperCase()}${total > 1 ? ` (${s.segment! + 1}/${total})` : ""}`, "phase-form");
        break;
      }
      case "hold":
        setPrompt("HOLD", `${s.seal!.toUpperCase()}: keep it steady`, "phase-hold");
        break;
      case "release":
        setPrompt("RELEASE", "Relax your hands", "phase-release");
        break;
      case "free":
        cardBox.replaceChildren(h("div", { class: "ph" }, "free motion"));
        setPrompt("MOVE", s.text ?? "", "phase-hold");
        break;
      case "end":
        setPrompt("DONE", "Saving…", "");
    }
  }

  async function record(p: ClipPlan) {
    state = "running";
    refreshChrome();
    const steps = buildTimeline(p, cfg);
    const clock = new PhaseClock();
    active = startRecording(d.stream, d.mime, video, clock, d.fallbackFps);
    try {
      await active.started;
      const t0 = clock.origin; // identical to the 'start'-event anchor used for the marks
      for (const s of steps) {
        await sleepUntil(t0 + s.at);
        applyStep(p, s);
        const now = await rAF(); // mark at the frame that paints the prompt
        for (const m of s.marks) clock.mark(m.segment, m.name, now);
        if (p.block === "free" && s.phase === "free" && s.at === steps.find((x) => x.phase === "free")!.at) clock.markFree("start", now);
        if (p.block === "free" && s.phase === "end") clock.markFree("end", now);
      }
      await sleepUntil(t0 + clipEndMs(steps, cfg));
      cancel = null;
      const res = await active.stop();
      active = null;
      const attempt = attempts.get(p.clip_id) ?? 1;
      const meta: ClipMeta = {
        clip_id: p.clip_id,
        block: p.block,
        label: p.label,
        sequence: p.sequence,
        take: p.take,
        attempt,
        mime: res.mime,
        ext: extForMime(res.mime),
        size_bytes: res.data.byteLength,
        duration_ms: res.durationMs,
        width: video.videoWidth,
        height: video.videoHeight,
        fps_measured: res.fps,
        segments: p.block === "free" ? undefined : clock.segments(p.block === "seal" ? [p.label!] : p.sequence!),
        free: p.block === "free" ? { free_start: clock.freeStart!, free_end: clock.freeEnd! } : undefined,
      };
      await store.putClip(rec.session_id, meta, res.data);
      done.add(p.clip_id);
      lastMetaMsg = `${(res.data.byteLength / 1e6).toFixed(1)} MB, ${res.fps} fps`;
      state = "review";
      setPrompt("SAVED ✓", p.block === "free" ? "Great, that's the last block." : "Nice.", "phase-release");
      refreshChrome();
    } catch (e) {
      active?.abort();
      active = null;
      cancel = null;
      if (!(e instanceof Cancelled)) {
        promptBox.replaceChildren(h("div", { class: "error" }, `Recording failed: ${(e as Error).message}. Press Space to retry.`));
        state = "ready";
        refreshChrome();
      }
    }
  }

  function primary() {
    if (!alive) return;
    if (state === "ready") void record(plan[cursor]);
    else if (state === "review") {
      cursor++;
      if (cursor >= plan.length) return d.onFinished();
      void record(plan[cursor]); // card + countdown are shown as part of the clip
      renderCard(plan[cursor]);
    } else if (state === "paused") togglePause();
  }

  async function retake() {
    if (state !== "review") return;
    const p = plan[cursor];
    await store.deleteClip(rec.session_id, p.clip_id);
    done.delete(p.clip_id);
    attempts.set(p.clip_id, (attempts.get(p.clip_id) ?? 1) + 1);
    lastMetaMsg = "retaking";
    showReady();
  }

  function togglePause() {
    if (state === "paused") {
      state = pausedFrom;
      if (state === "review") refreshChrome();
      else showReady();
      return;
    }
    if (state === "running") {
      if (!cancel) return; // already stopping/saving: let it finish
      cancel();
      active?.abort();
      pausedFrom = "ready";
    } else pausedFrom = state === "review" ? "review" : "ready";
    state = "paused";
    setPrompt("PAUSED", "Take your time", "");
    refreshChrome();
  }

  const onKey = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.code === "Space") (e.preventDefault(), primary());
    else if (e.key === "r" || e.key === "R") void retake();
    else if (e.key === "p" || e.key === "P") togglePause();
  };
  window.addEventListener("keydown", onKey);

  // --- start: resume from the first unrecorded clip ---
  void (async () => {
    for (const m of await store.listClipMetas(rec.session_id)) done.add(m.clip_id);
    const i = plan.findIndex((p) => !done.has(p.clip_id));
    if (i < 0) return d.onFinished();
    cursor = i;
    showReady();
  })();

  return () => {
    alive = false;
    window.removeEventListener("keydown", onKey);
    cancel?.();
    active?.abort();
  };
}
