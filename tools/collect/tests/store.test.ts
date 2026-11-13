import "fake-indexeddb/auto";
import { beforeEach, describe, expect, it } from "vitest";
import { unzipSync } from "fflate";
import { Store } from "../src/store";
import { collectBundle, uploadBundle, zipBundle } from "../src/export";
import { generatePlan } from "../src/protocol";
import { makeRecord, makeValidator, simulateClip } from "./helpers";

let n = 0;
let store: Store;
beforeEach(async () => {
  store = await Store.open(`test-db-${n++}`);
});

const bytes = (len: number, fill: number) => new Uint8Array(len).fill(fill).buffer;

describe("IndexedDB store", () => {
  it("saves and reads a session; latestInProgress ignores complete ones", async () => {
    const a = makeRecord("A");
    const b = { ...makeRecord("B"), session_id: "20261113-110000-zzzz", started_at: "2026-11-13T11:00:00.000Z" };
    const c = { ...makeRecord("C"), session_id: "20261113-120000-yyyy", started_at: "2026-11-13T12:00:00.000Z", status: "complete" as const };
    for (const r of [a, b, c]) await store.saveSession(r);
    expect((await store.getSession(a.session_id))?.protocol.seed).toBe("A");
    expect((await store.latestInProgress())?.session_id).toBe(b.session_id);
  });

  it("round-trips clip bytes + meta, and survives reopening the DB (reload)", async () => {
    const rec = makeRecord();
    await store.saveSession(rec);
    await store.putClip(rec.session_id, simulateClip(rec.plan[0]), bytes(1000, 7));
    await store.putClip(rec.session_id, simulateClip(rec.plan[1]), bytes(10, 9));
    store.close();
    const reopened = await Store.open(`test-db-${n - 1}`);
    const metas = await reopened.listClipMetas(rec.session_id);
    expect(metas.map((m) => m.clip_id).sort()).toEqual([rec.plan[0].clip_id, rec.plan[1].clip_id].sort());
    const data = new Uint8Array((await reopened.getClipData(rec.session_id, rec.plan[0].clip_id))!);
    expect(data.length).toBe(1000);
    expect(data[0]).toBe(7);
  });

  it("retake: putClip replaces, deleteClip removes meta and data", async () => {
    const rec = makeRecord();
    const p = rec.plan[0];
    await store.putClip(rec.session_id, simulateClip(p, 0, 1), bytes(5, 1));
    await store.putClip(rec.session_id, simulateClip(p, 0, 2), bytes(8, 2));
    expect((await store.getClipMeta(rec.session_id, p.clip_id))?.attempt).toBe(2);
    expect((await store.listClipMetas(rec.session_id)).length).toBe(1);
    await store.deleteClip(rec.session_id, p.clip_id);
    expect(await store.getClipData(rec.session_id, p.clip_id)).toBeUndefined();
    expect(await store.listClipMetas(rec.session_id)).toEqual([]);
  });

  it("isolates sessions and deleteSession removes everything", async () => {
    const r1 = makeRecord();
    const r2 = { ...makeRecord(), session_id: "other-session-1" };
    await store.saveSession(r1);
    await store.saveSession(r2);
    await store.putClip(r1.session_id, simulateClip(r1.plan[0]), bytes(1, 1));
    await store.putClip(r2.session_id, simulateClip(r2.plan[0]), bytes(1, 1));
    await store.deleteSession(r1.session_id);
    expect(await store.getSession(r1.session_id)).toBeUndefined();
    expect(await store.listClipMetas(r1.session_id)).toEqual([]);
    expect((await store.listClipMetas(r2.session_id)).length).toBe(1);
  });

  it("resume: the first plan item without a stored clip is the next one", async () => {
    const rec = makeRecord();
    for (const p of rec.plan.slice(0, 4)) await store.putClip(rec.session_id, simulateClip(p), bytes(1, 1));
    const doneIds = new Set((await store.listClipMetas(rec.session_id)).map((m) => m.clip_id));
    expect(rec.plan.findIndex((p) => !doneIds.has(p.clip_id))).toBe(4);
  });
});

describe("export", () => {
  it("zip contains session.json (schema-valid) and clips/<id>.<ext>", async () => {
    const rec = makeRecord("Z", "complete");
    for (const p of rec.plan.slice(0, 3)) await store.putClip(rec.session_id, simulateClip(p), bytes(64, 5));
    const files = unzipSync(zipBundle(await collectBundle(store, rec)));
    expect(Object.keys(files).sort()).toEqual(
      ["session.json", ...rec.plan.slice(0, 3).map((p) => `clips/${p.clip_id}.webm`)].sort(),
    );
    const json = JSON.parse(new TextDecoder().decode(files["session.json"]));
    expect(makeValidator()(json)).toBe(true);
    expect(files[`clips/${rec.plan[0].clip_id}.webm`].length).toBe(64);
  });

  it("upload: one clip per request, session.json last, retries 5xx, not 4xx", async () => {
    const rec = makeRecord("U", "complete");
    for (const p of rec.plan.slice(0, 2)) await store.putClip(rec.session_id, simulateClip(p), bytes(8, 1));
    const bundle = await collectBundle(store, rec);
    const calls: string[] = [];
    let failures = 1;
    const fetchImpl = (async (url: string, init: RequestInit) => {
      calls.push(`${init.method} ${url.split("/api/v1/")[1]}`);
      if (url.includes(rec.plan[0].clip_id) && failures-- > 0) return new Response("boom", { status: 503 });
      expect((init.headers as Record<string, string>)["X-Collect-Token"]).toBe("tok");
      return new Response("ok", { status: 200 });
    }) as unknown as typeof fetch;
    const progress: number[] = [];
    await uploadBundle(bundle, { baseUrl: "https://h/", token: "tok", fetchImpl, sleep: async () => {}, onProgress: (d) => progress.push(d) });
    const base = `S-7K3QMD/${rec.session_id}`;
    expect(calls).toEqual([
      `PUT ${base}/clips/${rec.plan[0].clip_id}.webm`,
      `PUT ${base}/clips/${rec.plan[0].clip_id}.webm`, // retry
      `PUT ${base}/clips/${rec.plan[1].clip_id}.webm`,
      `PUT ${base}/session.json`,
    ]);
    expect(progress).toEqual([1, 2, 3]);

    const calls401: number[] = [];
    const f401 = (async () => (calls401.push(1), new Response("no", { status: 401 }))) as unknown as typeof fetch;
    await expect(uploadBundle(bundle, { baseUrl: "https://h", token: "x", fetchImpl: f401, sleep: async () => {} })).rejects.toThrow(/401/);
    expect(calls401.length).toBe(1);
  });
});

describe("plan sanity", () => {
  it("generatePlan ids work as store keys", () => {
    expect(new Set(generatePlan("k").map((p) => p.clip_id)).size).toBe(43);
  });
});
