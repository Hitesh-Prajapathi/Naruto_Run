/** Zip export and chunked upload. */
import { zipSync } from "fflate";
import { buildSessionJson, type ClipMeta, type SessionRecord } from "./session";
import type { Store } from "./store";

export interface ExportBundle {
  sessionJson: ReturnType<typeof buildSessionJson>;
  clips: { meta: ClipMeta; data: ArrayBuffer }[];
}

export async function collectBundle(store: Store, rec: SessionRecord): Promise<ExportBundle> {
  const metas = await store.listClipMetas(rec.session_id);
  const clips: ExportBundle["clips"] = [];
  for (const meta of metas) {
    const data = await store.getClipData(rec.session_id, meta.clip_id);
    if (data) clips.push({ meta, data });
  }
  const present = clips.map((c) => c.meta);
  return { sessionJson: buildSessionJson(rec, present), clips };
}

/** session.json + clips/<clip_id>.<ext>. Video is already compressed, so it is stored (level 0). */
export function zipBundle(b: ExportBundle): Uint8Array {
  const files: Record<string, [Uint8Array, { level: 0 | 6 }]> = {
    "session.json": [new TextEncoder().encode(JSON.stringify(b.sessionJson, null, 2)), { level: 6 }],
  };
  for (const c of b.clips) files[`clips/${c.meta.clip_id}.${c.meta.ext}`] = [new Uint8Array(c.data), { level: 0 }];
  return zipSync(files);
}

export interface UploadOptions {
  baseUrl: string;
  token: string;
  retries?: number;
  backoffMs?: number;
  fetchImpl?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
  onProgress?: (done: number, total: number, what: string) => void;
}

async function postWithRetry(url: string, init: RequestInit, o: UploadOptions): Promise<void> {
  const f = o.fetchImpl ?? fetch;
  const sleep = o.sleep ?? ((ms) => new Promise<void>((r) => setTimeout(r, ms)));
  const retries = o.retries ?? 4;
  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const r = await f(url, init);
      if (r.ok) return;
      // 4xx (except 408/429) will not succeed on retry.
      if (r.status >= 400 && r.status < 500 && r.status !== 408 && r.status !== 429) {
        throw Object.assign(new Error(`HTTP ${r.status}: ${await r.text().catch(() => "")}`), { fatal: true });
      }
      lastErr = new Error(`HTTP ${r.status}`);
    } catch (e) {
      if ((e as { fatal?: boolean }).fatal) throw e;
      lastErr = e;
    }
    if (attempt < retries) await sleep((o.backoffMs ?? 1000) * 2 ** attempt);
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

/** Uploads clips one at a time, then session.json last (its presence marks a complete upload). */
export async function uploadBundle(b: ExportBundle, o: UploadOptions): Promise<void> {
  const { subject_id } = b.sessionJson.subject;
  const { session_id } = b.sessionJson.session;
  const base = `${o.baseUrl.replace(/\/$/, "")}/api/v1/${encodeURIComponent(subject_id)}/${encodeURIComponent(session_id)}`;
  const total = b.clips.length + 1;
  let n = 0;
  for (const c of b.clips) {
    await postWithRetry(
      `${base}/clips/${encodeURIComponent(c.meta.clip_id)}.${c.meta.ext}`,
      { method: "PUT", headers: { "X-Collect-Token": o.token, "Content-Type": c.meta.mime }, body: c.data },
      o,
    );
    o.onProgress?.(++n, total, c.meta.clip_id);
  }
  await postWithRetry(
    `${base}/session.json`,
    {
      method: "PUT",
      headers: { "X-Collect-Token": o.token, "Content-Type": "application/json" },
      body: JSON.stringify(b.sessionJson),
    },
    o,
  );
  o.onProgress?.(++n, total, "session.json");
}
