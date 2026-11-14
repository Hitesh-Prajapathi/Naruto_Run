/**
 * IndexedDB persistence: clips are written as soon as they are recorded, so a tab reload / crash
 * loses at most the clip in flight. Blobs are stored as ArrayBuffers (older Safari versions had
 * trouble with Blobs in IndexedDB).
 */
import type { ClipMeta, SessionRecord } from "./session";

const DB_NAME = "narutocv-collect";
const DB_VERSION = 1;

export interface ClipBlobRow {
  key: string; // `${session_id}/${clip_id}`
  data: ArrayBuffer;
  mime: string;
}

function req<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((res, rej) => {
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
function done(tx: IDBTransaction): Promise<void> {
  return new Promise((res, rej) => {
    tx.oncomplete = () => res();
    tx.onerror = () => rej(tx.error);
    tx.onabort = () => rej(tx.error);
  });
}

export class Store {
  private constructor(private db: IDBDatabase) {}

  static async open(name: string = DB_NAME): Promise<Store> {
    const open = indexedDB.open(name, DB_VERSION);
    open.onupgradeneeded = () => {
      const db = open.result;
      db.createObjectStore("sessions", { keyPath: "session_id" });
      db.createObjectStore("clipMeta", { keyPath: "key" }); // {key, session_id, meta}
      db.createObjectStore("clipBlobs", { keyPath: "key" });
    };
    return new Store(await req(open));
  }

  close() {
    this.db.close();
  }

  async saveSession(rec: SessionRecord): Promise<void> {
    const tx = this.db.transaction("sessions", "readwrite");
    tx.objectStore("sessions").put(rec);
    await done(tx);
  }
  async getSession(id: string): Promise<SessionRecord | undefined> {
    return req(this.db.transaction("sessions").objectStore("sessions").get(id));
  }
  async listSessions(): Promise<SessionRecord[]> {
    return req(this.db.transaction("sessions").objectStore("sessions").getAll());
  }
  /** Most recently started session that is still in progress. */
  async latestInProgress(): Promise<SessionRecord | undefined> {
    const all = (await this.listSessions()).filter((s) => s.status === "in_progress");
    all.sort((a, b) => b.started_at.localeCompare(a.started_at));
    return all[0];
  }

  /** Atomic: meta + blob together, replacing any previous recording of the same clip_id. */
  async putClip(sessionId: string, meta: ClipMeta, data: ArrayBuffer): Promise<void> {
    const key = `${sessionId}/${meta.clip_id}`;
    const tx = this.db.transaction(["clipMeta", "clipBlobs"], "readwrite");
    tx.objectStore("clipMeta").put({ key, session_id: sessionId, meta });
    tx.objectStore("clipBlobs").put({ key, data, mime: meta.mime } satisfies ClipBlobRow);
    await done(tx);
  }
  async getClipMeta(sessionId: string, clipId: string): Promise<ClipMeta | undefined> {
    const row = await req(this.db.transaction("clipMeta").objectStore("clipMeta").get(`${sessionId}/${clipId}`));
    return row?.meta;
  }
  async listClipMetas(sessionId: string): Promise<ClipMeta[]> {
    const rows: { session_id: string; meta: ClipMeta }[] = await req(
      this.db.transaction("clipMeta").objectStore("clipMeta").getAll(),
    );
    return rows.filter((r) => r.session_id === sessionId).map((r) => r.meta);
  }
  async getClipData(sessionId: string, clipId: string): Promise<ArrayBuffer | undefined> {
    const row: ClipBlobRow | undefined = await req(
      this.db.transaction("clipBlobs").objectStore("clipBlobs").get(`${sessionId}/${clipId}`),
    );
    return row?.data;
  }
  async deleteClip(sessionId: string, clipId: string): Promise<void> {
    const key = `${sessionId}/${clipId}`;
    const tx = this.db.transaction(["clipMeta", "clipBlobs"], "readwrite");
    tx.objectStore("clipMeta").delete(key);
    tx.objectStore("clipBlobs").delete(key);
    await done(tx);
  }
  async deleteSession(sessionId: string): Promise<void> {
    for (const m of await this.listClipMetas(sessionId)) await this.deleteClip(sessionId, m.clip_id);
    const tx = this.db.transaction("sessions", "readwrite");
    tx.objectStore("sessions").delete(sessionId);
    await done(tx);
  }
}
