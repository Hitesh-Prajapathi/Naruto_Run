# NarutoCV collection tool

Static browser app (Vite + TypeScript, no framework) that records **prompted video** of hand seals from many subjects, with clean metadata for leave-one-subject-out evaluation (plan section 2.3). Schema: `schemas/collection_session_v1.schema.json`.

## Run locally

```bash
cd tools/collect
npm i
npm run dev          # http://localhost:5173  (localhost counts as a secure context, so the camera works)
npm run build        # -> dist/
npx vitest run       # unit tests
npm run smoke        # headless Chromium + fake camera end-to-end (needs: npm run build; npx playwright install chromium)
```

Editable content: `src/content/consent.md` (consent text), `src/content/seals.json` (seal descriptions; **dragon is "TBD by project owner"**), `src/protocol.ts` (seals, takes, durations, combos, free-motion suggestions; bump `PROTOCOL_VERSION` when changing), `public/seals/*.jpg` (your own reference photos, see `public/seals/README.md`).

Shortcuts: **Space** start / next, **R** retake the last clip, **P** pause/resume. Pausing mid-clip discards that clip. Progress is saved per clip in IndexedDB; reloading the tab offers to resume.

## Host on GitHub Pages (remote friends)

Camera access needs https; GitHub Pages provides it. `vite.config.ts` uses `base: './'`, so any sub-path works.

```bash
cd tools/collect && npm run build
# publish dist/ e.g. with the gh-pages branch:
npx gh-pages -d dist        # (or copy dist/ to the branch/folder Pages serves)
```
Do not commit reference photos you do not want public. Optionally set `VITE_UPLOAD_URL` (and see `.env.example`) before building to enable the Upload button; otherwise subjects use **Download session** and send you the zip.

## Receiver (optional upload target)

```bash
pip install aiohttp jsonschema                 # only these two; not added to any pyproject
cd tools/collect/server
python receive.py --self-test                  # 18 checks, temp dir
COLLECT_TOKEN=pick-a-secret python receive.py --out ../../../data/raw/collection --port 8787
```
- **LAN**: friends on your network can't use `http://<lan-ip>` for the *camera page* (not a secure context) and an https Pages site can't post to an `http://` receiver (mixed content). Options: serve the receiver with `--cert/--key` (self-signed certs need to be trusted by each browser), or expose it through an https tunnel (e.g. cloudflared/ngrok) and set `VITE_UPLOAD_URL` to that URL, or just collect zips. Use `--allow-origin https://<you>.github.io` to restrict CORS.
- The shared token is a speed bump, not strong security: tell subjects the token and let them type it (don't bake it into the build). Anyone with the token can upload.
- Clips are uploaded one at a time with exponential-backoff retries; `session.json` goes last and is schema-validated (422 on violation, ids must match the URL, size caps enforced).

## Data layout

```
data/raw/collection/<subject_id>/<session_id>/
  session.json                      # schema collection_session_v1
  clips/<clip_id>.webm|mp4          # unmirrored video; clip_id like c012_seal_ram_t2
```
The downloaded zip has the same inner layout (`session.json` + `clips/`).
Per clip: `block` (seal/combo/free), `label` or `sequence`, `take`, `attempt`, `mime`, `duration_ms`, `width`, `height`, `fps_measured`, `phase_offsets_ms` (`form_start, hold_start, hold_end, release_end`, ms from clip start; combos also have per-seal `segments`; free clips have `free_start/free_end`). A clip begins at the countdown; the seal is formed at `form_start`.

**Timing accuracy:** offsets are `performance.now()` at the animation frame that paints each prompt, relative to `performance.now()` in the MediaRecorder `start` event. Expected accuracy is about +/- 1 camera frame (+/- 33 ms at 30 fps); in the headless fake-camera run the first-clip marks were within ~20 ms of nominal. These are PROMPT times, not true gesture onset (human reaction adds 150-400 ms), so trim using landmark stability inside the HOLD window. Not verified on real cameras or Safari.

## Consent and deletion procedure

1. Subject ticks the consent box (text in `src/content/consent.md`; its FNV-1a hash is stored in `session.json` so you know which wording was accepted).
2. Subject sees their random `subject_id` (e.g. `S-7K3QMD`) on the form and again at the end. No names or emails are collected; keep any "who got which ID" list private and separate, or don't keep one.
3. To honour a withdrawal: `python server/receive.py --out data/raw/collection --delete-subject S-7K3QMD`, then delete that subject from any other copies (DVC remote, backups, derived datasets, `splits.yaml`). Subjects can also clear their local browser copy via the button on the final screen.

## Browser support / known limits

Recording prefers `video/webm;codecs=vp9`, then vp8, then `video/mp4` (Safari). Live-recording behaviour in Safari, Android Chrome and with real cameras is **untested here**; only desktop Chromium with Chrome's fake camera was exercised (`npm run smoke`). Phones in portrait record portrait video (e.g. 720x1280). Memory: a clip is held in RAM until saved (the 2-minute free clip is roughly 30-45 MB).
