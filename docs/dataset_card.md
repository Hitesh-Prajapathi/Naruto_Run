# Dataset card

Two datasets: the **legacy v0 dataset** (documented, with its known flaws) and the **v1 multi-subject collection** (template, to be filled in as data arrives).

---

## Part 1. Legacy v0 dataset ("Pure Naruto Hand Sign Data")

### Summary

| Property | Value |
|---|---|
| Content | Still images of 13 classes: 12 hand seals + `zero` (no seal) |
| Raw size | 2,245 images (2,159 in the original `train/` folder + 86 in `test/`) |
| Redistribution | Not distributed; excluded from git (`.gitignore`). License and consent status of the original source: **unknown, needs owner confirmation** |
| Used for | Training and evaluating the v0 `best_model_A.onnx`; the landmark spike |
| Status | **Legacy. Do not use to claim generalisation.** |

### Provenance (from `03_refinement_execution_plan.md` section 0.2)

Filenames are UUID1s, which encode the capturing machine and a timestamp.

- **1,797 of 2,245 images (80%)** come from one machine, captured **within 21 minutes on 2020-02-21**. These are burst frames of near-duplicate content.
- The other **437** come from one other source over 4 days. All 86 images of the original "test" folder are from a single day of that source.
- 11 images carry no UUID (excluded from the session-based splits in the landmark spike).
- Effectively **one subject, two sessions**. No subject, camera, lighting, or demographic metadata exists. Skin-tone and handedness coverage are unknown.

### Class counts per original split

Counted from the files in the raw folders (original `train/` and `test/`, before any merging or re-splitting).

| class | train | test | total |
|---|---:|---:|---:|
| bird | 188 | 4 | 192 |
| boar | 172 | 10 | 182 |
| dog | 263 | 10 | 273 |
| dragon | 146 | 10 | 156 |
| hare | 174 | 10 | 184 |
| horse | 157 | 5 | 162 |
| monkey | 136 | 5 | 141 |
| ox | 169 | 7 | 176 |
| ram | 117 | 5 | 122 |
| rat | 125 | 5 | 130 |
| snake | 146 | 5 | 151 |
| tiger | 167 | 5 | 172 |
| zero | 199 | 5 | 204 |
| **all** | **2,159** | **86** | **2,245** |

Class balance is uneven (ram 122 vs dog 273 in total); the v0 pipeline balanced classes by augmentation.

### How v0 was prepared (and why that is a problem)

`training/legacy/prepare_dataset.py` (kept only to reproduce the baseline):

1. Merges the original train and test folders per class.
2. Applies a **random stratified 80/10/10 split per frame** (seed 42), so near-identical burst frames appear in train, val and test.
3. Centre-crops/resizes to 224 x 224 and **augments to balance each class to 220** (the source of the "2,860 images" figure; about 1,795 real train images, the rest augmented copies).

Consequence: the reported 99.64% val / 99.11% test accuracy measures memorisation. See [ADR 0001](adr/0001-leakage-finding-and-subject-disjoint-evaluation.md) and `docs/results/v0-baseline.md` (TBD, in progress).

### Known gaps

- Single subject, two sessions; no held-out subjects.
- Whole-frame images (background, face, clothing included), so models learn the room.
- Canonical seal definitions were never fixed; the dragon variant in the data may not match how it is performed live ([seal spec](seals.md)).
- Landmark detectability on this data is poor for interlocked seals ([spike](results/landmark_spike_v0.md)).

---

## Part 2. v1 multi-subject collection (TEMPLATE, not yet collected)

Collected with `tools/collect` (see `tools/collect/README.md`; schema `schemas/collection_session_v1.schema.json`). Fill in after data arrives. Report **aggregates only**; never publish per-person demographic data.

### Summary (TBD)

| Property | Value |
|---|---|
| Dataset version | TBD |
| Git SHA / DVC hash | TBD |
| Number of subjects | TBD (plan: minimum 15, target 25; vault subjects 3 to 5) |
| Sessions per subject | TBD (plan: 1 to 2, different room/light) |
| Takes per seal per session | 3 (protocol default) |
| Continuous labeled video | TBD (plan: 30 to 60 min) |
| Total clips / total duration | TBD |
| Splits | `splits.yaml`: LOSO folds + vault subject list, TBD |

### Collection protocol (from `tools/collect/src/protocol.ts`, protocol version 1.0.0)

1. **Consent screen** (text in `tools/collect/src/content/consent.md`; its hash is stored per session), then a random pseudonymous `subject_id`. No names or emails are collected.
2. **Metadata form:** camera type (laptop / external / phone), lighting (low / normal / backlit), background (plain / cluttered), distance (0.5 / 1 / 2 m), dominant hand (L / R), optional self-reported Monk Skin Tone bucket (1 to 10, or prefer not to say).
3. **Seal block:** for each of the 13 seals (order reshuffled per take), 3 takes of: countdown 3 s, form 2 s, hold 3 s, release 1.5 s.
4. **Combo block:** 3 prompted combos (tiger-horse, snake-dragon, dog-rat), one clip each.
5. **Free-motion block:** 2 minutes of natural non-seal hand motion (talking with hands, touching face, phone, clapping, prayer hands, stretching). These are hard negatives and feed the false-activations-per-minute metric.
6. Output: video clips (WebM/MP4, unmirrored) plus per-clip prompt timestamps in `session.json`. Prompt times are not true gesture onset; trim using landmark stability inside the hold window.

Data layout: `<subject_id>/<session_id>/session.json` and `clips/<clip_id>.webm|mp4`.

### Consent, retention and deletion

- Purpose, retention and right-to-delete wording: `tools/collect/src/content/consent.md` (owner to review before recruiting).
- Retention period and storage location: TBD.
- Deletion procedure: one subject, one folder; `receive.py --delete-subject <id>`, then remove from every copy (DVC remote, backups, derived data, `splits.yaml`).

### Coverage tables (aggregated only, TBD)

Fill with counts of subjects and clips per bucket. Suppress or merge any bucket with very few subjects so individuals cannot be identified.

| Slice | Buckets | Subjects | Clips |
|---|---|---:|---:|
| Lighting | low / normal / backlit | TBD | TBD |
| Background | plain / cluttered | TBD | TBD |
| Distance | 0.5 / 1 / 2 m | TBD | TBD |
| Camera | laptop / external / phone | TBD | TBD |
| Dominant hand | L / R | TBD | TBD |
| Monk Skin Tone | grouped buckets (e.g. 1-3, 4-6, 7-10) + not stated | TBD | TBD |

### Class balance (TBD)

Clips and labeled frames per seal after auto-trim and review: TBD.

### Labelling

Auto-labelled from prompts, transitions trimmed by landmark stability, then reviewed (accept / trim / reject) in a review tool (planned). Review rates: TBD.

### Known gaps and risks (fill in)

TBD. Expected: skin-tone coverage limited by who can be recruited; Safari, Android and real-camera behaviour of the recorder untested; subjects are likely students/friends, not a representative population.
