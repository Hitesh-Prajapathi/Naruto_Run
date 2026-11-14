# 0001. Leakage finding and subject-disjoint evaluation

Status: Accepted

## Context

The v0 image model reported 99.64% validation and 99.11% test accuracy. Live webcam tests contradicted this: dragon was detected 0/3 in both live runs, ram was predicted as rat 5 of 6 times, and the code carried 13 hand-tuned confidence floors to patch individual failures.

Investigating the data (see `03_refinement_execution_plan.md` section 0.2) showed:

- Filenames are UUID1s, which encode the capturing machine and time. 1,797 of 2,245 images (80%) came from one machine within 21 minutes on one day; the other 437 came from one other source over 4 days. This is effectively one subject and two sessions.
- `prepare_dataset.py` merged the original train and test folders and then did a random stratified 80/10/10 split per frame. Near-identical consecutive burst frames therefore landed in train, val and test.
- The "2,860 training images" figure included augmented copies (about 1,795 real images).

The landmark spike independently showed the same effect: a landmark-only classifier scored 0.886 (logreg) on a random split but 0.487 when trained on one session and tested on the other (`docs/results/landmark_spike_v0.md`).

## Decision

1. Treat the v0 accuracy figures as measuring memorisation, not recognition, and say so in the README next to the honest numbers.
2. All future evaluation is **subject-disjoint** (leave-one-subject-out, plus a never-touched vault of held-out subjects). Where only legacy data exists, use group-disjoint evaluation by capture session.
3. Frozen split files per dataset version; the legacy data stays as one extra source grouped by session and never decides a test result alone.
4. Report macro-F1, per-class precision/recall, confusion matrix, ECE and bootstrap confidence intervals, resampled by subject where possible.
5. Collect a multi-subject, prompted video dataset (`tools/collect`) to make this possible.

## Consequences

- Reported numbers will be much lower than 99%. That is expected and is the point of the project's story.
- A prerequisite is real data from many people, which costs recruitment effort. Minimum viable is about 10 subjects with honest confidence intervals.
- The v0 baseline is preserved (tag `v0-baseline`) so the before/after comparison is reproducible.
- The image-model group-disjoint number is produced in `docs/results/v0-baseline.md` (TBD at the time of writing).
