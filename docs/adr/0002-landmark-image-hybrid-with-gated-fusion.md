# 0002. Landmark + image hybrid with gated fusion

Status: Proposed (design validated only on a proxy; implementation pending)

## Context

Root causes of the v0 failures include a whole-frame classifier that learned the 2020 room rather than the hands, and seals (ram vs rat, dog vs hare) that differ in finger topology more than in pixels. Hand landmarks are independent of background, lighting and skin tone and should separate such pairs directly.

The landmark spike (`docs/results/landmark_spike_v0.md`, proxy run on the legacy 2,245 images, one subject and two sessions) tested whether MediaPipe can see both hands:

- Overall only 52.7% of images get 2 hands, 38.8% get 1, 8.6% get none.
- Landmark-friendly: horse (99.4% two-hand), zero (93.6%), bird (88%), though in the second source only horse stays high (97%).
- Middle: monkey 78%, dragon 71%, hare 64%, snake 59%, rat 51%. Dragon's two detected hands carry the same handedness label 65.8% of the time, so handedness cannot be used for slot ordering.
- Interlocked seals fail: ram 3.3% two-hand (94% one merged hand), tiger 5.2% (42.4% no hand), ox 12.5%, boar 16.5% (21% none), dog 37.4%.
- A landmark-only classifier loses about 40 points between a random split (0.886) and cross-session (0.487), partly from missing hands.

## Decision

Build a hybrid, with neither branch trusted alone:

- **[A] Landmark branch:** canonicalised two-hand landmarks, joint-angle and inter-hand-distance features, small MLP, trained with hand-dropout augmentation that mimics the measured detection rates.
- **[B] Image branch:** a classifier on a two-hand union crop (pose-wrist fallback when fewer than 2 hands), trained with background replacement and crop-jitter augmentation.
- **Gated (stacked) fusion:** a logistic-regression stacker over both branches' outputs plus `hand_count`, handedness scores and landmark quality, so that the fusion leans on the image branch when landmarks are missing.
- Expected roles from the proxy: landmark-led for horse (and maybe zero, bird); both branches for dragon, monkey, hare, snake, rat; image-led for ram, tiger, ox, boar, dog.

## Consequences

- More moving parts: two models, a stacker, and a feature spec that must match between Python and TS.
- The spike is a proxy. The real multi-subject recording must re-measure per-person, per-seal detection rates, jitter and the crop-scale effect before the fusion weights are trusted.
- Cross-session comparison against the image model is still open; experiments E1 vs E3 in the plan answer it.
- The dragon 0/3 live failure is not explained by detectability, so it also depends on the canonical seal spec ([docs/seals.md](../seals.md)).
