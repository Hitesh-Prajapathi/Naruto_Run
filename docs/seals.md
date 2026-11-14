# Seal specification

> **STATUS: DRAFT, NEEDS OWNER VERIFICATION.** Every description below comes from `tools/collect/src/content/seals.json`, which was written *without reference photos*. The project owner must verify each entry against canonical reference photos, choose ONE variant per seal, and fill in the photo placeholders. **The dragon variant has not been chosen at all.** Left/right are from the performer's point of view.

This is the intended single source of truth for what each seal looks like. Collection subjects are shown the same text and photo ([`tools/collect`](../tools/collect/README.md)). The detectability notes are measured, from the [landmark spike](results/landmark_spike_v0.md): a proxy run on the legacy 2,245 images (essentially one subject, two sessions) using MediaPipe HandLandmarker at full resolution, threshold 0.50. They are a de-risking signal, not final.

Reading the numbers: "0 / 1 / 2 hands" is the percentage of images where the landmarker returned that many hands. "S1a recall" is the landmark-only logistic-regression recall when trained on the large burst session and tested on the other session (the realistic direction); all images, no-detection counted as wrong.

Class list (13): bird, boar, dog, dragon, hare, horse, monkey, ox, ram, rat, snake, tiger, zero (`zero` means "no seal").

## Open questions for the owner

- [ ] Verify every description against reference photos (front and side).
- [ ] **Pick one dragon variant** and describe it finger by finger. (Dragon was never detected in 2 live runs, 0/3 each, and the spike shows detectability is not the cause, which points to a spec/data mismatch.)
- [ ] Record the left/right convention and handedness so mirror augmentation can be validated per class.
- [ ] Confirm the confusable pairs below (ram vs rat, dog vs hare, tiger vs rat) and add any others.
- [ ] Provide photos as `tools/collect/public/seals/<seal>.jpg` and link them here.

## bird

- **Description (needs owner verification against reference photos; status: draft):** Hands in front of the chest, one palm flat facing down, the other hand's fingertips pointing up beneath it, thumbs spread wide so the thumbs and index fingers make an open beak-like shape.
- **Confusable with:** none listed (needs owner review)
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 2.1 / 9.9 / 88.0 %; same handedness label on both hands: 0.6 %; S1a landmark-only recall 0.29.
- **Spike note:** landmark-friendly in the burst source (98.2% two-hand) but only 32.1% two-hand in the second source; not safe across sessions.
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/bird.jpg`._

## boar

- **Description (needs owner verification against reference photos; status: draft):** Both hands curled into loose fists held low in front of the chest, backs of the hands facing outward, knuckles close together.
- **Confusable with:** none listed (needs owner review)
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 20.9 / 62.6 / 16.5 %; same handedness label on both hands: 20.0 %; S1a landmark-only recall 0.81.
- **Spike note:** image-led; single-hand landmarks still carry signal (S1a recall 0.81), keep as a weak input.
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/boar.jpg`._

## dog

- **Description (needs owner verification against reference photos; status: draft):** One hand flat, palm down, over the other hand; the lower hand's fingers extend forward while the upper hand's thumb and fingers cover it. Fingers together, no gaps.
- **Confusable with:** hare
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 7.7 / 54.9 / 37.4 %; same handedness label on both hands: 2.9 %; S1a landmark-only recall 0.40.
- **Spike note:** image-led; mostly one hand, 48.8% no hand in the second source.
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/dog.jpg`._

## dragon (**variant NOT chosen**)

- **Description (needs owner verification against reference photos; status: tbd):** TBD by project owner. The dragon variant has not been pinned yet (the live model never detects it). Owner: choose ONE variant, describe it here finger by finger, and add public/seals/dragon.jpg.
- **Confusable with:** none listed (needs owner review)
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 2.6 / 26.3 / 71.2 %; same handedness label on both hands: 65.8 %; S1a landmark-only recall 0.68.
- **Spike note:** both branches; handedness is unreliable (two hands share a label 65.8% of the time, 71.3% in the burst source), so order hands by x, not by label.
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/dragon.jpg`._

## hare

- **Description (needs owner verification against reference photos; status: draft):** Hands stacked in front of the chest: the lower hand flat palm up, the upper hand's index and middle fingers extended forward over it, remaining fingers folded, thumbs tucked.
- **Confusable with:** dog
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 8.2 / 27.7 / 64.1 %; same handedness label on both hands: 0.0 %; S1a landmark-only recall 0.33.
- **Spike note:** both branches; collapses on the 224 centre crop (64% to 2% two-hand).
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/hare.jpg`._

## horse

- **Description (needs owner verification against reference photos; status: draft):** Palms pressed together in front of the chest, both index fingers extended and pressed together pointing up, other fingers curled over each other.
- **Confusable with:** none listed (needs owner review)
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 0.0 / 0.6 / 99.4 %; same handedness label on both hands: 0.0 %; S1a landmark-only recall 0.88.
- **Spike note:** landmark-led; the only seal that stays high in both sources (97.1% in the second).
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/horse.jpg`._

## monkey

- **Description (needs owner verification against reference photos; status: draft):** One hand's palm presses flat on the back of the other hand, fingers together and pointing forward, forearms roughly horizontal.
- **Confusable with:** none listed (needs owner review)
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 7.8 / 14.2 / 78.0 %; same handedness label on both hands: 0.9 %; S1a landmark-only recall 0.03.
- **Spike note:** both branches; S1a recall is 0.03, so landmarks alone fail across sessions.
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/monkey.jpg`._

## ox

- **Description (needs owner verification against reference photos; status: draft):** One hand held vertically with fingers pointing up and the other held horizontally, fingers pointing sideways, the two forming a cross shape at the chest.
- **Confusable with:** none listed (needs owner review)
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 7.4 / 80.1 / 12.5 %; same handedness label on both hands: 9.1 %; S1a landmark-only recall 0.23.
- **Spike note:** image-led; the 224 crop raised two-hand detection from 12.5% to 71.6% (hypothesis: palm-detector scale sensitivity, untested).
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/ox.jpg`._

## ram

- **Description (needs owner verification against reference photos; status: draft):** One hand pointing forward with index and middle fingers extended and together, the other hand wrapping around it from beneath, thumbs overlapping; ring and little fingers curled.
- **Confusable with:** rat
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 2.5 / 94.3 / 3.3 %; same handedness label on both hands: 0.0 %; S1a landmark-only recall 0.54.
- **Spike note:** image-led; interlocked fingers read as a single merged hand (94.3% one hand); confused with bird and tiger in S1a.
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/ram.jpg`._

## rat

- **Description (needs owner verification against reference photos; status: draft):** Hands together at the chest with fingers interlaced and curled; the two index fingers rest against each other pointing forward, thumbs side by side.
- **Confusable with:** ram, tiger
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 0.0 / 49.2 / 50.8 %; same handedness label on both hands: 1.5 %; S1a landmark-only recall 0.43.
- **Spike note:** both branches; 12 of 30 test rat frames in S1a go to snake, so the image branch is needed even when landmarks exist.
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/rat.jpg`._

## snake

- **Description (needs owner verification against reference photos; status: draft):** Palms together at the chest, fingers interlaced and pointing forward, thumbs resting on top of each other, forearms level.
- **Confusable with:** none listed (needs owner review)
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 5.3 / 35.8 / 58.9 %; same handedness label on both hands: 3.4 %; S1a landmark-only recall 0.60.
- **Spike note:** both branches; in S1a, rat (12 frames) and tiger (9) are often predicted as snake.
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/snake.jpg`._

## tiger

- **Description (needs owner verification against reference photos; status: draft):** Palms together, index and middle fingers of both hands extended up and pressed together; ring and little fingers curled and interlocked, thumbs against the curled fingers.
- **Confusable with:** rat
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 42.4 / 52.3 / 5.2 %; same handedness label on both hands: 11.1 %; S1a landmark-only recall 0.59.
- **Spike note:** image-led; 42.4% of images have no hand detected at all (50.8% in the burst source).
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/tiger.jpg`._

## zero

- **Description (needs owner verification against reference photos; status: draft):** No seal. Let both hands rest naturally in your lap or at your sides, relaxed and not touching.
- **Confusable with:** none listed (needs owner review)
- **Landmark detectability (spike, all sources):** 0 / 1 / 2 hands = 1.5 / 4.9 / 93.6 %; same handedness label on both hands: 0.0 %; S1a landmark-only recall 0.50.
- **Spike note:** landmark-friendly in the burst source (100% two-hand) but 50% in the second source; the "no seal" class.
- **Reference photo:** _placeholder, to be added as `tools/collect/public/seals/zero.jpg`._
