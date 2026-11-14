"""Offline augmentation used to balance classes.

Re-implementation of ``apply_augmentation`` from ``cv_model/data/prepare_dataset.py``
(same operations and parameter ranges) but driven by an explicit seeded RNG, so
every split-building run is reproducible.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

BALANCE_TARGET = 220  # per-class train size used by prepare_dataset.py


def apply_augmentation(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    # 1. random rotation in [-15, 15] degrees
    img = img.rotate(
        float(rng.uniform(-15, 15)), resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0)
    )
    # 2. colour jitter (p = 0.8)
    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.7, 1.3)))
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.8, 1.2)))
        img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.8, 1.2)))
    # 3. gaussian blur (p = 0.3)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.1, 1.0))))
    # 4. random zoom-crop, scale in [0.85, 1.0]
    scale = float(rng.uniform(0.85, 1.0))
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    left = int(rng.integers(0, w - new_w + 1))
    top = int(rng.integers(0, h - new_h + 1))
    img = img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.Resampling.LANCZOS)
    # 5. additive gaussian noise (p = 0.2)
    if rng.random() < 0.2:
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0, rng.uniform(0.01, 0.03), arr.shape), 0.0, 1.0) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))
    return img
