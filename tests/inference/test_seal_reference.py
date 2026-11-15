from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from narutocv.inference.seal_reference import SealReferencePanel, load_reference_images

LABELS = ("bird", "dragon", "zero")


def _write(path: Path, value: int, size: tuple[int, int] = (64, 36)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    image[:, : size[0] // 2] = 255 - value  # left half differs, so mirroring is visible
    cv2.imwrite(str(path), image)


def test_curated_photo_wins_over_dataset_and_missing_is_none(tmp_path: Path) -> None:
    curated, dataset = tmp_path / "curated", tmp_path / "dataset"
    _write(curated / "bird.png", 10)
    _write(dataset / "bird" / "a.png", 200)
    _write(dataset / "dragon" / "a.png", 100, size=(100, 100))
    images = load_reference_images(LABELS, curated_dir=curated, dataset_dir=dataset)
    assert images["zero"] is None
    assert images["bird"] is not None and images["bird"].shape[:2] == (36, 64)
    # Mirrored: the originally-left (255 - value) half is now on the right.
    assert images["bird"][0, -1, 0] == 245 and images["bird"][0, 0, 0] == 10
    # Dataset frames are cropped to the hands area (x 15-85 %, y 30-100 %).
    assert images["dragon"] is not None and images["dragon"].shape[:2] == (70, 70)


def test_render_shapes_and_attach() -> None:
    images = {"bird": np.zeros((36, 64, 3), np.uint8), "dragon": None, "zero": None}
    panel = SealReferencePanel(LABELS, images)
    assert panel.render(720).shape == (720, 504, 3)
    assert panel.render(360).shape == (360, 252, 3)
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = panel.attach(frame, highlight=("dragon",), large_label="dragon")
    assert out.shape == (720, 1784, 3)
    # Hidden and revealed renders differ (the large target photo only shows when revealed).
    hidden = panel.render(720, hidden_note="hidden")
    shown = panel.render(720, highlight=("bird",), large_label="bird")
    assert not np.array_equal(hidden, shown)
