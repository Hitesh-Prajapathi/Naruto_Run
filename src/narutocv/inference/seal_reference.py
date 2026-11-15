"""Reference panel of hand-seal photos, drawn beside the guided camera evaluation.

Display only: nothing here touches recognition.  The panel shows a grid of all seals at all
times, and the target seal as a large photo only once the evaluation reveals the target (the
"PERFORM NOW" phase), so the hidden-target protocol is kept.

Photo source, per label, first hit wins:
1. ``tools/collect/public/seals/<label>.{png,jpg,jpeg,webp}`` (curated reference photos)
2. one fixed frame from ``Pure Naruto Hand Sign Data/train/<label>/`` (the 2020 dataset),
   cropped to the hands area.
Photos are mirrored to match the mirrored live view.  A missing photo becomes a text tile.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURATED_DIR = REPO_ROOT / "tools" / "collect" / "public" / "seals"
DEFAULT_DATASET_DIR = REPO_ROOT / "Pure Naruto Hand Sign Data" / "train"
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# Hands sit in the lower middle of the dataset frames (seated subject, desk-level camera).
_DATASET_CROP = (0.15, 0.30, 0.85, 1.00)  # x0, y0, x1, y1 as fractions
# Position in the sorted file list of the frame to use (deterministic, mid-session).
_DATASET_PICK = 0.6

# The dataset's "zero" class is two fists pressed together, while the evaluation prompt for
# "zero" asks for relaxed hands.  The panel says so instead of picking a side.
CAPTIONS = {"zero": "zero: dataset = fists together; prompt = relaxed hands"}

# Layout at the reference height; the panel is resized to the camera frame's height.
_REF_H = 720
_PANEL_W = 504
_COLS = 4
_LARGE_H = 283
_TILE_LABEL_H = 18
_GREEN = (70, 255, 100)
_GREY = (90, 90, 90)
_TEXT = (235, 235, 235)


def _load_curated(label: str, curated_dir: Path) -> Optional[np.ndarray]:
    for suffix in _IMAGE_SUFFIXES:
        path = curated_dir / f"{label}{suffix}"
        if path.is_file():
            image = cv2.imread(str(path))
            if image is not None:
                return image
    return None


def _load_dataset(label: str, dataset_dir: Path) -> Optional[np.ndarray]:
    folder = dataset_dir / label
    if not folder.is_dir():
        return None
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not files:
        return None
    image = cv2.imread(str(files[min(len(files) - 1, int(len(files) * _DATASET_PICK))]))
    if image is None:
        return None
    h, w = image.shape[:2]
    x0, y0, x1, y1 = _DATASET_CROP
    return image[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]


def load_reference_images(
    labels: Iterable[str],
    *,
    curated_dir: Path = DEFAULT_CURATED_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, Optional[np.ndarray]]:
    """Mirrored reference photo per label (``None`` when no photo is found)."""
    images: dict[str, Optional[np.ndarray]] = {}
    for label in labels:
        image = _load_curated(label, curated_dir)
        if image is None:
            image = _load_dataset(label, dataset_dir)
        images[label] = None if image is None else cv2.flip(image, 1)
    return images


def _fit(image: Optional[np.ndarray], width: int, height: int, fallback: str) -> np.ndarray:
    """Letterbox ``image`` into a ``width`` x ``height`` tile (text tile if ``None``)."""
    tile = np.full((height, width, 3), 30, dtype=np.uint8)
    if image is None:
        cv2.putText(tile, fallback, (6, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1)
        return tile
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    x, y = (width - nw) // 2, (height - nh) // 2
    tile[y : y + nh, x : x + nw] = resized
    return tile


def _put(panel: np.ndarray, text: str, org: tuple[int, int], scale: float, color: Any) -> None:
    cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class SealReferencePanel:
    """Builds the side panel; tiles are rendered once, the per-frame cost is a copy + resize."""

    def __init__(self, labels: Sequence[str], images: dict[str, Optional[np.ndarray]]):
        self.labels = tuple(labels)
        self.images = images
        self._tile_w = _PANEL_W // _COLS
        self._tile_h = int(self._tile_w * 9 / 16)
        self._tiles = {
            label: _fit(images.get(label), self._tile_w - 4, self._tile_h, label)
            for label in self.labels
        }
        self._large = {
            label: _fit(images.get(label), _PANEL_W, _LARGE_H, label) for label in self.labels
        }

    @classmethod
    def from_disk(cls, labels: Sequence[str], **kwargs: Path) -> SealReferencePanel:
        return cls(labels, load_reference_images(labels, **kwargs))

    def render(
        self,
        height: int,
        *,
        highlight: Sequence[str] = (),
        large_label: Optional[str] = None,
        hidden_note: str = "",
    ) -> np.ndarray:
        """Panel image of the given ``height``.

        ``highlight`` labels get a green frame in the grid; ``large_label`` (if any) is shown
        as the large photo, otherwise ``hidden_note`` is written in that area.
        """
        panel = np.full((_REF_H, _PANEL_W, 3), 12, dtype=np.uint8)
        _put(panel, "HAND SEAL REFERENCE (mirrored like your view)", (8, 20), 0.5, _TEXT)
        top = 28
        if large_label is not None and large_label in self._large:
            panel[top : top + _LARGE_H, :] = self._large[large_label]
            cv2.rectangle(panel, (1, top), (_PANEL_W - 2, top + _LARGE_H - 1), _GREEN, 2)
            caption = CAPTIONS.get(large_label, large_label.upper())
            _put(panel, caption, (8, top + _LARGE_H + 18), 0.55, _GREEN)
        else:
            cv2.rectangle(panel, (1, top), (_PANEL_W - 2, top + _LARGE_H - 1), _GREY, 1)
            if hidden_note:
                _put(panel, hidden_note, (16, top + _LARGE_H // 2), 0.55, _TEXT)
        y0 = top + _LARGE_H + 28
        row_h = self._tile_h + _TILE_LABEL_H
        marked = set(highlight)
        for index, label in enumerate(self.labels):
            row, col = divmod(index, _COLS)
            x, y = col * self._tile_w + 2, y0 + row * row_h
            if y + row_h > _REF_H:
                break
            panel[y : y + self._tile_h, x : x + self._tile_w - 4] = self._tiles[label]
            on = label in marked
            if on:
                cv2.rectangle(
                    panel, (x, y), (x + self._tile_w - 5, y + self._tile_h - 1), _GREEN, 2
                )
            _put(panel, label, (x + 2, y + self._tile_h + 13), 0.45, _GREEN if on else _TEXT)
        if height == _REF_H:
            return panel
        width = max(1, round(_PANEL_W * height / _REF_H))
        return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)

    def attach(self, frame: np.ndarray, **kwargs: Any) -> np.ndarray:
        """``frame`` with the panel concatenated on its right."""
        return np.hstack([frame, self.render(frame.shape[0], **kwargs)])
