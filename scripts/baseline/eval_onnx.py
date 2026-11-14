"""Reference evaluation of the DEPLOYED best_model_A.onnx (CONTAMINATED).

The deployed model was trained on a random split of *all* sources, so any image
from the raw dataset may have been a training image.  These numbers are NOT an
honest generalisation estimate; they exist only to (a) reproduce the claimed
val/test accuracy as a preprocessing self-check and (b) sit beside the honest
retrained-model numbers.

Runtime preprocessing replicated from
``cv_model/inference/combined_pipeline.py::_classify_crop_timed``:
BGR->RGB, cv2.resize(224, INTER_AREA), /255, CHW, no mean/std, softmax only if
the output does not already sum to 1.

Pipelines:
  png224   input is a pre-made 224 PNG (PIL centre-crop + LANCZOS = the training
           pipeline); then the runtime tensor math (resize 224->224 is a no-op).
  rawcrop  raw image -> centre square crop -> runtime math (cv2 INTER_AREA 224),
           i.e. what the runtime does to a camera crop.

    PYTHONPATH=src python scripts/baseline/eval_onnx.py --raw-root "$RAW_ROOT" --prepared "$PREPARED_ROOT"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from narutocv.eval.metrics import bootstrap_ci, compute_metrics, confusion_matrix_png, worst_classes

CLASSES = [
    "bird",
    "boar",
    "dog",
    "dragon",
    "hare",
    "horse",
    "monkey",
    "ox",
    "ram",
    "rat",
    "snake",
    "tiger",
    "zero",
]
MODEL = Path("cv_model/models/best_model_A.onnx")


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def classify_bgr(session, in_name, out_name, crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    out = np.asarray(session.run([out_name], {in_name: tensor})[0]).reshape(-1)
    return out if np.isclose(out.sum(), 1.0, atol=1e-3) else softmax(out)


def centre_square(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return img_bgr[top : top + s, left : left + s]


def run(session, paths, mode: str) -> np.ndarray:
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    probs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # BGR, 3-channel (drops alpha)
        if mode == "rawcrop":
            img = centre_square(img)
        probs.append(classify_bgr(session, in_name, out_name, img))
    return np.stack(probs)


def summarise(name, y, probs, note) -> dict:
    m = compute_metrics(y, probs, CLASSES)
    ci = bootstrap_ci(y, probs, len(CLASSES), n_boot=1000, seed=0)
    return {"name": name, "note": note, "metrics": m, "ci95": ci, "worst3": worst_classes(m, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("artifacts/splits"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/onnx_ref"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    n_out = session.get_outputs()[0].shape
    print("onnx output shape", n_out, "classes", len(CLASSES))

    results = []
    # 1. self-check on the prepared val/test PNGs (what the model was trained/validated on)
    for split in ("val", "test"):
        paths, y = [], []
        for ci_, c in enumerate(CLASSES):
            for f in sorted((a.prepared / split / c).glob("*.png")):
                paths.append(f)
                y.append(ci_)
        probs = run(session, paths, "png224")
        np.save(a.out / f"prepared_{split}_probs.npy", probs)
        results.append(
            summarise(
                f"prepared_{split}",
                np.array(y),
                probs,
                "claimed-split reproduction; leaky random frame split",
            )
        )
    # 2. S0 / S1a test sets, both pipelines
    for sp in ("s0", "s1a"):
        test = pd.read_csv(a.splits / sp / "test.csv")
        y = np.array([CLASSES.index(c) for c in test["cls"]])
        for mode, paths in (
            ("png224", list(test["cache_path"])),
            ("rawcrop", [a.raw_root / p for p in test["path"]]),
        ):
            probs = run(session, paths, mode)
            np.save(a.out / f"{sp}_{mode}_probs.npy", probs)
            np.save(a.out / f"{sp}_labels.npy", y)
            res = summarise(
                f"{sp}_test_{mode}", y, probs, "CONTAMINATED: model trained on all sources"
            )
            results.append(res)
            if mode == "png224":
                confusion_matrix_png(
                    res["metrics"]["confusion_matrix"],
                    CLASSES,
                    a.out / f"{sp}_onnx_cm.png",
                    f"deployed ONNX (contaminated) on {sp} test",
                )
    (a.out / "onnx_results.json").write_text(json.dumps(results, indent=2))
    for r in results:
        m = r["metrics"]
        print(
            f"{r['name']:24s} n={m['n']:5d} acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} "
            f"ece={m['ece']:.4f}"
        )


if __name__ == "__main__":
    main()
