"""Train yolov8n-cls (notebook recipe) on one materialised split and predict its test set.

    PYTHONPATH=src python scripts/baseline/train_split.py s0 [s1a ...] [--epochs 50]

Outputs to artifacts/runs/<split>/: ultralytics run dir, test_probs.npy,
test_labels.npy, test_paths.json and run_meta.json.
Recipe = cv_model/training/V1_Model_training.ipynb approach A (epochs=50, imgsz=224,
batch=64, patience=10, lr0=0.01, lrf=0.01, warmup_epochs=3, weight_decay=5e-4,
label_smoothing=0.1, flips off), run on one Apple MPS device instead of 2xT4 DDP.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
ROOT = Path("artifacts").resolve()
WEIGHTS = Path("cv_model/models/pretrained/yolov8n-cls.pt").resolve()


def run_one(name: str, epochs: int, device: str, project: Path) -> dict:
    import torch
    import ultralytics
    from ultralytics import YOLO

    split_dir = ROOT / "splits" / name
    run_dir = project / name
    model = YOLO(str(WEIGHTS))
    args = dict(
        data=str(split_dir / "data"),
        epochs=epochs,
        imgsz=224,
        batch=64,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        patience=10,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        label_smoothing=0.1,
        flipud=0.0,
        fliplr=0.0,
        pretrained=True,
        seed=42,
        deterministic=False,
        plots=False,
        workers=4,
        verbose=False,
    )
    t0 = time.time()
    model.train(**args)
    train_s = time.time() - t0

    best = run_dir / "weights" / "best.pt"
    results_csv = pd.read_csv(run_dir / "results.csv")
    results_csv.columns = [c.strip() for c in results_csv.columns]
    acc_col = "metrics/accuracy_top1"
    best_epoch = int(results_csv[acc_col].idxmax()) + 1

    # predict test set
    test = pd.read_csv(split_dir / "test.csv")
    m = YOLO(str(best))
    names = m.names  # idx -> class
    assert [names[i] for i in range(len(CLASSES))] == CLASSES, names
    probs = []
    paths = list(test["cache_path"])
    for i in range(0, len(paths), 64):
        for r in m.predict(paths[i : i + 64], imgsz=224, device=device, verbose=False):
            probs.append(r.probs.data.cpu().numpy())
    probs = np.stack(probs)
    labels = np.array([CLASSES.index(c) for c in test["cls"]])
    np.save(run_dir / "test_probs.npy", probs)
    np.save(run_dir / "test_labels.npy", labels)
    (run_dir / "test_paths.json").write_text(json.dumps(list(test["path"])))
    meta = {
        "split": name,
        "epochs_requested": epochs,
        "epochs_completed": int(len(results_csv)),
        "best_epoch_by_val_top1": best_epoch,
        "best_val_top1": float(results_csv[acc_col].max()),
        "train_wall_seconds": round(train_s, 1),
        "device": device,
        "test_accuracy": float((probs.argmax(1) == labels).mean()),
        "n_test": int(len(labels)),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "train_args": args,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("splits", nargs="+")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--project", type=Path, default=ROOT / "runs")
    a = p.parse_args()
    os.environ.setdefault("YOLO_OFFLINE", "1")
    device = a.device
    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    for name in a.splits:
        meta = run_one(name, a.epochs, device, a.project.resolve())
        print(json.dumps({k: v for k, v in meta.items() if k != "train_args"}))


if __name__ == "__main__":
    main()
