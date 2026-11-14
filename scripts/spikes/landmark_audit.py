"""Phase 2.2 spike: HandLandmarker detectability audit on the existing dataset.

Writes artifacts/landmarks_v0.parquet (long format: one row per image x config).
Configs: full-res @0.5, full-res @0.45, 224 centre-crop @0.5, 224 centre-crop @0.45.
"""
import re
import sys
import uuid
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ROOT = Path("/Users/hiteshprajapathi/Desktop/Naruto_Run")
DATA = ROOT / "Pure Naruto Hand Sign Data"
MODEL = ROOT / "cv_model/models/hand_landmarker.task"
OUT = Path(__file__).resolve().parents[2] / "artifacts/landmarks_v0.parquet"
UUID_RE = re.compile(r"([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})")
CONFIGS = [("full_0.50", "full", 0.5), ("full_0.45", "full", 0.45),
           ("crop224_0.50", "crop", 0.5), ("crop224_0.45", "crop", 0.45)]


def session_info(name):
    """Return (source_node, capture_date, session_id) from the UUID1 in a filename."""
    m = UUID_RE.search(name)
    if not m:
        return None, None, None
    try:
        u = uuid.UUID("-".join(m.groups()))
        ts = (u.time - 0x01B21DD213814000) / 1e7
        date = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None, None, None
    node = m.group(5)
    return node, date, f"{node}_{date}"


def make_landmarker(conf):
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=conf, min_hand_presence_confidence=conf,
        min_tracking_confidence=conf)
    return vision.HandLandmarker.create_from_options(opts)


def center_crop224(rgb):
    h, w = rgb.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(rgb[y:y + s, x:x + s], (224, 224), interpolation=cv2.INTER_AREA)


def main():
    files = sorted(p for p in DATA.glob("*/*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(len(files), "images", flush=True)
    lms = {c[0]: make_landmarker(c[2]) for c in CONFIGS}
    rows = []
    for i, p in enumerate(files):
        bgr = cv2.imread(str(p))
        node, date, sid = session_info(p.name)
        base = dict(path=str(p.relative_to(DATA)), split=p.parent.parent.name, cls=p.parent.name,
                    source_node=node, capture_date=date, session_id=sid)
        if bgr is None:
            print("unreadable", p, file=sys.stderr)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop = center_crop224(rgb)
        for name, mode, _ in CONFIGS:
            img = rgb if mode == "full" else crop
            res = lms[name].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
            row = dict(base, config=name, height=img.shape[0], width=img.shape[1], n_hands=len(res.hand_landmarks))
            for k in range(2):
                if k < len(res.hand_landmarks):
                    cat = res.handedness[k][0]
                    row[f"h{k}_label"] = cat.category_name
                    row[f"h{k}_score"] = float(cat.score)
                    row[f"h{k}_img"] = np.array([[l.x, l.y, l.z] for l in res.hand_landmarks[k]], np.float32).ravel().tolist()
                    row[f"h{k}_world"] = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[k]], np.float32).ravel().tolist()
                else:
                    row[f"h{k}_label"] = None
                    row[f"h{k}_score"] = np.nan
                    row[f"h{k}_img"] = None
                    row[f"h{k}_world"] = None
            rows.append(row)
        if i % 200 == 0:
            print(i, flush=True)
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
