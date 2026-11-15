"""Phase 2.2 spike, task 2: per-class detectability tables + overlay montages.

Writes artifacts/detectability.md (tables) and docs/results/img/landmark_spike_overlay_<class>.jpg.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "Pure Naruto Hand Sign Data"
PRIMARY = "full_0.50"
BIG, SMALL = "48f17fc25591", "0242ac1c0002"
CONNECT = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def stats(g):
    n = len(g)
    scores = pd.concat([g["h0_score"], g["h1_score"]]).dropna()
    two = g[g["n_hands"] == 2]
    same = (two["h0_label"] == two["h1_label"]).mean() * 100 if len(two) else np.nan
    return pd.Series(
        dict(
            n=n,
            p0=(g.n_hands == 0).mean() * 100,
            p1=(g.n_hands == 1).mean() * 100,
            p2=(g.n_hands == 2).mean() * 100,
            med_score=scores.median() if len(scores) else np.nan,
            same=same,
            n2=len(two),
        )
    )


def fmt(v, d=1):
    return "-" if pd.isna(v) else f"{v:.{d}f}"


def table(df, title):
    rows = [
        "| class | n | 0 hands % | 1 hand % | 2 hands % | median handedness score | same-label % (of 2-hand) |",
        "|---|---|---|---|---|---|---|",
    ]
    s = df.groupby("cls").apply(stats, include_groups=False)
    for c, r in s.iterrows():
        rows.append(
            f"| {c} | {int(r.n)} | {fmt(r.p0)} | {fmt(r.p1)} | {fmt(r.p2)} | {fmt(r.med_score, 3)} | "
            f"{fmt(r.same)} ({int(r.n2)}) |"
        )
    t = stats(df)
    rows.append(
        f"| **all** | {int(t.n)} | {fmt(t.p0)} | {fmt(t.p1)} | {fmt(t.p2)} | {fmt(t.med_score, 3)} | "
        f"{fmt(t.same)} ({int(t.n2)}) |"
    )
    return f"**{title}**\n\n" + "\n".join(rows) + "\n"


def draw(row, size=320):
    img = cv2.imread(str(DATA / row["path"]))
    h, w = img.shape[:2]
    pts_all = []
    for k in range(int(row["n_hands"])):
        lm = np.array(row[f"h{k}_img"]).reshape(21, 3)
        pts = np.stack([lm[:, 0] * w, lm[:, 1] * h], 1)
        pts_all.append(pts)
        col = (0, 200, 0) if k == 0 else (0, 140, 255)
        for a, b in CONNECT:
            cv2.line(
                img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), col, max(2, w // 300)
            )
        for p in pts:
            cv2.circle(img, tuple(p.astype(int)), max(3, w // 200), (0, 0, 255), -1)
    if pts_all:  # tight crop around the hands
        allp = np.concatenate(pts_all)
        x0, y0 = allp.min(0)
        x1, y1 = allp.max(0)
        side = max(x1 - x0, y1 - y0) * 1.5 + 20
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    else:  # nothing detected: central half of the frame
        side, cx, cy = min(h, w) * 0.6, w / 2, h / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    cv2.putText(
        crop,
        f"{int(row['n_hands'])} hands",
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return crop


def main():
    df = pd.read_parquet(REPO / "artifacts/landmarks_v0.parquet")
    out = []
    for cfg in df["config"].unique():
        d = df[df["config"] == cfg]
        if cfg == PRIMARY:
            out.append(f"### Config {cfg}: all images\n\n" + table(d, "all sources"))
            out.append(table(d[d.source_node == BIG], f"source node {BIG} (burst session)"))
            out.append(
                table(
                    d[d.source_node == SMALL],
                    f"source node {SMALL} (4-day session, incl. original test set)",
                )
            )
        t = stats(d)
        out.append(
            f"CONFIG_SUMMARY {cfg}: n={int(t.n)} 0={t.p0:.1f}% 1={t.p1:.1f}% 2={t.p2:.1f}% "
            f"median_score={t.med_score:.3f} same_label={t.same:.1f}% of {int(t.n2)} two-hand images\n"
        )
    # per-config per-class 2-hand % compact comparison
    comp = (
        df.groupby(["cls", "config"])
        .apply(lambda g: (g.n_hands == 2).mean() * 100, include_groups=False)
        .unstack()
    )
    out.append(
        "### 2-hand detection rate (%) by class and config\n\n" + comp.round(1).to_markdown() + "\n"
    )
    (REPO / "artifacts/detectability.md").write_text("\n".join(out))

    # overlays for the 5 classes with lowest 2-hand rate (primary config)
    p = df[df["config"] == PRIMARY]
    worst = (
        p.groupby("cls")
        .apply(lambda g: (g.n_hands == 2).mean(), include_groups=False)
        .sort_values()
        .index[:5]
    )
    print("worst:", list(worst))
    img_dir = REPO / "docs/results/img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for c in worst:
        g = p[p.cls == c]
        bad = g[g.n_hands < 2]
        pick = pd.concat(
            [
                bad.sample(min(2, len(bad)), random_state=0),
                g[g.n_hands == 2].sample(min(1, (g.n_hands == 2).sum()), random_state=0),
            ]
        )
        if len(pick) < 3:
            pick = pd.concat([pick, g.drop(pick.index).sample(3 - len(pick), random_state=1)])
        tiles = [draw(r) for _, r in pick.iterrows()]
        cv2.imwrite(
            str(img_dir / f"landmark_spike_overlay_{c}.jpg"),
            np.hstack(tiles),
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
    print((REPO / "artifacts/detectability.md").read_text())


if __name__ == "__main__":
    main()
