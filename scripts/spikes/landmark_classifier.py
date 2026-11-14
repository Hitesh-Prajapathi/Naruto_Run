"""Phase 2.2 spike, task 3: landmark-only classifier preview (early peek at E3).

Usage: python landmark_classifier.py [config]   (default full_0.50)
Reads artifacts/landmarks_v0.parquet, writes artifacts/classifier_<config>.json and,
for the confusion matrix, docs/results/img/landmark_spike_confusion_s1a_<config>.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
CFG = sys.argv[1] if len(sys.argv) > 1 else "full_0.50"
NODE_BIG, NODE_SMALL = "48f17fc25591", "0242ac1c0002"
NODET = "no_detection"
N_BOOT = 1000


def canon_hand(lm63, label, w, h):
    """Wrist origin, scale by wrist->middle-MCP, mirror left hands. Returns 63 floats."""
    pts = np.asarray(lm63, np.float32).reshape(21, 3).copy()
    pts[:, 0] *= w   # undo normalisation so x/y have equal units (z is roughly x-scaled)
    pts[:, 1] *= h
    pts[:, 2] *= w
    pts -= pts[0]
    scale = np.linalg.norm(pts[9, :2]) + 1e-6
    pts /= scale
    if label == "Left":
        pts[:, 0] *= -1
    return pts.ravel(), scale


def featurize(row):
    """78-ish dim feature vector: 2 hands x 63 + wrist vector (3) + mask (3)."""
    n = int(row["n_hands"])
    w, h = row["width"], row["height"]
    hands = []
    for k in range(n):
        vec, scale = canon_hand(row[f"h{k}_img"], row[f"h{k}_label"], w, h)
        wrist = np.array([row[f"h{k}_img"][0] * w, row[f"h{k}_img"][1] * h, row[f"h{k}_img"][2] * w])
        hands.append((row[f"h{k}_label"], wrist, vec, scale))
    if n == 2:  # order by handedness (Left first), tie-break by wrist x
        if hands[0][0] == hands[1][0]:   # same label (mislabel signal): tie-break by wrist x
            hands.sort(key=lambda t: t[1][0])
        else:                            # Left first
            hands.sort(key=lambda t: t[0] != "Left")
    f = np.zeros(63 * 2 + 3 + 3, np.float32)
    for i, (_, _, vec, _) in enumerate(hands):
        f[i * 63:(i + 1) * 63] = vec
    if n == 2:
        mean_scale = (hands[0][3] + hands[1][3]) / 2
        f[126:129] = (hands[1][1] - hands[0][1]) / mean_scale
    f[129:132] = [n >= 1, n >= 2, n]
    return f


def boot_ci(y_true, y_pred, classes, rng):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs, f1s = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, labels=sorted(set(yt)), average="macro", zero_division=0))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return q(accs), q(f1s)


def metrics(y_true, y_pred, classes, rng):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return None
    acc_ci, f1_ci = boot_ci(y_true, y_pred, classes, rng)
    return dict(n=int(len(y_true)), acc=float(accuracy_score(y_true, y_pred)), acc_ci=acc_ci,
                macro_f1=float(f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)),
                f1_ci=f1_ci,
                per_class_recall={c: float(((y_pred == c) & (y_true == c)).sum() / max((y_true == c).sum(), 1))
                                  for c in classes if (y_true == c).any()})


def make_models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, random_state=0)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                                              max_iter=600, random_state=0)),
    }


def run_split(name, df, X, tr_mask, te_mask, classes, rng, results, conf_store=None):
    """Train on tr_mask rows with >=1 hand, evaluate on te_mask rows (all + >=1 hand)."""
    has = (df["n_hands"] >= 1).values
    tr = tr_mask & has
    for mname, model in make_models().items():
        model.fit(X[tr], df["cls"].values[tr])
        te = te_mask
        pred = np.full(len(df), NODET, dtype=object)
        te_has = te & has
        pred[te_has] = model.predict(X[te_has])
        y = df["cls"].values
        res = {"all": metrics(y[te], pred[te], classes, rng),
               "ge1hand": metrics(y[te_has], pred[te_has], classes, rng),
               "n_train": int(tr.sum())}
        results[f"{name}/{mname}"] = res
        if conf_store is not None and mname == "logreg":
            conf_store[name] = (y[te].copy(), pred[te].copy())
        print(name, mname, res["all"]["acc"], res["all"]["macro_f1"],
              res["ge1hand"]["acc"] if res["ge1hand"] else None, flush=True)


def main():
    df = pd.read_parquet(REPO / "artifacts/landmarks_v0.parquet")
    df = df[df["config"] == CFG].reset_index(drop=True)
    classes = sorted(df["cls"].unique())
    X = np.stack([featurize(r) for _, r in df.iterrows()])
    rng = np.random.default_rng(42)
    results, conf = {}, {}

    nouuid = df["source_node"].isna().values
    idx = np.where(~nouuid)[0]
    tr_i, te_i = train_test_split(idx, test_size=0.2, stratify=df["cls"].values[idx], random_state=42)
    m_tr = np.zeros(len(df), bool); m_tr[tr_i] = True
    m_te = np.zeros(len(df), bool); m_te[te_i] = True
    run_split("S0_random_leaky", df, X, m_tr, m_te, classes, rng, results, conf)
    big = (df["source_node"] == NODE_BIG).values
    small = (df["source_node"] == NODE_SMALL).values
    run_split("S1a_train48f1_test0242", df, X, big, small, classes, rng, results, conf)
    run_split("S1b_train0242_test48f1", df, X, small, big, classes, rng, results, conf)

    (REPO / "artifacts").mkdir(exist_ok=True)
    (REPO / f"artifacts/classifier_{CFG}.json").write_text(json.dumps(
        dict(config=CFG, classes=classes, results=results,
             n_images=int(len(df)), n_nouuid=int(nouuid.sum()),
             n_big=int(big.sum()), n_small=int(small.sum())), indent=1))

    # confusion matrix S1a (logreg), rows = true, cols = predicted incl. no_detection
    yt, yp = conf["S1a_train48f1_test0242"]
    labels = classes + [NODET]
    cm = confusion_matrix(yt, yp, labels=labels)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=90)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"S1a (train 48f1, test 0242) logreg, {CFG}, counts")
    fig.tight_layout()
    out = REPO / f"docs/results/img/landmark_spike_confusion_s1a_{CFG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    np.savetxt(REPO / f"artifacts/confusion_s1a_{CFG}.csv", cm, fmt="%d", delimiter=",", header=",".join(labels))
    print("done", out)


if __name__ == "__main__":
    main()
