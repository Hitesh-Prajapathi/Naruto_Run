"""Classification metrics reused across all phases (pure numpy + matplotlib).

Conventions
-----------
* ``y_true`` / ``y_pred`` are integer class indices in ``range(n_classes)``.
* Macro-F1 / macro-precision / macro-recall average over classes with
  **support > 0** (classes absent from ``y_true`` are excluded), so held-out
  sessions that lack some classes are scored on the classes they contain.
  Per-class P/R/F1 use 0 when a denominator is 0.
* ECE uses ``n_bins`` equal-width confidence bins with edges
  ``linspace(0, 1, n_bins + 1)`` and the convention ``(lo, hi]``; a confidence
  of exactly 0 falls into bin 0.  Confidence = max probability, correct =
  (argmax == y_true).  ECE = sum_b (n_b / N) * |acc_b - conf_b|.
* Bootstrap CIs resample *within each true class* (stratified), with replacement,
  keeping class counts fixed; seeded; percentile 95% interval.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_prf(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    pred = cm.sum(axis=0).astype(float)
    support = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support.astype(int)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1_from_cm(cm: np.ndarray) -> float:
    _, _, f1, support = per_class_prf(cm)
    return float(f1[support > 0].mean())


def ece_bins(probs: np.ndarray, y_true, n_bins: int = 15):
    """Return (ece, per-bin table dict) with the (lo, hi] convention."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    n = len(conf)
    counts = np.bincount(idx, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    nz = counts > 0
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_conf[nz] = conf_sum[nz] / counts[nz]
    bin_acc[nz] = acc_sum[nz] / counts[nz]
    ece = float(np.sum(counts / n * np.abs(bin_acc - bin_conf)))
    return ece, {"edges": edges, "counts": counts, "conf": bin_conf, "acc": bin_acc}


def ece(probs, y_true, n_bins: int = 15) -> float:
    return ece_bins(probs, y_true, n_bins)[0]


def compute_metrics(y_true, probs, class_names: list[str], n_bins: int = 15) -> dict:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = probs.argmax(axis=1)
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    p, r, f1, support = per_class_prf(cm)
    present = support > 0
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": float(f1[present].mean()),
        "macro_precision": float(p[present].mean()),
        "macro_recall": float(r[present].mean()),
        "ece": ece(probs, y_true, n_bins),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                "support": int(support[i])}
            for i, c in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }


def worst_classes(metrics: dict, k: int = 3) -> list[tuple[str, float, int]]:
    """k lowest-F1 classes among those with support > 0: (name, f1, support)."""
    items = [(c, v["f1"], v["support"]) for c, v in metrics["per_class"].items() if v["support"] > 0]
    return sorted(items, key=lambda t: (t[1], t[0]))[:k]


def bootstrap_ci(
    y_true, probs, n_classes: int, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
    n_bins: int = 15,
) -> dict:
    """Stratified (per true class) bootstrap percentile CIs for accuracy, macro-F1, ECE."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y_true == c) for c in range(n_classes)]
    groups = [g for g in groups if len(g)]
    pred = probs.argmax(axis=1)
    stats = {"accuracy": [], "macro_f1": [], "ece": []}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        cm = confusion_matrix(y_true[idx], pred[idx], n_classes)
        stats["accuracy"].append(np.trace(cm) / cm.sum())
        stats["macro_f1"].append(macro_f1_from_cm(cm))
        stats["ece"].append(ece(probs[idx], y_true[idx], n_bins))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in stats.items()}


def reliability_diagram(probs, y_true, path, n_bins: int = 15, title: str = "Reliability") -> float:
    """Save a reliability diagram PNG; returns the ECE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    value, tab = ece_bins(probs, y_true, n_bins)
    edges = tab["edges"]
    centres = (edges[:-1] + edges[1:]) / 2
    nz = tab["counts"] > 0
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=100)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.bar(centres[nz], tab["acc"][nz], width=1.0 / n_bins, edgecolor="k", alpha=0.7,
           label="accuracy")
    ax.plot(tab["conf"][nz], tab["acc"][nz], "o", color="C3", ms=3, label="(mean conf, acc)")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}  ECE={value:.3f}", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return value


def confusion_matrix_png(cm, class_names, path, title: str = "", normalize: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(6, 5.4), dpi=90)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5,
                        color="white" if shown[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
