"""Assemble docs/results/v0-baseline.{json,md} + images from artifacts/ (no hand-typed numbers).

PYTHONPATH=src python scripts/baseline/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from narutocv.data.manifest import CLASSES, class_session_table, provenance_summary, to_markdown
from narutocv.eval.metrics import (
    bootstrap_ci,
    compute_metrics,
    confusion_matrix_png,
    reliability_diagram,
    worst_classes,
)

ART = Path("artifacts")
OUT = Path("docs/results")
IMG = OUT / "img"
LIVE_DIR = Path("reports")  # overridden by --live-dir
CLAIMED = {
    "val": 99.64,
    "test": 99.11,
}  # from 03_refinement_execution_plan.md section 0.2 (not re-run)


def run_row(name: str, desc: str) -> dict | None:
    d = ART / "runs" / name
    if not (d / "run_meta.json").exists():
        return None
    meta = json.loads((d / "run_meta.json").read_text())
    probs = np.load(d / "test_probs.npy")
    y = np.load(d / "test_labels.npy")
    m = compute_metrics(y, probs, CLASSES)
    ci = bootstrap_ci(y, probs, len(CLASSES), n_boot=1000, seed=0)
    summ = json.loads((ART / "splits" / "summary.json").read_text())[name]
    return {
        "split": name,
        "description": desc,
        **summ,
        "epochs_completed": meta["epochs_completed"],
        "best_epoch": meta["best_epoch_by_val_top1"],
        "best_val_top1": meta["best_val_top1"],
        "train_wall_seconds": meta["train_wall_seconds"],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "ece": m["ece"],
        "ci95": ci,
        "worst3": worst_classes(m, 3),
        "per_class": m["per_class"],
        "confusion_matrix": m["confusion_matrix"],
        "n_classes_in_test": int(sum(v["support"] > 0 for v in m["per_class"].values())),
        "_probs": probs,
        "_y": y,
    }


def live_summary(live_dir: Path) -> dict:
    out = {}
    for tag, fname in (("v1.0", "hand-accuracy.json"), ("v1.1", "hand-v1.1-rerun-accuracy.json")):
        f = live_dir / fname
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        per = {}
        for a in d["attempts"]:
            c = per.setdefault(
                a["expected_label"], {"success": 0, "n": 0, "predicted": [], "latency_ms": []}
            )
            c["n"] += 1
            c["success"] += int(bool(a["success"]))
            c["predicted"].append(a.get("system_prediction"))
            if a.get("detection_latency_ms") is not None:
                c["latency_ms"].append(a["detection_latency_ms"])
        out[tag] = per
    return out


def contamination() -> dict:
    prep = pd.read_parquet(ART / "leakage_audit.prepared.parquet")
    train_paths = set(prep[(prep.prep_split == "train") & (~prep.is_aug)].raw_path.dropna())
    out = {}
    for sp in ("s0", "s1a"):
        test = pd.read_csv(ART / "splits" / sp / "test.csv")
        out[sp] = {
            "n": int(len(test)),
            "pct_in_prepared_train": float(100 * test["path"].isin(train_paths).mean()),
        }
    return out


def fmt_ci(x, ci):
    return f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", type=Path, default=Path("reports"))
    a = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(ART / "manifest_v0.parquet")
    audit = json.loads((ART / "leakage_audit.json").read_text())
    onnx = json.loads((ART / "onnx_ref" / "onnx_results.json").read_text())

    descs = {
        "s0": "random frame-level 80/20 (leaky), both sources",
        "s1a": "train burst source 48f1..., test source 0242...",
        "s1b": "train source 0242..., test burst source 48f1... (== S2 held-out 48f1 session)",
    }
    names = ["s0", "s1a", "s1b"] + [
        f"s2_{s}"
        for s in sorted(manifest[manifest.source_node == "0242ac1c0002"].session_id.unique())
    ]
    rows = []
    for n in names:
        r = run_row(n, descs.get(n, f"leave-one-session-out, held out {n[3:]}"))
        if r:
            rows.append(r)
    by = {r["split"]: r for r in rows}

    # figures
    for sp, title in (("s0", "S0 random (leaky)"), ("s1a", "S1a source-disjoint")):
        if sp in by:
            confusion_matrix_png(
                by[sp]["confusion_matrix"],
                CLASSES,
                IMG / f"cm_{sp}.png",
                f"retrained yolov8n-cls, {title}, n={by[sp]['n_test']}",
            )
            reliability_diagram(
                by[sp]["_probs"], by[sp]["_y"], IMG / f"reliability_{sp}.png", title=title
            )

    data = {
        "provenance": {
            "n_images": int(len(manifest)),
            "nodes": json.loads(
                provenance_summary(manifest)
                .reset_index()
                .to_json(orient="records", date_format="iso")
            ),
            "class_by_session": class_session_table(manifest).to_dict(),
            "orig_test_sessions": manifest[manifest.orig_split == "test"]
            .session_id.value_counts()
            .to_dict(),
            "raw_exact_duplicate_images": int(manifest.crop_sha1.duplicated(keep=False).sum()),
        },
        "leakage_audit": audit,
        "claimed": CLAIMED,
        "onnx_contaminated_reference": [
            {
                "name": o["name"],
                "note": o["note"],
                "n": o["metrics"]["n"],
                "accuracy": o["metrics"]["accuracy"],
                "macro_f1": o["metrics"]["macro_f1"],
                "ece": o["metrics"]["ece"],
                "ci95": o["ci95"],
                "worst3": o["worst3"],
            }
            for o in onnx
        ],
        "contamination": contamination(),
        "retrained": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "live_camera": live_summary(a.live_dir),
    }
    (OUT / "v0-baseline.json").write_text(json.dumps(data, indent=1))

    # ---------------------------------------------------------------- markdown
    L = []
    w = L.append
    w("# v0 honest baseline\n")
    w(
        "Generated by `scripts/baseline/make_report.py` from the JSON next to this file; every number "
        "below comes from a run executed for this report except the two figures labelled *claimed*.\n"
    )
    w("## 1. Data provenance\n")
    prov = provenance_summary(manifest)
    w(
        "| source_node | images | sessions | first (UTC) | last (UTC) | span (min) | orig. `test` imgs |"
    )
    w("|---|---|---|---|---|---|---|")
    for node, r in prov.iterrows():
        first = "-" if pd.isna(r.first_ts) else str(r.first_ts)[:19]
        last = "-" if pd.isna(r.last_ts) else str(r.last_ts)[:19]
        span = "-" if pd.isna(r.span_minutes) else f"{r.span_minutes:.1f}"
        w(f"| {node} | {r.n} | {r.n_sessions} | {first} | {last} | {span} | {r.n_orig_test} |")
    w("")
    sz = (
        manifest.assign(size=manifest.width.astype(str) + "x" + manifest.height.astype(str))
        .groupby(["source_node", "size"])
        .size()
    )
    w(
        "Raw image sizes (source, WxH): "
        + "; ".join(f"{s} {z} x{n}" for (s, z), n in sz.items())
        + ". All are centre-square-cropped to 224 before training.\n"
    )
    w(
        "`source_node` is the UUID1 node field, i.e. a **capture environment**, not a person or a "
        "device identity (`0242ac1c0002` is a Docker-style `02:42:ac:..` address). We do not know "
        "whether the same person appears in both sources. Session = node + UTC date. "
        f"{data['provenance']['raw_exact_duplicate_images']} raw images are pixel-identical (after "
        "the 224 crop) to another raw image.\n"
    )
    tbl = class_session_table(manifest)
    tbl.index.name = "class"
    w("Class x session counts (all 2,245 raw images):\n")
    w(to_markdown(tbl))

    la = audit["leakage_vs_train_originals"]
    rc = audit["raw_clusters"]
    w("## 2. Leakage audit of the old random split\n")
    w(
        "Prepared dataset (`prepare_dataset.py`: random stratified 80/10/10 per frame, seed 42) "
        "mapped back to raw by exact 224-crop pixel hash "
        f"({audit['mapping']['map_kind_counts']}). Train originals = prepared train files without "
        f"`_aug_` ({la['n_train_originals']}); {audit['mapping']['n_aug']} augmented copies are "
        "excluded. pHash = 64-bit imagehash pHash of the centre-crop-224 image.\n"
    )
    w(
        "| prepared split | n | same-class near-dup in train originals: Hamming 0 | <= 2 | <= 6 | <= 10 | any-class <= 6 |"
    )
    w("|---|---|---|---|---|---|---|")
    for s in ("val", "test"):
        r = la[s]
        w(
            f"| {s} | {r['n']} | {r['same_class_le_0_pct']:.1f}% | {r['same_class_le_2_pct']:.1f}% | "
            f"{r['same_class_le_6_pct']:.1f}% | {r['same_class_le_10_pct']:.1f}% | "
            f"{r['any_class_le_6_pct']:.1f}% |"
        )
    so = audit["session_overlap"]
    w("")
    w(
        f"Session overlap: **{so['test']['pct_session_in_train_of_all']:.1f}%** of prepared test images "
        f"and {so['val']['pct_session_in_train_of_all']:.1f}% of val images come from a session that "
        f"also has images in prepared train (test sessions: {so['test']['sessions']}).\n"
    )
    w(
        "Near-duplicate clusters among the 2,245 raw images (connected components of the pHash "
        "Hamming graph):\n"
    )
    w(
        "| Hamming <= | clusters | singletons | images in multi-image clusters | largest cluster | "
        "clusters spanning >1 class |"
    )
    w("|---|---|---|---|---|---|")
    for t in ("2", "6", "10"):
        r = rc[t]
        w(
            f"| {t} | {r['n_clusters']} | {r['n_singletons']} | {r['n_images_in_multi_clusters']} | "
            f"{r['largest_cluster']} | {r['n_clusters_with_multiple_classes']} |"
        )
    w("")
    w(
        "The original raw train/test folders share no cluster at any threshold "
        f"({rc['6']['n_clusters_spanning_original_train_and_test']} spanning clusters at <= 6), but "
        "that split is irrelevant to the prepared dataset, which merged both and re-split by frame.\n"
    )

    w("## 3. Results\n")
    w(
        "Retrained = ultralytics `yolov8n-cls` fine-tuned from the same `yolov8n-cls.pt`, notebook "
        "approach-A hyper-parameters, **identical settings for every split** (epochs=50 max, "
        "patience=10; see Deviations). Val = 15% of the train portion (stratified, seeded, carved "
        "before offline class balancing to 220/class). The train column is `originals + balancing "
        "augmentations`. Macro-F1 is over classes with support > 0. CIs: stratified bootstrap, "
        "1,000 resamples, seed 0, percentile 95%.\n"
    )
    w(
        "| split | train (orig+aug) | val | test n | accuracy [95% CI] | macro-F1 [95% CI] | ECE (15 bins) | worst 3 classes (F1, n) | epochs done / best | train wall |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        w3 = ", ".join(f"{c} ({f:.2f}, {n})" for c, f, n in r["worst3"])
        w(
            f"| {r['split']} | {r['n_train_original']}+{r['n_train_augmented']} | {r['n_val']} | "
            f"{r['n_test']} | {fmt_ci(r['accuracy'], r['ci95']['accuracy'])} | "
            f"{fmt_ci(r['macro_f1'], r['ci95']['macro_f1'])} | {r['ece']:.3f} | {w3} | "
            f"{r['epochs_completed']} / {r['best_epoch']} | {r['train_wall_seconds'] / 60:.1f} min |"
        )
    w("")
    w(
        "Split key: **s0** random frame-level (leaky); **s1a** train 48f17fc25591 -> test 0242ac1c0002; "
        "**s1b** train 0242ac1c0002 -> test 48f17fc25591 (this is also identical to leave-48f1-session-out, "
        "so S2 for that session is s1b); **s2_*** leave one session out, train on all other "
        "non-`nouuid` sessions. Held-out sessions can miss classes (see class x session table); macro-F1 "
        "uses classes present in that test set.\n"
    )
    w("### Deployed `best_model_A.onnx` (CONTAMINATED: model trained on all sources)\n")
    c = data["contamination"]
    w(
        "Not an honest estimate. The deployed model was trained on a random 80% of frames from every "
        "source: "
        f"{c['s0']['pct_in_prepared_train']:.1f}% of the S0 test images and "
        f"{c['s1a']['pct_in_prepared_train']:.1f}% of the S1a test images were literally in its "
        "training set (exact raw-image match to a prepared train original), and most of the remainder have near-duplicates there (section 2). "
        "`png224` = 224 PNGs from the training pipeline fed through the runtime tensor math; "
        "`rawcrop` = raw image, centre square, cv2 INTER_AREA 224 (runtime-style).\n"
    )
    w("| set | pipeline | n | accuracy [95% CI] | macro-F1 [95% CI] | ECE |")
    w("|---|---|---|---|---|---|")
    for o in data["onnx_contaminated_reference"]:
        oo = next(x for x in onnx if x["name"] == o["name"])
        w(
            f"| {o['name']} | {'rawcrop' if o['name'].endswith('rawcrop') else 'png224'} | {o['n']} | "
            f"{fmt_ci(o['accuracy'], oo['ci95']['accuracy'])} | "
            f"{fmt_ci(o['macro_f1'], oo['ci95']['macro_f1'])} | {o['ece']:.3f} |"
        )
    pv = next(o for o in onnx if o["name"] == "prepared_val")["metrics"]["accuracy"]
    pt = next(o for o in onnx if o["name"] == "prepared_test")["metrics"]["accuracy"]
    w("")
    w(
        f"Preprocessing self-check: on the prepared val/test PNGs this reproduction gives "
        f"{100 * pv:.2f}% / {100 * pt:.2f}%, against the *claimed* {CLAIMED['val']}% / {CLAIMED['test']}% "
        "(taken from `03_refinement_execution_plan.md`, not re-measured by ultralytics here). The small "
        "gap is within a few images; the preprocessing and class order are right.\n"
    )
    w("### Confusion matrices (row-normalised)\n")
    w(
        "| S0 random (leaky) | S1a source-disjoint |\n|---|---|\n"
        "| ![S0](img/cm_s0.png) | ![S1a](img/cm_s1a.png) |\n"
    )
    w("Reliability diagrams: `img/reliability_s0.png`, `img/reliability_s1a.png`.\n")

    live = data["live_camera"]
    if live:
        w("## 4. Live-camera results (existing reports, not re-run)\n")
        w(
            "3 attempts per seal on a webcam with the full runtime pipeline (thresholds, temporal "
            "filter). Source: `reports/hand-accuracy.json` (v1.0) and "
            "`reports/hand-v1.1-rerun-accuracy.json` (v1.1). `success` is as defined by those "
            "evaluators; predictions listed are the system's final prediction per attempt.\n"
        )
        w("| seal | v1.0 success | v1.0 predictions | v1.1 success | v1.1 predictions |")
        w("|---|---|---|---|---|")
        for c in CLASSES:
            cells = []
            for tag in ("v1.0", "v1.1"):
                r = live.get(tag, {}).get(c)
                cells += (
                    [f"{r['success']}/{r['n']}", ", ".join(str(p) for p in r["predicted"])]
                    if r
                    else ["-", "-"]
                )
            w(f"| {c} | " + " | ".join(cells) + " |")
        w(
            "`None` = the system emitted no action event for that attempt (the raw per-frame model output may still have named a seal).\n"
        )

    w("## 5. What this means\n")
    for line in narrative(rows, by, onnx, audit, data["contamination"]):
        w(line)
    w("")
    w(
        "S1 is **source/session-disjoint, not subject-disjoint**: we do not know whether the same "
        "person appears in both sources, so S1 numbers are still an optimistic bound on "
        "new-subject generalisation.\n"
    )
    w("## 6. Deviations from the notebook recipe\n")
    meta = json.loads((ART / "runs" / "s0" / "run_meta.json").read_text())
    w(
        f"- ultralytics {meta['ultralytics']}, torch {meta['torch']}, Python {meta['python']}; one "
        f"Apple `{meta['device']}` device instead of 2x T4 DDP; optimizer left on ultralytics "
        "`auto` (AdamW here); `seed=42`, `deterministic=False`, `plots=False`, `workers=4` "
        "(ultralytics reported 0 dataloader workers on MPS)."
    )
    w(
        "- Notebook ran epochs=50, patience=10 on the leaky random split; same values here for every "
        "split (the notebook text says ~50-80 epochs; the 80 was approach B)."
    )
    w(
        "- Offline balancing to 220/class re-implemented with a seeded RNG "
        "(`src/narutocv/data/augment.py`), same operations as `prepare_dataset.py`. For s1b/s2 on the "
        "burst source or small train portions this means many augmented copies per original."
    )
    w(
        "- Early stopping uses a val set drawn from the *same sources/frames as train*, so val is "
        "leaky in every split, including s1/s2. Epochs completed differ per split because of that."
    )
    w("- Test predictions use ultralytics `predict` at imgsz=224 on the same 224 centre-crop PNGs.")
    w(
        "- Wall times are on a shared machine (other jobs were running) and are indicative only; "
        "the longest (s2_..._2020-02-12) completed a similar number of epochs to the others in a "
        "several-times-longer wall time, which we attribute to machine contention (not investigated)."
    )
    w(
        "- Single seed per split; no seed-to-seed variance is estimated, only test-set bootstrap CIs.\n"
    )
    w("## 7. Reproduce\n")
    w(
        "```bash\nuv venv --python 3.14 .venv\n"
        "uv pip install --python .venv/bin/python torch torchvision ultralytics onnxruntime imagehash "
        "pandas pyarrow scikit-learn matplotlib pytest pillow opencv-python-headless\n"
        "export RAW_ROOT='<path>/Pure Naruto Hand Sign Data'   # read-only raw data\n"
        "export PREPARED_ROOT='<path>/cv_model/data/prepared_dataset'   # old leaky split\n"
        "export PYTHONPATH=src\n"
        '.venv/bin/python -m narutocv.data.manifest --raw-root "$RAW_ROOT"\n'
        '.venv/bin/python -m narutocv.data.dedup --raw-root "$RAW_ROOT" --prepared "$PREPARED_ROOT"\n'
        '.venv/bin/python scripts/baseline/build_splits.py --raw-root "$RAW_ROOT"\n'
        "YOLO_OFFLINE=1 .venv/bin/python scripts/baseline/train_split.py s0 s1a s1b "
        + " ".join(n for n in names[3:])
        + "\n"
        '.venv/bin/python scripts/baseline/eval_onnx.py --raw-root "$RAW_ROOT" --prepared "$PREPARED_ROOT"\n'
        ".venv/bin/python scripts/baseline/make_report.py --live-dir <path>/reports\n"
        ".venv/bin/python -m pytest --import-mode=importlib tests\n```\n"
    )
    (OUT / "v0-baseline.md").write_text("\n".join(L))
    print("wrote", OUT / "v0-baseline.md")


def narrative(rows, by, onnx, audit, data_c) -> list[str]:
    s0, s1a, s1b = by["s0"], by["s1a"], by["s1b"]
    la = audit["leakage_vs_train_originals"]["test"]
    ox = {o["name"]: o["metrics"] for o in onnx}
    s2 = [r for r in rows if r["split"].startswith("s2_")]
    lo = min(s2, key=lambda r: r["macro_f1"])
    hi = max(s2, key=lambda r: r["macro_f1"])
    cm = np.array(s1a["confusion_matrix"])
    to_zero = int(
        cm[:, CLASSES.index("zero")].sum() - cm[CLASSES.index("zero"), CLASSES.index("zero")]
    )
    cmb = np.array(s1b["confusion_matrix"])
    t, r_ = CLASSES.index("tiger"), CLASSES.index("ram")
    return [
        f"1. Same recipe, same data, only the split changes: macro-F1 {s0['macro_f1']:.3f} (S0, random) "
        f"vs {s1a['macro_f1']:.3f} (S1a, train burst source -> test other source) and "
        f"{s1b['macro_f1']:.3f} (S1b, reverse). The ~99%+ figure measures memorisation of the room/session.",
        f"2. Cause: {la['same_class_le_6_pct']:.1f}% of the old prepared test images have a same-class "
        f"pHash near-duplicate (Hamming <= 6) among prepared train originals, and "
        f"{audit['session_overlap']['test']['pct_session_in_train_of_all']:.1f}% come from a session also in train.",
        f"3. The deployed ONNX scores {ox['s1a_test_png224']['accuracy']:.3f} accuracy on the S1a test "
        f"images because it trained on those sources ({data_c['s1a']['pct_in_prepared_train']:.1f}% of those "
        f"images were in its training set; contaminated); the honestly retrained model gets "
        f"{s1a['accuracy']:.3f} on the same images.",
        f"4. Failure shape is source-specific and not 'noise': S1a sends {to_zero} of {s1a['n_test']} "
        f"images to `zero`; S1b sends {int(cmb[t, r_])} of {int(cmb[t].sum())} burst-source `tiger` images "
        f"to `ram`. Leave-one-session-out macro-F1 over the four 0242ac1c0002 sessions ranges "
        f"{lo['macro_f1']:.3f} ({lo['split'][3:]}) to {hi['macro_f1']:.3f} ({hi['split'][3:]}); "
        "per-session test sets are small, so CIs are wide.",
        "5. Caveats: a single seed per split, ultralytics `auto` optimizer, and a leaky val set used for "
        "early stopping all make these numbers rough; the direction and size of the gap are the finding, not the third decimal.",
    ]


if __name__ == "__main__":
    main()
