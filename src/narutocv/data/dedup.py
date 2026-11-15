"""Near-duplicate clustering and the old-split leakage audit.

* ``cluster_by_hamming``: connected components over pHash Hamming <= threshold.
* ``audit``: maps every prepared val/test/train image back to a raw image
  (exact pixel SHA-1 of the 224 crop first, then pHash Hamming <= 2 within the
  same class) and measures how many prepared val/test images have a near-
  duplicate (Hamming <= T) among prepared *train originals* (``_aug`` files
  excluded, since they are copies of train originals).

Hashes are 64-bit pHash of the centre-crop-224 image.

    PYTHONPATH=src python -m narutocv.data.dedup --raw-root "$RAW_ROOT" \
        --prepared "$PREPARED_ROOT" --out artifacts/leakage_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .manifest import CLASSES, crop_sha1, phash_hex

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hex_to_u64(hexes) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two uint64 arrays -> (len(a), len(b))."""
    x = (a[:, None] ^ b[None, :]).astype(np.uint64)
    as_bytes = x.view(np.uint8).reshape(*x.shape, 8)
    return _POP[as_bytes].sum(axis=-1).astype(np.int16)


def cluster_by_hamming(hashes: np.ndarray, threshold: int) -> np.ndarray:
    """Cluster labels (0..k-1) = connected components of the (dist <= threshold) graph."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = hamming_matrix(hashes, hashes)
    ii, jj = np.nonzero(np.triu(d <= threshold, k=1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def cluster_report(df: pd.DataFrame, thresholds=(2, 6, 10)) -> dict:
    """Per-threshold: n clusters, size stats, clusters mixing classes / sessions."""
    h = hex_to_u64(df["phash"])
    out = {}
    for t in thresholds:
        lab = cluster_by_hamming(h, t)
        g = df.assign(_c=lab).groupby("_c")
        sizes = g.size()
        out[str(t)] = {
            "n_images": int(len(df)),
            "n_clusters": int(len(sizes)),
            "n_singletons": int((sizes == 1).sum()),
            "n_images_in_multi_clusters": int(sizes[sizes > 1].sum()),
            "largest_cluster": int(sizes.max()),
            "n_clusters_with_multiple_classes": int((g["cls"].nunique() > 1).sum()),
            "n_clusters_spanning_original_train_and_test": int(
                (g["orig_split"].nunique() > 1).sum()
            ),
        }
    return out


def load_prepared(prepared_root: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            d = prepared_root / split / cls
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.png")):
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    rows.append(
                        {
                            "prep_split": split,
                            "cls": cls,
                            "file": f.name,
                            "is_aug": "_aug_" in f.name,
                            "phash": phash_hex(im),
                            "sha1": crop_sha1(im),
                        }
                    )
    return pd.DataFrame(rows)


def map_to_raw(prep: pd.DataFrame, manifest: pd.DataFrame, max_dist: int = 2) -> pd.DataFrame:
    """Add raw_path / raw_session / map_dist / map_kind to non-aug prepared rows."""
    prep = prep.copy()
    prep["raw_path"] = None
    prep["raw_session"] = None
    prep["map_dist"] = np.nan
    prep["map_kind"] = "none"
    prep["n_raw_candidates"] = 0
    sha_index = manifest.groupby("crop_sha1")["path"].apply(list).to_dict()
    mh = hex_to_u64(manifest["phash"])
    for i, row in prep.iterrows():
        if row["is_aug"]:
            prep.at[i, "map_kind"] = "aug"
            continue
        exact = [p for p in sha_index.get(row["sha1"], []) if p.split("/")[1] == row["cls"]]
        if exact:
            m = manifest[manifest["path"] == exact[0]].iloc[0]
            prep.at[i, "raw_path"], prep.at[i, "raw_session"] = m["path"], m["session_id"]
            prep.at[i, "map_dist"], prep.at[i, "map_kind"] = 0, "exact_pixels"
            prep.at[i, "n_raw_candidates"] = len(exact)
            continue
        d = hamming_matrix(hex_to_u64([row["phash"]]), mh)[0]
        ok = np.flatnonzero((d <= max_dist) & (manifest["cls"].to_numpy() == row["cls"]))
        if len(ok):
            best = ok[np.argmin(d[ok])]
            prep.at[i, "raw_path"] = manifest.at[best, "path"]
            prep.at[i, "raw_session"] = manifest.at[best, "session_id"]
            prep.at[i, "map_dist"], prep.at[i, "map_kind"] = float(d[best]), "phash"
            prep.at[i, "n_raw_candidates"] = len(ok)
    return prep


def leakage_vs_train(prep: pd.DataFrame, thresholds=(0, 2, 6, 10)) -> dict:
    """% of val/test images with a pHash near-duplicate among prepared train originals."""
    train = prep[(prep["prep_split"] == "train") & (~prep["is_aug"])]
    th = hex_to_u64(train["phash"])
    tcls = train["cls"].to_numpy()
    out = {"n_train_originals": int(len(train))}
    for split in ("val", "test"):
        q = prep[prep["prep_split"] == split]
        d = hamming_matrix(hex_to_u64(q["phash"]), th)
        same = tcls[None, :] == q["cls"].to_numpy()[:, None]
        d_same = np.where(same, d, 999)
        res = {"n": int(len(q))}
        for t in thresholds:
            res[f"any_class_le_{t}_pct"] = float(100 * (d.min(axis=1) <= t).mean())
            res[f"same_class_le_{t}_pct"] = float(100 * (d_same.min(axis=1) <= t).mean())
        res["median_nearest_same_class_dist"] = float(np.median(d_same.min(axis=1)))
        out[split] = res
    return out


def session_overlap(prep: pd.DataFrame) -> dict:
    """% of prepared test (and val) images whose raw session also appears in prepared train."""
    train_sessions = set(
        prep[(prep["prep_split"] == "train") & (~prep["is_aug"])]["raw_session"].dropna()
    )
    out = {"train_sessions": sorted(train_sessions)}
    for split in ("val", "test"):
        q = prep[prep["prep_split"] == split]
        mapped = q[q["raw_session"].notna()]
        out[split] = {
            "n": int(len(q)),
            "n_mapped_to_raw": int(len(mapped)),
            "pct_session_in_train_of_mapped": float(
                100 * mapped["raw_session"].isin(train_sessions).mean()
            ),
            "pct_session_in_train_of_all": float(
                100 * q["raw_session"].isin(train_sessions).sum() / len(q)
            ),
            "sessions": mapped["raw_session"].value_counts().to_dict(),
        }
    return out


def audit(raw_root_unused, manifest: pd.DataFrame, prepared_root: Path) -> dict:
    prep = map_to_raw(load_prepared(prepared_root), manifest)
    non_aug = prep[~prep["is_aug"]]
    mapping = {
        "n_prepared": int(len(prep)),
        "n_prepared_non_aug": int(len(non_aug)),
        "n_aug": int(prep["is_aug"].sum()),
        "by_split": prep.groupby(["prep_split", "is_aug"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict("records"),
        "map_kind_counts": non_aug["map_kind"].value_counts().to_dict(),
        "n_non_aug_with_multiple_raw_candidates": int((non_aug["n_raw_candidates"] > 1).sum()),
    }
    return {
        "raw_clusters": cluster_report(manifest),
        "mapping": mapping,
        "leakage_vs_train_originals": leakage_vs_train(prep),
        "session_overlap": session_overlap(prep),
    }, prep


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest_v0.parquet"))
    p.add_argument("--out", type=Path, default=Path("artifacts/leakage_audit.json"))
    a = p.parse_args(argv)
    manifest = pd.read_parquet(a.manifest)
    result, prep = audit(a.raw_root, manifest, a.prepared)
    a.out.write_text(json.dumps(result, indent=2, default=str))
    prep.drop(columns=["sha1"]).to_parquet(a.out.with_suffix(".prepared.parquet"), index=False)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
