"""Split definitions and on-disk materialisation for the v0 leakage experiment.

Splits (all exclude ``nouuid`` images, which have no capture provenance):

* ``s0``       random, leaky: frame-level stratified 80/20, seed 42.
* ``s1a``      source-disjoint: train on node 48f17fc25591, test on 0242ac1c0002.
* ``s1b``      reverse of s1a.
* ``s2_<sid>`` leave-one-session-out: test on session ``<sid>``, train on the rest.

In every split 15% of the *train portion* is carved out (stratified by class,
seeded, from the same pool as train) as validation for early stopping.  The val
carve-out happens BEFORE offline balancing, so augmented copies of a val frame
can never be in train.  The ultralytics data root only ever contains
``train/`` and ``val/``; the test list is stored outside it as ``test.csv``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .augment import BALANCE_TARGET, apply_augmentation
from .manifest import CLASSES, centre_crop_224

BURST_NODE = "48f17fc25591"
OTHER_NODE = "0242ac1c0002"
SEED = 42


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["source_node"] != "nouuid"].reset_index(drop=True)


def split_s0(df: pd.DataFrame, seed: int = SEED, test_size: float = 0.2) -> pd.DataFrame:
    df = eligible(df)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=df["cls"], random_state=seed
    )
    return _assign(df, train_idx, test_idx)


def split_by_node(df: pd.DataFrame, train_node: str, test_node: str) -> pd.DataFrame:
    df = eligible(df)
    train_idx = np.flatnonzero((df["source_node"] == train_node).to_numpy())
    test_idx = np.flatnonzero((df["source_node"] == test_node).to_numpy())
    return _assign(df, train_idx, test_idx)


def split_loso(df: pd.DataFrame, held_out_session: str) -> pd.DataFrame:
    df = eligible(df)
    test_mask = (df["session_id"] == held_out_session).to_numpy()
    if not test_mask.any():
        raise ValueError(f"unknown session {held_out_session!r}")
    return _assign(df, np.flatnonzero(~test_mask), np.flatnonzero(test_mask))


def _assign(df: pd.DataFrame, train_idx, test_idx) -> pd.DataFrame:
    out = df.copy()
    out["role"] = "unused"
    out.loc[out.index[test_idx], "role"] = "test"
    out.loc[out.index[train_idx], "role"] = "train"
    return out


def carve_val(split: pd.DataFrame, frac: float = 0.15, seed: int = SEED) -> pd.DataFrame:
    """Move ``frac`` of the train rows (stratified by class) to role ``val``."""
    out = split.copy()
    train = out[out["role"] == "train"]
    _, val_idx = train_test_split(
        train.index.to_numpy(), test_size=frac, stratify=train["cls"], random_state=seed
    )
    out.loc[val_idx, "role"] = "val"
    return out


def build_split(df: pd.DataFrame, name: str, seed: int = SEED) -> pd.DataFrame:
    if name == "s0":
        base = split_s0(df, seed)
    elif name == "s1a":
        base = split_by_node(df, BURST_NODE, OTHER_NODE)
    elif name == "s1b":
        base = split_by_node(df, OTHER_NODE, BURST_NODE)
    elif name.startswith("s2_"):
        base = split_loso(df, name[3:])
    else:
        raise ValueError(f"unknown split {name!r}")
    return carve_val(base, 0.15, seed)


def split_names(df: pd.DataFrame) -> list[str]:
    sessions = sorted(s for s in eligible(df)["session_id"].unique())
    return ["s0", "s1a", "s1b"] + [f"s2_{s}" for s in sessions]


def cache_name(row) -> str:
    return f"{row['orig_split']}_{Path(row['path']).stem}.png"


def ensure_crop_cache(df: pd.DataFrame, raw_root: Path, cache_dir: Path) -> pd.DataFrame:
    """Write each centre-crop-224 image exactly once; add a ``cache_path`` column."""
    out = df.copy()
    paths = []
    for _, row in out.iterrows():
        dest = cache_dir / row["cls"] / cache_name(row)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(raw_root / row["path"]) as raw:
                centre_crop_224(raw).save(dest, "PNG")
        paths.append(str(dest))
    out["cache_path"] = paths
    return out


def materialise(
    split: pd.DataFrame,
    out_dir: Path,
    seed: int = SEED,
    balance_target: int = BALANCE_TARGET,
) -> dict:
    """Create ``out_dir/data/{train,val}/<cls>/`` (symlinks + balancing augmentations)
    and ``out_dir/test.csv`` / ``out_dir/split.csv``.  Returns a summary dict."""
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for role in ("train", "val"):
        for cls in CLASSES:
            (data_dir / role / cls).mkdir(parents=True, exist_ok=True)

    aug_total = 0
    for cls in CLASSES:
        tr = split[(split["role"] == "train") & (split["cls"] == cls)]
        va = split[(split["role"] == "val") & (split["cls"] == cls)]
        for _, row in tr.iterrows():
            _link(row["cache_path"], data_dir / "train" / cls / Path(row["cache_path"]).name)
        for _, row in va.iterrows():
            _link(row["cache_path"], data_dir / "val" / cls / Path(row["cache_path"]).name)
        originals = list(tr["cache_path"])
        deficit = balance_target - len(originals)
        for i in range(max(0, deficit)):
            with Image.open(originals[i % len(originals)]) as src:
                aug = apply_augmentation(src.convert("RGB"), rng)
            aug.save(data_dir / "train" / cls / f"{cls}_aug_{i:05d}.png", "PNG")
            aug_total += 1

    split.drop(columns=["capture_ts"], errors="ignore").to_csv(out_dir / "split.csv", index=False)
    split[split["role"] == "test"].drop(columns=["capture_ts"], errors="ignore").to_csv(
        out_dir / "test.csv", index=False
    )
    counts = split["role"].value_counts().to_dict()
    return {
        "n_train_original": int(counts.get("train", 0)),
        "n_train_augmented": aug_total,
        "n_val": int(counts.get("val", 0)),
        "n_test": int(counts.get("test", 0)),
    }


def _link(src: str, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)
