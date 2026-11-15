"""Materialise the v0 leakage-experiment splits under artifacts/splits/.

PYTHONPATH=src python scripts/baseline/build_splits.py --raw-root "$RAW_ROOT"
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from narutocv.data.splits import build_split, ensure_crop_cache, materialise, split_names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest_v0.parquet"))
    p.add_argument("--out", type=Path, default=Path("artifacts/splits"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/cache224"))
    p.add_argument("--only", nargs="*", help="subset of split names")
    args = p.parse_args()

    df = pd.read_parquet(args.manifest)
    df = ensure_crop_cache(
        df[df["source_node"] != "nouuid"].reset_index(drop=True),
        args.raw_root,
        args.cache.resolve(),
    )
    summary = {}
    for name in args.only or split_names(df):
        split = build_split(df, name)
        dest = args.out / name
        if dest.exists():
            shutil.rmtree(dest)
        summary[name] = materialise(split, dest.resolve())
        print(name, summary[name])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
