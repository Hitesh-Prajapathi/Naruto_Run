"""Build a manifest of every raw hand-sign image.

Each raw filename embeds a UUID1 (``IMG_<uuid>.png``).  A UUID1 encodes the
capture time (UTC, 100 ns ticks since 1582-10-15) and a 48-bit *node* id.  We
use the node as ``source_node`` (a capture environment, NOT a person or a
device identity: ``0242ac1c0002`` has the Docker-style ``02:42:ac:..`` form)
and ``session_id = f"{node}_{utc_date}"``.  Files without a UUID1 get
``source_node = "nouuid"`` and ``session_id = "nouuid"``.

Perceptual hashes use the same centre-square-crop + LANCZOS resize to 224 that
``cv_model/data/prepare_dataset.py`` applied before splitting.

Usage::

    PYTHONPATH=src python -m narutocv.data.manifest --raw-root "$RAW_ROOT"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import uuid
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

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
IMAGE_SIZE = 224
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_UUID_EPOCH = dt.datetime(1582, 10, 15, tzinfo=dt.UTC)


def parse_uuid1(filename: str) -> tuple[str, dt.datetime] | None:
    """Return ``(node_hex, utc_datetime)`` for a filename holding a UUID1, else None."""
    match = _UUID_RE.search(filename)
    if match is None:
        return None
    value = uuid.UUID(match.group(0))
    if value.version != 1:
        return None
    ts = _UUID_EPOCH + dt.timedelta(microseconds=value.time // 10)
    return f"{value.node:012x}", ts


def session_id_for(source_node: str, capture_ts: dt.datetime | None) -> str:
    if source_node == "nouuid" or capture_ts is None:
        return "nouuid"
    return f"{source_node}_{capture_ts.date().isoformat()}"


def centre_crop_224(img: Image.Image) -> Image.Image:
    """Exactly the training-time normalisation: RGB, centre square crop, LANCZOS 224."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


def phash_hex(img224: Image.Image) -> str:
    return str(imagehash.phash(img224))  # 64-bit hash as 16 hex chars


def crop_sha1(img224: Image.Image) -> str:
    return hashlib.sha1(img224.tobytes()).hexdigest()


def iter_raw_files(raw_root: Path):
    for split in ("train", "test"):
        for cls in CLASSES:
            d = raw_root / split / cls
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.name.startswith(".") or f.suffix.lower() not in IMAGE_EXTS:
                    continue
                yield split, cls, f


def build_manifest(raw_root: Path) -> pd.DataFrame:
    rows = []
    for split, cls, f in iter_raw_files(raw_root):
        parsed = parse_uuid1(f.name)
        node, ts = parsed if parsed else ("nouuid", None)
        with Image.open(f) as raw:
            width, height = raw.size
            crop = centre_crop_224(raw)
        rows.append(
            {
                "path": f.relative_to(raw_root).as_posix(),
                "cls": cls,
                "orig_split": split,
                "source_node": node,
                "capture_ts": pd.Timestamp(ts) if ts else pd.NaT,
                "session_id": session_id_for(node, ts),
                "width": width,
                "height": height,
                "phash": phash_hex(crop),
                "crop_sha1": crop_sha1(crop),
            }
        )
    df = pd.DataFrame(rows)
    df["capture_ts"] = pd.to_datetime(df["capture_ts"], utc=True)
    return df


def class_session_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(df["cls"], df["session_id"])
    table = table.reindex([c for c in CLASSES if c in table.index])
    table["total"] = table.sum(axis=1)
    table.loc["TOTAL"] = table.sum(axis=0)
    return table


def provenance_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("source_node")
    out = g.agg(
        n=("path", "size"),
        first_ts=("capture_ts", "min"),
        last_ts=("capture_ts", "max"),
        n_sessions=("session_id", "nunique"),
        n_orig_test=("orig_split", lambda s: int((s == "test").sum())),
    )
    out["span_minutes"] = (out["last_ts"] - out["first_ts"]).dt.total_seconds() / 60.0
    return out.sort_values("n", ascending=False)


def to_markdown(table: pd.DataFrame) -> str:
    cols = [table.index.name or ""] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in table.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/manifest_v0.parquet"))
    p.add_argument("--table-md", type=Path, default=Path("artifacts/class_session_table.md"))
    args = p.parse_args(argv)

    df = build_manifest(args.raw_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    table = class_session_table(df)
    table.index.name = "class"
    args.table_md.write_text(to_markdown(table))
    print(f"wrote {len(df)} rows -> {args.out}")
    print(to_markdown(table))
    print(provenance_summary(df).to_string())


if __name__ == "__main__":
    main()
