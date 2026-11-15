#!/usr/bin/env python3
"""Download and verify the model files listed in models/manifest.yaml.

    uv run python scripts/fetch_models.py            # fetch anything missing, verify all
    uv run python scripts/fetch_models.py --verify   # verify only, never download
    uv run python scripts/fetch_models.py --force    # re-download everything with a URL

Exit status is non-zero if any artifact is missing or fails its sha256/size check.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST = MODELS_DIR / "manifest.yaml"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def check(path: Path, sha256: str, size: int | None) -> str | None:
    """Return None if `path` matches the manifest entry, else a reason string."""
    if not path.is_file():
        return "missing"
    if size is not None and path.stat().st_size != size:
        return f"size {path.stat().st_size} != {size}"
    actual = sha256_of(path)
    if actual != sha256:
        return f"sha256 {actual} != {sha256}"
    return None


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
        # NamedTemporaryFile creates the file 0600; a model file is not a secret, and a
        # container that runs as another uid must be able to read it.
        tmp_path.chmod(0o644)
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="verify only, do not download")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    entries = yaml.safe_load(MANIFEST.read_text())["models"]
    failures = 0
    for entry in entries:
        name, url = entry["name"], entry.get("url")
        sha256, size = entry["sha256"], entry.get("size")
        dest = MODELS_DIR / name

        reason = check(dest, sha256, size)
        if reason is None and not (args.force and url):
            print(f"ok       {name}")
            continue
        if args.verify or not url:
            print(f"FAILED   {name}: {reason or 'ok'}", file=sys.stderr)
            failures += reason is not None
            continue

        print(f"fetching {name} ({size or '?'} bytes) from {url}")
        download(url, dest)
        reason = check(dest, sha256, size)
        if reason is not None:
            print(f"FAILED   {name}: {reason}", file=sys.stderr)
            dest.unlink(missing_ok=True)
            failures += 1
        else:
            print(f"ok       {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
