"""Identify the model bundle the server runs, from `models/manifest.yaml`.

The manifest lists individual artifacts (hand classifier, hand landmarker,
pose landmarker), each with a pinned sha256. A "bundle" is the set of them, so
its identity is derived rather than stored:

    bundle sha256 = sha256 of the sorted lines "<name>:<sha256>\\n"
    bundle id     = "narutocv-" + first 12 hex chars of the bundle sha256

The digests come from the manifest, not from hashing the files at runtime;
`scripts/fetch_models.py --verify` is what proves the files match the manifest
(the Docker image runs it at build time). If the manifest is missing or
unreadable the bundle is reported as `unversioned` and a client that asked for
a specific bundle id is refused.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("narutocv.transport")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "models" / "manifest.yaml"
MANIFEST_ENV = "NARUTOCV_MODEL_MANIFEST"
UNVERSIONED_ID = "unversioned"


@dataclass(frozen=True)
class ModelFile:
    name: str
    version: str
    sha256: str


@dataclass(frozen=True)
class ModelBundle:
    id: str
    sha256: Optional[str]
    files: tuple[ModelFile, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sha256": self.sha256,
            "files": [
                {"name": f.name, "version": f.version, "sha256": f.sha256} for f in self.files
            ],
        }


UNVERSIONED_BUNDLE = ModelBundle(id=UNVERSIONED_ID, sha256=None)


def bundle_from_entries(entries: list[dict[str, Any]]) -> ModelBundle:
    files = tuple(
        sorted(
            (
                ModelFile(
                    name=str(entry["name"]),
                    version=str(entry.get("version", "")),
                    sha256=str(entry["sha256"]),
                )
                for entry in entries
            ),
            key=lambda f: f.name,
        )
    )
    if not files:
        return UNVERSIONED_BUNDLE
    digest = hashlib.sha256("".join(f"{f.name}:{f.sha256}\n" for f in files).encode()).hexdigest()
    return ModelBundle(id=f"narutocv-{digest[:12]}", sha256=digest, files=files)


def load_model_bundle(manifest_path: Optional[Path] = None) -> ModelBundle:
    path = Path(manifest_path or os.environ.get(MANIFEST_ENV) or DEFAULT_MANIFEST)
    try:
        import yaml

        data = yaml.safe_load(path.read_text())
        return bundle_from_entries(list(data["models"]))
    except Exception as error:
        logger.warning(
            "model manifest unavailable; reporting the bundle as unversioned",
            extra={"manifest": str(path), "error": f"{type(error).__name__}: {error}"},
        )
        return UNVERSIONED_BUNDLE
