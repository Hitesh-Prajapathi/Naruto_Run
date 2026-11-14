"""Drift check: the committed golden fixtures must equal a fresh regeneration.

If this fails, a reference implementation (or the generator) changed.  Regenerate with
``uv run python scripts/golden/generate_golden.py``, review the diff, commit it, and port the
change to TypeScript until ``npx vitest run`` passes (see ``docs/specs/parity.md``).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden"


def _load_generator() -> ModuleType:
    path = ROOT / "scripts" / "golden" / "generate_golden.py"
    spec = importlib.util.spec_from_file_location("generate_golden", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_golden"] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()

TOLERANCES = {
    "features_v1.json": gen.FEATURES_DRIFT_TOL,
    "decoder_v1.json": gen.DECODER_DRIFT_TOL,
    "calibration_v1.json": gen.CALIBRATION_DRIFT_TOL,
}


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-standard JSON constant {token!r} in a golden fixture")


def _diff(a: Any, b: Any, tol: float, path: str = "$") -> str | None:
    """First difference between two JSON trees (floats compared with a tolerance)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            d = _diff(a[k], b[k], tol, f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _diff(x, y, tol, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return None if a is b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return None if a == b else f"{path}: {a} != {b}"
        return None if math.isclose(a, b, rel_tol=tol, abs_tol=tol) else f"{path}: {a} != {b}"
    return None if a == b else f"{path}: {a!r} != {b!r}"


@pytest.fixture(scope="module")
def regenerated() -> dict[str, Any]:
    # Round-trip through the serialiser so the comparison sees exactly what a file would.
    return {
        name: json.loads(gen.dumps_fixture(doc), parse_constant=_reject_constant)
        for name, doc in gen.build_all().items()
    }


@pytest.mark.parametrize("name", gen.FIXTURE_NAMES)
def test_fixture_is_committed_and_current(name: str, regenerated: dict[str, Any]) -> None:
    path = GOLDEN / name
    assert path.is_file(), f"{name} is missing: run scripts/golden/generate_golden.py"
    committed = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    difference = _diff(committed, regenerated[name], TOLERANCES[name])
    assert difference is None, (
        f"{name} is stale ({difference}). Regenerate: "
        "uv run python scripts/golden/generate_golden.py, commit it, then port the change to "
        "TypeScript (docs/specs/parity.md)."
    )


def test_fixture_sizes() -> None:
    feats = json.loads((GOLDEN / "features_v1.json").read_text(encoding="utf-8"))
    dec = json.loads((GOLDEN / "decoder_v1.json").read_text(encoding="utf-8"))
    assert len(feats["cases"]) >= 40
    assert len(dec["cases"]) >= 15
    assert any(c["expect_error"] for c in feats["cases"])
    assert any(c["expected"]["stats"]["gaps"] for c in dec["cases"])
    assert any(c["expected"]["stats"]["nonmonotonic"] for c in dec["cases"])
