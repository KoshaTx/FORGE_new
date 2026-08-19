#!/usr/bin/env python3
"""Audit exact Ugi adapter features and core-rooted tree serializations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.product.ugi_adapter_feature_audit import audit_ugi_adapter_features

REPO = Path(__file__).resolve().parents[1]


def _write(path: Path, result: dict) -> None:
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=REPO / "results/phase1/ugi_l1_semantics",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=REPO / "data/splits/phase1/ugi_l1_assignments.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_adapter_feature_audit.json",
    )
    args = parser.parse_args()
    result = audit_ugi_adapter_features(
        args.semantic_dir / "ugi_l1_semantic_products.csv.gz",
        args.semantic_dir / "ugi_l1_semantic_atoms.csv.gz",
        args.assignments,
    )
    _write(args.output, result)
    print(args.output)
    if result["status"] != "pass":
        raise RuntimeError("Ugi adapter feature audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
