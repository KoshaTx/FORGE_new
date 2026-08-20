#!/usr/bin/env python3
"""Materialize a hash-bound prereveal R0/without-C18 and R1/with-C18 pair."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.data.r1_prime_audit import sha256_file
from forge.route.assessment.ugi3_route_registry_pair_builder import (
    build_registry_pair,
    configured_output_paths,
)

REPO = Path(__file__).resolve().parents[1]


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_route_registry_pair_builder_v1.json",
    )
    args = parser.parse_args()
    payloads = build_registry_pair(REPO, args.config)
    paths = configured_output_paths(REPO, args.config)
    for label in (
        "r0_records",
        "r0_manifest",
        "r1_records",
        "r1_manifest",
        "registry_diff",
        "binding",
    ):
        _write_atomic(paths[label], payloads[label])
    result = {
        label: {"path": str(paths[label]), "sha256": sha256_file(paths[label])}
        for label in payloads
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
