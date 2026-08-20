#!/usr/bin/env python3
"""Materialize the inactive development target manifest for proposal-lane qualification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from forge.corpus.r1_prime_audit import sha256_file
from forge.synthesis.engine.single_step_benchmark_manifest import (
    OUTPUT_FILENAMES,
    SingleStepBenchmarkManifestError,
    build_manifest_payloads,
    configured_output_directory,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs/route/single_step_proposal_lane_qualification_manifest_v1.json"


def _publish_directory(output_directory: Path, payloads: dict[str, bytes]) -> None:
    if output_directory.exists():
        raise SingleStepBenchmarkManifestError(
            f"immutable benchmark-manifest output already exists: {output_directory}"
        )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        for label, payload in payloads.items():
            path = temporary / OUTPUT_FILENAMES[label]
            with path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if output_directory.exists():
            raise SingleStepBenchmarkManifestError(
                f"immutable benchmark-manifest output appeared during build: {output_directory}"
            )
        os.rename(temporary, output_directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    payloads = build_manifest_payloads(REPO, args.config)
    output_directory = configured_output_directory(REPO, args.config)
    _publish_directory(output_directory, payloads)
    result = {
        label: {
            "path": str(output_directory / OUTPUT_FILENAMES[label]),
            "sha256": sha256_file(output_directory / OUTPUT_FILENAMES[label]),
        }
        for label in sorted(payloads)
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
