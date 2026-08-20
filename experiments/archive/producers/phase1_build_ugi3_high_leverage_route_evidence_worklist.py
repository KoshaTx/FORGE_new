#!/usr/bin/env python3
"""Build the frozen development-only Ugi route-evidence worklist."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.synthesis.evidence.ugi3_high_leverage_route_evidence_worklist import (
    build_high_leverage_route_evidence_worklist,
    write_worklist_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_ugi3_high_leverage_route_evidence_worklist_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/ugi3_high_leverage_route_evidence_worklist_v1/manifest.json"),
    )
    arguments = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = arguments.config if arguments.config.is_absolute() else repo_root / arguments.config
    output = arguments.output if arguments.output.is_absolute() else repo_root / arguments.output
    manifest = build_high_leverage_route_evidence_worklist(config, repo_root=repo_root)
    write_worklist_manifest(manifest, output)


if __name__ == "__main__":
    main()
