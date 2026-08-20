#!/usr/bin/env python3
"""Build the M0-09 LNPDB source queue and component evidence ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.synthesis.sources.source_ledger import (
    SourceLedgerError,
    build_source_ledger,
    write_source_ledger,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_lnpdb_route_sources.json",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=REPO / "data/vendor",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REPO / "data/reference",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_09",
    )
    parser.add_argument(
        "--generated-utc",
        help="optional timezone-aware ISO-8601 timestamp for reproducible reruns",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result, source_rows, component_rows = build_source_ledger(
            args.config,
            args.vendor_dir,
            args.reference_dir,
            generated_utc=args.generated_utc,
            seed=args.seed,
        )
        write_source_ledger(result, source_rows, component_rows, args.output_dir)
    except SourceLedgerError as exc:
        print(f"M0-09 LNPDB source ledger failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output": str(args.output_dir / "route_source_ledger.json"),
                "publications": result["summary"]["unique_publications"],
                "pmc_linked": result["summary"]["publications_with_pmc_identifier"],
                "components": result["summary"]["normalized_role_components"],
                "orphan_components": result["summary"]["components_without_source"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
