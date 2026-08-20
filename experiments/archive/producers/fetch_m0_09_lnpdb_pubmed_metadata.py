#!/usr/bin/env python3
"""Fetch a minimal PubMed metadata snapshot for LNPDB-linked R0 publications."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.synthesis.sources.source_ledger import (
    SourceLedgerError,
    fetch_pubmed_snapshot,
    load_config,
    write_json,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_lnpdb_route_sources.json",
        help="source-ledger config containing explicit identifier corrections",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO / "data/vendor/lnpdb_fc7c389.csv",
        help="hash-pinned raw LNPDB CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data/reference/lnpdb_pubmed_metadata_2026-07-28.json",
        help="normalized metadata snapshot",
    )
    parser.add_argument(
        "--retrieved-utc",
        help="optional timezone-aware ISO-8601 timestamp",
    )
    parser.add_argument(
        "--additional-pmid",
        action="append",
        default=[],
        help="additional corrected PMID to include; may be repeated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        corrected_pmids = [
            str(override["resolved_pmid"])
            for override in config.get("publication_identity_overrides", [])
        ]
        snapshot = fetch_pubmed_snapshot(
            args.source,
            additional_pmids=[*corrected_pmids, *args.additional_pmid],
            retrieved_utc=args.retrieved_utc,
        )
        write_json(snapshot, args.output)
    except SourceLedgerError as exc:
        print(f"PubMed metadata fetch failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {len(snapshot['records'])} PubMed records to {args.output} "
        f"for source {snapshot['source_sha256']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
