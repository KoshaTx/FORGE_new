#!/usr/bin/env python3
"""Freeze the M0-07 graph-oracle corpus and pretraining leakage gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.oracle.oracle_graph_corpus import (
    OracleGraphCorpusError,
    run_oracle_graph_corpus_audit,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_oracle_graph_corpus.json",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/m0_07")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_oracle_graph_corpus_audit(
            args.config,
            args.output_dir,
            args.repo_root,
        )
    except OracleGraphCorpusError as exc:
        print(f"M0-07 graph corpus audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "graph_profiles": {
                    name: {
                        "records": profile["records"],
                        "atoms": profile["atoms"],
                        "directed_edges": profile["directed_edges"],
                        "maximum_atoms": profile["atom_count_distribution"]["maximum"],
                    }
                    for name, profile in result["graph_profiles"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
