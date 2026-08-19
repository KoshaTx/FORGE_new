#!/usr/bin/env python3
"""Render the frozen AiZynthFinder component route-hypothesis diagnostic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from forge.route.aizynthfinder_diagnostic import render_route_hypothesis_pdf

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=REPO_ROOT
        / "results/phase1/aizynthfinder_component_route_diagnostic_v1/result.json",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=REPO_ROOT / "output/pdf/FORGE_AiZynthFinder_route_hypothesis_audit.pdf",
    )
    arguments = parser.parse_args()
    result = json.loads(arguments.result.read_text())
    ledger_record = result["artifacts"]["route_hypothesis_ledger"]
    ledger_path = Path(ledger_record["path"])
    if not ledger_path.is_absolute():
        ledger_path = REPO_ROOT / ledger_path
    observed = _sha256_file(ledger_path)
    if observed != ledger_record["sha256"]:
        raise ValueError("route-hypothesis ledger SHA-256 mismatch")
    with gzip.open(ledger_path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    render_route_hypothesis_pdf(result=result, rows=rows, output_path=arguments.output_pdf)
    print(
        json.dumps(
            {
                "pdf": str(arguments.output_pdf),
                "planner_pages": 1 + sum(row["full_search"] is not None for row in rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
