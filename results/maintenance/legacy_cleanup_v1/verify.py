"""Verify this cleanup's exact source identities and supported entry points without jobs.

Run: PYTHONPATH=.:tools:paper .venv/bin/python results/maintenance/legacy_cleanup_v1/verify.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from forge_provenance.pins import load_moves
from forge_provenance.resolver import HistoricalPinArchive

from experiments._runtime.registry import registry
from experiments.catalog import SPECIFICATIONS, load_catalog
from forge.core.hashing import sha256_file


def main() -> None:
    receipt = Path(__file__).resolve().parent
    repo = receipt.parents[2]
    baseline = json.loads((receipt / "baseline.json").read_text())
    with gzip.open(receipt / "source-identities-before.json.gz", "rt") as handle:
        identities = json.load(handle)
    archive = HistoricalPinArchive.load(repo / "provenance/frozen-code/manifest.json", repo)
    moves = load_moves(repo / "docs/artifact_path_moves.json")
    current = {}
    lost = []
    recovered = []
    for row in identities:
        name = row["path"]
        if name not in current:
            path = repo / moves.get(name, name)
            current[name] = str(sha256_file(path)) if path.is_file() else None
        resolved = (
            current[name] == row["sha256"] or archive.resolve(name, row["sha256"]) is not None
        )
        if row["resolved"] and not resolved:
            lost.append([name, row["sha256"]])
        if not row["resolved"] and resolved:
            recovered.append([name, row["sha256"]])
    assert not lost, f"Previously resolvable source identities lost: {lost}"
    before = json.loads((receipt / "archive-before.json").read_text())
    after = json.loads((repo / "provenance/frozen-code/manifest.json").read_text())
    indexed = {(row["original_path"], row["sha256"]): row for row in after["entries"]}
    for row in before["entries"]:
        assert indexed[(row["original_path"], row["sha256"])] == row
        assert archive.resolve(row["original_path"], row["sha256"]) is not None
    load_catalog()
    assert SPECIFICATIONS == baseline["catalog"]
    stages = {
        name: f"{registry.resolve(name).__module__}:{registry.resolve(name).__name__}"
        for name in registry.identifiers()
    }
    assert stages == baseline["stages"]
    print(
        json.dumps(
            {
                "status": "pass",
                "source_identities": len(identities),
                "previously_resolvable": sum(row["resolved"] for row in identities),
                "previously_unresolved": sum(not row["resolved"] for row in identities),
                "newly_unresolved": lost,
                "recovered": recovered,
                "original_archive_entries_preserved": len(before["entries"]),
                "archive_entries_added": len(after["entries"]) - len(before["entries"]),
                "catalog_entries": len(SPECIFICATIONS),
                "stage_ids": len(stages),
                "training_calls": 0,
                "remote_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
