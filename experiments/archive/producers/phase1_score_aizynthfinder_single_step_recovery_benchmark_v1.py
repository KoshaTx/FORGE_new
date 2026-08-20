#!/usr/bin/env python3
"""Score a frozen proposal ledger against separately loaded hidden truth."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from forge.synthesis.engine.aizynthfinder_single_step_recovery import (
    AiZynthFinderRecoveryError,
    load_gzip_json,
    score_frozen_proposals,
    sha256_file,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AiZynthFinderRecoveryError(f"JSON object required: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise AiZynthFinderRecoveryError("proposal ledger contains malformed rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_aizynthfinder_single_step_recovery_benchmark_v1.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _json(config_path)
    output = root / str(config["output_directory"])
    proposal_result_path = output / "proposal_result.json"
    ledger_path = output / "proposal_ledger.jsonl.gz"
    result = _json(proposal_result_path)
    if sha256_file(ledger_path) != result["artifacts"]["proposal_ledger"]["sha256"]:
        raise AiZynthFinderRecoveryError("frozen proposal ledger hash changed")
    truth_record = config["inputs"]["scoring_truth"]
    truth_path = root / str(truth_record["path"])
    if sha256_file(truth_path) != truth_record["sha256"]:
        raise AiZynthFinderRecoveryError("hidden truth hash changed")
    score = score_frozen_proposals(
        proposal_result_payload=result,
        proposal_rows=_rows(ledger_path),
        truth_payload=load_gzip_json(truth_path),
        truth_path=str(truth_path.relative_to(root)),
        truth_sha256=sha256_file(truth_path),
    )
    score_path = output / "score.json"
    if score_path.exists():
        raise AiZynthFinderRecoveryError("write-once score already exists")
    score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")
    print(json.dumps(score["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
