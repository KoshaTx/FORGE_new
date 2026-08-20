#!/usr/bin/env python3
"""Score frozen Graph2Edits proposals after hidden-truth reveal."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from forge.route.engine.aizynthfinder_single_step_recovery import load_gzip_json, sha256_file
from forge.route.engine.graph2edits_single_step_recovery import score_graph2edits_recovery


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_graph2edits_single_step_recovery_benchmark_v1.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _json(config_path)
    output = root / config["output_directory"]
    result = _json(output / "proposal_result.json")
    ledger_path = output / "proposal_ledger.jsonl.gz"
    if sha256_file(ledger_path) != result["artifacts"]["proposal_ledger"]["sha256"]:
        raise RuntimeError("Graph2Edits proposal ledger hash changed")
    with gzip.open(ledger_path, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    truth_record = config["inputs"]["scoring_truth"]
    truth_path = root / truth_record["path"]
    if sha256_file(truth_path) != truth_record["sha256"]:
        raise RuntimeError("hidden truth hash changed")
    score = score_graph2edits_recovery(
        proposal_result=result,
        proposal_rows=rows,
        truth_payload=load_gzip_json(truth_path),
        truth_path=str(truth_path.relative_to(root)),
        truth_sha256=sha256_file(truth_path),
    )
    score_path = output / "score.json"
    if score_path.exists():
        raise RuntimeError("write-once score already exists")
    score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")
    print(json.dumps(score["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
