"""Build the frozen proposal-aware upstream-closure sensitivity artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.value.ugi3_proposal_aware_closure_sensitivity import (
    build_proposal_aware_closure_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi3_proposal_aware_closure_sensitivity_v1.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve()
    config = json.loads(config_path.read_text())
    output_dir = repo / str(config["output_directory"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("write-once closure-sensitivity directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    result, worklist = build_proposal_aware_closure_sensitivity(repo, config_path)
    (output_dir / "leaf_worklist.jsonl.gz").write_bytes(worklist)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
