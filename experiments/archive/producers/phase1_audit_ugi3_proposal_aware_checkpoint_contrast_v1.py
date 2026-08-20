"""Build the frozen proposal-aware checkpoint route contrast artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.archive.phase1.synthesis_value_audits.ugi3_proposal_aware_checkpoint_contrast import (
    build_proposal_aware_checkpoint_contrast,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi3_proposal_aware_checkpoint_contrast_v1.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve()
    config = json.loads(config_path.read_text())
    output_dir = repo / str(config["output_directory"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("write-once proposal-aware contrast directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    result, ledger = build_proposal_aware_checkpoint_contrast(repo, config_path)
    (output_dir / "component_ledger.jsonl.gz").write_bytes(ledger)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
