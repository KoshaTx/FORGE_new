from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.potency.ranking.ugi_production_full_support_rescoring_v3 import (
    build_full_support_rescoring_v3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_production_full_support_rescoring_v3.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_production_full_support_rescoring_v3"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    result, ledger = build_full_support_rescoring_v3(repo, repo / args.config)
    (output_dir / "terminal_rescoring.csv.gz").write_bytes(ledger)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
