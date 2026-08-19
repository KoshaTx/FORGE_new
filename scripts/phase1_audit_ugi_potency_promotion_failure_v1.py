from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.potency.ugi_potency_promotion_failure_audit import (
    audit_potency_promotion_failure,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_potency_promotion_failure_audit_v1"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (repo / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    result = audit_potency_promotion_failure(
        confirmatory_path=repo / "results/phase1/ugi_morphology_potency_matched_v1/result.json",
        continuous_path=repo
        / "results/phase1/ugi_continuous_novelty_matched_ranking_v1/result.json",
        signal_path=repo / "results/phase1/ugi_morphology_potency_signal_v1/result.json",
    )
    (output / "result.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
