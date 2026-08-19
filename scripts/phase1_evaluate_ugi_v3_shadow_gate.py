#!/usr/bin/env python3
"""Evaluate one challenger with the unchanged v3 gates, without selecting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.product.ugi_v3_shadow_gate import evaluate_v3_shadow_gate

REPO = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/model/phase1_ugi_v3_nonselecting_shadow_gate_v1.json"),
    )
    parser.add_argument("--challenger-config", type=Path, required=True)
    parser.add_argument("--challenger-config-sha256", required=True)
    parser.add_argument("--challenger-result", type=Path, required=True)
    parser.add_argument("--challenger-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_v3_shadow_gate(
        contract_path=_resolve(args.contract),
        challenger_config_path=_resolve(args.challenger_config),
        challenger_config_sha256=args.challenger_config_sha256,
        challenger_result_path=_resolve(args.challenger_result),
        challenger_result_sha256=args.challenger_result_sha256,
        output_path=_resolve(args.output),
        repository=REPO,
    )
    compact = {
        "status": result["status"],
        "target_candidate_id": result["target_candidate_id"],
        "unchanged_candidate_reproduction": result["unchanged_candidate_reproduction"]["status"],
        "baseline_gate": result["same_step_comparison"]["original_v3_baseline"]["gate"],
        "challenger_gate": result["same_step_comparison"]["challenger"]["gate"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
