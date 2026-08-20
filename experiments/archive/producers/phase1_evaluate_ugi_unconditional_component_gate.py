#!/usr/bin/env python3
"""Audit actual component novelty on a frozen unconditional program draw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.corpus.ugi_held_component_gate import evaluate_unconditional_component_gate

REPO = Path(__file__).resolve().parents[3]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--qualified-reactions", type=Path, required=True)
    parser.add_argument("--program-probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--sampling-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    result = evaluate_unconditional_component_gate(
        registry_path=_resolve(args.registry),
        qualified_reactions_path=_resolve(args.qualified_reactions),
        program_probe_path=_resolve(args.program_probe),
        checkpoints=tuple(_resolve(path) for path in args.checkpoint),
        sampling_results=tuple(_resolve(path) for path in args.sampling_result),
        output_path=_resolve(args.output),
        seed=args.seed,
    )
    summary = {
        step: {
            "valid_fraction": value["sampling_statistics"]["valid_fraction"],
            "exact_frozen_products": value["reference_comparison"]["all_frozen_ugi"][
                "exact_matches"
            ],
            "component_novelty": value["product_component_novelty_counts"],
            "all_handles_pass_fraction": value["all_three_handles_pass_fraction"],
        }
        for step, value in result["checkpoints_by_step"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
