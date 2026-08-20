#!/usr/bin/env python3
"""Evaluate held-component generalization at frozen joint-flow checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.corpus.ugi_held_component_gate import evaluate_held_component_gate

REPO = Path(__file__).resolve().parents[3]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--qualified-reactions", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--sampling-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    result = evaluate_held_component_gate(
        cache_path=_resolve(args.cache),
        registry_path=_resolve(args.registry),
        qualified_reactions_path=_resolve(args.qualified_reactions),
        probe_path=_resolve(args.probe),
        checkpoints=tuple(_resolve(path) for path in args.checkpoint),
        sampling_results=tuple(_resolve(path) for path in args.sampling_result),
        output_path=_resolve(args.output),
        seed=args.seed,
        batch_size=args.batch_size,
    )
    summary = {
        step: {
            "full_heldout_loss": value["heldout_loss"]["full_heldout"]["record_weighted_metrics"][
                "total"
            ],
            "probe_loss": value["heldout_loss"]["frozen_probe"]["record_weighted_metrics"]["total"],
            "component_novelty": value["generated_component_analysis"][
                "product_component_novelty_counts"
            ],
            "all_handles_pass_fraction": value["generated_component_analysis"][
                "all_three_handles_pass_fraction"
            ],
        }
        for step, value in result["checkpoints_by_step"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
