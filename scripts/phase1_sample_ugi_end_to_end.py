#!/usr/bin/env python3
"""Compose current Ugi morphology, closure, and chemistry checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.product.ugi_end_to_end_sampling import (
    UgiEndToEndSamplingError,
    sample_ugi_end_to_end,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--morphology-checkpoint",
        type=Path,
        default=(
            REPO / "results/phase1/ugi_morphology_expanded_regularized_full/checkpoint_best.pt"
        ),
    )
    parser.add_argument(
        "--closure-checkpoint",
        type=Path,
        default=REPO / "results/phase1/ugi_closure_smoke/checkpoint_latest.pt",
    )
    parser.add_argument(
        "--chemistry-checkpoint",
        type=Path,
        default=REPO / "results/phase1/ugi_chemistry_smoke/checkpoint_latest.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi_end_to_end_smoke",
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--morphology-steps", type=int, default=8)
    parser.add_argument("--chemistry-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--program-probe", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = sample_ugi_end_to_end(
            REPO,
            args.output_dir.resolve(),
            morphology_checkpoint_path=args.morphology_checkpoint.resolve(),
            closure_checkpoint_path=args.closure_checkpoint.resolve(),
            chemistry_checkpoint_path=args.chemistry_checkpoint.resolve(),
            sample_count=args.samples,
            morphology_steps=args.morphology_steps,
            chemistry_steps=args.chemistry_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            overwrite=args.overwrite,
            program_probe_path=(
                args.program_probe.resolve() if args.program_probe is not None else None
            ),
        )
    except (KeyError, TypeError, UgiEndToEndSamplingError) as exc:
        print(f"Ugi end-to-end sampling failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "statistics": result["statistics"],
                "sampling": result["sampling"],
                "render": result["render"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
