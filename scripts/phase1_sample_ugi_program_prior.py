#!/usr/bin/env python3
"""Freeze unconditional coarse Ugi programs from the training-fold prior."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from forge.design.flow.defog_feasibility import sha256_file
from forge.design.flow.ugi_program_prior import (
    load_program_prior,
    sample_weighted_program_prior,
)

REPO = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _branch_class(junctions: tuple[int, int, int]) -> str:
    aldehyde = junctions[1] > 0
    isocyanide = junctions[2] > 0
    if aldehyde and isocyanide:
        return "both_tail_origins_branched"
    if aldehyde:
        return "aldehyde_origin_branched"
    if isocyanide:
        return "isocyanide_origin_branched"
    return "linear_tail_origins"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    prior_path = args.prior if args.prior.is_absolute() else REPO / args.prior
    prior, bounds = load_program_prior(prior_path)
    programs = sample_weighted_program_prior(
        prior,
        count=args.count,
        rng=np.random.default_rng(args.seed),
        bounds=bounds,
    )
    rows = [
        {
            "product_id": f"program-prior-{index:06d}",
            "source_stratum": "training_fold_weighted_program_prior",
            "branch_class": _branch_class(program.junction_budgets),
            "component_novelty_class": "unconditioned_program_prior",
            "program": program.__dict__,
        }
        for index, program in enumerate(programs)
    ]
    result = {
        "schema_version": "phase1_ugi_program_probe.v1",
        "seed": args.seed,
        "fold": "training_program_prior",
        "input_prior": {
            "path": str(prior_path.relative_to(REPO)),
            "sha256": sha256_file(prior_path),
        },
        "samples": rows,
        "stratum_branch_counts": dict(sorted(Counter(row["branch_class"] for row in rows).items())),
    }
    output_path = args.output if args.output.is_absolute() else REPO / args.output
    _atomic_json(output_path, result)
    print(output_path)
    print(sha256_file(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
