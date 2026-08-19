#!/usr/bin/env python3
"""Evaluate checkpoint chemistry against exact role-program training cells."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_joint_sparse_training import _training_weights
from forge.product.ugi_program_matched_tail_chemistry import (
    compare_program_matched_tail_chemistry,
)
from forge.product.ugi_training_cache import load_ugi_training_cache

REPO = Path(__file__).resolve().parents[1]
REFERENCE_FOLDS = ("train", "calibration")
SAMPLING = {
    "mode": "source_stratified_role_family_raked",
    "source_mass": {
        "current_phase1_union": 0.5,
        "expanded_exact_forward_enumeration": 0.5,
    },
    "uniform_row_mixture": 0.5,
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt",
    )
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO
            / "results/phase1/ugi_decoration_coupling_checkpoint_screen_v1"
            / "program_matched_tail_chemistry_all_v1.json"
        ),
    )
    args = parser.parse_args()
    corpus, joint_records_by_fold = load_ugi_training_cache(args.cache)
    assignments = tuple(row for fold in REFERENCE_FOLDS for row in corpus.assignments_by_fold[fold])
    joint_records = tuple(row for fold in REFERENCE_FOLDS for row in joint_records_by_fold[fold])
    if any(
        str(assignment["product_id"]) != str(record.product_id)
        for assignment, record in zip(assignments, joint_records, strict=True)
    ):
        raise RuntimeError("training assignment and joint-record order differs")
    weights = _training_weights(assignments, SAMPLING)

    arms: dict[str, Any] = {}
    inputs: dict[str, Any] = {
        "training_cache": {"path": str(args.cache), "sha256": sha256_file(args.cache)}
    }
    for specification in args.sample:
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        sample = json.loads(path.read_text())
        arms[name] = compare_program_matched_tail_chemistry(
            sample=sample,
            assignments=assignments,
            joint_records=joint_records,
            training_weights=weights,
        )
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}

    output = {
        "schema_version": "phase1_ugi_program_matched_tail_chemistry_audit.v1",
        "status": "complete_descriptive_checkpoint_audit",
        "reference_folds": list(REFERENCE_FOLDS),
        "heldout_fold_used": False,
        "sampling": SAMPLING,
        "inputs": inputs,
        "arms": arms,
    }
    _atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "arms": {
                    name: {
                        "all_feature_mae": value[
                            "mean_absolute_error_across_tail_roles_all_features"
                        ],
                        "local_chemistry_mae": value[
                            "mean_absolute_error_across_tail_roles_local_chemistry"
                        ],
                        "minimum_exact_program_match_fraction": min(
                            role["exact_match_fraction"] for role in value["by_role"].values()
                        ),
                    }
                    for name, value in arms.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
