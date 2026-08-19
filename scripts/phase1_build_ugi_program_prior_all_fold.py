#!/usr/bin/env python3
"""Build the final all-fold coarse Ugi morphology-program prior."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.potency.ugi_semantic_annotations import ROLE_NAMES
from forge.product.defog_feasibility import sha256_file
from forge.product.ugi_morphology_corpus import source_stratified_family_weights
from forge.product.ugi_program_prior import (
    build_weighted_program_prior,
    validate_program_prior_support,
)
from forge.product.ugi_training_cache import load_ugi_training_cache

REPO = Path(__file__).resolve().parents[1]
ALL_FOLDS = ("train", "calibration", "heldout")


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


def build_all_fold_prior(config_path: Path) -> dict[str, Any]:
    """Build one role-factorized prior from every production-refit record."""

    config = json.loads(config_path.read_text())
    partition = config.get("training_partition", {})
    if (
        config.get("schema_version") != "phase1_ugi_joint_sparse_training_config.v1"
        or partition.get("mode") != "production_refit_all_folds"
        or tuple(partition.get("training_folds", ())) != ALL_FOLDS
    ):
        raise RuntimeError("all-fold program prior requires the frozen production-refit config")

    cache_record = config["inputs"]["prepared_cache"]
    cache_path = Path(cache_record["path"])
    if not cache_path.is_absolute():
        cache_path = REPO / cache_path
    observed_cache_hash = sha256_file(cache_path)
    if observed_cache_hash != cache_record["sha256"]:
        raise RuntimeError("program-prior cache hash changed")
    corpus, records_by_fold = load_ugi_training_cache(cache_path)

    assignments = []
    programs = []
    fold_counts: dict[str, int] = {}
    for fold in ALL_FOLDS:
        fold_assignments = tuple(corpus.assignments_by_fold[fold])
        fold_records = tuple(records_by_fold[fold])
        if len(fold_assignments) != len(fold_records):
            raise RuntimeError(f"assignment/record mismatch in {fold}")
        expected = int(config["expected_fold_counts"][fold])
        if len(fold_records) != expected:
            raise RuntimeError(f"unexpected all-fold program count in {fold}")
        assignments.extend(fold_assignments)
        programs.extend(record.program for record in fold_records)
        fold_counts[fold] = len(fold_records)

    sampling = config["sampling"]
    weights = source_stratified_family_weights(
        assignments,
        source_mass={str(key): float(value) for key, value in sampling["source_mass"].items()},
        uniform_row_mixture=float(sampling["uniform_row_mixture"]),
    )
    prior = build_weighted_program_prior(tuple(programs), weights)
    support_audit = validate_program_prior_support(prior, config["model"])
    if support_audit["status"] != "pass":
        raise RuntimeError("all-fold program prior exceeds the frozen model support")

    role_priors = {}
    for role, support, probabilities in zip(
        ROLE_NAMES,
        prior.support_by_role,
        prior.probabilities_by_role,
        strict=True,
    ):
        role_priors[role] = [
            {
                "node_count": state[0],
                "junction_budget": state[1],
                "cycle_rank": state[2],
                "attachment_count": state[3],
                "probability": float(probability),
            }
            for state, probability in zip(support, probabilities, strict=True)
        ]

    observed_source_mass: dict[str, float] = {}
    for row, weight in zip(assignments, weights, strict=True):
        source = str(row["source_stratum"])
        observed_source_mass[source] = observed_source_mass.get(source, 0.0) + float(weight)

    return {
        "schema_version": "phase1_ugi_program_prior.v1",
        "status": "pass",
        "population": "all_fold_production_refit",
        "inputs": {
            "production_refit_config": {
                "path": str(config_path.relative_to(REPO)),
                "sha256": sha256_file(config_path),
            },
            "prepared_cache": {
                "path": str(cache_path.relative_to(REPO)),
                "sha256": observed_cache_hash,
            },
        },
        "folds": list(ALL_FOLDS),
        "fold_counts": fold_counts,
        "products": len(programs),
        "bounds": {
            key: int(config["model"][key])
            for key in (
                "maximum_component_atoms",
                "maximum_total_atoms",
                "maximum_junction_budget",
                "maximum_cycle_rank",
                "maximum_attachment_count",
            )
        },
        "sampling_policy": {
            "estimator": "source_stratified_family_raked_product_program_marginals",
            "role_programs_sampled_independently": True,
            "component_ids_enter_neural_or_program_state": False,
            "whole_component_graphs_selected": False,
            "observed_source_mass": dict(sorted(observed_source_mass.items())),
        },
        "support_audit": support_audit,
        "role_priors": role_priors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_joint_sparse_balanced_v2_production_refit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_program_prior_v3_all_fold.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    output_path = args.output if args.output.is_absolute() else REPO / args.output
    _atomic_json(output_path, build_all_fold_prior(config_path.resolve()))
    print(output_path)
    print(sha256_file(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
