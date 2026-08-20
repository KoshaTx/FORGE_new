#!/usr/bin/env python3
"""Freeze the training-fold coarse-program prior for Ugi graph generation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.design.flow.defog_feasibility import sha256_file
from forge.design.corpus.ugi_morphology_corpus import source_stratified_family_weights
from forge.design.flow.ugi_program_prior import (
    build_weighted_program_prior,
    validate_program_prior_support,
)
from forge.design.training.ugi_training_cache import load_ugi_training_cache

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = json.loads(config_path.read_text())
    cache_record = config["inputs"]["prepared_cache"]
    cache_path = Path(cache_record["path"])
    if not cache_path.is_absolute():
        cache_path = REPO / cache_path
    observed_cache_hash = sha256_file(cache_path)
    if observed_cache_hash != cache_record["sha256"]:
        raise RuntimeError("program-prior cache hash changed")
    corpus, records_by_fold = load_ugi_training_cache(cache_path)
    assignments = corpus.assignments_by_fold["train"]
    records = records_by_fold["train"]
    sampling = config["sampling"]
    weights = source_stratified_family_weights(
        assignments,
        source_mass={str(key): float(value) for key, value in sampling["source_mass"].items()},
        uniform_row_mixture=float(sampling["uniform_row_mixture"]),
    )
    prior = build_weighted_program_prior(
        tuple(record.program for record in records),
        weights,
    )
    support_audit = validate_program_prior_support(prior, config["model"])
    if support_audit["status"] != "pass":
        raise RuntimeError("program prior exceeds the frozen model support")
    role_priors = {}
    for role, support, probabilities in zip(
        ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"),
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
    source_mass = {}
    for row, weight in zip(assignments, weights, strict=True):
        source = str(row["source_stratum"])
        source_mass[source] = source_mass.get(source, 0.0) + float(weight)
    result = {
        "schema_version": "phase1_ugi_program_prior.v1",
        "status": "pass",
        "inputs": {
            "training_config": {
                "path": str(config_path.relative_to(REPO)),
                "sha256": sha256_file(config_path),
            },
            "prepared_cache": {
                "path": str(cache_path.relative_to(REPO)),
                "sha256": observed_cache_hash,
            },
        },
        "training_fold_products": len(records),
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
            "observed_source_mass": dict(sorted(source_mass.items())),
        },
        "support_audit": support_audit,
        "role_priors": role_priors,
    }
    output_path = args.output if args.output.is_absolute() else REPO / args.output
    _atomic_json(output_path, result)
    print(output_path)
    print(sha256_file(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
