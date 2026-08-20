#!/usr/bin/env python3
"""Build a component-family-balanced morphology-prior challenger."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES
from forge.design.flow.ugi_program_prior import (
    blend_program_priors,
    build_component_family_balanced_program_prior,
    load_program_prior,
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


def _resolve(specification: dict[str, str]) -> Path:
    path = Path(specification["path"])
    if not path.is_absolute():
        path = REPO / path
    if sha256_file(path) != specification["sha256"]:
        raise RuntimeError(f"input hash changed: {path}")
    return path


def _positive_junction_fractions(prior: Any) -> dict[str, float]:
    return {
        role: sum(
            float(probability)
            for state, probability in zip(support, probabilities, strict=True)
            if state[1] > 0
        )
        for role, support, probabilities in zip(
            ROLE_NAMES,
            prior.support_by_role,
            prior.probabilities_by_role,
            strict=True,
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "phase1_ugi_program_prior_challenger_config.v1":
        raise RuntimeError("unsupported program-prior challenger config")
    cache_path = _resolve(config["inputs"]["prepared_cache"])
    baseline_path = _resolve(config["inputs"]["baseline_program_prior"])
    corpus, records_by_fold = load_ugi_training_cache(cache_path)
    assignments = corpus.assignments_by_fold["train"]
    records = records_by_fold["train"]
    component_keys = tuple(
        {role: str(row[f"{role}_smiles"]) for role in ROLE_NAMES} for row in assignments
    )
    component_families = tuple(
        {role: str(row[f"{role}_family_id"]) for role in ROLE_NAMES} for row in assignments
    )
    exploration, component_counts = build_component_family_balanced_program_prior(
        tuple(record.program for record in records),
        component_keys,
        component_families,
    )
    baseline, baseline_bounds = load_program_prior(baseline_path)
    bounds = {str(key): int(value) for key, value in config["bounds"].items()}
    if bounds != baseline_bounds:
        raise RuntimeError("challenger bounds differ from the production prior")
    exploration_mass = float(config["estimator"]["component_family_balanced_mass"])
    challenger = blend_program_priors(
        baseline,
        exploration,
        exploration_mass=exploration_mass,
    )
    support_audit = validate_program_prior_support(challenger, bounds)
    if support_audit["status"] != "pass":
        raise RuntimeError("challenger prior exceeds frozen model support")
    role_priors = {}
    for role, support, probabilities in zip(
        ROLE_NAMES,
        challenger.support_by_role,
        challenger.probabilities_by_role,
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
    output = {
        "schema_version": "phase1_ugi_program_prior.v1",
        "status": "pass",
        "inputs": {
            "config": {
                "path": str(config_path.relative_to(REPO)),
                "sha256": sha256_file(config_path),
            },
            "prepared_cache": config["inputs"]["prepared_cache"],
            "baseline_program_prior": config["inputs"]["baseline_program_prior"],
        },
        "training_fold_products": len(records),
        "bounds": bounds,
        "sampling_policy": {
            "estimator": "fixed_mixture_of_production_and_component_family_balanced_priors",
            "production_prior_mass": 1.0 - exploration_mass,
            "component_family_balanced_mass": exploration_mass,
            "role_programs_sampled_independently": True,
            "component_ids_enter_neural_or_program_state": False,
            "whole_component_graphs_selected": False,
            "training_unique_components_by_role": component_counts,
        },
        "diagnostic": {
            "positive_junction_fraction_by_estimator": {
                "production": _positive_junction_fractions(baseline),
                "component_family_balanced": _positive_junction_fractions(exploration),
                "challenger_mixture": _positive_junction_fractions(challenger),
            }
        },
        "support_audit": support_audit,
        "role_priors": role_priors,
    }
    output_path = args.output if args.output.is_absolute() else REPO / args.output
    _atomic_json(output_path, output)
    print(output_path)
    print(sha256_file(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
