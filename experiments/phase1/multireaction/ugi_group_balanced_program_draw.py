"""Build an identity-free group-balanced measured-Ugi morphology-program draw."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_measured_joint_program_prior import (
    UgiMeasuredJointProgramPrior,
    summarize_program_distribution,
)

CONFIG_SCHEMA = "forge.ugi_group_balanced_program_draw_config.v1"
RESULT_SCHEMA = "forge.ugi_measured_joint_morphology_program_draw.v2"


class UgiGroupBalancedProgramDrawError(ValueError):
    """The group-balanced program-draw contract is malformed."""


def run_ugi_group_balanced_program_draw(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compile and sample the train-fold-only group-balanced semantic prior."""

    config = read_json_object(
        config_path,
        error=UgiGroupBalancedProgramDrawError,
        label="group-balanced Ugi program-draw config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != {
        "schema_version",
        "scientific_question",
        "inputs",
        "reaction_id",
        "sample_count",
        "seed",
        "policy",
        "nonclaims",
    }:
        raise UgiGroupBalancedProgramDrawError("program-draw config fields changed")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {
        "production_cache",
        "qualified_reactions",
        "ugi_assignments",
    }:
        raise UgiGroupBalancedProgramDrawError("program-draw inputs changed")
    inputs = {label: resolve_pin(pin, repo, label=label) for label, pin in raw_inputs.items()}
    reaction_id = str(config["reaction_id"])
    sample_count = config["sample_count"]
    seed = config["seed"]
    policy = config["policy"]
    if (
        reaction_id != "ugi_3cr_agile"
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or policy
        != {
            "component_or_family_ids_enter_program": False,
            "heldout_access": False,
            "sampling": "one categorical draw per attempt without rejection or retry",
            "weighting": "equal_family_group_then_equal_eligible_product",
        }
    ):
        raise UgiGroupBalancedProgramDrawError("program-draw policy changed")
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        reaction_id=reaction_id,
        expected_sha256=str(raw_inputs["qualified_reactions"]["sha256"]),
        expected_training_assignments_sha256=str(raw_inputs["ugi_assignments"]["sha256"]),
    )
    cache = SynthesisProgramProductionCache(inputs["production_cache"])
    try:
        prior = UgiMeasuredJointProgramPrior.from_training_data(
            assignments_path=inputs["ugi_assignments"],
            cache=cache,
            ester_policy=ester_policy,
            reaction_id=reaction_id,
            expected_assignments_sha256=str(raw_inputs["ugi_assignments"]["sha256"]),
        )
        programs = prior.sample(count=sample_count, seed=seed)
    finally:
        cache.close()
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "scientific_question": config["scientific_question"],
        "source": "group_balanced_measured_training_joint_count_prior",
        "reaction_id": reaction_id,
        "seed": seed,
        "rows": len(programs),
        "config": pin_record(config_path, repo),
        "implementation": pin_record(Path(__file__), repo),
        "prior_implementation": pin_record(
            repo / "forge/model/ugi_measured_joint_program_prior.py", repo
        ),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "prior_audit": dict(prior.audit),
        "sample_program_distribution_projection": summarize_program_distribution(
            programs,
            np.full(len(programs), 1.0 / len(programs), dtype=np.float64),
        ),
        "component_identity_conditioning": False,
        "samples": [
            {
                "sample_index": index,
                "program": {
                    "attachment_counts": list(program.attachment_counts),
                    "cycle_ranks": list(program.cycle_ranks),
                    "junction_budgets": list(program.junction_budgets),
                    "node_counts": list(program.node_counts),
                },
            }
            for index, program in enumerate(programs)
        ],
        "training_generation_repair_retry_route_or_oracle_calls": 0,
        "candidate_selection": False,
        "heldout_access": False,
        "nonclaims": config["nonclaims"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise UgiGroupBalancedProgramDrawError(f"output already exists: {output_path}")
    write_json(output_path, result)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    run_ugi_group_balanced_program_draw(
        args.config.resolve(),
        args.repo.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiGroupBalancedProgramDrawError",
    "run_ugi_group_balanced_program_draw",
]
