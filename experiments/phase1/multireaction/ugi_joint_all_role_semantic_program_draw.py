"""Draw coherent identity-free Ugi programs and complete role semantics from training data."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.ugi_all_role_semantic_program import (
    UgiMeasuredJointAllRoleSemanticPrior,
)
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy

CONFIG_SCHEMA = "forge.ugi_joint_all_role_semantic_program_draw_config.v2"
SUBSTITUTION_CONFIG_SCHEMA = "forge.ugi_joint_all_role_substitution_semantic_program_draw_config.v3"
# The row wire format intentionally remains compatible with the frozen paired comparison loader.
RESULT_SCHEMA = "forge.ugi_all_role_semantic_program_draw.v1"


class UgiJointAllRoleSemanticProgramDrawError(ValueError):
    """The joint program/semantic draw contract or a pinned input changed."""


def _mean_targets(targets: tuple[Any, ...]) -> dict[str, float]:
    rows: list[dict[str, int]] = []
    for target in targets:
        mapping = target.to_mapping()
        rows.append(
            {
                **{f"amine.{key}": value for key, value in mapping["amine"].items()},
                **{f"tail_pair.{key}": value for key, value in mapping["tail_pair"].items()},
            }
        )
    fields = tuple(rows[0])
    matrix = np.asarray([[row[field] for field in fields] for row in rows], dtype=np.float64)
    return {field: float(matrix[:, index].mean()) for index, field in enumerate(fields)}


def _weighted_mean_targets(targets: tuple[Any, ...], probabilities: np.ndarray) -> dict[str, float]:
    rows: list[dict[str, int]] = []
    for target in targets:
        mapping = target.to_mapping()
        rows.append(
            {
                **{f"amine.{key}": value for key, value in mapping["amine"].items()},
                **{f"tail_pair.{key}": value for key, value in mapping["tail_pair"].items()},
            }
        )
    fields = tuple(rows[0])
    matrix = np.asarray([[row[field] for field in fields] for row in rows], dtype=np.float64)
    means = probabilities @ matrix
    return {field: float(means[index]) for index, field in enumerate(fields)}


def run_ugi_joint_all_role_semantic_program_draw(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compile and draw one coherent train-only joint program law."""

    config = read_json_object(
        config_path,
        error=UgiJointAllRoleSemanticProgramDrawError,
        label="Ugi joint all-role semantic program-draw config",
    )
    schema = config.get("schema_version")
    if schema not in {CONFIG_SCHEMA, SUBSTITUTION_CONFIG_SCHEMA} or set(config) != {
        "schema_version",
        "scientific_question",
        "inputs",
        "reaction_id",
        "sample_count",
        "seed",
        "policy",
        "nonclaims",
    }:
        raise UgiJointAllRoleSemanticProgramDrawError("joint program-draw config changed")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {
        "production_cache",
        "qualified_reactions",
        "ugi_assignments",
    }:
        raise UgiJointAllRoleSemanticProgramDrawError("joint program-draw inputs changed")
    expected_policy = {
        "weighting": "equal_family_group_then_equal_eligible_product_joint_program_semantics",
        "sampling": "one joint categorical draw per attempt without rejection or retry",
        "paired_amine_target": "identical_between_baseline_and_treatment",
        "aldehyde_direction": "reactive_aldehyde_on_alkoxy_ester_side",
        "component_or_family_ids_enter_program": False,
        "heldout_access": False,
    }
    if schema == SUBSTITUTION_CONFIG_SCHEMA:
        expected_policy["amine_substitution_semantics"] = (
            "exact_hydrogen_bond_donors_and_heavy_branch_atoms"
        )
    if config["policy"] != expected_policy:
        raise UgiJointAllRoleSemanticProgramDrawError("joint program-draw policy changed")
    count = config["sample_count"]
    seed = config["seed"]
    if (
        config["reaction_id"] != "ugi_3cr_agile"
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise UgiJointAllRoleSemanticProgramDrawError("joint program-draw runtime changed")
    inputs = {
        label: resolve_pin(pin, repo, label=label) for label, pin in sorted(raw_inputs.items())
    }
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        reaction_id=str(config["reaction_id"]),
        expected_sha256=str(raw_inputs["qualified_reactions"]["sha256"]),
        expected_training_assignments_sha256=str(raw_inputs["ugi_assignments"]["sha256"]),
    )
    cache = SynthesisProgramProductionCache(inputs["production_cache"])
    try:
        prior = UgiMeasuredJointAllRoleSemanticPrior.from_training_data(
            assignments_path=inputs["ugi_assignments"],
            cache=cache,
            ester_policy=ester_policy,
            reaction_id=str(config["reaction_id"]),
            expected_assignments_sha256=str(raw_inputs["ugi_assignments"]["sha256"]),
            include_amine_substitution_semantics=(schema == SUBSTITUTION_CONFIG_SCHEMA),
        )
        programs, targets = prior.sample(count=count, seed=seed)
    finally:
        cache.close()
    sample_target_means = _mean_targets(targets)
    analytic_target_means = _weighted_mean_targets(
        tuple(target for _, target in prior.support), prior.probabilities
    )
    maximum_mean_error = max(
        abs(sample_target_means[field] - analytic_target_means[field])
        for field in analytic_target_means
    )
    if maximum_mean_error > 0.25:
        raise UgiJointAllRoleSemanticProgramDrawError(
            "sampled joint program law does not reproduce its analytic target means"
        )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "scientific_question": config["scientific_question"],
        "reaction_id": config["reaction_id"],
        "joint_program_semantic_seed": seed,
        "source_coarse_program_seed": seed,
        "source_amine_semantic_seed": seed,
        "all_role_semantic_seed": seed,
        "rows": len(programs),
        "config": pin_record(config_path, repo),
        "implementation": pin_record(Path(__file__), repo),
        "prior_implementation": pin_record(
            repo / "forge/model/ugi_all_role_semantic_program.py", repo
        ),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "prior_audit": dict(prior.audit),
        "analytic_target_means": analytic_target_means,
        "sample_target_means": sample_target_means,
        "sampled_vs_analytic_max_abs_mean_error": float(maximum_mean_error),
        "sampled_vs_analytic_mean_gate": {"maximum_absolute_error": 0.25, "pass": True},
        "paired_amine_targets_identical": schema != SUBSTITUTION_CONFIG_SCHEMA,
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
                # The baseline remains the frozen four-coordinate amine program.  A v3 target's
                # substitution fields enter only the treatment's complete all-role target, so the
                # paired comparison changes exactly one semantic intervention.
                "baseline_amine_semantic_target": {
                    field: target.amine.to_mapping()[field]
                    for field in (
                        "heavy_atom_graph_diameter",
                        "carbon_skeleton_diameter",
                        "nitrogen_atoms",
                        "oxygen_atoms",
                    )
                },
                "all_role_semantic_target": target.to_mapping(),
            }
            for index, (program, target) in enumerate(zip(programs, targets, strict=True))
        ],
        "training_generation_repair_retry_route_or_oracle_calls": 0,
        "candidate_selection": False,
        "heldout_access": False,
        "nonclaims": config["nonclaims"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise UgiJointAllRoleSemanticProgramDrawError(f"output already exists: {output_path}")
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_ugi_joint_all_role_semantic_program_draw(
        args.config.resolve(), args.repo.resolve(), args.output.resolve()
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "SUBSTITUTION_CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiJointAllRoleSemanticProgramDrawError",
    "run_ugi_joint_all_role_semantic_program_draw",
]
