"""Freeze an entropy-calibrated joint Ugi morphology and semantic program draw."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import pin_record, resolve_pin, sha256_json
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.ugi_all_role_semantic_program import UgiMeasuredJointAllRoleSemanticPrior
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_diversity import (
    sample_tempered_indices,
    select_morphology_temperature,
    summarize_sampled_modes,
)

CONFIG_SCHEMA = "forge.ugi_morphology_diversity_program_draw_config.v1"
# Retain the established row wire format so downstream sampling has one canonical loader.
RESULT_SCHEMA = "forge.ugi_all_role_semantic_program_draw.v1"


class UgiMorphologyDiversityProgramDrawError(ValueError):
    """The diversity-calibrated program draw or one of its pins changed."""


def _validate_config(value: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "scientific_question",
        "inputs",
        "reaction_id",
        "sample_count",
        "seed",
        "temperature_selection",
        "policy",
        "nonclaims",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise UgiMorphologyDiversityProgramDrawError("morphology-diversity config changed")
    if value.get("schema_version") != CONFIG_SCHEMA or value.get("reaction_id") != "ugi_3cr_agile":
        raise UgiMorphologyDiversityProgramDrawError("morphology-diversity schema changed")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "production_cache",
        "qualified_reactions",
        "ugi_assignments",
    }:
        raise UgiMorphologyDiversityProgramDrawError("morphology-diversity inputs changed")
    count = value.get("sample_count")
    seed = value.get("seed")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 256
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise UgiMorphologyDiversityProgramDrawError("morphology-diversity runtime changed")
    selection = value.get("temperature_selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "calibration_attempts",
        "candidate_temperatures",
        "maximum_kl_from_measured_prior",
        "minimum_expected_unique_ratio",
    }:
        raise UgiMorphologyDiversityProgramDrawError(
            "morphology temperature-selection policy changed"
        )
    candidates = selection.get("candidate_temperatures")
    if (
        selection.get("calibration_attempts") != 256
        or not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
    ):
        raise UgiMorphologyDiversityProgramDrawError(
            "morphology temperature-selection runtime changed"
        )
    if value.get("policy") != {
        "candidate_selection": False,
        "component_or_family_identity_conditioning": False,
        "heldout_access": False,
        "iid_attempts": True,
        "repair_or_retry": False,
        "support": "unchanged_source_adjudicated_measured_train_joint_modes",
        "temperature_selected_analytically_before_generation": True,
    }:
        raise UgiMorphologyDiversityProgramDrawError("morphology-diversity policy changed")
    return dict(value)


def run_ugi_morphology_diversity_program_draw(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compile the train-only prior, flatten it within a KL budget, and draw IID requests."""

    config = _validate_config(
        read_json_object(
            config_path,
            error=UgiMorphologyDiversityProgramDrawError,
            label="Ugi morphology-diversity program-draw config",
        )
    )
    inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(config["inputs"].items())
    }
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        reaction_id=str(config["reaction_id"]),
        expected_sha256=str(config["inputs"]["qualified_reactions"]["sha256"]),
        expected_training_assignments_sha256=str(config["inputs"]["ugi_assignments"]["sha256"]),
    )
    cache = SynthesisProgramProductionCache(inputs["production_cache"])
    try:
        prior = UgiMeasuredJointAllRoleSemanticPrior.from_training_data(
            assignments_path=inputs["ugi_assignments"],
            cache=cache,
            ester_policy=ester_policy,
            reaction_id=str(config["reaction_id"]),
            expected_assignments_sha256=str(config["inputs"]["ugi_assignments"]["sha256"]),
            include_amine_substitution_semantics=True,
        )
    finally:
        cache.close()
    selection_config = config["temperature_selection"]
    selection = select_morphology_temperature(
        prior.probabilities,
        count=int(selection_config["calibration_attempts"]),
        candidate_temperatures=selection_config["candidate_temperatures"],
        minimum_expected_unique_ratio=float(selection_config["minimum_expected_unique_ratio"]),
        maximum_kl=float(selection_config["maximum_kl_from_measured_prior"]),
    )
    indices = sample_tempered_indices(
        selection.probabilities,
        count=int(config["sample_count"]),
        seed=int(config["seed"]),
    )
    selected = tuple(prior.support[int(index)] for index in indices.tolist())
    calibration_indices = indices[: int(selection_config["calibration_attempts"])]
    support_payload = [
        {
            "program": {
                "attachment_counts": list(program.attachment_counts),
                "cycle_ranks": list(program.cycle_ranks),
                "junction_budgets": list(program.junction_budgets),
                "node_counts": list(program.node_counts),
            },
            "target": target.to_mapping(),
        }
        for program, target in prior.support
    ]
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "scientific_question": config["scientific_question"],
        "reaction_id": config["reaction_id"],
        "joint_program_semantic_seed": int(config["seed"]),
        "source_coarse_program_seed": int(config["seed"]),
        "source_amine_semantic_seed": int(config["seed"]),
        "all_role_semantic_seed": int(config["seed"]),
        "rows": len(selected),
        "config": pin_record(config_path, repo),
        "implementation": pin_record(Path(__file__), repo),
        "diversity_implementation": pin_record(
            repo / "forge/model/ugi_morphology_diversity.py", repo
        ),
        "prior_implementation": pin_record(
            repo / "forge/model/ugi_all_role_semantic_program.py", repo
        ),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "prior_audit": dict(prior.audit),
        "support_sha256": str(sha256_json(support_payload)),
        "temperature_selection": selection.to_mapping(),
        "calibration_draw_summary": summarize_sampled_modes(
            calibration_indices, support_size=len(prior.support)
        ),
        "full_draw_summary": summarize_sampled_modes(indices, support_size=len(prior.support)),
        "paired_amine_targets_identical": True,
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
                "baseline_amine_semantic_target": target.amine.to_mapping(),
                "all_role_semantic_target": target.to_mapping(),
            }
            for index, (program, target) in enumerate(selected)
        ],
        "training_generation_repair_retry_route_or_oracle_calls": 0,
        "candidate_selection": False,
        "heldout_access": False,
        "nonclaims": config["nonclaims"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise UgiMorphologyDiversityProgramDrawError(f"output already exists: {output_path}")
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_ugi_morphology_diversity_program_draw(
        args.config.resolve(), args.repo.resolve(), args.output.resolve()
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiMorphologyDiversityProgramDrawError",
    "run_ugi_morphology_diversity_program_draw",
]
