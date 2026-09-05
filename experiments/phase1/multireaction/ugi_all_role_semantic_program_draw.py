"""Attach complete identity-free role semantics to the frozen Ugi program sequence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.ugi_all_role_semantic_program import UgiMeasuredAllRoleSemanticPrior
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_program import UgiMorphologyProgram

CONFIG_SCHEMA = "forge.ugi_all_role_semantic_program_draw_config.v1"
RESULT_SCHEMA = "forge.ugi_all_role_semantic_program_draw.v1"
AMINE_DRAW_SCHEMA = "forge.ugi_amine_semantic_program_draw.v1"


class UgiAllRoleSemanticProgramDrawError(ValueError):
    """The all-role draw contract or one pinned dependency changed."""


def _load_frozen_draw(
    path: Path,
    *,
    count: int,
) -> tuple[
    tuple[UgiMorphologyProgram, ...],
    tuple[UgiAmineSemanticTarget, ...],
    Mapping[str, Any],
]:
    document = read_json_object(
        path,
        error=UgiAllRoleSemanticProgramDrawError,
        label="frozen Ugi amine-semantic program draw",
    )
    rows = document.get("samples")
    if (
        document.get("schema_version") != AMINE_DRAW_SCHEMA
        or document.get("status") != "pass"
        or document.get("reaction_id") != "ugi_3cr_agile"
        or document.get("component_identity_conditioning") is not False
        or document.get("heldout_access") is not False
        or document.get("candidate_selection") is not False
        or not isinstance(rows, list)
        or count < 1
        or len(rows) < count
    ):
        raise UgiAllRoleSemanticProgramDrawError("frozen semantic program draw changed")
    programs: list[UgiMorphologyProgram] = []
    targets: list[UgiAmineSemanticTarget] = []
    for index, row in enumerate(rows[:count]):
        if not isinstance(row, Mapping) or row.get("sample_index") != index:
            raise UgiAllRoleSemanticProgramDrawError("frozen program rows are not ordered")
        raw_program = row.get("program")
        raw_target = row.get("amine_semantic_target")
        if not isinstance(raw_program, Mapping) or not isinstance(raw_target, Mapping):
            raise UgiAllRoleSemanticProgramDrawError("frozen program row changed")
        programs.append(
            UgiMorphologyProgram(
                node_counts=tuple(int(value) for value in raw_program["node_counts"]),
                junction_budgets=tuple(
                    int(value) for value in raw_program["junction_budgets"]
                ),
                cycle_ranks=tuple(int(value) for value in raw_program["cycle_ranks"]),
                attachment_counts=tuple(
                    int(value) for value in raw_program["attachment_counts"]
                ),
            )
        )
        targets.append(UgiAmineSemanticTarget.from_mapping(raw_target))
    return tuple(programs), tuple(targets), document


def _mean_targets(targets: tuple[Any, ...]) -> dict[str, float]:
    rows: list[dict[str, int]] = []
    for target in targets:
        mapping = target.to_mapping()
        rows.append({
            **{f"amine.{key}": value for key, value in mapping["amine"].items()},
            **{f"tail_pair.{key}": value for key, value in mapping["tail_pair"].items()},
        })
    fields = tuple(rows[0])
    matrix = np.asarray([[row[field] for field in fields] for row in rows], dtype=np.float64)
    return {field: float(matrix[:, index].mean()) for index, field in enumerate(fields)}


def run_ugi_all_role_semantic_program_draw(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compile the measured-train joint prior and sample once per frozen coarse program."""

    config = read_json_object(
        config_path,
        error=UgiAllRoleSemanticProgramDrawError,
        label="Ugi all-role semantic program-draw config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != {
        "schema_version",
        "scientific_question",
        "inputs",
        "reaction_id",
        "sample_count",
        "semantic_seed",
        "policy",
        "nonclaims",
    }:
        raise UgiAllRoleSemanticProgramDrawError("all-role program-draw config changed")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {
        "amine_semantic_program_draw",
        "production_cache",
        "qualified_reactions",
        "ugi_assignments",
    }:
        raise UgiAllRoleSemanticProgramDrawError("all-role program-draw inputs changed")
    if config["policy"] != {
        "coarse_program_sequence": "frozen amine-semantic joint-support draw",
        "semantic_weighting": (
            "equal component-family-triple mass within exact complete coarse program"
        ),
        "sampling": "one joint categorical draw per attempt without rejection or retry",
        "component_or_family_ids_enter_program": False,
        "heldout_access": False,
    }:
        raise UgiAllRoleSemanticProgramDrawError("all-role semantic policy changed")
    count = config["sample_count"]
    seed = config["semantic_seed"]
    if (
        config["reaction_id"] != "ugi_3cr_agile"
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise UgiAllRoleSemanticProgramDrawError("all-role semantic runtime changed")
    inputs = {
        label: resolve_pin(pin, repo, label=label) for label, pin in sorted(raw_inputs.items())
    }
    programs, baseline_targets, source_draw = _load_frozen_draw(
        inputs["amine_semantic_program_draw"], count=count
    )
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        reaction_id=str(config["reaction_id"]),
        expected_sha256=str(raw_inputs["qualified_reactions"]["sha256"]),
        expected_training_assignments_sha256=str(raw_inputs["ugi_assignments"]["sha256"]),
    )
    cache = SynthesisProgramProductionCache(inputs["production_cache"])
    try:
        prior = UgiMeasuredAllRoleSemanticPrior.from_training_data(
            assignments_path=inputs["ugi_assignments"],
            cache=cache,
            ester_policy=ester_policy,
            reaction_id=str(config["reaction_id"]),
            expected_assignments_sha256=str(raw_inputs["ugi_assignments"]["sha256"]),
        )
        targets = prior.sample_for_programs(programs, seed=seed)
    finally:
        cache.close()
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "scientific_question": config["scientific_question"],
        "reaction_id": config["reaction_id"],
        "source_coarse_program_seed": source_draw["coarse_program_seed"],
        "source_amine_semantic_seed": source_draw["semantic_seed"],
        "all_role_semantic_seed": seed,
        "rows": len(programs),
        "config": pin_record(config_path, repo),
        "implementation": pin_record(Path(__file__), repo),
        "prior_implementation": pin_record(
            repo / "forge/model/ugi_all_role_semantic_program.py", repo
        ),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "prior_audit": dict(prior.audit),
        "sample_target_means": _mean_targets(targets),
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
                "baseline_amine_semantic_target": baseline.to_mapping(),
                "all_role_semantic_target": target.to_mapping(),
            }
            for index, (program, baseline, target) in enumerate(
                zip(programs, baseline_targets, targets, strict=True)
            )
        ],
        "training_generation_repair_retry_route_or_oracle_calls": 0,
        "candidate_selection": False,
        "heldout_access": False,
        "nonclaims": config["nonclaims"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise UgiAllRoleSemanticProgramDrawError(f"output already exists: {output_path}")
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_ugi_all_role_semantic_program_draw(
        args.config.resolve(), args.repo.resolve(), args.output.resolve()
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiAllRoleSemanticProgramDrawError",
    "run_ugi_all_role_semantic_program_draw",
]
