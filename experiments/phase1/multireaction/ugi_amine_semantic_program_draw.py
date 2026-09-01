"""Attach identity-free measured amine semantics to a frozen coarse Ugi program draw."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.ugi_amine_semantic_program import UgiMeasuredAmineSemanticPrior
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_program import UgiMorphologyProgram

CONFIG_SCHEMA = "forge.ugi_amine_semantic_program_draw_config.v1"
RESULT_SCHEMA = "forge.ugi_amine_semantic_program_draw.v1"
COARSE_DRAW_SCHEMA = "forge.ugi_measured_joint_morphology_program_draw.v1"
COARSE_DRAW_MEMBER = "occurrence_weighted_program_draw.json"


class UgiAmineSemanticProgramDrawError(ValueError):
    """The semantic program-draw contract or one pinned input changed."""


def _read_coarse_draw(archive_path: Path) -> Mapping[str, Any]:
    try:
        with tarfile.open(archive_path, mode="r") as archive:
            member = archive.getmember(COARSE_DRAW_MEMBER)
            handle = archive.extractfile(member)
            if handle is None:
                raise UgiAmineSemanticProgramDrawError(
                    "comparison detail archive lacks the coarse program draw"
                )
            document = json.load(io.TextIOWrapper(handle, encoding="utf-8"))
    except (KeyError, OSError, tarfile.TarError, json.JSONDecodeError) as error:
        raise UgiAmineSemanticProgramDrawError(
            "comparison detail archive is not a readable pinned program source"
        ) from error
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != COARSE_DRAW_SCHEMA
        or document.get("source") != "measured_training_joint_count_prior"
        or document.get("reaction_id") != "ugi_3cr_agile"
        or document.get("component_identity_conditioning") is not False
        or not isinstance(document.get("samples"), list)
        or document.get("rows") != len(document["samples"])
    ):
        raise UgiAmineSemanticProgramDrawError("coarse Ugi program draw changed")
    return document


def _programs(document: Mapping[str, Any], *, count: int) -> tuple[UgiMorphologyProgram, ...]:
    rows = document["samples"]
    if count < 1 or count > len(rows):
        raise UgiAmineSemanticProgramDrawError("requested program count is outside the draw")
    output: list[UgiMorphologyProgram] = []
    for index, row in enumerate(rows[:count]):
        if not isinstance(row, Mapping) or row.get("sample_index") != index:
            raise UgiAmineSemanticProgramDrawError("coarse program rows are not ordered")
        value = row.get("program")
        if not isinstance(value, Mapping) or set(value) != {
            "attachment_counts",
            "cycle_ranks",
            "junction_budgets",
            "node_counts",
        }:
            raise UgiAmineSemanticProgramDrawError("coarse program fields changed")
        output.append(
            UgiMorphologyProgram(
                node_counts=tuple(int(item) for item in value["node_counts"]),
                junction_budgets=tuple(int(item) for item in value["junction_budgets"]),
                cycle_ranks=tuple(int(item) for item in value["cycle_ranks"]),
                attachment_counts=tuple(int(item) for item in value["attachment_counts"]),
            )
        )
    return tuple(output)


def _mean_targets(targets: tuple[Any, ...]) -> dict[str, float]:
    fields = tuple(targets[0].to_mapping())
    matrix = np.asarray(
        [[target.to_mapping()[field] for field in fields] for target in targets],
        dtype=np.float64,
    )
    return {field: float(matrix[:, index].mean()) for index, field in enumerate(fields)}


def run_ugi_amine_semantic_program_draw(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compile the conditional train-fold prior and draw one target per coarse program."""

    config = read_json_object(
        config_path,
        error=UgiAmineSemanticProgramDrawError,
        label="Ugi amine semantic program-draw config",
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
        raise UgiAmineSemanticProgramDrawError("semantic program-draw config changed")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {
        "comparison_details_archive",
        "production_cache",
        "qualified_reactions",
        "ugi_assignments",
    }:
        raise UgiAmineSemanticProgramDrawError("semantic program-draw inputs changed")
    policy = config["policy"]
    if policy != {
        "coarse_program_sequence": "frozen occurrence-weighted measured-train draw",
        "semantic_weighting": "equal amine-family mass within exact coarse amine program",
        "sampling": "one conditional categorical draw per attempt without rejection or retry",
        "component_or_family_ids_enter_program": False,
        "heldout_access": False,
    }:
        raise UgiAmineSemanticProgramDrawError("semantic program policy changed")
    sample_count = config["sample_count"]
    semantic_seed = config["semantic_seed"]
    if (
        config["reaction_id"] != "ugi_3cr_agile"
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or isinstance(semantic_seed, bool)
        or not isinstance(semantic_seed, int)
        or semantic_seed < 0
    ):
        raise UgiAmineSemanticProgramDrawError("semantic program runtime changed")
    inputs = {
        label: resolve_pin(pin, repo, label=label) for label, pin in sorted(raw_inputs.items())
    }
    coarse_document = _read_coarse_draw(inputs["comparison_details_archive"])
    programs = _programs(coarse_document, count=sample_count)
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        reaction_id=str(config["reaction_id"]),
        expected_sha256=str(raw_inputs["qualified_reactions"]["sha256"]),
        expected_training_assignments_sha256=str(raw_inputs["ugi_assignments"]["sha256"]),
    )
    cache = SynthesisProgramProductionCache(inputs["production_cache"])
    try:
        prior = UgiMeasuredAmineSemanticPrior.from_training_data(
            assignments_path=inputs["ugi_assignments"],
            cache=cache,
            ester_policy=ester_policy,
            reaction_id=str(config["reaction_id"]),
            expected_assignments_sha256=str(raw_inputs["ugi_assignments"]["sha256"]),
        )
        targets = prior.sample_for_programs(programs, seed=semantic_seed)
    finally:
        cache.close()
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "scientific_question": config["scientific_question"],
        "reaction_id": config["reaction_id"],
        "coarse_program_seed": coarse_document["seed"],
        "semantic_seed": semantic_seed,
        "rows": len(programs),
        "config": pin_record(config_path, repo),
        "implementation": pin_record(Path(__file__), repo),
        "prior_implementation": pin_record(
            repo / "forge/model/ugi_amine_semantic_program.py", repo
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
                "amine_semantic_target": target.to_mapping(),
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
        raise UgiAmineSemanticProgramDrawError(f"output already exists: {output_path}")
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
    run_ugi_amine_semantic_program_draw(
        args.config.resolve(), args.repo.resolve(), args.output.resolve()
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiAmineSemanticProgramDrawError",
    "run_ugi_amine_semantic_program_draw",
]
