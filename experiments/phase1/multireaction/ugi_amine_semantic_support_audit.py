"""Exhaustively audit exact topology support for a frozen amine-semantic draw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_amine_semantic_program_comparison import (
    UgiAmineSemanticProgramComparisonError,
    _load_draw,
    _semantic_support_audit,
)
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _topology_policy,
)
from forge.core.hashing import pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy

RESULT_SCHEMA = "forge.ugi_amine_semantic_support_audit.v2"


def audit_ugi_amine_semantic_support(
    config_path: Path, repo: Path, output_path: Path
) -> dict[str, Any]:
    """Audit every production request without loading a model or generating a molecule."""

    config = read_json_object(
        config_path,
        error=UgiAmineSemanticProgramComparisonError,
        label="Ugi amine semantic comparison config",
    )
    raw_inputs = config.get("inputs")
    profiles = config.get("profiles")
    if not isinstance(raw_inputs, dict) or not isinstance(profiles, dict):
        raise UgiAmineSemanticProgramComparisonError("semantic audit config changed")
    inputs = {
        label: resolve_pin(pin, repo, label=label) for label, pin in sorted(raw_inputs.items())
    }
    full = profiles.get("full")
    if not isinstance(full, dict) or int(full.get("program_count", 0)) < 1:
        raise UgiAmineSemanticProgramComparisonError("semantic full profile changed")
    programs, targets = _load_draw(
        inputs["amine_semantic_program_draw"], count=int(full["program_count"])
    )
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(inputs["qualified_reactions"]),
        expected_training_assignments_sha256=sha256_file(inputs["ugi_assignments"]),
    )
    audit = _semantic_support_audit(
        programs,
        targets,
        topology_policy=_topology_policy(inputs),
        ester_policy=ester_policy,
        local_chemistry_support=LocalChemistrySupport.from_mapping(
            read_json_object(
                inputs["role_morphology_policy"],
                error=UgiAmineSemanticProgramComparisonError,
                label="role-local chemistry support",
            )
        ),
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if audit["all_pairs_supported"] else "fail",
        "support": audit,
        "inputs": {
            "config": pin_record(config_path, repo),
            **{
                label: pin_record(inputs[label], repo)
                for label in (
                    "amine_semantic_program_draw",
                    "qualified_reactions",
                    "role_morphology_policy",
                    "topology_closure_config",
                    "topology_morphology_config",
                    "ugi_assignments",
                )
            },
        },
        "calls": {
            "model": 0,
            "generation": 0,
            "training": 0,
            "repair": 0,
            "retry": 0,
            "route": 0,
            "oracle": 0,
            "candidate_selection": 0,
        },
        "component_identity_conditioning": False,
        "heldout_access": False,
    }
    write_json(output_path, result)
    if result["status"] != "pass":
        raise UgiAmineSemanticProgramComparisonError(
            f"{len(audit['unsupported_pairs'])} semantic program pairs lack exact support"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    result = audit_ugi_amine_semantic_support(args.config, repo, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
