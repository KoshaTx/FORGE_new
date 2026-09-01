from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.phase1.product_l1.evaluation.ugi_amine_semantic_program_comparison import (
    UgiAmineSemanticProgramComparisonError,
    _load_draw,
    _runtime,
    _semantic_support_audit,
)
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _topology_policy,
)
from forge.core.hashing import sha256_file
from forge.core.io import read_json_object
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/ugi_amine_semantic_program_comparison_seed0_v1.json"
DRAW = REPO / "results/phase1/ugi_amine_semantic_program_draw_seed0_v2/program_draw.json"


def test_semantic_comparison_freezes_the_scientific_contract() -> None:
    config = json.loads(CONFIG.read_text())
    runtime = _runtime(config, profile="smoke", device="cpu")

    assert runtime["program_count"] == 8
    assert config["policy"]["component_identity_conditioning"] is False
    assert config["policy"]["repairs_or_retries"] is False
    assert config["policy"]["training_calls"] == 0


def test_semantic_comparison_rejects_retry_relaxation() -> None:
    config = json.loads(CONFIG.read_text())
    changed = copy.deepcopy(config)
    changed["policy"]["repairs_or_retries"] = True

    with pytest.raises(UgiAmineSemanticProgramComparisonError, match="policy changed"):
        _runtime(changed, profile="smoke", device="cpu")


def test_semantic_draw_loads_paired_programs_and_targets() -> None:
    programs, targets = _load_draw(DRAW, count=8)

    assert len(programs) == len(targets) == 8
    assert all(target.nitrogen_atoms >= 1 for target in targets)


def test_semantic_smoke_draw_has_nonempty_exact_topology_support() -> None:
    config = json.loads(CONFIG.read_text())
    inputs = {label: REPO / value["path"] for label, value in config["inputs"].items()}
    programs, targets = _load_draw(DRAW, count=8)
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
                error=ValueError,
                label="role-local chemistry support",
            )
        ),
    )

    assert audit["all_pairs_supported"] is True
    assert audit["minimum_feasible_topologies_per_pair"] >= 1
    assert audit["local_chemistry_support_conditioned"] is True
