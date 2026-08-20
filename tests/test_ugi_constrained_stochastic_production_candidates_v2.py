from __future__ import annotations

import json
from pathlib import Path

from forge.design.training.ugi_constrained_stochastic_production_candidates import (
    branch_class,
    exact_terminal_admission,
    program_shard,
)
from forge.design.corpus.ugi_held_component_gate import _reaction_contract

REPO = Path(__file__).resolve().parents[1]


def test_fresh_schedule_materializes_sampler_programs_without_outcomes() -> None:
    schedule = json.loads(
        (REPO / "results/phase1/ugi_production_candidate_schedule_v2/schedule.json").read_text()
    )
    shard = program_shard(schedule, arm="support_enriched", shard_index=0, shard_draws=16)
    assert len(shard["samples"]) == 16
    for index, sample in enumerate(shard["samples"]):
        coordinate = schedule["records"][index]["arms"]["support_enriched"]
        assert sample["program"] == coordinate["program"]
        assert sample["branch_class"] == branch_class(sample["program"])
        assert "smiles" not in sample


def test_terminal_admission_is_separate_from_exact_l1() -> None:
    reaction = _reaction_contract(REPO / "data/vendor/qualified_reactions_v1.json")
    source = json.loads(
        (
            REPO
            / "results/phase1/ugi_decoration_coupling_production_fresh_census_v1"
            / "shard_00/result.json"
        ).read_text()
    )
    terminal = next(
        row
        for row in source["samples"]
        if row.get("valid") is True
        and row.get("terminal_valid") is True
        and row.get("component_reconstruction_valid") is True
    )
    admission = exact_terminal_admission(terminal, reaction)
    assert admission["raw_molecule_valid"] is True
    assert admission["exact_l1"] is True
    assert admission["admitted"] is True
    assert set(admission["handles_by_role"]) == {
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    }

    invalid = dict(terminal)
    invalid["valid"] = False
    rejected = exact_terminal_admission(invalid, reaction)
    assert rejected == {
        "admitted": False,
        "reason": "invalid_or_nonexact_l1",
        "raw_molecule_valid": False,
        "exact_l1": False,
        "handles_by_role": {},
    }
