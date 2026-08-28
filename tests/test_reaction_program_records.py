from __future__ import annotations

import pytest

from forge.corpus.reaction_program_records import (
    admits_reaction_program_structure,
    repeat_component_smiles,
)


def test_structure_supervision_separates_transform_consistency_from_source_execution() -> None:
    assert admits_reaction_program_structure(
        {"disposition": "admit_exact", "semantic_origin_status": "exact"}
    )
    assert admits_reaction_program_structure(
        {
            "disposition": "admit_transform_consistency",
            "semantic_origin_status": "exact",
        }
    )
    assert not admits_reaction_program_structure(
        {"disposition": "abstain", "semantic_origin_status": "exact"}
    )
    assert not admits_reaction_program_structure(
        {
            "disposition": "admit_transform_consistency",
            "semantic_origin_status": "ambiguous",
        }
    )


def test_repeat_component_smiles_reads_homogeneous_v1_rows() -> None:
    assert repeat_component_smiles({"step_count": "3", "repeat_component_smiles": "CCN"}) == (
        "CCN",
        "CCN",
        "CCN",
    )


def test_repeat_component_smiles_reads_ordered_v2_rows() -> None:
    assert repeat_component_smiles(
        {
            "step_count": "3",
            "repeat_component_smiles": "",
            "repeat_component_smiles_json": '["CCN","CCCN","CCCCN"]',
        }
    ) == ("CCN", "CCCN", "CCCCN")


@pytest.mark.parametrize(
    "row",
    [
        {"step_count": "2", "repeat_component_smiles_json": '["CCN"]'},
        {"step_count": "2", "repeat_component_smiles_json": "not-json"},
        {"step_count": "0", "repeat_component_smiles": "CCN"},
    ],
)
def test_repeat_component_smiles_rejects_incomplete_rows(row: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        repeat_component_smiles(row)
