from __future__ import annotations

from forge.route.terminals.ugi3_precursor_leaf_closure import (
    ALDEHYDE_ROLE,
    HEAD_ROLE,
    ISOCYANIDE_ROLE,
)
from forge.route.sources.ugi3_source_neutral_proposal_adjudication import (
    FATTY_ALDEHYDE,
    ISOCYANIDE,
)
from forge.route.sources.ugi3_source_neutral_proposal_adjudication_v2 import (
    _matching_forward_precursors,
    _program_from_target,
)


def test_program_is_inferred_without_legacy_family_label() -> None:
    aldehyde = _program_from_target(role=ALDEHYDE_ROLE, target="CCCCCCCC(=O)OCCCCCC=O")
    isocyanide = _program_from_target(role=ISOCYANIDE_ROLE, target="[C-]#[N+]CCCCCCCCCCCC")
    assert aldehyde is not None and aldehyde[0] == FATTY_ALDEHYDE
    assert isocyanide is not None and isocyanide[0] == ISOCYANIDE
    assert _program_from_target(role=HEAD_ROLE, target="CN(C)CCCCN") is None


def test_forward_match_requires_expected_precursor_not_family_name() -> None:
    row = {
        "raw_discovery_resolution": {
            "traces": [
                {
                    "target_reconstructed": True,
                    "reaction_id": "ugi3_upstream_primary_alcohol_oxidation_exact_source_v1",
                    "role_ordered_reactants": ["CCCCO"],
                }
            ]
        }
    }
    assert _matching_forward_precursors(
        row, program="primary_alcohol_oxidation", expected_precursor="CCCCO"
    ) == ("CCCCO",)
    assert not _matching_forward_precursors(
        row, program="primary_alcohol_oxidation", expected_precursor="CCCO"
    )
