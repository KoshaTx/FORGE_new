from __future__ import annotations

import pytest

from forge.value.audit.ugi3_source_bounded_route_value_contrast import (
    Ugi3SourceBoundedContrastError,
    source_bounded_utility,
)


def _component(role: str, smiles: str, evidence: str) -> dict[str, str]:
    return {
        "role": role,
        "canonical_smiles": smiles,
        "graded_evidence_class": evidence,
    }


def test_source_bounded_utility_requires_all_three_roles() -> None:
    qualified = {("isocyanide_tail", "iso")}
    components = [
        _component("amine_head", "amine", "exact_complete_current"),
        _component("oxoester_aldehyde_body_tail", "aldehyde", "exact_complete_current"),
        _component("isocyanide_tail", "iso", "missing_knowledge"),
    ]
    assert source_bounded_utility(components, qualified) == 1
    assert source_bounded_utility(components, set()) == 0


def test_source_bounded_utility_rejects_malformed_terminal() -> None:
    with pytest.raises(Ugi3SourceBoundedContrastError):
        source_bounded_utility([], set())
