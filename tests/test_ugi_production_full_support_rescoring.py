from __future__ import annotations

import pytest

from forge.bio.ugi_production_full_support_rescoring import (
    UgiProductionFullSupportRescoringError,
    pattern_id_for_roles,
)


@pytest.mark.parametrize(
    ("roles", "expected"),
    [
        ((), "known_components_novel_combination"),
        (("amine",), "amine_only"),
        (("aldehyde",), "aldehyde_only"),
        (("isocyanide",), "isocyanide_only"),
        (("amine", "aldehyde"), "amine_aldehyde"),
        (("amine", "isocyanide"), "amine_isocyanide"),
        (("aldehyde", "isocyanide"), "aldehyde_isocyanide"),
        (("amine", "aldehyde", "isocyanide"), "amine_aldehyde_isocyanide"),
    ],
)
def test_pattern_id_for_every_ordered_novelty_pattern(roles, expected):
    assert pattern_id_for_roles(roles) == expected


def test_pattern_id_rejects_noncanonical_role_order():
    with pytest.raises(UgiProductionFullSupportRescoringError):
        pattern_id_for_roles(("isocyanide", "aldehyde"))
