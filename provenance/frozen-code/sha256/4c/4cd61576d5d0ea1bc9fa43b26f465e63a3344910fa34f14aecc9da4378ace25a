from __future__ import annotations

from forge.bio.ugi_distributional_applicability_v2 import (
    CountChemicalReference,
    _identity_excluded_reference,
)


def test_count_fingerprint_distinguishes_long_chain_homologues() -> None:
    reference = CountChemicalReference(("CCCCCCCC",))
    distance = reference.distance("CCCCCCCCCC")

    assert distance.exact_identity_seen is False
    assert distance.fingerprint > 0.0


def test_identity_excluded_reference_does_not_collapse_component_distance() -> None:
    reference = _identity_excluded_reference(
        ("CCCC", "CCCC", "CCCCCC", "CCCCCCCC"),
        "CCCC",
    )
    distance = reference.distance("CCCC")

    assert distance.exact_identity_seen is False
    assert distance.fingerprint > 0.0
