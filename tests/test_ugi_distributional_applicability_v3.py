from __future__ import annotations

from forge.potency.ugi_distributional_applicability_v3 import (
    _product_reference_excluding_role_identity,
)


def test_product_reference_excludes_every_same_component_product() -> None:
    rows = (
        {"product_smiles": "CCCCN", "A_smiles": "CN"},
        {"product_smiles": "CCCCCCN", "A_smiles": "CN"},
        {"product_smiles": "CCCCO", "A_smiles": "CO"},
    )
    reference = _product_reference_excluding_role_identity(
        rows,
        role_field="A_smiles",
        query_component="CN",
    )

    assert reference.canonical_set == {"CCCCO"}
    assert reference.distance("CCCCN").exact_identity_seen is False
