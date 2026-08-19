from __future__ import annotations

from forge.value.ugi3_fresh_pool_route_coverage import _exact_l1_rows


def test_exact_l1_rows_require_all_three_gates() -> None:
    components = {
        "amine_head": "CN",
        "oxoester_aldehyde_body_tail": "CC=O",
        "isocyanide_tail": "[C-]#[N+]C",
    }
    accepted = {
        "valid": True,
        "component_reconstruction_valid": True,
        "l1_forward_verification": {"exact_product_reconstructed": True},
        "component_smiles_by_role": components,
    }
    rejected = dict(accepted, valid=False)
    assert _exact_l1_rows({"samples": [accepted, rejected]}) == [(0, accepted)]
