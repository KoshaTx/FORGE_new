from __future__ import annotations

from forge.design.audit.ugi_fresh_pool_audit import (
    UgiFreshPoolAuditError,
    summarize_fresh_pool_rows,
    validate_sampling_metadata,
)
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES


def _row(product: str, components: dict[str, str]) -> dict[str, object]:
    return {
        "valid": True,
        "smiles": product,
        "component_reconstruction_valid": True,
        "l1_forward_verification": {"exact_product_reconstructed": True},
        "component_smiles_by_role": components,
    }


def test_fresh_pool_summary_separates_exact_catalog_and_open_components() -> None:
    components = {
        "amine_head": "CN",
        "oxoester_aldehyde_body_tail": "CC=O",
        "isocyanide_tail": "[C-]#[N+]C",
    }
    catalog = {
        ("amine_head", "CN"): True,
        ("oxoester_aldehyde_body_tail", "CC=O"): False,
    }
    rows = [
        _row("CC", components),
        {"valid": False, "failure_type": "TerminalSupportFailure"},
    ]
    result = summarize_fresh_pool_rows(
        rows,
        selection_visible_products={"CC"},
        catalog=catalog,
    )
    assert result["attempted_draws"] == 2
    assert result["valid_products"] == 1
    assert result["exact_l1_fraction_of_valid"] == 1.0
    assert result["exact_selection_visible_products"] == 1
    assert result["products_by_structural_provenance"] == {
        "product_uses_only_original_ugi_components": 0,
        "product_contains_admitted_transferred_or_expanded_known_component": 0,
        "product_contains_catalog_absent_component": 1,
    }
    role_counts = result["component_occurrences_by_role_and_structural_provenance"]
    assert set(role_counts) == set(ROLE_NAMES)
    assert role_counts["amine_head"]["original_ugi_component"] == 1
    assert (
        role_counts["oxoester_aldehyde_body_tail"][
            "admitted_transferred_or_expanded_known_component"
        ]
        == 1
    )
    assert role_counts["isocyanide_tail"]["graph_absent_from_frozen_424_component_catalog"] == 1


def test_sampling_result_schema_nests_terminal_decoder_under_sampling() -> None:
    sample = {
        "seed": 20260814,
        "sampling": {
            "terminal_decoder": {
                "mode": "bond_stochastic",
                "seed": 20260815,
                "temperature": 1.0,
            }
        },
    }
    contract = {
        "flow_seed": 20260814,
        "terminal_decoder_mode": "bond_stochastic",
        "terminal_seed": 20260815,
        "terminal_temperature": 1.0,
    }
    validate_sampling_metadata(sample, contract)

    sample["sampling"]["terminal_decoder"]["mode"] = "argmax"
    try:
        validate_sampling_metadata(sample, contract)
    except UgiFreshPoolAuditError as exc:
        assert str(exc) == "terminal decoder changed"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("decoder mismatch was not rejected")
