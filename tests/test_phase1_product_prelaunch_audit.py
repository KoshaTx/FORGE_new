from __future__ import annotations

from forge.design.flow.defog_feasibility import AtomState
from forge.design.audit.phase1_prelaunch_audit import audit_r1_rows


def _expected(rows: int) -> dict:
    return {
        "r1_rows": rows,
        "maximum_heavy_atoms": 282,
        "maximum_closure_slots": 12,
        "required_components_per_product": 1,
        "minimum_realism_weight": 0.0,
        "required_size_bin_match": True,
        "required_r1_state_subset_of_r0": True,
    }


def test_exact_kekulized_r1_support_passes() -> None:
    rows = [
        {
            "canonical_smiles": "CCN",
            "size_bin": "le40",
            "realism_weight": "1.0",
        },
        {
            "canonical_smiles": "c1ccncc1",
            "size_bin": "le40",
            "realism_weight": "0.2",
        },
    ]
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("N", 0, False),
    )
    result = audit_r1_rows(rows, vocabulary, _expected(len(rows)))

    assert result["all_rows_supported"] is True
    assert result["maximum_heavy_atoms"] == 6
    assert result["maximum_closures"] == 1
    assert result["unsupported_atom_states"] == []
    assert result["failure_counts"] == {}


def test_prelaunch_audit_reports_every_support_failure() -> None:
    rows = [
        {
            "canonical_smiles": "C.C",
            "size_bin": "41_64",
            "realism_weight": "-1",
        },
        {
            "canonical_smiles": "C[Si](C)C",
            "size_bin": "le40",
            "realism_weight": "1",
        },
    ]
    vocabulary = (AtomState("C", 0, False),)
    result = audit_r1_rows(rows, vocabulary, _expected(3))

    assert result["all_rows_supported"] is False
    assert result["failure_counts"] == {
        "component_count_mismatch": 1,
        "invalid_realism_weight": 1,
        "row_count_mismatch": 1,
        "size_bin_mismatch": 1,
        "unsupported_atom_state": 1,
    }
