from __future__ import annotations

import numpy as np

from forge.eval.ugi_distribution_overlap import (
    ViewReference,
    _identity_record,
    _identity_reference,
    _joint_percentiles,
    _midrank_percentile,
    _non_agile_broad_rows,
    _weighted_sample_rows,
)


def test_joint_percentile_uses_one_calibrated_multiview_score() -> None:
    baseline = [
        {
            "product_fingerprint": value,
            "product_descriptor": value,
            "amine_fingerprint": value,
            "amine_descriptor": value,
        }
        for value in (0.1, 0.2, 0.3, 0.4)
    ]
    query = [
        {
            "product_fingerprint": 0.15,
            "product_descriptor": 0.15,
            "amine_fingerprint": 0.15,
            "amine_descriptor": 0.15,
        },
        {
            "product_fingerprint": 0.45,
            "product_descriptor": 0.45,
            "amine_fingerprint": 0.45,
            "amine_descriptor": 0.45,
        },
    ]
    values = _joint_percentiles(query, baseline, ("product", "amine"))
    assert values[0] < values[1]
    assert 0.0 < values[0] <= 1.0
    assert 0.0 < values[1] <= 1.0


def test_tied_nonconformity_uses_midrank_instead_of_outer_tail() -> None:
    tied = np.asarray([0.2] * 10)
    assert _midrank_percentile(tied, 0.2) == 0.5
    baseline = [{"product_fingerprint": 0.2, "product_descriptor": 0.2} for _ in range(10)]
    assert _joint_percentiles(baseline, baseline, ("product",)) == [0.5] * 10


def test_view_reference_excludes_exact_constitutional_identity() -> None:
    reference = ViewReference(("CC", "CCC", "CCCC"), fingerprint_cap=10, seed=7)
    distance = reference.distance("CCC")
    assert distance.fingerprint > 0
    assert distance.descriptor > 0


def test_weighted_query_sampling_is_deterministic_and_nonuniform() -> None:
    rows = [{"record_id": str(index)} for index in range(3)]
    weights = np.asarray([0.0, 0.0, 1.0])
    first = _weighted_sample_rows(rows, weights, 12, 17)
    second = _weighted_sample_rows(rows, weights, 12, 17)
    assert first == second
    assert {row["record_id"] for row in first} == {"2"}


def test_non_agile_broad_reference_excludes_direct_agile_sources() -> None:
    base = {
        "r0_structure_id": "r0",
        "canonical_constitutional_smiles": "CC",
        "r0_pretraining_eligible": "true",
    }
    rows = [
        {**base, "all_available_source_ids": "lnpdb_v1"},
        {
            **base,
            "r0_structure_id": "agile",
            "all_available_source_ids": "agile_virtual12k|lnpdb_v1",
        },
    ]
    assert [row["record_id"] for row in _non_agile_broad_rows(rows)] == ["r0"]


def test_identity_is_recomputed_for_each_named_reference() -> None:
    reference_rows = [
        {
            "product_smiles": "CCN",
            "amine_smiles": "CN",
            "aldehyde_smiles": "CC=O",
            "isocyanide_smiles": "[C-]#[N+]C",
        },
        {
            "product_smiles": "CCO",
            "amine_smiles": "CCN",
            "aldehyde_smiles": "CCC=O",
            "isocyanide_smiles": "[C-]#[N+]CC",
        },
    ]
    query = {
        "product_smiles": "CCCO",
        "amine_smiles": "CN",
        "aldehyde_smiles": "CCC=O",
        "isocyanide_smiles": "[C-]#[N+]C",
    }
    product_seen, triple_seen, unseen = _identity_record(query, _identity_reference(reference_rows))
    assert product_seen is False
    assert triple_seen is False
    assert unseen == []
