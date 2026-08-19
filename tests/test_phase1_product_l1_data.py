from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import pytest

from forge.product.phase1_data import (
    ASSIGNMENT_FIELDS,
    FOLDS,
    ROLES,
    Phase1DataError,
    _build_assignments,
    _build_provenance_rows,
    _component_fold_map,
    _merge_ugi_products,
    _parse_virtual_products,
    _validate_config,
)


def _config() -> dict:
    return {
        "schema_version": "phase1_product_l1_data_config.v3",
        "seed": 17,
        "inputs": {
            name: {"path": f"{name}.csv", "sha256": "0" * 64}
            for name in (
                "r0_constitutional",
                "r0_assignments",
                "r1_reaction_enumerated",
                "ugi_virtual_products",
                "ugi_measured_semantics",
                "qualified_reactions",
            )
        },
        "broad_stream": {
            "primary_split_scheme": "source_study",
            "r0_fraction": 0.9,
            "r1_fraction": 0.1,
            "r1_sampling_weight": "realism_weight",
            "raw_reaction_family_sampling_allowed": False,
        },
        "ugi_l1": {
            "roles": list(ROLES),
            "required_decomposition_status": "one_exact_qualified_ugi_decomposition",
            "required_candidate_count": 1,
            "component_split_fractions": {
                "train": 0.7,
                "calibration": 0.15,
                "heldout": 0.15,
            },
            "duplicate_product_upweighting_allowed": False,
        },
        "joint_training": {
            "broad_to_ugi_replay_ratios": [
                [0.75, 0.25],
                [0.5, 0.5],
                [0.25, 0.75],
            ],
            "biological_guidance_enabled": False,
            "synthesis_value_guidance_enabled": False,
        },
    }


def _virtual_row(index: int, components: dict[str, str], product: str) -> dict[str, str]:
    return {
        "source_row_index": str(index),
        "canonical_product_smiles": product,
        "decomposition_status": "one_exact_qualified_ugi_decomposition",
        "candidate_count": "1",
        "candidate_routes_json": json.dumps([{"components": components}]),
    }


def test_config_prohibits_raw_r1_family_sampling_and_guidance() -> None:
    config = _config()
    _validate_config(config)

    config["broad_stream"]["r1_sampling_weight"] = "reaction_family"
    with pytest.raises(Phase1DataError, match="realism_weight"):
        _validate_config(config)


def test_component_fold_map_is_deterministic_and_nonempty() -> None:
    components = [f"C{index}" for index in range(12)]
    fractions = {"train": 0.7, "calibration": 0.15, "heldout": 0.15}

    first = _component_fold_map(components, fractions, "amine_head", 19)
    second = _component_fold_map(list(reversed(components)), fractions, "amine_head", 19)

    assert first == second
    assert set(first.values()) == set(FOLDS)
    assert Counter(first.values()) == {"train": 8, "calibration": 2, "heldout": 2}


def test_exact_virtual_products_receive_role_specific_component_holdouts() -> None:
    config = _config()
    rows = []
    amines = ["CN", "CCN", "CCCN", "CCCCN"]
    aldehydes = ["CC=O", "CCC=O", "CCCC=O", "CCCCC=O", "CCCCCC=O", "CCCCCCC=O"]
    isocyanides = ["[C-]#[N+]C", "[C-]#[N+]CC", "[C-]#[N+]CCC"]
    for index in range(12):
        components = {
            "amine_head": amines[index % len(amines)],
            "oxoester_aldehyde_body_tail": aldehydes[index % len(aldehydes)],
            "isocyanide_tail": isocyanides[index % len(isocyanides)],
        }
        rows.append(_virtual_row(index, components, "C" * (index + 2)))

    products = _parse_virtual_products(rows, config)
    assignments = _build_assignments(products, {"CC"}, config)

    assert len(assignments) == 12
    assert tuple(assignments[0]) == ASSIGNMENT_FIELDS
    assert sum(row["is_source_adjudicated_measured_product"] == "true" for row in assignments) == 1
    for row in assignments:
        role_folds = {row[f"held_{role}_fold"] for role in ROLES}
        expected = (
            "heldout"
            if "heldout" in role_folds
            else "calibration" if "calibration" in role_folds else "train"
        )
        assert row["primary_product_fold"] == expected
    for role in ROLES:
        component_to_fold = {}
        for row in assignments:
            component = row[f"{role}_smiles"]
            fold = row[f"held_{role}_fold"]
            if component in component_to_fold:
                assert component_to_fold[component] == fold
            component_to_fold[component] = fold
        assert set(component_to_fold.values()) == set(FOLDS)


def test_duplicate_virtual_constitutions_collapse_without_upweighting() -> None:
    config = _config()
    components = {
        "amine_head": "CN",
        "oxoester_aldehyde_body_tail": "CC=O",
        "isocyanide_tail": "[C-]#[N+]C",
    }
    rows = [
        _virtual_row(0, components, "C/C=C/C"),
        _virtual_row(1, components, "C/C=C\\C"),
    ]

    products = _parse_virtual_products(rows, config)
    assert len(products) == 1
    provenance = _build_provenance_rows(products)
    assert len(provenance) == 2
    assert {row["constitutional_group_size"] for row in provenance} == {"2"}


def test_conflicting_constitutional_component_mappings_fail_loudly() -> None:
    config = _config()
    first = {
        "amine_head": "CN",
        "oxoester_aldehyde_body_tail": "CC=O",
        "isocyanide_tail": "[C-]#[N+]C",
    }
    second = {**first, "amine_head": "CCN"}
    rows = [
        _virtual_row(0, first, "C/C=C/C"),
        _virtual_row(1, second, "C/C=C\\C"),
    ]

    with pytest.raises(Phase1DataError, match="component mappings disagree"):
        _parse_virtual_products(rows, config)


def test_measured_products_take_priority_without_duplicate_upweighting() -> None:
    virtual = [
        {
            "product_id": "VUGI-00000",
            "canonical_product_smiles": "P0",
            **{f"{role}_smiles": f"{role}-0" for role in ROLES},
            "_source_records": [
                {
                    "source_kind": "virtual",
                    "source_product_id": "VUGI-00000",
                }
            ],
        },
        {
            "product_id": "VUGI-00001",
            "canonical_product_smiles": "P1",
            **{f"{role}_smiles": f"{role}-1" for role in ROLES},
            "_source_records": [
                {
                    "source_kind": "virtual",
                    "source_product_id": "VUGI-00001",
                }
            ],
        },
    ]
    measured = [
        {
            "product_id": "MUGI-A1B1C1",
            "canonical_product_smiles": "P1",
            **{f"{role}_smiles": f"{role}-1" for role in ROLES},
            "_source_records": [
                {
                    "source_kind": "measured",
                    "source_product_id": "MUGI-A1B1C1",
                }
            ],
        },
        {
            "product_id": "MUGI-A2B2C2",
            "canonical_product_smiles": "P2",
            **{f"{role}_smiles": f"{role}-2" for role in ROLES},
            "_source_records": [
                {
                    "source_kind": "measured",
                    "source_product_id": "MUGI-A2B2C2",
                }
            ],
        },
    ]

    union, overlap = _merge_ugi_products(virtual, measured)

    assert overlap == 1
    assert len(union) == 3
    assert {row["canonical_product_smiles"] for row in union} == {"P0", "P1", "P2"}
    assert next(row for row in union if row["canonical_product_smiles"] == "P1")[
        "product_id"
    ].startswith("MUGI-")


@pytest.mark.needs_r1
def test_repository_phase1_contract_matches_frozen_inputs(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "data/vendor/r1_reaction_enumerated_support_v1.csv").exists():
        pytest.skip("optional R1 asset is absent")
    config = json.loads((repo / "configs/model/phase1_product_l1_data.json").read_text())
    config["outputs"] = {
        "ugi_assignments": str(tmp_path / "ugi_l1_assignments.csv.gz"),
        "ugi_provenance": str(tmp_path / "ugi_l1_constitutional_provenance.csv.gz"),
        "manifest": str(tmp_path / "manifest.json"),
        "result": str(tmp_path / "result.json"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    from forge.product.phase1_data import freeze_phase1_data_contract

    result = freeze_phase1_data_contract(config_path, repo)
    assert result["summary"]["r0_rows"] == 15_229
    assert result["summary"]["r1_rows"] == 464_265
    assert result["summary"]["ugi_l1_products"] == 12_386
    assert result["summary"]["source_adjudicated_measured_ugi_products"] == 1_100
    with gzip.open(tmp_path / "ugi_l1_assignments.csv.gz", "rt") as handle:
        assert sum(1 for _ in handle) == 12_387
    with gzip.open(tmp_path / "ugi_l1_constitutional_provenance.csv.gz", "rt") as handle:
        assert sum(1 for _ in handle) == 13_377
