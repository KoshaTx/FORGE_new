from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

from forge.core.io import read_csv_rows, read_json_object
from forge.corpus.component_splits import product_fold
from forge.corpus.multireaction_mixed_expansion import (
    _components,
    _encode_origins,
    _load_config,
    _model_support_reason,
    _program_specs,
    _requests,
    _unrank_combination,
)
from forge.model.vocabulary import load_atom_vocabulary

REPO = Path(__file__).resolve().parents[1]
CAPACITY_CONFIG = REPO / "configs/multireaction/bl_lx_mixed_repeat_expansion_v1.json"
CONFIG = REPO / "configs/multireaction/bl_lx_mixed_repeat_expansion_v2.json"
STRICT_CONFIG = REPO / "configs/multireaction/bl_lx_mixed_repeat_expansion_v3.json"
CHECKED = REPO / "results/phase1/bl_lx_mixed_repeat_expansion_v1"
CAPACITY_RESULT = REPO / "results/phase1/bl_lx_mixed_repeat_capacity_v1/result.json"


def test_multiset_unranking_is_complete_and_lexicographic() -> None:
    assert [_unrank_combination(3, 2, rank) for rank in range(6)] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    ]


def test_semantic_origins_encode_adapter_states_not_chemistry_role_names() -> None:
    origin_codes, core_codes = _encode_origins(
        ("accumulator", "repeat", "repeat"),
        ("", "map_1", "map_2"),
    )
    assert origin_codes == bytes((0, 1, 1))
    assert core_codes == bytes((0, 1, 2))


def test_mixed_requests_are_deterministic_distinct_and_fold_conditioned() -> None:
    config, paths = _load_config(CONFIG, REPO)
    specs = _program_specs(paths["program_config"])
    _, components = _components(paths["component_registry"], specs)
    arguments = {
        "seed": int(config["seed"]),
        "program_id": "bl_2023_repeated_aza_michael",
        "fold": "heldout",
        "count": 50,
        "components": components.values(),
        "maximum_counter": 100_000,
        "start_index": 0,
    }
    first = _requests(**arguments)
    second = _requests(**arguments)
    assert first == second
    assert len({value.attempt_id for value in first}) == 50
    for request in first:
        repeat_ids = tuple(value.component_id for value in request.repeats)
        assert len(set(repeat_ids)) >= 2
        assert repeat_ids == tuple(sorted(repeat_ids))
        assert len(request.repeats) == request.head.step_count
        assert (
            product_fold(
                (
                    request.head.family_fold,
                    *(component.family_fold for component in request.repeats),
                )
            )
            == "heldout"
        )


def test_failed_capacity_contract_preserves_the_frozen_ugi_exposure() -> None:
    config, _ = _load_config(CAPACITY_CONFIG, REPO)
    assert config["target_products_by_program_fold"] == {
        "bl_2023_repeated_aza_michael": {
            "train": 66_464,
            "calibration": 14_242,
            "heldout": 14_242,
        },
        "lx_2024_repeated_reductive_amination": {
            "train": 66_464,
            "calibration": 14_242,
            "heldout": 14_242,
        },
    }


def test_mixed_targets_are_exposure_balanced_without_relaxing_chemistry() -> None:
    config, _ = _load_config(CONFIG, REPO)
    assert config["target_products_by_program_fold"] == {
        "bl_2023_repeated_aza_michael": {
            "train": 30_000,
            "calibration": 8_000,
            "heldout": 8_000,
        },
        "lx_2024_repeated_reductive_amination": {
            "train": 30_000,
            "calibration": 8_000,
            "heldout": 8_000,
        },
    }
    assert config["policy"]["evidence_stratum_mass"] == {
        "source_executed": 0.5,
        "computed_homogeneous_transform_consistency": 0.25,
        "computed_mixed_transform_consistency": 0.25,
    }


def test_strict_expansion_uses_the_pinned_atom_and_closure_support() -> None:
    config, paths = _load_config(STRICT_CONFIG, REPO)
    policy = dict(config["policy"])
    policy["_allowed_atom_states"] = [
        list(state.key()) for state in load_atom_vocabulary(paths["atom_vocabulary"])
    ]
    supported = Chem.MolFromSmiles("CCN")
    unsupported_state = Chem.MolFromSmiles("C[NH2]")
    excessive_closures = Chem.MolFromSmiles("C12C3C4C1C5C2C3C45")
    assert supported is not None and unsupported_state is not None and excessive_closures is not None
    assert _model_support_reason(supported, policy) == ""
    assert (
        _model_support_reason(unsupported_state, policy)
        == "forward_product_outside_pinned_atom_vocabulary"
    )
    assert (
        _model_support_reason(excessive_closures, policy)
        == "forward_product_exceeds_declared_closure_support"
    )


def test_checked_mixed_expansion_is_balanced_and_evidence_separated() -> None:
    result = read_json_object(CHECKED / "result.json")
    assert result["summary"]["total_products"] == 92_000
    assert result["summary"]["v1_products_preserved"] == 22_951
    assert result["summary"]["mixed_products_admitted"] == 69_049
    assert result["summary"]["products_by_program_fold"] == {
        "bl_2023_repeated_aza_michael|calibration": 8_000,
        "bl_2023_repeated_aza_michael|heldout": 8_000,
        "bl_2023_repeated_aza_michael|train": 30_000,
        "lx_2024_repeated_reductive_amination|calibration": 8_000,
        "lx_2024_repeated_reductive_amination|heldout": 8_000,
        "lx_2024_repeated_reductive_amination|train": 30_000,
    }
    assert result["claims_boundary"]["mixed_transform_consistency_is_observed_synthesis"] is False
    assert result["claims_boundary"]["mixed_transform_consistency_is_route_closure"] is False
    assert result["claims_boundary"]["reductive_amination_substructure_hit_rate_reported"] is False

    atlas = read_csv_rows(CHECKED / "reaction_program_atlas.csv.gz")
    mixed = [
        row for row in atlas if row["evidence_basis"] == "computed_mixed_transform_consistency"
    ]
    assert len(mixed) == 69_049
    assert all(len(set(json.loads(row["repeat_component_smiles_json"]))) >= 2 for row in mixed)
    assert all(row["semantic_origin_status"] == "exact" for row in mixed)

    stratum_mass: dict[tuple[str, str, str], float] = defaultdict(float)
    for row in read_csv_rows(CHECKED / "component_family_splits.csv.gz"):
        stratum_mass[(row["product_fold"], row["program_id"], row["evidence_stratum"])] += float(
            row["source_balanced_weight"]
        )
    expected_mass = {
        "source_executed": 0.5,
        "computed_homogeneous_transform_consistency": 0.25,
        "computed_mixed_transform_consistency": 0.25,
    }
    assert len(stratum_mass) == 18
    assert all(
        abs(value - expected_mass[stratum]) < 1e-8
        for (_, _, stratum), value in stratum_mass.items()
    )


def test_ugi_matched_capacity_failure_is_frozen_without_gate_relaxation() -> None:
    result = read_json_object(CAPACITY_RESULT)
    assert result["required_mixed_products"] == 61_707
    assert result["admitted_unique_factorization_products"] == 35_217
    assert result["gate"] == "fail"
    assert result["claims_boundary"]["chemical_gate_relaxed"] is False
