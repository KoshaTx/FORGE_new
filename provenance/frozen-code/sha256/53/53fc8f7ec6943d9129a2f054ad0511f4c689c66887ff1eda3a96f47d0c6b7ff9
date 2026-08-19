from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.value.ugi3_fresh_pool_route_coverage_v5 import (
    TARGET_ROLE,
    TARGET_SMILES,
    TARGET_SOURCE_CLASS,
    Ugi3FreshPoolRouteCoverageV5Error,
    build_fresh_pool_route_coverage_v5,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_fresh_pool_route_coverage_v5.json"
RESULT = REPO / "results/phase1/ugi3_fresh_pool_route_coverage_v5/result.json"
COMPONENTS = REPO / (
    "results/phase1/ugi3_fresh_pool_route_coverage_v5/" "component_synthesis_values.json.gz"
)
PRODUCTS = REPO / (
    "results/phase1/ugi3_fresh_pool_route_coverage_v5/" "product_synthesis_values.json.gz"
)
BASE_COMPONENTS = REPO / (
    "results/phase1/ugi3_fresh_pool_route_coverage_v4/" "component_synthesis_values.json.gz"
)
BASE_PRODUCTS = REPO / (
    "results/phase1/ugi3_fresh_pool_route_coverage_v4/" "product_synthesis_values.json.gz"
)


def _gzip_payload(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _target_in_product(record: dict) -> bool:
    return any(
        component["role"] == TARGET_ROLE
        and component["value"]["target"]["canonical_smiles"] == TARGET_SMILES
        for component in record["value"]["components"]
    )


def _complete_component_count(record: dict) -> int:
    return sum(
        component["value"]["assessment_outcome"] == "complete"
        for component in record["value"]["components"]
    )


def _tampered_config(tmp_path: Path, mutate) -> Path:
    config = json.loads(CONFIG.read_text())
    mutate(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def test_base_v4_independently_has_expected_c16_occurrence_branches() -> None:
    rows = _gzip_payload(BASE_PRODUCTS)["records"]
    target_rows = [record for record in rows if _target_in_product(record)]
    prior_bins = Counter(_complete_component_count(record) for record in target_rows)
    all_bins = Counter(_complete_component_count(record) for record in rows)
    assert len(target_rows) == 99
    assert prior_bins == {0: 41, 1: 40, 2: 18}
    assert all_bins == {0: 1252, 1: 1663, 2: 875, 3: 185}
    assert sum(record["value"]["route_complete"] for record in rows) == 185


def test_exact_c16_refresh_is_reproducible_nonselecting_and_config_owned() -> None:
    first, first_components, first_products = build_fresh_pool_route_coverage_v5(REPO, CONFIG)
    second, second_components, second_products = build_fresh_pool_route_coverage_v5(REPO, CONFIG)
    assert first == second == json.loads(RESULT.read_text())
    assert first_components == second_components == COMPONENTS.read_bytes()
    assert first_products == second_products == PRODUCTS.read_bytes()
    assert first["config_sha256"] == sha256_file(CONFIG)
    assert first["summary"]["target_occurrences"] == {f"{TARGET_ROLE}\t{TARGET_SMILES}": 99}
    assert first["summary"]["target_occurrences_by_prior_complete_component_count"] == {
        "0": 41,
        "1": 40,
        "2": 18,
    }
    assert first["summary"]["product_outcomes_before"]["complete"] == 185
    assert first["summary"]["product_outcomes_after"]["complete"] == 203
    assert first["summary"]["newly_complete_products"] == 18
    assert first["summary"]["products_by_route_complete_component_count"] == {
        "0": 1211,
        "1": 1664,
        "2": 897,
        "3": 203,
    }
    assert first["adjudication"] == {
        "holdout_reveal_authorized": False,
        "homologue_scope_promoted": False,
        "prospective_candidate_selection_changed": False,
        "reaction_family_promoted": False,
        "route_mining_priority_update_authorized": True,
        "synthesis_guidance_authorized": False,
    }


def test_exact_c16_refresh_changes_one_component_record_only() -> None:
    before = _gzip_payload(BASE_COMPONENTS)["records"]
    after = _gzip_payload(COMPONENTS)["records"]
    assert len(before) == len(after) == 1919
    differences = [
        (left, right) for left, right in zip(before, after, strict=True) if left != right
    ]
    assert len(differences) == 1
    left, right = differences[0]
    assert (left["role"], left["canonical_smiles"]) == (
        TARGET_ROLE,
        TARGET_SMILES,
    )
    assert (right["role"], right["canonical_smiles"]) == (
        TARGET_ROLE,
        TARGET_SMILES,
    )
    assert {key: value for key, value in left.items() if key not in {"source_class", "value"}} == {
        key: value for key, value in right.items() if key not in {"source_class", "value"}
    }
    assert left["value"]["assessment_outcome"] == "missing_knowledge"
    assert right["value"]["assessment_outcome"] == "complete"
    assert right["value"]["route_step_count"] == 4
    assert right["value"]["leaf_count"] == 2
    assert right["value"]["current_terminal_leaf_count"] == 2
    assert right["source_class"] == TARGET_SOURCE_CLASS


def test_product_metadata_and_non_target_values_remain_unchanged() -> None:
    before = _gzip_payload(BASE_PRODUCTS)["records"]
    after = _gzip_payload(PRODUCTS)["records"]
    assert len(before) == len(after) == 3975
    value_changes = 0
    metadata_changes = 0
    transitions: Counter[str] = Counter()
    final_bins: Counter[int] = Counter()
    for left, right in zip(before, after, strict=True):
        left_metadata = {key: value for key, value in left.items() if key != "value"}
        right_metadata = {key: value for key, value in right.items() if key != "value"}
        metadata_changes += int(left_metadata != right_metadata)
        changed = left["value"] != right["value"]
        value_changes += int(changed)
        assert changed == _target_in_product(left)
        left_by_role = {item["role"]: item["value"] for item in left["value"]["components"]}
        right_by_role = {item["role"]: item["value"] for item in right["value"]["components"]}
        for role in left_by_role:
            if role != TARGET_ROLE or left_by_role[role]["target"]["canonical_smiles"] != (
                TARGET_SMILES
            ):
                assert left_by_role[role] == right_by_role[role]
        before_count = _complete_component_count(left)
        after_count = _complete_component_count(right)
        transitions[f"{before_count}_to_{after_count}"] += 1
        final_bins[after_count] += 1
        assert right["value"]["scalar_value"] is None
        assert right["value"]["success_probability"] is None
        assert right["value"]["product_smiles"] == left["value"]["product_smiles"]
        assert right["value"]["l1_forward_consistent"] == (left["value"]["l1_forward_consistent"])
    assert value_changes == 99
    assert metadata_changes == 0
    assert transitions == {
        "0_to_0": 1211,
        "0_to_1": 41,
        "1_to_1": 1623,
        "1_to_2": 40,
        "2_to_2": 857,
        "2_to_3": 18,
        "3_to_3": 185,
    }
    assert final_bins == {0: 1211, 1: 1664, 2: 897, 3: 203}


def test_exact_route_result_must_remain_four_step_two_leaf_complete(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text())
    result_path = REPO / config["inputs"]["exact_route_result"]["path"]
    altered = json.loads(result_path.read_text())
    altered["summary"]["exact_steps"] = 3
    altered["summary"]["terminal_leaf_count"] = 1
    changed_result = tmp_path / "changed_exact_result.json"
    changed_result.write_text(json.dumps(altered, indent=2, sort_keys=True) + "\n")
    config["inputs"]["exact_route_result"] = {
        "path": str(changed_result),
        "sha256": sha256_file(changed_result),
    }
    changed_config = tmp_path / "changed_config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV5Error,
        match="does not authorize exact-identity route closure",
    ):
        build_fresh_pool_route_coverage_v5(REPO, changed_config)


def test_exact_result_must_own_assessment(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    assessment_path = REPO / config["inputs"]["exact_route_assessment"]["path"]
    changed_assessment = tmp_path / "changed_assessment.json.gz"
    changed_assessment.write_bytes(assessment_path.read_bytes() + b"tamper")
    config["inputs"]["exact_route_assessment"] = {
        "path": str(changed_assessment),
        "sha256": sha256_file(changed_assessment),
    }
    changed_config = tmp_path / "changed_config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV5Error,
        match="result/config/assessment ownership failed",
    ):
        build_fresh_pool_route_coverage_v5(REPO, changed_config)


def test_exact_result_must_own_exact_route_config(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    exact_config_path = REPO / config["inputs"]["exact_route_config"]["path"]
    altered = json.loads(exact_config_path.read_text())
    altered["scope"] = "tampered"
    changed_exact_config = tmp_path / "changed_exact_config.json"
    changed_exact_config.write_text(json.dumps(altered, indent=2, sort_keys=True) + "\n")
    config["inputs"]["exact_route_config"] = {
        "path": str(changed_exact_config),
        "sha256": sha256_file(changed_exact_config),
    }
    changed_config = tmp_path / "changed_config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV5Error,
        match="result/config/assessment ownership failed",
    ):
        build_fresh_pool_route_coverage_v5(REPO, changed_config)


def test_exact_terminal_evidence_must_be_current_at_frozen_assessment(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text())
    exact_config_path = REPO / config["inputs"]["exact_route_config"]["path"]
    exact_result_path = REPO / config["inputs"]["exact_route_result"]["path"]
    altered_config = json.loads(exact_config_path.read_text())
    expired_timestamp = "2026-09-02T02:43:32Z"
    altered_config["assessment_as_of_utc"] = expired_timestamp
    altered_config["expected_summary"]["assessment_as_of_utc"] = expired_timestamp
    changed_exact_config = tmp_path / "expired_exact_config.json"
    changed_exact_config.write_text(json.dumps(altered_config, indent=2, sort_keys=True) + "\n")

    altered_result = json.loads(exact_result_path.read_text())
    altered_result["config_sha256"] = sha256_file(changed_exact_config)
    altered_result["summary"]["assessment_as_of_utc"] = expired_timestamp
    changed_exact_result = tmp_path / "expired_exact_result.json"
    changed_exact_result.write_text(json.dumps(altered_result, indent=2, sort_keys=True) + "\n")

    config["inputs"]["exact_route_config"] = {
        "path": str(changed_exact_config),
        "sha256": sha256_file(changed_exact_config),
    }
    config["inputs"]["exact_route_result"] = {
        "path": str(changed_exact_result),
        "sha256": sha256_file(changed_exact_result),
    }
    changed_config = tmp_path / "changed_config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV5Error,
        match="terminal evidence is not current",
    ):
        build_fresh_pool_route_coverage_v5(REPO, changed_config)


def test_v5_policy_cannot_promote_scope_guidance_scalar_or_holdout(
    tmp_path: Path,
) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["assessment_policy"].update(
            {
                "homologue_scope_promoted": True,
                "scalarization_authorized": True,
                "synthesis_guidance_authorized": True,
                "holdout_reveal_authorized": True,
            }
        ),
    )
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV5Error,
        match="assessment policy changed",
    ):
        build_fresh_pool_route_coverage_v5(REPO, config)


def test_immutable_v4_config_ownership_is_enforced(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    base_config_path = REPO / config["inputs"]["base_config"]["path"]
    altered = json.loads(base_config_path.read_text())
    altered["assessment_policy"]["synthesis_guidance_authorized"] = True
    changed_base_config = tmp_path / "changed_v4_config.json"
    changed_base_config.write_text(json.dumps(altered, indent=2, sort_keys=True) + "\n")
    config["inputs"]["base_config"] = {
        "path": str(changed_base_config),
        "sha256": sha256_file(changed_base_config),
    }
    changed_config = tmp_path / "changed_config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV5Error,
        match="base v4 result/config ownership failed",
    ):
        build_fresh_pool_route_coverage_v5(REPO, changed_config)
