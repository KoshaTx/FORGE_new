from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from experiments.archive.phase1.synthesis_value_coverage.ugi3_fresh_pool_route_coverage_v4 import (
    TARGET_ROLE,
    TARGET_SMILES,
    Ugi3FreshPoolRouteCoverageV4Error,
    build_fresh_pool_route_coverage_v4,
)
from forge.corpus.r1_prime_audit import sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_fresh_pool_route_coverage_v4.json"
RESULT = REPO / "results/phase1/ugi3_fresh_pool_route_coverage_v4/result.json"
COMPONENTS = REPO / (
    "results/phase1/ugi3_fresh_pool_route_coverage_v4/" "component_synthesis_values.json.gz"
)
PRODUCTS = REPO / (
    "results/phase1/ugi3_fresh_pool_route_coverage_v4/" "product_synthesis_values.json.gz"
)


def _gzip_payload(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def test_exact_c18_refresh_is_reproducible_and_nonselecting() -> None:
    first, first_components, first_products = build_fresh_pool_route_coverage_v4(REPO, CONFIG)
    second, second_components, second_products = build_fresh_pool_route_coverage_v4(REPO, CONFIG)
    assert first == second == json.loads(RESULT.read_text())
    assert first_components == second_components == COMPONENTS.read_bytes()
    assert first_products == second_products == PRODUCTS.read_bytes()
    assert first["config_sha256"] == sha256_file(CONFIG)
    assert first["summary"]["product_outcomes_before"]["complete"] == 172
    assert first["summary"]["product_outcomes_after"]["complete"] == 185
    assert first["summary"]["newly_complete_products"] == 13
    assert first["summary"]["target_occurrences"] == {f"{TARGET_ROLE}\t{TARGET_SMILES}": 107}
    assert first["adjudication"] == {
        "holdout_reveal_authorized": False,
        "prospective_candidate_selection_changed": False,
        "reaction_family_promoted": False,
        "route_mining_priority_update_authorized": True,
        "synthesis_guidance_authorized": False,
    }


def test_exact_c18_refresh_changes_only_one_component_identity() -> None:
    config = json.loads(CONFIG.read_text())
    base = _gzip_payload(REPO / config["inputs"]["base_component_values"]["path"])
    refreshed = _gzip_payload(COMPONENTS)
    assert len(base["records"]) == len(refreshed["records"])
    differences = []
    for before, after in zip(base["records"], refreshed["records"], strict=True):
        if before != after:
            differences.append((before, after))
    assert len(differences) == 1
    before, after = differences[0]
    assert (before["role"], before["canonical_smiles"]) == (
        TARGET_ROLE,
        TARGET_SMILES,
    )
    assert (after["role"], after["canonical_smiles"]) == (
        TARGET_ROLE,
        TARGET_SMILES,
    )
    assert before["value"]["assessment_outcome"] == "missing_knowledge"
    assert after["value"]["assessment_outcome"] == "complete"
    assert after["source_class"] == "exact_c18_three_step_route_current_terminal"


def test_exact_route_result_must_remain_complete_and_exact(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    result_path = REPO / config["inputs"]["exact_route_result"]["path"]
    altered = json.loads(result_path.read_text())
    altered["summary"]["route_complete"] = False
    changed_result = tmp_path / "changed_exact_result.json"
    changed_result.write_text(json.dumps(altered, indent=2, sort_keys=True) + "\n")
    config["inputs"]["exact_route_result"] = {
        "path": str(changed_result),
        "sha256": sha256_file(changed_result),
    }
    changed_config = tmp_path / "changed_config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV4Error,
        match="does not reproduce from its owned config",
    ):
        build_fresh_pool_route_coverage_v4(REPO, changed_config)


def _rebind_tampered_exact_config(tmp_path: Path, mutate) -> Path:
    v4 = json.loads(CONFIG.read_text())
    exact_config_path = REPO / v4["inputs"]["exact_route_config"]["path"]
    exact_config = json.loads(exact_config_path.read_text())
    mutate(exact_config)
    changed_exact_config = tmp_path / "changed_exact_config.json"
    changed_exact_config.write_text(json.dumps(exact_config, indent=2, sort_keys=True) + "\n")

    exact_result_path = REPO / v4["inputs"]["exact_route_result"]["path"]
    exact_result = json.loads(exact_result_path.read_text())
    exact_result["config_sha256"] = sha256_file(changed_exact_config)
    changed_exact_result = tmp_path / "changed_exact_result.json"
    changed_exact_result.write_text(json.dumps(exact_result, indent=2, sort_keys=True) + "\n")

    v4["inputs"]["exact_route_config"] = {
        "path": str(changed_exact_config),
        "sha256": sha256_file(changed_exact_config),
    }
    v4["inputs"]["exact_route_result"] = {
        "path": str(changed_exact_result),
        "sha256": sha256_file(changed_exact_result),
    }
    changed_v4 = tmp_path / "changed_v4.json"
    changed_v4.write_text(json.dumps(v4, indent=2, sort_keys=True) + "\n")
    return changed_v4


def test_v4_rejects_contradictory_homologue_authorization(tmp_path: Path) -> None:
    changed_v4 = _rebind_tampered_exact_config(
        tmp_path,
        lambda value: value["limitations"].update({"homologue_promotion_authorized": True}),
    )
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV4Error,
        match="exact-pair-only limitations",
    ):
        build_fresh_pool_route_coverage_v4(REPO, changed_v4)


def test_v4_rejects_expired_exact_route_assessment(tmp_path: Path) -> None:
    changed_v4 = _rebind_tampered_exact_config(
        tmp_path,
        lambda value: value.update({"assessment_as_of_utc": "2026-09-02T02:24:03Z"}),
    )
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV4Error,
        match="outside the vendor observation window",
    ):
        build_fresh_pool_route_coverage_v4(REPO, changed_v4)


def test_policy_cannot_authorize_guidance_or_holdout_reveal(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["assessment_policy"]["synthesis_guidance_authorized"] = True
    config["assessment_policy"]["holdout_reveal_authorized"] = True
    changed_config = tmp_path / "changed_policy.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        Ugi3FreshPoolRouteCoverageV4Error,
        match="assessment policy changed",
    ):
        build_fresh_pool_route_coverage_v4(REPO, changed_config)
