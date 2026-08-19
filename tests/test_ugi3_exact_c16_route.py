from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.planner import (
    AssessmentOutcome,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.ugi3_exact_c16_route import (
    Ugi3ExactC16RouteError,
    build_exact_c16_route_audit,
    load_exact_c16_route_overlay,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_exact_c16_route_v1.json"
RESULT = REPO / "results/phase1/ugi3_exact_c16_route_v1/result.json"
STEP_LEDGER = REPO / ("results/phase1/ugi3_exact_c16_route_v1/step_verification_ledger.json.gz")
ASSESSMENT = REPO / "results/phase1/ugi3_exact_c16_route_v1/assessment.json.gz"
INPUT_PATHS = {
    "audit_source": REPO / "src/forge/route/ugi3_exact_c16_route.py",
    "audit_runner": REPO / "scripts/phase1_qualify_ugi3_exact_c16_route.py",
    "audit_tests": REPO / "tests/test_ugi3_exact_c16_route.py",
    "qualified_forward_source": REPO / "src/forge/route/qualified_forward.py",
    "planner_source": REPO / "src/forge/route/planner.py",
    "synthesis_value_source": REPO / "src/forge/value/synthesis.py",
    "reaction_registry": REPO / "configs/route/phase1_ugi3_exact_c16_qualified_reactions_v1.json",
    "chain_variant": REPO
    / "configs/route/variants/ugi3_exact_c16_c3_c13_chain_construction_v1.json",
    "deprotection_variant": REPO / "configs/route/variants/ugi3_exact_c16_thp_deprotection_v1.json",
    "zipper_variant": REPO / "configs/route/variants/ugi3_exact_c16_alkyne_zipper_v1.json",
    "oxidation_variant": REPO
    / "configs/route/variants/ugi3_exact_c16_primary_alcohol_oxidation_v1.json",
    "source_manifest": REPO
    / "data/source_cache/phase1_ugi3_exact_c16_alkynyl_aldehyde/source_manifest_v1.json",
    "route_primary": REPO
    / "data/source_cache/phase1_ugi3_exact_c16_alkynyl_aldehyde/au2020334152a1_wo2021035214a1.pdf",
    "oxidation_primary": REPO
    / "data/source_cache/phase1_ugi3_exact_c16_alkynyl_aldehyde/zheng_ja311416v_si_001.pdf",
    "conflicted_source": REPO
    / "data/source_cache/phase1_ugi3_exact_c16_alkynyl_aldehyde/wo2024073486a2_rejected_source_conflict.pdf",
    "terminal_observations": REPO
    / "data/source_cache/phase1_ugi3_exact_c16_alkynyl_aldehyde/terminal_observations_2026-08-03.json",
}


class RecordingMissingSource:
    def __init__(self) -> None:
        self.calls: list[RouteTarget] = []

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls.append(target)
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(),
            detail="no exact evidence in delegated source",
        )


def _budget() -> PlannerBudgetLedger:
    return PlannerBudgetLedger(
        PlannerBudgetLimits(
            maximum_depth=4,
            maximum_logical_planner_calls=1,
            maximum_expansions=4,
            maximum_product_candidates=4,
            maximum_verifier_calls=4,
            maximum_elapsed_milliseconds=0,
        )
    )


def _tampered_config(tmp_path: Path, mutate) -> Path:
    config = json.loads(CONFIG.read_text())
    mutate(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def _tampered_observation(tmp_path: Path, mutate) -> tuple[Path, dict[str, Path]]:
    observation = json.loads(INPUT_PATHS["terminal_observations"].read_text())
    mutate(observation)
    observation_path = tmp_path / "terminal_observations.json"
    observation_path.write_text(json.dumps(observation, indent=2, sort_keys=True) + "\n")
    config = json.loads(CONFIG.read_text())
    config["inputs"]["terminal_observations"]["sha256"] = sha256_file(observation_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    inputs = dict(INPUT_PATHS)
    inputs["terminal_observations"] = observation_path
    return config_path, inputs


def test_exact_c16_route_is_four_step_two_leaf_l2_l3_closed() -> None:
    first, first_steps, first_assessment = build_exact_c16_route_audit(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
    )
    second, second_steps, second_assessment = build_exact_c16_route_audit(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
    )
    assert first == second
    assert first_steps == second_steps
    assert first_assessment == second_assessment
    assert first["summary"] == {
        "assessment_as_of_utc": "2026-08-03T03:04:10Z",
        "assessment_outcome": "complete",
        "development_product_coverage_changed": False,
        "evidence_support": "exact_identity",
        "exact_steps": 4,
        "forward_consistency": "exact_unique",
        "maximum_route_depth": 4,
        "randomized_smiles_total_successes": 400,
        "randomized_smiles_trials_per_step": 100,
        "reaction_family_admitted": False,
        "rejected_source_conflict_count": 1,
        "rejected_source_conflict_used": False,
        "route_complete": True,
        "route_step_count": 4,
        "target_carbon_count": 16,
        "target_role": "oxoester_aldehyde_body_tail",
        "target_smiles": "C#CCCCCCCCCCCCCCC=O",
        "terminal_availability": "current_closed",
        "terminal_leaf_count": 2,
        "unassessed_terminal_leaf_count": 0,
        "uniquely_forward_verified_steps": 4,
    }
    assert first["adjudication"] == {
        "conflicted_source_admitted": False,
        "coverage_update_authorized": False,
        "current_l3_procurement_closed": True,
        "family_template_admitted": False,
        "holdout_reveal_authorized": False,
        "homologue_template_admitted": False,
        "independent_exact_l2_route_admitted": True,
        "route_complete": True,
        "synthesis_guidance_authorized": False,
    }
    assert first["rejected_evidence"][0]["status"] == "rejected_source_conflict"
    assert first["rejected_evidence"][0]["used_in_route"] is False


def test_frozen_overlay_preserves_two_terminal_leaves_and_delegates_neighbors() -> None:
    base = RecordingMissingSource()
    overlay = load_exact_c16_route_overlay(
        base_source=base,
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
        stored_result_path=RESULT,
        stored_step_ledger_path=STEP_LEDGER,
        stored_assessment_path=ASSESSMENT,
    )
    budget = _budget()
    assessment = RecursiveRouteAssessor(overlay).assess(overlay.target, budget)
    assert assessment.outcome is AssessmentOutcome.COMPLETE
    assert budget.expansions == 4
    assert budget.verifier_calls == 4
    assert base.calls == []

    neighbor = RouteTarget(
        role="oxoester_aldehyde_body_tail",
        canonical_smiles="C#CCCCCCCCCCCCCC=O",
    )
    delegated = RecursiveRouteAssessor(overlay).assess(neighbor, _budget())
    assert delegated.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert base.calls == [neighbor]


def test_rejected_source_conflict_cannot_support_a_route_step(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["steps"][0].update(
            {
                "primary_source_input": "conflicted_source",
                "conditions_source_sha256": sha256_file(INPUT_PATHS["conflicted_source"]),
            }
        ),
    )
    with pytest.raises(Ugi3ExactC16RouteError, match="rejected C16 source conflict"):
        build_exact_c16_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_rejected_source_conflict_must_remain_carbon_incoherent(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["rejected_evidence"][0]["carbon_audit"].update(
            {"reactant_carbon_sum": 16, "is_carbon_coherent": True}
        ),
    )
    with pytest.raises(Ugi3ExactC16RouteError, match="exact carbon audit"):
        build_exact_c16_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_route_chain_carbon_identity_cannot_change(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["molecules"]["1_bromotridecane"].update(
            {"route_chain_carbon_count": 12}
        ),
    )
    with pytest.raises(Ugi3ExactC16RouteError, match="route-chain carbon continuity"):
        build_exact_c16_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_exact_pair_registry_cannot_be_generalized_or_changed(tmp_path: Path) -> None:
    registry = json.loads(INPUT_PATHS["reaction_registry"].read_text())
    registry["reactions"][0]["exact_reactant_smiles"][1] = "CCCCCCCCCCCCBr"
    changed_registry = tmp_path / "registry.json"
    changed_registry.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    config = json.loads(CONFIG.read_text())
    config["inputs"]["reaction_registry"]["sha256"] = sha256_file(changed_registry)
    changed_config = tmp_path / "config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    inputs = dict(INPUT_PATHS)
    inputs["reaction_registry"] = changed_registry
    with pytest.raises(Ugi3ExactC16RouteError, match="not exact-pair qualified"):
        build_exact_c16_route_audit(config_path=changed_config, input_paths=inputs)


def test_terminal_cannot_close_without_explicit_current_availability(tmp_path: Path) -> None:
    observation = json.loads(INPUT_PATHS["terminal_observations"].read_text())
    observation["items"][0]["availability_observation"].update(
        {
            "explicit_current_stock_or_shipping_observed": False,
            "current_item_level_procurement_closed": False,
        }
    )
    changed_observation = tmp_path / "terminal_observations.json"
    changed_observation.write_text(json.dumps(observation, indent=2, sort_keys=True) + "\n")
    config = json.loads(CONFIG.read_text())
    config["inputs"]["terminal_observations"]["sha256"] = sha256_file(changed_observation)
    changed_config = tmp_path / "config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    inputs = dict(INPUT_PATHS)
    inputs["terminal_observations"] = changed_observation
    with pytest.raises(Ugi3ExactC16RouteError, match="vendor observation"):
        build_exact_c16_route_audit(config_path=changed_config, input_paths=inputs)


def test_terminal_evidence_cannot_close_after_expiry(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value.update({"assessment_as_of_utc": "2026-09-02T02:43:32Z"}),
    )
    with pytest.raises(Ugi3ExactC16RouteError, match="evidence is expired"):
        build_exact_c16_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_terminal_observation_cannot_be_future_dated(tmp_path: Path) -> None:
    def mutate(observation):
        observation["recorded_at_utc"] = "2026-08-04T02:43:31Z"
        observation["refresh_policy"]["expires_at_utc"] = "2026-09-03T02:43:31Z"
        for item in observation["items"]:
            item["observed_at_utc"] = "2026-08-04T02:43:31Z"
            item["expires_at_utc"] = "2026-09-03T02:43:31Z"

    config, inputs = _tampered_observation(tmp_path, mutate)
    with pytest.raises(Ugi3ExactC16RouteError, match="future-dated"):
        build_exact_c16_route_audit(config_path=config, input_paths=inputs)


def test_terminal_refresh_policy_must_be_well_formed(tmp_path: Path) -> None:
    config, inputs = _tampered_observation(
        tmp_path,
        lambda value: value["refresh_policy"].update({"validity_days": "30"}),
    )
    with pytest.raises(Ugi3ExactC16RouteError, match="refresh policy is malformed"):
        build_exact_c16_route_audit(config_path=config, input_paths=inputs)


def test_assessment_time_requires_strict_iso_utc(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value.update({"assessment_as_of_utc": "2026-08-03 03:04:10+00:00"}),
    )
    with pytest.raises(Ugi3ExactC16RouteError, match="strict ISO UTC"):
        build_exact_c16_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_family_or_homologue_promotion_is_fail_closed(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["limitations"].update({"homologue_promotion_authorized": True}),
    )
    with pytest.raises(Ugi3ExactC16RouteError, match="exact-pair-only limitations"):
        build_exact_c16_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_conditions_source_hash_is_enforced(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["steps"][3].update({"conditions_source_sha256": "0" * 64}),
    )
    with pytest.raises(Ugi3ExactC16RouteError, match="conditions source hash changed"):
        build_exact_c16_route_audit(config_path=config, input_paths=INPUT_PATHS)
