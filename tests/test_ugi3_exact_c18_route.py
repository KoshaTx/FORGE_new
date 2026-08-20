from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.engine.planner import (
    AssessmentOutcome,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.evidence.ugi3_exact_c18_route import (
    Ugi3ExactC18RouteError,
    build_exact_c18_route_audit,
    load_exact_c18_route_overlay,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_exact_c18_route_v1.json"
RESULT = REPO / "results/phase1/ugi3_exact_c18_route_v1/result.json"
STEP_LEDGER = REPO / ("results/phase1/ugi3_exact_c18_route_v1/step_verification_ledger.json.gz")
ASSESSMENT = REPO / "results/phase1/ugi3_exact_c18_route_v1/assessment.json.gz"
INPUT_PATHS = {
    "audit_source": REPO / "src/forge/route/ugi3_exact_c18_route.py",
    "audit_runner": REPO / "scripts/phase1_qualify_ugi3_exact_c18_route.py",
    "audit_tests": REPO / "tests/test_ugi3_exact_c18_route.py",
    "qualified_forward_source": REPO / "src/forge/route/qualified_forward.py",
    "planner_source": REPO / "src/forge/route/planner.py",
    "synthesis_value_source": REPO / "src/forge/value/synthesis.py",
    "reaction_registry": REPO / "configs/route/phase1_ugi3_exact_c18_qualified_reactions_v1.json",
    "reduction_variant": REPO
    / "configs/route/variants/ugi3_exact_c18_carboxylic_acid_reduction_v1.json",
    "zipper_variant": REPO / "configs/route/variants/ugi3_exact_c18_alkyne_zipper_v1.json",
    "oxidation_variant": REPO
    / "configs/route/variants/ugi3_exact_c18_primary_alcohol_oxidation_v1.json",
    "amara_primary_article": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/amara_cell_chem_biol_2019.pdf",
    "oppolzer_primary_si": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/oppolzer_jo000463n_si_001.pdf",
    "oppolzer_article_transcription": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/oppolzer_article_transcription_lookchem_2026-08-03.html",
    "terminal_procurement_observation": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/mce_hy_w341997_product_observation_2026-08-03.json",
    "terminal_identity_datasheet": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/mce_hy_w341997_datasheet_2026-08-03.pdf",
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


def test_exact_c18_route_is_three_step_l2_and_l3_closed() -> None:
    first, first_steps, first_assessment = build_exact_c18_route_audit(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
    )
    second, second_steps, second_assessment = build_exact_c18_route_audit(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
    )
    assert first == second
    assert first_steps == second_steps
    assert first_assessment == second_assessment
    assert first["summary"] == {
        "assessment_as_of_utc": "2026-08-03T02:30:00Z",
        "assessment_outcome": "complete",
        "development_product_coverage_changed": True,
        "evidence_support": "exact_identity",
        "exact_steps": 3,
        "forward_consistency": "exact_unique",
        "maximum_route_depth": 3,
        "randomized_smiles_total_successes": 300,
        "randomized_smiles_trials_per_step": 100,
        "reaction_family_admitted": False,
        "route_complete": True,
        "route_step_count": 3,
        "target_role": "oxoester_aldehyde_body_tail",
        "target_smiles": "C#CCCCCCCCCCCCCCCCC=O",
        "terminal_availability": "current_closed",
        "terminal_expires_at_utc": "2026-09-02T02:24:02Z",
        "terminal_observed_at_utc": "2026-08-03T02:24:02Z",
        "unassessed_terminal_leaf_count": 0,
        "uniquely_forward_verified_steps": 3,
    }
    assert first["adjudication"] == {
        "current_l3_procurement_closed": True,
        "exact_l2_chain_admitted": True,
        "family_template_admitted": False,
        "holdout_reveal_authorized": False,
        "route_complete": True,
        "synthesis_guidance_authorized": False,
    }
    step_rows = json.loads(gzip.decompress(first_steps))["rows"]
    assert [row["conditions_source_input"] for row in step_rows] == [
        "amara_primary_article",
        "amara_primary_article",
        "oppolzer_article_transcription",
    ]
    assert all(row["conditions_source_sha256"] for row in step_rows)
    assert all(row["conditions_source_locator"] for row in step_rows)


def test_frozen_overlay_preserves_exact_chain_and_delegates_neighbors() -> None:
    base = RecordingMissingSource()
    overlay = load_exact_c18_route_overlay(
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
    assert budget.expansions == 3
    assert budget.verifier_calls == 3
    assert base.calls == []

    neighbor = RouteTarget(
        role="oxoester_aldehyde_body_tail",
        canonical_smiles="C#CCCCCCCCCCCCCCC=O",
    )
    delegated = RecursiveRouteAssessor(overlay).assess(neighbor, _budget())
    assert delegated.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert base.calls == [neighbor]


def _tampered_config(tmp_path: Path, mutate) -> Path:
    config = json.loads(CONFIG.read_text())
    mutate(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def test_terminal_cannot_close_without_explicit_current_availability(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["terminal_procurement"].update(
            {"explicit_current_stock_or_shipping_observed": False}
        ),
    )
    with pytest.raises(Ugi3ExactC18RouteError, match="inconsistent with explicit"):
        build_exact_c18_route_audit(config_path=config, input_paths=INPUT_PATHS)


@pytest.mark.parametrize(
    "assessment_as_of_utc",
    ["2026-08-03T02:24:01Z", "2026-09-02T02:24:03Z"],
)
def test_terminal_assessment_time_must_be_inside_observation_window(
    tmp_path: Path, assessment_as_of_utc: str
) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value.update({"assessment_as_of_utc": assessment_as_of_utc}),
    )
    with pytest.raises(Ugi3ExactC18RouteError, match="outside the vendor observation window"):
        build_exact_c18_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_procurement_policy_cannot_change_silently(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["procurement_policy"].update({"expiry_days": 31}),
    )
    with pytest.raises(Ugi3ExactC18RouteError, match="procurement policy changed"):
        build_exact_c18_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_procurement_timestamp_requires_strict_utc(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value.update({"assessment_as_of_utc": "2026-08-03 02:30:00"}),
    )
    with pytest.raises(Ugi3ExactC18RouteError, match="ISO-8601 UTC"):
        build_exact_c18_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_contradictory_homologue_authorization_is_rejected(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["limitations"].update({"homologue_promotion_authorized": True}),
    )
    with pytest.raises(Ugi3ExactC18RouteError, match="exact-pair-only limitations"):
        build_exact_c18_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_exact_pair_identity_cannot_be_generalized_or_changed(tmp_path: Path) -> None:
    registry = json.loads(INPUT_PATHS["reaction_registry"].read_text())
    registry["reactions"][1]["exact_product_smiles"] = "C#CCCCCCCCCCCCCCCO"
    changed_registry = tmp_path / "registry.json"
    changed_registry.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    config = json.loads(CONFIG.read_text())
    config["inputs"]["reaction_registry"]["sha256"] = sha256_file(changed_registry)
    changed_config = tmp_path / "config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    inputs = dict(INPUT_PATHS)
    inputs["reaction_registry"] = changed_registry
    with pytest.raises(Ugi3ExactC18RouteError, match="not exact-pair qualified"):
        build_exact_c18_route_audit(config_path=changed_config, input_paths=inputs)


def test_route_steps_must_form_one_contiguous_chain(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["steps"].reverse(),
    )
    with pytest.raises(Ugi3ExactC18RouteError, match="not contiguous"):
        build_exact_c18_route_audit(config_path=config, input_paths=INPUT_PATHS)


def test_conditions_source_hash_is_enforced(tmp_path: Path) -> None:
    config = _tampered_config(
        tmp_path,
        lambda value: value["steps"][2].update({"conditions_source_sha256": "0" * 64}),
    )
    with pytest.raises(Ugi3ExactC18RouteError, match="conditions source hash changed"):
        build_exact_c18_route_audit(config_path=config, input_paths=INPUT_PATHS)
