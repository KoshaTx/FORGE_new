import json
from dataclasses import replace
from gzip import open as gzip_open
from pathlib import Path

from forge.route.engine.planner import AssessmentOutcome
from forge.value.guidance.ugi_exact_closure_guidance import (
    GuidanceDisposition,
    exact_closure_potential_from_product_value,
    smc_utility_bridge_from_exact_closure,
)
from forge.value.guidance.ugi_route_completion_utility_qualification import (
    build_route_completion_utility_qualification,
)
from forge.value.synthesis.synthesis import ProductSynthesisValue

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_route_completion_utility_v1.json"


def _censored_product() -> ProductSynthesisValue:
    with gzip_open(
        REPO / "results/phase1/ugi3_fresh_pool_route_coverage_v6/product_synthesis_values.json.gz",
        "rt",
    ) as handle:
        value = ProductSynthesisValue.from_dict(json.load(handle)["records"][0]["value"])
    role, component = value.components[0]
    censored = replace(
        component,
        assessment_outcome=AssessmentOutcome.EXECUTION_ERROR,
        current_terminal_leaf_count=0,
        unavailable_terminal_leaf_count=0,
        unassessed_terminal_leaf_count=0,
        missing_knowledge_leaf_count=0,
        outside_support_leaf_count=0,
        incompatible_leaf_count=0,
        budget_exhausted_leaf_count=0,
        invalid_input_leaf_count=0,
        execution_error_leaf_count=component.leaf_count,
    )
    return replace(value, components=((role, censored), *value.components[1:]))


def test_v6_utility_census_and_sparse_signal_are_frozen() -> None:
    result = build_route_completion_utility_qualification(REPO, CONFIG)

    assert result["status"] == "binary_exact_dossier_route_completion_utility_qualified"
    assert result["census"]["product_count"] == 3975
    assert result["census"]["support_bonus_count"] == 203
    assert result["census"]["neutral_count"] == 3772
    assert result["census"]["censored_count"] == 0
    assert result["risk"]["sparse_signal"] is True
    assert result["adjudication"]["candidate_selection_authorized"] is False


def test_censored_product_retains_none_and_receives_no_bonus() -> None:
    potential = exact_closure_potential_from_product_value(_censored_product())
    bridge = smc_utility_bridge_from_exact_closure(potential)

    assert potential.disposition is GuidanceDisposition.CENSOR
    assert potential.smc_potential is None
    assert bridge.route_completion_utility is None
    assert bridge.incremental_log_weight == 0.0
    assert bridge.support_bonus is False
    assert bridge.to_dict()["evidence_upgraded"] is False
    assert bridge.to_dict()["success_probability"] is None


def test_nonexact_l1_cannot_receive_positive_bonus_or_evidence_upgrade() -> None:
    value = replace(_censored_product(), l1_forward_consistent=False)
    potential = exact_closure_potential_from_product_value(value)
    bridge = smc_utility_bridge_from_exact_closure(potential)

    assert potential.disposition is GuidanceDisposition.CENSOR
    assert potential.smc_potential is None
    assert bridge.incremental_log_weight == 0.0
    assert bridge.support_bonus is False
    assert bridge.to_dict()["evidence_upgraded"] is False
