from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from forge.synthesis.engine.single_step_benchmark_manifest import _implementation_evidence

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/single_step_proposal_lane_qualification_benchmark_v1.json"

EXPECTED_LANES = (
    "strict_lipid_precedented",
    "unrestricted_graph2edits_proposal_only",
    "strict_first_hybrid",
)
EXPECTED_STRATA = {
    "known_exact_l2_routes": 18,
    "held_reaction_families": 18,
    "linear_aldehydes": 12,
    "branched_aldehydes": 12,
    "unsaturated_aldehydes": 12,
    "ester_containing_aldehydes": 12,
    "isocyanide_formamide_precursors": 12,
    "heterocyclic_amine_heads": 12,
    "adversarial_incompatibles": 12,
}
EXPECTED_METRICS = {
    "known_route_top_k_recovery_by_exact_canonical_reactant_multiset",
    "exact_forward_unique_pass_fraction",
    "operational_screen_yield",
    "proposal_source_provenance_counts",
    "invalid_raw_hypothesis_fraction",
    "canonical_duplicate_fraction",
    "latency_p50_p95_and_max_seconds",
    "peak_host_and_accelerator_memory",
    "accepted_proposals_per_target",
    "accepted_proposals_per_matched_budget",
}


def _load() -> dict:
    return json.loads(CONFIG.read_text())


def _validate(spec: dict) -> None:
    assert spec["schema_version"] == ("forge.single_step_proposal_lane_qualification_benchmark.v1")
    assert spec["status"] == "frozen_specification_not_executed"
    assert tuple(lane["lane_id"] for lane in spec["lanes"]) == EXPECTED_LANES
    assert all(
        spec["activation"][field] is False
        for field in (
            "benchmark_execution_authorized",
            "graph2edits_backend_active",
            "hybrid_production_lane_active",
            "candidate_selection_authorized",
        )
    )
    assert spec["activation"]["strict_lane_remains_production_default"] is True

    policy = spec["target_manifest_policy"]
    assert policy["manifest_required_before_any_backend_execution"] is True
    assert policy["manifest_currently_frozen"] is False
    assert policy["development_sources_only"] is True
    assert policy["sealed_holdout_access_forbidden"] is True
    assert policy["known_routes_hidden_from_lane_and_retained_for_scoring"] is True
    assert policy["held_family_local_records_and_templates_hidden_from_all_lanes"] is True
    strata = {row["stratum_id"]: row["count"] for row in spec["target_strata"]}
    assert strata == EXPECTED_STRATA
    assert sum(strata.values()) == policy["target_count"] == 120

    budget = spec["matched_budget"]["per_target"]
    assert budget == {
        "proposal_source_calls": 1,
        "raw_hypothesis_attempts": 200,
        "canonical_hypotheses_considered": 200,
        "forward_verifier_calls": 50,
        "operational_screen_calls": 50,
        "accepted_proposal_cap": 20,
        "wall_seconds": 30,
        "cpu_threads": 1,
        "peak_host_memory_mebibytes": 12288,
        "peak_accelerator_memory_mebibytes": 12288,
    }
    hybrid = spec["matched_budget"]["hybrid_allocation"]
    assert hybrid["strict_reserved_hypotheses"] + hybrid["learned_reserved_hypotheses"] == (
        budget["raw_hypothesis_attempts"]
    )
    assert hybrid["cross_source_canonical_deduplication"] is True
    assert all(spec["matched_budget"]["matching_policy"].values())

    verifier = spec["forward_verifier_resolution"]
    assert verifier["proposal_source_may_select_or_define_verifier"] is False
    assert verifier["graph2edits_reaction_class_is_verifier_evidence"] is False
    assert verifier["try_all_independently_admitted_transform_and_role_assignments"] is True
    assert verifier["ugi_l1_registry_may_verify_l2_proposals"] is False
    assert verifier["proposal_output_may_verify_itself"] is False
    assert verifier["admitted_registries"] == [
        "qualified_l2_exact_source_reactions",
        "qualified_l2_exact_c16_reactions",
        "qualified_l2_exact_c18_reactions",
    ]
    assert verifier["acceptance_rule"] == {
        "required_matching_admitted_transform_assignments": 1,
        "required_forward_products_for_matching_assignment": 1,
        "required_exact_constitutional_target_reconstruction": True,
        "reactant_multiset_must_be_canonical_and_connected": True,
    }
    assert set(verifier["censor_rules"].values()) == {
        "censor_no_verifier",
        "reject_forward_mismatch",
        "censor_ambiguous_forward_products",
        "censor_ambiguous_transform_assignment",
        "censor_budget_exhausted",
    }

    assert spec["metrics"]["top_k"] == [1, 5, 10, 20]
    assert set(spec["metrics"]["required"]) == EXPECTED_METRICS
    assert (
        spec["metrics"]["reporting"]["model_score_reported_only_as_opaque_within_lane_rank"] is True
    )

    authority = spec["scientific_authority"]
    for field in (
        "proposal_records_are_route_evidence",
        "model_score_may_enter_v_syn",
        "model_score_may_set_evidence_tier",
        "model_score_may_close_route",
        "exact_forward_unique_is_experimental_evidence",
        "operational_screen_pass_is_experimental_evidence",
    ):
        assert authority[field] is False
    assert authority["independent_evidence_and_l3_closure_remain_required"] is True
    assert authority["proposal_source_provenance_is_retained"] is True

    gates = spec["prerequisite_gates"]
    assert gates["all_gates_passed"] is False
    assert gates["institutional_license_review_complete"] is False
    assert gates["isolated_runtime_lock_frozen"] is False
    assert gates["ten_repeat_byte_identity_on_production_device"] is False
    assert gates["development_target_manifest_frozen"] is False
    assert spec["scope_guards"] == {
        "graph2edits_executed": False,
        "benchmark_targets_materialized": False,
        "adapter_files_modified": False,
        "route_evidence_created": False,
        "v_syn_modified": False,
        "production_planner_modified": False,
        "sealed_holdout_accessed": False,
        "candidate_selected": False,
    }


def test_frozen_specification_has_all_required_lanes_strata_budgets_and_metrics() -> None:
    _validate(_load())


def test_every_artifact_pin_authenticates_and_avoids_forbidden_holdout() -> None:
    spec = _load()
    forbidden = tuple(spec["forbidden_inputs"])
    assert spec["artifacts"]
    for record in spec["artifacts"].values():
        relative = record["path"]
        assert not any(relative.startswith(prefix) for prefix in forbidden)
        path = (
            _implementation_evidence(REPO, record, label=relative)
            if Path(relative).suffix == ".py"
            else REPO / relative
        )
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_candidate_manifest_is_inactive_and_proposal_only() -> None:
    spec = _load()
    candidate_path = REPO / spec["artifacts"]["graph2edits_candidate_manifest"]["path"]
    candidate = json.loads(candidate_path.read_text())

    assert candidate["activation_authorized"] is False
    assert candidate["runtime"]["environment_lock_complete"] is False
    assert candidate["scientific_authority"] == {
        "proposal_only": True,
        "may_create_evidence": False,
        "may_set_synthesis_success_probability": False,
        "may_close_route": False,
        "may_enter_synthesis_value_without_independent_screening": False,
    }


def test_l2_verifier_uses_upstream_registries_not_the_ugi_l1_registry() -> None:
    spec = _load()
    artifacts = spec["artifacts"]
    verifier = spec["forward_verifier_resolution"]

    assert "qualified_l1_assembly_reaction" in artifacts
    assert "qualified_l1_assembly_reaction" not in verifier["admitted_registries"]
    assert set(verifier["admitted_registries"]) <= set(artifacts)

    allowed_statuses = {
        "qualified_for_exact_source_forward_verification_only",
        "qualified_for_one_exact_substrate_product_pair_only",
    }
    observed_statuses: set[str] = set()
    for label in verifier["admitted_registries"]:
        registry = json.loads((REPO / artifacts[label]["path"]).read_text())
        assert registry["reactions"]
        observed_statuses.update(reaction["status"] for reaction in registry["reactions"])
    assert observed_statuses == allowed_statuses


def test_unrestricted_lane_is_not_a_lipid_family_whitelist() -> None:
    lanes = {lane["lane_id"]: lane for lane in _load()["lanes"]}
    strict = lanes["strict_lipid_precedented"]
    learned = lanes["unrestricted_graph2edits_proposal_only"]
    hybrid = lanes["strict_first_hybrid"]

    assert strict["lipid_family_whitelist_applies"] is True
    assert learned["lipid_family_whitelist_applies"] is False
    assert hybrid["lipid_family_whitelist_applies"] is False
    assert learned["sources"] == ["graph2edits_model_proposal_only"]
    assert "graph2edits_model_proposal_only" in hybrid["sources"]


def test_promotion_is_hybrid_only_and_fail_closed() -> None:
    policy = _load()["promotion_rule"]
    assert policy["only_hybrid_lane_may_be_promoted"] is True
    assert policy["all_prerequisite_gates_must_pass"] is True
    assert policy["hybrid_must_improve_held_family_stratum"] is True
    assert policy["minimum_additional_nonadversarial_chemotype_strata_improved"] == 3
    assert policy["adversarial_incompatible_accepted_proposals_maximum"] == 0
    assert policy["tie_or_failed_gate_keeps_backend_inactive"] is True
    assert policy["no_retry_or_threshold_revision_after_scored_run"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        lambda spec: spec["activation"].update({"graph2edits_backend_active": True}),
        lambda spec: spec["scientific_authority"].update({"model_score_may_enter_v_syn": True}),
        lambda spec: spec["target_manifest_policy"].update(
            {"sealed_holdout_access_forbidden": False}
        ),
        lambda spec: spec["matched_budget"]["hybrid_allocation"].update(
            {"learned_reserved_hypotheses": 101}
        ),
        lambda spec: spec["forward_verifier_resolution"].update(
            {"proposal_output_may_verify_itself": True}
        ),
    ),
)
def test_safety_or_budget_mutation_breaks_the_frozen_contract(mutation) -> None:
    spec = deepcopy(_load())
    mutation(spec)
    with pytest.raises(AssertionError):
        _validate(spec)
