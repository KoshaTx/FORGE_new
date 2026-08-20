"""Focused tests for the frozen bounded hybrid Ugi route cascade.

These protect the scientific invariants that an agent optimizing for a green
run would otherwise break: arm blindness, proposal-only authority, the
four-state component vocabulary, denominator preservation and deterministic
artifact bytes.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.phase1.synthesis_guidance.route_cascade import (
    COMPLETE,
    COMPONENT_STATES,
    CONFIG_SCHEMA_VERSION,
    EXPANDED_THEN_UNRESOLVED,
    EXPECTED_INPUTS,
    EXPECTED_SCOPE,
    NEVER_EXPANDED,
    NOT_ASSESSED_OUTCOME,
    REJECTED,
    SEARCH_CENSORED,
    UNRESOLVED,
    UgiBoundedHybridRouteCascadeError,
    assert_expected_population,
    attrition_csv_rows,
    blindness_receipt,
    build_candidate_population,
    collect_route_leaves,
    component_descriptors,
    component_sha256,
    csv_gzip_bytes,
    derive_final_component_state,
    jsonl_gzip_bytes,
    load_cascade_contract,
    product_route_state,
    read_csv_gzip,
    rescoring_index,
    resolve_potency_reporting,
    sha256_payload,
    tail_length_bucket,
    unique_component_targets,
    unresolved_disposition,
    unresolved_leaf_classes,
)
from forge.potency.annotations import ROLE_NAMES

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi_bounded_hybrid_route_cascade_v1.json"
OUTPUT = REPO / "results/phase1/ugi_bounded_hybrid_route_cascade_v1"

requires_config = pytest.mark.skipif(
    not CONFIG.is_file(), reason="frozen bounded hybrid cascade config is absent"
)


@pytest.fixture(scope="module")
def contract():
    if not CONFIG.is_file():
        pytest.skip("frozen bounded hybrid cascade config is absent")
    return load_cascade_contract(REPO, CONFIG)


@pytest.fixture(scope="module")
def records(contract):
    return build_candidate_population(contract)


# ---------------------------------------------------------------------------
# config contract
# ---------------------------------------------------------------------------


@requires_config
def test_config_schema_and_scope_are_frozen(contract):
    assert contract.config["schema_version"] == CONFIG_SCHEMA_VERSION
    assert contract.config["scope"] == EXPECTED_SCOPE
    assert set(contract.config["inputs"]) == EXPECTED_INPUTS
    assert contract.config["scope"]["arm_blind_single_lane"] is True
    assert contract.config["scope"]["guided_post_hoc_cache_split_inherited"] is False
    assert contract.config["scope"]["public_stock_solution_is_evidence"] is False
    assert contract.config["scope"]["prospective_panel_lock"] is False
    assert contract.config["scope"]["preregistration_created"] is False
    snapshot = contract.config["procurement_snapshot"]
    assert snapshot["is_live_availability"] is False
    assert snapshot["refresh_required_before_panel_lock"] is True


@requires_config
def test_scope_drift_fails_closed(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["scope"]["public_stock_solution_is_evidence"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(UgiBoundedHybridRouteCascadeError, match="scope changed"):
        load_cascade_contract(REPO, path)


@requires_config
def test_tampered_input_hash_fails_closed(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["inputs"]["shortlist_ledger"]["sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(UgiBoundedHybridRouteCascadeError, match="input pin changed"):
        load_cascade_contract(REPO, path)


@requires_config
def test_missing_input_pin_fails_closed(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["inputs"].pop("graded_evidence_ledger")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(UgiBoundedHybridRouteCascadeError, match="input set changed"):
        load_cascade_contract(REPO, path)


@requires_config
def test_procurement_snapshot_cannot_claim_live_availability(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["procurement_snapshot"]["is_live_availability"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(UgiBoundedHybridRouteCascadeError, match="procurement snapshot"):
        load_cascade_contract(REPO, path)


# ---------------------------------------------------------------------------
# population and denominators
# ---------------------------------------------------------------------------


@requires_config
def test_population_matches_frozen_expectation(contract, records):
    observed = assert_expected_population(contract, records)
    assert observed["rows"] == 256
    assert observed["unique_products"] == 256
    assert observed["arms"] == {
        "branch_exploration": 64,
        "broad_prior": 96,
        "support_enriched": 96,
    }
    assert observed["unique_components_total"] == 158
    assert observed["unique_components"] == {
        "amine_head": 48,
        "isocyanide_tail": 19,
        "oxoester_aldehyde_body_tail": 91,
    }


@requires_config
def test_population_drift_fails_closed(contract, records):
    class _Drifted:
        repo = contract.repo
        config_path = contract.config_path
        paths = contract.paths
        config = {
            **contract.config,
            "expected_population": {
                **contract.config["expected_population"],
                "rows": 255,
            },
        }

    with pytest.raises(UgiBoundedHybridRouteCascadeError, match="population changed"):
        assert_expected_population(_Drifted(), records)


@requires_config
def test_conservative_high_is_resolved_from_an_authoritative_source(contract, records):
    """The two shortlist cohorts use different schemas; neither may default to False."""

    index = rescoring_index(read_csv_gzip(contract.paths["terminal_rescoring_ledger"]))
    resolved = [resolve_potency_reporting(record, index) for record in records]
    assert len(resolved) == 256
    sources = Counter(item["conservative_high_source"] for item in resolved)
    # 192 preserved linear rows lack the flag and must come from the frozen ledger.
    assert sources["terminal_rescoring_v3"] == 192
    assert sources["shortlist_ledger"] == 64
    high = [item for item in resolved if item["conservative_high_potency"]]
    # 192 potency-exploitation rows plus the 15 conservative-high branch rows.
    assert len(high) == 207


@requires_config
def test_potency_resolution_fails_closed_on_identity_mismatch(contract, records):
    record = next(item for item in records if "conservative_high_potency" not in item.shortlist_row)
    index = rescoring_index(read_csv_gzip(contract.paths["terminal_rescoring_ledger"]))
    tampered = dict(index)
    row = dict(tampered[(record.arm_id, record.draw_index)])
    row["canonical_product"] = "CCO"
    tampered[(record.arm_id, record.draw_index)] = row
    with pytest.raises(UgiBoundedHybridRouteCascadeError, match="product identity differs"):
        resolve_potency_reporting(record, tampered)


@requires_config
def test_every_candidate_carries_exact_constitutional_l1(records):
    for record in records:
        verification = record.native_terminal["l1_forward_verification"]
        assert verification["exact_product_reconstructed"] is True
        assert set(record.components) == set(ROLE_NAMES)


# ---------------------------------------------------------------------------
# arm blindness
# ---------------------------------------------------------------------------


@requires_config
def test_blindness_receipt_is_invariant_to_arm_labels(contract, records):
    baseline = blindness_receipt(contract, records)

    class _Relabelled:
        def __init__(self, record, arm_id):
            self.arm_id = arm_id
            self.draw_index = record.draw_index
            self.canonical_product = record.canonical_product
            self.components = record.components
            self.shortlist_row = record.shortlist_row
            self.generation_row = record.generation_row

    rotated = [
        _Relabelled(record, {"broad_prior": "support_enriched"}.get(record.arm_id, "broad_prior"))
        for record in records
    ]
    assert blindness_receipt(contract, rotated) == baseline
    assert baseline["arm_influences_route_search"] is False
    assert baseline["potency_influences_route_search"] is False
    assert "arm_id" in baseline["reporting_only_fields"]
    assert "conservative_high_potency" in baseline["reporting_only_fields"]
    assert baseline["route_input_fields"] == ["canonical_product", "components"]


@requires_config
def test_component_targets_are_arm_independent(records):
    targets = unique_component_targets(records)
    shuffled = list(reversed(records))
    assert unique_component_targets(shuffled) == targets
    assert len(targets) == 158


def test_component_identity_excludes_arm_and_potency():
    key = component_sha256("amine_head", "NCCN1CCCCC1")
    assert key == sha256_payload(
        {
            "schema": "forge.ugi_bounded_hybrid_route_component_identity.v1",
            "role": "amine_head",
            "canonical_smiles": "NCCN1CCCCC1",
        }
    )
    assert key != component_sha256("isocyanide_tail", "NCCN1CCCCC1")


# ---------------------------------------------------------------------------
# state vocabulary and roll-up
# ---------------------------------------------------------------------------


def test_four_state_vocabulary_is_exhaustive_and_exclusive():
    assert set(COMPONENT_STATES) == {COMPLETE, UNRESOLVED, REJECTED, SEARCH_CENSORED}
    assert len(COMPONENT_STATES) == 4
    outcomes = (
        "complete",
        "incompatible",
        "outside_support",
        "missing_knowledge",
        "budget_exhausted",
        "invalid_input",
        "execution_error",
    )
    for outcome in outcomes:
        for censored in (False, True):
            state, basis = derive_final_component_state(
                strict_complete=outcome == "complete" and not censored,
                assessment_outcome=outcome,
                proposal_search_censored=censored,
            )
            assert state in COMPONENT_STATES
            assert basis


def test_only_independent_exact_closure_returns_complete():
    state, basis = derive_final_component_state(
        strict_complete=True,
        assessment_outcome="complete",
        proposal_search_censored=False,
    )
    assert state == COMPLETE
    assert "independent_exact_recursive_closure" in basis
    # A proposal-rich but unclosed component can never become complete.
    for outcome in ("missing_knowledge", "outside_support"):
        state, _ = derive_final_component_state(
            strict_complete=False,
            assessment_outcome=outcome,
            proposal_search_censored=False,
        )
        assert state == UNRESOLVED


def test_component_without_admitted_parent_is_search_censored_not_unresolved():
    state, basis = derive_final_component_state(
        strict_complete=False,
        assessment_outcome=NOT_ASSESSED_OUTCOME,
        proposal_search_censored=False,
    )
    assert state == SEARCH_CENSORED
    assert NOT_ASSESSED_OUTCOME in basis


def test_strict_closure_disagreeing_with_outcome_fails_closed():
    with pytest.raises(UgiBoundedHybridRouteCascadeError, match="strict closure disagrees"):
        derive_final_component_state(
            strict_complete=True,
            assessment_outcome="missing_knowledge",
            proposal_search_censored=False,
        )


def test_censored_and_rejected_states_are_distinguished():
    for outcome in ("budget_exhausted", "invalid_input", "execution_error"):
        state, _ = derive_final_component_state(
            strict_complete=False,
            assessment_outcome=outcome,
            proposal_search_censored=False,
        )
        assert state == SEARCH_CENSORED
    state, _ = derive_final_component_state(
        strict_complete=False,
        assessment_outcome="incompatible",
        proposal_search_censored=False,
    )
    assert state == REJECTED
    state, basis = derive_final_component_state(
        strict_complete=False,
        assessment_outcome="missing_knowledge",
        proposal_search_censored=True,
    )
    assert state == SEARCH_CENSORED
    assert "proposal_engine" in basis


def test_unresolved_disposition_separates_index_miss_from_failed_search():
    """The closure rate must never be readable as a search outcome without this split."""

    unexpanded = {"target": {}, "outcome": "missing_knowledge", "evidence": [], "children": []}
    expanded = {
        "target": {},
        "outcome": "missing_knowledge",
        "evidence": [],
        "children": [
            {"target": {}, "outcome": "missing_knowledge", "evidence": [], "children": []}
        ],
    }
    assert unresolved_disposition(unexpanded, strict_complete=False) == NEVER_EXPANDED
    assert unresolved_disposition(expanded, strict_complete=False) == EXPANDED_THEN_UNRESOLVED
    assert unresolved_disposition(None, strict_complete=False) == NEVER_EXPANDED
    # A closed component has no unresolved disposition at all.
    assert unresolved_disposition(expanded, strict_complete=True) is None
    assert unresolved_disposition(unexpanded, strict_complete=True) is None


def test_completed_run_reports_the_index_miss_split():
    result = _load_result("result.json")
    interp = result["interpretation"]
    assert interp["not_a_synthesizability_claim"] is True
    assert interp["proposal_engines_closed_zero_routes"] is True
    assert "NOT a retrosynthetic search result" in interp["what_route_complete_measures"]
    total = interp["unresolved_never_expanded"] + interp["unresolved_after_search"]
    assert total == sum(
        count for state, count in result["summary"]["component_states"].items() if state != COMPLETE
    )
    # The whole point: index misses dominate, so the split must be visible.
    assert interp["unresolved_never_expanded"] > 0


def test_product_requires_all_three_roles_to_close():
    assert product_route_state([COMPLETE, COMPLETE, COMPLETE]) == COMPLETE
    assert product_route_state([COMPLETE, COMPLETE, UNRESOLVED]) == UNRESOLVED
    assert product_route_state([COMPLETE, REJECTED, UNRESOLVED]) == REJECTED
    assert product_route_state([COMPLETE, SEARCH_CENSORED, UNRESOLVED]) == SEARCH_CENSORED
    with pytest.raises(UgiBoundedHybridRouteCascadeError, match="three typed component states"):
        product_route_state([COMPLETE, COMPLETE])


# ---------------------------------------------------------------------------
# route-tree leaves
# ---------------------------------------------------------------------------


def test_route_leaf_collection_and_unresolved_classes():
    tree = {
        "target": {"role": "amine_head", "canonical_smiles": "CCN"},
        "outcome": "missing_knowledge",
        "evidence": [],
        "children": [
            {
                "target": {"role": "amine_head", "canonical_smiles": "CC=O"},
                "outcome": "complete",
                "evidence": [{"tier": "exact_terminal_current"}],
                "children": [],
            },
            {
                "target": {"role": "amine_head", "canonical_smiles": "CCO"},
                "outcome": "missing_knowledge",
                "evidence": [],
                "children": [],
            },
        ],
    }
    leaves = collect_route_leaves(tree)
    assert len(leaves) == 2
    assert {leaf["leaf_smiles"] for leaf in leaves} == {"CC=O", "CCO"}
    assert unresolved_leaf_classes(leaves) == {"missing_knowledge": 1}
    assert collect_route_leaves({**tree, "children": []})[0]["depth"] == 0


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------


def test_descriptors_separate_ester_ether_and_unsaturation():
    ester = component_descriptors("CCCCCC(C)CCC(=O)OCCCCCC=O")
    assert ester["ester_groups"] == 1
    assert ester["ether_oxygens"] == 0
    assert ester["carbon_branch_points"] == 1
    unsaturated = component_descriptors("[C-]#[N+]CCCCCCCCC=CCCCCCCCC")
    assert unsaturated["unsaturated"] is True
    assert unsaturated["carbon_carbon_double_bonds"] == 1
    saturated = component_descriptors("NCCN1CCCCC1")
    assert saturated["unsaturated"] is False


def test_tail_length_buckets_are_monotone():
    assert tail_length_bucket(4) == "c00_c08"
    assert tail_length_bucket(10) == "c09_c12"
    assert tail_length_bucket(16) == "c13_c16"
    assert tail_length_bucket(18) == "c17_c20"
    assert tail_length_bucket(24) == "c21_plus"


# ---------------------------------------------------------------------------
# deterministic artifacts
# ---------------------------------------------------------------------------


def test_gzip_artifacts_are_byte_deterministic():
    rows = [{"b": 2, "a": 1}, {"a": 3, "b": 4}]
    assert jsonl_gzip_bytes(rows) == jsonl_gzip_bytes(rows)
    decoded = gzip.decompress(jsonl_gzip_bytes(rows)).decode().splitlines()
    assert decoded == ['{"a":1,"b":2}', '{"a":3,"b":4}']
    csv_rows = [{"grouping": "arm", "group": "broad_prior", "rows": 96}]
    first = csv_gzip_bytes(("grouping", "group", "rows"), csv_rows)
    assert first == csv_gzip_bytes(("grouping", "group", "rows"), csv_rows)
    assert gzip.decompress(first).decode().startswith("grouping,group,rows\n")


def test_attrition_csv_rows_flatten_both_groupings():
    group = {
        "rows": 1,
        "unique_products": 1,
        "exact_l1_pass_rows": 1,
        "route_complete_rows": 0,
        "route_complete_unique_products": 0,
        "exact_evidence_current_route_rows": 0,
        "graph2edits_proposal_coverage_rows": 1,
        "aizynthfinder_incremental_coverage_rows": 0,
        "union_proposal_coverage_rows": 1,
        "conservative_high_rows": 0,
        "conservative_high_route_complete_rows": 0,
        "branched_rows": 0,
        "branched_route_complete_rows": 0,
        "unsaturated_rows": 0,
        "corpus_absent_rows": 1,
        "corpus_absent_route_complete_rows": 0,
        "unique_components_present": {
            "amine_head": 1,
            "oxoester_aldehyde_body_tail": 1,
            "isocyanide_tail": 1,
        },
    }
    rows = attrition_csv_rows({"by_arm": {"broad_prior": group}, "by_cohort": {"c": group}})
    assert [row["grouping"] for row in rows] == ["arm", "cohort"]
    assert rows[0]["unique_amines"] == 1


# ---------------------------------------------------------------------------
# completed-run assertions (skipped until the cascade has been executed)
# ---------------------------------------------------------------------------


def _load_result(name: str):
    path = OUTPUT / name
    if not path.is_file():
        pytest.skip(f"cascade stage artifact is absent: {name}")
    return json.loads(path.read_bytes())


def _load_ledger(name: str):
    path = OUTPUT / name
    if not path.is_file():
        pytest.skip(f"cascade ledger is absent: {name}")
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_completed_run_preserves_denominators():
    result = _load_result("result.json")
    assert result["summary"]["candidates"] == 256
    assert result["summary"]["unique_products"] == 256
    assert result["summary"]["components"] == 158
    assert sum(result["summary"]["product_route_states"].values()) == 256
    assert sum(result["summary"]["component_states"].values()) == 158


def test_completed_run_never_locks_a_panel():
    result = _load_result("result.json")
    assert result["decision"]["panel_locked"] is False
    assert result["decision"]["preregistration_created"] is False
    assert result["decision"]["candidate_selection_performed"] is False
    assert result["decision"]["procurement_refresh_required_before_panel_lock"] is True


def test_completed_run_cache_key_excludes_reporting_fields():
    result = _load_result("result.json")
    cache = result["cache"]
    assert cache["single_lane"] is True
    assert cache["guided_post_hoc_split_inherited"] is False
    assert set(cache["cache_key_fields"]) == {
        "schema_version",
        "target",
        "context",
        "resource_usage_before_call",
    }
    for excluded in ("arm_id", "cohort", "authority_tier", "potency_utility"):
        assert excluded in cache["cache_key_excludes"]


def test_completed_run_records_every_non_admitted_candidate():
    """A gate failure must be retained with its reason, never silently dropped."""

    result = _load_result("result.json")
    rows = _load_ledger("candidate_route_ledger.jsonl.gz")
    admitted = [row for row in rows if row["route_assessment_admitted"]]
    not_admitted = [row for row in rows if not row["route_assessment_admitted"]]
    assert len(admitted) + len(not_admitted) == 256
    assert len(admitted) == result["summary"]["route_assessment_admitted_candidates"]
    assert len(not_admitted) == len(result["summary"]["route_assessment_not_admitted_candidates"])
    for row in not_admitted:
        failure = row["admission_failure"]
        assert failure["gate_was_not_relaxed"] is True
        assert failure["candidate_retained_in_denominator"] is True
        assert failure["detail"]["out_of_vocabulary_atom_states"]
        assert row["product_route_state"] != COMPLETE


def test_completed_run_component_ledger_authority():
    rows = _load_ledger("component_route_ledger.jsonl.gz")
    assert len(rows) == 158
    for row in rows:
        assert row["final_component_state"] in COMPONENT_STATES
        if row["final_component_state"] == COMPLETE:
            assert row["exact_evidence"]["strict_complete"] is True
        assert row["graded_family_projection"]["family_projection_is_exact_route_closure"] is False
        assert row["adjudication"]["proposal_only_engines_cannot_close_route"] is True


def test_completed_run_candidate_rollup_requires_three_roles():
    rows = _load_ledger("candidate_route_ledger.jsonl.gz")
    assert len(rows) == 256
    for row in rows:
        states = [item["final_component_state"] for item in row["roles"]]
        assert row["product_route_state"] == product_route_state(states)
        assert row["exact_l1_reverified"] is True
        assert row["route_inputs_excluded_reporting_fields"] is True
        assert "arm_id" in row["reporting"]
