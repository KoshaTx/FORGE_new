from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.route.evidence.ugi3_high_leverage_route_evidence_worklist import (
    MANIFEST_SCHEMA_VERSION,
    RouteEvidenceWorklistError,
    build_high_leverage_route_evidence_worklist,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_high_leverage_route_evidence_worklist_v1.json"


def _build() -> dict:
    return build_high_leverage_route_evidence_worklist(CONFIG, repo_root=REPO)


def test_worklist_is_cardinality_minimal_and_matches_frozen_25pct_audit() -> None:
    result = _build()
    selection = result["selection"]

    assert result["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert result["development_only"] is True
    assert result["sealed_holdout_accessed"] is False
    assert selection["selected_component_count"] == 16
    assert selection["target_unlock_count"] == 201
    assert selection["prior_prefix_unlock_count"] == 199
    assert selection["potential_unlock_count"] == 207
    assert selection["potential_unlock_fraction"] == pytest.approx(207 / 802)
    assert selection["components_with_existing_upstream_program"] == 13
    assert selection["heads_without_existing_program"] == 3
    assert selection["strict_existing_program_only_sensitivity"] == {
        "selected_component_count": 18,
        "prior_prefix_unlock_count": 198,
        "potential_unlock_count": 204,
        "potential_unlock_fraction": pytest.approx(204 / 802),
        "interpretation": (
            "excludes the three high-priority heads whose exact upstream program "
            "is not established"
        ),
    }


def test_every_component_is_missing_knowledge_nonpromoting_and_exactly_grouped() -> None:
    result = _build()
    components = result["components"]
    groups = result["groups"]

    assert sum(item["potential_unlock_count"] for item in components) == 207
    assert sum(item["component_count"] for item in groups) == 16
    assert sum(item["potential_unlock_count"] for item in groups) == 207
    assert all(
        item["current_evidence_state"]["assessment_outcome"] == "missing_knowledge"
        for item in components
    )
    assert all(item["route_asserted"] is False for item in components)
    assert all(item["procurement_asserted"] is False for item in components)
    assert all(item["new_reaction_family_admitted"] is False for item in components)
    assert all(
        item["learned_proposal_search"]["can_supply_evidence"] is False for item in components
    )
    assert all(item["learned_proposal_search"]["can_close_route"] is False for item in components)
    assert all(
        item["learned_proposal_search"]["can_admit_reaction_family"] is False for item in components
    )
    assert {item["existing_upstream_program"] for item in components} == {
        None,
        "fatty_acid_diol_esterification_then_alcohol_oxidation",
        "primary_alcohol_oxidation",
        "primary_amine_formylation_then_formamide_dehydration",
    }


def test_required_leaves_and_missing_evidence_are_explicit() -> None:
    result = _build()
    components = result["components"]

    for item in components:
        state = item["current_evidence_state"]
        assert state["projected_leaf_count"] == len(item["required_leaves"])
        assert set(state["unresolved_projected_leaves"]).issubset(item["required_leaves"])
        assert item["precise_missing_evidence"]
        if item["existing_upstream_program"] is None:
            assert item["role"] == "amine_head"
            assert item["required_leaves"] == []
            assert {entry["evidence_type"] for entry in item["precise_missing_evidence"]} == {
                "current_exact_procurement_or_stocked_terminal",
                "exact_upstream_route_if_not_currently_procured",
            }
        else:
            assert any(
                entry["evidence_type"] == "exact_substrate_execution_or_bounded_scope_qualification"
                for entry in item["precise_missing_evidence"]
            )


def test_config_rejects_any_attempt_to_substitute_a_holdout_path(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["inputs"]["priority_result"][
        "path"
    ] = "results/phase1/ugi3_route_saturation_holdout_v1/records.json"
    malicious = tmp_path / "malicious.json"
    malicious.write_text(json.dumps(config))

    with pytest.raises(RouteEvidenceWorklistError, match="frozen development path"):
        build_high_leverage_route_evidence_worklist(malicious, repo_root=REPO)
