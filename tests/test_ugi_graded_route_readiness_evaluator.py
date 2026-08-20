from __future__ import annotations

import json
from pathlib import Path

from forge.design.schedule.ugi_graded_route_readiness_evaluator import (
    GRADED_ROUTE_READINESS_POLICY,
    UGI_GRADED_ROUTE_READINESS_POLICY_SHA256,
    load_graded_component_index,
    load_semantic_proposal_index,
)
from forge.design.guidance.ugi_graded_route_readiness_guidance import (
    _generation_only_post_hoc_equivalence,
)

REPO = Path(__file__).resolve().parents[1]


def test_policy_is_nonprobabilistic_and_proposal_scores_are_excluded() -> None:
    assert len(UGI_GRADED_ROUTE_READINESS_POLICY_SHA256) == 64
    assert GRADED_ROUTE_READINESS_POLICY["proposal_model_score_used"] is False
    assert GRADED_ROUTE_READINESS_POLICY["utility_is_synthesis_success_probability"] is False


def test_frozen_evidence_and_proposal_indices_load() -> None:
    evidence = load_graded_component_index(
        REPO
        / "results/phase1/ugi3_graded_family_evidence_audit_v1/graded_family_evidence_ledger.csv.gz"
    )
    proposals = load_semantic_proposal_index(
        REPO / "results/phase1/graph2edits_semantic_readjudication_v1/semantic_ledger.jsonl.gz"
    )
    assert len(evidence) == 1919
    assert len(proposals) == 30
    assert all("model_score" not in value for value in proposals.values())


def test_generation_equivalence_excludes_only_route_assessment_fields() -> None:
    admission = {
        "terminal_sha256": "a" * 64,
        "generation_trace_sha256": "b" * 64,
        "route_assessment": {"policy": "old"},
    }
    group = {
        "shadow_terminal_ids": ["terminal"],
        "shadow_terminal_sha256s": ["c" * 64],
        "shadow_trace_sha256s": ["d" * 64],
    }
    zero = {
        "post_hoc": {
            "productive_terminal_ids": ["terminal"],
            "productive_canonical_identities": ["identity"],
            "productive_admissions": [admission],
            "checkpoint_groups": [group],
            "productive_route_completion_utilities": [0.0],
        }
    }
    current = json.loads(json.dumps(zero))
    current["post_hoc"]["productive_admissions"][0]["route_assessment"] = {"policy": "graded"}
    current["post_hoc"]["productive_route_completion_utilities"] = [1.0]
    result = _generation_only_post_hoc_equivalence(current, zero)
    assert result["all_generation_fields_bitwise_equal"] is True
    nested = _generation_only_post_hoc_equivalence(current, {"run": zero})
    assert nested["all_generation_fields_bitwise_equal"] is True
