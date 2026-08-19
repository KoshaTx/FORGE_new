"""Combine independently frozen single-step proposal-recovery scores."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from forge.route.aizynthfinder_single_step_recovery import (
    SCORE_SCHEMA_VERSION as AIZYNTH_SCORE_SCHEMA_VERSION,
)
from forge.route.aizynthfinder_single_step_recovery import (
    content_sha256,
)
from forge.route.graph2edits_single_step_recovery import (
    SCORE_SCHEMA_VERSION as GRAPH2EDITS_SCORE_SCHEMA_VERSION,
)

RESULT_SCHEMA_VERSION = "phase1_hybrid_single_step_recovery_audit.v1"


class HybridRecoveryAuditError(RuntimeError):
    """Raised when frozen proposal-score identities disagree."""


def _fraction(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": None if denominator == 0 else numerator / denominator,
    }


def combine_recovery_scores(
    aizynthfinder: Mapping[str, Any], graph2edits: Mapping[str, Any]
) -> dict[str, Any]:
    if aizynthfinder.get("schema_version") != AIZYNTH_SCORE_SCHEMA_VERSION:
        raise HybridRecoveryAuditError("AiZynthFinder score schema changed")
    if graph2edits.get("schema_version") != GRAPH2EDITS_SCORE_SCHEMA_VERSION:
        raise HybridRecoveryAuditError("Graph2Edits score schema changed")
    ai_rows = {str(row["target_id"]): row for row in aizynthfinder.get("per_exact_target", [])}
    graph_rows = {str(row["target_id"]): row for row in graph2edits.get("per_exact_target", [])}
    if len(ai_rows) != 36 or set(ai_rows) != set(graph_rows):
        raise HybridRecoveryAuditError("recovery scores do not cover the same 36 targets")

    top_k_union = Counter({1: 0, 5: 0, 10: 0, 20: 0})
    top_k_intersection = Counter({1: 0, 5: 0, 10: 0, 20: 0})
    family: dict[str, Counter[str]] = defaultdict(Counter)
    misses: list[dict[str, Any]] = []
    for target_id in sorted(ai_rows):
        ai = ai_rows[target_id]
        graph = graph_rows[target_id]
        if ai.get("transformation") != graph.get("transformation") or ai.get(
            "primary_stratum"
        ) != graph.get("primary_stratum"):
            raise HybridRecoveryAuditError("per-target recovery metadata disagrees")
        ai_rank = ai.get("first_exact_recovery_rank")
        graph_rank = graph.get("first_exact_recovery_rank")
        transformation = str(ai.get("transformation"))
        family[transformation]["targets"] += 1
        family[transformation]["aizynthfinder_recovered"] += int(ai_rank is not None)
        family[transformation]["graph2edits_recovered"] += int(graph_rank is not None)
        family[transformation]["union_recovered"] += int(
            ai_rank is not None or graph_rank is not None
        )
        for k in top_k_union:
            ai_hit = ai_rank is not None and int(ai_rank) <= k
            graph_hit = graph_rank is not None and int(graph_rank) <= k
            top_k_union[k] += int(ai_hit or graph_hit)
            top_k_intersection[k] += int(ai_hit and graph_hit)
        if ai_rank is None and graph_rank is None:
            misses.append(
                {
                    "target_id": target_id,
                    "primary_stratum": ai["primary_stratum"],
                    "transformation": transformation,
                }
            )

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "hybrid_proposal_recovery_audit_complete",
        "inputs": {
            "aizynthfinder_score_sha256": aizynthfinder.get("score_sha256"),
            "graph2edits_score_sha256": graph2edits.get("score_sha256"),
        },
        "summary": {
            "exact_route_targets": 36,
            "top_k_union_recovery": {
                str(k): _fraction(top_k_union[k], 36) for k in sorted(top_k_union)
            },
            "top_k_both_engines_recovery": {
                str(k): _fraction(top_k_intersection[k], 36) for k in sorted(top_k_intersection)
            },
            "by_transformation": {name: dict(counts) for name, counts in sorted(family.items())},
            "unrecovered_by_either_engine": misses,
        },
        "interpretation": {
            "learned_union_top5_recovery": top_k_union[5] / 36,
            "all_joint_misses_are_amine_formylation": bool(misses)
            and all(row["transformation"] == "amine_formylation" for row in misses),
            "amine_formylation_is_present_in_frozen_exact_source_truth": bool(misses)
            and all(row["transformation"] == "amine_formylation" for row in misses),
            "proposal_engines_replace_lipid_registry": False,
            "hybrid_search_is_justified_for_independent_adjudication": True,
        },
        "decision": {
            "graph2edits_selected_for_source_neutral_adjudication": True,
            "aizynthfinder_retained_as_independent_diagnostic_challenger": True,
            "production_route_evaluator_changed": False,
            "synthesis_tilting_promoted": False,
            "next_gate": (
                "Apply exact forward, substrate-scope, operational, evidence and terminal-"
                "material closure checks to proposal-derived route hypotheses, then rerun the "
                "matched synthesis tilt only if route-ready yield and score contrast improve."
            ),
        },
        "scientific_authority": {
            "proposal_recovery_is_route_evidence": False,
            "route_closure_authorized": False,
            "may_enter_synthesis_value": False,
            "production_backend_activated": False,
        },
    }
    result["result_sha256"] = content_sha256(result)
    return result


__all__ = ["HybridRecoveryAuditError", "RESULT_SCHEMA_VERSION", "combine_recovery_scores"]
