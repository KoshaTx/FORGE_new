"""Scoring and receipt helpers for frozen Graph2Edits route recovery."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from forge.route.engine.aizynthfinder_single_step_recovery import (
    AiZynthFinderRecoveryError,
    canonical_connected_smiles,
    content_sha256,
)

RESULT_SCHEMA_VERSION = "phase1_graph2edits_single_step_recovery_proposal_result.v1"
SCORE_SCHEMA_VERSION = "phase1_graph2edits_single_step_recovery_score.v1"


def _fraction(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": None if denominator == 0 else numerator / denominator,
    }


def score_graph2edits_recovery(
    *,
    proposal_result: Mapping[str, Any],
    proposal_rows: Iterable[Mapping[str, Any]],
    truth_payload: Mapping[str, Any],
    truth_path: str,
    truth_sha256: str,
) -> dict[str, Any]:
    if proposal_result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise AiZynthFinderRecoveryError("Graph2Edits proposal result schema changed")
    if proposal_result.get("hidden_truth_loaded_during_proposal_execution") is not False:
        raise AiZynthFinderRecoveryError("Graph2Edits proposal execution accessed truth")
    if truth_payload.get("schema_version") != "forge.single_step_benchmark_scoring_truth.v1":
        raise AiZynthFinderRecoveryError("truth schema changed")
    if truth_payload.get("lane_input") is not False:
        raise AiZynthFinderRecoveryError("truth was marked as proposal input")
    rows = list(proposal_rows)
    by_target = {str(row.get("target_id")): row for row in rows}
    if len(by_target) != len(rows) or len(rows) != 120:
        raise AiZynthFinderRecoveryError("Graph2Edits ledger must cover 120 unique targets")
    raw_truth = truth_payload.get("records")
    if not isinstance(raw_truth, list):
        raise AiZynthFinderRecoveryError("truth records are malformed")
    exact = [
        row
        for row in raw_truth
        if isinstance(row, dict)
        and row.get("truth_kind") == "documented_exact_forward_unique_reactant_multiset"
    ]
    adversarial = [
        row
        for row in raw_truth
        if isinstance(row, dict)
        and row.get("truth_kind") == "valid_connected_wrong_handle_role_swap_control"
    ]
    if len(exact) != 36 or len(adversarial) != 12:
        raise AiZynthFinderRecoveryError("frozen truth census changed")
    counts = Counter({1: 0, 5: 0, 10: 0, 20: 0})
    by_stratum: dict[str, Counter[int]] = defaultdict(Counter)
    stratum_denominators: Counter[str] = Counter()
    per_target: list[dict[str, Any]] = []
    for truth in exact:
        target_id = str(truth.get("target_id"))
        row = by_target[target_id]
        expected = tuple(
            sorted(
                canonical_connected_smiles(value, label="truth reactant")
                for value in truth.get("canonical_reactant_multiset", [])
            )
        )
        proposals = row.get("proposals")
        if not isinstance(proposals, list):
            raise AiZynthFinderRecoveryError("Graph2Edits proposal list is malformed")
        first_rank: int | None = None
        for proposal in proposals:
            reactants = tuple(
                sorted(
                    canonical_connected_smiles(value, label="Graph2Edits reactant")
                    for value in proposal.get("canonical_reactants", [])
                )
            )
            if reactants == expected:
                first_rank = int(proposal["rank"])
                break
        stratum = str(row.get("primary_stratum"))
        stratum_denominators[stratum] += 1
        for k in counts:
            recovered = first_rank is not None and first_rank <= k
            counts[k] += int(recovered)
            by_stratum[stratum][k] += int(recovered)
        per_target.append(
            {
                "target_id": target_id,
                "primary_stratum": stratum,
                "transformation": truth.get("transformation"),
                "first_exact_recovery_rank": first_rank,
            }
        )
    score: dict[str, Any] = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "proposal_recovery_scored_after_ledger_freeze",
        "proposal_result_sha256": proposal_result.get("result_sha256"),
        "proposal_ledger_sha256": proposal_result.get("artifacts", {})
        .get("proposal_ledger", {})
        .get("sha256"),
        "scoring_truth": {"path": truth_path, "sha256": truth_sha256},
        "summary": {
            "exact_route_targets": len(exact),
            "adversarial_controls_reserved_for_full_operational_benchmark": len(adversarial),
            "known_route_top_k": {str(k): _fraction(counts[k], len(exact)) for k in sorted(counts)},
            "known_route_top_k_by_primary_stratum": {
                stratum: {
                    str(k): _fraction(values[k], stratum_denominators[stratum])
                    for k in sorted(counts)
                }
                for stratum, values in sorted(by_stratum.items())
            },
        },
        "per_exact_target": sorted(per_target, key=lambda row: row["target_id"]),
        "scientific_authority": {
            "exact_recovery_is_route_evidence": False,
            "route_evidence_created": False,
            "route_closure_authorized": False,
            "may_enter_synthesis_value": False,
            "production_backend_activated": False,
            "synthesis_tilting_activated": False,
        },
    }
    score["score_sha256"] = content_sha256(score)
    return score


__all__ = ["RESULT_SCHEMA_VERSION", "SCORE_SCHEMA_VERSION", "score_graph2edits_recovery"]
