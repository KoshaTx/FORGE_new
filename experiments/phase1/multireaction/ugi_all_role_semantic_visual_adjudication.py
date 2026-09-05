"""Unblind and adjudicate a completed all-role Ugi morphology review sheet."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.multireaction.ugi_all_role_semantic_visual_review import (
    BLINDING_KEY_SCHEMA,
    FROZEN_ATTEMPT_INDICES,
    REVIEW_SHEET_SCHEMA,
    UgiAllRoleSemanticVisualReviewError,
    _validate_config,
)
from experiments.phase1.multireaction.ugi_all_role_semantic_visual_review import (
    RESULT_SCHEMA as PACKET_RESULT_SCHEMA,
)
from forge.core.hashing import artifact_record, pin_record
from forge.core.io import read_json_object, write_json

RESULT_SCHEMA = "forge.ugi_all_role_semantic_visual_adjudication.v1"


class UgiAllRoleSemanticVisualAdjudicationError(ValueError):
    """The completed blinded review or its frozen decision contract changed."""


def _read(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(
        path,
        error=UgiAllRoleSemanticVisualAdjudicationError,
        label=label,
    )


def _validate_packet_result(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != PACKET_RESULT_SCHEMA
        or value.get("status") != "ready_for_blinded_review"
        or value.get("quantitative_gate_passed") is not True
        or value.get("fixed_attempt_indices") != list(FROZEN_ATTEMPT_INDICES)
        or value.get("pairs") != len(FROZEN_ATTEMPT_INDICES)
        or value.get("blinded") is not True
        or value.get("candidate_selection") is not False
        or value.get("invalid_or_failed_attempts_retained") is not True
        or value.get("repair_or_retry") is not False
    ):
        raise UgiAllRoleSemanticVisualAdjudicationError(
            "visual-review packet result is inadmissible"
        )


def _validate_key(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = value.get("rows")
    if (
        value.get("schema_version") != BLINDING_KEY_SCHEMA
        or value.get("status") != "sealed_until_review_complete"
        or not isinstance(rows, list)
        or len(rows) != len(FROZEN_ATTEMPT_INDICES)
    ):
        raise UgiAllRoleSemanticVisualAdjudicationError("blinding key is inadmissible")
    indexed: dict[str, dict[str, Any]] = {}
    expected_arms = {"amine_semantic", "all_role_semantic"}
    for rank, (expected_index, raw) in enumerate(
        zip(FROZEN_ATTEMPT_INDICES, rows, strict=True), start=1
    ):
        pair_id = f"P{rank:02d}"
        if (
            not isinstance(raw, Mapping)
            or set(raw)
            != {"pair_id", "attempt_index", "slot_A_arm", "slot_B_arm"}
            or raw.get("pair_id") != pair_id
            or raw.get("attempt_index") != expected_index
            or {raw.get("slot_A_arm"), raw.get("slot_B_arm")} != expected_arms
        ):
            raise UgiAllRoleSemanticVisualAdjudicationError(
                f"blinding key row changed: {pair_id}"
            )
        indexed[pair_id] = dict(raw)
    return indexed


def _validate_sheet(
    value: Mapping[str, Any], decision_rule: Mapping[str, Any]
) -> list[dict[str, Any]]:
    required = {
        "schema_version",
        "status",
        "reviewer_id",
        "allowed_preferences",
        "allowed_pathology_labels",
        "decision_rule",
        "reviews",
    }
    reviewer_id = value.get("reviewer_id")
    rows = value.get("reviews")
    if (
        set(value) != required
        or value.get("schema_version") != REVIEW_SHEET_SCHEMA
        or value.get("status") != "complete"
        or not isinstance(reviewer_id, str)
        or not reviewer_id.strip()
        or value.get("allowed_preferences") != ["A", "B", "tie", "unassessable"]
        or value.get("allowed_pathology_labels")
        != ["A", "B", "both", "neither", "unassessable"]
        or value.get("decision_rule") != decision_rule
        or not isinstance(rows, list)
        or len(rows) != len(FROZEN_ATTEMPT_INDICES)
    ):
        raise UgiAllRoleSemanticVisualAdjudicationError("completed review sheet is inadmissible")
    checked: list[dict[str, Any]] = []
    preference_values = {"A", "B", "tie", "unassessable"}
    pathology_values = {"A", "B", "both", "neither", "unassessable"}
    for rank, raw in enumerate(rows, start=1):
        pair_id = f"P{rank:02d}"
        if (
            not isinstance(raw, Mapping)
            or set(raw)
            != {
                "pair_id",
                "tail_morphology_preference",
                "head_tail_balance_preference",
                "unsupported_ring_or_heteroatom_pathology",
                "reviewer_note",
            }
            or raw.get("pair_id") != pair_id
            or raw.get("tail_morphology_preference") not in preference_values
            or raw.get("head_tail_balance_preference") not in preference_values
            or raw.get("unsupported_ring_or_heteroatom_pathology") not in pathology_values
            or (
                raw.get("reviewer_note") is not None
                and not isinstance(raw.get("reviewer_note"), str)
            )
        ):
            raise UgiAllRoleSemanticVisualAdjudicationError(
                f"completed review row is inadmissible: {pair_id}"
            )
        checked.append(dict(raw))
    return checked


def _preference_summary(
    rows: list[dict[str, Any]],
    key: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = str(row[field])
        if value in {"A", "B"}:
            counts[str(key[str(row["pair_id"])][f"slot_{value}_arm"])] += 1
        else:
            counts[value] += 1
    return {
        "all_role_semantic": counts["all_role_semantic"],
        "amine_semantic": counts["amine_semantic"],
        "tie": counts["tie"],
        "unassessable": counts["unassessable"],
        "assessable": len(rows) - counts["unassessable"],
    }


def _pathology_summary(
    rows: list[dict[str, Any]], key: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = str(row["unsupported_ring_or_heteroatom_pathology"])
        if value in {"A", "B"}:
            counts[str(key[str(row["pair_id"])][f"slot_{value}_arm"])] += 1
        elif value == "both":
            counts["all_role_semantic"] += 1
            counts["amine_semantic"] += 1
            counts["both"] += 1
        else:
            counts[value] += 1
    return {
        "all_role_semantic": counts["all_role_semantic"],
        "amine_semantic": counts["amine_semantic"],
        "both": counts["both"],
        "neither": counts["neither"],
        "unassessable": counts["unassessable"],
        "assessable": len(rows) - counts["unassessable"],
    }


def run_ugi_all_role_semantic_visual_adjudication(
    config_path: Path,
    repo: Path,
    packet_result_path: Path,
    completed_review_sheet_path: Path,
    blinding_key_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Apply the frozen visual gate after every blinded response is complete."""

    try:
        config = _validate_config(
            _read(config_path, "all-role visual-review config")
        )
    except UgiAllRoleSemanticVisualReviewError as error:
        raise UgiAllRoleSemanticVisualAdjudicationError(str(error)) from error
    packet = _read(packet_result_path, "visual-review packet result")
    _validate_packet_result(packet)
    key = _validate_key(_read(blinding_key_path, "visual-review blinding key"))
    sheet_document = _read(completed_review_sheet_path, "completed visual-review sheet")
    rows = _validate_sheet(sheet_document, config["decision_rule"])
    tail = _preference_summary(rows, key, "tail_morphology_preference")
    balance = _preference_summary(rows, key, "head_tail_balance_preference")
    pathology = _pathology_summary(rows, key)
    rule = config["decision_rule"]
    minimum_assessable = int(rule["minimum_assessable_pairs_per_criterion"])
    minimum_net = int(rule["minimum_net_treatment_preferences"])
    checks = {
        "tail_morphology_assessable": tail["assessable"] >= minimum_assessable,
        "tail_morphology_treatment_majority": (
            tail["all_role_semantic"] > tail["amine_semantic"]
            if rule["require_treatment_preference_majority"]
            else True
        ),
        "tail_morphology_minimum_net_preference": (
            tail["all_role_semantic"] - tail["amine_semantic"] >= minimum_net
        ),
        "head_tail_balance_assessable": balance["assessable"] >= minimum_assessable,
        "head_tail_balance_treatment_majority": (
            balance["all_role_semantic"] > balance["amine_semantic"]
            if rule["require_treatment_preference_majority"]
            else True
        ),
        "head_tail_balance_minimum_net_preference": (
            balance["all_role_semantic"] - balance["amine_semantic"] >= minimum_net
        ),
        "pathology_assessable": pathology["assessable"] >= minimum_assessable,
        "treatment_pathologies_bounded": pathology["all_role_semantic"]
        <= int(rule["maximum_treatment_pathology_pairs"]),
        "treatment_pathologies_not_increased": (
            pathology["all_role_semantic"] <= pathology["amine_semantic"]
            if rule["require_treatment_pathologies_no_more_than_baseline"]
            else True
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "reviewer_id": str(sheet_document["reviewer_id"]),
        "pairs": len(rows),
        "summaries": {
            "tail_morphology_preference": tail,
            "head_tail_balance_preference": balance,
            "unsupported_ring_or_heteroatom_pathology": pathology,
        },
        "decision_rule": dict(rule),
        "checks": checks,
        "visual_decision": "pass_seed0_visual_gate" if passed else "do_not_promote",
        "candidate_selection": False,
        "inputs": {
            "config": pin_record(config_path, repo),
            "packet_result": artifact_record(packet_result_path),
            "completed_review_sheet": artifact_record(completed_review_sheet_path),
            "blinding_key": artifact_record(blinding_key_path),
        },
        "nonclaims": [
            "This is a descriptive blinded seed-0 development review, not inferential evidence.",
            "The visual result cannot override a failed quantitative or chemistry-preservation gate.",
        ],
    }
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--packet-result", required=True, type=Path)
    parser.add_argument("--completed-review-sheet", required=True, type=Path)
    parser.add_argument("--blinding-key", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_ugi_all_role_semantic_visual_adjudication(
        args.config.resolve(),
        args.repo.resolve(),
        args.packet_result.resolve(),
        args.completed_review_sheet.resolve(),
        args.blinding_key.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RESULT_SCHEMA",
    "UgiAllRoleSemanticVisualAdjudicationError",
    "run_ugi_all_role_semantic_visual_adjudication",
]
