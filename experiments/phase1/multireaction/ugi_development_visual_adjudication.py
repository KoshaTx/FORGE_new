"""Unblind and adjudicate a completed Ugi development morphology review."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.multireaction.ugi_development_visual_review import (
    LEGACY_PREFERENCE_CRITERIA,
    PATHOLOGY_CRITERION,
    SUPPORTED_CRITERIA,
)
from forge.core.hashing import artifact_record
from forge.core.io import read_json_object, write_json

RESULT_SCHEMA = "forge.ugi_development_visual_adjudication.v1"


class UgiDevelopmentVisualAdjudicationError(ValueError):
    """The blinded packet, completed sheet, or blinding key is inadmissible."""


def _read(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(path, error=UgiDevelopmentVisualAdjudicationError, label=label)


def run_ugi_development_visual_adjudication(
    packet_result_path: Path,
    completed_review_sheet_path: Path,
    blinding_key_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Apply the frozen visual gate after every blinded field is complete."""

    packet = _read(packet_result_path, "development visual-review packet result")
    sheet = _read(completed_review_sheet_path, "completed development visual-review sheet")
    key = _read(blinding_key_path, "development visual-review blinding key")
    if (
        packet.get("schema_version") != "forge.ugi_development_visual_review.v1"
        or packet.get("status") != "ready_for_blinded_review"
        or packet.get("blinded") is not True
        or packet.get("candidate_selection") is not False
        or packet.get("repair_or_retry") is not False
    ):
        raise UgiDevelopmentVisualAdjudicationError("visual-review packet is inadmissible")
    if (
        sheet.get("schema_version") != "forge.ugi_development_blinded_review_sheet.v1"
        or sheet.get("status") != "complete"
        or not sheet.get("reviewer_id")
        or key.get("schema_version") != "forge.ugi_development_blinding_key.v1"
        or key.get("status") != "sealed_until_review_complete"
    ):
        raise UgiDevelopmentVisualAdjudicationError("completed review or blinding key changed")
    reviews = sheet.get("reviews")
    key_rows = key.get("rows")
    if (
        not isinstance(reviews, list)
        or not isinstance(key_rows, list)
        or len(reviews) != packet.get("pairs")
        or len(key_rows) != len(reviews)
    ):
        raise UgiDevelopmentVisualAdjudicationError("review pair count changed")
    keyed: dict[str, Mapping[str, Any]] = {}
    arms: set[str] = set()
    for row in key_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "pair_id",
            "attempt_index",
            "slot_A_arm",
            "slot_B_arm",
        }:
            raise UgiDevelopmentVisualAdjudicationError("blinding-key row changed")
        pair_id = str(row["pair_id"])
        if pair_id in keyed or row["slot_A_arm"] == row["slot_B_arm"]:
            raise UgiDevelopmentVisualAdjudicationError("blinding-key arms are invalid")
        keyed[pair_id] = row
        arms.update((str(row["slot_A_arm"]), str(row["slot_B_arm"])))
    if len(arms) != 2:
        raise UgiDevelopmentVisualAdjudicationError("blinding key does not contain two arms")
    packet_baseline = packet.get("baseline_arm")
    packet_treatment = packet.get("treatment_arm")
    if packet_baseline is None and packet_treatment is None:
        # Backward compatibility for already frozen v1 packets.  New packets always record the
        # exact arm identities above, so arbitrary successor names remain scientifically valid.
        treatment = next((arm for arm in arms if "local_chemistry" in arm), None)
        if treatment is None:
            raise UgiDevelopmentVisualAdjudicationError("treatment arm cannot be identified")
        baseline = next(arm for arm in arms if arm != treatment)
    else:
        if (
            not isinstance(packet_baseline, str)
            or not packet_baseline
            or not isinstance(packet_treatment, str)
            or not packet_treatment
            or packet_baseline == packet_treatment
            or {packet_baseline, packet_treatment} != arms
        ):
            raise UgiDevelopmentVisualAdjudicationError(
                "packet arm identities disagree with the blinding key"
            )
        baseline = packet_baseline
        treatment = packet_treatment
    raw_preference_criteria = packet.get("preference_criteria", LEGACY_PREFERENCE_CRITERIA)
    pathology_criterion = packet.get("pathology_criterion", PATHOLOGY_CRITERION)
    if (
        not isinstance(raw_preference_criteria, (list, tuple))
        or any(not isinstance(value, str) for value in raw_preference_criteria)
        or not isinstance(pathology_criterion, str)
        or tuple((*raw_preference_criteria, pathology_criterion)) not in SUPPORTED_CRITERIA
    ):
        raise UgiDevelopmentVisualAdjudicationError("visual-review criteria are unsupported")
    preference_criteria = tuple(raw_preference_criteria)
    sheet_preference_criteria = sheet.get("preference_criteria", list(preference_criteria))
    sheet_pathology_criterion = sheet.get("pathology_criterion", pathology_criterion)
    if (
        sheet_preference_criteria != list(preference_criteria)
        or sheet_pathology_criterion != pathology_criterion
    ):
        raise UgiDevelopmentVisualAdjudicationError(
            "completed review criteria disagree with the packet"
        )
    preference_counts = {
        criterion: Counter({baseline: 0, treatment: 0, "tie": 0, "unassessable": 0})
        for criterion in preference_criteria
    }
    pathology_counts = Counter(
        {baseline: 0, treatment: 0, "both": 0, "neither": 0, "unassessable": 0}
    )
    for review in reviews:
        expected_review_fields = {
            "pair_id",
            *preference_criteria,
            pathology_criterion,
            "reviewer_note",
        }
        if not isinstance(review, Mapping) or set(review) != expected_review_fields:
            raise UgiDevelopmentVisualAdjudicationError("completed review row changed")
        pair_id = str(review["pair_id"])
        row_key = keyed.get(pair_id)
        if row_key is None:
            raise UgiDevelopmentVisualAdjudicationError("review pair is absent from the key")
        for criterion, counts in preference_counts.items():
            value = review[criterion]
            if value in {"tie", "unassessable"}:
                counts[str(value)] += 1
            elif value in {"A", "B"}:
                counts[str(row_key[f"slot_{value}_arm"])] += 1
            else:
                raise UgiDevelopmentVisualAdjudicationError("review preference is invalid")
        pathology = review[pathology_criterion]
        if pathology in {"both", "neither", "unassessable"}:
            pathology_counts[str(pathology)] += 1
            if pathology == "both":
                pathology_counts[baseline] += 1
                pathology_counts[treatment] += 1
        elif pathology in {"A", "B"}:
            pathology_counts[str(row_key[f"slot_{pathology}_arm"])] += 1
        else:
            raise UgiDevelopmentVisualAdjudicationError("review pathology label is invalid")
    rule = sheet.get("decision_rule")
    if not isinstance(rule, Mapping):
        raise UgiDevelopmentVisualAdjudicationError("visual decision rule is absent")
    criterion_gates = {}
    for criterion, counts in preference_counts.items():
        assessable = len(reviews) - counts["unassessable"]
        criterion_gates[criterion] = {
            "assessable_pairs": assessable,
            "counts": dict(counts),
            "minimum_assessable_pairs_met": assessable
            >= int(rule["minimum_assessable_pairs_per_criterion"]),
            "treatment_preference_majority": counts[treatment] > counts[baseline],
            "minimum_net_treatment_preferences_met": (
                counts[treatment] - counts[baseline]
                >= int(rule["minimum_net_treatment_preferences"])
            ),
        }
    pathology_gate = {
        "counts": dict(pathology_counts),
        "treatment_within_absolute_maximum": pathology_counts[treatment]
        <= int(rule["maximum_treatment_pathology_pairs"]),
        "treatment_no_more_than_baseline": pathology_counts[treatment]
        <= pathology_counts[baseline],
    }
    passed = all(
        row["minimum_assessable_pairs_met"]
        and row["treatment_preference_majority"]
        and row["minimum_net_treatment_preferences_met"]
        for row in criterion_gates.values()
    ) and all(value for key_name, value in pathology_gate.items() if key_name != "counts")
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "baseline_arm": baseline,
        "treatment_arm": treatment,
        "pairs": len(reviews),
        "preference_criteria": list(preference_criteria),
        "pathology_criterion": pathology_criterion,
        "criterion_gates": criterion_gates,
        "pathology_gate": pathology_gate,
        "visual_decision": "pass_seed0_visual_gate" if passed else "fail_seed0_visual_gate",
        "promotion_decision": "eligible_for_full_comparison" if passed else "do_not_promote",
        "inputs": {
            "packet_result": artifact_record(packet_result_path),
            "completed_review_sheet": artifact_record(completed_review_sheet_path),
            "blinding_key": artifact_record(blinding_key_path),
        },
        "candidate_selection": False,
        "repair_or_retry": False,
    }
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-result", type=Path, required=True)
    parser.add_argument("--completed-review-sheet", type=Path, required=True)
    parser.add_argument("--blinding-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_ugi_development_visual_adjudication(
        args.packet_result.resolve(),
        args.completed_review_sheet.resolve(),
        args.blinding_key.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
