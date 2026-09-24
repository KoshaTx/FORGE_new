"""Checkpoint recovered split contracts and the full corrected protection overlay."""

import json
from collections import Counter
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    names = {
        "previous_worklist": "results/phase1/compose_lipid_training_readiness_v5/all-family-worklist.json",
        "current_readiness": "results/phase1/compose_lipid_readiness_b5_v2/result.json",
        "validation": "results/phase1/compose_lipid_training_goal_validation_v6/validation_report.json",
        "source_recovery": "results/phase1/compose_lipid_split_source_recovery_v2/result.json",
        "corrected_selected_replay": "results/phase1/compose_lipid_split_source_recovery_v2/frozen-group-replay.json",
        "historical_morphology_replay": "results/phase1/compose_lipid_split_source_recovery_v2/historical-morphology-replay.json",
        "historical_frozen_groups": "results/phase1/compose_lipid_split_source_recovery_v2/historical-frozen-group-recovery.json",
        "full_corrected_projection": "results/phase1/compose_lipid_full_split_projection_v1/audit/result.json",
    }
    paths = {k: ROOT / v for k, v in names.items()}
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    old, current, checks = (
        data[k] for k in ("previous_worklist", "current_readiness", "validation")
    )
    projection = data["full_corrected_projection"]
    for result in (data["corrected_selected_replay"], data["historical_morphology_replay"]):
        if result["pass"] is not True or result["failures"]:
            raise ValueError("A source split replay did not pass")
    if (
        data["historical_frozen_groups"]["pass"] is not True
        or data["historical_frozen_groups"]["assignment_failures"]
        or data["historical_frozen_groups"]["panel_failures"]
    ):
        raise ValueError("Historical frozen group recovery did not pass")
    if data["source_recovery"]["missing_required_split_files"]:
        raise ValueError("Required split source recovery is incomplete")
    effective = Counter()
    crossings = projection["summary"]["disposition_crossings"]
    for row in crossings:
        effective[row["disposition"]] += row["rows"]
        if row["prior_disposition"] == "protected" and row["disposition"] != "protected":
            raise ValueError("Projection removed an existing holdout")
        if (
            row["prior_disposition"] == "eligible_for_program_preparation"
            and row["disposition"] != "eligible_for_program_preparation"
        ):
            raise ValueError("Projection changed the previously qualified evidence population")
    if sum(effective.values()) != current["summary"]["source_records"]:
        raise ValueError("Projection does not account for the complete source population")
    families = old["families"]
    for family, row in families.items():
        row["new_exact_computed_reconstructions"] = 0
        row["current_partition_audit"] = projection["summary"]["by_family"][family]
    receipts = dict(old["receipts"])
    receipts["original_split_input_request_resolved"] = receipts.pop("missing_split_inputs")
    receipts.update({k: pin(ROOT, p) for k, p in paths.items() if k != "previous_worklist"})
    inputs = {k: pin(ROOT, p) for k, p in paths.items()}
    for k, value in inputs.items():
        resolve_pin(value, ROOT, label=k)
    dump(
        HERE / "all-family-worklist.json",
        {
            **old,
            "schema_version": "forge.compose_lipid_all_family_worklist.v6",
            "date": "2026-09-21",
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": inputs,
            "receipts": receipts,
            "families": families,
            "summary": {
                **old["summary"],
                "new_exact_reconstructions": 0,
                "by_family": {
                    family: {**row, "new_exact_computed_reconstructions": 0}
                    for family, row in old["summary"]["by_family"].items()
                },
                "preparation_dispositions": dict(effective),
                "prior_preparation_dispositions": old["summary"]["preparation_dispositions"],
                "newly_protected_by_corrected_group_extension": projection["summary"]["totals"][
                    "newly_protected"
                ],
            },
            "split_source_status": {
                "required_source_files_recovered": 3,
                "missing_required_source_files": [],
                "corrected_selected_rows_reproduced": data["corrected_selected_replay"]["counts"][
                    "all_checks_pass"
                ],
                "historical_morphology_rows_reproduced": data["historical_morphology_replay"][
                    "counts"
                ]["all_keys_exact"],
                "historical_assignments_and_panels_reproduced": data["historical_frozen_groups"][
                    "source_rows_reproduced"
                ],
                "full_corrected_projection_rows": projection["summary"]["totals"]["rows"],
                "full_universe_partition_qualified": False,
                "remaining_work": [
                    "Apply the exactly recovered historical frozen groups across the full corpus.",
                    "Establish constitutional product overlap across the complete universe and prior protected folds.",
                    "Resolve source-study identity before releasing unassigned records for program preparation.",
                ],
            },
            "blocked_dependencies": [
                item
                for item in old["blocked_dependencies"]
                if not item.startswith("Original three full-universe split-producer")
            ],
            "validation": {
                k: checks[k]
                for k in (
                    "focused",
                    "full",
                    "new_failures",
                    "new_errors",
                    "vendor_verify_exit_code",
                    "source_snapshot_unchanged",
                )
            },
            "training_ready": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
