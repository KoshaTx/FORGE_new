"""Freeze the complete partition, carried chemistry coverage and all-family conflict."""

import json
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    paths = {
        "previous_worklist": ROOT
        / "results/phase1/compose_lipid_training_readiness_v6/all-family-worklist.json",
        "partition": ROOT / "results/phase1/compose_lipid_full_partition_v1/audit-v2/result.json",
        "features": ROOT
        / "results/phase1/compose_lipid_full_partition_features_v1/audit/result.json",
        "population_audit": HERE / "population-audit.json",
        "incremental_audit": HERE / "incremental-replay-audit.json",
        "family_coverage_conflict": HERE / "b5-frozen-holdout-conflict.json",
        "validation": ROOT
        / "results/phase1/compose_lipid_training_goal_validation_v7/validation_report.json",
        "historical_component_bindings": ROOT
        / "results/phase1/compose_lipid_split_source_recovery_v2/historical-global-component-exclusions.json",
        "incremental_replay_request": ROOT
        / "results/phase1/compose_lipid_full_replay_v1/michael/request.json",
    }
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    old = data["previous_worklist"]
    partition = data["partition"]
    current = data["incremental_audit"]
    checks = data["validation"]
    if not partition["eligible_view_qualified"] or not partition["all_nonprotected_rows_resolved"]:
        raise ValueError("Complete preparation partition is unresolved")
    if (
        data["population_audit"]["reader_verified_rows"]
        != partition["summary"]["totals"]["eligible_for_program_preparation"]
    ):
        raise ValueError("Protected reader does not cover the current population")
    inputs = {name: pin(ROOT, path) for name, path in paths.items()}
    for name, value in inputs.items():
        resolve_pin(value, ROOT, label=name)
    families = {}
    for family, row in old["families"].items():
        counts = current["summary"]["by_family"][family]
        families[family] = {
            **row,
            "current_partition_audit": partition["summary"]["by_family"][family],
            "eligible_preparation_rows": counts["eligible_preparation_rows"],
            "exact_computed_reconstructions": counts["exact_computed_reconstructions"],
            "historical_exact_now_protected": counts["historical_exact_now_protected"],
            "pending_eligible_reconstructions": counts["eligible_pending_chemistry"],
            "new_exact_computed_reconstructions": counts["new_exact_computed_reconstructions"],
            "expanded_population_work": "Apply qualified input-derived programs to every newly eligible recipe; retain unsupported/ambiguous recipes as explicit pending evidence.",
            "training_ready": False,
        }
        if family == "vitamin_b5_multistep":
            families[family][
                "next_program_work"
            ] = "Frozen shared-core precursor holdout excludes every B5 row; no B5 training admission is possible under the current frozen protection rules."
            families[family][
                "expanded_population_work"
            ] = "Keep existing computed evidence as protected evidence; do not release it into training."
    formal = [f for f, r in families.items() if r["formal_family"]]
    covered = [f for f in formal if families[f]["eligible_preparation_rows"] > 0]
    totals = current["summary"]["totals"]
    dump(
        HERE / "all-family-worklist.json",
        {
            "schema_version": "forge.compose_lipid_all_family_worklist.v7",
            "date": "2026-09-21",
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": inputs,
            "families": families,
            "summary": {
                "source_records": totals["source_records"],
                "preparation_dispositions": partition["summary"]["totals"],
                "eligible_preparation_rows": totals["eligible_preparation_rows"],
                "exact_reconstructions": totals["exact_computed_reconstructions"],
                "prior_exact_now_protected": totals["historical_exact_now_protected"],
                "pending_eligible_reconstructions": totals["eligible_pending_chemistry"],
                "new_exact_reconstructions": totals["new_exact_computed_reconstructions"],
                "formal_families": len(formal),
                "families_with_eligible_rows": len(covered),
                "families_without_eligible_rows": sorted(set(formal) - set(covered)),
                "maximum_source_heavy_atoms": partition["summary"]["maximum_heavy_atoms"],
                "maximum_eligible_heavy_atoms": partition["summary"][
                    "eligible_maximum_heavy_atoms"
                ],
                "eligible_above_96_atoms": partition["summary"]["eligible_above_96_atoms"],
                "eligible_unique_constitutions": partition["summary"][
                    "eligible_unique_constitutions"
                ],
            },
            "split_source_status": {
                **old["split_source_status"],
                "full_universe_partition_qualified": True,
                "full_universe_identity_qualified_records": data["features"]["counts"][
                    "identity_qualified_rows"
                ],
                "unresolved_nonprotected_rows": 0,
                "remaining_partition_work": [],
                "remaining_work": [],
                "source_study_unresolved_but_already_protected": data["features"]["counts"][
                    "source_study_unresolved"
                ],
            },
            "blocked_dependencies": [
                *old["blocked_dependencies"],
                data["family_coverage_conflict"]["conclusion"],
            ],
            "remaining_training_work": [
                "Resolve the all-23-family objective conflict with frozen B5 precursor exclusion without weakening existing gates.",
                "Complete input-derived exact chemistry qualification on the expanded eligible population.",
                "Freeze family/source-balanced weights and validate whole-graph plus program representation on full eligible molecular support.",
                "Build and verify the final admitted training artifact and pass required repository validation.",
            ],
            "incremental_replay_scope": "Three Michael families; all 109718 pending eligible rows evaluated in atomic restartable CPU shards; every completed record independently reconciled to current eligible target and complete source recipe.",
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
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
