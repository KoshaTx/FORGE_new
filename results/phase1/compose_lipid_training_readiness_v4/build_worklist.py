"""Preserve all-family coverage and open gates after protected B5 source-profile replay."""

import json
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    previous = ROOT / "results/phase1/compose_lipid_training_readiness_v3/all-family-worklist.json"
    readiness = ROOT / "results/phase1/compose_lipid_readiness_b5_v1/result.json"
    validation = (
        ROOT / "results/phase1/compose_lipid_training_goal_validation_v4/validation_report.json"
    )
    old = json.loads(previous.read_text())
    current = json.loads(readiness.read_text())
    checks = json.loads(validation.read_text())
    families = old["families"]
    for name, row in current["summary"]["by_family"].items():
        families[name].update(row)
        families[name]["training_ready"] = False
    families["vitamin_b5_multistep"].update(
        next_program_work="Adjudicate 1049 records with reported-tail identities outside the current series/site-specific contract. Preserve the original tasks and distinguish global catalogue labels from site-specific experimental claims before any separate computed-variant qualification.",
        source_review_status="Six independent source controls pass; all three earlier complete-precursor scope counterexamples are rejected. All 4077 eligible original task records authenticated; 3028 pass complete input profiles and every staged exactness gate. I7/I8/I9, source R stereochemistry, head ester/amide, original component IDs/roles/quantities and source discrepancies remain distinct.",
    )
    receipts = {
        **old["receipts"],
        **{
            name: pin(ROOT, ROOT / path)
            for name, path in {
                "b5_original_tasks": "results/phase1/compose_lipid_b5_original_tasks_v1/result.json",
                "b5_profile_replay": "results/phase1/compose_lipid_supplied_b5_v1/result.json",
                "b5_scope_registry": "data/vendor/qualified_b5_staged_source_program_v1.json",
                "b5_source_scope_audit": "results/phase1/compose_lipid_training_readiness_v4/source-scope-audit.json",
            }.items()
        },
    }
    dump(
        HERE / "all-family-worklist.json",
        {
            **old,
            "schema_version": "forge.compose_lipid_all_family_worklist.v4",
            "date": "2026-09-20",
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "previous_worklist": pin(ROOT, previous),
                "current_readiness": pin(ROOT, readiness),
                "validation": pin(ROOT, validation),
                "source_scope_audit": pin(ROOT, HERE / "source-scope-audit.json"),
            },
            "receipts": receipts,
            "summary": {
                **current["summary"],
                "families_with_exact_computed_evidence": sum(
                    r["exact_computed_reconstructions"] > 0 for r in families.values()
                ),
                "pending_eligible_reconstructions": sum(
                    r["pending_eligible_reconstructions"] for r in families.values()
                ),
            },
            "families": families,
            "remaining_gates": current["remaining_gates"],
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
            "independent_work_remaining": [
                "Adjudicate the B5 reported-tail interpretation and any separately justified computed-variant scope; do not overwrite original task labels or promote source execution.",
                "Remaining AEMA, amine alkylation, amine epoxide and ketone acyclic source programs.",
                "Pending attachment, charge and precursor-support adjudication in already replayed families.",
                "Final full-universe partition, eligible-population representation, constitutional deduplication and source-balanced training assembly after qualification.",
            ],
        },
    )


if __name__ == "__main__":
    main()
