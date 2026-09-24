"""Account for every family and preserve open gates at the Han/B5 checkpoint."""

import json
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    previous = ROOT / "results/phase1/compose_lipid_training_readiness_v2/all-family-worklist.json"
    readiness = ROOT / "results/phase1/compose_lipid_readiness_han_db_v1/result.json"
    validation = (
        ROOT / "results/phase1/compose_lipid_training_goal_validation_v3/validation_report.json"
    )
    old = json.loads(previous.read_text())
    current = json.loads(readiness.read_text())
    checks = json.loads(validation.read_text())
    families = old["families"]
    for name, row in current["summary"]["by_family"].items():
        families[name].update(row)
        families[name]["training_ready"] = False
    families["epoxide_opening_o_acylation"].update(
        next_program_work="Current eligible population completely replayed under documented grouped source stages; final full-universe partition and training assembly remain pending.",
        source_review_status="Primary article, source SI drawing and two exact intermediate/final source controls reviewed. Acyl chlorides, incorporated quantity two and complete event paths preserved.",
    )
    families["vitamin_b5_multistep"].update(
        next_program_work="Close the three explicit draft complete-precursor scope gaps, qualify source metadata-to-program bindings, and replay all 4077 eligible rows without pooling I7/I8/I9 or head ester/amide variants.",
        source_review_status="Primary SI reviewed; six source controls and 16 mechanical checks pass. Three deliberate scope counterexamples remain, source discrepancies retained; draft is not corpus-qualified.",
    )
    receipts = {
        **old["receipts"],
        **{
            name: pin(ROOT, ROOT / "results/phase1" / path)
            for name, path in {
                "han_db_replay": "compose_lipid_supplied_han_db_v1/result.json",
                "han_db_adjudication": "compose_lipid_han_db_source_v1/adjudication.json",
                "b5_control_transcriptions": "compose_lipid_b5_source_v1/control-transcriptions.json",
                "b5_draft_replay": "compose_lipid_b5_source_v1/draft-control-replay.json",
                "b5_draft_scope_gaps": "compose_lipid_b5_source_v1/draft-adversarial-audit.json",
                "b5_source_bindings": "compose_lipid_b5_source_v1/eligible-bindings.json",
                "b5_scope_proposal": "compose_lipid_b5_source_v1/complete-terminal-scope-proposal.json",
            }.items()
        },
    }
    dump(
        HERE / "all-family-worklist.json",
        {
            **old,
            "schema_version": "forge.compose_lipid_all_family_worklist.v3",
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
                for k in [
                    "focused",
                    "full",
                    "new_failures",
                    "new_errors",
                    "vendor_verify_exit_code",
                    "source_snapshot_unchanged",
                ]
            },
            "training_ready": False,
            "training_calls": 0,
            "independent_work_remaining": [
                "Complete B5 terminal scope and original source role/program binding; preserve all negative source-conflict and draft-scope receipts.",
                "Remaining AEMA, amine alkylation, amine epoxide and ketone acyclic source programs.",
                "Pending attachment, charge and precursor-support adjudication in already replayed families.",
                "Final eligible-population representation, constitutional deduplication and source-balanced training assembly after corpus qualification.",
            ],
        },
    )


if __name__ == "__main__":
    main()
