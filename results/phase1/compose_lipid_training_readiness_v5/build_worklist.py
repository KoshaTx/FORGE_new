"""Checkpoint narrow source-pair qualification and authenticated epoxide input preparation."""

import json
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    previous = ROOT / "results/phase1/compose_lipid_training_readiness_v4/all-family-worklist.json"
    readiness = ROOT / "results/phase1/compose_lipid_readiness_b5_v2/result.json"
    validation = (
        ROOT / "results/phase1/compose_lipid_training_goal_validation_v5/validation_report.json"
    )
    comparison = ROOT / "results/phase1/compose_lipid_b5_routing_v2/replay-comparison.json"
    epoxide = ROOT / "results/phase1/compose_lipid_epoxide_source_v1/component-contract-audit.json"
    old, current, checks = (json.loads(p.read_text()) for p in (previous, readiness, validation))
    families = old["families"]
    for family, row in current["summary"]["by_family"].items():
        families[family].update(row, training_ready=False)
    families["vitamin_b5_multistep"].update(
        next_program_work="1041 reported-tail contract disagreements remain. Preserve original task labels and site/series scope; independently justify any additional computed-variant profile.",
        source_review_status="All 4077 original recipes authenticated. Exact source 11/11 input pairs in the one-tail-knob lane now select the existing I9 homo program without rewriting original metadata. Eight added exact replays; 4069 replay rows unchanged; 3036 exact total. Six independent source controls and all existing scope/site/quantity gates retained.",
        unreviewed_new_pdf_count=0,
    )
    families["amine_epoxide_opening"].update(
        next_program_work="All 3765 eligible original tasks and complete precursor bindings are authenticated; explicit occupancy 1-6 agrees with component quantities and available N-H inventory. Recover and adjudicate Anderson SI exact controls and occupancy/regioselectivity scope. Do not promote structural input checks to reaction qualification.",
        source_review_status="Every eligible task declares Anderson subseries 20080679. The available Han source program belongs to a separate series and does not close the Anderson evidence gap. The 3765 input checks admit zero chemistry or training rows.",
        unreviewed_new_pdf_count=0,
    )
    receipt_paths = {
        "b5_profile_replay_v2": "results/phase1/compose_lipid_supplied_b5_v2/result.json",
        "b5_source_pair_routing": "data/vendor/qualified_b5_source_pair_routing_v2.json",
        "b5_replay_comparison": str(comparison.relative_to(ROOT)),
        "epoxide_original_tasks": "results/phase1/compose_lipid_epoxide_original_tasks_v1/result.json",
        "epoxide_component_contract": str(epoxide.relative_to(ROOT)),
    }
    dump(
        HERE / "all-family-worklist.json",
        {
            **old,
            "schema_version": "forge.compose_lipid_all_family_worklist.v5",
            "date": "2026-09-20",
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "previous_worklist": pin(ROOT, previous),
                "current_readiness": pin(ROOT, readiness),
                "validation": pin(ROOT, validation),
                "b5_increment_audit": pin(ROOT, comparison),
                "epoxide_input_audit": pin(ROOT, epoxide),
            },
            "receipts": {
                **old["receipts"],
                **{k: pin(ROOT, ROOT / p) for k, p in receipt_paths.items()},
            },
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
            "blocked_dependencies": [
                *old["blocked_dependencies"],
                "Anderson DOI 10.1073/pnas.0910603106 SI exact controls and applicable occupancy/site evidence remain unavailable through attempted primary-source routes.",
            ],
            "training_ready": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
