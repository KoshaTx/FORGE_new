"""Update full-family readiness accounting without admitting unresolved records."""

import json
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    old_path = ROOT / "results/phase1/compose_lipid_training_readiness_v1/all-family-worklist.json"
    readiness = ROOT / "results/phase1/compose_lipid_readiness_miao_thiol_yne_v1/result.json"
    validation = (
        ROOT / "results/phase1/compose_lipid_training_goal_validation_v2/validation_report.json"
    )
    old = json.loads(old_path.read_text())
    current = json.loads(readiness.read_text())
    checks = json.loads(validation.read_text())
    families = old["families"]
    steps = {
        "ketone_ugi4": "Complete current eligible replay; retain distinct source head-N proof and the separately pinned exact LNPDB role vocabulary profile. Final full-universe split and training assembly still pending.",
        "iphos_ring_opening": "Resolve 581 pending rows: 315 multiple unfiltered attachment outcomes, 148 unrepresented source connectivity/charge forms, 118 unsupported per-N event counts. Preserve permanent quaternary N versus removable ammonium H; do not neutralize target structures.",
        "alpha_isocyanoester_dihydroimidazole": "Complete current eligible computed replay for the three explicitly drawn Iso4/5/6 source precursors. Preserve the initial 0/1469 Iso5-only negative result and distinguish library scope from isolated Iso5 characterization.",
        "preassembled_thiol_yne_tail_amidation": "Resolve 906 oxygen-containing thiol precursor scope exclusions using transferable primary-source support before broader admission. Preserve the two copies of the same global thiol identity and the two documented stages.",
        "epoxide_opening_o_acylation": "Two independent full source controls now transcribed and inventory/mass checked. Implement grouped epoxide-addition stage followed by grouped O-acylation, preserving identical copies and source acyl chlorides. Do not infer intermediate selectivity or use acid substitutes.",
    }
    for name, row in current["summary"]["by_family"].items():
        families[name].update(row)
        if name in steps:
            families[name]["next_program_work"] = steps[name]
            families[name].pop("unreviewed_new_pdf_count", None)
            families[name][
                "source_review_status"
            ] = "Program-specific primary sources reviewed; see pinned adjudications and control transcriptions. This does not claim every bibliography item has been reviewed."
        families[name]["training_ready"] = False
    receipts = {
        "ketone_ugi4": "compose_lipid_supplied_ketone_ugi4_profiles_v1/result.json",
        "iphos_neutral": "compose_lipid_supplied_iphos_neutral_proton_transfer_v1/result.json",
        "iphos_zwitterion": "compose_lipid_supplied_iphos_zwitterion_v1/result.json",
        "iphos_anionic": "compose_lipid_supplied_iphos_deprotonated_v1/result.json",
        "miao_initial_negative": "compose_lipid_supplied_miao_cyclic_v1/result.json",
        "miao_drawn_scope": "compose_lipid_supplied_miao_cyclic_v2/result.json",
        "thiol_yne": "compose_lipid_supplied_thiol_yne_v1/result.json",
        "next_han_db_controls": "compose_lipid_han_db_source_v1/control-transcriptions.json",
        "missing_split_inputs": "compose_lipid_training_readiness_v1/required-split-inputs.json",
        "historical_result_recovery": "compose_lipid_training_goal_validation_v1/historical-result-recovery-audit.json",
    }
    dump(
        HERE / "all-family-worklist.json",
        {
            "schema_version": "forge.compose_lipid_all_family_worklist.v2",
            "date": "2026-09-20",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "previous_worklist": pin(ROOT, old_path),
                "current_readiness": pin(ROOT, readiness),
                "validation": pin(ROOT, validation),
                "source_scope_audit": pin(ROOT, HERE / "source-scope-audit.json"),
            },
            "receipts": {
                name: pin(ROOT, ROOT / "results/phase1" / path) for name, path in receipts.items()
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
                "focused": checks["focused"],
                "full": checks["full"],
                "new_failures": checks["new_failures"],
                "new_errors": checks["new_errors"],
                "vendor_verify_exit_code": checks["vendor_verify_exit_code"],
                "source_snapshot_unchanged": checks["source_snapshot_unchanged"],
            },
            "training_ready": False,
            "training_calls": 0,
            "blocked_dependencies": [
                "Original three full-universe split-producer/configuration files are still unavailable; unassigned records stay unassigned.",
                "Original acid/epoxide SI DOI 10.1002/adhm.202302691 remains unavailable through attempted publisher routes.",
                "Frozen historical result artifacts referenced by repository tests are still unavailable; archive path requested.",
            ],
            "independent_work_remaining": [
                "Grouped source programs for Han DB and AEMA.",
                "Remaining amine alkylation, amine epoxide, ketone acyclic and vitamin B5 source programs.",
                "Remaining role/site/charge and precursor-support adjudication.",
                "Final eligible-population representation, constitutional deduplication and source-balanced training assembly after corpus qualification.",
            ],
        },
    )


if __name__ == "__main__":
    main()
