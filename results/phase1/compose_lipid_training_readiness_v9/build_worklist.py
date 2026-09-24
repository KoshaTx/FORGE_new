"""Advance only independently reconciled source-mechanism replay evidence."""

import json
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    previous_path = (
        ROOT / "results/phase1/compose_lipid_training_readiness_v8/all-family-worklist.json"
    )
    current_path = HERE / "combined-replay-audit.json"
    previous = json.loads(previous_path.read_text())
    current = json.loads(current_path.read_text())
    families = {}
    for family, row in previous["families"].items():
        counts = current["summary"]["by_family"][family]
        added = counts["exact_computed_reconstructions"] - row["exact_computed_reconstructions"]
        if added != (current["comparison"]["new_exact"] if family == "maleate_addition" else 0):
            raise ValueError("Source mechanism replay changed another family's evidence")
        families[family] = {
            **row,
            "exact_computed_reconstructions": counts["exact_computed_reconstructions"],
            "pending_eligible_reconstructions": counts["eligible_pending_chemistry"],
            "new_exact_computed_reconstructions": added,
        }
        if family == "maleate_addition":
            families[family][
                "next_program_work"
            ] = "Complete joint program representation, final admission and balanced weights."
            families[family]["source_mechanism_replay"] = {
                "selection": "original_primary_metadata.mechanism_only",
                "carried_exact_reconstructions": current["comparison"][
                    "carried_exact_reconstructions"
                ],
                "new_exact_reconstructions": added,
                "source_roles_quantities_and_ids_modified": False,
            }
    totals = current["summary"]["totals"]
    if (
        totals["exact_computed_reconstructions"] + totals["eligible_pending_chemistry"]
        != previous["summary"]["eligible_preparation_rows"]
    ):
        raise ValueError("Updated chemistry evidence changed the eligible population")
    dump(
        HERE / "all-family-worklist.json",
        {
            **previous,
            "schema_version": "forge.compose_lipid_all_family_worklist.v9",
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "previous_worklist": pin(ROOT, previous_path),
                "combined_audit": pin(ROOT, current_path),
                "source_inventory_conflicts": pin(ROOT, HERE / "aryl-inventory-audit.json"),
                "fresh_validation": pin(ROOT, HERE / "validation-report.json"),
                "source_access_followup": pin(
                    ROOT,
                    ROOT
                    / "results/phase1/compose_lipid_acid_epoxide_source_v2/access-followup.json",
                ),
            },
            "families": families,
            "summary": {
                **previous["summary"],
                "exact_reconstructions": totals["exact_computed_reconstructions"],
                "pending_eligible_reconstructions": totals["eligible_pending_chemistry"],
                "new_exact_reconstructions": current["comparison"]["new_exact"],
                "new_replayed_recipes": previous["families"]["maleate_addition"][
                    "pending_eligible_reconstructions"
                ],
                "families_with_complete_eligible_replay": sum(
                    row["formal_family"]
                    and row["eligible_preparation_rows"] > 0
                    and row["pending_eligible_reconstructions"] == 0
                    for row in families.values()
                ),
            },
            "incremental_replay_scope": "All pending eligible maleate recipes; original source mechanism chooses the existing qualified amine or thiol program before target access. Prior exact evidence is carried unchanged.",
            "training_ready": False,
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
