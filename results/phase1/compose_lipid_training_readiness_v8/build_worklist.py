"""Publish reconciled bulk chemistry and full product representation coverage."""

import json
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    paths = {
        "previous_worklist": ROOT
        / "results/phase1/compose_lipid_training_readiness_v7/all-family-worklist.json",
        "combined_audit": HERE / "combined-replay-audit.json",
        "representation_audit": HERE / "representation-audit.json",
        "miao_acyclic_source_review": HERE / "miao-acyclic-source-review.json",
        "bulk_replay": ROOT
        / "results/phase1/compose_lipid_full_program_replay_v1/programs/result.json",
        "iphos_replay": ROOT
        / "results/phase1/compose_lipid_full_iphos_replay_v1/anionic/result.json",
    }
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    old = data["previous_worklist"]
    current = data["combined_audit"]
    representation = data["representation_audit"]
    if (
        not representation["pass"]
        or representation["verified_eligible_rows"] != old["summary"]["eligible_preparation_rows"]
    ):
        raise ValueError("Product representation does not cover the complete eligible partition")
    inputs = {name: pin(ROOT, path) for name, path in paths.items()}
    for name, value in inputs.items():
        resolve_pin(value, ROOT, label=name)
    replay_counts = {}
    for key in ("bulk_replay", "iphos_replay"):
        if data[key]["complete"] is not True:
            raise ValueError("Replay is incomplete")
        for family, counts in data[key]["summary"].items():
            if family in replay_counts:
                raise ValueError("Two new replay batches overlap a family")
            replay_counts[family] = counts
    families = {}
    for family, prior in old["families"].items():
        counts = current["summary"]["by_family"][family]
        exact = counts["exact_computed_reconstructions"]
        pending = counts["eligible_pending_chemistry"]
        families[family] = {
            **prior,
            "exact_computed_reconstructions": exact,
            "pending_eligible_reconstructions": pending,
            "new_exact_computed_reconstructions": exact - prior["exact_computed_reconstructions"],
            "bulk_replay": replay_counts.get(family),
            "product_representation_qualified": counts["eligible_preparation_rows"] > 0,
            "joint_program_representation_qualified": False,
            "training_ready": False,
        }
        if family in replay_counts:
            families[family]["expanded_population_work"] = (
                "All pending eligible recipes evaluated with the declared qualified program; "
                "unresolved outputs retain zero training admission."
            )
            families[family]["next_program_work"] = (
                "Complete joint program representation, final admission and balanced weights."
                if not pending
                else "Adjudicate remaining failed checks using input-derived source evidence; "
                "do not choose sites or charge variants using the target."
            )
        if family == "ketone_isocyanide_amide":
            families[family]["unreviewed_new_pdf_count"] = 0
            families[family]["source_review_status"] = data["miao_acyclic_source_review"][
                "conclusion"
            ]
            families[family]["next_program_work"] = (
                "Obtain an independent acyclic product control and applicable branch-selection "
                "evidence; qualify the registry transform and explicit oxygen accounting."
            )
    totals = current["summary"]["totals"]
    if (
        sum(r["exact_computed_reconstructions"] for r in families.values())
        != totals["exact_computed_reconstructions"]
    ):
        raise ValueError("Family evidence totals disagree")
    if (
        totals["exact_computed_reconstructions"] + totals["eligible_pending_chemistry"]
        != old["summary"]["eligible_preparation_rows"]
    ):
        raise ValueError("Evidence accounting changed the eligible population")
    new_exact = totals["exact_computed_reconstructions"] - old["summary"]["exact_reconstructions"]
    replayed = sum(row["rows"] for row in replay_counts.values())
    dump(
        HERE / "all-family-worklist.json",
        {
            **old,
            "schema_version": "forge.compose_lipid_all_family_worklist.v8",
            "date": "2026-09-21",
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": inputs,
            "families": families,
            "summary": {
                **old["summary"],
                "exact_reconstructions": totals["exact_computed_reconstructions"],
                "pending_eligible_reconstructions": totals["eligible_pending_chemistry"],
                "new_exact_reconstructions": new_exact,
                "new_replayed_recipes": replayed,
                "families_with_exact_evidence": sum(
                    r["formal_family"] and r["exact_computed_reconstructions"] > 0
                    for r in families.values()
                ),
                "families_with_complete_eligible_replay": sum(
                    r["formal_family"]
                    and r["eligible_preparation_rows"] > 0
                    and r["pending_eligible_reconstructions"] == 0
                    for r in families.values()
                ),
                "product_representation_verified_rows": representation["verified_eligible_rows"],
            },
            "representation": {
                "eligible_product_graphs_pass": True,
                "maximum_observed": representation["maximum_observed"],
                "model_smoke_support_atoms": representation["model_smoke_support_atoms"],
                "final_admitted_training_vocabulary_refit_required": True,
                "joint_synthesis_program_representation_qualified": False,
            },
            "incremental_replay_scope": (
                "Thirteen existing qualified family programs plus one fixed anionic iPhos "
                "program; complete eligible pending populations, atomic restartable CPU shards, "
                "independent row-by-row reconciliation to the protected partition and recipes."
            ),
            "remaining_training_work": [
                "Resolve the all-23-family conflict with the frozen B5 precursor exclusion.",
                "Qualify unsupported families and adjudicate explicit failed recipe checks.",
                "Qualify joint synthesis-program representation and refit vocabulary on final admission.",
                "Freeze family/source-balanced weights and the final admitted training artifact.",
                "Restore missing historical test artifacts and pass repository validation.",
            ],
            "validation_reused_from_previous_checkpoint": True,
            "training_ready": False,
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
