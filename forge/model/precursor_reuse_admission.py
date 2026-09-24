"""Admit exact completions without losing TRAIN novelty or concentrating product counts.

This is explicit constrained selection over a bounded proposal ledger. It is not evidence that
the frozen neural model learned the constraints. The complete original population is retained.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from forge.model.precursor_reuse import PrecursorReuseError


def admit_completions(
    originals: list[dict], completions: list[dict], train_products: set[str]
) -> list[dict]:
    """Visit attempts in saved order, admitting the least-edited feasible generated proposal.

    Moving one occurrence from multiplicity a to b changes the squared-count sum by
    2*(b-a+1). Requiring b<a prevents its increase, so inverse-Simpson diversity cannot
    fall at fixed valid count. It also prevents loss of a distinct graph. Existing exact
    outputs are immutable, so their unique and novel subsets cannot be removed.
    """
    if len(originals) != len(completions):
        raise PrecursorReuseError("admission populations differ")
    inventory = defaultdict(Counter)
    for row in originals:
        if row["valid_connected"]:
            inventory[row["program_id"]][row["canonical_smiles"]] += 1
    decisions = []
    for original, completion in zip(originals, completions, strict=True):
        smiles = original["canonical_smiles"]
        counts = inventory[original["program_id"]]
        decision = {
            "selected_smiles": smiles,
            "selected_donor": None,
            "disposition": "retained_original",
            "proposal_checks": [],
        }
        if not original["valid_connected"]:
            decision["disposition"] = "original_invalid"
        elif original["assembly"]["status"] == "exact_computed_program":
            decision["disposition"] = "original_exact"
        else:
            eligible = []
            for proposal in completion["proposals"]:
                if proposal["status"] != "exact_computed_program":
                    continue
                candidate = proposal["smiles"]
                if not candidate:
                    raise PrecursorReuseError("exact completion lacks a graph")
                status = "eligible"
                if smiles not in train_products and candidate in train_products:
                    status = "would_reduce_train_novelty"
                elif counts[candidate] >= counts[smiles]:
                    status = "would_concentrate_products"
                decision["proposal_checks"].append(
                    {
                        "donor": proposal["donor"],
                        "status": status,
                        "original_multiplicity": counts[smiles],
                        "candidate_multiplicity": counts[candidate],
                    }
                )
                if status == "eligible":
                    eligible.append(proposal)
            if eligible:
                chosen = min(
                    eligible,
                    key=lambda p: (p["atom_edits"] + p["bond_edits"], p["smiles"], p["donor"]),
                )
                counts[smiles] -= 1
                counts[chosen["smiles"]] += 1
                decision.update(
                    selected_smiles=chosen["smiles"],
                    selected_donor=chosen["donor"],
                    disposition="admitted_exact_completion",
                )
        decisions.append(decision)
    return decisions
