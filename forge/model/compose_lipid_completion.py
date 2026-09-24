"""Checked first-donor completion with separate original and selected product ledgers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from forge.model.compose_lipid_generation import propose_generated_reuse
from forge.model.precursor_reuse_admission import admit_completions
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph

POLICY_ID = "compose_lipid_first_donor_checked_v1"


def _check(check_product: Callable, layout: Any, smiles: str) -> dict:
    checked = check_product(layout, smiles)
    if (
        not isinstance(checked, dict)
        or type(checked.get("exact")) is not bool
        or checked.get("status") not in {"evaluated", "abstained"}
        or (checked["exact"] and checked["status"] != "evaluated")
    ):
        raise ValueError("Completion requires an explicit full-program assessment")
    return checked


def check_generated_repeat_completion(
    layout: Any,
    state: Mapping[str, list],
    atoms: Any,
    *,
    check_product: Callable[[Any, str], dict],
    reason: str | None = None,
) -> dict:
    """Check an original and at most one fixed first-donor proposal.

    The supplied checker must be the qualified full source-program checker, including
    precursor domains, identity, balance and unique forward replay. Exceptions propagate.
    No source product or precursor graph is used to propose the completion. This function
    records chemistry feasibility only; population admission is separate.
    """
    return _check_completion(
        layout,
        state,
        atoms,
        check_product=check_product,
        reason=reason,
        policy_id=POLICY_ID,
        propose=lambda layout, state, atoms: propose_generated_reuse(
            layout, state, atoms, maximum_proposals=64
        ),
    )


def _check_completion(layout, state, atoms, *, check_product, reason, policy_id, propose):
    row = dict(
        policy_id=policy_id,
        family=layout.family,
        program=layout.record.program_id,
        original_smiles=None,
        original_check=dict(status="invalid_product", exact=False),
        reason=reason,
        status="original_invalid",
        proposals=[],
        source_check_calls=0,
    )
    if reason is not None:
        return row
    nodes, edges = state_graph(state)
    if not fixed_graph_preserved(nodes, edges, layout.record):
        raise ValueError("Completion input changed the fixed assembly graph")
    smiles = graph_smiles(nodes, edges, atoms)
    if smiles is None:
        return row
    original = _check(check_product, layout, smiles)
    row.update(original_smiles=smiles, original_check=original, source_check_calls=1)
    if original["exact"]:
        row["status"] = "original_exact"
        return row
    if original["status"] != "evaluated":
        row["status"] = "original_unassessed"
        return row
    proposed = propose(layout, state, atoms)
    row["status"] = proposed["status"]
    if not proposed["proposals"]:
        return row
    first = dict(proposed["proposals"][0], donor=0)
    if first["smiles"]:
        checked = _check(check_product, layout, first["smiles"])
        first.update(
            reaction_check=checked,
            status="exact_computed_program" if checked["exact"] else "program_check_failed",
        )
        row["source_check_calls"] += 1
    row["proposals"] = [first]
    return row


def admit_generated_repeat_completions(
    rows: Sequence[dict], *, is_train_product: Callable[[str], bool], policy_id: str = POLICY_ID
) -> list[dict]:
    """Admit checked first donors while preserving per-family novelty and diversity.

    All requested attempts are returned in input order, including invalid and rejected
    outputs. Existing exact products are immutable. The TRAIN membership callback must
    consult the pinned training population, never calibration or heldout data.
    """
    originals = []
    train_products = set()
    for row in rows:
        if row["policy_id"] != policy_id or len(row["proposals"]) > 1:
            raise ValueError("Completion ledger differs from the first-donor policy")
        for proposal in row["proposals"]:
            if proposal["donor"] != 0 or any(proposal["donors"]):
                raise ValueError("Completion changed the fixed donor choice")
            assessed = proposal.get("reaction_check", {})
            if (proposal["status"] == "exact_computed_program") != (
                assessed.get("status") == "evaluated" and assessed.get("exact") is True
            ):
                raise ValueError("Proposal status differs from its full-program assessment")
        smiles = row["original_smiles"]
        originals.append(
            dict(
                program_id=row["family"],
                canonical_smiles=smiles,
                valid_connected=smiles is not None,
                assembly=dict(
                    status="exact_computed_program" if row["original_check"]["exact"] else "failed"
                ),
            )
        )
        for candidate in [smiles, *(p["smiles"] for p in row["proposals"])]:
            if candidate and is_train_product(candidate):
                train_products.add(candidate)
    decisions = admit_completions(originals, list(rows), train_products)
    output = []
    for row, decision in zip(rows, decisions, strict=True):
        selected_check = row["original_check"]
        if decision["selected_donor"] is not None:
            selected_check = row["proposals"][0]["reaction_check"]
        output.append(
            dict(
                **row,
                decision=decision,
                selected_smiles=decision["selected_smiles"],
                selected_check=selected_check,
                accepted=selected_check["exact"],
                changed=decision["selected_smiles"] != row["original_smiles"],
            )
        )
    return output
