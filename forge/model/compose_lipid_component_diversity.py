"""Finite proposal selection with full-source and per-role diversity guards.

This changes selection, not the learned component distribution. Every rejected
proposal and original request remains in the denominator; no resampling occurs.
"""

from collections import Counter, defaultdict


def accepted_components(assessment):
    """Require a unique fully accepted precursor tuple, not a substructure hit."""
    if assessment.get("exact") is not True or assessment.get("status") != "evaluated":
        raise ValueError("Components require an evaluated full-source exact assessment")
    witnesses = {
        tuple(sorted(parts.items()))
        for check in assessment.get("checks", [])
        for parts in check.get("accepted_components", [])
    }
    if len(witnesses) != 1 or not next(iter(witnesses)):
        raise ValueError("Component diversity requires one accepted precursor tuple")
    return dict(next(iter(witnesses)))


def admit_component_diverse_proposals(rows, *, is_train_product):
    """Keep exact baselines; add proposals without decreasing any role's Simpson count.

    Pick candidates with the most previously unseen role components first, then
    the lowest summed component multiplicity. Ties follow request/proposal order.
    The bounded pool is fixed before selection. Integer arithmetic avoids relaxed
    numerical tolerances. Product novelty and multiplicity guards also apply.
    """
    products = defaultdict(Counter)
    components = defaultdict(lambda: defaultdict(Counter))
    selected, pending = [], []
    for i, row in enumerate(rows):
        old, assessment = row["baseline_smiles"], row["baseline_check"]
        if type(assessment.get("exact")) is not bool:
            raise ValueError("Baseline assessment must declare exact as a boolean")
        if old:
            products[row["family"]][old] += 1
        if assessment["exact"]:
            if not old:
                raise ValueError("Exact baseline has no product")
            for role, smiles in accepted_components(assessment).items():
                components[row["family"]][role][smiles] += 1
        selected.append(
            dict(
                **row,
                selected_smiles=old,
                selected_check=assessment,
                disposition=(
                    "baseline_exact_immutable" if assessment["exact"] else "retained_baseline"
                ),
                accepted=assessment["exact"],
                proposal_decisions=[],
            )
        )
        for j, proposal in enumerate(row.get("proposals", [])):
            decision = dict(index=j, disposition="not_full_source_exact")
            selected[-1]["proposal_decisions"].append(decision)
            if proposal.get("check", {}).get("exact") is True:
                parts = accepted_components(proposal["check"])
                if not proposal.get("smiles"):
                    raise ValueError("Exact proposal has no product")
                if assessment["exact"]:
                    decision["disposition"] = "baseline_exact_immutable"
                else:
                    pending.append((i, j, parts))
                    decision["disposition"] = "pending"
    while pending:

        def priority(item):
            i, j, parts = item
            inventory = components[rows[i]["family"]]
            counts = [inventory.get(role, {}).get(smiles, 0) for role, smiles in parts.items()]
            return (-sum(c == 0 for c in counts), sum(counts), i, j)

        i, j, parts = min(pending, key=priority)
        pending.remove((i, j, parts))
        row = selected[i]
        decision = row["proposal_decisions"][j]
        if row["accepted"]:
            decision["disposition"] = "request_already_completed"
            continue
        family, old = row["family"], row["baseline_smiles"]
        proposal = row["proposals"][j]
        candidate = proposal["smiles"]
        inventory = products[family]
        if old and not is_train_product(old) and is_train_product(candidate):
            reason = "would_reduce_train_novelty"
        elif (old and inventory[candidate] >= inventory[old]) or (not old and inventory[candidate]):
            reason = "would_concentrate_products"
        elif components[family] and set(parts) != set(components[family]):
            reason = "inconsistent_component_roles"
        else:
            reason = None
            for role, smiles in parts.items():
                counter = components[family][role]
                n = sum(counter.values())
                squares = sum(c * c for c in counter.values())
                if (n + 1) ** 2 * squares < n**2 * (squares + 2 * counter[smiles] + 1):
                    reason = "would_reduce_component_effective_count"
                    break
        if reason:
            decision["disposition"] = reason
            continue
        if old:
            inventory[old] -= 1
        inventory[candidate] += 1
        for role, smiles in parts.items():
            components[family][role][smiles] += 1
        row.update(
            selected_smiles=candidate,
            selected_check=proposal["check"],
            accepted=True,
            disposition="admitted_component_diverse_proposal",
        )
        decision["disposition"] = row["disposition"]
    return selected
