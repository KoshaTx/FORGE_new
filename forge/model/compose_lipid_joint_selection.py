"""Bounded joint admission with exact final component-diversity checks.

The optimizer chooses among an already assessed finite pool. It cannot create
products, replace exact baselines, change chemistry checks, or discard requests.
Inverse Simpson constraints are linearized with integer component counts. The
returned assignment is independently checked using integer arithmetic before
any result is admitted; a solver tolerance never relaxes a scientific gate.
"""

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from forge.model.compose_lipid_component_diversity import accepted_components


def _solve(candidates, inventory, *, node_limit, time_limit):
    """Binary choices plus prefix indicators encode each squared role count."""
    objective, coefficients, lower, upper = [], [], [], []

    def variable(cost=0):
        objective.append(cost)
        return len(objective) - 1

    def constraint(terms, low=-np.inf, high=np.inf):
        coefficients.append(terms)
        lower.append(low)
        upper.append(high)

    requests, products, role_options = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, candidate in enumerate(candidates):
        # Cardinality dominates the entire deterministic secondary objective.
        variable(-1 + (i + 1) / ((len(candidates) + 1) ** 2 * 2))
        requests[candidate["row"]].append(i)
        products[candidate["smiles"]].append(i)
        for role, smiles in candidate["parts"].items():
            role_options[role, smiles].append(i)
    for indices in [*requests.values(), *products.values()]:
        constraint({i: 1 for i in indices}, high=1)
    z = [variable() for _ in range(len(requests) + 1)]
    constraint(dict.fromkeys(z, 1), low=1, high=1)
    constraint(
        {**{i: 1 for i in range(len(candidates))}, **{v: -k for k, v in enumerate(z)}},
        low=0,
        high=0,
    )
    increments = defaultdict(dict)
    for (role, smiles), indices in sorted(role_options.items()):
        limit = len({candidates[i]["row"] for i in indices})
        prefix = [variable() for _ in range(limit)]
        constraint({**dict.fromkeys(indices, 1), **dict.fromkeys(prefix, -1)}, low=0, high=0)
        for left, right in zip(prefix, prefix[1:]):
            constraint({left: 1, right: -1}, low=0)
        old = inventory[role][smiles]
        for k, v in enumerate(prefix, 1):
            increments[role][v] = 2 * old + 2 * k - 1
    for role, counts in inventory.items():
        n = sum(counts.values())
        squares = sum(c * c for c in counts.values())
        # n² * (old squared sum + new increments) <= old sum * (n + k)².
        terms = {i: n * n * v for i, v in increments[role].items()}
        terms.update({v: -squares * (n + k) ** 2 for k, v in enumerate(z)})
        constraint(terms, high=-n * n * squares)
    rr, cc, values = [], [], []
    for r, terms in enumerate(coefficients):
        for c, v in terms.items():
            rr.append(r)
            cc.append(c)
            values.append(v)
    matrix = coo_matrix((values, (rr, cc)), shape=(len(coefficients), len(objective))).tocsc()
    result = milp(
        np.asarray(objective),
        integrality=np.ones(len(objective)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options=dict(node_limit=node_limit, time_limit=time_limit, mip_rel_gap=0.0),
    )
    info = dict(
        status=int(result.status),
        message=result.message,
        optimal=result.status == 0,
        candidates=len(candidates),
        requests=len(requests),
        nodes=int(getattr(result, "mip_node_count", 0) or 0),
    )
    if result.x is None:
        return [], dict(info, disposition="no_incumbent_retained_baseline")
    chosen = result.x[: len(candidates)]
    if not np.isfinite(chosen).all() or not np.allclose(chosen, np.rint(chosen), atol=1e-6, rtol=0):
        raise ValueError("Joint selector returned a nonintegral assignment")
    return [candidates[i] for i in np.flatnonzero(np.rint(chosen))], info


def admit_joint_component_proposals(rows, *, is_train_product, node_limit=10000, time_limit=5.0):
    """Admit a jointly balanced set, preserving all final baseline quality floors.

    Products already present anywhere in a family's baseline are conservatively
    excluded from the additions. New products are unique within each family.
    Exact baseline tuples remain immutable, so component unique counts cannot
    decrease. A novel valid baseline can only be replaced by a novel product.
    """
    if (
        type(node_limit) is not int
        or node_limit < 1
        or not np.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("Finite positive solver bounds are required")
    selected, families = [], defaultdict(list)
    for i, row in enumerate(rows):
        old, check = row["baseline_smiles"], row["baseline_check"]
        if type(check.get("exact")) is not bool:
            raise ValueError("Baseline exact must be boolean")
        if check["exact"]:
            if not old:
                raise ValueError("Exact baseline has no product")
            accepted_components(check)
        selected.append(
            dict(
                **row,
                selected_smiles=old,
                selected_check=check,
                accepted=check["exact"],
                disposition="baseline_exact_immutable" if check["exact"] else "retained_baseline",
                proposal_decisions=[],
            )
        )
        families[row["family"]].append(i)
    reports = {}
    for family, indices in sorted(families.items()):
        products = {rows[i]["baseline_smiles"] for i in indices if rows[i]["baseline_smiles"]}
        inventory = defaultdict(Counter)
        for i in indices:
            if rows[i]["baseline_check"]["exact"]:
                for role, smiles in accepted_components(rows[i]["baseline_check"]).items():
                    inventory[role][smiles] += 1
        candidates = []
        for i in indices:
            row = rows[i]
            seen = set()
            for j, proposal in enumerate(row.get("proposals", [])):
                decision = dict(index=j, disposition="not_full_source_exact")
                selected[i]["proposal_decisions"].append(decision)
                check, smiles = proposal.get("check", {}), proposal.get("smiles")
                if check.get("exact") is not True:
                    continue
                parts = accepted_components(check)
                if not smiles:
                    raise ValueError("Exact proposal has no product")
                if row["baseline_check"]["exact"]:
                    reason = "baseline_exact_immutable"
                elif smiles in seen:
                    reason = "duplicate_proposal"
                elif smiles in products:
                    reason = "already_present_baseline_product"
                elif (
                    row["baseline_smiles"]
                    and not is_train_product(row["baseline_smiles"])
                    and is_train_product(smiles)
                ):
                    reason = "would_reduce_train_novelty"
                elif inventory and set(parts) != set(inventory):
                    reason = "inconsistent_component_roles"
                else:
                    reason = "not_selected_jointly"
                    candidates.append(dict(row=i, proposal=j, smiles=smiles, parts=parts))
                seen.add(smiles)
                decision["disposition"] = reason
        if candidates and not inventory:
            # A consistent role space is required even when no baseline is exact.
            if len({tuple(sorted(c["parts"])) for c in candidates}) != 1:
                raise ValueError("Ambiguous component role space without an exact baseline")
        chosen, report = (
            _solve(candidates, inventory, node_limit=node_limit, time_limit=time_limit)
            if candidates
            else ([], dict(status=0, optimal=True, candidates=0, requests=0))
        )
        # Independent integer checks: a solver incumbent is never scientific admission.
        if len({c["row"] for c in chosen}) != len(chosen) or len(
            {c["smiles"] for c in chosen}
        ) != len(chosen):
            raise ValueError("Joint selection violates request or product uniqueness")
        after = {r: c.copy() for r, c in inventory.items()}
        for c in chosen:
            for role, smiles in c["parts"].items():
                after.setdefault(role, Counter())[smiles] += 1
        for role, counts in inventory.items():
            n, nn = sum(counts.values()), sum(after[role].values())
            s, ss = sum(v * v for v in counts.values()), sum(v * v for v in after[role].values())
            if nn * nn * s < n * n * ss:
                raise ValueError("Joint assignment decreases exact component diversity")
        for c in chosen:
            row, j = selected[c["row"]], c["proposal"]
            proposal = row["proposals"][j]
            row.update(
                selected_smiles=proposal["smiles"],
                selected_check=proposal["check"],
                accepted=True,
                disposition="admitted_joint_component_proposal",
            )
            row["proposal_decisions"][j]["disposition"] = row["disposition"]
        reports[family] = dict(report, admitted=len(chosen))
    return selected, reports
