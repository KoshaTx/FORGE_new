"""Bounded component-coupled terminal proposals from generated atoms and logits.

All chemical predicates are supplied by qualified source policies. No reference
component graph is retrieved. Local edits preserve the atom support, adjacency,
cycle counts, source origins and fixed assembly graph. An independent complete
source replay is required even when every construction predicate is satisfied.
"""

import itertools
from dataclasses import replace

import numpy as np
from rdkit import Chem, rdBase

from forge.corpus.compose_lipid_source_program import retained_amine
from forge.model.compose_lipid_generation import _repeat_correspondences
from forge.model.defog_feasibility import graph_to_molecule
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS, INDEX_TO_DENSE_BOND


def _molecule(nodes, edges, atoms):
    try:
        with rdBase.BlockLogs():
            molecule = graph_to_molecule(nodes, edges, atoms)
            Chem.SanitizeMol(molecule)
        return molecule
    except (ValueError, RuntimeError):
        return None


def _violations(nodes, edges, atoms, layout, policies):
    molecule = _molecule(nodes, edges, atoms)
    if molecule is None:
        return None
    failures = []
    for policy in policies:
        blocks = [b for b in layout.record.component_blocks if b.role == policy["role"]]
        slot_groups = [tuple(range(b.start, b.stop)) for b in blocks]
        if layout.quantities[policy["role"]] == 1:
            slot_groups = [tuple(i for group in slot_groups for i in group)]
        elif len(slot_groups) != layout.quantities[policy["role"]]:
            raise ValueError("Repeated source correspondence is not uniquely connected")
        for slots in slot_groups:
            selected = set(slots)
            for i in slots:
                state = atoms[nodes[i]]
                atomic = Chem.GetPeriodicTable().GetAtomicNumber(state.symbol)
                if (
                    "allowed_atomic_numbers" in policy
                    and atomic not in policy["allowed_atomic_numbers"]
                ) or (policy.get("allow_aromatic_atoms") is False and state.aromatic):
                    failures.append(dict(kind="atom_domain", slots=(i,), cost=1, policy=policy))
            for symbol, expected in policy.get("product_element_counts", {}).items():
                observed = sum(atoms[nodes[i]].symbol == symbol for i in slots)
                if observed != expected:
                    failures.append(
                        dict(
                            kind="element_count",
                            symbol=symbol,
                            difference=expected - observed,
                            slots=slots,
                            cost=abs(expected - observed),
                            policy=policy,
                        )
                    )
            for query in policy.get("required_queries", []):
                matches = molecule.GetSubstructMatches(
                    Chem.MolFromSmarts(query["smarts"]), maxMatches=257
                )
                matches = [
                    m
                    for m in matches
                    if m[0] in selected
                    and (not query.get("whole_component") or set(m) <= selected)
                    and (
                        not query.get("exclude_core")
                        or layout.record.core_position_states[m[0]] == 1
                    )
                ]
                if len(matches) >= 256:
                    raise ValueError("Component query enumeration saturated")
                if len(matches) < query["minimum_matches"]:
                    failures.append(
                        dict(
                            kind="required_query",
                            slots=slots,
                            cost=query["minimum_matches"] - len(matches),
                            policy=policy,
                        )
                    )
            if "retained_amine_policy" in policy:
                matches = set(retained_amine(molecule, policy["retained_amine_policy"])) & selected
                if not matches:
                    failures.append(dict(kind="retained_amine", slots=slots, cost=1, policy=policy))
            for query in policy.get("remaining_handles", []):
                matches = [
                    m
                    for m in molecule.GetSubstructMatches(
                        Chem.MolFromSmarts(query["smarts"]), maxMatches=257
                    )
                    if set(m) <= selected
                    and (
                        not query.get("exclude_core")
                        or all(layout.record.core_position_states[i] == 1 for i in m)
                    )
                ]
                if len(matches) >= 256:
                    raise ValueError("Remaining handle enumeration saturated")
                if len(matches) > query["maximum_matches"]:
                    failures.append(
                        dict(
                            kind="extra_handle",
                            slots=tuple(sorted({i for m in matches for i in m})),
                            cost=len(matches) - query["maximum_matches"],
                            policy=policy,
                        )
                    )
    return failures


def _conserved(nodes, edges, layout, reference_edges):
    if not fixed_graph_preserved(nodes, edges, layout.record):
        return False
    for block in layout.record.component_blocks:
        selected = np.s_[block.start : block.stop, block.start : block.stop]
        if np.count_nonzero(edges[selected]) != np.count_nonzero(reference_edges[selected]):
            return False
    units = np.asarray([0, *BOND_VALENCE_UNITS.tolist()])[edges].sum(1)
    core = layout.core_units >= 0
    return np.array_equal(units[core], layout.core_units[core])


def _correspondences(layout, aliases):
    positions = np.asarray(
        [aliases.get(int(p), int(p)) for p in layout.record.core_position_states]
    )
    grouped, reason = _repeat_correspondences(
        replace(layout, record=replace(layout.record, core_position_states=positions))
    )
    return grouped, reason


def _copy(nodes, edges, groups, donors):
    n, e = nodes.copy(), edges.copy()
    for group, donor in zip(groups, donors, strict=True):
        source = group[donor]
        for target in group:
            n[target] = nodes[source]
            e[np.ix_(target, target)] = edges[np.ix_(source, source)]
    return n, e


def _tied_locations(groups, index):
    for group in groups:
        for occurrence in group:
            if index in occurrence:
                column = occurrence.index(index)
                return tuple(o[column] for o in group)
    return (index,)


def _mutations(nodes, edges, predictions, atoms, layout, state, groups, failures, limit):
    """Rank finite model-supported changes; the constraint objective chooses among them."""
    edits = {}
    fixed = layout.record.fixed_atom_mask
    state_atoms = {
        i: Chem.GetPeriodicTable().GetAtomicNumber(a.symbol) for i, a in enumerate(atoms)
    }
    for failure in failures:
        policy = failure["policy"]
        for i in failure["slots"]:
            tied = _tied_locations(groups, i)
            if i != min(tied) or any(fixed[j] for j in tied):
                continue
            choices = []
            for value, atom in enumerate(atoms):
                if value == nodes[i]:
                    continue
                if (
                    "allowed_atomic_numbers" in policy
                    and state_atoms[value] not in policy["allowed_atomic_numbers"]
                ):
                    continue
                if policy.get("allow_aromatic_atoms") is False and atom.aromatic:
                    continue
                if failure["kind"] == "element_count":
                    if failure["difference"] > 0 and atom.symbol != failure["symbol"]:
                        continue
                    if failure["difference"] < 0 and (
                        atoms[nodes[i]].symbol != failure["symbol"]
                        or atom.symbol == failure["symbol"]
                    ):
                        continue
                gain = sum(
                    float(predictions["nodes"][j, value] - predictions["nodes"][j, nodes[j]])
                    for j in tied
                )
                choices.append((gain, value))
            # All vocabulary states remain candidates before the global bounded ranking.
            for gain, value in choices:
                edits[("atom", tied, value)] = gain
            if failure["kind"] in {"extra_handle", "required_query", "retained_amine"}:
                for j in np.flatnonzero(edges[i]):
                    j = int(j)
                    if fixed[j]:
                        continue
                    other = _tied_locations(groups, j)
                    if len(other) != len(tied):
                        continue
                    pairs = tuple(
                        sorted(tuple(sorted((a, b))) for a, b in zip(tied, other, strict=True))
                    )
                    if any(a == b or not edges[a, b] for a, b in pairs):
                        continue
                    for bond, value in enumerate(INDEX_TO_DENSE_BOND):
                        current_bond = list(INDEX_TO_DENSE_BOND).index(int(edges[i, j]))
                        if max(bond, current_bond) >= predictions["parent_bonds"].shape[-1]:
                            continue
                        if value == edges[i, j]:
                            continue
                        gains = []
                        for a, b in pairs:
                            if int(state["parents"][b]) == a:
                                gains.append(
                                    float(
                                        predictions["parent_bonds"][b, bond]
                                        - predictions["parent_bonds"][
                                            b, list(INDEX_TO_DENSE_BOND).index(int(edges[a, b]))
                                        ]
                                    )
                                )
                            else:
                                gains.append(0.0)
                        edits[("bond", pairs, int(value))] = sum(gains)
    return sorted(edits.items(), key=lambda x: (-x[1], x[0]))[:limit]


def _branch_mutations(nodes, edges, predictions, atoms, layout, groups, failures, limit):
    """Rank an atom change coupled to one local branch relocation outside cores.

    Moving j--k to i--k in an i--j--k path changes substitution without changing
    the component's atoms or edge count. Sanitization and connectivity still
    apply. This lets a retained-site requirement and a competing-site exclusion
    be met together when either edit on its own cannot improve the predicates.
    """
    edits = {}
    for failure in failures:
        if failure["kind"] not in {"required_query", "retained_amine"}:
            continue
        slots = set(failure["slots"])
        for i in sorted(slots):
            tied = _tied_locations(groups, i)
            if i != min(tied) or any(layout.record.fixed_atom_mask[t] for t in tied):
                continue
            for j in np.flatnonzero(edges[i] == 1):
                if j not in slots or layout.record.fixed_atom_mask[j]:
                    continue
                middle = _tied_locations(groups, int(j))
                for k in np.flatnonzero(edges[j] == 1):
                    if k == i or k not in slots or edges[i, k] or layout.record.fixed_atom_mask[k]:
                        continue
                    outer = _tied_locations(groups, int(k))
                    if len(tied) != len(middle) or len(tied) != len(outer):
                        continue
                    triples = tuple(zip(tied, middle, outer, strict=True))
                    if any(edges[a, c] or edges[b, c] != 1 for a, b, c in triples):
                        continue
                    for value, atom in enumerate(atoms):
                        if value == nodes[i] or atom.aromatic:
                            continue
                        atomic = Chem.GetPeriodicTable().GetAtomicNumber(atom.symbol)
                        allowed = failure["policy"].get("allowed_atomic_numbers")
                        if allowed is not None and atomic not in allowed:
                            continue
                        gain = sum(
                            float(
                                predictions["nodes"][a, value] - predictions["nodes"][a, nodes[a]]
                            )
                            for a in tied
                        )
                        edits[(triples, value)] = gain
    return sorted(edits.items(), key=lambda x: (-x[1], x[0]))[:limit]


def propose_component_constraints(
    layout,
    state,
    predictions,
    atoms,
    policies,
    *,
    core_aliases=(),
    maximum_edits=4,
    maximum_mutations=128,
    maximum_donors=64,
    allow_branch_edits=False,
):
    """Return a predicate-screened donor proposal and one bounded coupled repair proposal.

    Full source checks and population diversity selection are separate. A lower
    predicate deficit does not make an intermediate molecule an accepted result.
    """
    if any(type(v) is not int or v < 1 for v in (maximum_edits, maximum_mutations, maximum_donors)):
        raise ValueError("Construction budgets must be positive integers")
    if any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("Component constraints require finite model predictions")
    roles = {b.role for b in layout.record.component_blocks}
    if any(p["role"] not in roles for p in policies):
        raise ValueError("Construction policy references a missing source role")
    nodes, edges = state_graph(state)
    if not _conserved(nodes, edges, layout, edges):
        raise ValueError("Component construction input changes the assembly core")
    groups, reason = _correspondences(layout, dict(core_aliases))
    if reason:
        return dict(status=reason, proposals=[])
    donors = list(itertools.product(*(range(len(g)) for g in groups)))
    if len(donors) > maximum_donors:
        return dict(status="donor_budget_exceeded", proposals=[])
    choices = []
    for indices in donors:
        n, e = _copy(nodes, edges, groups, indices)
        if np.count_nonzero(e) != np.count_nonzero(edges) or not _conserved(n, e, layout, edges):
            continue
        violations = _violations(n, e, atoms, layout, policies)
        if violations is None:
            continue
        score = sum(float(predictions["nodes"][i, n[i]]) for i in range(len(n)))
        choices.append((sum(v["cost"] for v in violations), -score, indices, n, e, violations))
    if not choices:
        return dict(status="no_valid_generated_donor", proposals=[])
    cost, _, chosen, n, e, failures = min(choices, key=lambda v: v[:3])
    proposals = []

    def emit(label, history):
        return dict(
            policy=label,
            smiles=graph_smiles(n, e, atoms) if not failures else None,
            status=(
                "component_predicates_satisfied"
                if not failures
                else "component_predicates_unresolved"
            ),
            remaining_deficit=sum(v["cost"] for v in failures),
            donors=list(chosen),
            edits=history,
            nodes=n.tolist(),
            edges=e.tolist(),
        )

    proposals.append(emit("whole_component_donor", []))
    history = []
    evaluated = 0
    for _ in range(maximum_edits):
        if not failures:
            break
        candidates = []
        for (kind, locations, value), gain in _mutations(
            n, e, predictions, atoms, layout, state, groups, failures, maximum_mutations
        ):
            nn, ee = n.copy(), e.copy()
            if kind == "atom":
                nn[list(locations)] = value
            else:
                for a, b in locations:
                    ee[a, b] = ee[b, a] = value
            if not _conserved(nn, ee, layout, edges):
                continue
            vv = _violations(nn, ee, atoms, layout, policies)
            evaluated += 1
            if vv is None:
                continue
            cc = sum(v["cost"] for v in vv)
            if cc < cost:
                candidates.append((cc, -gain, (kind, locations, value), nn, ee, vv))
        if not candidates:
            if not allow_branch_edits:
                break
            for (triples, value), gain in _branch_mutations(
                n, e, predictions, atoms, layout, groups, failures, maximum_mutations
            ):
                nn, ee = n.copy(), e.copy()
                for a, b, c in triples:
                    nn[a] = value
                    ee[b, c] = ee[c, b] = 0
                    ee[a, c] = ee[c, a] = 1
                if not _conserved(nn, ee, layout, edges):
                    continue
                vv = _violations(nn, ee, atoms, layout, policies)
                evaluated += 1
                if vv is None:
                    continue
                cc = sum(v["cost"] for v in vv)
                if cc < cost and graph_smiles(nn, ee, atoms):
                    candidates.append((cc, -gain, ("branch_atom", triples, value), nn, ee, vv))
            if not candidates:
                break
        cost, _, edit, n, e, failures = min(candidates, key=lambda v: v[:3])
        history.append(dict(kind=edit[0], locations=edit[1], value=edit[2]))
    if history:
        proposals.append(emit("coupled_component_repair", history))
    return dict(
        status="bounded_component_construction",
        proposals=proposals,
        donors_considered=len(donors),
        mutations_evaluated=evaluated,
    )
