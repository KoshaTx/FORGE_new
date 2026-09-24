"""Bounded source-scaffold completion and single-bond core attachment readout.

These are explicit generation restrictions. Chemistry admission remains the
complete qualified source executor, independent of the proposal mechanism.
"""

from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
from rdkit import Chem

from forge.assembly.program import _repair_molecule
from forge.model.defog_feasibility import graph_to_molecule
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS


def admit_family_proposals(rows, *, is_train_product):
    """Admit one fully checked proposal while retaining the full baseline population.

    Unlike completion of a valid molecule, core readout can recover an abstention.
    A recovered abstention must add a distinct graph; valid replacements must
    preserve TRAIN novelty and cannot increase the squared multiplicity sum.
    """
    counts = defaultdict(Counter)
    for row in rows:
        if row["baseline_smiles"]:
            counts[row["family"]][row["baseline_smiles"]] += 1
    selected = []
    for row in rows:
        old, candidate = row["baseline_smiles"], row.get("proposal_smiles")
        baseline, proposal = row["baseline_check"], row.get("proposal_check", {})
        if type(baseline.get("exact")) is not bool or (
            baseline["exact"] and (not old or baseline.get("status") != "evaluated")
        ):
            raise ValueError("Invalid full-source baseline assessment")
        if proposal.get("exact") and (not candidate or proposal.get("status") != "evaluated"):
            raise ValueError("Exact proposal requires an evaluated full-source assessment")
        reason = "retained_baseline"
        chosen, check = old, baseline
        inventory = counts[row["family"]]
        if baseline["exact"]:
            reason = "baseline_exact_immutable"
        elif candidate and proposal.get("exact") is True:
            if old and not is_train_product(old) and is_train_product(candidate):
                reason = "would_reduce_train_novelty"
            elif (old and inventory[candidate] >= inventory[old]) or (
                not old and inventory[candidate] != 0
            ):
                reason = "would_concentrate_products"
            else:
                if old:
                    inventory[old] -= 1
                inventory[candidate] += 1
                chosen, check, reason = candidate, proposal, "admitted_full_source_proposal"
        selected.append(
            dict(
                **row,
                selected_smiles=chosen,
                selected_check=check,
                disposition=reason,
                accepted=check["exact"],
            )
        )
    return selected


def reserve_core_attachments(layout, parent_scores):
    """Reserve single-bond exterior attachments before greedy tree construction.

    Required valence comes from the admitted anonymous core condition. This
    opt-in readout restricts its exterior attachments to single bonds. It never
    changes node counts, closure slots, core chemistry or the original layout.
    Scarce parent choices are allocated first; this is a bounded greedy policy,
    not a claim of globally optimal constrained decoding.
    """
    record, graph = layout.record, layout.record.graph
    count = record.node_count
    if parent_scores.shape != (count, count) or not np.isfinite(parent_scores).all():
        raise ValueError("Core reservation requires finite square parent scores")
    used = np.zeros(count, dtype=np.int64)
    for child in np.flatnonzero(record.fixed_parent_bond_mask):
        used[[child, graph.parents[child]]] += int(BOND_VALENCE_UNITS[graph.parent_bonds[child]])
    for slot in np.flatnonzero(record.fixed_closure_bond_mask):
        used[[graph.closure_left[slot], graph.closure_right[slot]]] += int(
            BOND_VALENCE_UNITS[graph.closure_bonds[slot]]
        )
    requests = []
    first_children = []
    for block in record.component_blocks:
        exterior = [
            i for i in range(block.start, block.stop) if record.core_position_states[i] == 1
        ]
        if exterior and not record.fixed_parent_bond_mask[exterior[0]]:
            first_children.append(exterior[0])
        for core in range(block.start, block.stop):
            if layout.core_units[core] < 0:
                continue
            deficit = int(layout.core_units[core] - used[core])
            if deficit < 0 or deficit % 2:
                return None, "core_deficit_outside_single_bond_policy"
            candidates = tuple(
                i
                for i in range(core + 1, block.stop)
                if not record.fixed_parent_bond_mask[i] and record.core_position_states[i] == 1
            )
            requests.extend([(core, candidates)] * (deficit // 2))
    parents, bonds = graph.parents.copy(), graph.parent_bonds.copy()
    fixed = record.fixed_parent_bond_mask.copy()
    reserved = set()
    # The first exterior atom has no earlier exterior parent. Reserving all
    # core capacity for later children would make this atom impossible to place.
    for child in first_children:
        choices = [j for j, (_, candidates) in enumerate(requests) if child in candidates]
        if not choices:
            return None, "first_exterior_atom_has_no_core_attachment"
        choice = max(choices, key=lambda j: (float(parent_scores[child, requests[j][0]]), -j))
        core, _ = requests.pop(choice)
        parents[child], bonds[child], fixed[child] = core, 0, True
        reserved.add(child)
    for core, candidates in sorted(requests, key=lambda x: (len(x[1]), x[0])):
        available = [i for i in candidates if i not in reserved]
        if not available:
            return None, "insufficient_ordered_core_attachment_slots"
        child = max(available, key=lambda i: (float(parent_scores[i, core]), -i))
        reserved.add(child)
        parents[child], bonds[child], fixed[child] = core, 0, True
    return (
        replace(
            layout,
            record=replace(
                record,
                graph=replace(graph, parents=parents, parent_bonds=bonds),
                fixed_parent_bond_mask=fixed,
            ),
        ),
        None,
    )


def reserve_ordered_core_attachments(layout, parent_scores):
    """Reserve incoming edges of later core atoms before allocating core children.

    A multistage component may contain several core islands separated by its
    generated exterior. Later islands need an incoming tree edge, which already
    consumes part of their required valence. Treating their entire deficit as
    outgoing children falsely declares such layouts infeasible. The additional
    reservations stay inside the original component and use only single bonds.
    """
    record, graph = layout.record, layout.record.graph
    count = record.node_count
    if parent_scores.shape != (count, count) or not np.isfinite(parent_scores).all():
        raise ValueError("Core reservation requires finite square parent scores")
    parents, bonds = graph.parents.copy(), graph.parent_bonds.copy()
    fixed = record.fixed_parent_bond_mask.copy()
    incoming = np.zeros(count, dtype=np.int64)
    occupied = {
        tuple(sorted((int(graph.closure_left[s]), int(graph.closure_right[s]))))
        for s in np.flatnonzero(record.fixed_closure_bond_mask)
    }
    for block in record.component_blocks:
        for child in range(max(1, block.start), block.stop):
            if record.core_position_states[child] <= 1 or fixed[child]:
                continue
            earlier = [
                p
                for p in range(block.start, child)
                if record.core_position_states[p] == 1
                and incoming[p] < 3
                and (p, child) not in occupied
            ]
            if not earlier:
                return None, "later_core_has_no_exterior_parent"
            parent = max(earlier, key=lambda p: (float(parent_scores[child, p]), -p))
            parents[child], bonds[child], fixed[child] = parent, 0, True
            incoming[parent] += 1
    reserved = replace(
        layout,
        record=replace(
            record,
            graph=replace(graph, parents=parents, parent_bonds=bonds),
            fixed_parent_bond_mask=fixed,
        ),
    )
    return reserve_core_attachments(reserved, parent_scores)


def propose_scaffold_completion(layout, state, atoms, adapter, reaction, *, maximum_matches=256):
    """Copy one generated internal arm using registered precursor cut bonds.

    An intact registered scaffold and an unambiguous, size-preserving arm
    correspondence are required. Missing scaffolds abstain; no stored precursor
    or source product supplies atoms. Reverse provenance binds the scaffold to
    the generated component block rather than an arbitrary product substructure.
    """
    if type(maximum_matches) is not int or maximum_matches < 2:
        raise ValueError("Scaffold search bound must be an integer greater than one")
    if adapter.reaction_id != reaction["reaction_id"]:
        raise ValueError("Scaffold reaction and adapter differ")
    contract = reaction["precursor_scaffolds"]
    role = contract["precursor_role"]
    blocks = [b for b in layout.record.component_blocks if b.role == role]
    if len(blocks) != 1:
        return dict(status="scaffold_role_not_one_connected_block", proposals=[])
    block = blocks[0]
    nodes, edges = state_graph(state)
    if not fixed_graph_preserved(nodes, edges, layout.record):
        raise ValueError("Scaffold input changed the immutable assembly graph")
    product = graph_to_molecule(nodes, edges, atoms)
    outcomes = adapter.reaction.reverse.RunReactants((product,), maxProducts=maximum_matches)
    if len(outcomes) >= maximum_matches:
        return dict(status="scaffold_inverse_bound_reached", proposals=[])
    witnesses = {}
    for outcome in outcomes:
        precursor = _repair_molecule(outcome[adapter.roles.index(role)])
        if precursor is None:
            continue
        original = {
            a.GetIdx(): a.GetIntProp("react_atom_idx")
            for a in precursor.GetAtoms()
            if a.HasProp("react_atom_idx")
        }
        if set(original.values()) != set(range(block.start, block.stop)):
            continue
        for rule in contract["scaffolds"]:
            if not rule["identical_detached_arms"]:
                continue
            query = Chem.MolFromSmarts(rule["core_smarts"])
            maps = {a.GetAtomMapNum(): a.GetIdx() for a in query.GetAtoms() if a.GetAtomMapNum()}
            matches = precursor.GetSubstructMatches(
                query, uniquify=False, maxMatches=maximum_matches
            )
            if len(matches) >= maximum_matches:
                return dict(status="scaffold_match_bound_reached", proposals=[])
            for match in matches:
                ports = [(match[maps[a]], match[maps[b]]) for a, b in rule["cut_bond_maps"]]
                removed = {tuple(sorted(p)) for p in ports}
                arms = []
                for inner, outer in ports:
                    todo, arm = [outer], set()
                    while todo:
                        i = todo.pop()
                        if i in arm:
                            continue
                        arm.add(i)
                        todo.extend(
                            a.GetIdx()
                            for a in precursor.GetAtomWithIdx(i).GetNeighbors()
                            if tuple(sorted((i, a.GetIdx()))) not in removed
                            and a.GetIdx() not in arm
                        )
                    if inner in arm or match[maps[rule["anchor_map"]]] in arm:
                        break
                    if any(i not in original for i in arm):
                        break
                    arms.append([original[outer], *sorted(original[i] for i in arm - {outer})])
                if len(arms) != len(ports) or len({i for arm in arms for i in arm}) != sum(
                    len(arm) for arm in arms
                ):
                    continue
                # Symmetry-related SMARTS matches describe the same atom sets.
                key = (rule["id"], tuple(sorted(tuple(sorted(arm)) for arm in arms)))
                ordered = sorted(arms, key=lambda a: tuple(a))
                witnesses[key] = (rule["id"], ordered)
    if len(witnesses) != 1:
        return dict(status="missing_or_ambiguous_registered_scaffold", proposals=[])
    scaffold_id, arms = next(iter(witnesses.values()))
    if len({len(a) for a in arms}) != 1:
        return dict(status="unequal_generated_scaffold_arm_sizes", proposals=[])
    if any(layout.record.fixed_atom_mask[i] for arm in arms for i in arm):
        return dict(status="scaffold_arm_overlaps_fixed_core", proposals=[])
    changed_nodes, changed_edges = nodes.copy(), edges.copy()
    donor = arms[0]
    for target in arms[1:]:
        changed_nodes[target] = nodes[donor]
        changed_edges[np.ix_(target, target)] = edges[np.ix_(donor, donor)]
    if np.count_nonzero(changed_edges) != np.count_nonzero(edges):
        return dict(status="scaffold_completion_changes_cycle_count", proposals=[])
    if not fixed_graph_preserved(changed_nodes, changed_edges, layout.record):
        raise ValueError("Scaffold completion changed immutable assembly graph")
    smiles = graph_smiles(changed_nodes, changed_edges, atoms)
    return dict(
        status="proposals_complete",
        proposals=[
            dict(
                donor=0,
                donors=[0],
                smiles=smiles,
                scaffold_id=scaffold_id,
                status="requires_full_source_check" if smiles else "invalid_or_disconnected",
                atom_edits=int(np.count_nonzero(changed_nodes != nodes)),
                bond_edits=int(np.count_nonzero(np.triu(changed_edges != edges, k=1))),
            )
        ],
    )
