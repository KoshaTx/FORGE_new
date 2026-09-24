"""Registry-derived equality and symmetric-arm proposals on generated graphs.

These are explicit terminal completions, not samples from the unchanged neural flow.
They use no source precursor atoms. Full source-program assessment remains mandatory.
"""

from dataclasses import dataclass, replace

import numpy as np
from rdkit.Chem import rdChemReactions

from forge.model.compose_lipid_completion import POLICY_ID as REPEAT_POLICY_ID
from forge.model.compose_lipid_completion import _check, _check_completion
from forge.model.compose_lipid_generation import _repeat_correspondences
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS

POLICY_ID = "compose_lipid_registry_constraints_checked_v2"


@dataclass(frozen=True)
class SourceConstraints:
    program_id: str
    core_aliases: tuple[tuple[int, int], ...] = ()
    symmetric_anchors: tuple[tuple[int, ...], ...] = ()


def query_correspondences(left, right, *, maximum_states=4096):
    """Exact query-graph isomorphisms, ignoring only atom-map labels.

    RDKit query-to-query matching need not establish query equality. Compare query
    predicates and bond predicates explicitly; a search bound fails loudly.
    """
    if left.GetNumAtoms() != right.GetNumAtoms():
        return []

    def labels(mol):
        atoms = [(a.DescribeQuery(), a.GetDegree()) for a in mol.GetAtoms()]
        bonds = {}
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            label = bond.DescribeQuery()
            # An implicit SMARTS bond also admits aromatic bonds. With an explicitly
            # aliphatic endpoint that alternative cannot describe an aromatic ring.
            aliphatic = any(
                f"AtomType {atom.GetAtomicNum()} = val" in atom.DescribeQuery()
                and "AtomOr" not in atom.DescribeQuery()
                and "!= val" not in atom.DescribeQuery()
                for atom in (bond.GetBeginAtom(), bond.GetEndAtom())
            )
            if label == "SingleOrAromaticBond 1 = val\n" and aliphatic:
                label = "BondOrder 1 = val\n"
            bonds[a, b] = bonds[b, a] = label
        return atoms, bonds

    la, lb = labels(left)
    ra, rb = labels(right)
    candidates = [[j for j, label in enumerate(ra) if label == value] for value in la]
    order = sorted(range(len(la)), key=lambda i: (len(candidates[i]), -la[i][1], i))
    found, mapping = [], {}
    visited = 0

    def visit(depth):
        nonlocal visited
        visited += 1
        if visited > maximum_states:
            raise ValueError("Registry query correspondence exceeded its explicit search bound")
        if depth == len(order):
            found.append(tuple(mapping[i] for i in range(len(la))))
            return
        i = order[depth]
        for j in candidates[i]:
            if j in mapping.values() or any(
                lb.get((i, k)) != rb.get((j, v)) for k, v in mapping.items()
            ):
                continue
            mapping[i] = j
            visit(depth + 1)
            del mapping[i]

    visit(0)
    return sorted(found)


def compile_constraints(
    registry, *, program_id, core_vocabulary, sequential_program=None, reaction_id=None
):
    """Compile only mapped relationships present in the supplied qualified registry.

    Sequential equal-role queries must have a unique correspondence. Symmetric
    precursor attachment orbits are an optional generation restriction, not a
    claim that all admitted products must be symmetric.
    """
    if (sequential_program is None) == (reaction_id is None):
        raise ValueError("Select exactly one registered program or reaction")
    positions = {name: i for i, name in enumerate(core_vocabulary)}
    reactions = {r["reaction_id"]: r for r in registry["reactions"]}
    aliases, anchors = {}, []

    def position(suffix):
        return positions[f"{program_id}:{suffix}"]

    if sequential_program is not None:
        program = next(p for p in registry["programs"] if p["program_id"] == sequential_program)
        for step, stage in enumerate(program["stages"], 1):
            reaction = reactions[stage["reaction_id"]]
            rxn = rdChemReactions.ReactionFromSmarts(reaction["atom_mapped_reaction_smarts"])
            roles = {r["name"]: i for i, r in enumerate(reaction["reactant_roles"])}
            for group in program.get("equal_component_groups", []):
                present = [role for role in group if role in roles]
                if len(present) < 2:
                    continue
                first = rxn.GetReactantTemplate(roles[present[0]])
                for role in present[1:]:
                    other = rxn.GetReactantTemplate(roles[role])
                    correspondences = query_correspondences(first, other)
                    if len(correspondences) != 1:
                        raise ValueError("Declared equal-role query correspondence is ambiguous")
                    for a, b in enumerate(correspondences[0]):
                        ma, mb = (
                            first.GetAtomWithIdx(a).GetAtomMapNum(),
                            other.GetAtomWithIdx(b).GetAtomMapNum(),
                        )
                        if ma and mb:
                            aliases[position(f"step_{step}:map_{mb}")] = position(
                                f"step_{step}:map_{ma}"
                            )
    else:
        reaction = reactions[reaction_id]
        rxn = rdChemReactions.ReactionFromSmarts(reaction["atom_mapped_reaction_smarts"])
        repeated = reaction["source_program"]["repeated_roles"]
        for i, role in enumerate(reaction["reactant_roles"]):
            if role["name"] not in repeated:
                continue
            query = rxn.GetReactantTemplate(i)
            matches = query_correspondences(query, query)
            seen = set()
            for atom in query.GetAtoms():
                index = atom.GetIdx()
                if index in seen:
                    continue
                orbit = sorted({match[index] for match in matches})
                seen.update(orbit)
                maps = [query.GetAtomWithIdx(j).GetAtomMapNum() for j in orbit]
                if len(orbit) > 1 and all(maps):
                    anchors.append(tuple(position(f"map_{m}") for m in maps))
    return SourceConstraints(program_id, tuple(sorted(aliases.items())), tuple(anchors))


def _arm_groups(layout, edges, constraints):
    """Resolve generated exterior branches attached to equivalent template atoms."""
    record = layout.record
    core = record.core_position_states > 1
    groups = []
    for block in record.component_blocks:
        allowed = set(range(block.start, block.stop))
        if not any(
            int(record.core_position_states[i]) in orbit
            for i in allowed
            for orbit in constraints.symmetric_anchors
        ):
            continue
        remaining = {i for i in allowed if not core[i]}
        branches = {}
        while remaining:
            todo, component = [min(remaining)], set()
            while todo:
                i = todo.pop()
                if i in component:
                    continue
                component.add(i)
                todo.extend(
                    int(j)
                    for j in np.flatnonzero(edges[i])
                    if j in remaining and j not in component
                )
            remaining.difference_update(component)
            boundary = {
                (i, int(j))
                for i in component
                for j in np.flatnonzero(edges[i])
                if j not in component
            }
            if len(boundary) != 1:
                return None, "exterior_branch_has_multiple_attachments"
            exterior, anchor = next(iter(boundary))
            if anchor not in allowed or not core[anchor] or anchor in branches:
                return None, "exterior_branch_attachment_unresolved"
            branches[anchor] = (exterior, component)
        for orbit in constraints.symmetric_anchors:
            found = [
                [i for i in allowed if int(record.core_position_states[i]) == label]
                for label in orbit
            ]
            if not any(found):
                continue
            if any(len(x) != 1 for x in found):
                return None, "symmetric_core_occurrence_unresolved"
            roots = [x[0] for x in found]
            if not any(i in branches for i in roots):
                continue
            if not all(i in branches for i in roots):
                return None, "symmetric_arm_missing"
            occurrences = []
            for root in roots:
                exterior, component = branches[root]
                occurrences.append([root, exterior, *sorted(component - {exterior})])
            if len({len(x) for x in occurrences}) != 1:
                return None, "unequal_generated_arm_sizes"
            groups.append(occurrences)
    return groups, None


def propose_source_completion(layout, state, atoms, constraints):
    """One deterministic generated donor per registered relation, preserving support."""
    if layout.record.program_id != constraints.program_id:
        raise ValueError("Source constraints belong to another program")
    nodes, edges = state_graph(state)
    if not fixed_graph_preserved(nodes, edges, layout.record):
        raise ValueError("Completion input changed the immutable assembly graph")
    aliases = dict(constraints.core_aliases)
    positions = np.asarray(
        [aliases.get(int(p), int(p)) for p in layout.record.core_position_states]
    )
    correspondence_layout = replace(
        layout, record=replace(layout.record, core_position_states=positions)
    )
    repeated, reason = _repeat_correspondences(correspondence_layout)
    if reason:
        return dict(status=reason, proposals=[])
    arms, reason = (
        _arm_groups(layout, edges, constraints) if constraints.symmetric_anchors else ([], None)
    )
    if reason:
        return dict(status=reason, proposals=[])
    groups = [*arms, *repeated]
    if not groups:
        return dict(status="no_registered_relation", proposals=[])
    changed_nodes, changed_edges = nodes.copy(), edges.copy()
    for occurrences in groups:
        source = occurrences[0]
        donor_nodes = changed_nodes[source].copy()
        donor_edges = changed_edges[np.ix_(source, source)].copy()
        for target in occurrences[1:]:
            changed_nodes[target] = donor_nodes
            changed_edges[np.ix_(target, target)] = donor_edges
    reason = None
    if np.count_nonzero(changed_edges) != np.count_nonzero(edges):
        reason = "changed_edge_or_cycle_count"
    elif not fixed_graph_preserved(changed_nodes, changed_edges, layout.record):
        reason = "changed_fixed_core"
    units = np.asarray([0, *BOND_VALENCE_UNITS.tolist()])[changed_edges].sum(1)
    core = layout.core_units >= 0
    if reason is None and not np.array_equal(units[core], layout.core_units[core]):
        reason = "changed_core_saturation"
    smiles = None if reason else graph_smiles(changed_nodes, changed_edges, atoms)
    return dict(
        status="proposals_complete",
        proposals=[
            dict(
                donor=0,
                donors=[0] * len(groups),
                smiles=smiles,
                status=reason
                or ("requires_exact_program_check" if smiles else "invalid_or_disconnected"),
                atom_edits=int(np.count_nonzero(changed_nodes != nodes)),
                bond_edits=int(np.count_nonzero(np.triu(changed_edges != edges, k=1))),
                registered_repeat_groups=len(repeated),
                symmetric_arm_groups=len(arms),
            )
        ],
    )


def check_source_completion(
    layout, state, atoms, constraints, *, check_product, reason=None, baseline=None
):
    """Extend an admitted repeat completion without replacing its exact products."""
    if constraints.program_id != layout.record.program_id:
        raise ValueError("Source constraints belong to another program")
    if baseline is not None:
        if (
            baseline["policy_id"] != REPEAT_POLICY_ID
            or baseline["program"] != layout.record.program_id
            or baseline["family"] != layout.family
            or type(baseline["accepted"]) is not bool
            or baseline["accepted"] != baseline["selected_check"]["exact"]
        ):
            raise ValueError("Mismatched admitted baseline completion")
        if baseline["accepted"]:
            nodes, edges = state_graph(state)
            if (
                reason is not None
                or not fixed_graph_preserved(nodes, edges, layout.record)
                or graph_smiles(nodes, edges, atoms) != baseline["original_smiles"]
            ):
                raise ValueError("Baseline completion does not belong to this generated state")
            checked = _check(check_product, layout, baseline["selected_smiles"])
            if not checked["exact"]:
                raise ValueError("Previously admitted completion failed full-program replay")
            return dict(
                policy_id=POLICY_ID,
                family=layout.family,
                program=layout.record.program_id,
                original_smiles=baseline["selected_smiles"],
                original_check=checked,
                reason=None,
                status="original_exact",
                proposals=[],
                source_check_calls=1,
                preserved_baseline_completion=True,
            )
        if baseline["changed"]:
            raise ValueError("A rejected baseline must retain its original product")
    return _check_completion(
        layout,
        state,
        atoms,
        check_product=check_product,
        reason=reason,
        policy_id=POLICY_ID,
        propose=lambda layout, state, atoms: propose_source_completion(
            layout, state, atoms, constraints
        ),
    )
