"""Registry-derived terminal scaffolds, with model-generated equal-size arms.

This is a constructive readout intervention, not an annotation of the learned
flow. Temporary decoding regions never become precursor-origin labels. The
original layout, atom count, assembly core and per-role cycle counts stay fixed.
Every proposal requires the independent, complete source executor afterwards.
"""

from dataclasses import replace

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from forge.model.compose_lipid_generation import closure_allocation
from forge.model.defog_feasibility import AtomState
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.sparse_topology_feasibility import SPARSE_BOND_TO_INDEX
from forge.model.synthesis_program_sampling import _strict_terminal_record


def _walk(molecule, root, allowed):
    order, parent = [root], {root: None}
    for node in order:
        for neighbor in sorted(a.GetIdx() for a in molecule.GetAtomWithIdx(node).GetNeighbors()):
            if neighbor in allowed and neighbor not in parent:
                parent[neighbor] = node
                order.append(neighbor)
    if set(order) != set(allowed):
        raise ValueError("Registered scaffold region is disconnected")
    return order, parent


def _compile(rule, reaction, role):
    """Materialize and validate the finite registered query, not a source example."""
    query = Chem.MolFromSmarts(rule["core_smarts"])
    if query is None or not rule["identical_detached_arms"]:
        raise ValueError("Construction requires a registered identical-arm scaffold")
    builder = Chem.RWMol()
    for atom in query.GetAtoms():
        if atom.GetAtomicNum() <= 0:
            raise ValueError("Scaffold query has no concrete atomic number")
        value = Chem.Atom(atom.GetAtomicNum())
        value.SetFormalCharge(atom.GetFormalCharge())
        value.SetIsAromatic(atom.GetIsAromatic())
        builder.AddAtom(value)
    for bond in query.GetBonds():
        builder.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    molecule = builder.GetMol()
    Chem.SanitizeMol(molecule)
    # Never silently approximate a query that its concrete graph does not satisfy.
    if tuple(range(query.GetNumAtoms())) not in molecule.GetSubstructMatches(
        query, uniquify=False, maxMatches=256
    ):
        raise ValueError("Materialized scaffold does not satisfy its registered query")
    maps = {a.GetAtomMapNum(): a.GetIdx() for a in query.GetAtoms() if a.GetAtomMapNum()}
    anchor = maps[rule["anchor_map"]]
    transform = rdChemReactions.ReactionFromSmarts(reaction["atom_mapped_reaction_smarts"])
    roles = [r["name"] for r in reaction["reactant_roles"]]
    template = transform.GetReactantTemplate(roles.index(role))
    matches = molecule.GetSubstructMatches(template, uniquify=False, maxMatches=256)
    product_maps = {
        a.GetAtomMapNum() for product in transform.GetProducts() for a in product.GetAtoms()
    }
    correspondences = set()
    for match in matches:
        retained = tuple(
            match[a.GetIdx()] for a in template.GetAtoms() if a.GetAtomMapNum() in product_maps
        )
        removed = tuple(
            match[a.GetIdx()] for a in template.GetAtoms() if a.GetAtomMapNum() not in product_maps
        )
        correspondences.add((retained, removed))
    if len(matches) >= 256 or len(correspondences) != 1:
        raise ValueError("Scaffold has an ambiguous reaction-site correspondence")
    retained, removed = next(iter(correspondences))
    if retained != (anchor,):
        raise ValueError("Construction supports one retained scaffold anchor")
    ports = [(maps[a], maps[b]) for a, b in rule["cut_bond_maps"]]
    if len(set(ports)) != len(ports) or len(ports) < 2:
        raise ValueError("Scaffold requires distinct registered arm cuts")
    cut = Chem.RWMol(molecule)
    for inner, outer in ports:
        cut.RemoveBond(inner, outer)
    fragments = [set(f) for f in Chem.GetMolFrags(cut.GetMol(), sanitizeFrags=False)]
    scaffold = next(f for f in fragments if anchor in f) - set(removed)
    arms = [next(f for f in fragments if outer in f) for _, outer in ports]
    if (
        any(anchor in arm for arm in arms)
        or len(set.union(scaffold, *arms, set(removed))) != (molecule.GetNumAtoms())
        or sum(map(len, arms)) != len(set.union(*arms))
    ):
        raise ValueError("Scaffold cuts do not partition the registered query")
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    return molecule, anchor, scaffold, ports, arms


def construct_scaffold_proposals(
    layout, predictions, atoms, reaction, *, atom_aware_topology=False
):
    """At most one deterministic readout per registry rule; no retries or fragments.

    Currently supports tree arms and a scaffold whose only surviving reaction
    atom is its anchor. Other support is explicitly abstained, never resized.
    Predictions are unbatched NumPy arrays from the unchanged flow trajectory.
    """
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("Scaffold construction requires finite predictions")
    contract = reaction["precursor_scaffolds"]
    role = contract["precursor_role"]
    record = layout.record
    blocks = [b for b in record.component_blocks if b.role == role]
    if len(blocks) != 1:
        return dict(status="scaffold_role_not_one_block", proposals=[])
    block = blocks[0]
    core = np.flatnonzero(record.fixed_atom_mask[block.start : block.stop]) + block.start
    if core.tolist() != [block.start]:
        return dict(status="scaffold_requires_one_leading_core_anchor", proposals=[])
    results = []
    for rule in contract["scaffolds"]:
        molecule, anchor, scaffold, ports, fragments = _compile(rule, reaction, role)
        order, tree = _walk(molecule, anchor, scaffold)
        remaining = block.atom_count - len(order)
        arm_count = len(ports)
        scaffold_edges = sum(
            b.GetBeginAtomIdx() in scaffold and b.GetEndAtomIdx() in scaffold
            for b in molecule.GetBonds()
        )
        cycles = scaffold_edges - len(scaffold) + 1
        entry = dict(scaffold_id=rule["id"], smiles=None)
        if remaining < 0 or remaining % arm_count:
            results.append(dict(entry, status="incompatible_equal_arm_atom_budget"))
            continue
        arm_size = remaining // arm_count
        if any(arm_size < len(fragment) for fragment in fragments):
            results.append(dict(entry, status="insufficient_arm_atom_budget"))
            continue
        if layout.variable_closures_by_role[block.role_state] != cycles:
            results.append(dict(entry, status="non_tree_arm_cycle_budget_not_supported"))
            continue
        # Decoding regions confine new edges; these are not new precursor roles.
        regions = [replace(block, stop=block.start + len(order))]
        mapping = {old: block.start + i for i, old in enumerate(order)}
        trees = [tree]
        arm_slots = []
        for j, ((inner, outer), fragment) in enumerate(zip(ports, fragments, strict=True)):
            start = regions[0].stop + j * arm_size
            regions.append(replace(block, start=start, stop=start + arm_size))
            local, parents = _walk(molecule, outer, fragment)
            parents[outer] = inner
            mapping.update({old: start + i for i, old in enumerate(local)})
            trees.append(parents)
            arm_slots.append(list(range(start, start + arm_size)))
        graph = record.graph
        nodes, parents, bonds = (
            x.copy() for x in (graph.node_states, graph.parents, graph.parent_bonds)
        )
        left, right, cbonds = (
            x.copy() for x in (graph.closure_left, graph.closure_right, graph.closure_bonds)
        )
        fixed_atoms = record.fixed_atom_mask.copy()
        fixed_parents = record.fixed_parent_bond_mask.copy()
        fixed_closures = record.fixed_closure_bond_mask.copy()
        atom_lookup = {atom: i for i, atom in enumerate(atoms)}
        for old, new in mapping.items():
            if old == anchor:
                continue  # Product anchor chemistry belongs to the immutable layout.
            atom = molecule.GetAtomWithIdx(old)
            state = AtomState(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetIsAromatic())
            if state not in atom_lookup:
                raise ValueError("Registered scaffold atom is outside the admitted vocabulary")
            nodes[new], fixed_atoms[new] = atom_lookup[state], True
        tree_edges = set()
        for local in trees:
            for child, parent in local.items():
                if parent is None:
                    continue
                a, b = mapping[parent], mapping[child]
                if a >= b:
                    raise ValueError("Scaffold tree violates ordered atom support")
                bond = molecule.GetBondBetweenAtoms(parent, child)
                parents[b], bonds[b], fixed_parents[b] = (
                    a,
                    SPARSE_BOND_TO_INDEX[bond.GetBondType()],
                    True,
                )
                tree_edges.add(tuple(sorted((parent, child))))
        allocation = closure_allocation(layout)
        free = [i for i, value in enumerate(allocation) if value == block.role_state]
        residual = [
            b
            for b in molecule.GetBonds()
            if b.GetBeginAtomIdx() in mapping
            and b.GetEndAtomIdx() in mapping
            and tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) not in tree_edges
        ]
        if len(residual) != len(free):
            raise ValueError("Registry scaffold and reserved cycle budget disagree")
        for slot, bond in zip(free, residual, strict=True):
            left[slot], right[slot] = sorted(
                (mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()])
            )
            cbonds[slot], fixed_closures[slot], allocation[slot] = (
                SPARSE_BOND_TO_INDEX[bond.GetBondType()],
                True,
                -1,
            )
        changed = replace(
            record,
            graph=replace(
                graph,
                node_states=nodes,
                parents=parents,
                parent_bonds=bonds,
                closure_left=left,
                closure_right=right,
                closure_bonds=cbonds,
            ),
            fixed_atom_mask=fixed_atoms,
            fixed_parent_bond_mask=fixed_parents,
            fixed_closure_bond_mask=fixed_closures,
            component_blocks=tuple(
                region
                for b in record.component_blocks
                for region in (regions if b == block else [b])
            ),
        )
        # Only the donor is stochastic/model generated. Replicas get the same
        # readout logits in corresponding slots, then exact graph equality below.
        pred = {key: value[None].copy() for key, value in predictions.items()}
        allowed = np.array(
            [
                Chem.GetPeriodicTable().GetAtomicNumber(a.symbol) in rule["arm_atomic_numbers"]
                and (rule["arm_aromatic_atoms_allowed"] or not a.aromatic)
                for a in atoms
            ]
        )
        for slots in arm_slots:
            pred["nodes"][0, slots] = np.where(allowed, pred["nodes"][0, slots], -1e9)
        donor = arm_slots[0]
        for target in arm_slots[1:]:
            for key in ("nodes", "parent_bonds"):
                pred[key][0, target] = pred[key][0, donor]
            pred["parents"][0][np.ix_(target, target)] = pred["parents"][0][np.ix_(donor, donor)]
        state, reason = _strict_terminal_record(
            pred,
            0,
            changed,
            atoms,
            qualified_core_units=layout.core_units,
            confine_origin_edges=True,
            reserve_fixed_closures=True,
            origin_closure_roles=allocation,
            origin_ring_sizes=layout.ring_sizes_by_role,
            atom_aware_topology=atom_aware_topology,
        )
        if state is None:
            results.append(dict(entry, status=reason))
            continue
        nodes, edges = state_graph({key: value.tolist() for key, value in state.items()})
        donor = arm_slots[0]
        for target in arm_slots[1:]:
            nodes[target] = nodes[donor]
            edges[np.ix_(target, target)] = edges[np.ix_(donor, donor)]
        if not fixed_graph_preserved(nodes, edges, record):
            raise ValueError("Constructive scaffold changed the immutable reaction core")
        if np.count_nonzero(np.triu(edges)) != record.node_count - 1 + graph.closure_count:
            raise ValueError("Constructive scaffold changed the cycle count")
        smiles = graph_smiles(nodes, edges, atoms)
        results.append(
            dict(
                entry,
                smiles=smiles,
                status="requires_full_source_check" if smiles else "invalid_product",
                nodes=nodes.tolist(),
                edges=edges.tolist(),
                arm_size=arm_size,
                arm_count=arm_count,
                donor=0,
            )
        )
    return dict(status="bounded_registry_scaffold_readouts", proposals=results)
