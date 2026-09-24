"""Template-symmetric internal arms from generated atoms, with fixed size and core."""

import itertools
import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
from rdkit import Chem

from forge.model.compose_lipid_family_rules import reserve_ordered_core_attachments
from forge.model.compose_lipid_generation import constrained_readout
from forge.model.compose_lipid_layout import collate_generated_layouts
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS


def template_automorphisms(template, maximum=256):
    """Atom-query and concrete-bond automorphisms with mapping labels removed.

    Atom predicates must be identical. Implicit and explicit single bonds share
    their concrete bond type. This proposes a symmetry restriction; independent
    full replay still checks every actual orientation of the complete precursor.
    """
    query = Chem.Mol(template)
    groups = defaultdict(list)
    for atom in query.GetAtoms():
        atom.SetAtomMapNum(0)
        groups[atom.GetSmarts()].append(atom.GetIdx())
    if math.prod(math.factorial(len(g)) for g in groups.values()) > maximum:
        return ()
    choices = [list(itertools.permutations(g)) for g in groups.values()]
    bonds = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondType()
        for b in query.GetBonds()
    }
    result = []
    for values in itertools.product(*choices):
        mapping = dict(
            zip(
                itertools.chain.from_iterable(groups.values()),
                itertools.chain.from_iterable(values),
                strict=True,
            )
        )
        transformed = {
            tuple(sorted((mapping[a], mapping[b]))): value for (a, b), value in bonds.items()
        }
        if transformed == bonds:
            result.append(tuple(mapping[i] for i in range(query.GetNumAtoms())))
    return tuple(result)


def construct_symmetric_role_arms(
    layout, predictions, atoms, adapter, role, core_vocabulary, *, atom_aware_topology=False
):
    """Two tree-arm readouts only when registry-template symmetry supports the ports.

    Internal arm slots stay within their source component. Other ring/size
    supports abstain explicitly. Every returned graph needs full source replay.
    """
    template = adapter.reaction.forward.GetReactantTemplate(adapter.roles.index(role))
    automorphisms = template_automorphisms(template)
    maps = {a.GetAtomMapNum(): a.GetIdx() for a in template.GetAtoms()}
    record, graph = layout.record, layout.record.graph
    used = np.zeros(record.node_count, dtype=np.int64)
    for c in np.flatnonzero(record.fixed_parent_bond_mask):
        used[[c, graph.parents[c]]] += int(BOND_VALENCE_UNITS[graph.parent_bonds[c]])
    for s in np.flatnonzero(record.fixed_closure_bond_mask):
        used[[graph.closure_left[s], graph.closure_right[s]]] += int(
            BOND_VALENCE_UNITS[graph.closure_bonds[s]]
        )
    parents, bonds, fixed = (
        graph.parents.copy(),
        graph.parent_bonds.copy(),
        record.fixed_parent_bond_mask.copy(),
    )
    blocks, arm_pairs = [], []
    for block in record.component_blocks:
        if block.role != role:
            blocks.append(block)
            continue
        if layout.variable_closures_by_role[block.role_state]:
            return dict(status="symmetric_arm_cycles_not_supported", proposals=[])
        ports = [
            i
            for i in range(block.start, block.stop)
            if layout.core_units[i] >= 0 and layout.core_units[i] - used[i] == 2
        ]
        if len(ports) != 2:
            return dict(status="requires_two_single_bond_ports", proposals=[])
        pmap = [
            int(core_vocabulary[int(record.core_position_states[i])].rsplit(":map_", 1)[1])
            for i in ports
        ]
        if not any(
            maps.get(pmap[0]) is not None and m[maps[pmap[0]]] == maps.get(pmap[1])
            for m in automorphisms
        ):
            return dict(status="ports_not_template_symmetric", proposals=[])
        runs = []
        for exterior, group in itertools.groupby(
            range(block.start, block.stop), key=lambda i: record.core_position_states[i] == 1
        ):
            slots = list(group)
            blocks.append(replace(block, start=slots[0], stop=slots[-1] + 1))
            if exterior:
                runs.append(slots)
        if len(runs) != 2 or len(runs[0]) != len(runs[1]):
            return dict(status="unequal_or_unresolved_internal_arm_slots", proposals=[])
        remaining = set(ports)
        for run in runs:
            eligible = [i for i in remaining if i < run[0]]
            if not eligible:
                return dict(status="internal_arm_order_not_supported", proposals=[])
            parent = max(eligible, key=lambda i: (float(predictions["parents"][run[0], i]), -i))
            parents[run[0]], bonds[run[0]], fixed[run[0]] = parent, 0, True
            remaining.remove(parent)
        arm_pairs.append(runs)
    if not arm_pairs:
        return dict(status="no_source_role", proposals=[])
    pinned = replace(
        layout,
        record=replace(
            record,
            graph=replace(graph, parents=parents, parent_bonds=bonds),
            fixed_parent_bond_mask=fixed,
        ),
    )
    pinned, reason = reserve_ordered_core_attachments(
        pinned, predictions["parents"][: record.node_count, : record.node_count]
    )
    if reason:
        return dict(status=reason, proposals=[])
    temporary = replace(pinned, record=replace(pinned.record, component_blocks=tuple(blocks)))
    import torch

    pred = {k: torch.as_tensor(v[None]) for k, v in predictions.items()}
    states, reasons = constrained_readout(
        pred,
        collate_generated_layouts([layout], maximum_closures=12),
        [temporary],
        atoms,
        atom_aware_topology=atom_aware_topology,
    )
    if reasons[0]:
        return dict(status=reasons[0], proposals=[])
    state = {
        k: v[0, : (graph.closure_count if k.startswith("closure") else record.node_count)].numpy()
        for k, v in states.items()
    }
    proposals = []
    for donor in (0, 1):
        new = {k: v.copy() for k, v in state.items()}
        for runs in arm_pairs:
            source = runs[donor]
            for target in runs:
                new["nodes"][target] = state["nodes"][source]
                correspondence = dict(zip(source, target, strict=True))
                for src, dst in zip(source[1:], target[1:], strict=True):
                    new["parents"][dst] = correspondence[int(state["parents"][src])]
                    new["parent_bonds"][dst] = state["parent_bonds"][src]
        serial = {k: v.tolist() for k, v in new.items()}
        n, e = state_graph(serial)
        assert fixed_graph_preserved(n, e, record)
        assert np.count_nonzero(np.triu(e)) == record.node_count - 1 + graph.closure_count
        proposals.append(
            dict(
                donor=donor,
                state=serial,
                nodes=n.tolist(),
                edges=e.tolist(),
                smiles=graph_smiles(n, e, atoms),
            )
        )
    return dict(status="requires_full_source_check", proposals=proposals)
