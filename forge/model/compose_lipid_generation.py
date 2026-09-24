"""Generation and optional constrained readouts for anonymous multireaction layouts."""

import itertools

import numpy as np
import torch

from forge.model.compose_lipid_layout import collate_generated_layouts
from forge.model.compose_lipid_sampling import sample
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS
from forge.model.synthesis_program_sampling import (
    _program_conditioning,
    decode_synthesis_program_strict_argmax,
)
from forge.model.synthesis_program_training import move_tensors


def closure_allocation(layout):
    """Assign each variable closure slot to its sampled source role."""
    roles = [
        role
        for role, count in sorted(layout.variable_closures_by_role.items())
        for _ in range(count)
    ]
    output = [-1] * layout.record.graph.closure_count
    slots = np.flatnonzero(~layout.record.fixed_closure_bond_mask)
    if len(slots) != len(roles):
        raise ValueError("Sampled closure budget differs from layout slots")
    for slot, role in zip(slots, roles, strict=True):
        output[slot] = role
    return output


def constrained_readout(
    predictions,
    batch,
    layouts,
    atoms,
    *,
    rings=False,
    morphology=False,
    topology_only=False,
    atom_aware_topology=False,
):
    return decode_synthesis_program_strict_argmax(
        predictions,
        batch,
        [x.record for x in layouts],
        atoms,
        qualified_core_units=[x.core_units for x in layouts],
        confine_origin_edges=True,
        reserve_fixed_closures=True,
        origin_closure_roles=[closure_allocation(x) for x in layouts] if rings else None,
        origin_ring_sizes=[x.ring_sizes_by_role for x in layouts] if rings else None,
        exact_origin_morphology=morphology,
        topology_only=topology_only,
        atom_aware_topology=atom_aware_topology,
    )


def generate_comparison(model, layouts, atoms, node, bond, *, steps, seed, device):
    """One joint flow with paired readouts; no topology retries or chemistry reweighting."""
    batch = move_tensors(
        collate_generated_layouts(layouts, maximum_closures=model.maximum_closures), device
    )
    model.eval()
    with torch.inference_mode():
        raw, predictions = sample(
            model, batch, node, bond, steps=steps, seed=seed, return_predictions=True
        )
        result = {"raw": (raw, (None,) * len(layouts))}
        for name, kwargs in (
            ("strict", {}),
            ("rings", dict(rings=True)),
            ("morphology", dict(rings=True, morphology=True)),
        ):
            result[name] = constrained_readout(predictions, batch, layouts, atoms, **kwargs)
        # Optional one-pass factorization: fix topology before refreshing chemistry logits.
        topology, topology_reasons = constrained_readout(
            predictions, batch, layouts, atoms, rings=True, topology_only=True
        )
        state = {k: v.clone() for k, v in raw.items()}
        for k in ("parents", "closure_left", "closure_right"):
            state[k] = topology[k].to(device)
        refreshed = model(
            **state,
            t=torch.ones(len(layouts), device=device),
            **_program_conditioning(model, batch),
        )
        for k in ("parents", "closure_left", "closure_right"):
            forced = torch.full_like(refreshed[k], -1e9)
            forced.scatter_(-1, state[k].unsqueeze(-1), 1e9)
            refreshed[k] = forced
        terminal, chemistry_reasons = constrained_readout(
            refreshed, batch, layouts, atoms, rings=True
        )
        # Failed topology attempts stay failed; chemistry never rescues or replaces them.
        result["topology_then_chemistry"] = (
            terminal,
            tuple(a or b for a, b in zip(topology_reasons, chemistry_reasons, strict=True)),
        )
    return result


def _repeat_correspondences(layout):
    """Anonymous generated-slot bijections, never clean source-fragment matches."""
    r = layout.record
    groups = []
    for role, quantity in sorted(layout.quantities.items()):
        if quantity < 2:
            continue
        blocks = [b for b in r.component_blocks if b.role == role]
        if len(blocks) != quantity or len({b.atom_count for b in blocks}) != 1:
            return None, "unequal_or_unresolved_occurrences"
        aligned, signatures = [], []
        for block in blocks:
            core = [i for i in range(block.start, block.stop) if r.core_position_states[i] > 1]
            core.sort(key=lambda i: int(r.core_position_states[i]))
            labels = [int(r.core_position_states[i]) for i in core]
            if len(labels) != len(set(labels)):
                return None, "ambiguous_core_correspondence"
            signatures.append(
                [
                    (
                        int(r.core_position_states[i]),
                        int(r.graph.node_states[i]),
                        int(layout.core_units[i]),
                    )
                    for i in core
                ]
            )
            exterior = [i for i in range(block.start, block.stop) if r.core_position_states[i] == 1]
            aligned.append(core + exterior)
        if any(signature != signatures[0] for signature in signatures[1:]):
            return None, "incompatible_repeat_core_conditions"
        groups.append(aligned)
    return groups, None


def propose_generated_reuse(layout, state, atoms, *, maximum_proposals=64):
    """Explicit completion proposals from generated donors, with unchanged core connections.

    This is a separately reported completion intervention. No proposed product is
    accepted here: every valid proposal still requires the frozen full-program
    checker and the declared novelty/diversity admission policy.
    """
    if type(maximum_proposals) is not int or maximum_proposals < 1:
        raise ValueError("maximum_proposals must be a positive integer")
    groups, reason = _repeat_correspondences(layout)
    if reason or not groups:
        return dict(status=reason or "no_repeated_source_role", proposals=[])
    count = int(np.prod([len(g) for g in groups]))
    if count > maximum_proposals:
        return dict(status="proposal_budget_exceeded", proposals=[])
    nodes, edges = state_graph(state)
    if not fixed_graph_preserved(nodes, edges, layout.record):
        raise ValueError("Generated input changed fixed assembly chemistry")
    proposals = []
    for donors in itertools.product(*(range(len(g)) for g in groups)):
        changed_nodes, changed_edges = nodes.copy(), edges.copy()
        for occurrences, donor in zip(groups, donors, strict=True):
            source = occurrences[donor]
            for target in occurrences:
                changed_nodes[target] = nodes[source]
                changed_edges[np.ix_(target, target)] = edges[np.ix_(source, source)]
        reason = None
        if np.count_nonzero(changed_edges) != np.count_nonzero(edges):
            reason = "changed_edge_or_cycle_count"
        elif not fixed_graph_preserved(changed_nodes, changed_edges, layout.record):
            reason = "changed_fixed_core"
        used = np.asarray([0, *BOND_VALENCE_UNITS.tolist()])[changed_edges].sum(1)
        core = layout.core_units >= 0
        if reason is None and not np.array_equal(used[core], layout.core_units[core]):
            reason = "changed_core_saturation"
        smiles = None if reason else graph_smiles(changed_nodes, changed_edges, atoms)
        proposals.append(
            dict(
                donors=list(donors),
                smiles=smiles,
                status=reason
                or ("requires_exact_program_check" if smiles else "invalid_or_disconnected"),
                atom_edits=int(np.count_nonzero(changed_nodes != nodes)),
                bond_edits=int(np.count_nonzero(np.triu(changed_edges != edges, k=1))),
            )
        )
    return dict(status="proposals_complete", proposals=proposals)
