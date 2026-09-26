"""Bounded variable-closure proposals on an unchanged generated spanning tree.

Ring lengths come only from the request. This repairs a conditioning mismatch,
not a chemistry validity rule. The caller must independently replay source and
design checks and retain the original candidate and diversity floors.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
from rdkit import rdBase

from forge.model.compose_lipid_generation import _repeat_correspondences
from forge.model.compose_lipid_layout import tree_ring_size
from forge.model.compose_lipid_structural_audit import audit_layout_rings
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS
from forge.model.synthesis_program_sampling import _atom_capacity_table


def validate_unchanged_tree(layout, original, proposed, atoms):
    """Reject changed morphology, fixed chemistry, duplicate or cross-origin edges."""
    for key in ("nodes", "parents", "parent_bonds", "closure_bonds"):
        if list(original[key]) != list(proposed[key]):
            raise ValueError("Closure repair changed " + key)
    if any(len(proposed[k]) != len(original[k]) for k in ("closure_left", "closure_right")):
        raise ValueError("Closure repair changed closure count")
    record = layout.record
    nodes, edges = state_graph(proposed)
    if len(nodes) != record.node_count or not fixed_graph_preserved(nodes, edges, record):
        raise ValueError("Closure repair changed requested size or fixed graph")
    occupied = {tuple(sorted((i, int(p)))) for i, p in enumerate(proposed["parents"]) if i}
    for slot, (a, b) in enumerate(
        zip(proposed["closure_left"], proposed["closure_right"], strict=True)
    ):
        a, b = int(a), int(b)
        pair = tuple(sorted((a, b)))
        if a == b or pair in occupied:
            raise ValueError("Duplicate or self closure edge")
        occupied.add(pair)
        changed = (a, b) != (original["closure_left"][slot], original["closure_right"][slot])
        if not changed:
            continue
        if record.fixed_closure_bond_mask[slot]:
            raise ValueError("Changed fixed closure")
        blocks = [
            block
            for block in record.component_blocks
            if a in range(block.start, block.stop) and b in range(block.start, block.stop)
        ]
        if len(blocks) != 1:
            raise ValueError("Cross-origin or cross-component closure")
        if record.core_position_states[a] > 1 and record.core_position_states[b] > 1:
            raise ValueError("Changed core-core closure")
        if tree_ring_size(proposed["parents"], a, b) not in layout.ring_sizes_by_role.get(
            blocks[0].role_state, ()
        ):
            raise ValueError("Closure differs from requested ring sizes")
    units = np.asarray([0, *BOND_VALENCE_UNITS.tolist()])[edges].sum(1)
    if np.any(units > _atom_capacity_table(atoms)[nodes]):
        raise ValueError("Closure exceeds atom valence capacity")
    core = np.asarray(layout.core_units) >= 0
    if not np.array_equal(units[core], np.asarray(layout.core_units)[core]):
        raise ValueError("Closure changes fixed core saturation")
    return nodes, edges


def propose_closure_endpoints(
    layout,
    state,
    predictions: Mapping,
    atoms,
    *,
    maximum_proposals=2,
    beam_width=16,
    maximum_pairs_per_group=4096,
):
    """Rank at most two proposals without chemistry calls, retries, or replacing atoms."""
    if any(
        type(v) is not int or v < 1
        for v in (maximum_proposals, beam_width, maximum_pairs_per_group)
    ):
        raise ValueError("Positive integer search bounds required")
    original = {k: list(map(int, v)) for k, v in state.items()}
    nodes, edges = state_graph(original)
    if len(nodes) != layout.record.node_count or not fixed_graph_preserved(
        nodes, edges, layout.record
    ):
        raise ValueError("Invalid source-fixed input")
    before = audit_layout_rings(edges, layout, tree_state=original, tree_basis="saved_parent_tree")
    report = {
        "proposals": [],
        "before": before,
        "status": "unchanged",
        "groups": [],
        "costs": {"endpoint_pairs_considered": 0, "states_checked": 0, "graphs_sanitized": 0},
        "bounds": {
            "maximum_proposals": maximum_proposals,
            "beam_width": beam_width,
            "maximum_pairs_per_group": maximum_pairs_per_group,
        },
        "search_censored": False,
    }
    if not before["fundamental_size_mismatch"] and not before["cycle_allocation_mismatch"]:
        report["status"] = "already_matches_request_noop"
        return report
    if before["cycle_allocation_mismatch"] or before["unexpected_cross_origin_edges"]:
        report["status"] = "role_cycle_mismatch_outside_endpoint_only_scope"
        return report
    logits = {k: np.asarray(predictions[k]) for k in ("closure_left", "closure_right")}
    slots = len(original["closure_left"])
    if any(
        v.ndim != 2 or v.shape[0] < slots or v.shape[1] < len(nodes) or not np.isfinite(v).all()
        for v in logits.values()
    ):
        raise ValueError("Finite full-support closure logits required")
    repeat, reason = _repeat_correspondences(layout)
    if reason:
        report["status"] = "repeat_mapping_abstention:" + reason
        return report
    repeats = {frozenset(order): group for group in repeat for order in group}
    used = set()
    groups = []
    for slot, (a, b) in enumerate(
        zip(original["closure_left"], original["closure_right"], strict=True)
    ):
        if slot in used or layout.record.fixed_closure_bond_mask[slot]:
            continue
        block = next(
            (
                b0
                for b0 in layout.record.component_blocks
                if a in range(b0.start, b0.stop) and b in range(b0.start, b0.stop)
            ),
            None,
        )
        if block is None or tree_ring_size(
            original["parents"], a, b
        ) in layout.ring_sizes_by_role.get(block.role_state, ()):
            continue
        members = list(range(block.start, block.stop))
        repeat_group = repeats.get(frozenset(members), [members])
        source = next(order for order in repeat_group if set(order) == set(members))
        replacements = []
        for target in repeat_group:
            if not np.array_equal(nodes[source], nodes[target]) or not np.array_equal(
                edges[np.ix_(source, source)], edges[np.ix_(target, target)]
            ):
                report["status"] = "repeat_graph_alignment_abstention"
                return report
            mapping = dict(zip(source, target, strict=True))
            equivalent = [
                j
                for j, (x, y) in enumerate(
                    zip(original["closure_left"], original["closure_right"], strict=True)
                )
                if {x, y} == {mapping[a], mapping[b]}
            ]
            if len(equivalent) != 1:
                report["status"] = "repeat_closure_basis_abstention"
                return report
            replacements.append((equivalent[0], mapping))
        used.update(j for j, _ in replacements)
        groups.append((members, replacements, layout.ring_sizes_by_role.get(block.role_state, ())))
    beam = [(0.0, original)]
    for members, replacements, allowed in groups:
        options = []
        for a in members:
            for b in members:
                if a >= b or tree_ring_size(original["parents"], a, b) not in allowed:
                    continue
                if edges[a, b] or (
                    layout.record.core_position_states[a] > 1
                    and layout.record.core_position_states[b] > 1
                ):
                    continue
                score = sum(
                    max(
                        float(logits["closure_left"][j, m[a]] + logits["closure_right"][j, m[b]]),
                        float(logits["closure_left"][j, m[b]] + logits["closure_right"][j, m[a]]),
                    )
                    for j, m in replacements
                )
                options.append((score, a, b))
        options.sort(key=lambda x: (-x[0], x[1], x[2]))
        report["search_censored"] |= len(options) > maximum_pairs_per_group
        options = options[:maximum_pairs_per_group]
        report["groups"].append(
            {
                "slots": [j for j, _ in replacements],
                "role_members": members,
                "requested_sizes": list(allowed),
                "pairs": len(options),
            }
        )
        candidates = []
        for score, previous in beam:
            for added, a, b in options:
                report["costs"]["endpoint_pairs_considered"] += 1
                trial = copy.deepcopy(previous)
                for j, m in replacements:
                    pair = (m[a], m[b])
                    if float(
                        logits["closure_left"][j, pair[1]] + logits["closure_right"][j, pair[0]]
                    ) > float(
                        logits["closure_left"][j, pair[0]] + logits["closure_right"][j, pair[1]]
                    ):
                        pair = pair[::-1]
                    trial["closure_left"][j], trial["closure_right"][j] = pair
                report["costs"]["states_checked"] += 1
                try:
                    tn, te = validate_unchanged_tree(layout, original, trial, atoms)
                except ValueError:
                    continue
                report["costs"]["graphs_sanitized"] += 1
                with rdBase.BlockLogs():
                    smiles = graph_smiles(tn, te, atoms)
                if smiles is not None:
                    candidates.append((score + added, smiles, trial))
        unique = {}
        for item in sorted(candidates, key=lambda x: (-x[0], x[1])):
            unique.setdefault(item[1], item)
        beam = [(v[0], v[2]) for v in list(unique.values())[:beam_width]]
        if not beam:
            report["status"] = "no_valid_bounded_endpoint_assignment"
            return report
    for score, trial in beam[:maximum_proposals]:
        tn, te = validate_unchanged_tree(layout, original, trial, atoms)
        after = audit_layout_rings(
            te, layout, tree_state=trial, tree_basis="endpoint_repair_same_parent_tree"
        )
        if after["fundamental_size_mismatch"] or after["cycle_allocation_mismatch"]:
            continue
        report["proposals"].append(
            {
                "nodes": tn.tolist(),
                "edges": te.tolist(),
                "tree_state": trial,
                "tree_basis": "endpoint_repair_same_parent_tree",
                "smiles": graph_smiles(tn, te, atoms),
                "closure_logit_score": score,
                "ring": after,
                "model_likelihood": None,
                "trajectory_log_probability": None,
                "route_verdict": None,
            }
        )
    report["status"] = (
        "proposed" if report["proposals"] else "no_complete_requested_ring_assignment"
    )
    return report
