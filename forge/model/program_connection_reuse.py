"""Joint occurrence-interior and registry-qualified connection completion.

Connections are additional source-derived program context. Generated atom states and internal
precursor bonds remain generated. The original full-depth checker and support gates are intact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_programs import LibraryProgramLimits
from forge.model.defog_feasibility import BOND_TYPE_TO_INDEX
from forge.model.precursor_occurrence_reuse import (
    OccurrenceReusePlan,
    copy_occurrence_graph,
    qualify_occurrence_reuse,
)
from forge.model.precursor_reuse import PrecursorReuseError
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph


@dataclass(frozen=True)
class ProgramConnectionPlan:
    record_id: str
    status: str
    occurrence_plan: OccurrenceReusePlan
    connections: tuple[tuple[int, int, int], ...] = ()
    reason: str | None = None

    def __post_init__(self):
        if self.record_id != self.occurrence_plan.record_id or bool(self.connections) != (
            self.status == "qualified"
        ):
            raise PrecursorReuseError("inconsistent connection-plan identity/status")
        if self.connections and (
            self.occurrence_plan.status != "qualified"
            or len(self.connections) != len(self.occurrence_plan.occurrences)
            or len({(a, b) for a, b, _ in self.connections}) != len(self.connections)
            or any(type(i) is not int for edge in self.connections for i in edge)
        ):
            raise PrecursorReuseError("invalid qualified connection program")


def occurrence_labels(node_count: int, plan: OccurrenceReusePlan) -> np.ndarray:
    labels = np.zeros(node_count, dtype=np.int64)
    for index, occurrence in enumerate(plan.occurrences, 1):
        if any(not 0 <= i < node_count for i in occurrence):
            raise PrecursorReuseError("occurrence exceeds connection graph")
        labels[list(occurrence)] = index
    return labels


def qualify_program_connections(record, adapter, policy, limits, vocabulary):
    occurrence = qualify_occurrence_reuse(record, adapter, policy, limits)

    def result(status, connections=(), reason=None):
        return ProgramConnectionPlan(
            record.graph.structure_id, status, occurrence, connections, reason
        )

    if occurrence.status != "qualified":
        return result(occurrence.status, reason=occurrence.reason)
    labels = occurrence_labels(record.node_count, occurrence)
    if not np.any(labels == 0):
        return result("missing_accumulator_atoms")
    _, edges = state_graph(
        {
            "nodes": record.graph.node_states.tolist(),
            "parents": record.graph.parents.tolist(),
            "parent_bonds": record.graph.parent_bonds.tolist(),
            "closure_left": record.graph.closure_left.tolist(),
            "closure_right": record.graph.closure_right.tolist(),
            "closure_bonds": record.graph.closure_bonds.tolist(),
        }
    )
    template = adapter.reaction.forward.GetProductTemplate(0)
    allowed = set()
    for bond in template.GetBonds():
        left, right = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if left > 0 and right > 0:
            allowed.add(
                (
                    tuple(sorted((f"map_{left}", f"map_{right}"))),
                    BOND_TYPE_TO_INDEX[bond.GetBondType()],
                )
            )
    positions = [
        vocabulary.core_position_states[int(i)].removeprefix(record.program_id + ":")
        for i in record.core_position_states
    ]
    connections = []
    adjacency = {i: set() for i in range(len(occurrence.occurrences) + 1)}
    for a, b in zip(*np.nonzero(np.triu(edges, k=1)), strict=True):
        if labels[a] == labels[b]:
            continue
        pair = (tuple(sorted((positions[a], positions[b]))), int(edges[a, b]))
        if pair not in allowed:
            return result("connection_not_in_product_template")
        connections.append((int(a), int(b), int(edges[a, b])))
        adjacency[int(labels[a])].add(int(labels[b]))
        adjacency[int(labels[b])].add(int(labels[a]))
    reached, todo = {0}, [0]
    while todo:
        for neighbor in adjacency[todo.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                todo.append(neighbor)
    if len(reached) != len(adjacency) or len(connections) != len(occurrence.occurrences):
        return result("connection_program_not_a_tree")
    return result("qualified", tuple(connections))


def connect_generated_occurrences(nodes, edges, plan: ProgramConnectionPlan, donor: int):
    if plan.status != "qualified":
        raise PrecursorReuseError("connection program is not qualified")
    output_nodes, output_edges = copy_occurrence_graph(nodes, edges, plan.occurrence_plan, donor)
    labels = occurrence_labels(len(nodes), plan.occurrence_plan)
    output_edges[labels[:, None] != labels[None, :]] = 0
    for left, right, kind in plan.connections:
        if (
            not 0 <= left < right < len(nodes)
            or labels[left] == labels[right]
            or kind not in BOND_TYPE_TO_INDEX.values()
        ):
            raise PrecursorReuseError("invalid cross-occurrence connection")
        output_edges[left, right] = output_edges[right, left] = kind
    return output_nodes, output_edges


def propose_program_connections(
    *, record, plan, terminal, original, atoms, adapter, policy, limits
):
    result = {"disposition": "no_proposal", "proposals": [], "additional_assembly_checks": 0}
    if plan.record_id != record.graph.structure_id:
        raise PrecursorReuseError("connection plan belongs to another record")
    if plan.status != "qualified":
        result["disposition"] = plan.status
        return result
    if not original["valid_connected"]:
        result["disposition"] = "original_invalid"
        return result
    if original["assembly"]["status"] == "exact_computed_program":
        result["disposition"] = "original_exact"
        return result
    if terminal["state"] is None:
        raise PrecursorReuseError("valid original lacks terminal graph")
    nodes, edges = state_graph(terminal["state"])
    if graph_smiles(nodes, edges, atoms) != original[
        "canonical_smiles"
    ] or not fixed_graph_preserved(nodes, edges, record):
        raise PrecursorReuseError("connection terminal differs from original/fixed graph")
    for donor in range(len(plan.occurrence_plan.occurrences)):
        changed_nodes, changed_edges = connect_generated_occurrences(nodes, edges, plan, donor)
        row = {
            "donor": donor,
            "smiles": None,
            "status": "unassessed",
            "assembly": None,
            "atom_edits": int(np.count_nonzero(changed_nodes != nodes)),
            "bond_edits": int(np.count_nonzero(np.triu(changed_edges != edges, k=1))),
        }
        if np.count_nonzero(changed_edges) != np.count_nonzero(edges):
            row["status"] = "changed_cycle_count"
        elif not fixed_graph_preserved(changed_nodes, changed_edges, record):
            row["status"] = "changed_fixed_graph"
        else:
            row["smiles"] = graph_smiles(changed_nodes, changed_edges, atoms)
            if row["smiles"] is None:
                row["status"] = "invalid_or_disconnected"
            else:
                check = check_generated_program(
                    adapter,
                    row["smiles"],
                    depth=record.program_depth,
                    accumulator_role=policy["accumulator_role"],
                    limits=LibraryProgramLimits(policy["maximum_steps"], **limits),
                )
                row["assembly"], row["status"] = asdict(check), check.status
                result["additional_assembly_checks"] += 1
        result["proposals"].append(row)
    result["disposition"] = "proposals_complete"
    return result
