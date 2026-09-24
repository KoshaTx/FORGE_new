"""Exact occurrence-aware graph copying, including joined same-role precursor occurrences."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_occurrences import (
    OccurrenceTraceError,
    normalize_occurrences,
    trace_precursor_occurrences,
)
from forge.assembly.library_programs import LibraryProgramLimits
from forge.model.precursor_reuse import PrecursorReuseError
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph


@dataclass(frozen=True)
class OccurrenceReusePlan:
    record_id: str
    occurrences: tuple[tuple[int, ...], ...]
    status: str
    source_programs: int = 0
    reason: str | None = None

    def __post_init__(self):
        if not self.record_id or bool(self.occurrences) != (self.status == "qualified"):
            raise PrecursorReuseError("invalid occurrence plan")
        if self.occurrences and (
            len(self.occurrences) < 2
            or self.source_programs < 1
            or normalize_occurrences(self.occurrences) != self.occurrences
        ):
            raise PrecursorReuseError("invalid normalized occurrence correspondence")


def qualify_occurrence_reuse(record, adapter, policy, limits, maximum_matches=4096):
    def abstain(status, count=0, reason=None):
        return OccurrenceReusePlan(record.graph.structure_id, (), status, count, reason)

    accumulator = policy["accumulator_role"]
    if accumulator is None or record.program_depth < 2:
        return abstain("not_repeated")
    bound = LibraryProgramLimits(policy["maximum_steps"], **limits)
    strict = check_generated_program(
        adapter,
        record.graph.canonical_smiles,
        depth=record.program_depth,
        accumulator_role=accumulator,
        limits=bound,
    )
    if not strict.exact:
        return abstain("source_reuse_not_verified", reason=strict.reason)
    relations = set()
    inverse = {int(old): new for new, old in enumerate(record.canonical_atom_order)}
    for witness in strict.programs:
        try:
            canonical = trace_precursor_occurrences(
                adapter,
                witness["components"],
                witness["intermediate_products"],
                accumulator_role=accumulator,
                limits=bound,
                maximum_matches=maximum_matches,
            )
            relation = normalize_occurrences(
                [[inverse[i] for i in occurrence] for occurrence in canonical]
            )
        except OccurrenceTraceError as error:
            return abstain("trace_not_qualified", len(strict.programs), str(error))
        relations.add(relation)
        if len(relations) > 1:
            return abstain("witness_correspondence_conflict", len(strict.programs))
    occurrences = next(iter(relations))
    co_role = next(role for role in adapter.roles if role != accumulator)
    expected = {
        i for b in record.component_blocks if b.role == co_role for i in range(b.start, b.stop)
    }
    if len(occurrences) != record.program_depth or {i for o in occurrences for i in o} != expected:
        return abstain("source_role_partition_conflict", len(strict.programs))
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
    reference = list(occurrences[0])
    for occurrence in occurrences[1:]:
        target = list(occurrence)
        if not np.array_equal(
            record.graph.node_states[target], record.graph.node_states[reference]
        ) or not np.array_equal(edges[np.ix_(target, target)], edges[np.ix_(reference, reference)]):
            return abstain("product_occurrence_states_differ", len(strict.programs))
    return OccurrenceReusePlan(
        record.graph.structure_id, occurrences, "qualified", len(strict.programs)
    )


def copy_occurrence_graph(nodes, edges, plan: OccurrenceReusePlan, donor: int):
    if (
        plan.status != "qualified"
        or type(donor) is not int
        or not 0 <= donor < len(plan.occurrences)
    ):
        raise PrecursorReuseError("unqualified occurrence copy or invalid donor")
    if edges.shape != (len(nodes), len(nodes)) or any(
        i >= len(nodes) for o in plan.occurrences for i in o
    ):
        raise PrecursorReuseError("occurrence copy exceeds graph")
    source = list(plan.occurrences[donor])
    output_nodes, output_edges = nodes.copy(), edges.copy()
    for occurrence in plan.occurrences:
        target = list(occurrence)
        output_nodes[target] = nodes[source]
        output_edges[np.ix_(target, target)] = edges[np.ix_(source, source)]
    return output_nodes, output_edges


def propose_occurrence_reuse(*, record, plan, terminal, original, atoms, adapter, policy, limits):
    result = {"disposition": "no_proposal", "proposals": [], "additional_assembly_checks": 0}
    if plan.record_id != record.graph.structure_id:
        raise PrecursorReuseError("occurrence plan belongs to another record")
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
        raise PrecursorReuseError("valid original has no terminal state")
    nodes, edges = state_graph(terminal["state"])
    if graph_smiles(nodes, edges, atoms) != original[
        "canonical_smiles"
    ] or not fixed_graph_preserved(nodes, edges, record):
        raise PrecursorReuseError("occurrence terminal differs from original/fixed graph")
    for donor in range(len(plan.occurrences)):
        changed_nodes, changed_edges = copy_occurrence_graph(nodes, edges, plan, donor)
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
