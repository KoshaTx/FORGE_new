"""Bounded graph-copy completion of an already generated repeated-precursor product.

This is a separate terminal proposal stage, not a change to the frozen flow or assembly gate.
Only generated atom/bond states are copied. Qualified source correspondence supplies addresses.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from rdkit import Chem
from torch import nn

from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_programs import LibraryProgramLimits
from forge.model.defog_feasibility import graph_to_molecule
from forge.model.precursor_reuse import PrecursorReuseError, PrecursorReusePlan
from forge.model.sparse_topology_feasibility import INDEX_TO_DENSE_BOND
from forge.model.synthesis_program_sampling import _strict_terminal_record


class TerminalTrace(nn.Module):
    """Capture the unchanged deterministic terminal readout for independent completion."""

    def __init__(self, base: Any, records: list, atoms: tuple) -> None:
        super().__init__()
        self.base = base
        self.maximum_closures = base.maximum_closures
        self.records, self.atoms = records, atoms
        self.terminals: list[dict] = []

    def forward(self, **inputs: Any) -> dict:
        predictions = self.base(**inputs)
        if bool((inputs["t"] == 1).all()):
            offset = len(self.terminals)
            batch = len(inputs["t"])
            if offset + batch > len(self.records):
                raise PrecursorReuseError("terminal trace received extra batches")
            arrays = {k: v.detach().cpu().numpy() for k, v in predictions.items()}
            for i, record in enumerate(self.records[offset : offset + batch]):
                state, reason = _strict_terminal_record(arrays, i, record, self.atoms)
                self.terminals.append(
                    {
                        "state": (
                            None if state is None else {k: v.tolist() for k, v in state.items()}
                        ),
                        "reason": reason,
                    }
                )
        return predictions


def state_graph(state: dict) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(state["nodes"], dtype=np.int64)
    edges = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
    for child in range(1, len(nodes)):
        parent = state["parents"][child]
        if type(parent) is not int or not 0 <= parent < child:
            raise PrecursorReuseError("invalid generated parent")
        edges[child, parent] = edges[parent, child] = INDEX_TO_DENSE_BOND[
            state["parent_bonds"][child]
        ]
    for a, b, kind in zip(
        state["closure_left"], state["closure_right"], state["closure_bonds"], strict=True
    ):
        if not 0 <= a < len(nodes) or not 0 <= b < len(nodes) or a == b or edges[a, b]:
            raise PrecursorReuseError("invalid generated closure")
        edges[a, b] = edges[b, a] = INDEX_TO_DENSE_BOND[kind]
    return nodes, edges


def graph_smiles(nodes: np.ndarray, edges: np.ndarray, atoms: tuple) -> str | None:
    try:
        molecule = graph_to_molecule(nodes, edges, atoms)
        smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        reparsed = Chem.MolFromSmiles(smiles)
        if reparsed is None or len(Chem.GetMolFrags(reparsed)) != 1:
            return None
        return Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=False)
    except (ValueError, RuntimeError, KeyError):
        return None


def copy_generated_component(
    nodes: np.ndarray, edges: np.ndarray, record: Any, plan: PrecursorReusePlan, donor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Copy one generated internal graph via the declared equality relation; keep cross edges."""
    if (
        plan.record_id != record.graph.structure_id
        or len(plan.groups) != len(nodes)
        or plan.status != "qualified"
    ):
        raise PrecursorReuseError("unqualified or mismatched graph-copy relation")
    blocks = [b for b in record.component_blocks if any(plan.groups[b.start : b.stop])]
    if type(donor) is not int or not 0 <= donor < len(blocks):
        raise PrecursorReuseError("invalid donor occurrence")
    donor_block = blocks[donor]
    source = {plan.groups[i]: i for i in range(donor_block.start, donor_block.stop)}
    if 0 in source or len(source) != donor_block.atom_count:
        raise PrecursorReuseError("donor correspondence is incomplete")
    output_nodes, output_edges = nodes.copy(), edges.copy()
    for block in blocks:
        targets = list(range(block.start, block.stop))
        labels = [plan.groups[i] for i in targets]
        if set(labels) != set(source) or len(labels) != len(set(labels)):
            raise PrecursorReuseError("copy correspondence is not bijective")
        origins = [source[g] for g in labels]
        output_nodes[targets] = nodes[origins]
        output_edges[np.ix_(targets, targets)] = edges[np.ix_(origins, origins)]
    return output_nodes, output_edges


def fixed_graph_preserved(nodes: np.ndarray, edges: np.ndarray, record: Any) -> bool:
    if not np.array_equal(
        nodes[record.fixed_atom_mask], record.graph.node_states[record.fixed_atom_mask]
    ):
        return False
    for child in np.flatnonzero(record.fixed_parent_bond_mask):
        if (
            edges[child, record.graph.parents[child]]
            != INDEX_TO_DENSE_BOND[int(record.graph.parent_bonds[child])]
        ):
            return False
    for slot in np.flatnonzero(record.fixed_closure_bond_mask):
        if (
            edges[record.graph.closure_left[slot], record.graph.closure_right[slot]]
            != INDEX_TO_DENSE_BOND[int(record.graph.closure_bonds[slot])]
        ):
            return False
    return True


def complete_reuse(
    *,
    record: Any,
    plan: PrecursorReusePlan,
    terminal: dict,
    original: dict,
    atoms: tuple,
    adapter: Any,
    policy: dict,
    limits: dict,
) -> dict:
    """One proposal per generated occurrence; exact, size-preserving candidates or original.

    Existing exact programs are retained. Candidate choice uses only graph-edit count and a
    deterministic graph tie-break, with no reference novelty, training identity or batch history.
    Every rejected proposal remains in the return value. No molecular generation retry occurs.
    """
    result = {
        "selected_smiles": original["canonical_smiles"],
        "selected_donor": None,
        "disposition": "retained_original",
        "proposals": [],
        "additional_assembly_checks": 0,
    }
    if plan.status != "qualified":
        result["disposition"] = "relation_" + plan.status
        return result
    if not original["valid_connected"] or terminal["state"] is None:
        result["disposition"] = "original_invalid"
        return result
    if original["assembly"]["status"] == "exact_computed_program":
        result["disposition"] = "original_exact"
        return result
    nodes, edges = state_graph(terminal["state"])
    if graph_smiles(nodes, edges, atoms) != original[
        "canonical_smiles"
    ] or not fixed_graph_preserved(nodes, edges, record):
        raise PrecursorReuseError("terminal trace differs from original/fixed chemistry")
    blocks = [b for b in record.component_blocks if any(plan.groups[b.start : b.stop])]
    for donor in range(len(blocks)):
        proposed_nodes, proposed_edges = copy_generated_component(nodes, edges, record, plan, donor)
        row = {
            "donor": donor,
            "smiles": None,
            "status": "unassessed",
            "assembly": None,
            "atom_edits": int(np.count_nonzero(nodes != proposed_nodes)),
            "bond_edits": int(np.count_nonzero(np.triu(edges != proposed_edges, k=1))),
        }
        if np.count_nonzero(proposed_edges) != np.count_nonzero(edges):
            row["status"] = "changed_cycle_count"
        elif not fixed_graph_preserved(proposed_nodes, proposed_edges, record):
            row["status"] = "changed_fixed_graph"
        else:
            row["smiles"] = graph_smiles(proposed_nodes, proposed_edges, atoms)
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
                row["assembly"] = asdict(check)
                row["status"] = check.status
                result["additional_assembly_checks"] += 1
        result["proposals"].append(row)
    eligible = [p for p in result["proposals"] if p["status"] == "exact_computed_program"]
    if eligible:
        chosen = min(
            eligible, key=lambda p: (p["atom_edits"] + p["bond_edits"], p["smiles"], p["donor"])
        )
        result.update(
            selected_smiles=chosen["smiles"],
            selected_donor=chosen["donor"],
            disposition="exact_graph_copy_completion",
        )
    return result
