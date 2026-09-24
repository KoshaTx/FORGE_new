"""Bounded source-core terminal proposals on a replay of the frozen neural trajectories."""

from dataclasses import replace

import numpy as np

from forge.model.combinatorial_core_scaffold import apply_core_scaffold, extract_core
from forge.model.fixed_closure_decoding import decode_fixed_closure_reserved_argmax
from forge.model.precursor_reuse_projection import TerminalTrace
from forge.model.reaction_program_flow import collate_synthesis_program_layouts
from forge.model.synthesis_program_sampling import _fixed_state_exact_records, _terminal_smiles


def source_core_layout(record):
    """Expose only the source's declared core and the original adapter-fixed coordinates."""
    _, core = extract_core(record)
    g = record.graph
    graph = replace(
        g,
        canonical_smiles="",
        node_states=np.where(record.fixed_atom_mask, g.node_states, 0),
        parents=np.where(record.fixed_parent_bond_mask, g.parents, 0),
        parent_bonds=np.where(record.fixed_parent_bond_mask, g.parent_bonds, 0),
        closure_left=np.where(record.fixed_closure_bond_mask, g.closure_left, 0),
        closure_right=np.where(record.fixed_closure_bond_mask, g.closure_right, 0),
        closure_bonds=np.where(record.fixed_closure_bond_mask, g.closure_bonds, 0),
        edges=np.empty((0, 0), dtype=np.int8),
    )
    return apply_core_scaffold(replace(record, graph=graph), core)


class SourceCoreTrace(TerminalTrace):
    """Capture one core-constrained readout per original trajectory; return neural scores intact."""

    def __init__(self, base, records, atoms):
        super().__init__(base, records, atoms)
        plans = [source_core_layout(record) for record in records]
        self.scaffolds = [
            p if p is not None else r for (p, _), r in zip(plans, records, strict=True)
        ]
        self.qualification = [reason for _, reason in plans]
        self.core_rows = []
        self.core_terminals = []

    def forward(self, **inputs):
        predictions = super().forward(**inputs)
        if bool((inputs["t"] == 1).all()):
            offset, batch = len(self.core_rows), len(inputs["t"])
            local = self.scaffolds[offset : offset + batch]
            layout = collate_synthesis_program_layouts(
                local, maximum_closures=self.maximum_closures
            )
            terminal, reasons = decode_fixed_closure_reserved_argmax(
                predictions, layout, local, self.atoms
            )
            if not _fixed_state_exact_records(terminal, local):
                raise ValueError("source core decoder changed an immutable coordinate")
            for i, (record, reason) in enumerate(zip(local, reasons, strict=True)):
                reason = self.qualification[offset + i] or reason
                smiles = (
                    None
                    if reason
                    else _terminal_smiles(
                        terminal, i, record.node_count, record.graph.closure_count, self.atoms
                    )
                )
                self.core_rows.append(
                    {
                        "sample_index": offset + i,
                        "program_id": record.program_id,
                        "layout_record_id": record.graph.structure_id,
                        "canonical_smiles": smiles,
                        "valid": smiles is not None,
                        "constraint_abstention_reason": reason,
                    }
                )
                self.core_terminals.append(
                    {
                        "reason": reason,
                        "state": (
                            None
                            if reason
                            else {
                                key: value[
                                    i,
                                    : (
                                        record.graph.closure_count
                                        if key.startswith("closure")
                                        else record.node_count
                                    ),
                                ]
                                .cpu()
                                .tolist()
                                for key, value in terminal.items()
                            }
                        ),
                    }
                )
        return predictions
