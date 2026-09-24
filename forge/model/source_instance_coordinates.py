"""Source precursor quantities are independent of connected product-origin regions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.model.reaction_program_flow import collate_synthesis_program_records
from forge.model.synthesis_program_graph import (
    SynthesisProgramGraphError,
    SynthesisProgramGraphRecord,
)


@dataclass(frozen=True)
class SourceInstanceCoordinates:
    """Non-identity instance labels and serialization positions in product node order."""

    instance_states: np.ndarray
    position_states: np.ndarray
    repeat_group_states: np.ndarray
    source_quantities: tuple[tuple[str, int], ...]


def source_instance_coordinates(
    record: SynthesisProgramGraphRecord, source_quantities: Mapping[str, int]
) -> SourceInstanceCoordinates:
    """Bind connected origin blocks to the complete declared source quantities.

    A singleton source precursor can contribute disconnected product regions. For
    repeated roles, one connected region per occurrence remains required; additional
    regions need independently qualified instance assignments and cannot be guessed.
    No precursor identifiers or molecular fingerprints enter these coordinates.
    """
    blocks_by_role = Counter(block.role for block in record.component_blocks)
    if set(source_quantities) != set(blocks_by_role):
        raise SynthesisProgramGraphError("Source quantity roles do not cover every product origin")
    if any(type(quantity) is not int or quantity < 1 for quantity in source_quantities.values()):
        raise SynthesisProgramGraphError("Source precursor quantities must be positive integers")
    for role, quantity in source_quantities.items():
        if quantity > 1 and blocks_by_role[role] != quantity:
            raise SynthesisProgramGraphError(
                f"Repeated precursor instance membership is unresolved for {role}: "
                f"{quantity} source occurrences, {blocks_by_role[role]} connected origin regions"
            )
    instances = np.zeros(record.node_count, dtype=np.int64)
    positions = np.zeros(record.node_count, dtype=np.int64)
    repeats = np.zeros(record.node_count, dtype=np.int64)
    assigned, occurrences, offsets = {}, Counter(), Counter()
    for block in record.component_blocks:
        quantity = source_quantities[block.role]
        occurrence = occurrences[block.role] if quantity > 1 else 0
        key = block.role, occurrence
        if key not in assigned:
            assigned[key] = len(assigned) + 1
        instance = assigned[key]
        instances[block.start : block.stop] = instance
        positions[block.start : block.stop] = np.arange(
            offsets[instance] + 1, offsets[instance] + block.atom_count + 1, dtype=np.int64
        )
        offsets[instance] += block.atom_count
        if quantity > 1:
            repeats[block.start : block.stop] = block.role_state
            occurrences[block.role] += 1
    if len(assigned) != sum(source_quantities.values()):
        raise SynthesisProgramGraphError("Source instance coordinates changed precursor quantities")
    return SourceInstanceCoordinates(
        instances, positions, repeats, tuple(sorted(source_quantities.items()))
    )


def collate_source_instance_records(
    records: Sequence[SynthesisProgramGraphRecord],
    source_quantities: Sequence[Mapping[str, int]],
    *,
    maximum_nodes: int | None = None,
    maximum_closures: int,
    conditioning_mode: str = "program",
) -> dict[str, Any]:
    """Use qualified source instances for repeat conditioning and component-based losses.

    Graphs, connected serialization blocks, fixed masks, atom origins, reaction cores
    and role-level morphology keep their shared contracts. Null conditioning remains
    unchanged. Source quantities are checked even when all conditions are suppressed.
    """
    import torch

    if len(records) != len(source_quantities):
        raise SynthesisProgramGraphError("Source quantities and product records are misaligned")
    coordinates = [
        source_instance_coordinates(record, quantities)
        for record, quantities in zip(records, source_quantities, strict=True)
    ]
    batch = collate_synthesis_program_records(
        records,
        maximum_nodes=maximum_nodes,
        maximum_closures=maximum_closures,
        conditioning_mode=conditioning_mode,
    )
    if conditioning_mode == "program":
        for index, (record, source) in enumerate(zip(records, coordinates, strict=True)):
            for key, values in (
                ("component_instance_states", source.instance_states),
                ("component_position_states", source.position_states),
                ("repeat_group_states", source.repeat_group_states),
            ):
                batch[key][index, : record.node_count] = torch.from_numpy(values)
    return batch
