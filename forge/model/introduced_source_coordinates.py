"""Keep explicitly introduced core atoms outside source precursor instances."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np

from forge.assembly.introduced_atom_origins import ASSEMBLY_INTRODUCED
from forge.model.reaction_program_flow import collate_synthesis_program_records
from forge.model.source_instance_coordinates import (
    SourceInstanceCoordinates,
    source_instance_coordinates,
)
from forge.model.synthesis_program_graph import (
    SynthesisProgramGraphError,
    SynthesisProgramGraphRecord,
)


def source_and_introduced_coordinates(
    record: SynthesisProgramGraphRecord,
    source_quantities: Mapping[str, int],
    *,
    introduced_roles: Collection[str],
) -> SourceInstanceCoordinates:
    """Bind source instances after an explicit single-introduction qualification.

    The caller must bind introduced roles to a qualified reaction contract. The
    sole introduced core atom retains graph, role and core supervision, with zero
    precursor instance, position and repeat coordinates. Source quantities exclude it.
    """
    if not introduced_roles:
        return source_instance_coordinates(record, source_quantities)
    if tuple(sorted(introduced_roles)) != (ASSEMBLY_INTRODUCED,):
        raise SynthesisProgramGraphError("Only the qualified single-introduction role is supported")
    blocks = record.component_blocks
    counts = Counter(block.role for block in blocks)
    if set(source_quantities) | {ASSEMBLY_INTRODUCED} != set(counts) or (
        ASSEMBLY_INTRODUCED in source_quantities
    ):
        raise SynthesisProgramGraphError("Source quantities must cover only precursor origins")
    if any(type(q) is not int or q < 1 for q in source_quantities.values()):
        raise SynthesisProgramGraphError("Source precursor quantities must be positive integers")
    introduced = [block for block in blocks if block.role == ASSEMBLY_INTRODUCED]
    if (
        len(introduced) != 1
        or introduced[0].atom_count != 1
        or record.core_position_states[introduced[0].start] <= 1
    ):
        raise SynthesisProgramGraphError("Exactly one qualified introduced core atom is required")
    for role, quantity in source_quantities.items():
        if quantity > 1 and counts[role] != quantity:
            raise SynthesisProgramGraphError(
                f"Repeated precursor instance membership is unresolved for {role}"
            )
    instances = np.zeros(record.node_count, dtype=np.int64)
    positions = np.zeros(record.node_count, dtype=np.int64)
    repeats = np.zeros(record.node_count, dtype=np.int64)
    assigned, occurrences, offsets = {}, Counter(), Counter()
    for block in blocks:
        if block.role == ASSEMBLY_INTRODUCED:
            continue
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


def collate_source_and_introduced_records(
    records: Sequence[SynthesisProgramGraphRecord],
    source_quantities: Sequence[Mapping[str, int]],
    *,
    introduced_roles: Sequence[Collection[str]],
    maximum_nodes: int | None = None,
    maximum_closures: int,
    conditioning_mode: str = "program",
) -> dict[str, Any]:
    """Preserve shared graph tensors while excluding introduced atoms from instances."""
    import torch

    if len(records) != len(source_quantities) or len(records) != len(introduced_roles):
        raise SynthesisProgramGraphError(
            "Source quantities, introduced roles and records misaligned"
        )
    coordinates = [
        source_and_introduced_coordinates(record, quantities, introduced_roles=introduced)
        for record, quantities, introduced in zip(
            records, source_quantities, introduced_roles, strict=True
        )
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
