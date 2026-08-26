from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from forge.model.synthesis_program_layout import (
    SynthesisProgramLayoutPrior,
    _compile_program_distribution,
    _RecordSummary,
)

UGI_BLOCKS = (
    (3, 2, ((10, 1),)),
    (4, 1, ((14, 1),)),
    (5, 6, ((12, 1), (13, 1))),
    (6, 7, ((11, 1),)),
)
INTERNAL_EDGES = (
    ((10, 3, True), (11, 6, True), 0),
    ((11, 6, True), (12, 5, True), 0),
    ((12, 5, True), (13, 5, True), 0),
    ((12, 5, True), (14, 4, True), 1),
)
ONE_HEAD_ATTACHMENT = tuple(
    sorted(
        (
            ((1, 3, False), (10, 3, True), 0),
            ((1, 5, False), (13, 5, True), 0),
            ((1, 6, False), (11, 6, True), 0),
            *INTERNAL_EDGES,
        )
    )
)
TWO_HEAD_ATTACHMENTS = tuple(sorted((ONE_HEAD_ATTACHMENT[0], *ONE_HEAD_ATTACHMENT)))


def _summary(*, amine_size: int, fixed_signature: tuple[tuple[object, ...], ...]) -> _RecordSummary:
    return _RecordSummary(
        depth=1,
        closure_count=0,
        blocks=((3, amine_size, ((10, 1),)), *UGI_BLOCKS[1:]),
        fixed_signature=fixed_signature,
        weight=0.5,
    )


def _prior() -> SynthesisProgramLayoutPrior:
    prior = object.__new__(SynthesisProgramLayoutPrior)
    prior.vocabulary = SimpleNamespace(
        role_states=(
            "unassigned",
            "aldehyde_tail",
            "alkyl_tail",
            "amine_head",
            "assembly_introduced",
            "isocyanide_tail",
            "oxoester_aldehyde_body_tail",
        ),
        program_to_index={"ugi_3cr_agile": 1},
    )
    prior.maximum_heavy_atoms = 194
    prior.maximum_closures = 3
    prior._distributions = {
        "ugi_3cr_agile": _compile_program_distribution(
            (
                _summary(amine_size=2, fixed_signature=ONE_HEAD_ATTACHMENT),
                _summary(amine_size=3, fixed_signature=TWO_HEAD_ATTACHMENTS),
            )
        )
    }
    prior._fixed_node_states = {"ugi_3cr_agile": {10: 1, 11: 1, 12: 1, 13: 1, 14: 2}}
    return prior


def test_factorized_ugi_sizes_are_conditioned_on_joint_fixed_attachment_contract() -> None:
    prior = _prior()
    rng = np.random.default_rng(3561515994897245985)
    observed_signatures: set[tuple[tuple[object, ...], ...]] = set()
    for sample_index in range(512):
        depth, closures, blocks, fixed = prior._sample_fields("ugi_3cr_agile", rng)
        amine_size = next(size for role, size, _ in blocks if role == 3)
        observed_signatures.add(fixed)
        assert amine_size == (3 if fixed == TWO_HEAD_ATTACHMENTS else 2)
        prior._ugi_layout(
            program_id="ugi_3cr_agile",
            depth=depth,
            closure_count=closures,
            blocks=blocks,
            fixed_signature=fixed,
            sample_index=sample_index,
        )
    assert observed_signatures == {ONE_HEAD_ATTACHMENT, TWO_HEAD_ATTACHMENTS}


def test_factorized_layout_support_gate_exhausts_each_semantic_bundle() -> None:
    support = _prior().validate_support()
    assert support == {
        "programs": 1,
        "semantic_bundles": 2,
        "component_size_support_cells": 8,
    }


def test_repeated_components_retain_their_own_reaction_core_signature() -> None:
    signature = ((2, 1), (3, 2), (4, 1))
    blocks = (
        (1, 5, ((5, 1),)),
        (2, 8, signature),
        (2, 8, signature),
    )
    core, spans = SynthesisProgramLayoutPrior._assign_core_positions(
        blocks,
        np.random.default_rng(17),
    )
    assert spans == [(1, 0, 5), (2, 5, 13), (2, 13, 21)]
    for _, start, stop in spans[1:]:
        observed = tuple(
            sorted((state, int((core[start:stop] == state).sum())) for state in (2, 3, 4))
        )
        assert observed == signature


def test_factorized_summary_contains_counts_but_no_component_identity() -> None:
    summary = _summary(amine_size=2, fixed_signature=ONE_HEAD_ATTACHMENT)
    assert summary.blocks == UGI_BLOCKS
    assert all(isinstance(value, int) for block in summary.blocks for value in block[:2])


def test_repeated_bl_layout_fixes_each_registry_derived_reaction_junction() -> None:
    prior = object.__new__(SynthesisProgramLayoutPrior)
    prior.vocabulary = SimpleNamespace(
        role_states=("unassigned", "unused", "tail", "head"),
        program_to_index={"bl": 1},
    )
    prior.maximum_heavy_atoms = 64
    prior.maximum_closures = 0
    prior._fixed_node_states = {"bl": {2: 3, 3: 0, 4: 0, 5: 0, 6: 8, 7: 8}}
    head = (3, 4, ((2, 1),))
    tail = (2, 8, ((3, 1), (4, 1), (5, 1), (6, 1), (7, 1)))

    def edge(left: tuple[int, int], right: tuple[int, int], bond: int = 0):
        return tuple(sorted(((left[0], left[1], True), (right[0], right[1], True)))) + (bond,)

    per_step = (
        edge((2, 3), (3, 2)),
        edge((3, 2), (4, 2)),
        edge((4, 2), (5, 2)),
        edge((5, 2), (6, 2), 1),
        edge((5, 2), (7, 2)),
    )
    record = prior._fixed_core_layout(
        program_id="bl",
        depth=2,
        closure_count=0,
        blocks=(head, tail, tail),
        fixed_signature=tuple(sorted((*per_step, *per_step))),
        sample_index=0,
    )

    assert int(record.fixed_atom_mask.sum()) == 11
    assert int(record.fixed_parent_bond_mask.sum()) == 10
    assert [block.role for block in record.component_blocks] == ["head", "tail", "tail"]
    for block in record.component_blocks[1:]:
        assert tuple(record.core_position_states[block.start : block.start + 5]) == (3, 4, 5, 6, 7)
    head_index = int(np.flatnonzero(record.core_position_states == 2)[0])
    tail_junctions = np.flatnonzero(record.core_position_states == 3)
    assert all(record.graph.parents[index] == head_index for index in tail_junctions)
