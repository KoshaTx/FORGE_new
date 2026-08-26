from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from forge.model.defog_feasibility import AtomState
from forge.model.local_chemistry_support import (
    LEGACY_POLICY_SCHEMA,
    LocalChemistrySupport,
    LocalChemistrySupportError,
    assess_product_local_chemistry,
    build_local_chemistry_support,
)
from forge.model.reaction_program_flow import collate_synthesis_program_layouts
from forge.model.sparse_topology_feasibility import SparseGraphRecord
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)
from forge.model.synthesis_program_sampling import (
    LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
    SynthesisProgramSamplingError,
    decode_synthesis_program_strict_argmax,
    sample_synthesis_program_products,
)

PROGRAM = "test_program"
ATOM_VOCABULARY = (
    AtomState("C", 0, False, 0),
    AtomState("O", 0, False, 0),
    AtomState("N", 0, False, 0),
)


def _chain_record(
    structure_id: str,
    canonical_smiles: str,
    node_states: tuple[int, ...],
) -> SynthesisProgramGraphRecord:
    count = len(node_states)
    parents = np.asarray((0, *range(count - 1)), dtype=np.int64)
    parent_bonds = np.zeros(count, dtype=np.int64)
    edges = np.zeros((count, count), dtype=np.int8)
    for child in range(1, count):
        edges[child - 1, child] = 1
        edges[child, child - 1] = 1
    return SynthesisProgramGraphRecord(
        graph=SparseGraphRecord(
            structure_id=structure_id,
            canonical_smiles=canonical_smiles,
            node_states=np.asarray(node_states, dtype=np.int64),
            parents=parents,
            parent_bonds=parent_bonds,
            closure_left=np.zeros(0, dtype=np.int64),
            closure_right=np.zeros(0, dtype=np.int64),
            closure_bonds=np.zeros(0, dtype=np.int64),
            edges=edges,
        ),
        canonical_atom_order=np.arange(count, dtype=np.int64),
        program_id=PROGRAM,
        program_state=1,
        program_depth=1,
        role_states=np.ones(count, dtype=np.int64),
        core_position_states=np.zeros(count, dtype=np.int64),
        component_blocks=(SynthesisProgramComponentBlock("tail", 1, 0, count),),
        fixed_atom_mask=np.zeros(count, dtype=np.bool_),
        fixed_parent_bond_mask=np.zeros(count, dtype=np.bool_),
        fixed_closure_bond_mask=np.zeros(0, dtype=np.bool_),
    )


def _policy() -> LocalChemistrySupport:
    return build_local_chemistry_support(
        (
            _chain_record("carbon-oxygen", "CO", (0, 1)),
            _chain_record("carbon-nitrogen", "CN", (0, 2)),
        ),
        ATOM_VOCABULARY,
    )


def _three_membered_ring_record() -> SynthesisProgramGraphRecord:
    return _ring_record("triangle", "C1CC1", (0, 0, 0))


def _ring_record(
    structure_id: str,
    canonical_smiles: str,
    node_states: tuple[int, ...],
) -> SynthesisProgramGraphRecord:
    record = _chain_record(structure_id, canonical_smiles, node_states)
    count = len(node_states)
    graph = SparseGraphRecord(
        structure_id=record.graph.structure_id,
        canonical_smiles=record.graph.canonical_smiles,
        node_states=record.graph.node_states,
        parents=record.graph.parents,
        parent_bonds=record.graph.parent_bonds,
        closure_left=np.asarray([0], dtype=np.int64),
        closure_right=np.asarray([count - 1], dtype=np.int64),
        closure_bonds=np.asarray([0], dtype=np.int64),
        edges=np.asarray(record.graph.edges).copy(),
    )
    graph.edges[0, count - 1] = 1
    graph.edges[count - 1, 0] = 1
    return SynthesisProgramGraphRecord(
        graph=graph,
        canonical_atom_order=record.canonical_atom_order,
        program_id=record.program_id,
        program_state=record.program_state,
        program_depth=record.program_depth,
        role_states=record.role_states,
        core_position_states=record.core_position_states,
        component_blocks=record.component_blocks,
        fixed_atom_mask=record.fixed_atom_mask,
        fixed_parent_bond_mask=record.fixed_parent_bond_mask,
        fixed_closure_bond_mask=np.asarray([False]),
    )


def _predictions(layout: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch, nodes = layout["nodes"].shape
    closures = layout["closure_bonds"].shape[1]
    predictions = {
        "nodes": torch.zeros((batch, nodes, len(ATOM_VOCABULARY))),
        "parents": torch.zeros((batch, nodes, nodes)),
        "parent_bonds": torch.zeros((batch, nodes, 4)),
        "closure_left": torch.zeros((batch, closures, nodes)),
        "closure_right": torch.zeros((batch, closures, nodes)),
        "closure_bonds": torch.zeros((batch, closures, 4)),
    }
    predictions["parents"].scatter_(-1, layout["parents"].unsqueeze(-1), 10.0)
    predictions["parent_bonds"].scatter_(-1, layout["parent_bonds"].unsqueeze(-1), 10.0)
    if closures:
        predictions["closure_left"].scatter_(-1, layout["closure_left"].unsqueeze(-1), 10.0)
        predictions["closure_right"].scatter_(-1, layout["closure_right"].unsqueeze(-1), 10.0)
    return predictions


def _prefer_record_topology(
    predictions: dict[str, torch.Tensor],
    record: SynthesisProgramGraphRecord,
) -> None:
    for child, parent in enumerate(record.graph.parents.tolist()):
        predictions["parents"][0, child, int(parent)] = 20.0
    for slot, (left, right) in enumerate(
        zip(record.graph.closure_left, record.graph.closure_right, strict=True)
    ):
        predictions["closure_left"][0, slot, int(left)] = 20.0
        predictions["closure_right"][0, slot, int(right)] = 20.0


def test_local_chemistry_policy_roundtrips_without_component_identity() -> None:
    support = _policy()
    payload = support.to_mapping()

    assert LocalChemistrySupport.from_mapping(payload).to_mapping() == payload
    assert payload["policy"]["complete_component_identities_stored"] is False
    serialized = repr(payload).lower()
    assert "structure_id" not in serialized
    assert "canonical_smiles" not in serialized

    malformed = copy.deepcopy(payload)
    malformed["programs"][PROGRAM]["unexpected_component_id"] = "carbon-oxygen"
    with pytest.raises(LocalChemistrySupportError, match="malformed"):
        LocalChemistrySupport.from_mapping(malformed)


def test_legacy_policy_roundtrip_does_not_claim_full_cycle_support() -> None:
    support = build_local_chemistry_support(
        (_three_membered_ring_record(),),
        ATOM_VOCABULARY,
        schema_version=LEGACY_POLICY_SCHEMA,
    )
    payload = support.to_mapping()
    restored = LocalChemistrySupport.from_mapping(payload)

    assert payload["schema_version"] == LEGACY_POLICY_SCHEMA
    assert "program_cycles" not in payload["programs"][PROGRAM]
    assert restored.enforces_role_cycles is False


@pytest.mark.parametrize(
    ("smiles", "failure"),
    (
        ("C1CO1", "unsupported_three_membered_ring"),
        ("COO", "unsupported_local_edge"),
        ("CNO", "unsupported_local_edge"),
    ),
)
def test_method_blind_assessment_abstains_on_unsupported_local_patterns(
    smiles: str,
    failure: str,
) -> None:
    result = assess_product_local_chemistry(smiles, program_id=PROGRAM, support=_policy())

    assert result["valid_connected"] is True
    assert result["local_chemistry_supported"] is False
    assert failure in result["failure_types"]


def test_method_blind_assessment_retains_supported_graph_and_invalid_denominator() -> None:
    support = _policy()

    clean = assess_product_local_chemistry("CO", program_id=PROGRAM, support=support)
    invalid = assess_product_local_chemistry(None, program_id=PROGRAM, support=support)

    assert clean["local_chemistry_supported"] is True
    assert clean["failure_types"] == []
    assert invalid["valid_connected"] is False
    assert invalid["failure_types"] == ["invalid_or_disconnected"]


def test_local_decoder_masks_unsupported_oxygen_oxygen_edge_without_repair() -> None:
    record = _chain_record("decode", "CO", (0, 1))
    layout = collate_synthesis_program_layouts((record,), maximum_closures=0)
    predictions = _predictions(layout)
    predictions["nodes"][:, :, 1] = 20.0
    predictions["nodes"][:, :, 0] = 10.0

    unconstrained, unconstrained_reasons = decode_synthesis_program_strict_argmax(
        predictions,
        layout,
        (record,),
        ATOM_VOCABULARY,
    )
    constrained, constrained_reasons = decode_synthesis_program_strict_argmax(
        predictions,
        layout,
        (record,),
        ATOM_VOCABULARY,
        _policy(),
    )

    assert unconstrained_reasons == (None,)
    assert unconstrained["nodes"][0, :2].tolist() == [1, 1]
    assert constrained_reasons == (None,)
    assert constrained["nodes"][0, :2].tolist() == [1, 0]


def test_local_decoder_masks_unsupported_three_membered_closure_before_atom_decode() -> None:
    record = _three_membered_ring_record()
    layout = collate_synthesis_program_layouts((record,), maximum_closures=1)
    predictions = _predictions(layout)

    _, unconstrained_reasons = decode_synthesis_program_strict_argmax(
        predictions,
        layout,
        (record,),
        ATOM_VOCABULARY,
    )
    _, constrained_reasons = decode_synthesis_program_strict_argmax(
        predictions,
        layout,
        (record,),
        ATOM_VOCABULARY,
        _policy(),
    )

    assert unconstrained_reasons == (None,)
    assert constrained_reasons == ("closure_pair_unavailable",)


def test_local_decoder_uses_complete_role_conditioned_ring_morphology() -> None:
    supported_ring = _ring_record("supported-five", "C1CCCO1", (0, 0, 0, 0, 1))
    support = build_local_chemistry_support(
        (
            _chain_record("four-atom-bound", "CCCC", (0, 0, 0, 0)),
            supported_ring,
        ),
        ATOM_VOCABULARY,
    )
    unsupported_ring = _ring_record("unsupported-four", "C1CCO1", (0, 0, 0, 1))

    unsupported_layout = collate_synthesis_program_layouts((unsupported_ring,), maximum_closures=1)
    unsupported_predictions = _predictions(unsupported_layout)
    _prefer_record_topology(unsupported_predictions, unsupported_ring)
    _, unsupported_reasons = decode_synthesis_program_strict_argmax(
        unsupported_predictions,
        unsupported_layout,
        (unsupported_ring,),
        ATOM_VOCABULARY,
        support,
    )
    supported_layout = collate_synthesis_program_layouts((supported_ring,), maximum_closures=1)
    supported_predictions = _predictions(supported_layout)
    _prefer_record_topology(supported_predictions, supported_ring)
    _, supported_reasons = decode_synthesis_program_strict_argmax(
        supported_predictions,
        supported_layout,
        (supported_ring,),
        ATOM_VOCABULARY,
        support,
    )

    assert support.enforces_role_cycles is True
    assert support.allows_any_role_cycle(PROGRAM, ("tail",) * 5)
    assert not support.allows_any_role_cycle(PROGRAM, ("tail",) * 4)
    assert unsupported_reasons == ("closure_pair_unavailable",)
    assert supported_reasons == (None,)


def test_local_decoder_policy_binding_fails_closed() -> None:
    record = _chain_record("binding", "CO", (0, 1))
    kwargs = {
        "model": None,
        "records": (record,),
        "atom_vocabulary": ATOM_VOCABULARY,
        "node_marginal": np.full(len(ATOM_VOCABULARY), 1 / len(ATOM_VOCABULARY)),
        "bond_marginal": np.full(4, 0.25),
        "samples_per_program": 1,
        "sample_steps": 2,
        "batch_size": 1,
        "seed": 7,
        "device": "cpu",
    }
    with pytest.raises(SynthesisProgramSamplingError, match="supplied together"):
        sample_synthesis_program_products(
            **kwargs,
            terminal_decode_policy=LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
        )
    with pytest.raises(SynthesisProgramSamplingError, match="supplied together"):
        sample_synthesis_program_products(**kwargs, local_chemistry_support=_policy())
