from dataclasses import replace

import numpy as np
import pytest
import torch

from forge.model.introduced_source_coordinates import (
    collate_source_and_introduced_records,
    source_and_introduced_coordinates,
)
from forge.model.reaction_program_flow import collate_synthesis_program_records
from forge.model.source_instance_coordinates import source_instance_coordinates
from forge.model.synthesis_program_graph import SynthesisProgramGraphError
from tests.test_remaining_program_origins import contracts, trace  # noqa: F401
from tests.test_source_instance_coordinates import record, repeated


@pytest.fixture(scope="module")
def ugi3(contracts):  # noqa: F811
    traced = trace(contracts, "ugi3")
    a = traced.annotations
    graph, _, _ = record(a.canonical_product_smiles, a.atom_roles, a.core_positions, "ugi3")
    quantities = {role: 1 for role in contracts["ugi3"][1].values()}
    return graph, quantities


def coordinates(graph, quantities):
    return source_and_introduced_coordinates(
        graph, quantities, introduced_roles=("assembly_introduced",)
    )


def test_introduced_oxygen_is_a_core_atom_but_never_a_fourth_precursor(ugi3):
    graph, quantities = ugi3
    result = coordinates(graph, quantities)
    assert result.source_quantities == tuple(sorted(quantities.items()))
    block = next(b for b in graph.component_blocks if b.role == "assembly_introduced")
    assert graph.core_position_states[block.start] > 1
    for array in (result.instance_states, result.position_states, result.repeat_group_states):
        assert array[block.start] == 0
    assert sorted(set(result.instance_states)) == [0, 1, 2, 3]
    assert np.count_nonzero(result.instance_states) == graph.node_count - 1
    for instance in (1, 2, 3):
        positions = result.position_states[result.instance_states == instance]
        np.testing.assert_array_equal(positions, np.arange(1, len(positions) + 1))
    assert not result.repeat_group_states.any()


def test_collation_retains_every_other_graph_and_semantic_tensor(ugi3):
    graph, quantities = ugi3
    prior = collate_synthesis_program_records([graph], maximum_closures=12)
    actual = collate_source_and_introduced_records(
        [graph], [quantities], introduced_roles=[("assembly_introduced",)], maximum_closures=12
    )
    assert set(prior) == set(actual)
    changed = {key for key in prior if not torch.equal(prior[key], actual[key])}
    assert changed == {"component_instance_states", "component_position_states"}
    assert actual["component_instance_states"].max() == 3
    assert prior["component_instance_states"].max() == 4


def test_no_introduction_retains_source_instance_contract():
    graph = repeated()[0]
    quantities = {"head": 1, "tail": 3}
    old = source_instance_coordinates(graph, quantities)
    actual = source_and_introduced_coordinates(graph, quantities, introduced_roles=())
    for name in ("instance_states", "position_states", "repeat_group_states"):
        np.testing.assert_array_equal(getattr(old, name), getattr(actual, name))


@pytest.mark.parametrize("invalid", [0, 1, True])
def test_introduced_atom_cannot_receive_even_a_dummy_source_quantity(ugi3, invalid):
    graph, quantities = ugi3
    with pytest.raises(SynthesisProgramGraphError, match="only precursor origins"):
        coordinates(graph, {**quantities, "assembly_introduced": invalid})


def test_introduction_requires_explicit_role_contract(ugi3):
    with pytest.raises(SynthesisProgramGraphError, match="cover every product origin"):
        source_and_introduced_coordinates(*ugi3, introduced_roles=())
    with pytest.raises(SynthesisProgramGraphError, match="single-introduction"):
        source_and_introduced_coordinates(*ugi3, introduced_roles=("amine",))


def test_introduced_exterior_atom_is_rejected(ugi3):
    graph, quantities = ugi3
    block = next(b for b in graph.component_blocks if b.role == "assembly_introduced")
    core = graph.core_position_states.copy()
    core[block.start] = 1
    with pytest.raises(SynthesisProgramGraphError, match="introduced core atom"):
        coordinates(replace(graph, core_position_states=core), quantities)


def test_null_conditioning_is_identical_and_still_validates_sources(ugi3):
    graph, quantities = ugi3
    kwargs = dict(maximum_closures=12, conditioning_mode="null")
    prior = collate_synthesis_program_records([graph], **kwargs)
    actual = collate_source_and_introduced_records(
        [graph], [quantities], introduced_roles=[("assembly_introduced",)], **kwargs
    )
    assert all(torch.equal(prior[key], actual[key]) for key in prior)
    with pytest.raises(SynthesisProgramGraphError):
        collate_source_and_introduced_records(
            [graph], [{}], introduced_roles=[("assembly_introduced",)], **kwargs
        )


def test_batch_introduction_contracts_cannot_be_misaligned(ugi3):
    with pytest.raises(SynthesisProgramGraphError, match="misaligned"):
        collate_source_and_introduced_records(
            [ugi3[0]], [ugi3[1]], introduced_roles=[], maximum_closures=12
        )
