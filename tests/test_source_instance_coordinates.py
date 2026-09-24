import json
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from forge.assembly.families import RegistryAssemblyAdapter
from forge.assembly.program_atom_origins import trace_repeated_program
from forge.assembly.repeated_components import RepeatBounds
from forge.core.hashing import resolve_pin
from forge.model.constitutional_program_graph import tensorize_constitutional_program_product
from forge.model.defog_feasibility import AtomState
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_flow import collate_synthesis_program_records
from forge.model.source_instance_coordinates import (
    collate_source_instance_records,
    source_instance_coordinates,
)
from forge.model.synthesis_program_graph import SynthesisProgramGraphError

ROOT = Path(__file__).resolve().parents[1]


def record(smiles, roles, core, program="test"):
    molecule = Chem.MolFromSmiles(smiles)
    vocabulary = ReactionProgramVocabulary.from_semantics(
        program_ids=(program,),
        roles=roles,
        core_positions=sorted(set(core) - {"exterior"}),
        maximum_steps=1,
    )
    atoms = QualifiedAtomVocabulary(
        tuple(
            sorted(
                {
                    AtomState(a.GetSymbol(), a.GetFormalCharge(), False, 0)
                    for a in molecule.GetAtoms()
                },
                key=AtomState.key,
            )
        )
    )
    graph = tensorize_constitutional_program_product(
        record_id="source-control",
        program_id=program,
        canonical_product_smiles=smiles,
        atom_roles=roles,
        atom_core_positions=core,
        program_depth=1,
        vocabulary=vocabulary,
        atom_vocabulary=atoms,
    )
    return graph, vocabulary, atoms


@pytest.fixture(scope="module")
def ugi4():
    cfg = json.loads(
        (ROOT / "configs/multireaction/compose_lipid_v8_ugi4_program_v1.json").read_text()
    )
    source = json.loads(
        resolve_pin(cfg["inputs"]["adjudication"], ROOT, label="source control").read_text()
    )
    adapter = RegistryAssemblyAdapter.from_registry(
        resolve_pin(cfg["inputs"]["registry"], ROOT, label="registry"),
        reaction_id=cfg["reaction_id"],
        expected_sha256=cfg["inputs"]["registry"]["sha256"],
    )
    control = source["source_controls"][0]
    traced = trace_repeated_program(
        adapter,
        control["components"],
        accumulator_role=adapter.roles[0],
        events=1,
        source_roles=source["source_contract"]["registry_to_source_roles"],
        bounds=RepeatBounds(maximum_outcomes=cfg["maximum_outcomes"]),
    )
    assert traced.disposition == "unique_forward_atom_coordinates"
    a = traced.annotations
    return (
        *record(a.canonical_product_smiles, a.atom_roles, a.core_positions, "ugi4"),
        source["source_contract"]["roles"],
    )


def repeated():
    return record(
        "CCN(CC)CC",
        ["head" if i == 2 else "tail" for i in range(7)],
        ["core" if i == 2 else "exterior" for i in range(7)],
    )


def test_transferred_oxygen_remains_part_of_one_acid_precursor(ugi4):
    graph, _, _, quantities = ugi4
    acid = [block for block in graph.component_blocks if block.role == "carboxylic_acid"]
    assert len(acid) == 2 and quantities["carboxylic_acid"] == 1
    coordinates = source_instance_coordinates(graph, quantities)
    first, second = acid
    assert coordinates.instance_states[first.start] == coordinates.instance_states[second.start]
    assert len(set(coordinates.instance_states)) == sum(quantities.values()) == 4
    assert not coordinates.repeat_group_states.any()
    for instance in np.unique(coordinates.instance_states):
        positions = coordinates.position_states[coordinates.instance_states == instance]
        np.testing.assert_array_equal(positions, np.arange(1, len(positions) + 1))


def test_fragmented_singleton_changes_only_instance_and_repeat_coordinates(ugi4):
    graph, _, _, quantities = ugi4
    prior = collate_synthesis_program_records([graph], maximum_closures=12)
    actual = collate_source_instance_records([graph], [quantities], maximum_closures=12)
    assert set(prior) == set(actual)
    changed = {key for key in prior if not torch.equal(prior[key], actual[key])}
    assert changed == {
        "component_instance_states",
        "component_position_states",
        "repeat_group_states",
    }
    assert actual["component_instance_states"].max() == 4
    assert prior["component_instance_states"].max() == 5


def test_repeated_connected_precursors_preserve_every_legacy_batch_tensor():
    graph, _, _ = repeated()
    quantities = {"head": 1, "tail": 3}
    prior = collate_synthesis_program_records([graph], maximum_closures=2, maximum_nodes=12)
    actual = collate_source_instance_records(
        [graph], [quantities], maximum_closures=2, maximum_nodes=12
    )
    assert set(prior) == set(actual)
    assert all(torch.equal(prior[key], actual[key]) for key in prior)
    coordinates = source_instance_coordinates(graph, quantities)
    tails = [
        coordinates.instance_states[block.start]
        for block in graph.component_blocks
        if block.role == "tail"
    ]
    assert len(set(tails)) == 3


def test_null_conditioning_remains_identical_even_for_fragmented_origins(ugi4):
    graph, _, _, quantities = ugi4
    prior = collate_synthesis_program_records(
        [graph], maximum_closures=12, conditioning_mode="null"
    )
    actual = collate_source_instance_records(
        [graph], [quantities], maximum_closures=12, conditioning_mode="null"
    )
    assert all(torch.equal(prior[key], actual[key]) for key in prior)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_source_quantity_cannot_be_coerced(ugi4, value):
    graph, _, _, quantities = ugi4
    with pytest.raises(SynthesisProgramGraphError, match="positive integers"):
        source_instance_coordinates(graph, {**quantities, "carboxylic_acid": value})


@pytest.mark.parametrize("extra", [False, True])
def test_complete_source_role_coverage_is_required(ugi4, extra):
    graph, _, _, quantities = ugi4
    changed = dict(quantities)
    if extra:
        changed["undeclared"] = 1
    else:
        changed.pop("carboxylic_acid")
    with pytest.raises(SynthesisProgramGraphError, match="cover every product origin"):
        source_instance_coordinates(graph, changed)


def test_repeated_fragment_membership_is_not_inferred_from_a_target(ugi4):
    graph, _, _, quantities = ugi4
    with pytest.raises(SynthesisProgramGraphError, match="membership is unresolved"):
        collate_source_instance_records(
            [graph],
            [{**quantities, "carboxylic_acid": 3}],
            maximum_closures=12,
            conditioning_mode="null",
        )


def test_batch_quantities_cannot_be_misaligned(ugi4):
    with pytest.raises(SynthesisProgramGraphError, match="misaligned"):
        collate_source_instance_records([ugi4[0]], [], maximum_closures=12)


def test_dictionary_order_does_not_change_instance_coordinates(ugi4):
    graph, _, _, quantities = ugi4
    first = source_instance_coordinates(graph, quantities)
    second = source_instance_coordinates(graph, dict(reversed(list(quantities.items()))))
    for name in ("instance_states", "position_states", "repeat_group_states"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
