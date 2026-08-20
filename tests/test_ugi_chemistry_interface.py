from __future__ import annotations

import csv
import gzip
import json
from dataclasses import fields, replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from experiments.archive.phase1.design_audits.canonical_representation_audit import (
    load_atom_vocabulary,
)
from experiments.archive.phase1.design_audits.ugi_chemistry_interface_audit import (
    audit_ugi_chemistry_interface,
)
from forge.model.ugi_adapter_features import ORIGIN_TO_INDEX, tensorize_ugi_l1_support_record
from forge.model.ugi_chemistry_interface import (
    ADAPTER_ATTACHMENT_BOND_STATE,
    ROOT_BOND_TARGET,
    WITHHELD_STATE,
    ChemistryTopologyCondition,
    assemble_ugi_chemistry_topology_condition,
    core_schema_from_record,
    materialize_chemistry_target,
    project_chemistry_topology_condition,
    recompute_adapter_distances,
    validate_chemistry_topology_condition,
)
from forge.model.ugi_morphology_program import split_ugi_support_morphology
from forge.potency.annotations import ROLE_NAMES

torch = pytest.importorskip("torch")

from forge.corpus.ugi_chemistry_corpus import (  # noqa: E402
    UgiChemistryRecord,
    load_expanded_ugi_chemistry_corpus,
)
from forge.model.ugi_chemistry_flow import (  # noqa: E402
    UgiChemistryFlow,
    UgiChemistrySample,
    _masked_terminal_choice,
    _terminal_channel_temperatures,
    _terminal_choice_modes,
    chemistry_sample_to_molecule,
    chemistry_source_marginals,
    collate_ugi_chemistry_records,
    decoration_source_marginal,
    noise_ugi_chemistry_batch,
    rstar_ugi_chemistry_step,
    sample_ugi_chemistry,
    ugi_chemistry_flow_loss,
    valence_constrained_terminal_sample,
)

REPO = Path(__file__).resolve().parents[1]


def test_stochastic_terminal_choice_is_seeded_and_respects_support() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0])
    valid = torch.tensor([True, False, True])

    def draw(seed: int) -> list[int]:
        generator = torch.Generator().manual_seed(seed)
        return [
            _masked_terminal_choice(
                logits,
                valid,
                mode="stochastic",
                generator=generator,
                temperature=1.0,
            )
            for _ in range(64)
        ]

    first = draw(271)
    second = draw(271)

    assert first == second
    assert set(first) == {0, 2}


@cache
def _source_rows() -> (
    tuple[dict[str, str], dict[str, str], tuple[dict[str, str], ...], tuple[dict[str, str], ...]]
):
    products_path = REPO / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_products.csv.gz"
    atoms_path = REPO / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_atoms.csv.gz"
    with gzip.open(products_path, "rt", newline="") as handle:
        products = list(csv.DictReader(handle))
    first = products[0]
    aromatic = next(
        row
        for row in products
        if any(
            atom.GetIsAromatic() for atom in Chem.MolFromSmiles(row["product_smiles"]).GetAtoms()
        )
    )
    selected = {first["product_id"], aromatic["product_id"]}
    atoms_by_product = {product_id: [] for product_id in selected}
    with gzip.open(atoms_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["product_id"] in atoms_by_product:
                atoms_by_product[row["product_id"]].append(row)
    return (
        first,
        aromatic,
        tuple(atoms_by_product[first["product_id"]]),
        tuple(atoms_by_product[aromatic["product_id"]]),
    )


@cache
def _records():
    first, aromatic, first_atoms, aromatic_atoms = _source_rows()
    vocabulary = load_atom_vocabulary(REPO / "results/phase1/product_v3_atom_vocabulary.json")
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    return (
        tensorize_ugi_l1_support_record(
            first,
            first_atoms,
            atom_to_index,
            preserve_aromaticity=True,
        ),
        tensorize_ugi_l1_support_record(
            aromatic,
            aromatic_atoms,
            atom_to_index,
            preserve_aromaticity=True,
        ),
        vocabulary,
        atom_to_index,
    )


def _chemistry_records() -> tuple[UgiChemistryRecord, UgiChemistryRecord]:
    first, aromatic, _, atom_to_index = _records()
    schema = core_schema_from_record(first)
    return tuple(
        UgiChemistryRecord(
            product_id=record.support_graph.structure_id,
            condition=project_chemistry_topology_condition(record, schema),
            target=materialize_chemistry_target(record, atom_to_index),
        )
        for record in (first, aromatic)
    )


@cache
def _expanded_multi_decoration_record():
    corpus = load_expanded_ugi_chemistry_corpus(
        REPO / "results/phase1/ugi_expanded_chemistry_exemplars/assignments.csv.gz",
        REPO / "results/phase1/ugi_expanded_chemistry_exemplars/semantic_products.csv.gz",
        REPO / "results/phase1/ugi_expanded_chemistry_exemplars/semantic_atoms.csv.gz",
        REPO / "results/phase1/product_v3_atom_vocabulary.json",
    )
    record = next(
        record
        for fold in ("train", "calibration", "heldout")
        for record in corpus.records_by_fold[fold]
        if record.target.decorations.count > 1
    )
    with gzip.open(
        REPO / "results/phase1/ugi_expanded_chemistry_exemplars/assignments.csv.gz",
        "rt",
        newline="",
    ) as handle:
        expected = next(
            row["canonical_product_smiles"]
            for row in csv.DictReader(handle)
            if row["product_id"] == record.product_id
        )
    return record, corpus.atom_vocabulary, expected


def test_chemistry_condition_structurally_excludes_target_channels() -> None:
    first, _, _, _ = _records()
    schema = core_schema_from_record(first)
    condition = project_chemistry_topology_condition(first, schema)
    field_names = {field.name for field in fields(ChemistryTopologyCondition)}

    assert "node_states" not in field_names
    assert "parent_bonds" not in field_names
    assert "closure_bonds" not in field_names
    assert "distance_to_core" not in field_names
    assert "distance_to_own_port" not in field_names
    assert "distances_to_all_ports" not in field_names
    assert np.all(condition.fixed_atom_states[~condition.fixed_atom_mask] == WITHHELD_STATE)
    assert np.all(
        condition.fixed_parent_bond_states[~condition.fixed_parent_bond_mask] == WITHHELD_STATE
    )
    assert np.all(
        condition.fixed_closure_bond_states[~condition.fixed_closure_bond_mask] == WITHHELD_STATE
    )
    validate_chemistry_topology_condition(condition)


def test_adapter_distances_are_recomputed_from_generated_topology() -> None:
    first, _, _, _ = _records()
    condition = project_chemistry_topology_condition(first, core_schema_from_record(first))
    recomputed = recompute_adapter_distances(condition)

    assert np.array_equal(recomputed.distance_to_core, first.support_adapter.distance_to_core)
    assert np.array_equal(
        recomputed.distance_to_own_port,
        first.support_adapter.distance_to_own_port,
    )
    assert np.array_equal(
        recomputed.distances_to_all_ports,
        first.support_adapter.distances_to_all_ports,
    )

    # Add one feasible within-origin sparse closure that shortens a distal
    # path. Recomputed features must follow that current generated topology.
    role = condition.origin_states[-1]
    candidates = np.flatnonzero(
        (condition.origin_states == role) & ~condition.fixed_atom_mask
    ).tolist()
    left = min(candidates, key=lambda index: recomputed.distance_to_own_port[index])
    right = max(candidates, key=lambda index: recomputed.distance_to_own_port[index])
    assert recomputed.distance_to_own_port[right] - recomputed.distance_to_own_port[left] > 1
    existing = {
        tuple(sorted((int(a), int(b))))
        for a, b in zip(condition.closure_left, condition.closure_right, strict=True)
    }
    existing.update(
        tuple(sorted((int(condition.parents[child]), child)))
        for child in range(1, condition.node_count)
    )
    assert tuple(sorted((left, right))) not in existing
    closure_pairs = sorted(
        [
            *zip(condition.closure_left.tolist(), condition.closure_right.tolist(), strict=True),
            tuple(sorted((left, right))),
        ]
    )
    changed_topology = replace(
        condition,
        closure_left=np.asarray([pair[0] for pair in closure_pairs], dtype=np.int64),
        closure_right=np.asarray([pair[1] for pair in closure_pairs], dtype=np.int64),
        fixed_closure_bond_mask=np.zeros(len(closure_pairs), dtype=bool),
        fixed_closure_bond_states=np.full(len(closure_pairs), WITHHELD_STATE, dtype=np.int64),
    )
    changed = recompute_adapter_distances(changed_topology)
    assert not np.array_equal(changed.distances_to_all_ports, recomputed.distances_to_all_ports)


def test_chemistry_target_retains_atoms_bonds_aromaticity_and_decorations() -> None:
    first, aromatic, vocabulary, atom_to_index = _records()
    first_target = materialize_chemistry_target(first, atom_to_index)
    aromatic_target = materialize_chemistry_target(aromatic, atom_to_index)

    assert np.array_equal(first_target.atom_states, first.support_graph.node_states)
    assert first_target.parent_bond_states[0] == ROOT_BOND_TARGET
    assert np.array_equal(
        first_target.parent_bond_states[1:],
        first.support_graph.parent_bonds[1:],
    )
    assert first_target.decorations.count == first.skeleton.removed_count
    assert np.all(first_target.decorations.anchor_indices < first.support_graph.node_count)
    assert any(vocabulary[int(state)].aromatic for state in aromatic_target.atom_states)
    assert 3 in aromatic_target.parent_bond_states or 3 in aromatic_target.closure_bond_states


def test_fixed_ugi_core_schema_is_component_invariant() -> None:
    first, aromatic, _, _ = _records()
    assert core_schema_from_record(first) == core_schema_from_record(aromatic)


def test_generated_exteriors_stitch_to_the_exact_training_topology() -> None:
    first, _, _, _ = _records()
    first_product, _, _, _ = _source_rows()
    components = json.loads(first_product["component_smiles_json"])
    morphology = split_ugi_support_morphology(
        first,
        components,
        product_id=first_product["product_id"],
    )
    assembled = assemble_ugi_chemistry_topology_condition(
        structure_id=first.support_graph.structure_id,
        offspring_by_role={
            component.role: component.offspring for component in morphology.components
        },
        attachment_counts_by_role={
            component.role: component.attachment_count for component in morphology.components
        },
        closure_left_by_role={
            component.role: component.closure_left for component in morphology.components
        },
        closure_right_by_role={
            component.role: component.closure_right for component in morphology.components
        },
        schema=core_schema_from_record(first),
    )
    projected = project_chemistry_topology_condition(first, core_schema_from_record(first))

    for field in fields(ChemistryTopologyCondition):
        left = getattr(assembled, field.name)
        right = getattr(projected, field.name)
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right), field.name
        else:
            assert left == right, field.name


def test_generated_two_port_amine_exterior_is_stitched_without_fragment_identity() -> None:
    first, _, _, _ = _records()
    roles = {
        "amine_head": np.asarray([0, 0], dtype=np.int64),
        "oxoester_aldehyde_body_tail": np.asarray([1, 0], dtype=np.int64),
        "isocyanide_tail": np.asarray([1, 0], dtype=np.int64),
    }
    empty = {role: np.asarray([], dtype=np.int64) for role in roles}
    condition = assemble_ugi_chemistry_topology_condition(
        structure_id="generated_two_port_amine",
        offspring_by_role=roles,
        attachment_counts_by_role={
            "amine_head": 2,
            "oxoester_aldehyde_body_tail": 1,
            "isocyanide_tail": 1,
        },
        closure_left_by_role=empty,
        closure_right_by_role=empty,
        schema=core_schema_from_record(first),
    )

    core_mask = condition.fixed_atom_mask
    amine_origin = condition.origin_states == ORIGIN_TO_INDEX["amine_head"]
    boundaries = [
        (int(condition.parents[child]), child)
        for child in range(1, condition.node_count)
        if amine_origin[int(condition.parents[child])]
        and amine_origin[child]
        and bool(core_mask[int(condition.parents[child])]) != bool(core_mask[child])
    ]

    assert len(boundaries) == 2
    assert len({left for left, _ in boundaries}) == 1
    for _, child in boundaries:
        assert condition.fixed_parent_bond_mask[child]
        assert condition.fixed_parent_bond_states[child] == ADAPTER_ATTACHMENT_BOND_STATE
    validate_chemistry_topology_condition(condition)


def test_small_corpus_chemistry_interface_audit_passes(tmp_path: Path) -> None:
    first, aromatic, first_atoms, aromatic_atoms = _source_rows()
    products_path = tmp_path / "products.csv.gz"
    atoms_path = tmp_path / "atoms.csv.gz"
    for path, rows in (
        (products_path, (first, aromatic)),
        (atoms_path, (*first_atoms, *aromatic_atoms)),
    ):
        with gzip.open(path, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = audit_ugi_chemistry_interface(
        products_path,
        atoms_path,
        REPO / "results/phase1/product_v3_atom_vocabulary.json",
    )

    assert result["status"] == "pass"
    assert result["counts"]["products"] == 2
    assert result["gates"]["condition_dataclass_excludes_target_channels"]


def test_chemistry_flow_noises_only_withheld_states_and_backpropagates() -> None:
    records = _chemistry_records()
    _, _, vocabulary, _ = _records()
    maximum_nodes = max(record.condition.node_count for record in records)
    maximum_closures = max(record.condition.closure_count for record in records)
    batch = collate_ugi_chemistry_records(
        records,
        maximum_nodes=maximum_nodes,
        maximum_closures=maximum_closures,
    )
    source_arrays = chemistry_source_marginals(
        records,
        atom_classes=len(vocabulary),
        bond_classes=4,
        probability_floor=1e-5,
    )
    sources = {
        key: torch.as_tensor(value, dtype=torch.float32) for key, value in source_arrays.items()
    }
    generator = torch.Generator().manual_seed(17)
    t = torch.tensor([0.25, 0.75])
    noisy = noise_ugi_chemistry_batch(batch, sources, t, generator)

    assert torch.equal(
        noisy["nodes"][batch["fixed_atom_mask"]], batch["nodes"][batch["fixed_atom_mask"]]
    )
    assert torch.equal(
        noisy["parent_bonds"][batch["fixed_parent_bond_mask"]],
        batch["parent_bonds"][batch["fixed_parent_bond_mask"]],
    )
    model = UgiChemistryFlow(
        atom_classes=len(vocabulary),
        bond_classes=4,
        maximum_nodes=maximum_nodes,
        maximum_distance=64,
        hidden_dim=32,
        layers=2,
        dropout=0.0,
    )
    predictions = model(
        nodes=noisy["nodes"],
        parent_bonds=noisy["parent_bonds"],
        closure_bonds=noisy["closure_bonds"],
        t=t,
        topology=batch,
    )
    loss, metrics = ugi_chemistry_flow_loss(predictions, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["total"] > 0
    assert predictions["nodes"].shape == (*batch["nodes"].shape, len(vocabulary))
    assert predictions["decoration_anchor"].shape == (
        len(records),
        maximum_nodes + 1,
    )
    assert model.atom_embedding.weight.grad is not None


def test_chemistry_rstar_step_preserves_fixed_core_and_valid_decoration_support() -> None:
    records = _chemistry_records()
    _, _, vocabulary, _ = _records()
    maximum_nodes = max(record.condition.node_count for record in records)
    maximum_closures = max(record.condition.closure_count for record in records)
    batch = collate_ugi_chemistry_records(
        records,
        maximum_nodes=maximum_nodes,
        maximum_closures=maximum_closures,
    )
    source_arrays = chemistry_source_marginals(
        records,
        atom_classes=len(vocabulary),
        bond_classes=4,
        probability_floor=1e-5,
    )
    sources = {
        key: torch.as_tensor(value, dtype=torch.float32) for key, value in source_arrays.items()
    }
    generator = torch.Generator().manual_seed(23)
    state = noise_ugi_chemistry_batch(batch, sources, torch.zeros(len(records)), generator)
    model = UgiChemistryFlow(
        atom_classes=len(vocabulary),
        bond_classes=4,
        maximum_nodes=maximum_nodes,
        maximum_distance=64,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
    )
    predictions = model(
        nodes=state["nodes"],
        parent_bonds=state["parent_bonds"],
        closure_bonds=state["closure_bonds"],
        t=torch.zeros(len(records)),
        topology=batch,
    )
    updated = rstar_ugi_chemistry_step(
        state,
        predictions,
        batch,
        sources,
        t=0.0,
        dt=0.1,
        generator=generator,
    )
    decoration_source = decoration_source_marginal(batch, sources["decoration"])

    assert torch.allclose(decoration_source.sum(dim=1), torch.ones(len(records)))
    assert torch.count_nonzero(decoration_source[:, 1:][~batch["atom_variable_mask"]]) == 0
    assert torch.equal(
        updated["nodes"][batch["fixed_atom_mask"]],
        batch["nodes"][batch["fixed_atom_mask"]],
    )


def test_exact_chemistry_target_reconstructs_the_complete_source_molecule() -> None:
    first_record, _ = _chemistry_records()
    first_product, _, _, _ = _source_rows()
    _, _, vocabulary, _ = _records()
    target = first_record.target
    sample = UgiChemistrySample(
        atom_states=target.atom_states.copy(),
        parent_bond_states=target.parent_bond_states.copy(),
        closure_bond_states=target.closure_bond_states.copy(),
        decoration_anchor=(
            int(target.decorations.anchor_indices[0]) + 1 if target.decorations.count else 0
        ),
    )
    molecule = chemistry_sample_to_molecule(
        first_record.condition,
        sample,
        vocabulary,
    )

    assert (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        == first_product["product_smiles"]
    )


def test_untrained_chemistry_sampler_preserves_core_support() -> None:
    records = _chemistry_records()
    _, _, vocabulary, _ = _records()
    maximum_nodes = max(record.condition.node_count for record in records)
    source_arrays = chemistry_source_marginals(
        records,
        atom_classes=len(vocabulary),
        bond_classes=4,
        probability_floor=1e-5,
    )
    model = UgiChemistryFlow(
        atom_classes=len(vocabulary),
        bond_classes=4,
        maximum_nodes=maximum_nodes,
        maximum_distance=64,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
    )
    samples, diagnostics = sample_ugi_chemistry(
        model,
        tuple(record.condition for record in records),
        source_arrays,
        atom_classes=len(vocabulary),
        sample_steps=2,
        batch_size=2,
        seed=31,
        device="cpu",
    )

    assert diagnostics["samples"] == 2
    for record, sample in zip(records, samples, strict=True):
        fixed = record.condition.fixed_atom_mask
        assert np.array_equal(
            sample.atom_states[fixed],
            record.condition.fixed_atom_states[fixed],
        )


def test_terminal_support_masks_oxygen_oxygen_adjacency() -> None:
    record, _ = _chemistry_records()
    _, _, vocabulary, _ = _records()
    condition = record.condition
    atom_logits = torch.zeros((1, condition.node_count, len(vocabulary)))
    for index, state in enumerate(vocabulary):
        if state.symbol == "O":
            atom_logits[:, :, index] = 10.0
        elif state.symbol == "C" and not state.aromatic and state.formal_charge == 0:
            atom_logits[:, :, index] = 9.0
    parent_logits = torch.zeros((1, condition.node_count, 4))
    parent_logits[:, :, 0] = 10.0
    closure_logits = torch.zeros((1, condition.closure_count, 4))
    if condition.closure_count:
        closure_logits[:, :, 0] = 10.0
    decoration_anchor = torch.zeros((1, condition.node_count + 1))
    decoration_anchor[:, 0] = 10.0

    sample = valence_constrained_terminal_sample(
        condition,
        {
            "nodes": atom_logits,
            "parent_bonds": parent_logits,
            "closure_bonds": closure_logits,
            "decoration_anchor": decoration_anchor,
        },
        0,
        vocabulary,
        1,
    )
    molecule = chemistry_sample_to_molecule(condition, sample, vocabulary)

    assert not any(
        bond.GetBeginAtom().GetAtomicNum() == 8 and bond.GetEndAtom().GetAtomicNum() == 8
        for bond in molecule.GetBonds()
    )


def test_terminal_decoder_modes_isolate_atom_and_decoration_sampling() -> None:
    assert _terminal_choice_modes("argmax") == ("argmax", "argmax", "argmax")
    assert _terminal_choice_modes("bond_stochastic") == (
        "argmax",
        "stochastic",
        "argmax",
    )
    assert _terminal_choice_modes("atom_bond_stochastic") == (
        "stochastic",
        "stochastic",
        "argmax",
    )
    assert _terminal_choice_modes("decoration_bond_stochastic") == (
        "argmax",
        "stochastic",
        "stochastic",
    )
    assert _terminal_choice_modes("stochastic") == (
        "stochastic",
        "stochastic",
        "stochastic",
    )


def test_terminal_channel_temperatures_support_role_conditional_atom_sampling() -> None:
    condition = _chemistry_records()[0].condition
    atom, bond, decoration = _terminal_channel_temperatures(
        condition,
        temperature=1.0,
        atom_temperature=None,
        bond_temperature=0.8,
        decoration_temperature=0.6,
        atom_temperatures_by_origin=(0.4, 0.5, 0.7),
    )

    expected = np.ones(condition.node_count)
    for role, temperature in zip(ROLE_NAMES, (0.4, 0.5, 0.7), strict=True):
        expected[condition.origin_states == ORIGIN_TO_INDEX[role]] = temperature
    assert np.allclose(atom, expected)
    assert bond == pytest.approx(0.8)
    assert decoration == pytest.approx(0.6)


def test_expanded_sparse_decorations_roundtrip_and_backpropagate() -> None:
    record, vocabulary, expected = _expanded_multi_decoration_record()
    target = record.target
    sample = UgiChemistrySample(
        atom_states=target.atom_states.copy(),
        parent_bond_states=target.parent_bond_states.copy(),
        closure_bond_states=target.closure_bond_states.copy(),
        decoration_anchor=0,
        decoration_anchors=target.decorations.anchor_indices.copy() + 1,
        decoration_atom_states=target.decorations.atom_states.copy(),
        decoration_bond_states=target.decorations.bond_states.copy(),
    )
    molecule = chemistry_sample_to_molecule(record.condition, sample, vocabulary)
    assert Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False) == expected

    batch = collate_ugi_chemistry_records(
        (record,),
        maximum_nodes=record.condition.node_count,
        maximum_closures=record.condition.closure_count,
        maximum_decorations=5,
    )
    sources = chemistry_source_marginals(
        (record,),
        atom_classes=len(vocabulary),
        bond_classes=4,
        probability_floor=1e-5,
        maximum_decorations=5,
    )
    assert sources["decoration"][0] > sources["decoration"][1]
    generator = torch.Generator().manual_seed(101)
    noisy = noise_ugi_chemistry_batch(
        batch,
        {key: torch.as_tensor(value, dtype=torch.float32) for key, value in sources.items()},
        torch.tensor([0.5]),
        generator,
    )
    model = UgiChemistryFlow(
        atom_classes=len(vocabulary),
        bond_classes=4,
        maximum_nodes=record.condition.node_count,
        maximum_distance=85,
        maximum_decorations=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
    )
    predictions = model(
        nodes=noisy["nodes"],
        parent_bonds=noisy["parent_bonds"],
        closure_bonds=noisy["closure_bonds"],
        t=torch.tensor([0.5]),
        topology=batch,
    )
    loss, metrics = ugi_chemistry_flow_loss(predictions, batch)
    loss.backward()

    assert predictions["decoration_anchors"].shape == (1, 5, record.condition.node_count + 1)
    assert metrics["decoration_atom_ce"] > 0
    assert torch.isfinite(loss)
