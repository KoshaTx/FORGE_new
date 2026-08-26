from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.corpus.synthesis_program_training import load_synthesis_program_training_cache
from forge.model.defog_feasibility import AtomState
from forge.model.reaction_program_flow import (
    collate_synthesis_program_layouts,
    collate_synthesis_program_records,
    decode_synthesis_program_argmax,
    noise_synthesis_program_batch,
    synthesis_program_flow_loss,
)
from forge.model.sparse_topology_feasibility import (
    SparseGraphRecord,
    _masked_sparse_losses,
    _noise_sparse_batch,
)
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)
from forge.model.synthesis_program_sampling import decode_synthesis_program_strict_argmax
from forge.model.synthesis_program_training import synthesis_program_reconstruction_metrics
from forge.model.tensor_checkpoint import decode_tensor_state, encode_tensor_state

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results/phase1/shared_synthesis_program_training_integration_v1"


def _cache():
    return load_synthesis_program_training_cache(RESULTS / "cache.json")


def _fake_predictions(clean: dict[str, torch.Tensor], node_classes: int) -> dict[str, torch.Tensor]:
    batch, nodes = clean["nodes"].shape
    closures = clean["closure_bonds"].shape[1]
    return {
        "nodes": torch.zeros((batch, nodes, node_classes)),
        "parents": torch.zeros((batch, nodes, nodes)),
        "parent_bonds": torch.zeros((batch, nodes, 4)),
        "closure_left": torch.zeros((batch, closures, nodes)),
        "closure_right": torch.zeros((batch, closures, nodes)),
        "closure_bonds": torch.zeros((batch, closures, 4)),
    }


def _target_predictions(
    clean: dict[str, torch.Tensor], node_classes: int
) -> dict[str, torch.Tensor]:
    predictions = _fake_predictions(clean, node_classes)
    for field in predictions:
        predictions[field].scatter_(-1, clean[field].unsqueeze(-1), 10.0)
    return predictions


def test_shared_cache_is_equal_mass_and_preserves_only_ugi_fixed_states() -> None:
    cache = _cache()
    assert [record.program_id for record in cache.records] == list(
        cache.vocabulary.program_states[1:]
    )
    assert np.allclose(cache.sampling_weights, np.full(3, 1 / 3))
    assert cache.source_folds == ("train", "train", "train")
    ugi = next(record for record in cache.records if record.program_id == "ugi_3cr_agile")
    assert int(ugi.fixed_atom_mask.sum()) == 5
    assert int(ugi.fixed_parent_bond_mask.sum()) == 7
    for record in cache.records:
        if record is ugi:
            continue
        assert not record.fixed_atom_mask.any()
        assert not record.fixed_parent_bond_mask.any()
        assert not record.fixed_closure_bond_mask.any()


def test_sampling_layout_scrubs_every_variable_target_and_keeps_fixed_states() -> None:
    cache = _cache()
    clean = collate_synthesis_program_records(cache.records, maximum_closures=3)
    layout = collate_synthesis_program_layouts(cache.records, maximum_closures=3)
    for field, variable_mask in {
        "nodes": "atom_variable_mask",
        "parents": "parent_variable_mask",
        "parent_bonds": "parent_bond_variable_mask",
        "closure_left": "closure_endpoint_variable_mask",
        "closure_right": "closure_endpoint_variable_mask",
        "closure_bonds": "closure_bond_variable_mask",
    }.items():
        assert torch.count_nonzero(layout[field][layout[variable_mask]]) == 0
    for field, fixed_mask in {
        "nodes": "fixed_atom_mask",
        "parents": "fixed_parent_mask",
        "parent_bonds": "fixed_parent_bond_mask",
        "closure_left": "fixed_closure_endpoint_mask",
        "closure_right": "fixed_closure_endpoint_mask",
        "closure_bonds": "fixed_closure_bond_mask",
    }.items():
        assert torch.equal(layout[field][layout[fixed_mask]], clean[field][clean[fixed_mask]])


def test_component_coordinates_express_repeat_equivalence_without_component_identity() -> None:
    cache = _cache()
    batch = collate_synthesis_program_records(cache.records, maximum_closures=3)
    for row_index, record in enumerate(cache.records):
        role_counts = {
            role: sum(block.role_state == role for block in record.component_blocks)
            for role in set(block.role_state for block in record.component_blocks)
        }
        for component_index, block in enumerate(record.component_blocks, start=1):
            observed = slice(block.start, block.stop)
            assert torch.equal(
                batch["component_instance_states"][row_index, observed],
                torch.full((block.atom_count,), component_index),
            )
            assert torch.equal(
                batch["component_position_states"][row_index, observed],
                torch.arange(1, block.atom_count + 1),
            )
            expected_group = block.role_state if role_counts[block.role_state] > 1 else 0
            assert torch.equal(
                batch["repeat_group_states"][row_index, observed],
                torch.full((block.atom_count,), expected_group),
            )
    assert "component_id" not in batch


def test_sampling_layout_can_mismatch_program_and_roles_without_changing_graph_support() -> None:
    cache = _cache()
    record = next(row for row in cache.records if row.program_id == "ugi_3cr_agile")
    factual = collate_synthesis_program_layouts((record,), maximum_closures=3)
    roles = sorted(set(int(value) for value in record.role_states if int(value) > 0))
    role_mapping = {value: roles[(index + 1) % len(roles)] for index, value in enumerate(roles)}
    mapped = collate_synthesis_program_layouts(
        (record,),
        maximum_closures=3,
        conditioning_mode="program_mapped",
        program_state_mapping={
            cache.vocabulary.program_to_index["ugi_3cr_agile"]: cache.vocabulary.program_to_index[
                "bl_2023_repeated_aza_michael"
            ]
        },
        role_state_mapping=role_mapping,
    )

    assert not torch.equal(mapped["program_states"], factual["program_states"])
    assert not torch.equal(mapped["role_states"], factual["role_states"])
    for field in (
        "core_position_states",
        "program_depths",
        "node_mask",
        "child_mask",
        "closure_mask",
        "nodes",
        "parents",
        "parent_bonds",
        "closure_left",
        "closure_right",
        "closure_bonds",
    ):
        assert torch.equal(mapped[field], factual[field])

    with pytest.raises(ValueError, match="null state"):
        collate_synthesis_program_layouts(
            (record,),
            maximum_closures=3,
            conditioning_mode="program_mapped",
            role_state_mapping={0: 1},
        )


def test_shared_noising_and_decoder_leave_ugi_fixed_states_exact() -> None:
    cache = _cache()
    clean = collate_synthesis_program_records(cache.records, maximum_closures=3)
    node_p0 = torch.full((len(cache.atom_vocabulary),), 1 / len(cache.atom_vocabulary))
    bond_p0 = torch.full((4,), 0.25)
    noisy = noise_synthesis_program_batch(
        clean,
        node_p0,
        bond_p0,
        torch.full((3,), 0.2),
        torch.Generator().manual_seed(91),
    )
    predictions = _fake_predictions(clean, len(cache.atom_vocabulary))
    decoded = decode_synthesis_program_argmax(predictions, clean)
    for state in (noisy, decoded):
        for field, mask_name in {
            "nodes": "fixed_atom_mask",
            "parents": "fixed_parent_mask",
            "parent_bonds": "fixed_parent_bond_mask",
            "closure_left": "fixed_closure_endpoint_mask",
            "closure_right": "fixed_closure_endpoint_mask",
            "closure_bonds": "fixed_closure_bond_mask",
        }.items():
            mask = clean[mask_name]
            assert torch.equal(state[field][mask], clean[field][mask])
    assert torch.equal(decoded["parent_bonds"][:, 0], clean["parent_bonds"][:, 0])


def test_strict_decoder_recovers_feasible_targets_without_repair() -> None:
    cache = _cache()
    records = cache.records
    clean = collate_synthesis_program_records(records, maximum_closures=3)
    layout = collate_synthesis_program_layouts(records, maximum_closures=3)
    predictions = _target_predictions(clean, len(cache.atom_vocabulary))
    terminal, reasons = decode_synthesis_program_strict_argmax(
        predictions,
        layout,
        records,
        cache.atom_vocabulary,
    )
    assert reasons == (None,) * len(records)
    for row, record in enumerate(records):
        for field in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        ):
            expected = getattr(record.graph, field if field != "nodes" else "node_states")
            count = record.graph.closure_count if field.startswith("closure") else record.node_count
            assert torch.equal(terminal[field][row, :count], torch.from_numpy(expected))


def test_strict_decoder_abstains_when_exact_closure_support_is_impossible() -> None:
    graph = SparseGraphRecord(
        structure_id="impossible-closure",
        canonical_smiles="CC",
        node_states=np.asarray([0, 0], dtype=np.int64),
        parents=np.asarray([0, 0], dtype=np.int64),
        parent_bonds=np.asarray([0, 0], dtype=np.int64),
        closure_left=np.asarray([0], dtype=np.int64),
        closure_right=np.asarray([1], dtype=np.int64),
        closure_bonds=np.asarray([0], dtype=np.int64),
        edges=np.zeros((2, 2), dtype=np.int8),
    )
    record = SynthesisProgramGraphRecord(
        graph=graph,
        canonical_atom_order=np.asarray([0, 1], dtype=np.int64),
        program_id="test",
        program_state=1,
        program_depth=1,
        role_states=np.asarray([1, 1], dtype=np.int64),
        core_position_states=np.asarray([1, 1], dtype=np.int64),
        component_blocks=(SynthesisProgramComponentBlock("tail", 1, 0, 2),),
        fixed_atom_mask=np.asarray([False, False]),
        fixed_parent_bond_mask=np.asarray([False, False]),
        fixed_closure_bond_mask=np.asarray([False]),
    )
    layout = collate_synthesis_program_layouts((record,), maximum_closures=1)
    predictions = _fake_predictions(layout, node_classes=1)
    terminal, reasons = decode_synthesis_program_strict_argmax(
        predictions,
        layout,
        (record,),
        (AtomState("C", 0, False, 0),),
    )
    assert reasons == ("closure_pair_unavailable",)
    assert terminal["nodes"].shape == layout["nodes"].shape


def test_zero_fixed_auxiliary_noising_and_loss_equal_generic_sparse_flow() -> None:
    cache = _cache()
    auxiliary = tuple(record for record in cache.records if not record.fixed_atom_mask.any())
    clean = collate_synthesis_program_records(auxiliary, maximum_closures=3)
    node_p0 = torch.full((len(cache.atom_vocabulary),), 1 / len(cache.atom_vocabulary))
    bond_p0 = torch.full((4,), 0.25)
    t = torch.full((2,), 0.37)
    shared = noise_synthesis_program_batch(
        clean, node_p0, bond_p0, t, torch.Generator().manual_seed(92)
    )
    generic = _noise_sparse_batch(clean, node_p0, bond_p0, t, torch.Generator().manual_seed(92))
    assert all(torch.equal(shared[key], generic[key]) for key in shared)
    predictions = _fake_predictions(clean, len(cache.atom_vocabulary))
    shared_loss, shared_metrics = synthesis_program_flow_loss(predictions, clean)
    generic_loss, generic_metrics = _masked_sparse_losses(predictions, clean)
    assert torch.equal(shared_loss, generic_loss)
    assert shared_metrics == generic_metrics


def test_integration_result_passes_without_authorizing_production() -> None:
    cache = json.loads((RESULTS / "cache_result.json").read_text())
    training = json.loads((RESULTS / "training/result.json").read_text())
    sampling = json.loads((RESULTS / "sampling/result.json").read_text())
    qualification = json.loads((RESULTS / "qualification/result.json").read_text())
    assert cache["status"] == training["status"] == sampling["status"] == "pass"
    assert cache["ugi_regression"]["atom_states_exact"] is True
    assert cache["ugi_regression"]["bond_states_and_endpoints_exact"] is True
    assert training["zero_fixed_equivalence"] == {
        "programs": [
            "bl_2023_repeated_aza_michael",
            "lx_2024_repeated_reductive_amination",
        ],
        "noise_tensors_exact": True,
        "loss_exact": True,
        "metric_values_exact": True,
    }
    assert sampling["metrics"]["fixed_state_failures"] == 0
    assert sampling["metrics"]["exact_target_graph"] == 12
    assert qualification["status"] == "integration_gate_pass"
    assert qualification["summary"]["production_training_authorized"] is False


def test_json_tensor_checkpoint_codec_roundtrips_without_pickle() -> None:
    state = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    restored = decode_tensor_state(encode_tensor_state(state))
    assert set(restored) == {"weight"}
    assert torch.equal(restored["weight"], state["weight"])


def test_shared_reconstruction_metric_requires_every_active_tensor_to_match() -> None:
    cache = _cache()
    clean = collate_synthesis_program_records(cache.records, maximum_closures=3)
    predictions = _fake_predictions(clean, len(cache.atom_vocabulary))
    for field in (
        "nodes",
        "parents",
        "parent_bonds",
        "closure_left",
        "closure_right",
        "closure_bonds",
    ):
        predictions[field].scatter_(-1, clean[field].unsqueeze(-1), 10.0)

    exact = synthesis_program_reconstruction_metrics(predictions, clean)
    predictions["nodes"][0, 0].zero_()
    predictions["nodes"][0, 0, (int(clean["nodes"][0, 0]) + 1) % len(cache.atom_vocabulary)] = 10
    changed = synthesis_program_reconstruction_metrics(predictions, clean)

    assert exact["exact_tensor_fraction"] == 1.0
    assert exact["fixed_states_exact"] is True
    assert changed["exact_tensor_records"] == exact["exact_tensor_records"] - 1
