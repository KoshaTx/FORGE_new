"""Restored losses, source conditions and strict decoding preserve their contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from rdkit import Chem

from forge.corpus.compose_lipid_restored_prior import smooth_sources
from forge.corpus.qualified_program_cache import (
    QualifiedProgramExample,
    collate_qualified_program_examples,
)
from forge.model.compose_lipid_core import (
    condition_on_qualified_core,
    core_condition,
    layout_with_sampled_core,
    sample_core_condition,
)
from forge.model.compose_lipid_training import compose_lipid_backward, validate_objective
from forge.model.synthesis_program_sampling import (
    _terminal_smiles,
    decode_synthesis_program_strict_argmax,
)
from forge.model.synthesis_program_training import build_synthesis_program_flow
from tests.test_source_instance_coordinates import record

OBJECTIVE = dict(
    offspring_weight=1.0,
    junction_consistency_weight=0.5,
    topology_conditioned_chemistry_weight=1.0,
    chemistry_loss_balancing="equal_present_role_mass",
    gradient_balancing="pcgrad",
    pcgrad_backend="sequential",
)


def source():
    # A ring touches a core atom: a blanket exterior-only closure policy would be wrong.
    r, vocab, atoms = record("N1CCCCC1CC", ["head"] * 8, ["center"] + ["exterior"] * 7)
    return QualifiedProgramExample(r, "family", (("head", "id", 1),), ()), vocab, atoms


def test_qualified_core_keeps_gold_graph_and_every_exterior_variable():
    e, _, _ = source()
    marked = condition_on_qualified_core(e)
    assert marked.record.graph is e.record.graph
    assert np.array_equal(marked.record.fixed_atom_mask, e.record.core_position_states > 1)
    assert not marked.record.fixed_parent_bond_mask.any()


def test_sampled_core_layout_is_independent_of_variable_target_values():
    e, _, _ = source()
    key, value = core_condition(e.record)
    bank = {json.dumps(key, separators=(",", ":")): [dict(value, mass=1.0)]}
    core, units = sample_core_condition(e.record, bank, np.random.default_rng(3))
    expected, reason = layout_with_sampled_core(e, core)
    assert reason is None and (units >= 0).sum() == 1
    poisoned = replace(
        e,
        record=replace(
            e.record,
            graph=replace(
                e.record.graph,
                node_states=np.full(e.record.node_count, 999),
                parents=np.zeros(e.record.node_count, dtype=np.int64),
                parent_bonds=np.full(e.record.node_count, 999),
            ),
        ),
    )
    actual, reason = layout_with_sampled_core(poisoned, core)
    assert reason is None
    for name in ("node_states", "parents", "parent_bonds"):
        np.testing.assert_array_equal(
            getattr(actual.record.graph, name), getattr(expected.record.graph, name)
        )


def test_strict_decoder_retains_ring_touching_core_and_exact_atom_count():
    e, _, atoms = source()
    e = condition_on_qualified_core(e)
    batch = collate_qualified_program_examples([e], maximum_closures=2)
    key, value = core_condition(e.record)
    _, units = sample_core_condition(
        e.record,
        {json.dumps(key, separators=(",", ":")): [dict(value, mass=1.0)]},
        np.random.default_rng(0),
    )
    classes = {
        "nodes": len(atoms),
        "parent_bonds": 3,
        "closure_bonds": 3,
        "parents": e.record.node_count,
        "closure_left": e.record.node_count,
        "closure_right": e.record.node_count,
    }
    predictions = {
        k: torch.nn.functional.one_hot(batch[k], n).float() * 100 for k, n in classes.items()
    }
    terminal, reasons = decode_synthesis_program_strict_argmax(
        predictions,
        batch,
        [e.record],
        atoms,
        qualified_core_units=[units],
        confine_origin_edges=True,
        reserve_fixed_closures=True,
    )
    assert reasons == (None,)
    assert _terminal_smiles(
        terminal, 0, e.record.node_count, e.record.graph.closure_count, atoms
    ) == Chem.MolToSmiles(Chem.MolFromSmiles(e.record.graph.canonical_smiles))


def test_smoothed_sources_keep_unsupported_cells_positive_and_normalized():
    counts = np.zeros((3, 4, 3))
    counts[1, 2] = [3, 7, 0]
    counts[2, 3] = [0, 0, 4]
    p = smooth_sources(counts)
    assert (p > 0).all()
    np.testing.assert_allclose(p.sum(-1), 1)
    np.testing.assert_allclose(p[1, 1], p[1, 0])
    global_p = (counts.sum((0, 1)) + 1e-5) / (counts.sum() + 3e-5)
    program_p = (global_p + counts[1].sum(0)) / (1 + counts[1].sum())
    np.testing.assert_allclose(p[1, 2], (program_p + counts[1, 2]) / (1 + counts[1, 2].sum()))


def test_restored_objectives_produce_gradients_for_new_heads_and_families():
    e, vocab, atoms = source()
    batch = collate_qualified_program_examples([e, e], maximum_closures=2)
    batch["family_states"] = torch.tensor([1, 2])
    batch["repeat_atom_groups"] = torch.zeros_like(batch["nodes"])
    batch["repeat_bond_groups"] = torch.zeros_like(batch["nodes"])
    torch.set_num_threads(1)
    torch.manual_seed(48)
    model = build_synthesis_program_flow(
        vocabulary=vocab,
        node_classes=len(atoms),
        device="cpu",
        model_config=dict(
            architecture="reaction_program_graph_transformer",
            hidden_dim=16,
            layers=1,
            attention_heads=2,
            expert_count=2,
            adapter_dim=4,
            maximum_heavy_atoms=254,
            maximum_closures=2,
            dropout=0.0,
            bond_classes=3,
            maximum_children=253,
            role_morphology_conditioning=True,
            program_routed_output_heads=True,
        ),
    )
    loss, metrics = compose_lipid_backward(
        model,
        batch,
        objective=OBJECTIVE,
        architecture="reaction_program_graph_transformer",
        node_marginal=torch.ones(len(atoms)) / len(atoms),
        bond_marginal=torch.ones(3) / 3,
        times=torch.tensor([0.25, 0.6]),
        generator=torch.Generator().manual_seed(51),
        semantic_weights=dict(role_weight=0.25, core_weight=0.25, repeat_consistency_weight=0.25),
        repeat_supervision="exact_fragment",
    )
    assert torch.isfinite(loss) and "projected_conflicts" in metrics
    parameters = dict(model.named_parameters())
    for name in ("offspring_output.weight",):
        assert name in parameters
        assert parameters[name].grad is not None and parameters[name].grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in parameters.items()
        if n.startswith("terminal_chemistry_adapter")
    )


def test_objective_typo_cannot_silently_disable_restoration():
    with pytest.raises(ValueError):
        validate_objective(dict(OBJECTIVE, offspring_weigth=1.0))
