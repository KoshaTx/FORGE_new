from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.phase1.multireaction.combinatorial_reuse_pilot import validate_config
from forge.assembly.families import load_assembly_libraries
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.precursor_reuse import (
    PrecursorReuseError,
    PrecursorReuseFlow,
    PrecursorReusePlan,
    ReuseSamplingView,
    group_tensor,
    pool_corresponding,
    qualify_reuse,
)
from forge.model.reaction_program_flow import (
    collate_synthesis_program_layouts,
    derive_role_morphology_states,
)
from forge.model.synthesis_program_sampling import (
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    config = json.loads(
        (REPO / "configs/multireaction/library_program_dataset_v1.json").read_text()
    )
    libraries = load_assembly_libraries(
        [
            (REPO / config["inputs"][k]["path"], config["inputs"][k]["sha256"])
            for k in config["registries"]
        ],
        expected_families=config["programs"],
    )
    model, vocabulary, atoms, nodes, bonds, _ = load_synthesis_program_checkpoint(
        REPO / "results/phase1/combinatorial_checkpoint_v1/checkpoint.json", device="cpu"
    )
    with SynthesisProgramProductionCache(
        REPO / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        candidates = [
            cache.record(int(i))
            for i in cache.indices(program_id="aza_michael_amine_acrylate", fold="train")[:100]
        ]
    chosen = None
    for record in candidates:
        plan = qualify_reuse(
            record,
            libraries[record.program_id],
            config["programs"][record.program_id],
            config["limits"],
        )
        if plan.status == "qualified":
            chosen = record, plan
            break
    assert (
        chosen is not None
    ), "no qualified repeated positive control in the first 100 train records"
    yield model, atoms, nodes, bonds, *chosen, config, libraries
    torch.set_num_threads(previous)


def test_pool_keeps_examples_separate_and_is_invariant_to_copy_order():
    hidden = torch.tensor([[[1.0], [3.0], [5.0]], [[100.0], [200.0], [900.0]]], requires_grad=True)
    groups = torch.tensor([[1, 1, 0], [1, 1, 0]])
    pooled = pool_corresponding(hidden, groups)
    assert pooled[:, :2, 0].tolist() == [[2.0, 2.0], [150.0, 150.0]]
    order = torch.tensor([1, 0, 2])
    assert torch.equal(pool_corresponding(hidden[:, order], groups[:, order]), pooled[:, order])
    pooled[:, 0].sum().backward()
    assert hidden.grad[:, :2].tolist() == [[[0.5], [0.5]], [[0.5], [0.5]]]


@pytest.mark.parametrize("groups", [(1, 0), (2, 2), (-1, -1), (True, True)])
def test_invalid_or_singleton_relations_fail(groups):
    with pytest.raises(PrecursorReuseError):
        PrecursorReusePlan("x", groups, "qualified", 1)


def test_qualified_relation_requires_exact_source_reuse(example):
    _, _, _, _, record, plan, config, libraries = example
    assert plan.exact_source_programs > 0
    assert max(plan.groups) > 0
    limits = {**config["limits"], "maximum_expansions": 1}
    blocked = qualify_reuse(
        record, libraries[record.program_id], config["programs"][record.program_id], limits
    )
    assert blocked.status == "source_reuse_not_verified"
    assert not any(blocked.groups)


def test_depth_one_does_not_infer_equality_from_repeated_roles(example):
    *_, record, plan, config, libraries = example
    record = replace(record, program_depth=1)
    result = qualify_reuse(
        record,
        libraries[record.program_id],
        config["programs"][record.program_id],
        config["limits"],
    )
    assert result.status == "not_repeated" and not any(result.groups)


def _forward(model, layout, groups, share):
    return model(reuse_groups=groups, share=share, t=torch.tensor([0.5]), **layout)


def test_zero_residual_reproduces_base_and_relation_path_has_gradients(example):
    base, _, _, _, record, plan, *_ = example
    model = PrecursorReuseFlow(copy.deepcopy(base)).eval()
    layout = collate_synthesis_program_layouts([record], maximum_closures=1)
    fields = {
        k: v
        for k, v in layout.items()
        if k
        in {
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
            "node_mask",
            "child_mask",
            "closure_mask",
            "program_states",
            "role_states",
            "core_position_states",
            "program_depths",
            "adapter_mask",
            "repeat_group_states",
            "component_position_states",
            "component_instance_states",
            "role_morphology_states",
        }
    }
    groups = group_tensor([plan], record.node_count)
    predictions = _forward(model, fields, groups, True)
    reference = base(t=torch.tensor([0.5]), **fields)
    assert all(torch.equal(predictions[k], reference[k]) for k in predictions)
    predictions["nodes"].square().sum().backward()
    assert model.reuse_projection.weight.grad.abs().sum() > 0


def test_changing_other_copy_affects_shared_context_but_not_self_context(example):
    base, _, _, _, record, plan, *_ = example
    groups = group_tensor([plan], record.node_count)
    own = torch.randn((1, record.node_count, base.backbone.hidden_dim))
    indices = torch.where(groups[0] == 1)[0]
    changed = own.clone()
    changed[0, indices[1]] += 3
    before, after = (pool_corresponding(h, groups) for h in (own, changed))
    assert not torch.equal(before[0, indices[0]], after[0, indices[0]])
    assert torch.equal(own[0, indices[0]], changed[0, indices[0]])


def test_sampler_sidecar_binding_rejects_wrong_record_and_reuse(example):
    base, _, _, _, record, plan, *_ = example
    model = PrecursorReuseFlow(copy.deepcopy(base))
    with pytest.raises(PrecursorReuseError, match="identity"):
        ReuseSamplingView(
            model, [record], [replace(plan, record_id="wrong")], batch_size=1, share=True
        )
    view = ReuseSamplingView(model, [record], [plan], batch_size=1, share=True)
    expected, _ = view.batches[0]
    changed = dict(expected)
    changed["program_depths"] = changed["program_depths"] + 1
    with pytest.raises(PrecursorReuseError, match="semantics"):
        view.prepare_program_memory(**changed)
    memory = view.prepare_program_memory(**expected)
    assert memory["groups"].shape == (1, record.node_count)
    view.assert_consumed()
    with pytest.raises(PrecursorReuseError, match="exhausted"):
        view.prepare_program_memory(**expected)
    with pytest.raises(PrecursorReuseError, match="unbound"):
        view(program_memory={})


def test_sampler_does_not_copy_qualified_source_atoms_or_pointers(example):
    base, atoms, nodes, bonds, record, plan, *_ = example
    record = replace(record, role_morphology_states=derive_role_morphology_states(record))
    states = record.graph.node_states.copy()
    states[~record.fixed_atom_mask] = (states[~record.fixed_atom_mask] + 1) % len(atoms)
    parents = record.graph.parents.copy()
    variable = ~record.fixed_parent_bond_mask
    replacement = np.maximum(np.arange(record.node_count) - 1, 0)
    parents[variable] = replacement[variable]
    changed = replace(
        record,
        graph=replace(record.graph, node_states=states, parents=parents, canonical_smiles="CC"),
    )
    network = PrecursorReuseFlow(copy.deepcopy(base))
    torch.nn.init.normal_(network.reuse_projection.weight, std=0.01)
    outputs = []
    for r in (record, changed):
        view = ReuseSamplingView(network, [r], [plan], batch_size=1, share=True)
        rows, report = sample_synthesis_program_products(
            view,
            [r],
            atoms,
            nodes,
            bonds,
            samples_per_program=1,
            sample_steps=2,
            batch_size=1,
            seed=17,
            device="cpu",
            terminal_decode_policy="strict_valence_topology_argmax",
        )
        view.assert_consumed()
        assert report["fixed_state_failures"] == 0
        outputs.append(rows[0]["canonical_smiles"])
    assert outputs[0] == outputs[1]


def test_experiment_contract_rejects_gate_changes_or_extra_budget():
    config = json.loads(
        (REPO / "configs/multireaction/combinatorial_reuse_pilot_v1.json").read_text()
    )
    validate_config(config)
    changed = copy.deepcopy(config)
    changed["policy"]["gate_changes"] = True
    with pytest.raises(PrecursorReuseError):
        validate_config(changed)
    config["training"]["steps"] = 769
    with pytest.raises(PrecursorReuseError):
        validate_config(config)


def test_symmetric_correspondence_abstains_without_losing_the_record(example):
    *_, config, libraries = example
    with SynthesisProgramProductionCache(
        REPO / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        records = [
            cache.record(int(i))
            for i in cache.indices(program_id="aza_michael_amine_acrylate", fold="train")[:100]
        ]
    plans = [
        qualify_reuse(
            r, libraries[r.program_id], config["programs"][r.program_id], config["limits"]
        )
        for r in records
    ]
    ambiguous = [p for p in plans if p.status == "ambiguous_atom_correspondence"]
    assert ambiguous and all(not any(p.groups) for p in ambiguous)
    assert len(plans) == len(records)


def test_correspondence_handles_different_serialization_orders(example):
    _, _, _, _, record, plan, *_ = example
    blocks = [b for b in record.component_blocks if any(plan.groups[b.start : b.stop])]
    assert len(blocks) == record.program_depth
    assert len({plan.groups[b.start : b.stop] for b in blocks}) > 1
    for group in set(plan.groups) - {0}:
        positions = np.where(np.asarray(plan.groups) == group)[0]
        assert len(set(record.graph.node_states[positions])) == 1
        assert len(set(record.core_position_states[positions])) == 1


def test_saved_pilot_recomputes_all_attempts_and_qualified_relations():
    from experiments.phase1.multireaction.combinatorial_reuse_verify import verify

    result = REPO / "results/phase1/combinatorial_reuse_pilot_v1/result.json"
    if not result.exists():
        pytest.skip("local reuse pilot artifact is absent")
    checked = verify(REPO, result)
    assert checked["attempts_recomputed"] == 4 * 12 * 64
    assert checked["checkpoint_reloads_verified"] == 2


@pytest.mark.parametrize("kind", ["promotion", "inputs", "decision"])
def test_verifier_rejects_forged_pilot_claims(tmp_path, kind):
    from experiments.phase1.multireaction.combinatorial_reuse_verify import verify

    source = REPO / "results/phase1/combinatorial_reuse_pilot_v1/result.json"
    if not source.exists():
        pytest.skip("local reuse pilot artifact is absent")
    result = json.loads(source.read_text())
    if kind == "promotion":
        result["model_promoted"] = True
    elif kind == "inputs":
        result["inputs"]["checkpoint"] = result["inputs"]["cache"]
    else:
        result["observed_preservation_rule_passed"] = not result[
            "observed_preservation_rule_passed"
        ]
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(result))
    with pytest.raises(PrecursorReuseError):
        verify(REPO, path)
