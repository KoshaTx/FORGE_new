from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from forge.corpus.synthesis_program_training import load_synthesis_program_training_cache
from forge.model.reaction_program_flow import (
    collate_synthesis_program_records,
    derive_role_morphology_states,
)
from forge.model.reaction_program_transformer import synthesis_program_offspring_targets
from forge.model.synthesis_program_sampling import (
    COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
    _terminal_role_morphology,
    sample_synthesis_program_products,
)
from forge.model.ugi_transformer_topology import (
    UgiTransformerTopologyPolicy,
    decode_ugi_exact_topology,
)

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "results/phase1/shared_synthesis_program_training_integration_v1/cache.json"


def _ugi_record():
    cache = load_synthesis_program_training_cache(CACHE)
    record = next(item for item in cache.records if item.program_id == "ugi_3cr_agile")
    return replace(record, role_morphology_states=derive_role_morphology_states(record))


def _policy() -> UgiTransformerTopologyPolicy:
    return UgiTransformerTopologyPolicy.from_mapping(
        {
            "allowed_ring_sizes": [5, 6, 7],
            "maximum_heavy_degree": 4,
            "maximum_adjacent_branch_run_by_role": {
                "amine_head": 2,
                "oxoester_aldehyde_body_tail": 1,
                "isocyanide_tail": 1,
            },
        }
    )


def _target_predictions(record):
    clean = collate_synthesis_program_records((record,), maximum_closures=3)
    offspring, _ = synthesis_program_offspring_targets(clean, maximum_children=3)
    offspring_logits = torch.full((1, record.node_count, 4), -30.0)
    offspring_logits.scatter_(2, offspring[:, : record.node_count, None], 30.0)
    left = torch.full((1, 3, record.node_count), -30.0)
    right = torch.full_like(left, -30.0)
    for slot, (left_node, right_node) in enumerate(
        zip(record.graph.closure_left, record.graph.closure_right, strict=True)
    ):
        left[0, slot, int(left_node)] = 30.0
        right[0, slot, int(right_node)] = 30.0
    return {"offspring": offspring_logits, "closure_left": left, "closure_right": right}


def test_exact_transformer_topology_round_trips_a_supported_ugi_program() -> None:
    record = _ugi_record()
    predictions = _target_predictions(record)
    first = decode_ugi_exact_topology(
        predictions,
        index=0,
        record=record,
        policy=_policy(),
        generator=torch.Generator().manual_seed(101),
    )
    second = decode_ugi_exact_topology(
        predictions,
        index=0,
        record=record,
        policy=_policy(),
        generator=torch.Generator().manual_seed(101),
    )

    assert np.array_equal(first.parents, record.graph.parents)
    assert np.array_equal(first.closure_left, record.graph.closure_left)
    assert np.array_equal(first.closure_right, record.graph.closure_right)
    assert np.array_equal(first.parents, second.parents)
    assert np.array_equal(first.closure_left, second.closure_left)
    assert np.array_equal(first.closure_right, second.closure_right)

    observed = _terminal_role_morphology(
        record,
        parents=first.parents,
        closure_left=first.closure_left,
        closure_right=first.closure_right,
    )
    expected = {
        role_state: tuple(int(value) - 1 for value in values[0])
        for role_state in sorted(set(int(value) for value in record.role_states if value > 0))
        if (values := np.unique(record.role_morphology_states[record.role_states == role_state], axis=0)).shape
        == (1, 4)
    }
    assert observed == expected


def test_exact_transformer_topology_rejects_an_unpinned_role_policy() -> None:
    with pytest.raises(ValueError, match="every precursor role"):
        UgiTransformerTopologyPolicy.from_mapping(
            {
                "allowed_ring_sizes": [5, 6],
                "maximum_heavy_degree": 4,
                "maximum_adjacent_branch_run_by_role": {"amine_head": 1},
            }
        )


class _TargetTopologyModel:
    maximum_closures = 3

    def __init__(self, record, node_classes: int) -> None:
        self.record = record
        self.node_classes = node_classes
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, **inputs):
        self.calls += 1
        batch, nodes = inputs["nodes"].shape
        clean = collate_synthesis_program_records((self.record,), maximum_closures=3)

        def categorical(target, classes):
            logits = torch.full((batch, *target.shape[1:], classes), -30.0)
            expanded = target.expand(batch, *target.shape[1:])
            return logits.scatter(-1, expanded.unsqueeze(-1), 30.0)

        parent_logits = torch.full((batch, nodes, nodes), -30.0)
        parent_logits.scatter_(
            2,
            clean["parents"].expand(batch, -1).unsqueeze(-1),
            30.0,
        )
        closure_left = torch.full((batch, 3, nodes), -30.0)
        closure_right = torch.full_like(closure_left, -30.0)
        closure_left.scatter_(
            2,
            clean["closure_left"].expand(batch, -1).unsqueeze(-1),
            30.0,
        )
        closure_right.scatter_(
            2,
            clean["closure_right"].expand(batch, -1).unsqueeze(-1),
            30.0,
        )
        offspring, _ = synthesis_program_offspring_targets(clean, maximum_children=3)
        return {
            "nodes": categorical(clean["nodes"], self.node_classes),
            "parents": parent_logits,
            "parent_bonds": categorical(clean["parent_bonds"], 4),
            "closure_left": closure_left,
            "closure_right": closure_right,
            "closure_bonds": categorical(clean["closure_bonds"], 4),
            "offspring": categorical(offspring, 4),
        }


def test_coupled_sampler_conditions_chemistry_on_exact_topology_without_repair() -> None:
    cache = load_synthesis_program_training_cache(CACHE)
    record = _ugi_record()
    model = _TargetTopologyModel(record, len(cache.atom_vocabulary))
    node_marginal = np.full(len(cache.atom_vocabulary), 1 / len(cache.atom_vocabulary))
    bond_marginal = np.full(4, 0.25)

    rows, receipt = sample_synthesis_program_products(
        model,
        (record,),
        cache.atom_vocabulary,
        node_marginal,
        bond_marginal,
        samples_per_program=1,
        sample_steps=2,
        batch_size=1,
        seed=211,
        device="cpu",
        terminal_decode_policy=COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
        ugi_topology_policy=_policy(),
    )

    assert model.calls == 4  # two flow steps, terminal prediction, topology-conditioned chemistry
    assert rows[0]["valid"] is True
    assert rows[0]["exact_target_graph"] is True
    assert receipt["strict_constraint_abstentions"] == 0
    assert receipt["topology_coupling_second_pass_applied"] is True
    assert receipt["topology_selection"] == "exact_program_conditional_sample_then_argmax_chemistry"
    assert receipt["topology_seed"] == 212
    assert receipt["repairs"] == {}
