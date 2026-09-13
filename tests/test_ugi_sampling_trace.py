from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forge.model import synthesis_program_sampling as sampling
from forge.model.defog_feasibility import AtomState
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.synthesis_program_graph import tensorize_synthesis_program_product
from forge.model.ugi_sampling_trace import SamplingTrace, SamplingTraceError
from forge.model.ugi_transformer_topology import _sample_constructive_ester_offspring


def _topology_draw(generator):
    return _sample_constructive_ester_offspring(
        torch.arange(68, dtype=torch.float64).reshape(17, 4) / 50,
        generator=generator,
        minimum_side_carbons=6,
        minimum_long_side_carbons=10,
        exact_full_side_carbons=(6, 10),
    )


def test_topology_observation_preserves_draw_and_rng_state():
    pristine = torch.Generator().manual_seed(123)
    expected = _topology_draw(pristine)
    observed = torch.Generator().manual_seed(123)
    with SamplingTrace(selected_attempts=(0,)) as trace:
        actual = _topology_draw(observed)
    assert np.array_equal(actual, expected)
    assert torch.equal(pristine.get_state(), observed.get_state())
    event = next(event for event in trace.events if event["kind"] == "categorical_choice")
    assert len(event["candidates"]) == len(event["final_probabilities"])
    assert sum(event["final_probabilities"]) == pytest.approx(1)
    assert event["candidates"][event["selected_candidate_index"]]["identity"] == actual.tolist()
    assert all(len(candidate["identity_sha256"]) == 64 for candidate in event["candidates"])
    json.dumps({"events": trace.events, "summary": trace.summary}, allow_nan=False)


def test_numpy_observation_preserves_state_and_actual_masked_law():
    logits = np.asarray([2.0, -5.0, 1.0, 0.0])
    valid = np.asarray([True, False, True, True])
    pristine = np.random.default_rng(33)
    expected = sampling._sample_allowed(logits, valid, generator=pristine, temperature=0.8)
    observed = np.random.default_rng(33)
    with SamplingTrace() as trace:
        actual = sampling._sample_allowed(
            logits, valid, generator=trace.observe_generator(observed), temperature=0.8
        )
    assert actual == expected
    assert pristine.bit_generator.state == observed.bit_generator.state
    event = trace.events[0]
    assert event["allowed_states"] == [0, 2, 3]
    assert event["selected_state"] == actual
    assert len(event["final_probabilities"]) == 3


class _FrequencyPolicy:
    """Fixture law deliberately unrelated to scores, to detect recomputed tracing laws."""

    tail_unsaturation_count_strategy = "frequency_resampled"
    tail_unsaturation_count_tolerance = 1
    tail_unsaturation_position_strategy = "model_ranked"
    uses_ranked_terminal_bonds = True
    uses_joint_realism = False
    uses_local_chemistry_bonds = False
    local_chemistry_prior = SimpleNamespace(
        unsaturation_count_frequencies=lambda role: ((0, 0, 7), (1, 0, 3))
    )

    def tail_unsaturation_count_options(self, **kwargs):
        return ((0, 0), (1, 0))

    def probabilities(self, model, semantic, joint):
        # Deliberately nonuniform positional distribution with no extra RNG.
        values = np.arange(1, len(model) + 1, dtype=float)
        return values / values.sum()


def _tail_draw(generator, policy):
    count = 8
    parents = np.asarray([-1, 0, 1, 2, 3, 4, 5, 6])
    record = SimpleNamespace(
        program_id=sampling.UGI_PROGRAM_ID,
        node_count=count,
        core_position_states=np.ones(count, dtype=int),
        fixed_parent_bond_mask=np.asarray([True, *([False] * (count - 1))]),
        fixed_closure_bond_mask=np.asarray([], dtype=bool),
        graph=SimpleNamespace(structure_id="tail-observer-fixture"),
    )
    predictions = {
        "parent_bonds": np.arange(count * 3, dtype=float).reshape(1, count, 3),
        "closure_bonds": np.empty((1, 0, 3)),
    }
    target = SimpleNamespace(
        tail_pair=SimpleNamespace(
            aldehyde_carbon_carbon_double_bonds=1,
            aldehyde_carbon_carbon_triple_bonds=0,
            isocyanide_carbon_carbon_double_bonds=0,
            isocyanide_carbon_carbon_triple_bonds=0,
        )
    )
    return sampling._select_ugi_all_role_tail_bonds(
        predictions,
        0,
        record,
        (AtomState("C", 0, False, 0),),
        np.zeros(count, dtype=int),
        parents,
        np.asarray([], dtype=int),
        np.asarray([], dtype=int),
        np.full(count, 4),
        np.full(count, 8),
        np.asarray([2, 4, 6]),
        ("oxoester_aldehyde_body_tail",) * 4 + ("isocyanide_tail",) * 4,
        SimpleNamespace(allows_role_edge=lambda *args: True),
        {},
        target,
        semantic_guidance_policy=policy,
        terminal_generator=generator,
    )


def test_tail_trace_captures_actual_count_groups_and_conditional_positions():
    pristine = np.random.default_rng(47)
    expected = _tail_draw(pristine, _FrequencyPolicy())
    observed = np.random.default_rng(47)
    with SamplingTrace() as trace:
        actual = _tail_draw(trace.observe_generator(observed), _FrequencyPolicy())
    assert actual == expected
    assert pristine.bit_generator.state == observed.bit_generator.state
    groups = [event for event in trace.events if event["kind"] == "tail_count_group_choice"]
    assert len(groups) == 2
    assert {event["context"]["role"] for event in groups} == {
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    }
    for event in groups:
        assert event["group_keys"] == [[0, 0], [1, 0]]
        assert event["final_probabilities"] == [0.7, 0.3]
        assert event["selected_group"] == event["group_keys"][event["selected_group_index"]]
    choices = [event for event in trace.events if event["kind"] == "categorical_choice"]
    assert len(choices) == 2
    for event in choices:
        assert event["selected_candidate_index"] in event["draw_candidate_indices"]
        assert len(event["final_probabilities"]) == len(event["draw_candidate_indices"])
    assert len([event for event in trace.events if event["kind"] == "terminal_assignment"]) == 2


def test_deterministic_tail_trace_observes_both_roles_and_first_argmax():
    expected = _tail_draw(None, None)
    with SamplingTrace() as trace:
        actual = _tail_draw(None, None)
    assert actual == expected
    decisions = [event for event in trace.events if event["kind"] == "terminal_assignment"]
    assert len(decisions) == 2
    for event in decisions:
        scores = [candidate["neural_score"] for candidate in event["candidates"]]
        assert event["selected_candidate_index"] == int(np.argmax(scores))
        assert sum(event["final_probabilities"]) == 1
        assert event["selection_law"] == "first_neural_argmax"


def test_amine_candidate_internal_argmax_is_not_a_terminal_selection():
    block = SimpleNamespace(role="amine_head", start=0, stop=4, atom_count=4)
    record = SimpleNamespace(
        program_id=sampling.UGI_PROGRAM_ID,
        node_count=4,
        component_blocks=(block,),
        core_position_states=np.ones(4, dtype=int),
        fixed_atom_mask=np.zeros(4, dtype=bool),
        graph=SimpleNamespace(
            structure_id="amine-observer-fixture", node_states=np.zeros(4, dtype=int)
        ),
    )
    args = (
        {"nodes": np.arange(8, dtype=float).reshape(1, 4, 2)},
        0,
        record,
        (AtomState("C", 0, False, 0), AtomState("N", 0, False, 0)),
        np.asarray([8, 6]),
        np.full(4, 4),
        ({1}, {0, 2}, {1, 3}, {2}),
        (),
        (),
        ("amine_head",) * 4,
        SimpleNamespace(
            component_is_within_observed_support=lambda *args, **kwargs: True,
            allows_role_edge_for_any_bond=lambda *args: True,
            enforces_role_cycles=False,
        ),
        SimpleNamespace(
            reaction_id=sampling.UGI_PROGRAM_ID,
            amine_role="amine_head",
            minimum_amine_exterior_nitrogens=1,
            maximum_amine_exterior_nitrogens=1,
        ),
    )
    expected = sampling._select_ugi_amine_atom_states(*args)
    with SamplingTrace() as trace:
        actual = sampling._select_ugi_amine_atom_states(*args)
    assert actual == expected
    internal = [
        event for event in trace.events if event["kind"] == "candidate_internal_atom_argmax"
    ]
    assert len(internal) == 16
    assert len({event["context"]["candidate_symbol_assignment_sha256"] for event in internal}) == 4
    assert all(event["context"]["role"] == "amine_head" for event in internal)
    assert all(event["context"]["field"] == "nodes" for event in internal)
    assert not any(event["kind"] == "coordinate_argmax" for event in trace.events)
    assert len([event for event in trace.events if event["kind"] == "terminal_assignment"]) == 1


class _FixtureModel:
    maximum_closures = 1

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, **inputs):
        self.calls += 1
        batch, nodes = inputs["nodes"].shape
        return {
            "nodes": torch.zeros((batch, nodes, 2)),
            "parents": torch.zeros((batch, nodes, nodes)),
            "parent_bonds": torch.zeros((batch, nodes, 3)),
            "closure_left": torch.zeros((batch, 1, nodes)),
            "closure_right": torch.zeros((batch, 1, nodes)),
            "closure_bonds": torch.zeros((batch, 1, 3)),
        }


def _sampler_draw():
    vocabulary = ReactionProgramVocabulary.from_semantics(
        program_ids=("fixture",),
        roles=("body",),
        core_positions=("fixture:core",),
        maximum_steps=1,
    )
    atoms = (AtomState("C", 0, False, 0), AtomState("N", 0, False, 0))
    record = tensorize_synthesis_program_product(
        record_id="observer-fixture",
        program_id="fixture",
        canonical_product_smiles="CCCN",
        atom_roles=("body",) * 4,
        atom_core_positions=("fixture:core", "exterior", "exterior", "exterior"),
        program_depth=1,
        vocabulary=vocabulary,
        atom_vocabulary=atoms,
    )
    model = _FixtureModel()
    result = sampling.sample_synthesis_program_products(
        model,
        (record, record),
        atoms,
        np.full(2, 0.5),
        np.full(3, 1 / 3),
        samples_per_program=1,
        sample_steps=2,
        batch_size=2,
        seed=77,
        device="cpu",
        terminal_decode_policy="strict_valence_topology_argmax",
    )
    return result, model.calls


def test_endpoint_only_and_full_observer_match_pristine_sampler():
    pristine, pristine_calls = _sampler_draw()
    rng_binding, multinomial_binding = np.random.default_rng, torch.multinomial
    with SamplingTrace(selected_attempts=(0,), decisions=False) as control:
        assert np.random.default_rng is rng_binding
        assert torch.multinomial is multinomial_binding
        controlled, control_calls = _sampler_draw()
    with SamplingTrace(selected_attempts=(0,)) as traced:
        observed, traced_calls = _sampler_draw()
    assert pristine == controlled == observed
    assert pristine_calls == control_calls == traced_calls == 3
    assert control.endpoint_state == traced.endpoint_state
    assert control.endpoint_digest == traced.endpoint_digest
    snapshot = traced.endpoint_state["sampler_returns"][0]
    assert snapshot["batch_offset"] == 0
    assert snapshot["total_attempts"] == len(snapshot["terminal_states"]["nodes"]) == 2
    assert snapshot["generator_states"]["generator"]["state_hex"]
    outcomes = [event for event in traced.events if event["kind"] == "attempt_outcome"]
    assert len(outcomes) == 1 and outcomes[0]["context"]["attempt_index"] == 0
    coordinates = [event for event in traced.events if event["kind"] == "coordinate_argmax"]
    assert {event["context"]["field"] for event in coordinates} == {
        "nodes",
        "parents",
        "parent_bonds",
    }
    for event in coordinates:
        context = event["context"]
        assert context["selection_stage"] == "terminal_coordinate"
        if context["field"] == "parent_bonds":
            assert len(context["edge"]) == 2
            assert context["roles"] == ["body", "body"]
        else:
            assert context["role"] == "body"
    assert control.events == []


@pytest.mark.parametrize("failure", [False, True])
def test_restores_bindings_and_existing_profile_on_success_or_exception(failure):
    previous = sys.getprofile()
    calls = []

    def previous_profiler(frame, event, arg):
        if event == "return" and frame.f_code.co_name == "_sample_constructive_ester_offspring":
            calls.append(event)

    rng, multinomial = np.random.default_rng, torch.multinomial
    sys.setprofile(previous_profiler)
    try:
        try:
            with SamplingTrace() as trace:
                _topology_draw(torch.Generator().manual_seed(1))
                if failure:
                    raise ValueError("original failure")
        except ValueError as error:
            assert str(error) == "original failure"
        assert sys.getprofile() is previous_profiler
        assert np.random.default_rng is rng
        assert torch.multinomial is multinomial
        assert calls == ["return"]
        assert trace.summary["status"] == ("failed" if failure else "complete")
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("bound", [{"max_candidates": 1}, {"max_events": 1}])
def test_overflow_fails_and_restores_global_state(bound):
    rng, multinomial, profile = np.random.default_rng, torch.multinomial, sys.getprofile()
    with pytest.raises(SamplingTraceError, match="max_"):
        with SamplingTrace(**bound) as trace:
            _topology_draw(torch.Generator().manual_seed(1))
    assert trace.summary["status"] == "failed"
    assert (np.random.default_rng, torch.multinomial, sys.getprofile()) == (
        rng,
        multinomial,
        profile,
    )


def test_nested_context_is_rejected_without_disabling_outer_observer():
    with SamplingTrace() as trace:
        with pytest.raises(SamplingTraceError, match="nested"):
            with SamplingTrace():
                pass
        _topology_draw(torch.Generator().manual_seed(1))
    assert trace.summary["status"] == "complete"
    assert trace.events


def test_unsupported_function_schema_fails_before_installing_hooks(monkeypatch):
    monkeypatch.setattr(sampling, "_select_ugi_amine_atom_states", lambda: None)
    rng, multinomial, profile = np.random.default_rng, torch.multinomial, sys.getprofile()
    with pytest.raises(SamplingTraceError, match="unsupported sampling implementation schema"):
        with SamplingTrace():
            pass
    assert (np.random.default_rng, torch.multinomial, sys.getprofile()) == (
        rng,
        multinomial,
        profile,
    )
