"""Output-exactness regression tests for the optimized Ugi sampling path.

These tests exist because the sampling implementation was optimized for throughput under a
byte-identical-output contract.  Each one pins a behaviour that an execution-only change must not
disturb: the exact attached-tree dynamic program, the strict terminal decoder's admissible-parent
set, and the step-invariant program memory reused across denoising steps.

The reference implementations below are transcriptions of the pre-optimization code.  They are
deliberately naive and are the specification the fast paths must reproduce exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_transformer import ReactionProgramGraphTransformer
from forge.model.ugi_morphology_program import (
    UgiMorphologyProgramError,
    _attached_choice_valid,
    _sample_attached_offspring_with_exact_budget,
    attached_program_feasible,
)

torch = pytest.importorskip("torch")


def _reference_attached_sample(
    logits,
    *,
    junction_budget: int,
    attachment_count: int,
    generator,
    maximum_adjacent_branch_run: int | None = None,
) -> np.ndarray:
    """Pre-optimization scalar dynamic program, kept verbatim as the specification."""

    node_count, child_classes = logits.shape
    maximum_children = child_classes - 1
    log_probabilities = logits.to(torch.float64).log_softmax(dim=-1)
    branch_limit = (
        node_count if maximum_adjacent_branch_run is None else maximum_adjacent_branch_run
    )
    suffix: list[dict[tuple[int, int, int], object]] = [dict() for _ in range(node_count + 1)]
    for incoming_run in range(branch_limit + 1):
        suffix[node_count][(0, 0, incoming_run)] = logits.new_tensor(0.0, dtype=torch.float64)
    for position in range(node_count - 1, -1, -1):
        positions_including_current = node_count - position
        for pending in range(1, positions_including_current + 1):
            for budget in range(junction_budget + 1):
                for incoming_run in range(branch_limit + 1):
                    terms = []
                    for children in range(maximum_children + 1):
                        state = _attached_choice_valid(
                            position=position,
                            pending=pending,
                            remaining_budget=budget,
                            children=children,
                            node_count=node_count,
                        )
                        next_run = incoming_run + 1 if children >= 2 else 0
                        suffix_state = None if state is None else (*state, next_run)
                        if (
                            next_run <= branch_limit
                            and suffix_state is not None
                            and suffix_state in suffix[position + 1]
                        ):
                            terms.append(
                                log_probabilities[position, children]
                                + suffix[position + 1][suffix_state]
                            )
                    if terms:
                        suffix[position][(pending, budget, incoming_run)] = torch.logsumexp(
                            torch.stack(terms), dim=0
                        )
    if (attachment_count, junction_budget, 0) not in suffix[0]:
        raise UgiMorphologyProgramError(
            "exact attached-tree decoder found no branch-run-compatible completion"
        )
    output = np.zeros(node_count, dtype=np.int64)
    pending = attachment_count
    remaining_budget = junction_budget
    current_branch_run = 0
    for position in range(node_count):
        choices: list[int] = []
        states: list[tuple[int, int, int]] = []
        weights = []
        for children in range(maximum_children + 1):
            state = _attached_choice_valid(
                position=position,
                pending=pending,
                remaining_budget=remaining_budget,
                children=children,
                node_count=node_count,
            )
            next_run = current_branch_run + 1 if children >= 2 else 0
            suffix_state = None if state is None else (*state, next_run)
            if (
                next_run > branch_limit
                or suffix_state is None
                or suffix_state not in suffix[position + 1]
            ):
                continue
            choices.append(children)
            states.append(suffix_state)
            weights.append(
                log_probabilities[position, children] + suffix[position + 1][suffix_state]
            )
        selected = int(
            torch.multinomial(torch.stack(weights).softmax(dim=0), 1, generator=generator)
        )
        output[position] = choices[selected]
        pending, remaining_budget, current_branch_run = states[selected]
    return output


_SHAPES = [
    (6, 1, 1, 3),
    (8, 2, 1, 3),
    (9, 2, 2, 4),
    (11, 3, 1, 4),
    (12, 0, 1, 3),
    (14, 2, 3, 4),
]


@pytest.mark.parametrize(("node_count", "junction_budget", "attachment_count", "classes"), _SHAPES)
def test_attached_tree_sampler_matches_the_scalar_reference_exactly(
    node_count: int, junction_budget: int, attachment_count: int, classes: int
) -> None:
    """The vectorized dynamic program must draw the identical tree from the identical stream."""

    if not attached_program_feasible(node_count, junction_budget, classes - 1, attachment_count):
        pytest.skip("declared program shape is infeasible by construction")
    for trial in range(6):
        logits = torch.randn(
            (node_count, classes),
            generator=torch.Generator().manual_seed(1000 + trial),
            dtype=torch.float32,
        )
        observed = _sample_attached_offspring_with_exact_budget(
            logits,
            junction_budget=junction_budget,
            attachment_count=attachment_count,
            generator=torch.Generator().manual_seed(7),
        )
        expected = _reference_attached_sample(
            logits,
            junction_budget=junction_budget,
            attachment_count=attachment_count,
            generator=torch.Generator().manual_seed(7),
        )
        assert np.array_equal(observed, expected)


@pytest.mark.parametrize("branch_run", [0, 1, 2, 3])
def test_attached_tree_sampler_matches_the_reference_under_a_binding_branch_limit(
    branch_run: int,
) -> None:
    """A branch-run budget below the tree size is not inert and must keep its own state."""

    logits = torch.randn((10, 4), generator=torch.Generator().manual_seed(21))

    def draw(sampler):
        try:
            return sampler(
                logits,
                junction_budget=2,
                attachment_count=1,
                generator=torch.Generator().manual_seed(4),
                maximum_adjacent_branch_run=branch_run,
            )
        except Exception as error:  # noqa: BLE001 - the failure mode is part of the contract
            return type(error).__name__

    observed = draw(_sample_attached_offspring_with_exact_budget)
    expected = draw(_reference_attached_sample)
    if isinstance(expected, str):
        assert observed == expected
    else:
        assert np.array_equal(observed, expected)


def test_attached_tree_sampler_consumes_the_same_generator_stream() -> None:
    """A different draw order would be a different sampler even at the same distribution."""

    logits = torch.randn((10, 4), generator=torch.Generator().manual_seed(3))
    fast = torch.Generator().manual_seed(11)
    slow = torch.Generator().manual_seed(11)
    for _ in range(4):
        observed = _sample_attached_offspring_with_exact_budget(
            logits, junction_budget=2, attachment_count=1, generator=fast
        )
        expected = _reference_attached_sample(
            logits, junction_budget=2, attachment_count=1, generator=slow
        )
        assert np.array_equal(observed, expected)
    assert torch.equal(fast.get_state(), slow.get_state())


def _transformer(**overrides):
    vocabulary = ReactionProgramVocabulary(
        program_states=("unconditioned", "ugi_3cr_agile"),
        role_states=("unassigned", "amine_head", "aldehyde_tail"),
        core_position_states=("unconditioned", "exterior", "core_a", "core_b"),
        maximum_steps=3,
    )
    torch.manual_seed(0)
    model = ReactionProgramGraphTransformer(
        vocabulary=vocabulary,
        node_classes=5,
        hidden_dim=32,
        layers=2,
        heads=4,
        expert_count=3,
        adapter_dim=8,
        maximum_closures=2,
        maximum_heavy_atoms=12,
        dropout=0.0,
        bond_classes=4,
        repeat_group_conditioning=True,
        role_morphology_conditioning=True,
        **overrides,
    )
    model.eval()
    return model


def _transformer_batch(nodes: int = 7, batch: int = 3):
    generator = torch.Generator().manual_seed(5)
    node_mask = torch.ones((batch, nodes), dtype=torch.bool)
    child_mask = node_mask.clone()
    child_mask[:, 0] = False
    parents = torch.zeros((batch, nodes), dtype=torch.long)
    for child in range(1, nodes):
        parents[:, child] = torch.randint(0, child, (batch,), generator=generator)
    return {
        "nodes": torch.randint(0, 5, (batch, nodes), generator=generator),
        "parents": parents,
        "parent_bonds": torch.randint(0, 4, (batch, nodes), generator=generator),
        "closure_left": torch.randint(0, nodes, (batch, 2), generator=generator),
        "closure_right": torch.randint(0, nodes, (batch, 2), generator=generator),
        "closure_bonds": torch.randint(0, 4, (batch, 2), generator=generator),
        "node_mask": node_mask,
        "child_mask": child_mask,
        "closure_mask": torch.ones((batch, 2), dtype=torch.bool),
        "program_states": torch.ones(batch, dtype=torch.long),
        "role_states": torch.randint(1, 3, (batch, nodes), generator=generator),
        "core_position_states": torch.randint(1, 4, (batch, nodes), generator=generator),
        "program_depths": torch.ones(batch, dtype=torch.long),
        "adapter_mask": node_mask.clone(),
        "repeat_group_states": torch.zeros((batch, nodes), dtype=torch.long),
        "component_position_states": torch.zeros((batch, nodes), dtype=torch.long),
        "component_instance_states": torch.ones((batch, nodes), dtype=torch.long),
        "role_morphology_states": torch.ones((batch, nodes, 4), dtype=torch.long),
    }


def test_reused_program_memory_reproduces_every_output_tensor_exactly() -> None:
    """Encoding the program once per batch must be reuse, not an approximation."""

    model = _transformer()
    batch = _transformer_batch()
    conditioning = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        }
    }
    memory_arguments = {
        key: batch[key]
        for key in (
            "program_states",
            "role_states",
            "core_position_states",
            "program_depths",
            "adapter_mask",
            "repeat_group_states",
            "component_position_states",
            "role_morphology_states",
        )
    }
    with torch.inference_mode():
        memory = model.prepare_program_memory(**memory_arguments)
        for step in range(4):
            time = torch.full((batch["nodes"].shape[0],), step / 4.0)
            state = {
                key: batch[key]
                for key in (
                    "nodes",
                    "parents",
                    "parent_bonds",
                    "closure_left",
                    "closure_right",
                    "closure_bonds",
                )
            }
            fresh = model(**state, t=time, **conditioning)
            reused = model(**state, t=time, **conditioning, program_memory=memory)
            assert set(fresh) == set(reused)
            for key in sorted(fresh):
                assert torch.equal(fresh[key], reused[key]), key


def test_program_memory_from_a_different_batch_is_refused() -> None:
    """A stale memory must fail closed rather than silently condition the wrong program."""

    model = _transformer()
    batch = _transformer_batch()
    other = _transformer_batch()
    memory_arguments = {
        key: other[key]
        for key in (
            "program_states",
            "role_states",
            "core_position_states",
            "program_depths",
            "adapter_mask",
            "repeat_group_states",
            "component_position_states",
            "role_morphology_states",
        )
    }
    conditioning = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        }
    }
    state = {
        key: batch[key]
        for key in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        )
    }
    with torch.inference_mode():
        memory = model.prepare_program_memory(**memory_arguments)
        with pytest.raises(Exception, match="different conditioning tensors"):
            model(
                **state,
                t=torch.zeros(batch["nodes"].shape[0]),
                **conditioning,
                program_memory=memory,
            )
