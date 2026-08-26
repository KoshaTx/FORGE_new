from __future__ import annotations

import pytest

from forge.assembly import ReactionProgramSpec
from forge.model.reaction_program_conditioning import (
    ReactionProgramConditioning,
    ReactionProgramVocabulary,
)

torch = pytest.importorskip("torch")


def _vocabulary() -> ReactionProgramVocabulary:
    return ReactionProgramVocabulary.from_specs(
        (
            ReactionProgramSpec("aza", "aza", "amine", "acrylate", 1, 4),
            ReactionProgramSpec("reductive", "reductive", "amine", "aldehyde", 1, 6),
        )
    )


def test_program_vocabulary_is_stable_and_contains_no_component_identity() -> None:
    vocabulary = _vocabulary()
    assert vocabulary.program_states == ("unconditioned", "aza", "reductive")
    assert vocabulary.role_states == ("unassigned", "acrylate", "aldehyde", "amine")
    assert vocabulary.core_position_states == ("unconditioned", "exterior")
    assert vocabulary.maximum_steps == 6


def test_program_vocabulary_namespaces_registry_core_positions() -> None:
    specs = (ReactionProgramSpec("aza", "aza", "amine", "acrylate", 1, 4),)
    vocabulary = ReactionProgramVocabulary.from_specs(
        specs,
        core_positions=("aza:map_2", "aza:map_1"),
    )
    assert vocabulary.core_position_states == (
        "unconditioned",
        "exterior",
        "aza:map_1",
        "aza:map_2",
    )


def test_program_conditioner_masks_non_adapter_nodes_exactly() -> None:
    vocabulary = _vocabulary()
    module = ReactionProgramConditioning(vocabulary=vocabulary, hidden_dim=8)
    mask = torch.tensor([[True, False, True], [False, False, True]])
    output = module(
        program_states=torch.tensor([1, 2]),
        role_states=torch.tensor([[3, 0, 1], [0, 0, 2]]),
        core_position_states=torch.tensor([[1, 0, 1], [0, 0, 1]]),
        program_depths=torch.tensor([1, 2]),
        adapter_mask=mask,
    )
    assert output.shape == (2, 3, 8)
    assert torch.equal(output[~mask], torch.zeros_like(output[~mask]))
