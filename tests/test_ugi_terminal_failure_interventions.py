from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.phase1.product_l1.training.ugi_training_cache import load_ugi_training_cache
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.ugi_chemistry_flow import (
    TERMINAL_DECODE_FAILURE_SCHEMA,
    UgiTerminalDecodeError,
    valence_constrained_terminal_sample,
)

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
POLICY = REPO / "results/phase1/local_morphology_support_v2/policy.json"
PROGRAM_ID = "ugi_3cr_agile"


@pytest.fixture(scope="module")
def terminal_fixture():
    if not CACHE.is_file() or not POLICY.is_file():
        pytest.skip("frozen Ugi training cache or local-morphology policy is absent")
    corpus, _ = load_ugi_training_cache(CACHE)
    condition = corpus.records_by_fold["train"][0].condition
    support = LocalChemistrySupport.from_mapping(json.loads(POLICY.read_text()))
    vocabulary = corpus.atom_vocabulary
    atom_logits = torch.zeros((1, condition.node_count, len(vocabulary)))
    carbon = next(
        index
        for index, state in enumerate(vocabulary)
        if state.symbol == "C" and not state.aromatic and state.formal_charge == 0
    )
    nitrogen = next(
        index
        for index, state in enumerate(vocabulary)
        if state.symbol == "N" and not state.aromatic and state.formal_charge == 0
    )
    oxygen = next(
        index
        for index, state in enumerate(vocabulary)
        if state.symbol == "O" and not state.aromatic and state.formal_charge == 0
    )
    atom_logits[:, :, carbon] = 10.0
    # This is a variable amine-head parent/child pair in the frozen record.
    left, right = 43, 44
    atom_logits[0, left, nitrogen] = 30.0
    atom_logits[0, right, oxygen] = 30.0
    parent_logits = torch.zeros((1, condition.node_count, 4))
    parent_logits[:, :, 0] = 10.0
    closure_logits = torch.zeros((1, condition.closure_count, 4))
    if condition.closure_count:
        closure_logits[:, :, 0] = 10.0
    maximum_decorations = 2
    terminal = {
        "nodes": atom_logits,
        "parent_bonds": parent_logits,
        "closure_bonds": closure_logits,
        "decoration_anchors": torch.full(
            (1, maximum_decorations, condition.node_count + 1), -20.0
        ),
        "decoration_atoms": torch.zeros((1, maximum_decorations, len(vocabulary))),
        "decoration_bonds": torch.zeros((1, maximum_decorations, 4)),
    }
    terminal["decoration_anchors"][:, :, 0] = 20.0
    return condition, vocabulary, support, terminal, left, right, nitrogen, oxygen


def test_role_local_decoder_masks_unsupported_nitrogen_oxygen_edge(
    terminal_fixture,
) -> None:
    condition, vocabulary, support, terminal, left, right, nitrogen, oxygen = terminal_fixture

    unconstrained = valence_constrained_terminal_sample(
        condition,
        terminal,
        0,
        vocabulary,
        2,
    )
    constrained = valence_constrained_terminal_sample(
        condition,
        terminal,
        0,
        vocabulary,
        2,
        local_chemistry_support=support,
        program_id=PROGRAM_ID,
        local_chemistry_constraint_scope="role_edges_only",
    )

    assert int(unconstrained.atom_states[left]) == nitrogen
    assert int(unconstrained.atom_states[right]) == oxygen
    assert not (
        vocabulary[int(constrained.atom_states[left])].symbol == "N"
        and vocabulary[int(constrained.atom_states[right])].symbol == "O"
    )


def test_role_local_terminal_failure_is_typed_and_serializable(terminal_fixture) -> None:
    condition, vocabulary, support, terminal, *_ = terminal_fixture
    empty_edges = {**support.role_edges, PROGRAM_ID: frozenset()}
    impossible = replace(support, role_edges=empty_edges)

    with pytest.raises(UgiTerminalDecodeError) as captured:
        valence_constrained_terminal_sample(
            condition,
            terminal,
            0,
            vocabulary,
            2,
            local_chemistry_support=impossible,
            program_id=PROGRAM_ID,
        )

    detail = captured.value.to_mapping()
    assert detail["schema_version"] == TERMINAL_DECODE_FAILURE_SCHEMA
    assert detail["stage"] == "atom_state"
    assert detail["code"] == "role_local_atom_state_unavailable"
    assert isinstance(detail["context"]["node"], int)
