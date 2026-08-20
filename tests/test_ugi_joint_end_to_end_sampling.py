from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forge.design.flow.ugi_adapter_features import ORIGIN_TO_INDEX
from forge.design.corpus.ugi_chemistry_corpus import load_expanded_ugi_chemistry_corpus
from forge.design.corpus.ugi_held_component_gate import load_ugi_reaction_contract
from forge.design.sampling.ugi_joint_end_to_end_sampling import (
    UgiJointEndToEndSamplingError,
    _annotate_l1_terminal_admission,
    _complete_reference_comparison,
    _load_matched_programs,
    _slice_matched_programs,
    complete_ugi_joint_terminals,
    flow_endpoint_chemistry_sample,
    sample_ugi_joint_end_to_end,
)
from forge.design.flow.ugi_joint_sparse_flow import UgiJointSparseTerminal
from forge.design.flow.ugi_morphology_program import UgiMorphologyProgram

torch = pytest.importorskip("torch")


def test_flow_endpoint_mapping_preserves_joint_channels_without_redrawing() -> None:
    roles = (
        ORIGIN_TO_INDEX["amine_head"],
        ORIGIN_TO_INDEX["oxoester_aldehyde_body_tail"],
        ORIGIN_TO_INDEX["isocyanide_tail"],
    )
    condition = SimpleNamespace(
        origin_states=np.asarray((roles[0], *roles), dtype=np.int64),
        fixed_atom_mask=np.asarray((True, False, False, False)),
        fixed_atom_states=np.asarray((7, -1, -1, -1), dtype=np.int64),
        fixed_parent_bond_states=np.asarray((-1, -1, -1, -1), dtype=np.int64),
        closure_count=1,
    )
    program = UgiMorphologyProgram(
        node_counts=(1, 1, 1),
        junction_budgets=(0, 0, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )
    terminal = UgiJointSparseTerminal(
        program=program,
        offspring=(np.zeros(1, dtype=np.int64),) * 3,
        atom_logits=np.zeros((3, 4), dtype=np.float32),
        parent_bond_logits=np.zeros((3, 4), dtype=np.float32),
        decoration_anchor_logits=np.zeros((3, 4), dtype=np.float32),
        decoration_atom_logits=np.zeros((3, 4), dtype=np.float32),
        decoration_bond_logits=np.zeros((3, 4), dtype=np.float32),
        hidden=np.zeros((3, 4), dtype=np.float32),
        flow_endpoint_atom_states=np.asarray((1, 2, 3), dtype=np.int64),
        flow_endpoint_parent_bond_states=np.asarray((0, 1, 2), dtype=np.int64),
        flow_endpoint_decoration_anchors=np.asarray((1, 3, 0), dtype=np.int64),
        flow_endpoint_decoration_atom_states=np.asarray((2, 1, 0), dtype=np.int64),
        flow_endpoint_decoration_bond_states=np.asarray((1, 0, 0), dtype=np.int64),
    )

    sample = flow_endpoint_chemistry_sample(
        condition,
        terminal,
        np.asarray((2,), dtype=np.int64),
    )

    assert sample.atom_states.tolist() == [7, 1, 2, 3]
    assert sample.parent_bond_states.tolist() == [-1, 0, 1, 2]
    assert sample.closure_bond_states.tolist() == [2]
    assert sample.decoration_anchors.tolist() == [2, 4, 0]
    assert sample.decoration_atom_states.tolist() == [2, 1, 0]
    assert sample.decoration_bond_states.tolist() == [1, 0, 0]


def test_matched_program_loader_preserves_evaluation_strata(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "product_id": "probe-1",
                        "source_stratum": "expanded_exact_forward_enumeration",
                        "branch_class": "both_tail_origins_branched",
                        "component_novelty_class": "bounded_structural_expansion_component",
                        "program": {
                            "node_counts": [4, 8, 9],
                            "junction_budgets": [1, 2, 1],
                            "cycle_ranks": [0, 0, 1],
                            "attachment_counts": [1, 1, 1],
                        },
                    }
                ]
            }
        )
    )

    programs, metadata = _load_matched_programs(path)

    assert programs[0].node_counts == (4, 8, 9)
    assert metadata == (
        {
            "product_id": "probe-1",
            "source_stratum": "expanded_exact_forward_enumeration",
            "branch_class": "both_tail_origins_branched",
            "component_novelty_class": "bounded_structural_expansion_component",
        },
    )


def test_matched_program_slicing_uses_a_declared_contiguous_block() -> None:
    programs = tuple(
        UgiMorphologyProgram(
            node_counts=(index + 1, 2, 3),
            junction_budgets=(0, 0, 0),
            cycle_ranks=(0, 0, 0),
            attachment_counts=(1, 1, 1),
        )
        for index in range(5)
    )
    metadata = tuple({"index": index} for index in range(5))

    selected, selected_metadata = _slice_matched_programs(programs, metadata, offset=1, limit=3)

    assert [program.node_counts[0] for program in selected] == [2, 3, 4]
    assert selected_metadata == ({"index": 1}, {"index": 2}, {"index": 3})


@pytest.mark.parametrize(("offset", "limit"), [(-1, 1), (0, 0)])
def test_matched_program_slicing_rejects_invalid_bounds(offset: int, limit: int) -> None:
    with pytest.raises(UgiJointEndToEndSamplingError):
        _slice_matched_programs((), (), offset=offset, limit=limit)


def test_deferred_reference_comparison_is_explicit_and_complete() -> None:
    result = {
        "status": "sampling_complete_reference_pending",
        "reference_comparison": None,
        "reference_comparison_status": "pending",
    }

    observed = _complete_reference_comparison(
        result,
        molecules=[],
        reference_corpus=None,
        mode="deferred",
    )

    assert observed["status"] == "complete_sampling_reference_deferred"
    assert observed["reference_comparison"] is None
    assert observed["reference_comparison_status"] == "deferred"


def test_reference_comparison_mode_rejects_silent_fallback() -> None:
    with pytest.raises(UgiJointEndToEndSamplingError, match="reference comparison mode"):
        _complete_reference_comparison(
            {"status": "sampling_complete_reference_pending"},
            molecules=[],
            reference_corpus=None,
            mode="skip",
        )


def test_full_reference_comparison_requires_the_frozen_corpus() -> None:
    with pytest.raises(UgiJointEndToEndSamplingError, match="frozen reference corpus"):
        _complete_reference_comparison(
            {"status": "sampling_complete_reference_pending"},
            molecules=[],
            reference_corpus=None,
            mode="full",
        )


def test_exact_l1_terminal_validity_requires_a_qualified_registry(tmp_path: Path) -> None:
    with pytest.raises(
        UgiJointEndToEndSamplingError,
        match="qualified reaction registry",
    ):
        sample_ugi_joint_end_to_end(
            Path(__file__).resolve().parents[1],
            tmp_path / "output",
            joint_checkpoint_path=tmp_path / "missing-joint.pt",
            closure_checkpoint_path=tmp_path / "missing-closure.pt",
            matched_staged_result_path=tmp_path / "missing-programs.json",
            sample_steps=8,
            batch_size=16,
            seed=1,
            overwrite=False,
            evaluate_exact_l1_terminal_admission=True,
        )


def test_l1_terminal_failure_remains_in_raw_validity_denominator() -> None:
    reaction = load_ugi_reaction_contract(
        Path(__file__).parents[1] / "data/vendor/qualified_reactions_v1.json"
    )
    row = {
        "valid": True,
        "failure_type": None,
        "smiles": "C=NC(CCCCCCCCCCCCCCCCCCCCCC)C(=O)NCCCCCCCCCCCCCC",
        "component_reconstruction_valid": True,
        "component_smiles_by_role": {
            "amine_head": "C=N",
            "oxoester_aldehyde_body_tail": "CCCCCCCCCCCCCCCCCCCCCCC=O",
            "isocyanide_tail": "[C-]#[N+]CCCCCCCCCCCCCC",
        },
    }

    _annotate_l1_terminal_admission(row, reaction)

    assert row["valid"] is True
    assert row["failure_type"] is None
    assert row["raw_molecule_valid"] is True
    assert row["terminal_valid"] is False
    assert row["terminal_failure_type"] == "L1ForwardConsistencyFailure"
    assert row["l1_forward_verification"]["exact_product_reconstructed"] is False


def test_terminal_completion_is_restartable_at_the_closure_rng_boundary() -> None:
    repo = Path(__file__).resolve().parents[1]
    corpus = load_expanded_ugi_chemistry_corpus(
        repo / "results/phase1/ugi_expanded_chemistry_exemplars/assignments.csv.gz",
        repo / "results/phase1/ugi_expanded_chemistry_exemplars/semantic_products.csv.gz",
        repo / "results/phase1/ugi_expanded_chemistry_exemplars/semantic_atoms.csv.gz",
        repo / "results/phase1/product_v3_atom_vocabulary.json",
    )
    program = UgiMorphologyProgram(
        node_counts=(1, 1, 1),
        junction_budgets=(0, 0, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )
    atom_classes = len(corpus.atom_vocabulary)
    terminal = UgiJointSparseTerminal(
        program=program,
        offspring=(np.zeros(1, dtype=np.int64),) * 3,
        atom_logits=np.zeros((3, atom_classes), dtype=np.float32),
        parent_bond_logits=np.zeros((3, 4), dtype=np.float32),
        decoration_anchor_logits=np.zeros((1, 4), dtype=np.float32),
        decoration_atom_logits=np.zeros((1, atom_classes), dtype=np.float32),
        decoration_bond_logits=np.zeros((1, 4), dtype=np.float32),
        hidden=np.zeros((3, 4), dtype=np.float32),
    )
    initial_state = torch.Generator().manual_seed(17).get_state()

    completion = complete_ugi_joint_terminals(
        SimpleNamespace(maximum_decorations=1),
        None,
        (terminal,),
        corpus,
        program_metadata=({"source": "restartable-test"},),
        closure_generator_state=initial_state,
        allowed_ring_sizes=(5, 6),
        maximum_heavy_degree=4,
    )

    assert len(completion.rows) == 1
    assert completion.rows[0]["source"] == "restartable-test"
    assert "valid" in completion.rows[0]
    assert torch.equal(completion.closure_generator_state, initial_state)


@pytest.mark.parametrize(
    "decoder_mode",
    [
        "stochastic",
        "bond_stochastic",
        "atom_bond_stochastic",
        "decoration_bond_stochastic",
    ],
)
def test_stochastic_terminal_completion_requires_explicit_rng_state(
    decoder_mode: str,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    corpus = load_expanded_ugi_chemistry_corpus(
        repo / "results/phase1/ugi_expanded_chemistry_exemplars/assignments.csv.gz",
        repo / "results/phase1/ugi_expanded_chemistry_exemplars/semantic_products.csv.gz",
        repo / "results/phase1/ugi_expanded_chemistry_exemplars/semantic_atoms.csv.gz",
        repo / "results/phase1/product_v3_atom_vocabulary.json",
    )
    program = UgiMorphologyProgram(
        node_counts=(1, 1, 1),
        junction_budgets=(0, 0, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )
    atom_classes = len(corpus.atom_vocabulary)
    terminal = UgiJointSparseTerminal(
        program=program,
        offspring=(np.zeros(1, dtype=np.int64),) * 3,
        atom_logits=np.zeros((3, atom_classes), dtype=np.float32),
        parent_bond_logits=np.zeros((3, 4), dtype=np.float32),
        decoration_anchor_logits=np.zeros((1, 4), dtype=np.float32),
        decoration_atom_logits=np.zeros((1, atom_classes), dtype=np.float32),
        decoration_bond_logits=np.zeros((1, 4), dtype=np.float32),
        hidden=np.zeros((3, 4), dtype=np.float32),
    )

    with pytest.raises(
        UgiJointEndToEndSamplingError,
        match="explicit generator state",
    ):
        complete_ugi_joint_terminals(
            SimpleNamespace(maximum_decorations=1),
            None,
            (terminal,),
            corpus,
            program_metadata=({},),
            closure_generator_state=torch.Generator().manual_seed(17).get_state(),
            allowed_ring_sizes=(5, 6),
            maximum_heavy_degree=4,
            terminal_decoder_mode=decoder_mode,
        )
