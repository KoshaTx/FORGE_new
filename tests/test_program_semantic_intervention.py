from __future__ import annotations

import pytest
import torch

from experiments.phase1.multireaction.program_semantic_intervention import (
    ProgramSemanticInterventionError,
    _condition_contracts,
    _validate_training_input,
    paired_intervention_summary,
)
from forge.core.hashing import sha256_file
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.synthesis_program_sampling import _terminal_smiles


def _row(
    index: int,
    *,
    smiles: str | None,
    valid: bool,
    exact_l1: bool,
) -> dict[str, object]:
    return {
        "sample_index": index,
        "layout_record_id": f"layout-{index}",
        "program_id": "ugi",
        "canonical_smiles": smiles,
        "valid": valid,
        "exact_l1_program": exact_l1,
    }


def test_paired_intervention_summary_preserves_attempt_pairing() -> None:
    factual = [
        _row(0, smiles="CC", valid=True, exact_l1=True),
        _row(1, smiles="CN", valid=True, exact_l1=True),
        _row(2, smiles=None, valid=False, exact_l1=False),
        _row(3, smiles="CO", valid=True, exact_l1=False),
    ]
    control = [
        _row(0, smiles="CC", valid=True, exact_l1=True),
        _row(1, smiles="NN", valid=True, exact_l1=False),
        _row(2, smiles="CN", valid=True, exact_l1=True),
        _row(3, smiles=None, valid=False, exact_l1=False),
    ]

    summary = paired_intervention_summary(factual, control)

    assert summary["both_exact_l1"] == 1
    assert summary["factual_only_exact_l1"] == 1
    assert summary["control_only_exact_l1"] == 1
    assert summary["neither_exact_l1"] == 1
    assert summary["factual_minus_control_exact_l1_yield"] == 0.0
    assert summary["canonical_output_change_fraction"] == 0.75
    assert summary["validity_change_fraction"] == 0.5
    assert summary["attempts_are_not_independent_training_replicates"] is True


def test_paired_intervention_summary_rejects_unpaired_rows() -> None:
    factual = [_row(0, smiles="CC", valid=True, exact_l1=True)]
    control = [_row(1, smiles="CC", valid=True, exact_l1=True)]

    with pytest.raises(ProgramSemanticInterventionError, match="identity changed"):
        paired_intervention_summary(factual, control)


def test_condition_contracts_are_inference_mismatches_not_label_renaming() -> None:
    vocabulary = ReactionProgramVocabulary.from_semantics(
        program_ids=("ugi", "bl", "lx"),
        roles=("amine", "aldehyde", "isocyanide", "head", "tail"),
        maximum_steps=2,
    )

    conditions = _condition_contracts(
        target_program="ugi",
        mismatch_programs=("bl", "lx"),
        ugi_roles=("amine", "aldehyde", "isocyanide"),
        vocabulary=vocabulary,
    )

    assert set(conditions) == {
        "factual",
        "mismatched_program__bl",
        "mismatched_program__lx",
        "mismatched_roles__forward_cycle",
        "mismatched_roles__reverse_cycle",
        "null_all_program_coordinates",
    }
    ugi_state = vocabulary.program_to_index["ugi"]
    assert conditions["mismatched_program__bl"]["program_state_mapping"] == {
        ugi_state: vocabulary.program_to_index["bl"]
    }
    forward = conditions["mismatched_roles__forward_cycle"]["role_state_mapping"]
    assert set(forward) == {
        vocabulary.role_to_index["amine"],
        vocabulary.role_to_index["aldehyde"],
        vocabulary.role_to_index["isocyanide"],
    }
    assert set(forward.values()) == set(forward)
    assert all(source != target for source, target in forward.items())


def test_terminal_smiles_does_not_publish_a_string_that_rdkit_cannot_reparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import forge.model.synthesis_program_sampling as sampling

    monkeypatch.setattr(sampling, "graph_to_molecule", lambda *_: object())
    monkeypatch.setattr(sampling.Chem, "MolToSmiles", lambda *_args, **_kwargs: "invalid")
    monkeypatch.setattr(sampling.Chem, "MolFromSmiles", lambda _value: None)
    state = {
        "nodes": torch.tensor([[0]]),
        "parents": torch.tensor([[0]]),
        "parent_bonds": torch.tensor([[0]]),
        "closure_left": torch.empty((1, 0), dtype=torch.long),
        "closure_right": torch.empty((1, 0), dtype=torch.long),
        "closure_bonds": torch.empty((1, 0), dtype=torch.long),
    }

    assert _terminal_smiles(state, 0, 1, 0, ()) is None


def test_training_validation_uses_the_configured_final_arm(tmp_path) -> None:
    design_path = tmp_path / "study_design.json"
    cache_path = tmp_path / "cache.npz"
    archive_path = tmp_path / "checkpoints.tar"
    design_path.write_bytes(b"design")
    cache_path.write_bytes(b"cache")
    archive_path.write_bytes(b"archive")
    snapshot = {"step": 1700, "filename": "step_00001700.pt", "sha256": "f" * 64}
    training = {
        "schema_version": "forge.synthesis_program_production_training_result.v1",
        "status": "pass",
        "profile": "full",
        "replicate": 0,
        "seed": 20260825,
        "design": {"sha256": str(sha256_file(design_path))},
        "cache": {"sha256": str(sha256_file(cache_path))},
        "checkpoint_archive": {"sha256": str(sha256_file(archive_path))},
        "arms": {
            "shared_three_program_conditioned": {"checkpoints": []},
            "bl_core_constrained_repeat_aware": {"checkpoints": [snapshot]},
        },
    }

    selected = _validate_training_input(
        training,
        replicate=0,
        seed=20260825,
        design_path=design_path,
        cache_path=cache_path,
        archive_path=archive_path,
        final_step=1700,
        target_arm="bl_core_constrained_repeat_aware",
    )

    assert selected == snapshot
