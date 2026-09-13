"""Behavioral checks for measured-only, fixed-state checkpoint attribution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from forge.core.hashing import PinError
from forge.model.ugi_realism_model_attribution import (
    UgiRealismModelAttributionError,
    classification_summary,
    coordinate_summaries,
    masked_probabilities,
    rank_law_probabilities,
    select_measured_train_rows,
)


def test_role_denominators_exclude_fixed_and_split_actual_corruption() -> None:
    logits = np.array([[[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0], [0.0, 20.0]]])
    target = np.array([[0, 1, 1, 1, 0]])
    noisy = np.array([[1, 1, 0, 1, 1]])
    result = coordinate_summaries(
        logits=logits,
        targets=target,
        noisy=noisy,
        active_mask=np.array([[True, True, True, True, False]]),
        role_states=np.array([[1, 1, 2, 2, 99]]),
        roles={"head": 1, "tail": 2},
        allowed=np.ones_like(logits, dtype=bool),
    )
    for role in ("head", "tail"):
        assert result[role]["all_variable"]["softmax"]["total"] == 2
        assert result[role]["actually_corrupted"]["softmax"]["total"] == 1
        assert result[role]["unchanged"]["softmax"]["total"] == 1
    assert result["head"]["actually_corrupted"]["softmax"]["accuracy"] == 1
    assert result["tail"]["actually_corrupted"]["softmax"]["accuracy"] == 0


def test_rank_law_keeps_mask_and_order_but_discards_logit_gap() -> None:
    logits = np.array([[12.0, 0.0, 99.0], [0.01, 0.0, 99.0]])
    allowed = np.array([[True, True, False], [True, True, False]])
    softmax = masked_probabilities(logits, allowed)
    ranked = rank_law_probabilities(
        logits,
        allowed,
        model_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.35,
    )
    assert np.all(ranked[:, 2] == 0)
    assert np.allclose(ranked[0], ranked[1])
    assert np.array_equal(softmax.argmax(axis=1), ranked.argmax(axis=1))
    assert ranked[0, 0] < softmax[0, 0]
    assert ranked[1, 0] > softmax[1, 0]
    # Rank normalization can increase confidence when the neural gap is small; it is not
    # universally a flattening transform.


def test_invalid_mask_and_excluded_target_fail() -> None:
    with pytest.raises(UgiRealismModelAttributionError, match="candidate masks"):
        masked_probabilities(np.array([[1.0, 0.0]]), np.array([[False, False]]))
    with pytest.raises(UgiRealismModelAttributionError, match="target excluded"):
        coordinate_summaries(
            logits=np.array([[[10.0, 0.0]]]),
            targets=np.array([[1]]),
            noisy=np.array([[0]]),
            active_mask=np.array([[True]]),
            role_states=np.array([[1]]),
            roles={"head": 1},
            allowed=np.array([[[True, False]]]),
        )


def test_classification_reports_calibration_and_empty_denominator() -> None:
    result = classification_summary(np.array([[0.9, 0.1], [0.9, 0.1]]), np.array([0, 1]))
    assert result["accuracy"] == 0.5
    assert result["mean_confidence"] == pytest.approx(0.9)
    assert result["ece"] == pytest.approx(0.4)
    assert result["multiclass_brier"] == pytest.approx(0.82)
    empty = classification_summary(np.zeros((0, 2)), np.array([], dtype=int))
    assert empty["total"] == 0 and empty["accuracy"] is None
    with pytest.raises(UgiRealismModelAttributionError):
        classification_summary(np.array([[1.1, -0.1]]), np.array([0]))


def _row(identifier: str, head: str, tail: str) -> dict[str, str]:
    return {
        "product_id": identifier,
        "primary_product_fold": "train",
        "is_source_adjudicated_measured_product": "true",
        "head_smiles": head,
        "tail_smiles": tail,
        "head_family_fold": "train",
        "tail_family_fold": "train",
    }


def test_diverse_selection_reads_no_nontrain_structural_fields() -> None:
    rows = [_row("one", "a", "x"), _row("two", "b", "y"), _row("three", "a", "x")]
    # These rows deliberately have no structure fields: the metadata exclusion must happen first.
    rows += [
        {"primary_product_fold": "heldout"},
        {"primary_product_fold": "calibration"},
        {"primary_product_fold": "train", "is_source_adjudicated_measured_product": "false"},
    ]
    selected, audit = select_measured_train_rows(rows, roles=["head", "tail"], records=2, seed=3)
    repeated, repeated_audit = select_measured_train_rows(
        reversed(rows), roles=["head", "tail"], records=2, seed=3
    )
    assert selected == repeated and audit == repeated_audit
    assert audit["selected_components_by_role"] == {"head": 2, "tail": 2}
    assert audit["eligible_measured_train_products"] == 3
    bad = _row("bad", "a", "x")
    bad["tail_family_fold"] = "heldout"
    with pytest.raises(UgiRealismModelAttributionError, match="fold disagrees"):
        select_measured_train_rows([bad], roles=["head", "tail"], records=1, seed=3)


def test_existing_noising_is_deterministic_and_preserves_adapter_states() -> None:
    torch = pytest.importorskip("torch")
    from forge.model.reaction_program_flow import noise_synthesis_program_batch

    clean = {
        "nodes": torch.tensor([[0, 1, 0, 1]]),
        "parents": torch.tensor([[0, 0, 1, 2]]),
        "parent_bonds": torch.tensor([[0, 1, 0, 1]]),
        "node_mask": torch.ones((1, 4), dtype=torch.bool),
        "atom_variable_mask": torch.tensor([[False, True, True, False]]),
        "parent_variable_mask": torch.tensor([[False, False, True, True]]),
        "parent_bond_variable_mask": torch.tensor([[False, False, True, True]]),
        "closure_left": torch.zeros((1, 1), dtype=torch.long),
        "closure_right": torch.ones((1, 1), dtype=torch.long),
        "closure_bonds": torch.ones((1, 1), dtype=torch.long),
        "closure_endpoint_variable_mask": torch.zeros((1, 1), dtype=torch.bool),
        "closure_bond_variable_mask": torch.zeros((1, 1), dtype=torch.bool),
    }

    def noise() -> dict:
        return noise_synthesis_program_batch(
            clean,
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.2]),
            torch.Generator().manual_seed(31),
        )

    first, second = noise(), noise()
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert torch.equal(first["nodes"][:, [0, 3]], clean["nodes"][:, [0, 3]])
    assert torch.equal(first["parents"][:, :2], clean["parents"][:, :2])
    assert torch.equal(first["closure_bonds"], clean["closure_bonds"])


def test_tampered_input_fails_before_model_or_output(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    from experiments.phase1.multireaction import ugi_realism_model_attribution as audit

    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/multireaction/ugi_realism_model_attribution_v1.json").read_text()
    )
    payload = tmp_path / "changed.bin"
    payload.write_bytes(b"tampered")
    config["inputs"]["base_checkpoint_archive"] = {"path": "changed.bin", "sha256": "0" * 64}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(audit, "_load_base_package", lambda **_: pytest.fail("model loaded"))
    with pytest.raises(PinError, match="changed"):
        audit.run_ugi_realism_model_attribution(tmp_path, config_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_runtime_parent_attribution_uses_training_mask() -> None:
    torch = pytest.importorskip("torch")
    from forge.model.sparse_topology_feasibility import _parent_candidate_mask

    # The raw largest score is an impossible future parent, which must not count as an error.
    logits = np.array([[[0.0, 0.0, 99.0], [4.0, 0.0, 99.0], [4.0, 0.0, 99.0]]])
    result = coordinate_summaries(
        logits=logits,
        targets=np.array([[0, 0, 0]]),
        noisy=np.array([[0, 0, 1]]),
        active_mask=np.array([[False, True, True]]),
        role_states=np.ones((1, 3), dtype=int),
        roles={"head": 1},
        allowed=_parent_candidate_mask(torch.ones((1, 3), dtype=torch.bool)).numpy(),
    )
    assert result["head"]["all_variable"]["softmax"]["accuracy"] == 1.0


def test_training_objective_uses_effective_checkpoint_and_loss_receipts() -> None:
    pytest.importorskip("torch")
    from experiments.phase1.multireaction.ugi_realism_model_attribution import (
        _training_objective_audit,
    )

    arguments = {
        "package": {
            "model_config": {
                "semantic_objective": {
                    "topology_conditioned_chemistry_weight": 1.0,
                }
            }
        },
        "training": {
            "model": {},
            "arms": {"arm": {"final_loss": {"program_3_topology_conditioned_chemistry_ce": 0.12}}},
        },
        "design": {"model": {}},
        "training_config": {"intervention": {"topology_conditioned_chemistry_second_pass": True}},
        "arm_id": "arm",
        "program_state": 3,
    }
    audit = _training_objective_audit(**arguments)
    assert audit["topology_conditioned_chemistry_weight"] == 1.0
    assert audit["base_design_snapshot_weight"] is None
    assert audit["final_ugi_topology_conditioned_chemistry_ce"] == 0.12
    arguments["training_config"]["intervention"][
        "topology_conditioned_chemistry_second_pass"
    ] = False
    with pytest.raises(UgiRealismModelAttributionError, match="disagree"):
        _training_objective_audit(**arguments)
    arguments["training_config"]["intervention"][
        "topology_conditioned_chemistry_second_pass"
    ] = True
    arguments["training"]["arms"]["arm"]["final_loss"] = {}
    with pytest.raises(UgiRealismModelAttributionError, match="loss receipt"):
        _training_objective_audit(**arguments)
