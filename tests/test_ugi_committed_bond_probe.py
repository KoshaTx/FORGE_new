"""Paired-input, selection and no-advancement invariants for the TRAIN sensitivity probe."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forge.core.hashing import PinError
from forge.model.defog_feasibility import AtomState
from forge.model.ugi_committed_bond_probe import (
    UgiCommittedBondProbeError,
    oracle_ester_commitment_masks,
    paired_committed_states,
    select_component_covering_panel,
    summarize_bond_response,
)


def _fixture():
    clean = {
        "nodes": torch.zeros((1, 7), dtype=torch.long),
        "parents": torch.tensor([[0, 0, 1, 2, 3, 4, 5]]),
        "parent_bonds": torch.zeros((1, 7), dtype=torch.long),
        "closure_left": torch.tensor([[1]]),
        "closure_right": torch.tensor([[2]]),
        "closure_bonds": torch.zeros((1, 1), dtype=torch.long),
        "node_mask": torch.ones((1, 7), dtype=torch.bool),
        "core_position_states": torch.tensor([[2, 1, 1, 1, 1, 1, 1]]),
        "role_states": torch.tensor([[0, 1, 1, 2, 2, 3, 3]]),
        "atom_variable_mask": torch.tensor([[False, True, True, True, True, True, True]]),
        "parent_variable_mask": torch.tensor([[False, False, True, True, True, True, True]]),
        "parent_bond_variable_mask": torch.tensor([[False, True, True, True, True, True, True]]),
        "closure_bond_variable_mask": torch.ones((1, 1), dtype=torch.bool),
        "closure_endpoint_variable_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    noisy = {
        key: clean[key].clone()
        for key in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        )
    }
    noisy["nodes"][0, 1:] = 1
    noisy["parents"][0, 2:] = 0
    noisy["parent_bonds"][0, [1, 2, 3, 5]] = 1
    commitments = {
        "parent_bonds": torch.tensor([[False, False, True, False, False, False, False]]),
        "closure_bonds": torch.zeros((1, 1), dtype=torch.bool),
    }
    return clean, noisy, commitments


def _predictions(clean, correct=True):
    result = {}
    for field in ("parent_bonds", "closure_bonds"):
        result[field] = torch.zeros((*clean[field].shape, 2))
        result[field][..., 0 if correct else 1] = 3
    return result


def test_commitment_changes_atoms_and_ester_bonds_only():
    clean, noisy, commitments = _fixture()
    stale, refreshed = paired_committed_states(clean, noisy, commitments)
    assert torch.equal(stale["parents"], clean["parents"])
    assert torch.equal(refreshed["nodes"], clean["nodes"])
    assert torch.equal(stale["nodes"], noisy["nodes"])
    assert stale["parent_bonds"][0, 2] == 1 and refreshed["parent_bonds"][0, 2] == 0
    for field, mask in commitments.items():
        assert torch.equal(stale[field][~mask], refreshed[field][~mask])
    assert noisy["parents"][0, 2] == 0  # Inputs were not mutated.
    commitments["parent_bonds"][0, 0] = True
    with pytest.raises(UgiCommittedBondProbeError, match="fixed or padded"):
        paired_committed_states(clean, noisy, commitments)


def test_fixed_state_corruption_fails():
    clean, noisy, commitments = _fixture()
    noisy["nodes"][0, 0] = 1
    with pytest.raises(UgiCommittedBondProbeError, match="fixed-state"):
        paired_committed_states(clean, noisy, commitments)


def _summary(clean, noisy, commitments, stale, refreshed, repeat=None):
    return summarize_bond_response(
        clean=clean,
        noisy=noisy,
        commitments=commitments,
        stale=stale,
        repeat=stale if repeat is None else repeat,
        refreshed=refreshed,
        roles={"head": 1, "body": 2, "tail": 3},
        repeat_atol=1e-6,
    )


def test_corrupted_denominators_exclude_commitments_and_keep_unchanged_separate():
    clean, noisy, commitments = _fixture()
    bad, good = _predictions(clean, False), _predictions(clean)
    result = _summary(clean, noisy, commitments, bad, good)
    assert result["gate_passed"]
    assert result["corrections"] == 3
    assert result["actually_corrupted_uncommitted_bonds"] == 3
    assert result["uncommitted_bonds"] == 6
    assert result["excluded_committed_bonds"]["parent_bonds"] == 1
    assert result["by_role"]["head"]["unchanged"]["stale"]["total"] == 1
    assert result["exact_bond_vector_recovery"]["all_uncommitted"] == {
        "records": 1,
        "stale": 0,
        "repeat": 0,
        "refreshed": 1,
    }


def test_improved_average_cannot_hide_role_accuracy_regression():
    clean, noisy, commitments = _fixture()
    stale, refreshed = _predictions(clean, False), _predictions(clean)
    stale["parent_bonds"][0, 5] = torch.tensor([0.1, 0.0])
    refreshed["parent_bonds"][0, 5] = torch.tensor([0.0, 0.1])
    result = _summary(clean, noisy, commitments, stale, refreshed)
    assert result["role_balanced_nll_improvement"] > 0
    assert not result["gate_checks"]["no_role_corrupted_accuracy_loss"]
    assert not result["gate_passed"]


def test_empty_corrupted_role_and_repeat_drift_prevent_advancement():
    clean, noisy, commitments = _fixture()
    stale, refreshed = _predictions(clean, False), _predictions(clean)
    noisy["parent_bonds"][0, 5] = 0
    result = _summary(clean, noisy, commitments, stale, refreshed)
    assert result["by_role"]["tail"]["actually_corrupted"]["stale"]["nll"] is None
    assert not result["gate_passed"]
    repeat = {key: value + 0.01 for key, value in stale.items()}
    result = _summary(clean, noisy, commitments, stale, refreshed, repeat)
    assert not result["gate_checks"]["identical_input_repeat_within_tolerance"]


def test_identical_conditions_do_not_pass_sensitivity_gate():
    clean, noisy, commitments = _fixture()
    predictions = _predictions(clean)
    assert not _summary(clean, noisy, commitments, predictions, predictions)["gate_passed"]


def test_component_coverage_uses_fixed_bound_and_metadata_before_structures():
    rows = [{"primary_product_fold": "heldout"}, {"primary_product_fold": "calibration"}]
    for index in range(3):
        rows.append(
            {
                "primary_product_fold": "train",
                "is_source_adjudicated_measured_product": "true",
                "product_id": str(index),
                "head_smiles": str(index),
                "head_family_fold": "train",
            }
        )
    selected, audit = select_component_covering_panel(rows, roles=["head"], records=3, seed=7)
    assert len(selected) == 3 and audit["all_admitted_measured_components_covered"]
    with pytest.raises(UgiCommittedBondProbeError, match="fixed panel"):
        select_component_covering_panel(rows, roles=["head"], records=2, seed=7)


def test_existing_ester_selector_constructs_only_exact_target_commitments():
    # A small artificial graph exercises the production selector; it is not corpus evidence.
    clean, _, _ = _fixture()
    clean["nodes"] = torch.tensor([[0, 0, 1, 0, 1, 0, 0]])
    clean["parents"] = torch.tensor([[0, 0, 1, 2, 3, 3, 5]])
    clean["parent_bonds"][0, 4] = 1
    clean["parent_bond_variable_mask"][0, 1] = False
    clean["closure_bonds"] = torch.empty((1, 0), dtype=torch.long)
    clean["closure_bond_variable_mask"] = torch.empty((1, 0), dtype=torch.bool)
    record = SimpleNamespace(
        program_id="test",
        node_count=7,
        component_blocks=[SimpleNamespace(role="body", start=0, stop=7)],
        core_position_states=clean["core_position_states"][0].numpy(),
        fixed_atom_mask=~clean["atom_variable_mask"][0].numpy(),
        fixed_parent_bond_mask=np.array([False, True, False, False, False, False, False]),
        fixed_closure_bond_mask=np.array([], dtype=bool),
        graph=SimpleNamespace(
            parents=clean["parents"][0].numpy(),
            closure_left=np.array([], dtype=int),
            closure_right=np.array([], dtype=int),
        ),
    )
    policy = SimpleNamespace(
        reaction_id="test",
        aldehyde_role="body",
        minimum_ester_side_carbons=0,
        minimum_ester_long_side_carbons=0,
    )
    masks, receipt = oracle_ester_commitment_masks(
        clean,
        [record],
        [AtomState("C", 0, False, 0), AtomState("O", 0, False, 0)],
        policy,
        bond_classes=4,
    )
    assert masks["parent_bonds"].sum() == 4
    assert masks["parent_bonds"][0].tolist() == [False, False, True, True, True, True, False]
    assert len(receipt[0]["committed_bonds"]) == 4
    clean["nodes"][0, 4] = 0
    with pytest.raises(UgiCommittedBondProbeError, match="atom assignment"):
        oracle_ester_commitment_masks(
            clean,
            [record],
            [AtomState("C", 0, False, 0), AtomState("O", 0, False, 0)],
            policy,
            bond_classes=4,
        )


def test_bad_pin_writes_failure_receipt_before_loading_model(tmp_path, monkeypatch):
    from experiments.phase1.multireaction import ugi_committed_bond_probe as runner

    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/multireaction/ugi_committed_bond_probe_v1.json").read_text()
    )
    (tmp_path / "payload").write_text("changed")
    config["inputs"]["base_checkpoint_archive"] = {"path": "payload", "sha256": "0" * 64}
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_load_base_package", lambda **_: pytest.fail("checkpoint loaded"))
    with pytest.raises(PinError):
        runner.run_ugi_committed_bond_probe(tmp_path, Path("config.json"), Path("output"))
    receipt = json.loads((tmp_path / "output/failure.json").read_text())
    assert receipt["status"] == "failed_no_admitted_metrics"
    assert not (tmp_path / "output/result.json").exists()


def test_admission_filters_existing_policy_before_component_selection(tmp_path, monkeypatch):
    from experiments.phase1.multireaction import ugi_committed_bond_probe as runner

    rows = [{"primary_product_fold": "heldout"}]
    for index in range(3):
        rows.append(
            {
                "primary_product_fold": "train",
                "is_source_adjudicated_measured_product": "true",
                "product_id": str(index),
                "canonical_product_smiles": f"product{index}",
                "head_smiles": str(index),
                "head_family_fold": "train",
            }
        )
    monkeypatch.setattr(runner, "iter_csv", lambda path: iter(rows))
    monkeypatch.setattr(runner, "program_from_layout_record", lambda record, **_: record)
    policy = object()

    def admitted(record, supplied):
        assert supplied is policy
        return record != 2

    monkeypatch.setattr(runner, "program_admitted_by_ester_policy", admitted)

    class Cache:
        vocabulary = object()

        def indices(self, *, program_id, fold):
            assert program_id == "test" and fold == "train"
            return [0, 1, 2, 3]

        def record_id(self, index):
            return str(index)

        def canonical_smiles(self, index):
            assert index != 3  # Non-measured TRAIN structures are not interpreted either.
            return f"product{index}"

        def record(self, index):
            assert index != 3
            return index

        def records(self, indices):
            return [self.record(index) for index in indices]

    records, selection = runner._admitted_panel(
        Cache(),
        {"ugi_assignments": tmp_path / "unused"},
        {"roles": ["head"], "records": 2, "seed": 7, "target_program": "test"},
        policy,
    )
    assert set(records) == {0, 1}
    assert selection["admitted_products"] == 2 and selection["measured_train_products"] == 3
    assert selection["nontrain_rows_masked_before_structure_access"] == 1
