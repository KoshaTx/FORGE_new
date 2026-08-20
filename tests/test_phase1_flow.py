from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

torch = pytest.importorskip("torch")

from forge.model.defog_feasibility import AtomState  # noqa: E402
from forge.model.phase1_flow import (  # noqa: E402
    Phase1FlowError,
    SparseWholeLipidFlow,
    TrainingGraphRecord,
    _checkpoint_package,
    _degree_continuation_log_prior,
    _evaluate_model,
    _flow_loss,
    _load_resume_checkpoint,
    _model_architecture_kwargs,
    _model_state_sha256,
    _noisy_topology_features,
    _parent_distance_log_prior,
    _save_checkpoint_atomic,
    _size_bucket,
    _source_mixture_diagnostic,
    _validate_config,
    _validate_effective_training_config,
    run_tiny_overfit_gate,
)
from forge.model.sparse_topology_feasibility import (  # noqa: E402
    _noise_sparse_batch,
    collate_sparse_records,
    tensorize_sparse_row,
)


def _record(structure_id: str, smiles: str, vocabulary: tuple[AtomState, ...]):
    molecule = Chem.MolFromSmiles(smiles)
    row = {
        "r0_structure_id": structure_id,
        "canonical_isomeric_smiles": smiles,
        "heavy_atoms": str(molecule.GetNumHeavyAtoms()),
        "elements": "|".join(sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})),
    }
    return tensorize_sparse_row(row, {state: index for index, state in enumerate(vocabulary)})


def _config() -> dict:
    return {
        "model": {
            "maximum_closure_slots": 12,
            "maximum_heavy_atoms": 282,
        },
        "training": {
            "learning_rate": 0.01,
            "gradient_clip_norm": 10.0,
        },
        "overfit_gate": {
            "device": "cpu",
            "records": 8,
            "steps": 120,
            "hidden_dim": 32,
            "layers": 2,
            "dropout": 0.0,
            "minimum_total_loss_reduction_fraction": 0.35,
        },
    }


def test_size_buckets_cover_declared_support() -> None:
    assert _size_bucket(40) == "le40"
    assert _size_bucket(41) == "41_64"
    assert _size_bucket(96) == "65_96"
    assert _size_bucket(128) == "97_128"
    assert _size_bucket(282) == "gt128"


def test_repository_v2_config_freezes_lipid_context_without_region_templates() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "configs/model/phase1_product_pretrain_v2.json").read_text())
    _validate_config(config)
    model = config["model"]
    assert model["topology_context"] is True
    assert model["parent_distance_buckets"] == 16
    assert not any("head" in key or "tail" in key or "linker" in key for key in model)

    invalid = json.loads(json.dumps(config))
    invalid["model"]["topology_context"] = False
    with pytest.raises(Phase1FlowError, match="requires noisy-state topology context"):
        _validate_config(invalid)


def test_repository_v3_config_adds_explicit_aromatic_and_lipid_root_support() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "configs/model/phase1_product_pretrain_v3.json").read_text())
    _validate_config(config)
    model = config["model"]
    assert model["bond_classes"] == 4
    assert model["preserve_aromaticity"] is True
    assert model["root_strategy"] == "lipid_polar"
    assert model["region_classes"] == 3
    assert model["region_scheme"] == "polar_structural_v2"
    assert model["r1_atom_vocabulary_extensions"] == [
        {
            "symbol": "O",
            "formal_charge": 0,
            "aromatic": True,
            "explicit_hydrogens": 0,
        }
    ]

    invalid = json.loads(json.dumps(config))
    invalid["model"]["bond_classes"] = 3
    with pytest.raises(Phase1FlowError, match="aromatic bond support"):
        _validate_config(invalid)


def test_source_mixture_diagnostic_exposes_drift_without_false_gate() -> None:
    diagnostic = _source_mixture_diagnostic(
        {"r0": 0.9, "r1": 0.1},
        {"r0": 199, "r1": 13},
        212,
    )
    assert diagnostic["expected_fractions"] == {"r0": 0.9, "r1": 0.1}
    assert diagnostic["observed_fractions"]["r0"] == pytest.approx(199 / 212)
    assert diagnostic["maximum_absolute_error"] == pytest.approx(abs(199 / 212 - 0.9))
    assert diagnostic["is_diagnostic_not_acceptance_gate"] is True

    with pytest.raises(Phase1FlowError, match="do not equal"):
        _source_mixture_diagnostic({"r0": 0.9, "r1": 0.1}, {"r0": 1}, 2)
    with pytest.raises(Phase1FlowError, match="outside"):
        _source_mixture_diagnostic(
            {"r0": 0.9, "r1": 0.1},
            {"r0": 1, "r1": 0, "unexpected": 1},
            2,
        )


def test_batch_budget_must_preserve_declared_heavy_atom_support() -> None:
    effective = {
        "model": {"maximum_heavy_atoms": 282},
        "training": {
            "maximum_graphs_per_batch": 8,
            "maximum_atoms_per_batch": 512,
            "maximum_parent_pointer_logits_per_batch": 32768,
        },
    }
    with pytest.raises(Phase1FlowError, match="parent-pointer budget"):
        _validate_effective_training_config(effective)

    effective["training"]["maximum_parent_pointer_logits_per_batch"] = 282 * 282
    _validate_effective_training_config(effective)


def test_sparse_whole_lipid_flow_predicts_counts_and_topology() -> None:
    vocabulary = (AtomState("C", 0, False), AtomState("N", 0, False))
    records = (
        _record("a", "CCCCN", vocabulary),
        _record("b", "C1CCNCC1", vocabulary),
    )
    clean = collate_sparse_records(records, 6, 12)
    generator = torch.Generator().manual_seed(7)
    node_p0 = torch.tensor([0.8, 0.2])
    bond_p0 = torch.tensor([0.8, 0.1, 0.1])
    t = torch.tensor([0.2, 0.8])
    noisy = _noise_sparse_batch(clean, node_p0, bond_p0, t, generator)
    model = SparseWholeLipidFlow(
        node_classes=2,
        hidden_dim=16,
        layers=2,
        maximum_closures=12,
        maximum_heavy_atoms=282,
        dropout=0.0,
    )

    predictions = model(
        noisy["nodes"],
        noisy["parents"],
        noisy["parent_bonds"],
        noisy["closure_left"],
        noisy["closure_right"],
        noisy["closure_bonds"],
        t,
        clean["node_mask"],
        clean["child_mask"],
        clean["closure_mask"],
    )
    loss, components = _flow_loss(predictions, clean)

    assert torch.isfinite(loss)
    assert predictions["node_count"].shape == (2, 283)
    assert predictions["closure_count"].shape == (2, 13)
    assert components["total_with_counts"] > components["total"]

    changed_closure_bonds = noisy["closure_bonds"].clone()
    changed_closure_bonds[1, 0] = (changed_closure_bonds[1, 0] + 1) % 3
    changed = model(
        noisy["nodes"],
        noisy["parents"],
        noisy["parent_bonds"],
        noisy["closure_left"],
        noisy["closure_right"],
        changed_closure_bonds,
        t,
        clean["node_mask"],
        clean["child_mask"],
        clean["closure_mask"],
    )
    assert not torch.equal(changed["nodes"][1], predictions["nodes"][1])
    assert not torch.equal(changed["closure_bonds"][1], predictions["closure_bonds"][1])

    first_validation = _evaluate_model(
        model,
        (records, records),
        device=torch.device("cpu"),
        maximum_closures=12,
        node_marginal=node_p0,
        bond_marginal=bond_p0,
        seed=19,
    )
    second_validation = _evaluate_model(
        model,
        (records, records),
        device=torch.device("cpu"),
        maximum_closures=12,
        node_marginal=node_p0,
        bond_marginal=bond_p0,
        seed=19,
    )
    assert second_validation == first_validation


def test_lipid_context_uses_noisy_state_and_relative_parent_distance() -> None:
    vocabulary = (AtomState("C", 0, False), AtomState("N", 0, False))
    records = (
        _record("chain", "CCCCN", vocabulary),
        _record("branch", "CC(C)CN", vocabulary),
    )
    clean = collate_sparse_records(records, 5, 12)
    features = _noisy_topology_features(
        clean["parents"],
        clean["parent_bonds"],
        clean["closure_left"],
        clean["closure_right"],
        clean["closure_bonds"],
        clean["node_mask"],
        clean["child_mask"],
        clean["closure_mask"],
    )
    changed_parents = clean["parents"].clone()
    changed_parents[0, 4] = 0
    changed = _noisy_topology_features(
        changed_parents,
        clean["parent_bonds"],
        clean["closure_left"],
        clean["closure_right"],
        clean["closure_bonds"],
        clean["node_mask"],
        clean["child_mask"],
        clean["closure_mask"],
    )
    assert features.shape == (2, 5, 4)
    assert not torch.equal(features[0], changed[0])

    model = SparseWholeLipidFlow(
        node_classes=2,
        hidden_dim=16,
        layers=1,
        maximum_closures=12,
        maximum_heavy_atoms=282,
        dropout=0.0,
        topology_context=True,
        parent_distance_buckets=4,
    )
    with torch.no_grad():
        model.parent_query.weight.zero_()
        model.parent_query.bias.zero_()
        model.parent_key.weight.zero_()
        model.parent_key.bias.zero_()
        model.parent_distance_bias.copy_(torch.tensor([0.0, -1.0, -2.0, -3.0, -4.0]))
    predictions = model(
        clean["nodes"][:1],
        clean["parents"][:1],
        clean["parent_bonds"][:1],
        clean["closure_left"][:1],
        clean["closure_right"][:1],
        clean["closure_bonds"][:1],
        torch.tensor([0.5]),
        clean["node_mask"][:1],
        clean["child_mask"][:1],
        clean["closure_mask"][:1],
    )
    assert predictions["parents"][0, 4, :4].tolist() == pytest.approx([-4.0, -3.0, -2.0, -1.0])


def test_parent_distance_prior_is_training_only_bounded_and_nonrestrictive() -> None:
    vocabulary = (AtomState("C", 0, False), AtomState("N", 0, False))
    records = (
        _record("chain", "CCCCN", vocabulary),
        _record("branch", "CC(C)CN", vocabulary),
    )
    prior = _parent_distance_log_prior(
        records,
        buckets=4,
        probability_floor=0.01,
        strength=1.0,
    )
    assert prior.shape == (5,)
    assert prior[0] == 0.0
    assert np.isfinite(prior).all()
    assert prior[1:].max() == pytest.approx(0.0)
    assert prior[1:].min() > np.log(0.01) - 1.0
    assert _model_architecture_kwargs({}) == {
        "bond_classes": 3,
        "topology_context": False,
        "parent_distance_buckets": 0,
        "closure_ring_size_buckets": 0,
        "region_classes": 0,
    }


def test_region_topology_priors_preserve_support_and_distinguish_tail_branching() -> None:
    records = (
        TrainingGraphRecord(
            structure_id="branched-head-linear-tail",
            canonical_smiles="",
            node_states=np.zeros(7, dtype=np.int64),
            parents=np.asarray([0, 0, 0, 0, 1, 4, 5], dtype=np.int64),
            parent_bonds=np.zeros(7, dtype=np.int64),
            closure_left=np.asarray([1], dtype=np.int64),
            closure_right=np.asarray([3], dtype=np.int64),
            closure_bonds=np.zeros(1, dtype=np.int64),
            region_states=np.asarray([0, 0, 0, 0, 2, 2, 2], dtype=np.int64),
        ),
        TrainingGraphRecord(
            structure_id="linear",
            canonical_smiles="",
            node_states=np.zeros(5, dtype=np.int64),
            parents=np.asarray([0, 0, 1, 2, 3], dtype=np.int64),
            parent_bonds=np.zeros(5, dtype=np.int64),
            closure_left=np.asarray([], dtype=np.int64),
            closure_right=np.asarray([], dtype=np.int64),
            closure_bonds=np.asarray([], dtype=np.int64),
            region_states=np.asarray([0, 1, 2, 2, 2], dtype=np.int64),
        ),
    )

    degree = _degree_continuation_log_prior(
        records,
        region_classes=3,
        maximum_degree=8,
        probability_floor=0.01,
    )
    assert degree is not None
    assert np.isfinite(degree[:, :-1]).all()
    assert np.isneginf(degree[:, -1]).all()
    assert degree[2, 2] < degree[0, 2]


def test_tiny_overfit_gate_reduces_joint_loss() -> None:
    torch.set_num_threads(1)
    vocabulary = (AtomState("C", 0, False), AtomState("N", 0, False))
    smiles = (
        "CCCCN",
        "CCCN",
        "CCCCCN",
        "CCNCC",
        "C1CCNCC1",
        "CC(C)CN",
        "CCN(C)CC",
        "CCCCCCN",
    )
    records = tuple(_record(str(index), value, vocabulary) for index, value in enumerate(smiles))

    result = run_tiny_overfit_gate(records, vocabulary, _config(), seed=23)
    repeated = run_tiny_overfit_gate(records, vocabulary, _config(), seed=23)

    assert result["status"] == "pass"
    assert result["loss_reduction_fraction"] >= 0.35
    for key in (
        "initial_mean_total_loss",
        "final_mean_total_loss",
        "loss_reduction_fraction",
    ):
        assert repeated[key] == pytest.approx(result[key], abs=0.0)


def test_resumable_checkpoint_restores_model_optimizer_and_rng(tmp_path: Path) -> None:
    vocabulary = (AtomState("C", 0, False), AtomState("N", 0, False))
    model_config = {
        "hidden_dim": 8,
        "layers": 1,
        "maximum_closure_slots": 12,
        "maximum_heavy_atoms": 282,
        "dropout": 0.0,
    }
    model = SparseWholeLipidFlow(
        node_classes=2,
        hidden_dim=8,
        layers=1,
        maximum_closures=12,
        maximum_heavy_atoms=282,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(101)
    rng = np.random.default_rng(202)
    package = _checkpoint_package(
        config_sha256="a" * 64,
        data_manifest_sha256="b" * 64,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        atom_vocabulary=vocabulary,
        node_marginal=np.asarray([0.8, 0.2]),
        bond_marginal=np.asarray([0.8, 0.1, 0.1]),
        step=1,
        generator=generator,
        rng=rng,
        losses=({"total_with_counts": 1.0},),
        source_totals={"r0": 1},
        bucket_totals={"le40": 1},
        examples_seen=1,
        validation_history=({"step": 0, "metrics": {"total_with_counts": 2.0}},),
        best_validation_step=0,
        best_validation_loss=2.0,
        elapsed_wall_seconds=3.0,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    _save_checkpoint_atomic(checkpoint, package)
    expected_generator_values = torch.rand(4, generator=generator)
    expected_numpy_values = rng.random(4)

    restored_model = SparseWholeLipidFlow(
        node_classes=2,
        hidden_dim=8,
        layers=1,
        maximum_closures=12,
        maximum_heavy_atoms=282,
        dropout=0.0,
    )
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_generator = torch.Generator().manual_seed(999)
    restored_rng = np.random.default_rng(999)
    restored = _load_resume_checkpoint(
        checkpoint,
        output_dir=tmp_path,
        config_sha256="a" * 64,
        data_manifest_sha256="b" * 64,
        model=restored_model,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
        generator=restored_generator,
        rng=restored_rng,
    )

    assert restored["step"] == 1
    assert _model_state_sha256(restored_model) == package["model_state_sha256"]
    assert torch.equal(torch.rand(4, generator=restored_generator), expected_generator_values)
    assert restored_rng.random(4) == pytest.approx(expected_numpy_values, abs=0.0)
