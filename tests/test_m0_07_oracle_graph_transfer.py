from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from forge.potency.oracle_graph import (
    DMPNNEncoder,
    GraphFeatureVocabulary,
    OracleGraphRecord,
    tensorize_smiles,
)
from forge.potency.oracle_graph_pretraining import RESULT_SCHEMA_VERSION
from forge.potency.oracle_graph_transfer import (
    FIT_SCHEMA_VERSION,
    OracleGraphTransferError,
    build_transfer_model,
    ensemble_transfer_fits,
    fit_transfer_job,
    load_full_pretraining_checkpoint,
    validate_completed_transfer_fit,
)


def _vocabulary() -> GraphFeatureVocabulary:
    return GraphFeatureVocabulary(
        elements=("C", "N", "O"),
        formal_charges=("-1", "0", "1"),
        total_degrees=("1", "2", "3", "4"),
        total_hydrogens=("0", "1", "2", "3"),
        hybridizations=("SP", "SP2", "SP3"),
        bond_types=("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"),
    )


def _checkpoint(vocabulary: GraphFeatureVocabulary, *, mode: str = "full") -> dict:
    torch.manual_seed(7)
    encoder = DMPNNEncoder(
        vocabulary.atom_feature_dim,
        vocabulary.bond_feature_dim,
        hidden_dim=8,
        depth=2,
        dropout=0.1,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode": mode,
        "data_policy": {
            "biological_labels_used": False,
            "component_annotations_used": False,
            "provenance_fields_used": False,
            "virtual_candidates_used": False,
            "stereochemistry_used": False,
            "graph_truncation_allowed": False,
        },
        "feature_vocabulary": vocabulary.to_dict(),
        "architecture": {
            "encoder": "sparse_dmpnn",
            "hidden_dim": 8,
            "depth": 2,
            "dropout": 0.1,
            "readout": "sum_mean",
        },
        "encoder_state_dict": encoder.state_dict(),
    }


def _records(vocabulary: GraphFeatureVocabulary) -> list[OracleGraphRecord]:
    smiles = (
        "CCO",
        "CCN",
        "CCCO",
        "CCCN",
        "CCCCO",
        "CCCCN",
        "CC(=O)O",
        "CC(=O)N",
        "CCC(=O)O",
        "CCC(=O)N",
        "COC",
        "CNC",
        "CCOC",
        "CCNC",
        "CC=O",
        "CC=N",
        "CCC=O",
        "CCC=N",
        "C1CCOCC1",
        "C1CCNCC1",
        "c1ccccc1",
        "c1ccncc1",
        "CC#N",
        "CCC#N",
    )
    records = []
    for index, value in enumerate(smiles):
        graph = tensorize_smiles(value, vocabulary, label=str(index))
        role = tensorize_smiles("CC", vocabulary, label=f"role-{index}")
        records.append(
            OracleGraphRecord(
                label=f"L{index:02d}",
                product=graph,
                amine=role,
                aldehyde=role,
                isocyanide=role,
                targets=torch.tensor(
                    [0.25 * index + (index % 3), -0.2 * index + (index % 4)],
                    dtype=torch.float32,
                ),
            )
        )
    return records


def _assignments(records: list[OracleGraphRecord]) -> list[dict[str, str]]:
    rows = []
    for index, record in enumerate(records):
        stage = "train" if index < 16 else "calibration" if index < 20 else "test"
        rows.append(
            {
                "scheme": "held_aldehyde_5fold",
                "fold": "0",
                "label": record.label,
                "stage": stage,
            }
        )
    return rows


def _row(variant: str, seed: int = 1729) -> dict:
    return {
        "job_id": f"{variant}-{seed}",
        "ensemble_id": f"{variant}-ensemble",
        "architecture": variant,
        "endpoint": "expt_Hela",
        "scheme": "held_aldehyde_5fold",
        "fold": 0,
        "seed": seed,
        "selection_eligible": "true",
        "train_rows": 16,
        "calibration_rows": 4,
        "test_rows": 4,
        "output_relative_path": f"unused/{variant}/{seed}.json",
    }


def _config() -> dict:
    return {
        "model": {
            "head_learning_rate": 0.01,
            "encoder_learning_rate": 0.001,
            "weight_decay": 1e-5,
            "maximum_graphs_per_batch": 16,
            "maximum_atoms_per_batch": 512,
            "maximum_directed_edges_per_batch": 1024,
        },
        "epoch_selection": {
            "inner_validation_fraction": 0.25,
            "maximum_epochs": 2,
            "patience": 2,
            "minimum_delta": 0.0,
        },
    }


def _fit(
    variant: str,
    records: list[OracleGraphRecord],
    vocabulary: GraphFeatureVocabulary,
    *,
    seed: int = 1729,
) -> dict:
    return fit_transfer_job(
        _row(variant, seed),
        records=records,
        label_to_index={record.label: index for index, record in enumerate(records)},
        assignments=_assignments(records),
        vocabulary=vocabulary,
        checkpoint=_checkpoint(vocabulary),
        checkpoint_sha256="a" * 64,
        config=_config(),
        config_sha256="b" * 64,
        input_hashes={"checkpoint": "a" * 64},
    )


def test_full_checkpoint_gate_rejects_profile_and_accepts_synthetic_full(
    tmp_path: Path,
) -> None:
    vocabulary = _vocabulary()
    profile_path = tmp_path / "profile.pt"
    torch.save(_checkpoint(vocabulary, mode="profile"), profile_path)
    with pytest.raises(OracleGraphTransferError, match="only the completed full"):
        load_full_pretraining_checkpoint(profile_path, vocabulary=vocabulary)

    full_path = tmp_path / "full.pt"
    torch.save(_checkpoint(vocabulary), full_path)
    loaded = load_full_pretraining_checkpoint(full_path, vocabulary=vocabulary)
    assert loaded["mode"] == "full"
    model = build_transfer_model(
        "r0_pretrained_frozen_linear",
        vocabulary=vocabulary,
        checkpoint=loaded,
        seed=1729,
    )
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


@pytest.mark.parametrize(
    "variant",
    ["r0_pretrained_frozen_linear", "r0_pretrained_finetuned_dmpnn"],
)
def test_transfer_fit_is_train_only_and_poison_invariant(variant: str) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    vocabulary = _vocabulary()
    records = _records(vocabulary)
    first = _fit(variant, records, vocabulary)
    poisoned = [
        (
            record
            if index < 16
            else replace(record, targets=record.targets + torch.tensor([1000.0, -1000.0]))
        )
        for index, record in enumerate(records)
    ]
    second = _fit(variant, poisoned, vocabulary)
    assert first["epoch_selection"] == second["epoch_selection"]
    assert first["refit"] == second["refit"]
    assert first["calibration"]["prediction"] == second["calibration"]["prediction"]
    assert first["test"]["prediction"] == second["test"]["prediction"]
    assert first["boundary"]["calibration_or_test_used_for_epoch_selection"] is False
    if variant == "r0_pretrained_frozen_linear":
        assert (
            first["refit"]["encoder_sha256_before_calibration_or_test_access"]
            == first["pretrained_encoder"]["source_encoder_state_sha256"]
        )


def test_three_seed_ensemble_calibrates_after_seed_combination() -> None:
    vocabulary = _vocabulary()
    records = _records(vocabulary)
    fits = [
        _fit("r0_pretrained_frozen_linear", records, vocabulary, seed=seed)
        for seed in (1729, 11729, 21729)
    ]
    metric, predictions = ensemble_transfer_fits(fits, coverages=(0.8, 0.9, 0.95))
    assert metric["seed_count"] == 3
    assert metric["calibration_rows"] == 4
    assert metric["test_rows"] == 4
    assert metric["conformal_q90"] >= 0.0
    assert len(predictions) == 4
    assert all(row["ensemble_standard_deviation"] >= 0.0 for row in predictions)


def test_resumable_fit_validation_rejects_stale_output(tmp_path: Path) -> None:
    row = _row("r0_pretrained_frozen_linear")
    path = tmp_path / "fit.json"
    payload = {
        "schema_version": FIT_SCHEMA_VERSION,
        "status": "completed",
        "job": row,
        "config_sha256": "b" * 64,
        "input_hashes": {"checkpoint": "a" * 64},
        "boundary": {"calibration_and_test_accessed_after_model_hash": True},
    }
    path.write_text(__import__("json").dumps(payload))
    assert validate_completed_transfer_fit(
        path,
        row=row,
        config_sha256="b" * 64,
        input_hashes={"checkpoint": "a" * 64},
    )
    payload["config_sha256"] = "c" * 64
    path.write_text(__import__("json").dumps(payload))
    with pytest.raises(OracleGraphTransferError, match="stale or corrupt"):
        validate_completed_transfer_fit(
            path,
            row=row,
            config_sha256="b" * 64,
            input_hashes={"checkpoint": "a" * 64},
        )
