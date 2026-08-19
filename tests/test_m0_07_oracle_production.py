from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn
import torch
from rdkit import rdBase
from sklearn.dummy import DummyRegressor

from forge.bio import oracle_graph_matrix as graph_matrix
from forge.bio.oracle_classical import build_feature_bundle
from forge.bio.oracle_freeze import RESULT_SCHEMA_VERSION as FREEZE_RESULT_SCHEMA_VERSION
from forge.bio.oracle_graph import (
    DMPNNEncoder,
    GraphFeatureVocabulary,
    OracleGraphRecord,
    WholeGraphRegressor,
    tensorize_smiles,
)
from forge.bio.oracle_production import (
    CHECKPOINT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    OracleProductionError,
    _build_applicability_index,
    _build_chemistry_contract,
    _conservative_domain_scores,
    build_domain_radii,
    classify_production_ugi_candidate,
    collect_selected_graph_epochs,
    load_production_checkpoint,
    predict_production_graph_records,
    predict_production_smiles,
    predict_production_ugi_smiles,
)

_TEST_FIT_CONFIG_SHA256 = "a" * 64
_TEST_FIT_INPUT_HASHES = {"curated_oracle_data": "b" * 64}


def _freeze_result() -> dict:
    schemes = [
        "lantern_scaffold_balanced",
        "held_head_5fold",
        "held_aldehyde_5fold",
        "held_isocyanide_5fold",
        "held_head_aldehyde_pair_5fold",
        "held_head_isocyanide_pair_5fold",
        "held_aldehyde_isocyanide_pair_5fold",
    ]
    domains = {
        "unseen_head": {
            "passes": False,
            "action": "abstain",
            "required_schemes": [
                "lantern_scaffold_balanced",
                "held_head_5fold",
            ],
        },
        "unseen_isocyanide": {
            "passes": True,
            "action": "conformal_lower_confidence_guidance",
            "required_schemes": [
                "lantern_scaffold_balanced",
                "held_isocyanide_5fold",
            ],
        },
    }
    return {
        "selection_contract": {
            "eligible_schemes": schemes,
            "endpoints": {"expt_Hela": 0.5, "expt_Raw": 0.5},
        },
        "applicability_policy": {
            "endpoints": {endpoint: {"domains": domains} for endpoint in ("expt_Hela", "expt_Raw")},
            "unsupported_domains": {
                "three_unseen_components": {
                    "action": "abstain",
                    "reason": "no held-three-role evidence",
                },
                "non_ugi_final_assembly": {
                    "action": "abstain",
                    "reason": "outside AGILE chemistry",
                },
                "in_vivo_endpoint": {
                    "action": "abstain",
                    "reason": "in vitro labels only",
                },
            },
        },
    }


def _metric_rows() -> list[dict]:
    result = _freeze_result()
    rows = []
    for endpoint in result["selection_contract"]["endpoints"]:
        for scheme_index, scheme in enumerate(result["selection_contract"]["eligible_schemes"]):
            folds = [0] if scheme == "lantern_scaffold_balanced" else range(5)
            for fold in folds:
                base = 1.0 + 0.1 * scheme_index + 0.01 * fold
                rows.append(
                    {
                        "representation": "role_dmpnn",
                        "model": "ensemble",
                        "endpoint": endpoint,
                        "scheme": scheme,
                        "fold": fold,
                        "conformal_q80": base,
                        "conformal_q90": base + 1.0,
                        "conformal_q95": base + 2.0,
                    }
                )
    rows.append(
        {
            "representation": "other",
            "model": "ensemble",
            "endpoint": "expt_Hela",
            "scheme": "lantern_scaffold_balanced",
            "fold": 0,
            "conformal_q80": 999,
            "conformal_q90": 999,
            "conformal_q95": 999,
        }
    )
    return rows


def test_domain_radii_use_conservative_fold_and_scheme_maxima(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.csv.gz"
    fieldnames = list(_metric_rows()[0])
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_metric_rows())
    result = build_domain_radii(
        metrics_path=path,
        selected={"representation": "role_dmpnn", "model": "ensemble"},
        freeze_result=_freeze_result(),
        coverages=[0.8, 0.9, 0.95],
    )
    hela = result["domain_radii"]["expt_Hela"]
    assert not hela["unseen_head"]["authorized"]
    assert hela["unseen_isocyanide"]["authorized"]
    assert hela["unseen_isocyanide"]["radii"]["q90"] == pytest.approx(2.34)
    assert not hela["three_unseen_components"]["authorized"]
    assert hela["three_unseen_components"]["action"] == "abstain"
    assert result["scheme_radii"]["expt_Hela"]["held_head_5fold"]["q95"] == pytest.approx(3.14)


def _write_fit(
    path: Path,
    *,
    repo_root: Path,
    architecture: str,
    endpoint: str,
    scheme: str,
    fold: int,
    seed: int,
    epoch: int,
    metadata_fold: int | None = None,
) -> tuple[Path, dict[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    job_id = hashlib.sha256(str(path).encode()).hexdigest()
    relative_path = str(path.resolve().relative_to(repo_root.resolve()))
    path.write_text(
        json.dumps(
            {
                "schema_version": "test_graph_fit.v1",
                "status": "completed",
                "job": {
                    "job_id": job_id,
                    "architecture": architecture,
                    "endpoint": endpoint,
                    "scheme": scheme,
                    "fold": fold if metadata_fold is None else metadata_fold,
                    "seed": seed,
                    "selection_eligible": "true",
                    "output_relative_path": relative_path,
                },
                "config_sha256": _TEST_FIT_CONFIG_SHA256,
                "input_hashes": _TEST_FIT_INPUT_HASHES,
                "epoch_selection": {
                    "best_epoch": epoch,
                    "calibration_or_test_targets_accessed": False,
                },
                "boundary": {
                    "calibration_or_test_used_for_scaling": False,
                    "calibration_or_test_used_for_epoch_selection": False,
                    "calibration_or_test_used_for_weight_updates": False,
                    "calibration_and_test_accessed_after_checkpoint_hash": True,
                },
            }
        )
    )
    return path.resolve(), {
        "job_id": job_id,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_selected_epoch_rule_uses_only_train_selected_fit_metadata(
    tmp_path: Path,
) -> None:
    freeze = _freeze_result()
    architecture = "role_dmpnn"
    seed = 1729
    values = []
    fit_source_index = {}
    for scheme_index, scheme in enumerate(freeze["selection_contract"]["eligible_schemes"]):
        folds = [0] if scheme == "lantern_scaffold_balanced" else range(5)
        for fold in folds:
            epoch = 10 + scheme_index + fold
            values.append(epoch)
            path, source = _write_fit(
                tmp_path
                / architecture
                / "expt_Hela"
                / scheme
                / f"fold-{fold}"
                / f"seed-{seed}.json",
                repo_root=tmp_path,
                architecture=architecture,
                endpoint="expt_Hela",
                scheme=scheme,
                fold=fold,
                seed=seed,
                epoch=epoch,
            )
            fit_source_index[path] = source
    result = collect_selected_graph_epochs(
        repo_root=tmp_path,
        fit_root=tmp_path,
        fit_source_index=fit_source_index,
        fit_schema_version="test_graph_fit.v1",
        expected_config_sha256=_TEST_FIT_CONFIG_SHA256,
        expected_input_hashes=_TEST_FIT_INPUT_HASHES,
        maximum_epoch=100,
        representation=architecture,
        endpoints=["expt_Hela"],
        repeat_seeds=[seed],
        eligible_schemes=freeze["selection_contract"]["eligible_schemes"],
    )
    expected = int(sorted(values)[len(values) // 2])
    assert result["expt_Hela"][str(seed)]["epoch_count"] == expected
    assert result["expt_Hela"][str(seed)]["source_fits"] == 31
    assert not result["expt_Hela"][str(seed)]["selection_targets_accessed"]


def test_selected_epoch_rule_fails_when_one_fold_is_missing(
    tmp_path: Path,
) -> None:
    freeze = _freeze_result()
    scheme = "lantern_scaffold_balanced"
    path, source = _write_fit(
        tmp_path / "role_dmpnn" / "expt_Hela" / scheme / "fold-0" / "seed-1729.json",
        repo_root=tmp_path,
        architecture="role_dmpnn",
        endpoint="expt_Hela",
        scheme=scheme,
        fold=0,
        seed=1729,
        epoch=10,
    )
    with pytest.raises(OracleProductionError, match="expected 5 epoch sources"):
        collect_selected_graph_epochs(
            repo_root=tmp_path,
            fit_root=tmp_path,
            fit_source_index={path: source},
            fit_schema_version="test_graph_fit.v1",
            expected_config_sha256=_TEST_FIT_CONFIG_SHA256,
            expected_input_hashes=_TEST_FIT_INPUT_HASHES,
            maximum_epoch=100,
            representation="role_dmpnn",
            endpoints=["expt_Hela"],
            repeat_seeds=[1729],
            eligible_schemes=freeze["selection_contract"]["eligible_schemes"],
        )


def test_selected_epoch_rule_rejects_hash_pinned_wrong_fold_metadata(
    tmp_path: Path,
) -> None:
    scheme = "lantern_scaffold_balanced"
    path, source = _write_fit(
        tmp_path / "role_dmpnn" / "expt_Hela" / scheme / "fold-0" / "seed-1729.json",
        repo_root=tmp_path,
        architecture="role_dmpnn",
        endpoint="expt_Hela",
        scheme=scheme,
        fold=0,
        metadata_fold=1,
        seed=1729,
        epoch=10,
    )
    with pytest.raises(OracleProductionError, match="invalid epoch source"):
        collect_selected_graph_epochs(
            repo_root=tmp_path,
            fit_root=tmp_path,
            fit_source_index={path: source},
            fit_schema_version="test_graph_fit.v1",
            expected_config_sha256=_TEST_FIT_CONFIG_SHA256,
            expected_input_hashes=_TEST_FIT_INPUT_HASHES,
            maximum_epoch=100,
            representation="role_dmpnn",
            endpoints=["expt_Hela"],
            repeat_seeds=[1729],
            eligible_schemes=[scheme],
        )


def test_checkpoint_loader_requires_hash_and_accepts_safe_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    selected = {
        "lane": "classical",
        "representation": "morgan_count",
        "model": "ridge",
    }
    curated_path = tmp_path / "curated.csv.gz"
    pd.DataFrame(
        {
            "label": [f"record-{index}" for index in range(1100)],
            "A_smiles": ["CN"] * 1100,
            "B_smiles": ["CC=O"] * 1100,
            "C_smiles": ["[C-]#[N+]"] * 1100,
        }
    ).to_csv(curated_path, index=False)
    curated_hash = hashlib.sha256(curated_path.read_bytes()).hexdigest()
    source_config_path = tmp_path / "classical.json"
    source_config_path.write_text(
        json.dumps(
            {
                "inputs": {
                    "curated_oracle_data": {
                        "path": curated_path.name,
                        "sha256": curated_hash,
                    }
                }
            }
        )
    )
    source_config_hash = hashlib.sha256(source_config_path.read_bytes()).hexdigest()
    selected_result_path = tmp_path / "classical_result.json"
    selected_result = {
        "schema_version": "test_classical_result.v1",
        "status": "completed_test_classical_lane",
        "configuration": {
            "path": source_config_path.name,
            "sha256": source_config_hash,
            "bytes": source_config_path.stat().st_size,
        },
    }
    selected_result_path.write_text(json.dumps(selected_result))
    registry_path = tmp_path / "registry.json"
    registry_path.write_bytes(Path("data/vendor/qualified_reactions_v1.json").read_bytes())
    variant_path = tmp_path / "variant.yaml"
    variant_path.write_bytes(Path("configs/assembly/ugi_variant.yaml").read_bytes())
    freeze_path = tmp_path / "freeze.json"
    freeze = {
        "schema_version": FREEZE_RESULT_SCHEMA_VERSION,
        "status": "oracle_architecture_and_applicability_policy_frozen",
        "selected_model": dict(selected),
        "inputs": {
            "classical": {
                "result": {
                    "path": selected_result_path.name,
                    "sha256": hashlib.sha256(selected_result_path.read_bytes()).hexdigest(),
                    "bytes": selected_result_path.stat().st_size,
                    "schema_version": selected_result["schema_version"],
                    "status": selected_result["status"],
                }
            }
        },
        "decision": {
            "one_representation_and_model_frozen_across_endpoints": True,
            "production_refit_required": True,
        },
    }
    freeze_path.write_text(json.dumps(freeze))
    config_path = tmp_path / "config.json"
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "seed": 1729,
        "inputs": {
            "freeze_result": freeze_path.name,
            "classical_config": source_config_path.name,
            "supervised_graph_config": "supervised.json",
            "label_free_r0_transfer_config": "transfer.json",
        },
        "chemistry": {
            "reaction_id": "ugi_3cr_agile",
            "qualified_reaction_registry": {
                "path": registry_path.name,
                "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            },
            "reaction_variant": {
                "path": variant_path.name,
                "sha256": hashlib.sha256(variant_path.read_bytes()).hexdigest(),
            },
            "maximum_forward_products": 128,
            "identity": "canonical_constitutional_smiles",
        },
        "training": {
            "endpoints": ["expt_Hela", "expt_Raw"],
            "ensemble_seeds": [1729, 11729, 21729],
            "records": "all_1100_reconciled_single_structure_records",
            "guided_generation_labels_used": False,
            "virtual_candidate_labels_used": False,
        },
        "uncertainty": {
            "coverages": [0.8, 0.9, 0.95],
            "guidance_score": ("ensemble_mean_minus_max_held_domain_empirical_residual_radius"),
            "raw_mean_maximization_allowed": False,
            "interpretation": (
                "conservative_empirical_held_domain_radius_not_a_finite_sample_"
                "coverage_guarantee_and_not_in_vivo_confidence"
            ),
        },
        "checkpoint": {
            "format": "torch_save_trusted_local_bundle",
            "path": path.name,
            "result_path": "production.json",
            "require_hash_before_inference": True,
            "stereochemistry_used": False,
            "graph_truncation_allowed": False,
        },
    }
    config_path.write_text(json.dumps(config))
    freeze_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    expected_applicability = _build_applicability_index(
        source_config_path,
        tmp_path,
    )
    expected_chemistry = _build_chemistry_contract(config, tmp_path)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "selected_model": selected,
            "freeze_result_sha256": freeze_hash,
            "production_config_sha256": config_hash,
            "data_policy": {
                "records": "all_1100_reconciled_single_structure_records",
                "virtual_candidate_labels_used": False,
                "guided_generation_labels_used": False,
                "stereochemistry_used": False,
                "graph_truncation_allowed": False,
            },
            "applicability_index": expected_applicability,
            "chemistry": expected_chemistry,
            "software": {
                "rdkit": rdBase.rdkitVersion,
                "scikit_learn": sklearn.__version__,
                "torch": str(torch.__version__),
            },
            "model": {
                "lane": "classical",
                "representation": "morgan_count",
                "model": "ridge",
                "trusted_local_pickle": b"serialized",
            },
        },
        path,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result_path = tmp_path / "production.json"
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "production_oracle_checkpoint_frozen",
        "configuration": {
            "path": config_path.name,
            "sha256": config_hash,
            "bytes": config_path.stat().st_size,
        },
        "freeze_result": {
            "path": freeze_path.name,
            "sha256": freeze_hash,
            "bytes": freeze_path.stat().st_size,
        },
        "selected_model": selected,
        "checkpoint": {
            "path": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "format": "torch_save_trusted_local_bundle",
            "hash_required_before_inference": True,
        },
        "decision": {"production_checkpoint_frozen": True},
    }
    result_path.write_text(json.dumps(result))
    loaded = load_production_checkpoint(result_path, repo_root=tmp_path)
    assert loaded["model"]["trusted_local_pickle"] == b"serialized"
    assert (
        loaded["_runtime_production_result_sha256"]
        == hashlib.sha256(result_path.read_bytes()).hexdigest()
    )
    original_freeze = freeze_path.read_bytes()
    freeze["selected_model"]["model"] = "rf"
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(OracleProductionError, match="frozen artifact record"):
        load_production_checkpoint(result_path, repo_root=tmp_path)
    freeze_path.write_bytes(original_freeze)

    original_checkpoint = path.read_bytes()
    altered_checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    altered_checkpoint["applicability_index"]["source"]["sha256"] = "f" * 64
    torch.save(altered_checkpoint, path)
    result["checkpoint"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["checkpoint"]["bytes"] = path.stat().st_size
    result_path.write_text(json.dumps(result))
    with pytest.raises(
        OracleProductionError,
        match="production applicability or chemistry contract changed",
    ):
        load_production_checkpoint(result_path, repo_root=tmp_path)

    path.write_bytes(original_checkpoint)
    result["checkpoint"]["bytes"] = path.stat().st_size
    result["checkpoint"]["sha256"] = "0" * 64
    result_path.write_text(json.dumps(result))
    with pytest.raises(OracleProductionError, match="hash mismatch"):
        load_production_checkpoint(result_path, repo_root=tmp_path)


def _verified_classical_checkpoint() -> dict:
    source_config = json.loads(Path("configs/bio/m0_07_oracle_classical.json").read_text())
    frame = pd.DataFrame(
        {
            "label": ["first", "second"],
            "model_smiles": ["CCCC", "CCO"],
        }
    )
    feature_bundle = build_feature_bundle(frame, source_config["representations"])
    feature_names = list(feature_bundle.feature_names["morgan_count"])
    estimator = DummyRegressor(strategy="constant", constant=4.0)
    estimator.fit(
        np.zeros((2, len(feature_names))),
        np.asarray([1.0, 2.0]),
    )
    payload = pickle.dumps(estimator, protocol=5)
    member = {
        "trusted_local_pickle": payload,
        "pickle_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "_runtime_hash_verified": True,
        "model": {
            "lane": "classical",
            "representation": "morgan_count",
            "representation_config": source_config["representations"],
            "feature_names": feature_names,
            "endpoint_models": {
                "expt_Hela": {"ensemble": [member, member, member]},
            },
        },
        "uncertainty": {
            "domain_radii": {
                "expt_Hela": {
                    "unseen_isocyanide": {
                        "authorized": True,
                        "radii": {"q90": 2.0},
                    },
                    "unseen_aldehyde": {
                        "authorized": False,
                        "radii": {"q90": 5.0},
                    },
                }
            }
        },
    }


def test_verified_classical_inference_and_domain_abstention() -> None:
    checkpoint = _verified_classical_checkpoint()
    prediction = predict_production_smiles(
        checkpoint,
        ["CCCC", "CCO"],
        endpoint="expt_Hela",
    )
    assert prediction["ensemble_mean"] == pytest.approx([4.0, 4.0])
    guided = _conservative_domain_scores(
        checkpoint,
        prediction,
        domain="unseen_isocyanide",
    )
    assert guided["scores"] == pytest.approx([2.0, 2.0])
    assert guided["action"] == "empirical_held_domain_radius_adjusted_guidance"
    abstained = _conservative_domain_scores(
        checkpoint,
        prediction,
        domain="unseen_aldehyde",
    )
    assert abstained["action"] == "abstain"
    assert abstained["scores"] is None

    unverified = dict(checkpoint)
    unverified.pop("_runtime_hash_verified")
    with pytest.raises(OracleProductionError, match="expected hash"):
        _conservative_domain_scores(
            unverified,
            prediction,
            domain="unseen_isocyanide",
        )


def test_candidate_domain_is_inferred_after_exact_ugi_reconstruction() -> None:
    checkpoint = _verified_classical_checkpoint()
    checkpoint["applicability_index"] = _build_applicability_index(
        Path("configs/bio/m0_07_oracle_classical.json"),
        Path("."),
    )
    production_config = json.loads(Path("configs/bio/m0_07_oracle_production.json").read_text())
    checkpoint["chemistry"] = _build_chemistry_contract(
        production_config,
        Path("."),
    )
    frame = pd.read_csv("results/m0_07/agile_oracle_curated.csv.gz")
    row = frame.iloc[0]
    candidate = {
        "label": str(row["label"]),
        "product_smiles": str(row["model_smiles"]),
        "amine_smiles": str(row["A_smiles"]),
        "aldehyde_smiles": str(row["B_smiles"]),
        "isocyanide_smiles": str(row["C_smiles"]),
    }
    classification = classify_production_ugi_candidate(checkpoint, candidate)
    assert classification["exact_forward_verified"]
    assert classification["domain"] == "known_components_novel_combination"
    assert classification["combination_seen_in_measured_training"]

    caller_misclassified = dict(candidate)
    caller_misclassified["domain"] = "known_components_novel_combination"
    checkpoint["applicability_index"]["components"] = {
        "amine": [],
        "aldehyde": [],
        "isocyanide": [],
    }
    derived = classify_production_ugi_candidate(checkpoint, caller_misclassified)
    assert derived["domain"] == "three_unseen_components"
    assert derived["unseen_component_roles"] == [
        "amine",
        "aldehyde",
        "isocyanide",
    ]

    incompatible = dict(candidate)
    incompatible["product_smiles"] = "CC"
    rejected = classify_production_ugi_candidate(checkpoint, incompatible)
    assert not rejected["exact_forward_verified"]
    assert rejected["domain"] == "non_ugi_final_assembly"


def _small_graph_vocabulary() -> GraphFeatureVocabulary:
    return GraphFeatureVocabulary(
        elements=("C", "N", "O"),
        formal_charges=("-1", "0", "1"),
        total_degrees=("1", "2", "3", "4"),
        total_hydrogens=("0", "1", "2", "3", "4"),
        hybridizations=("SP", "SP2", "SP3"),
        bond_types=("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"),
    )


def _small_graph_record(vocabulary: GraphFeatureVocabulary) -> OracleGraphRecord:
    return OracleGraphRecord(
        label="candidate-1",
        product=tensorize_smiles("CC(=O)NC", vocabulary, label="product"),
        amine=tensorize_smiles("CN", vocabulary, label="amine"),
        aldehyde=tensorize_smiles("CCC=O", vocabulary, label="aldehyde"),
        isocyanide=tensorize_smiles("[C-]#[N+]", vocabulary, label="isocyanide"),
        targets=torch.zeros(2, dtype=torch.float32),
    )


def _small_ugi_candidate() -> dict[str, str]:
    return {
        "label": "candidate-1",
        "product_smiles": "CC(=O)NC",
        "amine_smiles": "CN",
        "aldehyde_smiles": "CCC=O",
        "isocyanide_smiles": "[C-]#[N+]",
    }


def test_verified_supervised_graph_inference_reconstructs_selected_architecture() -> None:
    vocabulary = _small_graph_vocabulary()
    model_config = {
        "hidden_dim": 16,
        "depth": 2,
        "dropout": 0.0,
        "maximum_graphs_per_batch": 4,
        "maximum_atoms_per_batch": 128,
        "maximum_directed_edges_per_batch": 256,
    }
    members = []
    for fit_seed in (11, 12, 13):
        model = graph_matrix._new_model(
            "ugi_component_role_aware_dmpnn",
            vocabulary,
            model_config,
            fit_seed,
        )
        members.append({"fit_seed": fit_seed, "state_dict": model.state_dict()})
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "_runtime_hash_verified": True,
        "model": {
            "lane": "supervised_graph",
            "architecture": "ugi_component_role_aware_dmpnn",
            "model_config": model_config,
            "feature_vocabulary": vocabulary.to_dict(),
            "role_aware": True,
            "endpoint_models": {
                "expt_Hela": {
                    "target_mean": 3.0,
                    "target_scale": 2.0,
                    "ensemble": members,
                }
            },
        },
    }
    prediction = predict_production_ugi_smiles(
        checkpoint,
        [_small_ugi_candidate()],
        endpoint="expt_Hela",
    )
    assert prediction["records"] == 1
    assert np.isfinite(prediction["ensemble_mean"]).all()
    assert np.isfinite(prediction["ensemble_standard_deviation"]).all()


def test_verified_transfer_graph_inference_reconstructs_pretrained_encoder() -> None:
    vocabulary = _small_graph_vocabulary()
    architecture = {"hidden_dim": 16, "depth": 2, "dropout": 0.0}
    model_config = {
        "maximum_graphs_per_batch": 4,
        "maximum_atoms_per_batch": 128,
        "maximum_directed_edges_per_batch": 256,
    }
    members = []
    for fit_seed in (21, 22, 23):
        torch.manual_seed(fit_seed)
        encoder = DMPNNEncoder(
            vocabulary.atom_feature_dim,
            vocabulary.bond_feature_dim,
            architecture["hidden_dim"],
            architecture["depth"],
            architecture["dropout"],
        )
        model = WholeGraphRegressor(encoder, architecture["hidden_dim"], outputs=1)
        members.append({"fit_seed": fit_seed, "state_dict": model.state_dict()})
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "_runtime_hash_verified": True,
        "model": {
            "lane": "label_free_r0_transfer",
            "encoder_architecture": architecture,
            "model_config": model_config,
            "feature_vocabulary": vocabulary.to_dict(),
            "endpoint_models": {
                "expt_Hela": {
                    "target_mean": 3.0,
                    "target_scale": 2.0,
                    "ensemble": members,
                }
            },
        },
    }
    prediction = predict_production_graph_records(
        checkpoint,
        [_small_graph_record(vocabulary)],
        endpoint="expt_Hela",
    )
    assert prediction["records"] == 1
    assert np.isfinite(prediction["ensemble_mean"]).all()
    assert np.isfinite(prediction["ensemble_standard_deviation"]).all()
