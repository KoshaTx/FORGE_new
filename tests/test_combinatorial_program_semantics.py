from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest

from experiments.phase1.multireaction.combinatorial_overfit_gate import (
    run_combinatorial_overfit_gate,
)
from experiments.phase1.multireaction.combinatorial_training_pilot import (
    run_combinatorial_training_pilot,
)
from experiments.phase1.multireaction.combinatorial_training_smoke import (
    run_combinatorial_training_smoke,
)
from forge.assembly.families import constitutional_molecule
from forge.core.hashing import PinError, sha256_file
from forge.corpus.combinatorial_program_cache import build_combinatorial_program_cache
from forge.corpus.combinatorial_program_semantics import (
    CombinatorialProgramSemanticError,
    qualify_combinatorial_program_semantics,
)
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache

REPO = Path(__file__).resolve().parents[1]
FAMILY = "amide_coupling_acid_amine"


def _write_gzip(path: Path, text: str) -> None:
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as handle,
    ):
        handle.write(text)


@pytest.fixture
def semantic_repo(tmp_path):
    shipped = json.loads(
        (REPO / "configs/multireaction/combinatorial_program_semantics_v1.json").read_text()
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    renamed = {
        "atom_vocabulary": "atoms.json",
        "ugi_registry": "ugi.json",
        "family_registry": "families.json",
        "ugi_variant": "variant.yaml",
    }
    pins = {}
    for name, filename in renamed.items():
        source = REPO / shipped["inputs"][name]["path"]
        destination = inputs / filename
        shutil.copyfile(source, destination)
        pins[name] = {
            "path": str(destination.relative_to(tmp_path)),
            "sha256": str(sha256_file(destination)),
        }

    registry = json.loads((inputs / "families.json").read_text())
    reaction = next(row for row in registry["reactions"] if row["reaction_id"] == FAMILY)
    (inputs / "families.json").write_text(json.dumps({**registry, "reactions": [reaction]}))
    pins["family_registry"]["sha256"] = str(sha256_file(inputs / "families.json"))
    example = reaction["known_positive_examples"][0]
    product = constitutional_molecule(example["expected"])[0]
    product_id = hashlib.sha256(product.encode()).hexdigest()
    components = dict(
        zip(
            [row["name"] for row in reaction["reactant_roles"]],
            example["reactants"],
            strict=True,
        )
    )

    products_path = inputs / "products.csv.gz"
    fields = [
        "product_id",
        "canonical_smiles",
        "reaction_family",
        "fold",
        "representative_program_group",
        "unique_policy_qualified_program_available",
        "mean_realism_weight",
        "training_sampling_weight",
    ]
    rows = io.StringIO()
    writer = csv.DictWriter(rows, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow(
        {
            "product_id": product_id,
            "canonical_smiles": product,
            "reaction_family": FAMILY,
            "fold": "train",
            "representative_program_group": "group-1",
            "unique_policy_qualified_program_available": "1",
            "mean_realism_weight": "0.5",
            "training_sampling_weight": "1",
        }
    )
    _write_gzip(products_path, rows.getvalue())

    programs_path = inputs / "programs.jsonl.gz"
    group = {
        "group_id": "group-1",
        "reaction_id": FAMILY,
        "components": components,
        "accumulator_role": None,
        "targets": [
            {
                "product_smiles": product,
                "minimum_steps": 1,
                "policy_intermediate_products": [product],
                "policy_path_count_capped_at_two": 1,
                "search_complete_through_minimum_depth": True,
            }
        ],
    }
    _write_gzip(programs_path, json.dumps(group, sort_keys=True) + "\n")
    pins["products"] = {
        "path": str(products_path.relative_to(tmp_path)),
        "sha256": str(sha256_file(products_path)),
    }
    pins["programs"] = {
        "path": str(programs_path.relative_to(tmp_path)),
        "sha256": str(sha256_file(programs_path)),
    }
    dataset = {
        "schema_version": "forge.combinatorial_program_dataset.v1",
        "artifacts": {
            "products.csv.gz": pins["products"],
            "programs.jsonl.gz": pins["programs"],
        },
    }
    dataset_path = inputs / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    pins["program_dataset_result"] = {
        "path": str(dataset_path.relative_to(tmp_path)),
        "sha256": str(sha256_file(dataset_path)),
    }

    config = {
        "schema_version": shipped["schema_version"],
        "workers": 1,
        "maximum_outcomes": shipped["maximum_outcomes"],
        "required_folds": ["train"],
        "inputs": pins,
        "registries": ["family_registry"],
        "expected_selected_counts": {FAMILY: {"train": 1, "calibration": 0, "heldout": 0}},
        "programs": {FAMILY: shipped["programs"][FAMILY]},
        "support_bounds": shipped["support_bounds"],
        "policy": shipped["policy"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path, config_path


def test_exact_semantics_publish_deterministically_through_shared_tensor_contract(semantic_repo):
    repo, config = semantic_repo
    first = qualify_combinatorial_program_semantics(repo, config, Path("first"))
    second = qualify_combinatorial_program_semantics(repo, config, Path("second"))
    assert first["status"] == "pass"
    assert first["selected_fold_counts"][FAMILY]["train"] == 1
    assert first["semantic_fold_counts"][FAMILY]["train"] == 1
    assert first["representation"]["records"] == 1
    assert first["representation"]["abstentions"] == 0
    assert all(first["gates"].values())
    assert first["training_calls"] == first["generator_sampling_calls"] == 0
    assert first["representation"]["sha256"] == second["representation"]["sha256"]
    assert (
        first["artifacts"]["semantic_records.jsonl.gz"]["sha256"]
        == second["artifacts"]["semantic_records.jsonl.gz"]["sha256"]
    )
    with gzip.open(repo / first["artifacts"]["semantic_records.jsonl.gz"]["path"], "rt") as f:
        record = json.loads(f.read())
    assert set(record["origin_roles"]) == {
        "carboxylic_acid_tail",
        "amine_head",
    }
    assert set(filter(None, record["core_positions"])) == {"map_1", "map_2", "map_4"}
    assert record["fixed_atom_indices"] == []
    assert "ugi_source_mapping_multiplicity" not in record


@pytest.mark.parametrize("defect", ["pin", "policy", "folds", "population"])
def test_tampered_semantic_contract_never_publishes(semantic_repo, defect):
    repo, config_path = semantic_repo
    config = json.loads(config_path.read_text())
    if defect == "pin":
        config["inputs"]["products"]["sha256"] = "0" * 64
    elif defect == "policy":
        config["policy"]["raw_family_count_sampling"] = True
    elif defect == "folds":
        config["required_folds"] = ["train", "quarantine"]
    else:
        config["expected_selected_counts"][FAMILY]["train"] = 2
    config_path.write_text(json.dumps(config))
    with pytest.raises((CombinatorialProgramSemanticError, PinError)):
        qualify_combinatorial_program_semantics(repo, config_path, Path("bad"))
    assert not (repo / "bad").exists()


def test_admitted_semantics_pack_into_deterministic_shared_cache(semantic_repo):
    repo, semantic_config = semantic_repo
    semantic = qualify_combinatorial_program_semantics(repo, semantic_config, Path("semantic"))
    semantic_result = repo / "semantic/result.json"
    semantic_records = repo / semantic["artifacts"]["semantic_records.jsonl.gz"]["path"]
    source_config = json.loads(semantic_config.read_text())
    cache_config = {
        "schema_version": "forge.combinatorial_program_cache_config.v1",
        "required_folds": ["train"],
        "inputs": {
            "semantic_result": {
                "path": str(semantic_result.relative_to(repo)),
                "sha256": str(sha256_file(semantic_result)),
            },
            "semantic_records": {
                "path": str(semantic_records.relative_to(repo)),
                "sha256": str(sha256_file(semantic_records)),
            },
            "semantic_config": {
                "path": str(semantic_config.relative_to(repo)),
                "sha256": str(sha256_file(semantic_config)),
            },
            "atom_vocabulary": source_config["inputs"]["atom_vocabulary"],
        },
        "expected_fold_counts": semantic["semantic_fold_counts"],
        "support_bounds": source_config["support_bounds"],
        "weight_policy": {
            "within_family": "mean_realism_weight",
            "between_families": "equal_mass",
            "semantic_abstentions": "exclude_then_renormalize_within_family",
            "raw_family_count_sampling": False,
        },
    }
    path = repo / "cache.json"
    path.write_text(json.dumps(cache_config))
    first = build_combinatorial_program_cache(repo, path, Path("first-cache"))
    second = build_combinatorial_program_cache(repo, path, Path("second-cache"))
    assert first["status"] == "pass"
    assert first["records"] == 1
    assert all(first["gates"].values())
    assert first["artifacts"]["cache.npz"]["sha256"] == second["artifacts"]["cache.npz"]["sha256"]
    with SynthesisProgramProductionCache(repo / first["artifacts"]["cache.npz"]["path"]) as cache:
        assert cache.fold_counts() == semantic["semantic_fold_counts"]
        assert cache.program_id(0) == FAMILY
        assert cache.training_measure({FAMILY: 1.0}).tolist() == [1.0]
        assert cache.record(0).graph.canonical_smiles


def test_packed_semantics_complete_a_deterministic_local_optimizer_step(semantic_repo):
    repo, semantic_config = semantic_repo
    semantic = qualify_combinatorial_program_semantics(repo, semantic_config, Path("semantic"))
    semantic_result = repo / "semantic/result.json"
    semantic_records = repo / semantic["artifacts"]["semantic_records.jsonl.gz"]["path"]
    source_config = json.loads(semantic_config.read_text())
    cache_config = repo / "cache.json"
    cache_config.write_text(
        json.dumps(
            {
                "schema_version": "forge.combinatorial_program_cache_config.v1",
                "required_folds": ["train"],
                "inputs": {
                    "semantic_result": {
                        "path": str(semantic_result.relative_to(repo)),
                        "sha256": str(sha256_file(semantic_result)),
                    },
                    "semantic_records": {
                        "path": str(semantic_records.relative_to(repo)),
                        "sha256": str(sha256_file(semantic_records)),
                    },
                    "semantic_config": {
                        "path": str(semantic_config.relative_to(repo)),
                        "sha256": str(sha256_file(semantic_config)),
                    },
                    "atom_vocabulary": source_config["inputs"]["atom_vocabulary"],
                },
                "expected_fold_counts": semantic["semantic_fold_counts"],
                "support_bounds": source_config["support_bounds"],
                "weight_policy": {
                    "within_family": "mean_realism_weight",
                    "between_families": "equal_mass",
                    "semantic_abstentions": "exclude_then_renormalize_within_family",
                    "raw_family_count_sampling": False,
                },
            }
        )
    )
    cache = build_combinatorial_program_cache(repo, cache_config, Path("cache"))
    cache_result = repo / "cache/result.json"
    cache_path = repo / cache["artifacts"]["cache.npz"]["path"]
    smoke_config = repo / "smoke.json"
    smoke_config.write_text(
        json.dumps(
            {
                "schema_version": "forge.combinatorial_training_smoke_config.v1",
                "seed": 913,
                "expected_programs": 1,
                "record_selection": "lexicographically_first_train_record_per_program",
                "inputs": {
                    "cache_result": {
                        "path": str(cache_result.relative_to(repo)),
                        "sha256": str(sha256_file(cache_result)),
                    },
                    "cache": {
                        "path": str(cache_path.relative_to(repo)),
                        "sha256": str(sha256_file(cache_path)),
                    },
                    "cache_config": {
                        "path": str(cache_config.relative_to(repo)),
                        "sha256": str(sha256_file(cache_config)),
                    },
                },
                "model": {
                    "architecture": "reaction_program_sparse_whole_lipid_flow",
                    "hidden_dim": 16,
                    "layers": 1,
                    "maximum_closures": 1,
                    "maximum_heavy_atoms": 140,
                    "dropout": 0.0,
                    "bond_classes": 4,
                },
                "optimization": {
                    "optimizer_steps_per_replica": 1,
                    "determinism_replicas": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "flow_time": 0.5,
                    "gradient_clip_norm": 1.0,
                    "source_probability_floor": 0.00001,
                    "cpu_threads": 1,
                },
            }
        )
    )
    result = run_combinatorial_training_smoke(repo, smoke_config, Path("smoke"))
    assert result["status"] == "pass"
    assert all(result["gates"].values())
    assert result["selected_records"][0]["program_id"] == FAMILY
    assert result["optimizer_steps_executed"] == 2
    assert result["generator_sampling_calls"] == 0

    smoke_result = repo / "smoke/result.json"
    overfit_config = repo / "overfit.json"
    overfit_config.write_text(
        json.dumps(
            {
                "schema_version": "forge.combinatorial_overfit_gate_config.v1",
                "scientific_question": "Can the shared flow learn one fixed example?",
                "hypothesis": "The fixed-example loss will decrease.",
                "alternative_explanation": "A gradient can exist without useful learning.",
                "seed": 913,
                "expected_programs": 1,
                "record_selection": "lower_median_heavy_atom_count_then_record_id",
                "inputs": {
                    "cache_result": {
                        "path": str(cache_result.relative_to(repo)),
                        "sha256": str(sha256_file(cache_result)),
                    },
                    "cache": {
                        "path": str(cache_path.relative_to(repo)),
                        "sha256": str(sha256_file(cache_path)),
                    },
                    "cache_config": {
                        "path": str(cache_config.relative_to(repo)),
                        "sha256": str(sha256_file(cache_config)),
                    },
                    "training_smoke_result": {
                        "path": str(smoke_result.relative_to(repo)),
                        "sha256": str(sha256_file(smoke_result)),
                    },
                },
                "model": {
                    "architecture": "reaction_program_sparse_whole_lipid_flow",
                    "hidden_dim": 16,
                    "layers": 1,
                    "maximum_closures": 1,
                    "maximum_heavy_atoms": 140,
                    "dropout": 0.0,
                    "bond_classes": 4,
                },
                "optimization": {
                    "steps": 24,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "flow_time": 0.5,
                    "gradient_clip_norm": 1.0,
                    "source_probability_floor": 0.00001,
                    "cpu_threads": 1,
                    "corruption_policy": "one_fixed_draw_reused_for_learning_gate",
                    "family_loss_policy": "equal_program_mass",
                    "report_steps": [1, 24],
                },
                "acceptance": {
                    "maximum_final_to_initial_loss_ratio": 0.999,
                    "minimum_improved_programs": 1,
                    "minimum_exact_tensor_records": 0,
                },
                "candidate_selection": False,
                "generation_calls": 0,
                "remote_compute": False,
            }
        )
    )
    overfit = run_combinatorial_overfit_gate(repo, overfit_config, Path("overfit"))
    assert overfit["status"] == "pass"
    assert all(overfit["gates"].values())
    assert overfit["programs_with_improved_loss"] == [FAMILY]
    assert overfit["training"]["final_total_loss"] < overfit["training"]["initial_total_loss"]
    assert overfit["generator_sampling_calls"] == 0

    overfit_result = repo / "overfit/result.json"
    pilot_config = repo / "pilot.json"
    pilot_config.write_text(
        json.dumps(
            {
                "schema_version": "forge.combinatorial_training_pilot_config.v1",
                "scientific_question": "Does bounded training improve paired evaluation loss?",
                "hypothesis": "The evaluation loss will decrease.",
                "alternative_explanation": "One-step gradients may not transfer across noise.",
                "seed": 914,
                "expected_programs": 1,
                "expected_evaluation_records": 1,
                "inputs": {
                    "cache_result": {
                        "path": str(cache_result.relative_to(repo)),
                        "sha256": str(sha256_file(cache_result)),
                    },
                    "cache": {
                        "path": str(cache_path.relative_to(repo)),
                        "sha256": str(sha256_file(cache_path)),
                    },
                    "cache_config": {
                        "path": str(cache_config.relative_to(repo)),
                        "sha256": str(sha256_file(cache_config)),
                    },
                    "overfit_gate_result": {
                        "path": str(overfit_result.relative_to(repo)),
                        "sha256": str(sha256_file(overfit_result)),
                    },
                },
                "model": {
                    "architecture": "reaction_program_sparse_whole_lipid_flow",
                    "hidden_dim": 16,
                    "layers": 1,
                    "maximum_closures": 1,
                    "maximum_heavy_atoms": 140,
                    "dropout": 0.0,
                    "bond_classes": 4,
                },
                "training": {
                    "steps": 64,
                    "records_per_program_per_step": 1,
                    "sampling_policy": ("realism_weight_within_family_equal_mass_between_families"),
                    "family_loss_policy": "equal_program_mass",
                    "flow_time_policy": "uniform_bounded_per_record",
                    "minimum_flow_time": 0.02,
                    "maximum_flow_time": 0.98,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "gradient_clip_norm": 1.0,
                    "source_probability_floor": 0.00001,
                    "loss_window_steps": 4,
                    "report_steps": [1, 16, 64],
                    "cpu_threads": 1,
                },
                "evaluation": {
                    "fold": "train",
                    "batch_size": 1,
                    "flow_time": 0.5,
                    "corruption_policy": "fixed_paired_draws_before_and_after_training",
                    "maximum_records_per_program": 1,
                },
                "acceptance": {
                    "maximum_final_to_initial_training_loss_ratio": 0.999,
                    "maximum_final_to_initial_calibration_loss_ratio": 0.999,
                    "minimum_improved_calibration_programs": 1,
                    "require_ugi_exact_reconstruction_not_regressed": True,
                },
                "candidate_selection": False,
                "generation_calls": 0,
                "heldout_structure_access": False,
                "remote_compute": False,
            }
        )
    )
    pilot = run_combinatorial_training_pilot(repo, pilot_config, Path("pilot"))
    assert pilot["status"] == "pass"
    assert all(pilot["gates"].values())
    assert pilot["population"]["evaluation_records"] == 1
    assert pilot["calibration"]["programs_with_improved_loss"] == [FAMILY]
    assert pilot["generator_sampling_calls"] == 0
    assert pilot["heldout_structure_access"] is False
