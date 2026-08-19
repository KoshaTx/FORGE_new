from __future__ import annotations

import json
from pathlib import Path

from forge.data.r0_splits import sha256_file
from forge.product.defog_feasibility import AtomState
from forge.product.sparse_topology_feasibility import (
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
    tensorize_sparse_row,
)

REPO = Path(__file__).resolve().parents[1]


def _resolve(record: dict[str, str]) -> Path:
    path = REPO / record["path"]
    assert path.is_file()
    assert sha256_file(path) == record["sha256"]
    return path


def test_cuda_preflight_uses_pinned_production_support_and_ring_examples() -> None:
    config = json.loads((REPO / "configs/model/phase1_product_cuda_preflight.json").read_text())
    assert config["schema_version"] == "phase1_product_cuda_preflight_config.v1"
    product = json.loads(_resolve(config["product_config"]).read_text())
    vocabulary_source = json.loads(_resolve(config["atom_vocabulary_source"]).read_text())
    records = sorted(vocabulary_source["atom_vocabulary"], key=lambda row: int(row["index"]))
    assert [int(row["index"]) for row in records] == list(range(len(records)))
    vocabulary = tuple(
        AtomState(
            str(row["symbol"]),
            int(row["formal_charge"]),
            bool(row["aromatic"]),
        )
        for row in records
    )
    assert len(vocabulary) == 9
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}

    closures = 0
    for sample in config["samples"]:
        sparse = tensorize_sparse_row(
            {
                "r0_structure_id": sample["structure_id"],
                "canonical_isomeric_smiles": sample["canonical_isomeric_smiles"],
            },
            atom_to_index,
        )
        assert sparse.node_count <= int(product["model"]["maximum_heavy_atoms"])
        assert sparse.closure_count <= int(product["model"]["maximum_closure_slots"])
        assert sparse_roundtrip_exact(sparse)
        assert sparse_constitutional_roundtrip_exact(sparse, vocabulary)
        closures += sparse.closure_count

    assert closures >= int(config["acceptance"]["minimum_total_closures"])
    assert product["execution"] == {
        "device": "cuda",
        "precision": "float32",
        "mixed_precision": False,
        "compile": False,
        "deterministic_algorithms": True,
        "num_workers": 0,
    }


def _assert_sha256(value: str) -> None:
    assert len(value) == 64
    int(value, 16)


def test_historical_cuda_preflight_result_remains_self_describing() -> None:
    result = json.loads((REPO / "results/phase1/product_cuda_preflight.json").read_text())
    assert result["schema_version"] == "phase1_product_cuda_preflight_result.v1"
    assert result["status"] == "pass"
    assert result["decision"]["cuda_training_path_authorized"] is True

    inputs = result["inputs"]
    for key in ("preflight_config", "product_config", "atom_vocabulary_source"):
        record = inputs[key]
        assert sha256_file(REPO / record["path"]) == record["sha256"]
    _assert_sha256(inputs["runner"]["sha256"])
    for path, expected in inputs["mounted_sources"].items():
        assert (REPO / path).is_file()
        _assert_sha256(expected)

    measured = result["result"]
    assert measured["status"] == "pass"
    assert measured["device"]["name"] == "NVIDIA L4"
    assert measured["runtime"]["deterministic_algorithms"] is True
    assert measured["runtime"]["precision"] == "float32"
    assert measured["runtime"]["mixed_precision"] is False
    assert measured["determinism"] == {
        "exact_loss_history": True,
        "exact_model_state": True,
        "exact_resume_loss_history": True,
        "exact_resume_model_state": True,
    }
    assert len(measured["repetitions"]) == 2
    assert measured["total_closures"] >= result["acceptance"]["minimum_total_closures"]
    assert (
        measured["resumed_run"]["peak_memory_allocated_bytes"]
        < measured["device"]["total_memory_bytes"]
    )


def test_historical_lipid_context_cuda_preflight_result_remains_self_describing() -> None:
    result = json.loads((REPO / "results/phase1/product_cuda_preflight_v2.json").read_text())
    assert result["schema_version"] == "phase1_product_cuda_preflight_result.v1"
    assert result["status"] == "pass"
    assert result["decision"]["cuda_training_path_authorized"] is True
    assert result["task"] == "Phase 1 lipid-context sparse whole-lipid CUDA preflight"

    inputs = result["inputs"]
    for key in ("preflight_config", "product_config", "atom_vocabulary_source"):
        record = inputs[key]
        assert sha256_file(REPO / record["path"]) == record["sha256"]
    _assert_sha256(inputs["runner"]["sha256"])
    for path, expected in inputs["mounted_sources"].items():
        assert (REPO / path).is_file()
        _assert_sha256(expected)

    preflight = json.loads((REPO / inputs["preflight_config"]["path"]).read_text())
    product = json.loads((REPO / inputs["product_config"]["path"]).read_text())
    assert preflight["product_config"] == inputs["product_config"]
    assert product["schema_version"] == "phase1_product_pretrain_config.v2"
    assert product["model"]["topology_context"] is True
    assert product["model"]["parent_distance_buckets"] == 16

    measured = result["result"]
    assert measured["status"] == "pass"
    assert measured["device"]["name"] == "NVIDIA L4"
    assert measured["runtime"]["deterministic_algorithms"] is True
    assert measured["runtime"]["precision"] == "float32"
    assert measured["runtime"]["mixed_precision"] is False
    assert measured["determinism"] == {
        "exact_loss_history": True,
        "exact_model_state": True,
        "exact_resume_loss_history": True,
        "exact_resume_model_state": True,
    }
    assert len(measured["repetitions"]) == 2
    assert measured["total_closures"] >= result["acceptance"]["minimum_total_closures"]
    assert (
        measured["resumed_run"]["peak_memory_allocated_bytes"]
        < measured["device"]["total_memory_bytes"]
    )


def test_v3_cuda_preflight_freezes_lipid_native_topology_contract() -> None:
    config_path = REPO / "configs/model/phase1_product_cuda_preflight_v3.json"
    config = json.loads(config_path.read_text())
    product_path = REPO / config["product_config"]["path"]
    vocabulary_path = REPO / config["atom_vocabulary_source"]["path"]
    assert sha256_file(product_path) == config["product_config"]["sha256"]
    assert sha256_file(vocabulary_path) == config["atom_vocabulary_source"]["sha256"]

    product = json.loads(product_path.read_text())
    assert product["schema_version"] == "phase1_product_pretrain_config.v3"
    model = product["model"]
    assert model["region_scheme"] == "polar_structural_v2"
    assert model["degree_continuation_prior_strength"] == 1.0
    assert model["closure_ring_size_prior_strength"] == 2.0
    assert model["aromatic_cycle_sizes"] == [5, 6]


def test_v3_cuda_preflight_is_retained_but_invalidated_after_sampler_fix() -> None:
    result_path = REPO / "results/phase1/product_cuda_preflight_v3.json"
    result = json.loads(result_path.read_text())
    invalidation = json.loads(
        (REPO / "results/phase1/product_cuda_preflight_v3_invalidation.json").read_text()
    )
    assert result["schema_version"] == "phase1_product_cuda_preflight_result.v1"
    assert result["status"] == "pass"
    assert result["decision"]["cuda_training_path_authorized"] is True
    assert invalidation["status"] == "invalidated_historical_artifact"
    assert sha256_file(result_path) == invalidation["artifact"]["sha256"]

    inputs = result["inputs"]
    for key in ("preflight_config", "product_config", "atom_vocabulary_source", "runner"):
        record = inputs[key]
        assert sha256_file(REPO / record["path"]) == record["sha256"]
    changed_sources = {row["path"]: row for row in invalidation["changed_sources"]}
    assert set(changed_sources) == {
        "src/forge/product/defog_feasibility.py",
        "src/forge/product/sparse_topology_feasibility.py",
    }
    for path, expected in inputs["mounted_sources"].items():
        if path in changed_sources:
            changed = changed_sources[path]
            assert expected == changed["artifact_recorded_sha256"]
            assert sha256_file(REPO / path) == changed["current_sha256"]
            assert expected != changed["current_sha256"]
        else:
            assert sha256_file(REPO / path) == expected
    assert (
        invalidation["scientific_disposition"]["cuda_training_path_authorized_for_current_source"]
        is False
    )

    measured = result["result"]
    assert measured["status"] == "pass"
    assert measured["device"]["name"] == "NVIDIA L4"
    assert measured["determinism"] == {
        "exact_loss_history": True,
        "exact_model_state": True,
        "exact_resume_loss_history": True,
        "exact_resume_model_state": True,
    }
    assert measured["total_closures"] == 4
