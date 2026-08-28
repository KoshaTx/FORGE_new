from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from experiments._runtime.spec import ExperimentSpec
from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_calibration_adjudication import (
    _checkpoint_evidence,
    _gate_candidate,
    _mean_difference,
    _primary_difference,
)

REPO = Path(__file__).resolve().parents[1]


def _jsonl(schema: str, rows: list[dict]) -> bytes:
    payload = [json.dumps({"schema_version": schema, "rows": len(rows)})]
    payload.extend(json.dumps(row) for row in rows)
    return gzip.compress(("\n".join(payload) + "\n").encode())


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_checkpoint_evidence_reconstructs_method_blind_metrics(tmp_path: Path) -> None:
    common = [
        {
            "method_id": "candidate",
            "seed": 0,
            "attempt_index": 0,
            "valid": True,
            "canonical_smiles": "CCCN",
            "exact_l1_program": True,
            "exact_l1_trace_count": 1,
            "exact_l1_traces": [
                {
                    "components_by_role": {
                        "amine_head": "CN",
                        "oxoester_aldehyde_body_tail": "CC=O",
                        "isocyanide_tail": "[C-]#[N+]C",
                    }
                }
            ],
        },
        {
            "method_id": "candidate",
            "seed": 0,
            "attempt_index": 1,
            "valid": False,
            "canonical_smiles": None,
            "exact_l1_program": False,
            "exact_l1_trace_count": 0,
            "exact_l1_traces": [],
        },
    ]
    local = [
        {
            "method_id": "candidate",
            "seed": 0,
            "attempt_index": 0,
            "local_support_qualified_exact_l1": True,
        },
        {
            "method_id": "candidate",
            "seed": 0,
            "attempt_index": 1,
            "local_support_qualified_exact_l1": False,
        },
    ]
    morphology = [
        {
            "method_id": "candidate",
            "seed": 0,
            "attempt_index": 0,
            "trace_assessments": [
                {
                    "all_roles_within_observed_hard_bounds": True,
                    "all_role_ring_signatures_supported": True,
                    "tails_within_observed_hard_bounds": True,
                    "tail_ring_signatures_supported": True,
                }
            ],
        },
        {
            "method_id": "candidate",
            "seed": 0,
            "attempt_index": 1,
            "trace_assessments": [],
        },
    ]
    reported = {
        "valid_fraction_per_attempt": 0.5,
        "exact_l1_yield_per_attempt": 0.5,
        "local_support_qualified_exact_l1_yield_per_attempt": 0.5,
        "role_supported_exact_l1_yield_per_attempt": 0.5,
        "tail_supported_exact_l1_yield_per_attempt": 0.5,
        "component_novelty_fraction": 1.0,
        "effective_component_count": 3.0,
    }
    common_result = {
        "common_assessment": {
            "metrics": {
                "unique_open_ended_whole_product_novel_exact_l1_products_per_1000_attempts": 500.0
            }
        }
    }
    archive_path = tmp_path / "assessment.tar"
    with tarfile.open(archive_path, "w") as archive:
        prefix = "step_0300/assessment"
        _add_bytes(
            archive,
            f"{prefix}/common/assessed_attempts.jsonl.gz",
            _jsonl("forge.common_ugi_assessed_attempts.v1", common),
        )
        _add_bytes(
            archive,
            f"{prefix}/local_chemistry/assessed_attempts.jsonl.gz",
            _jsonl("forge.common_local_chemistry_attempt.v1", local),
        )
        _add_bytes(
            archive,
            f"{prefix}/role_morphology_attempts.jsonl.gz",
            _jsonl("forge.ugi_role_morphology_attempt.v1", morphology),
        )
        _add_bytes(
            archive,
            f"{prefix}/common/result.json",
            json.dumps(common_result).encode(),
        )

    evidence = _checkpoint_evidence(
        archive_path,
        arm_id="tree",
        step=300,
        programs=2,
        training_products={"CC"},
        training_components={
            "amine_head": {"N"},
            "oxoester_aldehyde_body_tail": {"CCC=O"},
            "isocyanide_tail": {"[C-]#[N+]CC"},
        },
        reported_metrics=reported,
    )
    assert evidence.metrics == pytest.approx(
        {
            **reported,
            "unique_open_ended_whole_product_novel_exact_l1_products_per_attempt": 0.5,
        }
    )


def test_paired_bootstrap_preserves_fixed_difference() -> None:
    indices = np.tile(np.arange(8), (100, 1))
    comparison = _mean_difference(
        np.ones(8, dtype=np.bool_),
        np.zeros(8, dtype=np.bool_),
        indices,
        0.95,
    )
    assert comparison == {
        "candidate": 1.0,
        "reference": 0.0,
        "difference": 1.0,
        "ci_lower": 1.0,
        "ci_upper": 1.0,
    }


def test_primary_bootstrap_counts_unique_products_per_attempt() -> None:
    indices = np.tile(np.arange(4), (100, 1))
    comparison = _primary_difference(
        ("A", "A", "B", None),
        ("A", None, None, None),
        indices,
        0.95,
    )
    assert comparison == {
        "candidate": 0.5,
        "reference": 0.25,
        "difference": 0.25,
        "ci_lower": 0.25,
        "ci_upper": 0.25,
    }


def test_gate_requires_both_retention_and_primary_improvement() -> None:
    config = json.loads(
        (
            REPO
            / "configs/model/phase1_ugi_tree_transformer_calibration_adjudication_v1.json"
        ).read_text()
    )
    gates = config["gates"]
    comparisons = {
        metric: {
            "candidate": 0.8,
            "reference": 0.8,
            "difference": 0.0,
            "ci_lower": -0.01,
            "ci_upper": 0.01,
        }
        for metric in gates["noninferiority_metrics"]
    }
    comparisons[gates["primary_metric"]] = {
        "candidate": 0.25,
        "reference": 0.20,
        "difference": 0.05,
        "ci_lower": 0.01,
        "ci_upper": 0.09,
    }
    comparisons["effective_component_count"] = {
        "candidate": 80.0,
        "reference": 100.0,
        "ratio": 0.8,
    }
    assert _gate_candidate(comparisons, gates)["status"] == "pass"
    comparisons["exact_l1_yield_per_attempt"]["ci_lower"] = -0.021
    assert _gate_candidate(comparisons, gates)["status"] == "fail"


def test_calibration_descriptors_are_exact_h100_and_arm_pinned() -> None:
    names = {
        "ugi_tree_calibration_dense_h100_v1.json": "dense_corrected_schedule",
        "ugi_tree_calibration_relations_h100_v1.json": "tree_relations_and_routing",
        "ugi_tree_calibration_consistency_h100_v1.json": "tree_plus_balanced_consistency",
        "ugi_tree_calibration_masking_h100_v1.json": "full_component_masking",
        "ugi_v0_calibration_h100_v1.json": "v0_reference",
    }
    root = REPO / "experiments/phase1/product_l1"
    for filename, arm_id in names.items():
        path = root / filename
        ExperimentSpec.load(path)
        value = json.loads(path.read_text())
        stage = value["stages"][0]
        assert value["metadata"]["arm_id"] == arm_id
        assert value["metadata"]["heldout_rows_used"] is False
        assert stage["resources"]["gpu_type"] == "H100!"
        assert stage["resources"]["device"] == "cuda"
        assert stage["resources"]["precision"] == "float32"
        assert value["profiles"] == ["full"]

    preflight = root / "ugi_tree_transformer_calibration_h100_preflight_v1.json"
    ExperimentSpec.load(preflight)
    value = json.loads(preflight.read_text())
    assert [stage["id"] for stage in value["stages"]] == ["tree", "v0"]
    assert all(stage["resources"]["gpu_type"] == "H100!" for stage in value["stages"])
    assert value["metadata"]["scientific_result"] is False
