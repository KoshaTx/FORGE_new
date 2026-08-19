from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from forge.route.graph2edits_runtime_qualification import (
    Graph2EditsRuntimeQualificationError,
    parse_hash_lock,
    verify_runtime_receipt,
)

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "configs/route/graph2edits_runtime_py311_macos_arm64.lock.txt"
RECEIPT = REPO / "configs/route/graph2edits_runtime_qualification_macos_arm64_py311_v1.json"


def _receipt() -> dict:
    return json.loads(RECEIPT.read_text())


def test_runtime_lock_preserves_compatibility_pins() -> None:
    versions = parse_hash_lock(LOCK)
    assert versions["torch"] == "2.2.2"
    assert versions["numpy"] == "1.26.4"
    assert versions["rdkit"] == "2024.3.6"
    assert versions["omegaconf"] == "2.3.0"
    assert versions["antlr4-python3-runtime"] == "4.9.3"


def test_frozen_runtime_receipt_is_inactive_nonbenchmark_and_self_authenticating() -> None:
    receipt = _receipt()
    verify_runtime_receipt(receipt, verify_artifacts=False, verify_environment=False)
    assert receipt["activation_authorized"] is False
    assert receipt["production_promotion_authorized"] is False
    assert receipt["scope_guards"]["frozen_120_target_benchmark_executed"] is False
    assert receipt["scope_guards"]["sealed_holdouts_accessed"] is False
    assert receipt["checkpoint_load"]["passed"] is True
    assert receipt["smoke_inference"]["repeat_count"] == 2
    assert receipt["smoke_inference"]["num_results_requested"] == 3
    assert receipt["platform"]["linux_gpu_qualified"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda receipt: receipt.update({"activation_authorized": True}),
        lambda receipt: receipt["scope_guards"].update({"sealed_holdouts_accessed": True}),
        lambda receipt: receipt["smoke_inference"].update({"target_smiles": "benchmark"}),
    ),
)
def test_runtime_receipt_mutations_fail_closed(mutation) -> None:
    receipt = deepcopy(_receipt())
    mutation(receipt)
    with pytest.raises(Graph2EditsRuntimeQualificationError):
        verify_runtime_receipt(receipt, verify_artifacts=False, verify_environment=False)
