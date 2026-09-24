from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest

from experiments.phase1.multireaction.combinatorial_graph_reuse import validate
from experiments.phase1.multireaction.combinatorial_graph_reuse_verify import verify
from experiments.phase1.multireaction.combinatorial_reuse_admission import (
    verify as verify_admission,
)
from forge.core.hashing import sha256_file
from forge.model.precursor_reuse import PrecursorReuseError

REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "results/phase1/combinatorial_graph_reuse_discovery_v1/result.json"
ADMISSION = REPO / "results/phase1/combinatorial_reuse_admission_discovery_v1/result.json"


@pytest.mark.parametrize(
    "coordinate,value",
    [("flow_steps", 1), ("flow_steps", 65), ("attempts_per_family", 257), ("batch_size", True)],
)
def test_graph_copy_budget_is_bounded(coordinate, value):
    config = json.loads(
        (REPO / "configs/multireaction/combinatorial_graph_reuse_discovery_v1.json").read_text()
    )
    validate(config)
    config["sampling"][coordinate] = value
    with pytest.raises(PrecursorReuseError):
        validate(config)


def test_graph_copy_rejects_gate_changes():
    config = json.loads(
        (REPO / "configs/multireaction/combinatorial_graph_reuse_discovery_v1.json").read_text()
    )
    config["policy"]["gate_changes"] = True
    with pytest.raises(PrecursorReuseError, match="policy"):
        validate(config)


def test_verification_rejects_false_preservation_claim(tmp_path):
    result = json.loads(PARENT.read_text())
    assert not result["all_family_preservation_screen_passed"]
    result["all_family_preservation_screen_passed"] = True
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(PrecursorReuseError, match="summary/decision"):
        verify(REPO, path)


def test_rehashed_completion_substitution_still_fails():
    result = json.loads(PARENT.read_text())
    values = [
        json.loads(s)
        for s in (REPO / result["artifacts"]["completions.jsonl"]["path"]).read_text().splitlines()
    ]
    row = next(r for r in values if r["selected_donor"] is not None)
    row["selected_donor"] = 999
    with tempfile.TemporaryDirectory(prefix=".reuse-test-", dir=REPO / "results") as temporary:
        work = Path(temporary)
        ledger = work / "completions.jsonl"
        ledger.write_text("".join(json.dumps(r) + "\n" for r in values))
        result["artifacts"]["completions.jsonl"] = {
            "path": str(ledger.relative_to(REPO)),
            "sha256": str(sha256_file(ledger)),
        }
        path = work / "result.json"
        path.write_text(json.dumps(result))
        with pytest.raises(PrecursorReuseError, match="completion/proposal ledger"):
            verify(REPO, path)


def test_admission_result_recomputes_and_rejects_false_gain(tmp_path):
    assert verify_admission(REPO, ADMISSION)["status"] == "verified"
    result = json.loads(ADMISSION.read_text())
    changed = copy.deepcopy(result)
    changed["total_exact_gain"] += 1
    path = tmp_path / "result.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(PrecursorReuseError, match="summary/decision"):
        verify_admission(REPO, path)


def test_admission_cannot_claim_unused_reference_selection(tmp_path):
    result = json.loads(ADMISSION.read_text())
    result["policy"]["uses_train_product_membership"] = False
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(PrecursorReuseError, match="contract"):
        verify_admission(REPO, path)
