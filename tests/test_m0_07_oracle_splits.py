from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from forge.potency.oracle.oracle_splits import (
    OracleSplitError,
    _audit_scaffold_split,
    _load_config,
    _validate_assignment_partition,
    assign_groups_to_folds,
    build_group_cv_assignments,
    build_oracle_splits,
)

REPO = Path(__file__).resolve().parents[1]


def _records() -> list[dict[str, str]]:
    return [{"label": f"A{index}B1C1"} for index in range(1, 11)]


def test_group_folds_are_deterministic_and_balanced() -> None:
    groups = [f"A{index}" for index in range(1, 11)]

    first = assign_groups_to_folds(groups, n_folds=5, seed=1729, namespace="heads")
    second = assign_groups_to_folds(groups, n_folds=5, seed=1729, namespace="heads")

    assert first == second
    counts = {fold: list(first.values()).count(fold) for fold in range(5)}
    assert counts == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}


def test_held_group_never_leaks_into_train_or_calibration() -> None:
    records = _records()
    groups = {row["label"]: row["label"].split("B", maxsplit=1)[0] for row in records}
    assignments = build_group_cv_assignments(
        records,
        scheme="held_head_5fold",
        groups=groups,
        n_folds=5,
        seed=1729,
        calibration_fraction_of_non_test=0.125,
    )

    summary = _validate_assignment_partition(assignments, {row["label"] for row in records})

    assert len(summary) == 5
    assert all(fold["test_group_leakage"] == 0 for fold in summary.values())
    assert all(fold["stage_rows"]["test"] == 2 for fold in summary.values())


def test_scaffold_audit_detects_leakage() -> None:
    scaffolds = ["s1", "s1", "s2", "s3"]
    leaking = _audit_scaffold_split(([0], [1], [2, 3]), scaffolds)
    clean = _audit_scaffold_split(([0, 1], [2], [3]), scaffolds)

    assert not leaking["scaffold_disjoint"]
    assert leaking["leaking_scaffolds"] == ["s1"]
    assert clean["scaffold_disjoint"]


def test_config_rejects_random_split_selection(tmp_path: Path) -> None:
    config = json.loads((REPO / "configs/bio/m0_07_oracle_splits.json").read_text())
    config["policy"]["model_selection_uses_random_split"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(OracleSplitError, match="model_selection_uses_random_split"):
        _load_config(path)


@pytest.mark.needs_vendor
def test_frozen_oracle_split_contract_when_present() -> None:
    result_path = REPO / "results/m0_07/oracle_split_manifest.json"
    if not result_path.exists():
        pytest.skip("M0-07 oracle split manifest has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_07_oracle_splits.v1"
    assert result["status"] == "oracle_evaluation_contract_frozen"
    assert result["summary"]["curated_records"] == 1100
    assert result["summary"]["virtual_candidate_records"] == 12276
    assert result["summary"]["component_cardinality"] == {"A": 20, "B": 11, "C": 5}
    assert not result["source_split_audits"]["lantern_murcko_source"]["scaffold_disjoint"]
    assert result["source_split_audits"]["lantern_scaffold_balanced"]["scaffold_disjoint"]
    assert not result["decision"]["oracle_model_frozen"]
    assert not result["evaluation_contract"]["model_selection_uses_random_split"]

    assignment_path = REPO / "results/m0_07/oracle_split_assignments.csv.gz"
    payload = assignment_path.read_bytes()
    metadata = result["artifacts"]["oracle_split_assignments.csv.gz"]
    assert len(payload) == metadata["bytes"]
    assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]

    with gzip.open(assignment_path, "rt") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["summary"]["assignment_rows"]
    assert {row["scheme"] for row in rows} == {
        "lantern_random",
        "lantern_scaffold_balanced",
        "held_head_5fold",
        "held_aldehyde_5fold",
        "held_isocyanide_5fold",
        "held_head_aldehyde_pair_5fold",
        "held_head_isocyanide_pair_5fold",
        "held_aldehyde_isocyanide_pair_5fold",
    }


# Manifest keys that record the machine the artifact was built on rather than
# the result itself.  Upgrading numpy or rdkit changes these and nothing else;
# folding them into a byte-exact comparison turns every dependency bump into a
# reproducibility failure and hides the failures that matter.
VOLATILE_MANIFEST_KEYS = ("software",)


def _without_build_environment(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key not in VOLATILE_MANIFEST_KEYS}


@pytest.mark.needs_vendor
def test_frozen_oracle_split_contract_reproduces_byte_exactly(tmp_path: Path) -> None:
    result_path = REPO / "results/m0_07/oracle_split_manifest.json"
    if not result_path.exists():
        pytest.skip("M0-07 oracle split manifest has not been generated")

    output_dir = tmp_path / "m0_07"
    build_oracle_splits(
        REPO / "configs/bio/m0_07_oracle_splits.json",
        output_dir,
        REPO,
    )

    # The split assignments are the scientific artifact: byte-exact, no exceptions.
    name = "oracle_split_assignments.csv.gz"
    assert (output_dir / name).read_bytes() == (REPO / "results/m0_07" / name).read_bytes()

    # The manifest must reproduce exactly apart from the recorded build environment.
    rebuilt = json.loads((output_dir / "oracle_split_manifest.json").read_text())
    stored = json.loads((REPO / "results/m0_07/oracle_split_manifest.json").read_text())
    assert _without_build_environment(rebuilt) == _without_build_environment(stored)


@pytest.mark.needs_vendor
def test_frozen_oracle_split_build_environment_is_reported(tmp_path: Path) -> None:
    """Surface environment drift without failing on it.

    A changed numpy or rdkit version does not invalidate the split, but it does
    mean the stored manifest no longer describes the current machine, so it is
    reported rather than silently accepted.
    """

    result_path = REPO / "results/m0_07/oracle_split_manifest.json"
    if not result_path.exists():
        pytest.skip("M0-07 oracle split manifest has not been generated")

    output_dir = tmp_path / "m0_07"
    build_oracle_splits(REPO / "configs/bio/m0_07_oracle_splits.json", output_dir, REPO)

    rebuilt = json.loads((output_dir / "oracle_split_manifest.json").read_text()).get(
        "software", {}
    )
    stored = json.loads(result_path.read_text()).get("software", {})
    drifted = {
        key: (stored.get(key), rebuilt.get(key))
        for key in sorted(set(stored) | set(rebuilt))
        if stored.get(key) != rebuilt.get(key)
    }
    if drifted:
        detail = "; ".join(
            f"{k}: stored {was!r} -> current {now!r}" for k, (was, now) in drifted.items()
        )
        pytest.skip(f"build environment differs from the frozen manifest ({detail})")
