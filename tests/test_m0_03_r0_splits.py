from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from forge.data.r0_splits import (
    FOLDS,
    SCHEMES,
    SplitError,
    _assign_group_folds,
    _connected_group_ids,
    _source_study_tokens,
    load_frozen_r0_splits,
)

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "data/splits/m0_03_constitutional"


def test_overlapping_annotations_are_kept_in_one_group_and_fold() -> None:
    token_sets = [
        {"study:a", "exact:1"},
        {"study:a", "study:b", "exact:2"},
        {"study:b", "exact:3"},
        {"study:c", "exact:4"},
        {"study:d", "exact:5"},
    ]
    group_ids, group_sizes = _connected_group_ids(token_sets, "study")
    folds = _assign_group_folds(
        group_sizes,
        {"R0_train": 0.7, "R0_cal": 0.15, "R0_heldout": 0.15},
        "test",
    )

    assert group_ids[0] == group_ids[1] == group_ids[2]
    assert len(set(group_ids)) == 3
    assert {folds[group_id] for group_id in group_ids} == set(FOLDS)


def test_group_assignment_is_deterministic_and_never_splits_a_group() -> None:
    sizes = {f"group-{index}": size for index, size in enumerate([8, 5, 4, 3, 2, 1])}
    fractions = {"R0_train": 0.7, "R0_cal": 0.15, "R0_heldout": 0.15}

    first = _assign_group_folds(sizes, fractions, "fixed-salt")
    second = _assign_group_folds(dict(reversed(list(sizes.items()))), fractions, "fixed-salt")

    assert first == second
    assert set(first) == set(sizes)
    assert set(first.values()) == set(FOLDS)


def test_source_without_study_id_gets_source_level_fallback() -> None:
    row = {
        "r0_structure_id": "R0-test",
        "observed_source_ids": "source_with_study|source_without_study",
        "study_split_groups_json": json.dumps({"source_with_study": ["study-1"]}),
    }

    assert _source_study_tokens(row) == {
        "study:source_with_study:study-1",
        "study:fallback-source:source_without_study",
    }


def test_frozen_m0_03_bundle_is_complete_and_leak_free() -> None:
    frozen = load_frozen_r0_splits(FROZEN)

    assert len(frozen.assignments) == 15_229
    assert frozen.manifest["all_groups_leak_free"] is True
    for scheme in SCHEMES:
        fold_ids = {fold: set(frozen.ids(scheme, fold)) for fold in FOLDS}
        assert all(fold_ids.values())
        assert not (fold_ids["R0_train"] & fold_ids["R0_cal"])
        assert not (fold_ids["R0_train"] & fold_ids["R0_heldout"])
        assert not (fold_ids["R0_cal"] & fold_ids["R0_heldout"])
        assert len(set.union(*fold_ids.values())) == 15_229


def test_loader_rejects_tampered_assignment(tmp_path: Path) -> None:
    manifest = json.loads((FROZEN / "manifest.json").read_text())
    rows = []
    with (FROZEN / "r0_fold_assignments.csv").open(newline="") as handle:
        rows.extend(csv.DictReader(handle))
    output = tmp_path / "m0_03"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with (output / "r0_fold_assignments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
        handle.write("tampered")

    with pytest.raises(SplitError, match="hash mismatch for frozen"):
        load_frozen_r0_splits(output)
