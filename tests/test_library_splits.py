from __future__ import annotations

import pytest

from forge.assembly.families import LibraryAssemblyError
from forge.corpus.library_splits import (
    FrozenIdentityFolds,
    component_partitions,
    conflicting_exposures,
    constitution_id,
    program_partition,
)

FRACTIONS = {"train": 0.7, "calibration": 0.15, "heldout": 0.15}


def test_same_constitution_in_different_roles_and_serializations_has_one_partition():
    frozen = FrozenIdentityFolds()
    molecules = ["CCO", "OCC", "CN", "NCC"]
    a = component_partitions(molecules, frozen, seed=4, fractions=FRACTIONS)
    b = component_partitions(reversed(molecules), frozen, seed=4, fractions=FRACTIONS)
    assert a == b and len(a) == 3
    assert constitution_id("[CH3:1][C@H:2](O)CC") == constitution_id("CCC(O)C")


def test_frozen_holdout_wins_without_rewriting_historical_assignments():
    identity = constitution_id("CCO")
    frozen = FrozenIdentityFolds()
    for fold in ("R0_train", "val", "test"):
        frozen.add(identity, fold, fold)
    assert frozen.effective(identity) == "heldout"
    assert frozen.folds[identity] == {"train", "calibration", "heldout"}
    assert component_partitions(["CCO"], frozen, seed=0, fractions=FRACTIONS)[identity] == "heldout"


def test_mixed_components_and_protected_intermediates_are_quarantined():
    a, b, p, middle = map(constitution_id, ["CCO", "CN", "CCN", "CCCN"])
    frozen = FrozenIdentityFolds()
    partitions = {a: "train", b: "heldout"}
    assert program_partition([a, b], p, [], partitions, frozen)[0] == "quarantine"
    partitions[b] = "train"
    frozen.add(middle, "calibration", "old split")
    assert program_partition([a, b], p, [middle], partitions, frozen) == (
        "quarantine",
        ("protected_product_or_intermediate",),
    )
    assert program_partition([a, b], p, [], partitions, frozen) == ("train", ())
    partitions[p] = "heldout"
    assert program_partition([a, b], p, [], partitions, frozen)[0] == "quarantine"


def test_all_conflicting_product_groups_quarantined_including_shared_intermediate():
    assert conflicting_exposures(
        [
            ("p1", "train", ["a", "intermediate"]),
            ("p2", "heldout", ["b", "intermediate"]),
            ("p3", "calibration", ["c"]),
            ("p4", "quarantine", ["c"]),
        ]
    ) == {"p1", "p2"}


def test_invalid_metadata_fails_closed():
    frozen = FrozenIdentityFolds()
    with pytest.raises(LibraryAssemblyError):
        frozen.add("short-id", "train", "fixture")
    with pytest.raises(LibraryAssemblyError):
        frozen.add(constitution_id("CC"), "unknown", "fixture")
    with pytest.raises(LibraryAssemblyError):
        program_partition(["missing"], "p", [], {}, frozen)
    with pytest.raises(LibraryAssemblyError):
        component_partitions(["CC"], frozen, seed=1, fractions={"train": 1.0})
