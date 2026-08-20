from __future__ import annotations

from experiments.archive.producers.phase1_audit_ugi_branch_arm_geometry import (
    _cached_component_branch_geometry,
    component_branch_geometry,
)


def test_component_branch_geometry_cache_reuses_the_computed_object() -> None:
    cache = {}

    first = _cached_component_branch_geometry(
        cache,
        "CCCC(C)CCCCCC=O",
        "oxoester_aldehyde_body_tail",
    )
    second = _cached_component_branch_geometry(
        cache,
        "CCCC(C)CCCCCC=O",
        "oxoester_aldehyde_body_tail",
    )

    assert first is second
    assert len(cache) == 1


def test_branch_geometry_terminates_when_a_distal_arm_contains_a_ring() -> None:
    observed = component_branch_geometry(
        "O=CCCCC(C)C1CCCCC1",
        "oxoester_aldehyde_body_tail",
    )

    branch = observed["first_outward_branch"]
    assert branch is not None
    assert branch["longer_distal_arm_carbons"] >= branch["shorter_distal_arm_carbons"] >= 1
