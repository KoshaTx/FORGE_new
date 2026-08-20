from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from forge.design.sampling.ugi_selected_generator_implementation import (
    SELECTED_GENERATOR_SOURCE_PATHS,
    SelectedGeneratorSourceArtifact,
    UgiSelectedGeneratorImplementationError,
    build_selected_generator_implementation_qualification,
    require_selected_generator_implementation_unchanged,
)

REPO = Path(__file__).resolve().parents[1]


def test_real_productive_implementation_is_complete_stable_and_typed() -> None:
    first = build_selected_generator_implementation_qualification(REPO)
    second = build_selected_generator_implementation_qualification(REPO)

    assert first == second
    assert first.implementation_sha256 == second.implementation_sha256
    assert tuple(source.path for source in first.sources) == SELECTED_GENERATOR_SOURCE_PATHS
    assert len(first.sources) == 34
    assert "src/forge/product/phase1_tree_topology_flow.py" in SELECTED_GENERATOR_SOURCE_PATHS
    assert "src/forge/product/ugi_synthesis_guidance.py" in SELECTED_GENERATOR_SOURCE_PATHS
    assert "src/forge/data/r1_prime_audit.py" in SELECTED_GENERATOR_SOURCE_PATHS
    assert {name for name, _ in first.runtime_versions} == {
        "numpy",
        "python",
        "rdkit",
        "torch",
    }


def test_unchanged_gate_rejects_even_one_bound_source_digest_change() -> None:
    qualification = build_selected_generator_implementation_qualification(REPO)
    changed_source = SelectedGeneratorSourceArtifact(
        path=qualification.sources[0].path,
        sha256="0" * 64,
    )
    changed = replace(
        qualification,
        sources=(changed_source, *qualification.sources[1:]),
    )

    with pytest.raises(
        UgiSelectedGeneratorImplementationError,
        match="changed after qualification",
    ):
        require_selected_generator_implementation_unchanged(REPO, changed)


def test_builder_fails_closed_when_productive_source_is_missing(tmp_path: Path) -> None:
    for relative_path in SELECTED_GENERATOR_SOURCE_PATHS[:-1]:
        source = REPO / relative_path
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    with pytest.raises(
        UgiSelectedGeneratorImplementationError,
        match="generator source is missing",
    ):
        build_selected_generator_implementation_qualification(tmp_path)
