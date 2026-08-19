from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.assembly import Ugi3AssemblyAdapter, Ugi3AssemblyError
from forge.core.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data/vendor/qualified_reactions_v1.json"


def _example() -> tuple[dict[str, str], str]:
    document = json.loads(REGISTRY.read_text())
    reaction = next(
        row for row in document["reactions"] if row["reaction_id"] == "ugi_3cr_agile"
    )
    roles = [row["name"] for row in reaction["reactant_roles"]]
    example = reaction["known_positive_examples"][0]
    return dict(zip(roles, example["reactants"], strict=True)), example["expected"]


def test_registry_example_reconstructs_exactly() -> None:
    adapter = Ugi3AssemblyAdapter.from_registry(
        REGISTRY,
        expected_sha256=str(sha256_file(REGISTRY)),
    )
    components, expected = _example()
    check = adapter.check_forward(components, expected)
    assert check.reaction_id == "ugi_3cr_agile"
    assert check.exact
    assert not check.saturated
    assert check.enumerated_outcomes == 1


def test_adapter_rejects_an_incomplete_role_mapping() -> None:
    adapter = Ugi3AssemblyAdapter.from_registry(REGISTRY)
    components, expected = _example()
    components.pop(next(iter(components)))
    with pytest.raises(Ugi3AssemblyError, match="roles differ"):
        adapter.check_forward(components, expected)
