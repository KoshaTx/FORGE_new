from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
    load_assembly_libraries,
)
from forge.core.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
REGISTRIES = (
    REPO / "data/vendor/qualified_reactions_v1.json",
    REPO / "data/vendor/qualified_reaction_families_v1.json",
)


def _entries():
    return [(path, raw) for path in REGISTRIES for raw in json.loads(path.read_text())["reactions"]]


@pytest.mark.parametrize("path,raw", _entries(), ids=[raw["reaction_id"] for _, raw in _entries()])
def test_all_registry_positive_negative_and_exact_inverse_controls(path, raw):
    adapter = RegistryAssemblyAdapter.from_registry(
        path, reaction_id=raw["reaction_id"], expected_sha256=str(sha256_file(path))
    )
    for example in raw["known_positive_examples"]:
        components = dict(zip(adapter.roles, example["reactants"], strict=True))
        check = adapter.check_forward(components, example["expected"])
        assert check.exact and not check.saturated
        assert all(value.qualified for value in adapter.assess_roles(components))
        traces = adapter.decompose(example["expected"])
        assert traces
        expected = tuple(
            (role, constitutional_molecule(smi)[0]) for role, smi in components.items()
        )
        assert expected in {trace.components for trace in traces}
        for trace in traces:
            assert adapter.check_forward(dict(trace.components), trace.product_smiles).exact
    for example in raw["known_negative_examples"]:
        components = dict(zip(adapter.roles, example["reactants"], strict=True))
        assert not adapter.forward_products(components).products


def _ugi():
    path = REGISTRIES[0]
    raw = json.loads(path.read_text())["reactions"][0]
    adapter = RegistryAssemblyAdapter.from_registry(
        path, reaction_id=raw["reaction_id"], expected_sha256=str(sha256_file(path))
    )
    example = raw["known_positive_examples"][0]
    return adapter, dict(zip(adapter.roles, example["reactants"], strict=True)), example["expected"]


@pytest.mark.parametrize("bound", [0, -1, True, 1.5, "64"])
def test_noninteger_and_nonpositive_bounds_rejected(bound):
    adapter, components, product = _ugi()
    with pytest.raises(LibraryAssemblyError, match="positive integer"):
        adapter.forward_products(components, maximum_outcomes=bound)
    with pytest.raises(LibraryAssemblyError, match="positive integer"):
        adapter.decompose(product, maximum_outcomes=bound)


def test_truncation_never_admitted_as_a_complete_decomposition():
    adapter, components, product = _ugi()
    assert adapter.forward_products(components, maximum_outcomes=1).saturated
    with pytest.raises(LibraryAssemblyError, match="saturated"):
        adapter.decompose(product, maximum_outcomes=1)


@pytest.mark.parametrize("bad", ["", "not-a-molecule", "CC.CC", "[13CH4]"])
def test_invalid_disconnected_or_isotope_labelled_components_are_explicit_errors(bad):
    adapter, components, _ = _ugi()
    components[adapter.roles[0]] = bad
    with pytest.raises(LibraryAssemblyError):
        adapter.forward_products(components)


def test_transform_hit_cannot_override_registry_multiplicity_policy():
    adapter, components, _ = _ugi()
    components[adapter.roles[0]] = "NCCNCCN"
    assessment = adapter.assess_roles(components)[0]
    assert assessment.handle_count == 3
    assert not assessment.qualified
    assert not adapter.forward_products(components).products


def test_unknown_role_and_wrong_product_do_not_pass():
    adapter, components, _ = _ugi()
    assert not adapter.check_forward(components, "CC").exact
    components["extra"] = "CC"
    with pytest.raises(LibraryAssemblyError, match="expected roles"):
        adapter.forward_products(components)


def test_identity_ignores_stereochemistry_and_atom_map_serialization():
    assert (
        constitutional_molecule("[CH3:7][C@H:8](O)CC")[0] == constitutional_molecule("CC(O)CC")[0]
    )


def test_library_union_is_exact_and_hash_pinned():
    families = [raw["reaction_id"] for _, raw in _entries()]
    pins = [(path, str(sha256_file(path))) for path in REGISTRIES]
    assert set(load_assembly_libraries(pins, expected_families=families)) == set(families)
    with pytest.raises(LibraryAssemblyError, match="hash mismatch"):
        load_assembly_libraries([(REGISTRIES[0], "0" * 64)], expected_families=families)
    with pytest.raises(LibraryAssemblyError, match="duplicate registry"):
        load_assembly_libraries(pins + pins, expected_families=families)
    with pytest.raises(LibraryAssemblyError, match="differ"):
        load_assembly_libraries(pins, expected_families=families[:-1])


@pytest.mark.parametrize("change", ["count", "duplicate_role", "semantics", "unqualified"])
def test_unsupported_contract_never_silently_coerced(tmp_path, change):
    from forge.assembly.registry import ReactionRegistryError

    document = json.loads(REGISTRIES[0].read_text())
    raw = document["reactions"][0]
    if change == "count":
        raw["reactant_roles"][0]["count"] = 2
    elif change == "duplicate_role":
        raw["reactant_roles"][1]["name"] = raw["reactant_roles"][0]["name"]
    elif change == "semantics":
        raw["reactant_roles"][0]["site_multiplicity_semantics"] = "invented"
    else:
        raw["status"] = "pending"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document))
    with pytest.raises((LibraryAssemblyError, ReactionRegistryError)):
        RegistryAssemblyAdapter.from_registry(
            path, reaction_id=raw["reaction_id"], expected_sha256=str(sha256_file(path))
        )
