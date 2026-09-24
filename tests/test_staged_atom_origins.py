import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem

from forge.assembly.families import LibraryAssemblyError
from forge.assembly.repeated_components import RepeatBounds
from forge.assembly.sequential_program import RegistrySequentialProgram
from forge.assembly.staged_atom_origins import trace_staged_program
from forge.assembly.staged_program import RegistryStagedProgram
from forge.core.hashing import resolve_pin

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def contracts():
    answer = {}
    for kind, path, cls in (
        (
            "thiol_yne",
            "results/phase1/compose_lipid_thiol_yne_source_v1/replay-config.json",
            RegistryStagedProgram,
        ),
        (
            "staar",
            "configs/multireaction/compose_lipid_v8_staar_program_v1.json",
            RegistrySequentialProgram,
        ),
    ):
        config = json.loads((ROOT / path).read_text())
        registry = resolve_pin(config["inputs"]["registry"], ROOT, label=kind)
        binding = next(iter(config["families"].values()))
        program = cls.from_registry(
            registry,
            program_id=binding["program_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
            bounds=RepeatBounds(**config["search_bounds"]),
        )
        if kind == "thiol_yne":
            source = json.loads(
                resolve_pin(
                    config["inputs"]["transform_controls"], ROOT, label="controls"
                ).read_text()
            )
            mapping = {
                role: origin
                for origin, roles in binding["source_role_occurrences"].items()
                for role in roles
            }
            controls = source["source_controls"]
            negative = source["regression_controls"]
        else:
            source = json.loads(
                resolve_pin(
                    config["inputs"]["adjudication"], ROOT, label="adjudication"
                ).read_text()
            )
            controls = json.loads(
                resolve_pin(
                    source["assets"]["control_transcriptions.json"], ROOT, label="controls"
                ).read_text()
            )["controls"]
            mapping, negative = {role: role for role in program.roles}, []
        answer[kind] = program, mapping, controls, negative
    return answer


@pytest.mark.parametrize("kind", ["thiol_yne", "staar"])
@pytest.mark.parametrize("index", [0, 1])
def test_source_products_and_every_stage_core_membership_survive(contracts, kind, index):
    program, mapping, controls, _ = contracts[kind]
    control = controls[index]
    traced = trace_staged_program(program, control["components"], source_roles=mapping)
    original = program.replay(control["components"], control["expected_product"])
    assert original["computed_consistency_pass"]
    assert traced.complete_search and traced.disposition == "unique_forward_atom_coordinates"
    assert traced.constitutional_products == tuple(original["forward_layers"][-1])
    assert traced.states_by_layer == tuple(len(layer) for layer in original["forward_layers"])
    for depth, adapter in enumerate(program.adapters, start=1):
        declared = {
            a.GetAtomMapNum()
            for product in adapter.reaction.forward.GetProducts()
            for a in product.GetAtoms()
            if a.GetAtomMapNum()
        }
        observed = [
            token
            for core in traced.annotations.core_positions
            for token in core.split("|")
            if token.startswith(f"step_{depth}:")
        ]
        assert sorted(observed) == sorted(f"step_{depth}:map_{number}" for number in declared)
    if kind == "staar":
        assert any("|" in value for value in traced.annotations.core_positions)
    assert set(traced.annotations.atom_roles) == set(mapping.values())


def test_all_competing_final_products_are_preserved(contracts):
    program, mapping, _, controls = contracts["thiol_yne"]
    control = controls[0]
    result = trace_staged_program(program, control["components"], source_roles=mapping)
    original = program.replay(control["components"], control["expected_product"])
    assert result.complete_search and result.annotations is None
    assert result.disposition == "ambiguous_forward_products"
    assert len(result.constitutional_products) == 2
    assert result.constitutional_products == tuple(original["forward_layers"][-1])


@pytest.mark.parametrize("bound", ["maximum_outcomes", "maximum_states", "maximum_transitions"])
def test_truncated_search_never_supplies_labels(contracts, bound):
    program, mapping, controls, _ = contracts["staar"]
    bounded = replace(program, bounds=replace(program.bounds, **{bound: 1}))
    result = trace_staged_program(bounded, controls[0]["components"], source_roles=mapping)
    assert not result.complete_search and result.annotations is None
    assert result.disposition.endswith("bound")


def test_undeclared_role_collapse_is_rejected(contracts):
    program, mapping, controls, _ = contracts["staar"]
    with pytest.raises(LibraryAssemblyError, match="collapse"):
        trace_staged_program(
            program, controls[0]["components"], source_roles={r: "same" for r in mapping}
        )


def test_equal_occurrences_must_share_a_source_role_and_structure(contracts):
    program, mapping, controls, _ = contracts["thiol_yne"]
    with pytest.raises(LibraryAssemblyError, match="collapse"):
        trace_staged_program(
            program, controls[0]["components"], source_roles={r: r for r in mapping}
        )
    first, second = program.specification["equal_component_groups"][0]
    components = dict(controls[0]["components"])
    components[second] = "C" + components[first]
    with pytest.raises(LibraryAssemblyError, match="different structures"):
        trace_staged_program(program, components, source_roles=mapping)


@pytest.mark.parametrize("kind", ["thiol_yne", "staar"])
def test_input_atom_order_does_not_choose_a_semantic_path(contracts, kind):
    program, mapping, controls, _ = contracts[kind]
    parts = controls[0]["components"]
    permuted = {}
    for role, smiles in parts.items():
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        permuted[role] = Chem.MolToSmiles(mol, canonical=False)
    assert trace_staged_program(program, parts, source_roles=mapping) == trace_staged_program(
        program, permuted, source_roles=mapping
    )
    assert "target" not in inspect.signature(trace_staged_program).parameters


def test_isotope_query_that_has_no_literal_isotope_is_rejected(contracts):
    program, mapping, controls, _ = contracts["staar"]
    original = program.adapters[0]
    query = Chem.MolFromSmarts("[!13C:1]")
    assert query.GetAtomWithIdx(0).GetIsotope() == 0
    forward = SimpleNamespace(
        GetReactants=lambda: (query,), GetProducts=original.reaction.forward.GetProducts
    )
    changed = SimpleNamespace(reaction=SimpleNamespace(forward=forward))
    with pytest.raises(LibraryAssemblyError, match="isotope-free registry queries"):
        trace_staged_program(
            replace(program, adapters=(changed, *program.adapters[1:])),
            controls[0]["components"],
            source_roles=mapping,
        )


@pytest.mark.parametrize("lost_origin", [False, True])
def test_same_product_paths_cannot_merge_different_origins(contracts, lost_origin):
    template = contracts["staar"][0].adapters[0].reaction.forward
    number = next(
        a.GetAtomMapNum() for p in template.GetProducts() for a in p.GetAtoms() if a.GetAtomMapNum()
    )

    def first(reactants, **kwargs):
        a = reactants[0].GetAtomWithIdx(0).GetIsotope()
        b = reactants[1].GetAtomWithIdx(0).GetIsotope()
        answer = []
        for tags in ((a, b, b), (b, a, b)):
            mol = Chem.MolFromSmiles("CCO")
            for atom, tag in zip(mol.GetAtoms(), tags):
                atom.SetIsotope(0 if lost_origin else tag)
            mol.GetAtomWithIdx(2).SetIntProp("old_mapno", number)
            answer.append((mol,))
        return tuple(answer)

    def second(reactants, **kwargs):
        assert not any(atom.HasProp("old_mapno") for atom in reactants[0].GetAtoms())
        return ((Chem.Mol(reactants[0]),),)

    def adapter(roles, forward):
        return SimpleNamespace(
            roles=roles,
            _assess=lambda values: [SimpleNamespace(qualified=True)] * len(values),
            reaction=SimpleNamespace(
                forward=SimpleNamespace(
                    GetReactants=template.GetReactants,
                    GetProducts=template.GetProducts,
                    RunReactants=forward,
                )
            ),
        )

    program = SimpleNamespace(
        roles=("initial", "side", "last"),
        bounds=RepeatBounds(),
        specification={
            "initial_role": "initial",
            "terminal_constraints": {r: {} for r in ("initial", "side", "last")},
            "product_constraints": {},
            "stages": [
                {"accumulator_role": "initial", "added_role": "side"},
                {"accumulator_role": "intermediate", "added_role": "last"},
            ],
        },
        adapters=(adapter(("initial", "side"), first), adapter(("intermediate", "last"), second)),
    )
    result = trace_staged_program(
        program, {r: "CC" for r in program.roles}, source_roles={r: r for r in program.roles}
    )
    assert result.annotations is None
    if lost_origin:
        assert result.disposition == "assembly_introduced_or_unresolved_atom_origin"
        assert not result.complete_search
    else:
        assert result.states_by_layer == (1, 2, 2)
        assert result.constitutional_products == ("CCO",)
        assert result.disposition == "ambiguous_forward_atom_coordinates"
