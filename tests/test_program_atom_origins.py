"""All-path tracing, graph symmetry, source policies and explicit search bounds."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem

from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.program_atom_origins import canonical_coordinates, trace_repeated_program
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.core.hashing import resolve_pin

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def michael():
    config = json.loads(
        (ROOT / "configs/multireaction/compose_lipid_supplied_michael_v1.json").read_text()
    )
    registry = resolve_pin(config["inputs"]["registry"], ROOT, label="Michael registry")
    document = json.loads(registry.read_text())
    source = json.loads(
        resolve_pin(config["inputs"]["transform_controls"], ROOT, label="controls").read_text()
    )
    adapters, programs = {}, {}
    for family, binding in config["families"].items():
        adapters[family] = RegistryAssemblyAdapter.from_registry(
            registry,
            reaction_id=binding["reaction_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
        )
        programs[family] = next(
            r["source_program"]
            for r in document["reactions"]
            if r["reaction_id"] == binding["reaction_id"]
        )
    return config, source, adapters, programs


def trace(contract, control, **overrides):
    config, _, adapters, programs = contract
    family = control["family"]
    arguments = {
        "accumulator_role": programs[family]["accumulator_role"],
        "events": control["events"],
        "source_roles": config["families"][family]["registry_to_source_roles"],
        "bounds": RepeatBounds(**config["search_bounds"]),
    }
    arguments.update(overrides)
    return trace_repeated_program(adapters[family], control["components"], **arguments)


@pytest.mark.parametrize("index", range(3))
def test_characterized_controls_preserve_precursor_blocks_and_core(michael, index):
    control = michael[1]["source_controls"][index]
    result = trace(michael, control)
    assert result.disposition == "unique_forward_atom_coordinates"
    assert result.complete_search
    expected = Chem.MolToSmiles(
        Chem.MolFromSmiles(control["expected_product"]), isomericSmiles=False
    )
    assert result.constitutional_products == (expected,)
    origin = result.annotations
    assert len(origin.atom_roles) == Chem.MolFromSmiles(expected).GetNumAtoms()
    for role, smiles in control["components"].items():
        quantity = 1 if role == "amine_head" else control["events"]
        assert origin.atom_roles.count(role) == quantity * Chem.MolFromSmiles(smiles).GetNumAtoms()
    assert "exterior" in origin.core_positions
    assert sum(value != "exterior" for value in origin.core_positions) == 11


def test_ambiguous_source_control_keeps_all_products_without_target_selection(michael):
    control = michael[1]["ambiguity_controls"][0]
    result = trace(michael, control)
    assert result.complete_search and result.annotations is None
    assert result.disposition == "ambiguous_forward_products"
    family = control["family"]
    program = michael[3][family]
    original = replay_repeated_components(
        michael[2][family],
        control["components"],
        control["expected_product"],
        accumulator_role=program["accumulator_role"],
        events=control["events"],
        byproducts_per_event=program["net_byproducts_per_event"],
        bounds=RepeatBounds(**michael[0]["search_bounds"]),
    )
    assert result.constitutional_products == tuple(original["forward_layers"][-1])


@pytest.mark.parametrize("bound", ["maximum_outcomes", "maximum_states", "maximum_transitions"])
def test_search_saturation_never_yields_coordinates(michael, bound):
    result = trace(michael, michael[1]["source_controls"][0], bounds=RepeatBounds(**{bound: 1}))
    assert not result.complete_search
    assert result.annotations is None
    assert result.disposition.endswith("bound")


@pytest.mark.parametrize("events", [0, True, 8])
def test_invalid_event_count_fails(michael, events):
    with pytest.raises(LibraryAssemblyError, match="bounded event count"):
        trace(michael, michael[1]["source_controls"][0], events=events)


def test_role_aliases_cannot_collapse(michael):
    with pytest.raises(LibraryAssemblyError):
        trace(
            michael,
            michael[1]["source_controls"][0],
            source_roles={"amine_head": "same", "acceptor_tail": "same"},
        )


def test_input_atom_order_changes_neither_origin_nor_core_coordinates(michael):
    control = michael[1]["source_controls"][0]
    alternate = {}
    for role, smiles in control["components"].items():
        molecule = Chem.MolFromSmiles(smiles)
        permuted = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
        alternate[role] = Chem.MolToSmiles(permuted, canonical=False)
    assert trace(michael, control) == trace(michael, {**control, "components": alternate})


def test_unqualified_head_remains_rejected(michael):
    control = michael[1]["source_controls"][0]
    changed = {**control, "components": {**control["components"], "amine_head": "CN(C)C"}}
    result = trace(michael, changed)
    assert result.disposition == "no_complete_forward_product"
    assert result.annotations is None


@pytest.mark.parametrize(
    "smiles", ["CC", "C1CCCCC1", "CC(C)(C)C", "C1NCN1", "CCOC(=O)CCNCCC(=O)OCC"]
)
def test_symmetry_classes_match_exhaustive_automorphisms(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(molecule, isomericSmiles=False))
    matches = molecule.GetSubstructMatches(canonical, uniquify=False, maxMatches=4096)
    assert len(matches) < 4096
    # Move a distinct label across every atom, independently of canonical atom order.
    for marked in range(-1, molecule.GetNumAtoms()):
        coordinates = tuple(
            ("marked" if i == marked else "other", "exterior")
            for i in range(molecule.GetNumAtoms())
        )
        reference = {tuple(coordinates[i] for i in match) for match in matches}
        result = canonical_coordinates(molecule, coordinates)
        assert (result is None) == (len(reference) != 1)
        if result is not None:
            assert tuple(zip(result.atom_roles, result.core_positions)) == next(iter(reference))


def test_core_symmetry_ambiguity_is_independent_of_precursor_origin():
    molecule = Chem.MolFromSmiles("CC")
    assert canonical_coordinates(molecule, (("head", "map_1"), ("head", "map_2"))) is None


def test_distinct_semantic_paths_are_not_merged_by_constitution(michael):
    control = michael[1]["source_controls"][0]
    # Reverse symmetric copies of the first addition by changing no chemistry: every
    # commuting placement must survive until the final core coordinates agree.
    head = "NCCCN"
    changed = {**control, "components": {**control["components"], "amine_head": head}, "events": 4}
    result = trace(michael, changed)
    assert result.complete_search
    assert max(result.states_by_layer) > 1
    assert result.disposition == "unique_forward_atom_coordinates"
    assert (
        result.annotations.atom_roles.count("amine_head") == Chem.MolFromSmiles(head).GetNumAtoms()
    )


def test_complete_search_with_no_target_parameter(michael):
    control = michael[1]["source_controls"][0]
    assert trace(michael, control) == trace(michael, {**control, "expected_product": "invalid"})


@pytest.mark.parametrize("lose_origin", [False, True])
def test_same_product_different_origins_cannot_be_deduplicated(michael, lose_origin):
    # A synthetic executor isolates the state-merging boundary; its chemistry is
    # intentionally not admitted. The real registry supplies the core namespace.
    template = michael[2]["aza_michael_acrylate"].reaction.forward

    def outcomes(reactants, **_):
        head = reactants[0].GetAtomWithIdx(0).GetIsotope()
        tail = reactants[1].GetAtomWithIdx(0).GetIsotope()
        output = []
        for tags in ((head, tail, tail), (tail, head, tail)):
            product = Chem.MolFromSmiles("CCO")
            for atom, tag in zip(product.GetAtoms(), tags):
                atom.SetIsotope(0 if lose_origin else tag)
            product.GetAtomWithIdx(2).SetIntProp("old_mapno", 1)
            output.append((product,))
        return tuple(output)

    adapter = SimpleNamespace(
        roles=("head", "tail"),
        _assess=lambda _: [SimpleNamespace(qualified=True)],
        reaction=SimpleNamespace(
            forward=SimpleNamespace(
                GetReactants=template.GetReactants,
                GetProducts=template.GetProducts,
                RunReactants=outcomes,
            )
        ),
    )
    result = trace_repeated_program(
        adapter,
        {"head": "C", "tail": "CO"},
        accumulator_role="head",
        events=1,
        source_roles={"head": "head", "tail": "tail"},
        bounds=RepeatBounds(),
    )
    assert result.annotations is None
    if lose_origin:
        assert result.disposition == "assembly_introduced_or_unresolved_atom_origin"
        assert not result.complete_search
    else:
        assert result.states_by_layer == (1, 2)
        assert result.disposition == "ambiguous_atom_coordinates"
        assert result.constitutional_products == ("CCO",)
