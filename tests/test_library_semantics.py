from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.assembly.families import LibraryAssemblyError, load_assembly_libraries
from forge.assembly.library_semantics import trace_library_atom_semantics

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source():
    config = json.loads(
        (REPO / "configs/multireaction/library_program_dataset_v1.json").read_text()
    )
    libraries = load_assembly_libraries(
        [
            (REPO / config["inputs"][name]["path"], config["inputs"][name]["sha256"])
            for name in config["registries"]
        ],
        expected_families=config["programs"],
    )
    rows = {
        row["reaction_id"]: row
        for name in config["registries"]
        for row in json.loads((REPO / config["inputs"][name]["path"]).read_text())["reactions"]
    }
    return libraries, rows


@pytest.mark.parametrize(
    "family",
    [
        "amide_coupling_acid_amine",
        "aza_michael_amine_acrylate",
        "carbamate_amine_chloroformate",
        "disulfide_coupling",
        "epoxide_opening_amine",
        "iphos_amine_dioxaphospholane",
        "passerini_3cr",
        "thiol_michael_thioether",
        "urea_amine_isocyanate",
    ],
)
def test_registry_controls_have_exact_unambiguous_generic_semantics(source, family):
    libraries, rows = source
    adapter = libraries[family]
    example = rows[family]["known_positive_examples"][0]
    components = dict(zip(adapter.roles, example["reactants"], strict=True))
    semantics = trace_library_atom_semantics(
        adapter,
        components,
        [example["expected"]],
        accumulator_role=None,
    )
    from forge.assembly.families import constitutional_molecule

    assert semantics.canonical_product_smiles == constitutional_molecule(example["expected"])[0]
    assert set(semantics.atom_origins) == set(adapter.roles)
    assert any(semantics.core_positions)
    assert semantics.step_count == 1


@pytest.mark.parametrize("family", ["acetal_aldehyde_diol", "reductive_amination_amine_aldehyde"])
def test_symmetric_registry_controls_abstain_from_nonunique_atom_semantics(source, family):
    libraries, rows = source
    adapter = libraries[family]
    example = rows[family]["known_positive_examples"][0]
    components = dict(zip(adapter.roles, example["reactants"], strict=True))
    with pytest.raises(LibraryAssemblyError, match="ambiguous|did not reconstruct|disagree"):
        trace_library_atom_semantics(
            adapter, components, [example["expected"]], accumulator_role=None
        )


def test_repeated_semantics_preserve_each_precursor_role_and_prior_core_positions(source):
    libraries, rows = source
    adapter = libraries["aza_michael_amine_acrylate"]
    example = rows[adapter.reaction_id]["known_positive_examples"][0]
    components = dict(zip(adapter.roles, example["reactants"], strict=True))
    components["amine_head"] = "NCCN"
    first = adapter.forward_products(components).products[0]
    second = adapter.forward_products({**components, "amine_head": first}).products[0]
    result = trace_library_atom_semantics(
        adapter,
        components,
        [first, second],
        accumulator_role="amine_head",
    )
    assert result.step_count == 2
    assert result.canonical_product_smiles == second
    assert set(result.atom_origins) == set(adapter.roles)
    assert result.core_positions.count("map_4") == 2


def test_acetal_template_symmetry_uses_declared_core_position_orbits(source):
    libraries, rows = source
    adapter = libraries["acetal_aldehyde_diol"]
    example = rows[adapter.reaction_id]["known_positive_examples"][0]
    components = dict(zip(adapter.roles, example["reactants"], strict=True))
    result = trace_library_atom_semantics(
        adapter,
        components,
        [example["expected"]],
        accumulator_role=None,
        core_position_aliases={
            "map_1": "map_1",
            "map_3": "map_3_or_map_6",
            "map_4": "map_4_or_map_5",
            "map_5": "map_4_or_map_5",
            "map_6": "map_3_or_map_6",
        },
    )
    assert set(filter(None, result.core_positions)) == {
        "map_1",
        "map_3_or_map_6",
        "map_4_or_map_5",
    }


@pytest.mark.parametrize("bad", [[], ["CC", "CCC"]])
def test_fixed_arity_depth_mismatch_fails(source, bad):
    libraries, rows = source
    adapter = libraries["passerini_3cr"]
    example = rows[adapter.reaction_id]["known_positive_examples"][0]
    components = dict(zip(adapter.roles, example["reactants"], strict=True))
    with pytest.raises(LibraryAssemblyError):
        trace_library_atom_semantics(adapter, components, bad, accumulator_role=None)


def test_policy_violation_and_saturation_never_produce_semantics(source):
    libraries, _ = source
    adapter = libraries["aza_michael_amine_acrylate"]
    components = {"amine_head": "NCCNCCN", "alkyl_acrylate_or_acrylamide_tail": "C=CC(=O)OCC"}
    with pytest.raises(LibraryAssemblyError, match="policy"):
        trace_library_atom_semantics(adapter, components, ["CC"], accumulator_role="amine_head")
    valid = {"amine_head": "CN", "alkyl_acrylate_or_acrylamide_tail": "C=CC(=O)OCC"}
    expected = adapter.forward_products(valid).products[0]
    with pytest.raises(LibraryAssemblyError, match="saturated"):
        trace_library_atom_semantics(
            adapter, valid, [expected], accumulator_role="amine_head", maximum_outcomes=1
        )
