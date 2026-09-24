"""Complete precursor scope, attachment-aware equality and bounded-search rejection."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.component_scope import RegistryComponentScopes
from forge.assembly.families import LibraryAssemblyError
from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/vendor/qualified_b5_staged_source_program_v1.json"
CONTROLS = ROOT / "results/phase1/compose_lipid_b5_source_v1/control-transcriptions.json"


@pytest.fixture(scope="module")
def registry():
    return json.loads(REGISTRY.read_text())


@pytest.fixture(scope="module")
def scopes():
    return RegistryComponentScopes.from_registry(REGISTRY, expected_sha256=sha256_file(REGISTRY))


@pytest.mark.parametrize("label", ["I71", "I81", "I91", "I93", "I95", "I97"])
def test_complete_independently_transcribed_source_controls(scopes, registry, label):
    c = next(c for c in json.loads(CONTROLS.read_text())["controls"] if c["label"] == label)
    for role, smiles in c["components"].items():
        scope = registry["profile_component_scopes"][c["profile"]][role]
        before = scopes.assess(scope, smiles)
        assert before["pass"] and before["complete_search"]
        mol = Chem.MolFromSmiles(smiles)
        reordered = Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False
        )
        assert scopes.assess(scope, reordered) == before


@pytest.mark.parametrize(
    "name,role,scope",
    [
        ("I7_unequal_internal_arms", "tail6_alcohol", "paired_tail_alcohol"),
        ("I8_secondary_tail_alcohol", "r2_tail_alcohol", "primary_tail_alcohol"),
        ("I8_unreported_head_scaffold", "head_acid", "I8_head_acid_spacer"),
    ],
)
def test_mechanically_exact_but_outside_complete_scope_is_rejected(scopes, name, role, scope):
    receipt = json.loads((CONTROLS.parent / "draft-adversarial-audit.json").read_text())
    gap = receipt["remaining_complete_terminal_scope_counterexamples"][name]
    assert gap["draft_mechanical_consistency_pass"]
    assert not scopes.assess(scope, gap["components"][role])["pass"]


@pytest.mark.parametrize(
    "scope,smiles",
    [
        ("primary_tail_alcohol", "CCCCCOCCO"),
        ("primary_tail_alcohol", "OCC1CCCCC1"),
        ("primary_tail_alcohol", "CC#CCCCO"),
        ("primary_tail_alcohol", "CC=CC=CCO"),
        ("primary_tail_alcohol", "CC(C)CC(C)CCO"),
        ("primary_tail_alcohol", "CCC(C)(C)CCO"),
        ("primary_tail_alcohol", "c1ccccc1CCO"),
        ("primary_tail_alcohol", "OCCCCCCO"),
        ("half_ester_tail_acid", "CCCCOC(=O)C(C)CC(=O)O"),
        ("half_ester_tail_acid", "CCCCOC(=O)COCC(=O)O"),
        ("half_ester_tail_acid", "CCCCOC(=O)CCC(=O)OC"),
        ("half_ester_tail_acid", "CCC(OC(=O)CCC(=O)O)CC"),
        ("half_ester_tail_acid", "CCCCOC(=O)C(=O)O"),
        ("I8_head_acid_spacer", "CN(C)C(C)CC(=O)O"),
        ("I9_head_amine_spacer", "CN(C)CCNC"),
        ("I9_head_amine_spacer", "CN(C)CCN(C)C"),
        ("source_core", "CC(C)(CO)CC(O)C(=O)NCCC(=O)O"),
    ],
)
def test_complete_scope_rejects_extra_functions_and_wrong_scaffolds(scopes, scope, smiles):
    assert not scopes.assess(scope, smiles)["pass"]


def test_equal_hydrocarbon_formula_does_not_erase_attachment_position(scopes):
    first = scopes.assess("primary_tail_alcohol", "CCC(C)CCO")
    second = scopes.assess("primary_tail_alcohol", "CC(C)CCCO")
    assert first["pass"] and second["pass"]
    assert first["fragment_signatures"][0]["bodies"] != second["fragment_signatures"][0]["bodies"]


def test_internal_pair_equality_preserves_attachment_position(scopes):
    same = "OCC(OC(=O)CCC(C)CC)COC(=O)CCC(C)CC"
    unequal = "OCC(OC(=O)CCC(C)CC)COC(=O)CC(C)CCC"
    assert scopes.assess("paired_tail_alcohol", same)["pass"]
    assert not scopes.assess("paired_tail_alcohol", unequal)["pass"]


def test_identical_bodies_compare_across_two_original_functional_groups(scopes):
    alcohol = scopes.assess("primary_tail_alcohol", "CCC(C)CCO")
    acid = scopes.assess("half_ester_tail_acid", "CCC(C)CCOC(=O)CCCC(=O)O")
    assert alcohol["pass"] and acid["pass"]
    assert alcohol["fragment_signatures"][0]["bodies"] == acid["fragment_signatures"][0]["bodies"]


def test_reported_and_generated_head_spacers_are_separate(scopes):
    assert scopes.assess("I8_head_acid_spacer", "CN(C)CCCC(=O)O")["pass"]
    assert not scopes.assess("I8_head_acid_reported", "CN(C)CCCC(=O)O")["pass"]
    assert scopes.assess("I8_head_acid_reported", "CN(C)CCC(=O)O")["pass"]


def test_head_spacer_has_no_silent_fixed_atom_cap(scopes):
    assert scopes.assess("I8_head_acid_spacer", "CN(C)" + "C" * 110 + "C(=O)O")["pass"]


def test_complete_cut_search_saturation_fails_closed(scopes):
    # Deliberately duplicated query matches, even when they would collapse to one body.
    specifications = copy.deepcopy(scopes.specifications)
    specifications["primary_tail_alcohol"]["query"] = "[OH1:1]-[C:2]"
    limited = RegistryComponentScopes(specifications, maximum_matches=1)
    result = limited.assess("primary_tail_alcohol", "OCCCCCO")
    assert not result["complete_search"] and not result["pass"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_search_bound(scopes, value):
    with pytest.raises(LibraryAssemblyError):
        replace(scopes, maximum_matches=value)


@pytest.mark.parametrize("smiles", ["CCO.CC", "[13CH3]CO", "*CCO", "bad"])
def test_invalid_precursors_raise(scopes, smiles):
    with pytest.raises(LibraryAssemblyError):
        scopes.assess("primary_tail_alcohol", smiles)


def test_registry_pin_is_required():
    with pytest.raises(LibraryAssemblyError, match="checksum"):
        RegistryComponentScopes.from_registry(REGISTRY, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "change", ["unknown_kind", "duplicate_cut", "missing_anchor", "unknown_field"]
)
def test_malformed_registry_cannot_silently_drop_a_check(scopes, change):
    data = copy.deepcopy(scopes.specifications)
    target = data["primary_tail_alcohol"]
    if change == "unknown_kind":
        target["kind"] = "substructure"
    elif change == "duplicate_cut":
        target["cut_bonds"] *= 2
    elif change == "missing_anchor":
        target["central_anchor_map"] = 100
    else:
        target["ignore_branching"] = True
    with pytest.raises(LibraryAssemblyError):
        RegistryComponentScopes(data)
