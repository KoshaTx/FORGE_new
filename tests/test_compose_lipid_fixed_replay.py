"""Ketone source chemistry, role-specific nitrogen retention and protected replay."""

import copy
import hashlib
import json
from functools import partial
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.corpus.compose_lipid_family_replay import replay_record
from forge.corpus.compose_lipid_fixed_replay import load_contract, qualify_controls

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multireaction/compose_lipid_supplied_ketone_ugi4_v1.json"
FAMILY = "ketone_ugi4"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


@pytest.fixture
def source(contract):
    return json.loads(contract[1]["transform_controls"].read_text())["source_controls"][0]


def test_source_fo32_forward_inverse_formula_and_retained_head_pass(contract):
    result = contract[-1]["FO_32"]["replay"]
    assert result["computed_consistency_pass"]
    assert all(result["checks"].values())
    assert result["inverse_candidate_count"] == 1
    assert result["retained_source_role_witnesses"]
    assert not result["experimental_execution_admitted"]


def test_ambiguous_target_remains_excluded_despite_matching_product(contract):
    result = contract[-1]["asymmetric_two_primary_amines"]["replay"]
    assert len(result["forward_products"]) == 2
    assert result["checks"]["distinct_source_role_handles_retained"]
    assert result["checks"]["source_reactive_site_witness"]
    assert not result["computed_consistency_pass"]


def test_other_role_basic_nitrogen_cannot_rescue_consumed_head(contract):
    result = contract[-1]["cross_role_basic_nitrogen_cannot_rescue_head"]["replay"]
    assert result["checks"]["unique_unfiltered_forward_exact"]
    assert result["checks"]["unique_unfiltered_inverse_exact"]
    assert result["checks"]["full_element_hydrogen_charge_balance"]
    assert not result["checks"]["distinct_source_role_handles_retained"]
    assert not result["computed_consistency_pass"]


@pytest.mark.parametrize("carbonyl", ["CCC=O", "CC(=O)O", "CC(=O)OC", "CC(=O)N"])
def test_aldehyde_acid_ester_amide_are_not_ketone_substitutes(contract, source, carbonyl):
    run = contract[2][FAMILY]["run"]
    parts = {**source["components"], "coupled_ketone": carbonyl}
    result = run(parts, source["expected_product"])
    assert not result["computed_consistency_pass"]
    assert result["forward_products"] == []


def test_enumeration_saturation_fails_loudly(contract, source):
    run = partial(contract[2][FAMILY]["run"], maximum_outcomes=1)
    with pytest.raises(LibraryAssemblyError, match="saturat"):
        run(source["components"], source["expected_product"])


def test_component_and_product_atom_reordering_is_invariant(contract, source):
    def reorder(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        return Chem.MolToSmiles(mol, canonical=False)

    result = contract[2][FAMILY]["run"](
        {r: reorder(s) for r, s in source["components"].items()},
        reorder(source["expected_product"]),
    )
    assert result["computed_consistency_pass"]


def test_retained_nitrogen_requires_original_role_even_if_order_changes(contract):
    document = json.loads(contract[1]["transform_controls"].read_text())
    c = document["regression_controls"][1]
    parts = {
        role: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=False, rootedAtAtom=1)
        for role, s in c["components"].items()
    }
    result = contract[2][FAMILY]["run"](parts, c["expected_product"])
    assert not result["checks"]["distinct_source_role_handles_retained"]


@pytest.mark.parametrize(
    "change",
    ["missing_negative", "formula", "expected", "duplicate", "attribution", "failed_check"],
)
def test_control_tampering_fails_before_corpus_access(contract, change):
    doc = json.loads(contract[1]["transform_controls"].read_text())
    if change == "missing_negative":
        doc["regression_controls"] = []
    elif change == "formula":
        doc["source_controls"][0]["expected_formula"] = "C"
    elif change == "expected":
        doc["regression_controls"][0]["expected_computed_consistency_pass"] = True
    elif change == "duplicate":
        doc["source_controls"] *= 2
    elif change == "attribution":
        doc["source_controls"][0]["source_asset"] = "missing"
    else:
        doc["regression_controls"][0]["required_failed_checks"] = [
            "full_element_hydrogen_charge_balance"
        ]
    with pytest.raises(ComposeLipidError):
        qualify_controls(doc, contract[2])


def test_protected_row_rejected_before_invalid_graph_decoding(contract):
    with pytest.raises(ComposeLipidError, match="Protected"):
        replay_record(
            {
                "source": {"constitution": "invalid"},
                "preparation": {"eligible_for_program_preparation": False},
            },
            {},
            contract[2][FAMILY],
        )


def prepared(source):
    structures = {f"id:{role}": smiles for role, smiles in source["components"].items()}
    identity = hashlib.sha256(
        constitutional_molecule(source["expected_product"])[0].encode()
    ).hexdigest()
    item = {
        "source": {"constitution": source["expected_product"]},
        "preparation": {
            "eligible_for_program_preparation": True,
            "constitution_id": identity,
            "component_instances": [[role, f"id:{role}", 1] for role in source["components"]],
        },
    }
    return item, structures


def test_complete_supplied_global_bindings_roundtrip(contract, source):
    item, structures = prepared(source)
    result = replay_record(item, structures, contract[2][FAMILY])
    assert result["disposition"] == "exact_computed_reconstruction"
    assert result["verified_target_constitution_id"] == item["preparation"]["constitution_id"]


@pytest.mark.parametrize("change", ["quantity", "two_head_ids", "unknown_id", "target_identity"])
def test_supplied_bindings_and_identity_cannot_be_target_selected(contract, source, change):
    item, structures = prepared(source)
    instances = item["preparation"]["component_instances"]
    if change == "quantity":
        instances[0][2] = 2
    elif change == "two_head_ids":
        instances.append(copy.deepcopy(instances[0]))
    elif change == "unknown_id":
        instances[0][1] = "unresolved"
    else:
        item["preparation"]["constitution_id"] = "0" * 64
    if change in ("unknown_id", "target_identity"):
        with pytest.raises(ComposeLipidError):
            replay_record(item, structures, contract[2][FAMILY])
    else:
        assert not replay_record(item, structures, contract[2][FAMILY])["computed_consistency_pass"]


@pytest.mark.parametrize("change", ["pin", "roles", "bound", "policy"])
def test_configuration_substitution_fails(contract, tmp_path, change):
    cfg = copy.deepcopy(contract[0])
    if change == "pin":
        cfg["inputs"]["registry"]["sha256"] = "0" * 64
    elif change == "roles":
        cfg["families"][FAMILY]["registry_to_source_roles"]["amine_head"] = "isocyanide"
    elif change == "bound":
        cfg["maximum_outcomes"] = True
    else:
        cfg["policy"]["training_admitted"] = True
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises((ValueError, ComposeLipidError)):
        load_contract(ROOT, p)
