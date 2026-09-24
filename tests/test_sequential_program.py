"""Source controls, competing paths and strict bounds for ordered programs."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.assembly.repeated_components import RepeatBounds
from forge.assembly.sequential_program import RegistrySequentialProgram
from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/vendor/qualified_staar_source_program_v1.json"
CONTROLS = json.loads(
    (
        ROOT / "results/phase1/compose_lipid_v8_staar_source_v1/control_transcriptions.json"
    ).read_text()
)["controls"]


@pytest.fixture
def program():
    return RegistrySequentialProgram.from_registry(
        REGISTRY, program_id="source_staar_two_stage", expected_sha256=str(sha256_file(REGISTRY))
    )


@pytest.mark.parametrize("control", CONTROLS, ids=[c["label"] for c in CONTROLS])
def test_independent_source_products_reconstruct_through_both_stages(program, control):
    replay = program.replay(control["components"], control["expected_product"])
    assert replay["computed_consistency_pass"]
    assert all(replay["checks"].values())
    assert len(replay["forward_layers"]) == 3
    assert all(len(layer) == 1 for layer in replay["forward_layers"])
    assert replay["forward_layers"][1] == ["CCCCCCCCC(CCCCCC)C(=O)NC(CCS)C(=O)NCCN(C)C"]
    assert not replay["experimental_selectivity_qualified"]


def test_alternative_sites_are_retained_even_when_the_target_is_found(program):
    components = {**CONTROLS[0]["components"], "amine_head": "CNCCCN"}
    target = CONTROLS[0]["expected_product"].replace("CN(C)CCN", "CNCCCN", 1)
    replay = program.replay(components, target)
    assert constitutional_molecule(target)[0] in replay["forward_layers"][-1]
    assert len(replay["forward_layers"][1]) == len(replay["forward_layers"][2]) == 2
    assert not replay["computed_consistency_pass"]
    assert not replay["checks"]["unique_each_forward_stage"]


@pytest.mark.parametrize(
    "bounds",
    [
        RepeatBounds(maximum_outcomes=1),
        RepeatBounds(maximum_states=1),
        RepeatBounds(maximum_transitions=1),
    ],
)
def test_truncation_invalidates_the_entire_program(program, bounds):
    bounded = replace(program, bounds=bounds)
    replay = bounded.replay(CONTROLS[0]["components"], CONTROLS[0]["expected_product"])
    assert not replay["computed_consistency_pass"]
    assert not replay["checks"]["complete_search"]
    assert replay["bound_reasons"] or replay["inverse"]["bound_reasons"]


def test_head_losing_its_only_basic_site_is_not_admitted(program):
    c = CONTROLS[0]
    components = {**c["components"], "amine_head": "CCN"}
    target = c["expected_product"].replace("CN(C)CCN", "CCN", 1)
    replay = program.replay(components, target)
    assert replay["checks"]["unique_forward_exact"]
    assert not replay["checks"]["product_constraints"]
    assert not replay["computed_consistency_pass"]


@pytest.mark.parametrize("kind", ["amino_tail", "extra_sulfur", "acrylamide", "wrong_ring"])
def test_out_of_scope_components_cannot_be_admitted(program, kind):
    c = CONTROLS[0]
    components = dict(c["components"])
    if kind == "amino_tail":
        components["acrylate_tail"] = "C=CC(=O)OCCN"
    elif kind == "extra_sulfur":
        components["thiolactone_region"] = "O=C1SCCC1NC(=O)CCSCC"
    elif kind == "acrylamide":
        components["acrylate_tail"] = "C=CC(=O)NCCCCCCCC"
    else:
        components["thiolactone_region"] = "O=C1SCCCC1NC(=O)CCCCCCCC"
    replay = program.replay(components, c["expected_product"])
    assert not replay["computed_consistency_pass"]


def test_single_cut_or_stage_cannot_stand_in_for_the_complete_program(program):
    replay = program.replay(CONTROLS[0]["components"], "CCCCCCCCC(CCCCCC)C(=O)NC(CCS)C(=O)NCCN(C)C")
    assert not replay["computed_consistency_pass"]
    assert not replay["checks"]["full_element_hydrogen_charge_balance"]
    assert not replay["checks"]["unique_complete_inverse"]


def test_atom_order_and_smiles_serialization_do_not_change_programs(program):
    def permute(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False
        )

    c = CONTROLS[0]
    original = program.replay(c["components"], c["expected_product"])
    assert (
        program.replay(
            {k: permute(v) for k, v in c["components"].items()}, permute(c["expected_product"])
        )
        == original
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "stage_order",
        "reuse_role",
        "missing_constraint",
        "negative_balance",
        "unrecognized_constraint",
    ],
)
def test_changed_program_contract_fails_closed(tmp_path, mutation):
    document = json.loads(REGISTRY.read_text())
    spec = document["programs"][0]
    if mutation == "stage_order":
        spec["stages"].reverse()
    elif mutation == "reuse_role":
        spec["stages"][1]["added_role"] = "amine_head"
    elif mutation == "missing_constraint":
        spec["terminal_constraints"].pop("amine_head")
    elif mutation == "negative_balance":
        spec["stages"][0]["net_byproducts"] = {"H": -1}
    else:
        spec["product_constraints"]["allow_incomplete"] = True
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document))
    with pytest.raises(LibraryAssemblyError):
        RegistrySequentialProgram.from_registry(
            path, program_id=spec["program_id"], expected_sha256=str(sha256_file(path))
        )


def test_stale_pin_and_incomplete_roles_are_rejected(program):
    with pytest.raises(LibraryAssemblyError, match="hash mismatch"):
        RegistrySequentialProgram.from_registry(
            REGISTRY, program_id="source_staar_two_stage", expected_sha256="0" * 64
        )
    missing = copy.deepcopy(CONTROLS[0]["components"])
    missing.pop("amine_head")
    with pytest.raises(LibraryAssemblyError, match="all terminal roles"):
        program.replay(missing, CONTROLS[0]["expected_product"])
