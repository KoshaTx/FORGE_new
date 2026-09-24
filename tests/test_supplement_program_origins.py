"""New source controls preserve component multiplicity and oxygen provenance."""

import json
from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem

from forge.core.hashing import resolve_pin
from results.phase1.compose_lipid_supplement_origins_v1.run import (
    checked_control,
    load_programs,
    select,
    trace,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.needs_vendor


@pytest.fixture(scope="module")
def contracts():
    config = json.loads(
        (ROOT / "results/phase1/compose_lipid_supplement_origins_v1/config.json").read_text()
    )
    executors, checked = load_programs(config)
    sources = {}
    for kind, key in (("aema", "source_control"), ("acid_epoxide", "transform_controls")):
        cfg = json.loads(resolve_pin(config["contracts"][kind], ROOT, label=kind).read_text())
        sources[kind] = json.loads(resolve_pin(cfg["inputs"][key], ROOT, label=key).read_text())
    return executors, checked, sources


@pytest.mark.parametrize(
    "family,kind,index,expected",
    [
        (
            "aema_aza_thiol_addition",
            "aema",
            3,
            {"amine_core": 10, "AEMA": 52, "thiol_periphery": 28},
        ),
        (
            "acid_epoxide_diester_multistep",
            "acid_epoxide",
            0,
            {"epoxide": 13, "hydrophobic_acid": 18, "amine_acid_head": 8},
        ),
    ],
)
def test_source_drawings_have_expected_origin_inventory(contracts, family, kind, index, expected):
    executors, _, sources = contracts
    source = sources[kind] if kind == "aema" else sources[kind]["source_controls"][0]
    executor = executors[family][index]
    traced = trace(executor, source["components"])
    assert traced.complete_search and traced.annotations is not None
    assert Counter(traced.annotations.atom_roles) == expected
    reversed_parts = {}
    for role, smiles in source["components"].items():
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        reversed_parts[role] = Chem.MolToSmiles(mol, canonical=False)
    assert trace(executor, reversed_parts).annotations == traced.annotations


def test_acid_head_loses_oxygen_and_epoxide_oxygen_keeps_both_stage_labels(contracts):
    executors, _, sources = contracts
    source = sources["acid_epoxide"]["source_controls"][0]
    annotation = trace(
        executors["acid_epoxide_diester_multistep"][0], source["components"]
    ).annotations
    mol = Chem.MolFromSmiles(annotation.canonical_product_smiles)
    oxygen = Counter()
    for atom, role, core in zip(
        mol.GetAtoms(), annotation.atom_roles, annotation.core_positions, strict=True
    ):
        if atom.GetSymbol() == "O":
            oxygen[role] += 1
            if role == "epoxide":
                assert "step_1:" in core and "step_2:" in core
    assert oxygen == {"amine_acid_head": 1, "epoxide": 1, "hydrophobic_acid": 2}


def test_wrong_target_does_not_erase_valid_forward_origins(contracts):
    _, checked, _ = contracts
    negatives = [c for c in checked["acid_epoxide"] if not c["expected_target_match"]]
    assert negatives
    assert any(c["semantic_replay"]["annotations"] is not None for c in negatives)
    assert all(not c["original_replay"]["computed_consistency_pass"] for c in negatives)


def test_counts_are_selected_from_complete_original_recipe(contracts):
    executors, _, _ = contracts
    for n in range(1, 7):
        prepared = {
            "family": "aema_aza_thiol_addition",
            "component_instances": [
                ["amine_core", "core", 1],
                ["AEMA", "linker", n],
                ["thiol_periphery", "tail", n],
            ],
        }
        assert select(prepared, executors)["quantities"]["AEMA"] == n
        prepared["component_instances"][-1][-1] = n + 1
        with pytest.raises(ValueError, match="exactly one"):
            select(prepared, executors)


def test_source_target_swap_fails_positive_control(contracts):
    executors, checked, sources = contracts
    source = dict(sources["acid_epoxide"]["source_controls"][0])
    source["expected_product"] = "CC"
    with pytest.raises(ValueError, match="disposition changed"):
        checked_control(
            executors["acid_epoxide_diester_multistep"][0],
            source,
            checked["acid_epoxide"][0]["original_replay"],
            True,
        )
