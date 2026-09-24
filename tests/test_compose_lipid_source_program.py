from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.atom_map_variant import derive_product_atom_map_variant
from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_source_program import (
    CHECKS,
    IMPLEMENTATION,
    REMAINING_HOLDS,
    RESULT_SCHEMA,
    SCOPE,
    _load,
    _summary,
    check_complete_program,
    map_bonds_match,
    verify_source_program,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/compose_lipid_v8_passerini_program_v1.json"


@pytest.fixture
def pair(tmp_path):
    config, paths, variant = _load(REPO, CONFIG)
    original = RegistryAssemblyAdapter.from_registry(
        paths["registry"],
        reaction_id=config["original_reaction_id"],
        expected_sha256=str(sha256_file(paths["registry"])),
    )
    path = tmp_path / "derived.json"
    path.write_text(json.dumps({"reactions": [variant]}))
    corrected = RegistryAssemblyAdapter.from_registry(
        path, reaction_id=variant["reaction_id"], expected_sha256=str(sha256_file(path))
    )
    raw = next(
        r
        for r in json.loads(paths["registry"].read_text())["reactions"]
        if r["reaction_id"] == config["original_reaction_id"]
    )
    return config, raw, original, corrected


def test_source_examples_keep_graph_identity_and_correct_oxygen_origins(pair):
    config, _, original, corrected = pair
    contract = config["source_contract"]
    assert not map_bonds_match(original, contract["expected_product_map_bonds"])
    assert map_bonds_match(corrected, contract["expected_product_map_bonds"])
    for row in config["source_controls"]:
        assert (
            original.forward_products(row["components"]).products
            == corrected.forward_products(row["components"]).products
        )
        check = check_complete_program(
            corrected, row["components"], row["expected_product"], contract
        )
        assert check["qualified"]
        old = check_complete_program(original, row["components"], row["expected_product"], contract)
        assert old["checks"]["unique_forward_exact"]
        assert not old["qualified"]


def test_variant_preserves_all_role_policies_and_frozen_source(pair):
    config, raw, _, _ = pair
    before = deepcopy(raw)
    variant = derive_product_atom_map_variant(raw, config["variant"])
    assert raw == before
    for key, value in raw.items():
        if key not in {"reaction_id", "reaction_version", "atom_mapped_reaction_smarts", "sources"}:
            assert variant[key] == value
    assert (
        variant["atom_mapped_reaction_smarts"].split(">>")[0]
        == raw["atom_mapped_reaction_smarts"].split(">>")[0]
    )


def test_generic_passerini_hit_without_a_retained_amine_is_not_source_qualified(pair):
    config, raw, _, corrected = pair
    example = raw["known_positive_examples"][0]
    components = dict(zip(corrected.roles, example["reactants"], strict=True))
    result = check_complete_program(
        corrected, components, example["expected"], config["source_contract"]
    )
    assert result["checks"]["unique_forward_exact"]
    assert not result["checks"]["retained_amine_in_acid_head"]
    assert not result["qualified"]


def test_source_target_substitution_is_rejected(pair):
    config, _, _, corrected = pair
    first, other = config["source_controls"]
    checked = check_complete_program(
        corrected, first["components"], other["expected_product"], config["source_contract"]
    )
    assert not checked["checks"]["unique_forward_exact"]
    assert not checked["checks"]["unique_inverse_exact"]
    assert not checked["qualified"]


def test_program_disposition_is_invariant_to_atom_order_and_smiles_serialization(pair):
    config, _, _, corrected = pair
    source = config["source_controls"][0]

    def reversed_smiles(smiles):
        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
        return Chem.MolToSmiles(molecule, canonical=False)

    original = check_complete_program(
        corrected, source["components"], source["expected_product"], config["source_contract"]
    )
    reordered = check_complete_program(
        corrected,
        {role: reversed_smiles(smiles) for role, smiles in source["components"].items()},
        reversed_smiles(source["expected_product"]),
        config["source_contract"],
    )
    assert reordered == original


@pytest.mark.parametrize(
    "permutation",
    [
        {"3": 3},
        {"3": 5},
        {"0": 5, "5": 0},
        {"3": 6, "6": 3},
        {"99": 100, "100": 99},
        {"03": 5, "5": 3},
        {"3": True, "5": 3},
    ],
)
def test_atom_map_variants_cannot_drop_duplicate_invent_or_change_elements(pair, permutation):
    config, raw, _, _ = pair
    spec = deepcopy(config["variant"])
    spec["product_map_permutation"] = permutation
    with pytest.raises(LibraryAssemblyError):
        derive_product_atom_map_variant(raw, spec)


@pytest.mark.parametrize("change", ["identity", "version", "source", "extra"])
def test_atom_map_variant_requires_a_distinct_attributed_version(pair, change):
    config, raw, _, _ = pair
    spec = deepcopy(config["variant"])
    if change == "identity":
        spec["reaction_id"] = raw["reaction_id"]
    elif change == "version":
        spec["reaction_version"] = raw["reaction_version"]
    elif change == "source":
        spec["source"] = {}
    else:
        spec["unreviewed_transform"] = True
    with pytest.raises(LibraryAssemblyError):
        derive_product_atom_map_variant(raw, spec)


@pytest.fixture
def ledger():
    """Synthetic receipt fixture, separate from corpus/source evidence."""
    components = [
        {
            "role": "test_role",
            "canonical_smiles": "C",
            "constitution_id": hashlib.sha256(b"C").hexdigest(),
            "historical_fold": "heldout",
        }
    ]
    config = {"family": "test_family", "source_contract": {"roles": {"test_role": 1}}}
    rows = [
        {
            "target_id": "synthetic_target",
            "family": "test_family",
            "components": components,
            "checks": dict.fromkeys(CHECKS, True),
            "qualified": True,
            "historical_protected_precursor": True,
            "training_admitted": False,
        }
    ]
    observed = {
        "synthetic_target": {
            "status": "exact_related_transform_protected_precursor",
            "components": deepcopy(components),
        }
    }
    return rows, config, observed


def test_program_qualification_does_not_admit_protected_training_rows(ledger):
    rows, config, observed = ledger
    assert _summary(rows, config, observed) == {
        "rows": 1,
        "complete_program_checks_pass": 1,
        "historical_protected_precursor": 1,
        "clear_of_known_historical_precursors": 0,
    }


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "missing",
        "outside",
        "admit",
        "family",
        "components",
        "hide_protection",
        "missing_check",
        "false_check",
        "coerce_check",
        "coerce_qualification",
    ],
)
def test_receipt_accounting_rejects_population_scope_and_check_tampering(ledger, change):
    rows, config, observed = ledger
    row = rows[0]
    if change == "duplicate":
        rows.append(deepcopy(row))
    elif change == "missing":
        rows.clear()
    elif change == "outside":
        row["target_id"] = "uninspected_target"
    elif change == "admit":
        row["training_admitted"] = True
    elif change == "family":
        row["family"] = "different_family"
    elif change == "components":
        row["components"][0]["historical_fold"] = "train"
    elif change == "hide_protection":
        row["historical_protected_precursor"] = False
    elif change == "missing_check":
        row["checks"].pop("unique_forward_exact")
    elif change == "false_check":
        row["checks"]["unique_forward_exact"] = False
    elif change == "coerce_check":
        row["checks"]["unique_forward_exact"] = 1
    else:
        row["qualified"] = 1
    with pytest.raises(ComposeLipidError):
        _summary(rows, config, observed)


@pytest.mark.parametrize(
    "change", ["training", "evidence", "scope", "holds", "omitted_source", "omitted_artifact"]
)
def test_verifier_rejects_expanded_claims_and_incomplete_manifests(tmp_path, change):
    config = json.loads(CONFIG.read_text())
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "single_family_program_qualified_training_unqualified",
        "config": {"path": str(CONFIG), "sha256": str(sha256_file(CONFIG))},
        "pretraining_result": config["pretraining_result"],
        "policy": config["policy"],
        "family": config["family"],
        "training_ready": False,
        "training_rows_admitted": 0,
        "other_source_families_qualified_by_this_receipt": [],
        "evidence_basis": "computed_transform_consistency",
        "scope": SCOPE,
        "remaining_holds": list(REMAINING_HOLDS),
        "source_event_program_qualified": True,
        "implementation": dict.fromkeys(IMPLEMENTATION),
        "artifacts": {"registry.json": None, "program_checks.jsonl.gz": None},
    }
    if change == "training":
        result["training_ready"] = True
    elif change == "evidence":
        result["evidence_basis"] = "experimentally_demonstrated"
    elif change == "scope":
        result["scope"] = "all_families"
    elif change == "holds":
        result["remaining_holds"] = []
    elif change == "omitted_source":
        result["implementation"] = {}
    else:
        result["artifacts"] = {}
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ComposeLipidError, match="receipt scope"):
        verify_source_program(REPO, path)
