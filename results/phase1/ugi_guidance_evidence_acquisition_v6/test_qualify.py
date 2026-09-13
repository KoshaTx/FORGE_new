"""Protect exact identity, stock evidence and expiry for the fresh supplier record."""

import copy
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from rdkit import Chem

SPEC = importlib.util.spec_from_file_location(
    "fresh_terminal_v6", Path(__file__).with_name("qualify.py")
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def source_args():
    config = audit.read(audit.CONFIG)
    expected = {
        **config["molecules"]["stearolic_acid"],
        "cas_number": config["terminal_procurement"]["cas_number"],
    }
    return [
        (audit.OUT / "batch_07/stearolic_accelsci.html").read_text(),
        (audit.OUT / "batch_08/accelsci_shipping.html").read_text(),
        expected,
    ]


def test_exact_identity_and_observed_grade_are_preserved(source_args):
    observed = audit.parse_accel(*source_args)
    assert observed["canonical_smiles"] == source_args[2]["canonical_smiles"]
    assert [r["pack_size"] for r in observed["packs"]] == ["250mg", "1g", "5g"]
    assert observed["item_specific_purity"] == "98%"
    assert observed["us_delivery_offered_by_supplier"]
    assert not observed["us_warehouse_inventory_established"]
    assert not observed["item_specific_delivery_date_or_quantity_confirmed"]


@pytest.mark.parametrize("replacement", ["Back order", "Inquiry", "", "<!-- Global Stock -->"])
def test_no_visible_stock_is_not_admitted(source_args, replacement):
    source_args[0] = source_args[0].replace("Global Stock", replacement)
    with pytest.raises(ValueError):
        audit.parse_accel(*source_args)


def test_cas_and_formula_cannot_rescue_wrong_graph(source_args):
    source_args[0] = source_args[0].replace(
        source_args[2]["canonical_smiles"], "CCCCCCCCC=CCCCCCCCC(=O)O"
    )
    with pytest.raises(ValueError, match="graph differs"):
        audit.parse_accel(*source_args)


def test_full_key_is_required(source_args):
    source_args[2]["inchi_key"] = source_args[2]["inchi_key"][:-1] + "X"
    with pytest.raises(ValueError, match="InChIKey"):
        audit.parse_accel(*source_args)


def test_shipping_without_explicit_us_destination_is_insufficient(source_args):
    source_args[1] = source_args[1].replace("United States", "Worldwide")
    with pytest.raises(ValueError, match="US destination"):
        audit.parse_accel(*source_args)


def test_atom_order_does_not_change_qualification(source_args):
    expected = audit.parse_accel(*source_args)
    molecule = Chem.MolFromSmiles(source_args[2]["canonical_smiles"])
    reversed_mol = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
    source_args[2]["canonical_smiles"] = Chem.MolToSmiles(reversed_mol, canonical=False)
    assert audit.parse_accel(*source_args) == expected


def test_conflicting_pack_is_not_silently_dropped(source_args):
    source_args[0] = source_args[0].replace("Global Stock", "Back order", 1)
    with pytest.raises(ValueError, match="Conflicting"):
        audit.parse_accel(*source_args)


def test_terminal_passes_unchanged_contract_but_expiry_and_tampering_fail(tmp_path):
    record, _ = audit.source("batch_07", "stearolic_accelsci")
    assessed = (
        datetime.fromisoformat(record["completed_at_utc"]) + timedelta(minutes=10)
    ).isoformat()
    observation, terminal = audit.derive(assessed)
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(observation))
    audit.check_terminal(terminal, path, assessed)
    expired = (
        datetime.fromisoformat(terminal["expires_at_utc"]) + timedelta(seconds=1)
    ).isoformat()
    future = (
        datetime.fromisoformat(terminal["observed_at_utc"]) - timedelta(seconds=1)
    ).isoformat()
    for date in (expired, future):
        with pytest.raises(ValueError, match="outside"):
            audit.check_terminal(terminal, path, date)
    changed = copy.deepcopy(observation)
    changed["item"]["catalog_number"] = "different-item"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="disagrees"):
        audit.check_terminal(terminal, path, assessed)


def test_missing_purity_is_not_admitted(source_args):
    source_args[0] = source_args[0].replace("98%", "unspecified")
    with pytest.raises(ValueError, match="purity"):
        audit.parse_accel(*source_args)
