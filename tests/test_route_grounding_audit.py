"""Regression tests for the role-specific route skeleton derivations.

Every one of these guards a bug that actually happened and that produced a confident 0%:

  - multi-product reaction outcomes were flattened into a set, so an ester cleaved to acid plus
    alcohol became two unrelated molecules and no acid/diol pair could be reassembled;
  - the recovered primary amine came back as [NH3+] and failed every functional-group test;
  - the fix for that stripped the charges off the legitimate [N+]#[C-] of an isocyanide and
    broke the forward verification that is supposed to confirm the disconnection.

The known-route cases are taken from the frozen chemist packet, so the audit and the dossier
cannot silently disagree about what a route is.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]


def _audit():
    path = REPO / "scripts/phase1_route_grounding_audit_v1.py"
    spec = importlib.util.spec_from_file_location("forge_route_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["forge_route_audit"] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _audit()


# ----------------------------------------------------------------- known routes


@pytest.mark.parametrize(
    "tail,acid,diol",
    [
        # L01/L02 in the frozen packet
        ("CCCCCCCCCCC(=O)OCCCCCC=O", "CCCCCCCCCCC(=O)O", "OCCCCCCO"),
        # a branched acid, L03
        ("CCCCCC(C)CCC(=O)OCCCCCC=O", "CCCCCC(C)CCC(=O)O", "OCCCCCCO"),
        # an unsaturated acid, L04
        ("CCCCCCCC=CC(=O)OCCCCCC=O", "CCCCCCCC=CC(=O)O", "OCCCCCCO"),
        # a longer diol, L10
        ("CCCCCCCCC=CCCCC(=O)OCCCCCCCC=O", "CCCCCCCCC=CCCCC(=O)O", "OCCCCCCCCO"),
    ],
)
def test_known_aldehyde_routes_recover_the_packet_precursors(tail, acid, diol):
    out = AUDIT.aldehyde_skeleton(tail)
    assert out["schema_applies"], out
    assert out["forward_verified"]
    assert out["acid"] == AUDIT.canon(acid)
    assert out["diol"] == AUDIT.canon(diol)


@pytest.mark.parametrize(
    "tail,amine",
    [
        # the C17 homologue used by L01, L02 and seven others
        ("[C-]#[N+]CCCCCCCCC=CCCCCCCC", "CCCCCCCC=CCCCCCCCCN"),
        # true oleyl, AGILE C5, used by fourteen candidates
        ("[C-]#[N+]CCCCCCCCC=CCCCCCCCC", "CCCCCCCCC=CCCCCCCCCN"),
        # saturated, L03
        ("[C-]#[N+]CCCCCCCCCCCCCC", "CCCCCCCCCCCCCCN"),
        # long saturated, L27
        ("[C-]#[N+]CCCCCCCCCCCCCCCCCCCC", "CCCCCCCCCCCCCCCCCCCCN"),
    ],
)
def test_known_isocyanide_routes_recover_the_packet_precursors(tail, amine):
    out = AUDIT.isocyanide_skeleton(tail)
    assert out["schema_applies"], out
    assert out["forward_verified"]
    assert out["amine"] == AUDIT.canon(amine)


# --------------------------------------------------- the three bugs, one test each


def test_multi_product_outcomes_keep_their_pairing():
    """Flattening a two-product disconnection destroys the acid/diol correspondence."""
    outcomes = AUDIT.run_reaction(AUDIT.CLEAVE_ESTER, "CCCCCCCCCCC(=O)OCCCCCC=O")
    assert outcomes, "ester cleavage produced nothing"
    assert all(isinstance(group, tuple) for group in outcomes)
    assert any(len(group) == 2 for group in outcomes), "the pair was flattened"


def test_recovered_primary_amine_is_neutral():
    """Reversing [C-]#[N+] hands back [NH3+] unless the charge is cleared."""
    amines = AUDIT.single_products("[C-]#[N+:1]>>[N:1]", "[C-]#[N+]CCCCCCCC")
    assert amines
    for smiles in amines:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert all(atom.GetFormalCharge() == 0 for atom in mol.GetAtoms()), smiles
        assert mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2]")), smiles


def test_neutralisation_preserves_a_legitimate_isocyanide():
    """The fix for the charged amine must not strip [N+]#[C-] from a real isocyanide."""
    mol = Chem.MolFromSmiles("[C-]#[N+]CCCCCCCC")
    kept = AUDIT._neutralise(Chem.Mol(mol))
    Chem.SanitizeMol(kept)
    assert kept.HasSubstructMatch(AUDIT.ISOCYANIDE), Chem.MolToSmiles(kept)
    charges = sorted(atom.GetFormalCharge() for atom in kept.GetAtoms() if atom.GetFormalCharge())
    assert charges == [-1, 1]


# ------------------------------------------------------------------ scope guards


def test_non_ester_aldehyde_is_reported_as_out_of_schema_not_as_a_failure():
    """Roughly half the declared aldehyde family is non-ester; that is support, not breakage.

    The training corpus contains 51 non-ester aldehyde components against 56 ester-linked ones,
    and the qualified reaction requires only [CX3H1]=[OX1] of this role. The Tail A preparation
    covers the ester subclass, so a non-ester aldehyde must be reported as outside that schema
    with the reason named, never as a route failure.
    """
    out = AUDIT.aldehyde_skeleton("CCCCCCCCCCCC=O")
    assert not out["schema_applies"]
    assert out["reason"] == "no ester linkage"


def test_a_molecule_that_is_not_an_isocyanide_is_rejected():
    out = AUDIT.isocyanide_skeleton("CCCCCCCCN")
    assert not out["schema_applies"]
    assert out["reason"] == "not an isocyanide"


def test_unparseable_input_does_not_raise():
    assert AUDIT.aldehyde_skeleton("not-a-molecule")["reason"] == "unparseable"
    assert AUDIT.isocyanide_skeleton("not-a-molecule")["reason"] == "unparseable"


def test_derivations_are_deterministic():
    tail = "CCCCCCCCCCC(=O)OCCCCCC=O"
    iso = "[C-]#[N+]CCCCCCCCC=CCCCCCCC"
    assert [AUDIT.aldehyde_skeleton(tail) for _ in range(3)][0] == AUDIT.aldehyde_skeleton(tail)
    assert [AUDIT.isocyanide_skeleton(iso) for _ in range(3)][0] == AUDIT.isocyanide_skeleton(iso)
