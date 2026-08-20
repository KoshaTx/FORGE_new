"""Regression tests for the aldehyde chemotype classifier.

The classifier decides which generated chemistry counts as absent from the measured library, so
a silent misclassification would move the headline support-gap numbers. These pin the label of
a representative structure from every category, drawn from the audit's own inspection sample.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _audit():
    path = REPO / "experiments/archive/producers/phase1_aldehyde_chemotype_audit_v1.py"
    spec = importlib.util.spec_from_file_location("forge_chemotype_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["forge_chemotype_audit"] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _audit()
PAT = AUDIT.compile_patterns()


def label(smiles: str) -> str:
    return AUDIT.label_aldehyde(smiles, PAT, {})["exclusive"]


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # the measured chemotype: acid esterified onto a diol, then oxidised
        ("CCCCCCCCCCC(=O)OCCCCCC=O", "ester_linked"),
        ("CCCCCCCCCC(=O)OCCOCCC=O", "ester_linked"),  # ester wins over a co-occurring ether
        ("CCCCCCCCCC(=O)CCCOCCC=O", "ketone_and_ether"),
        ("CCCCCCCCCCCC(=O)CCCCC=O", "ketone_containing"),
        ("CCCCCCCOCCCC=O", "ether_linked"),
        ("CCCCCCCCCCCCCCCCCCCCCC=O", "unlinked_chain_aldehyde"),
        ("C#CCCCCCCCC=CCCCCCCCCC=O", "unlinked_chain_aldehyde"),  # unsaturation is not a linkage
    ],
)
def test_chemotype_labels_are_stable(smiles, expected):
    assert label(smiles) == expected


def test_precedence_is_declared_and_covers_every_label():
    produced = {
        label(s)
        for s in (
            "CCCCCCCCCCC(=O)OCCCCCC=O",
            "CCCCCCCCCC(=O)CCCOCCC=O",
            "CCCCCCCCCCCC(=O)CCCCC=O",
            "CCCCCCCOCCCC=O",
            "CCCCCCCCCCCCCCCCCCCCCC=O",
        )
    }
    assert produced <= set(AUDIT.PRECEDENCE)
    assert len(AUDIT.PRECEDENCE) == len(set(AUDIT.PRECEDENCE))


def test_ester_takes_precedence_over_every_other_motif():
    """A component carrying both must land in the measured chemotype, not a novel one."""
    assert label("CCCCCCCCCC(=O)OCCOCCC=O") == "ester_linked"
    assert label("CCCCCCC(=O)OCCC(=O)CCC=O") == "ester_linked"


def test_unsaturation_does_not_create_a_linkage_label():
    for smiles in ("CCCCCC=CC=CCCCCCCCCCCCCCC=O", "C#CCCC=CCCCCC=O"):
        assert label(smiles) == "unlinked_chain_aldehyde"


def test_multi_label_motifs_are_reported_alongside_the_exclusive_label():
    out = AUDIT.label_aldehyde("CCCCCCCCCC(=O)CCCOCCC=O", PAT, {})
    assert out["exclusive"] == "ketone_and_ether"
    assert set(out["motifs"]) >= {"ether", "ketone"}


def test_unparseable_input_is_labelled_not_raised():
    assert label("not-a-molecule") == "unparsed"


def test_smarts_definitions_are_recorded_in_the_module():
    assert AUDIT.SMARTS["ester"] == "[CX3](=[OX1])[OX2][CX4]"
    for name in ("ether", "ketone", "alkene", "alkyne"):
        assert AUDIT.SMARTS[name]
