"""Both sides of an enumeration-membership comparison must be stereo-free.

This guards a bug that actually happened. The AGILE virtual enumeration retains
stereochemistry on 3,652 of its 12,276 products; the FORGE pipeline is constitutional and
emits none. A raw canonical-SMILES comparison therefore records every stereo-bearing library
member as chemistry the library never contained. In the first pass of the open-world funnel
that moved 798 designs from inside to outside and inflated the out-of-enumeration share from
87.6% to 90.3%.

It was caught only because the funnel reported 38 of 40 frozen candidates outside the
enumeration while the frozen panel's own novelty flag says 32. These tests pin the convention
so the two cannot drift apart again silently.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest
from rdkit import Chem, rdBase

REPO = Path(__file__).resolve().parents[1]
AGILE = REPO / "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz"
PANEL = REPO / "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"

pytestmark = pytest.mark.skipif(
    not (AGILE.exists() and PANEL.exists()),
    reason="frozen enumeration or panel artifact is unavailable",
)


def stereo_free(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule)


@pytest.fixture(scope="module")
def enumeration() -> dict[str, set[str]]:
    with gzip.open(AGILE, "rt", newline="") as handle:
        raw = {row["canonical_product_smiles"] for row in csv.DictReader(handle)}
    with rdBase.BlockLogs():
        free = {value for value in (stereo_free(s) for s in raw) if value}
    return {"raw": raw, "stereo_free": free}


@pytest.fixture(scope="module")
def panel() -> list[dict]:
    return [json.loads(line) for line in gzip.open(PANEL, "rt")]


def test_the_enumeration_actually_carries_stereochemistry(enumeration):
    """If this ever became false, the bug would be invisible rather than fixed."""
    with rdBase.BlockLogs():
        bearing = sum(
            1
            for smiles in enumeration["raw"]
            if (molecule := Chem.MolFromSmiles(smiles)) is not None
            and stereo_free(smiles) != Chem.MolToSmiles(molecule)
        )
    assert bearing > 0, "the enumeration no longer carries stereochemistry"
    assert bearing == 3652, f"stereo-bearing count moved from 3652 to {bearing}"


def test_stripping_stereo_does_not_collapse_distinct_enumeration_members(enumeration):
    """Stereo-free comparison is only safe if it does not merge library products."""
    assert len(enumeration["stereo_free"]) == len(enumeration["raw"]) == 12276


def test_panel_novelty_flag_matches_stereo_free_membership(enumeration, panel):
    inside = sum(
        1
        for record in panel
        if stereo_free(record["canonical_product"]) in enumeration["stereo_free"]
    )
    flagged = sum(1 for record in panel if not record["novelty"]["absent_from_agile_library"])
    assert inside == flagged == 8


def test_raw_membership_disagrees_and_is_the_wrong_convention(enumeration, panel):
    """The failing convention, pinned so a regression reads as a change and not as noise."""
    raw_inside = sum(1 for record in panel if record["canonical_product"] in enumeration["raw"])
    assert raw_inside == 2
    disagreeing = sorted(
        record["candidate_id"]
        for record in panel
        if (record["canonical_product"] in enumeration["raw"])
        != (not record["novelty"]["absent_from_agile_library"])
    )
    assert disagreeing == ["P01", "P03", "P05", "P06", "P08", "P12"]


def test_forge_products_carry_no_stereochemistry(panel):
    """The asymmetry is one-sided: stripping our own side must be a no-op."""
    with rdBase.BlockLogs():
        for record in panel:
            smiles = record["canonical_product"]
            molecule = Chem.MolFromSmiles(smiles)
            assert molecule is not None
            assert stereo_free(smiles) == Chem.MolToSmiles(molecule), smiles
