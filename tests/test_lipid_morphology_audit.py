from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from forge.product.lipid_morphology_audit import molecule_morphology  # noqa: E402


def test_morphology_recovers_path_like_hydrophobic_arms() -> None:
    molecule = Chem.MolFromSmiles("CCCCCCCCOC(=O)CN(CCO)CCCCCCCC")
    assert molecule is not None

    result = molecule_morphology(molecule)

    assert result["maximum_root_distance"] >= 8
    assert len(result["tail_components"]) == 2
    assert all(component["path_like"] for component in result["tail_components"])
    assert result["oxygen_environments"]["carbonyl"] == 1
    assert result["oxygen_environments"]["single_bond_bridge"] == 1
    assert result["oxygen_environments"]["terminal_single"] == 1


def test_morphology_detects_tail_junction_without_banning_it() -> None:
    molecule = Chem.MolFromSmiles("NCCCCCCCC(CCCC)(CCCC)CCCC")
    assert molecule is not None

    result = molecule_morphology(molecule)

    assert any(component["junction_atoms"] > 0 for component in result["tail_components"])
