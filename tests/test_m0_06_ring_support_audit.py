from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.design.audit.ring_support_audit import _ring_signature, sha256_file

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("smiles", "expected"),
    (
        (
            "CCCCN",
            {
                "ring_count": 0,
                "ring_sizes": [],
                "macrocycle": False,
                "fused": False,
                "spiro": False,
            },
        ),
        (
            "N1CCCCC1",
            {
                "ring_count": 1,
                "ring_sizes": [6],
                "macrocycle": False,
                "fused": False,
                "spiro": False,
            },
        ),
        (
            "c1ccc2ccccc2c1",
            {
                "ring_count": 2,
                "ring_sizes": [6, 6],
                "macrocycle": False,
                "fused": True,
                "spiro": False,
            },
        ),
        (
            "C1CCC2(CC1)CCCC2",
            {
                "ring_count": 2,
                "ring_sizes": [5, 6],
                "macrocycle": False,
                "fused": False,
                "spiro": True,
            },
        ),
        (
            "C1CCCCCCCCCCC1",
            {
                "ring_count": 1,
                "ring_sizes": [12],
                "macrocycle": True,
                "fused": False,
                "spiro": False,
            },
        ),
    ),
)
def test_ring_signature_distinguishes_lipid_topology_classes(
    smiles: str,
    expected: dict[str, object],
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    signature = _ring_signature(molecule, macrocycle_minimum=9)

    for field, value in expected.items():
        assert signature[field] == value


@pytest.mark.needs_vendor
def test_frozen_ring_support_result_contract_when_present() -> None:
    result_path = REPO / "results/m0_06_ring_support/result.json"
    if not result_path.exists():
        pytest.skip("ring-support audit has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_06_lipid_ring_support_result.v1"
    assert result["status"] == "completed_lipid_native_ring_support_audit"
    broad = result["broad_r0"]
    assert broad["records"] == 15_229
    assert broad["cyclic_records"] == 8_010
    assert broad["macrocycle_records"] == 477
    assert broad["fused_ring_records"] == 608
    assert broad["spiro_ring_records"] == 41
    assert broad["bridgehead_ring_records"] == 533
    assert broad["maximum_ring_size"] == 15
    assert broad["aromatic_ring_records"] == 1_891

    agile = result["reconciled_agile_ugi"]
    assert agile["product"]["records"] == 1_100
    assert agile["product"]["cyclic_records"] == 660
    assert agile["product"]["ring_size_instance_counts"] == {"5": 275, "6": 385}
    assert agile["product"]["multiple_ring_records"] == 0
    assert agile["product"]["macrocycle_records"] == 0
    assert agile["product"]["fused_ring_records"] == 0
    assert agile["product"]["spiro_ring_records"] == 0
    assert agile["product"]["bridgehead_ring_records"] == 0
    assert agile["components"]["aldehyde_component"]["cyclic_records"] == 0
    assert agile["components"]["isocyanide_component"]["cyclic_records"] == 0
    assert agile["ring_localization_counts"] == {"acyclic": 440, "amine_head": 660}
    assert agile["product_ring_sizes_match_amine_head_rows"] == 1_100

    decision = result["decision"]
    assert decision["primary_ugi_automatic_ring_gate_supported"]
    assert not decision["exact_ring_fragment_vocabulary_required"]
    assert not decision["primary_ugi_automatic_support"]["macrocycle_allowed"]

    for record in result["inputs"].values():
        path = REPO / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
    config_path = REPO / result["configuration"]["path"]
    assert sha256_file(config_path) == result["configuration"]["sha256"]
