"""Exact source identity extensions must not open an unrestricted N/O domain."""

import json
from pathlib import Path

import pytest

from forge.corpus.compose_lipid_zhou_replay import component_domain, load_contract, replay_record
from tests.test_compose_lipid_aema_replay import prepared

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def contract():
    return load_contract(
        ROOT, ROOT / "results/phase1/compose_lipid_user_supplements_v1/zhou-replay-config.json"
    )


def test_drawn_six_arm_aminothiol_control_preserves_nitrogen(contract):
    source = json.loads(contract[1]["source_control"].read_text())
    item, structures = prepared(source)
    result = replay_record(item, structures, contract[2])
    assert result["computed_consistency_pass"]
    assert result["balance"]["reactants"]["N"] == 10
    assert result["balance"]["product_and_net_byproducts"]["N"] == 10
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    "core,tail,expected",
    [
        ("NCCCn1ccnc1", "CCCCCCS", True),
        ("CN(CCCN)CCCN", "OCCCCCCCCCCCS", True),
        ("CN(CCCN)CCCN", "SCCc1ccccc1", True),
        ("CN(CCCN)CCCN", "CCCCOC(=O)CCCCS", False),
        ("CN(CCCN)CCCN", "NCCCCCCS", False),
        ("NCCc1ccccc1", "CCCCCCS", False),
    ],
)
def test_exact_source_extensions_do_not_admit_unreviewed_functionality(
    contract, core, tail, expected
):
    assert (
        component_domain({"amine_core": core, "thiol_periphery": tail}, contract[2][4]) is expected
    )


def test_fed_excess_cannot_add_branches(contract):
    source = json.loads(contract[1]["source_control"].read_text())
    item, structures = prepared(source)
    for row in item["preparation"]["component_instances"]:
        if row[0] == "thiol_periphery":
            row[2] = 7
    assert (
        replay_record(item, structures, contract[2])["disposition"]
        == "outside_qualified_program_multiplicity"
    )
