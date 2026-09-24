"""Legacy split features stay separate from actual complete-component inventories."""

import copy
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_historical_morphology import HistoricalMorphology

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def historical():
    path = ROOT / "data/vendor/compose_lipid_historical_morphology_policy_v1.json"
    return HistoricalMorphology.from_registry(ROOT, path, expected_sha256=sha256_file(path))


def describe(historical, **changes):
    args = dict(
        family="aema_aza_thiol_addition",
        metadata={},
        source_anchor=False,
        instances=[
            ["AEMA", "full-aema", 2],
            ["amine_core", "head", 1],
            ["thiol_periphery", "tail", 2],
        ],
        corrected_description={
            "regional_profile": {"heavy_atom_band": 6},
            "morphology_context": {},
        },
    )
    args.update(changes)
    return historical.describe(**args)


def test_original_virtual_partition_roles_do_not_rewrite_component_inventory(historical):
    instances = [["AEMA", "aema", 2], ["amine_core", "head", 1], ["thiol_periphery", "tail", 2]]
    before = copy.deepcopy(instances)
    first = describe(historical, instances=instances)
    # The original virtual partition descriptor did not encode the now-complete AEMA inventory.
    assert first == describe(historical, instances=[["unchanged_source_role", "other", 3]])
    assert instances == before


def test_original_source_anchor_uses_full_role_multiplicity(historical):
    first = describe(historical, source_anchor=True, instances=[["head", "a", 1], ["tail", "b", 2]])
    assert first != describe(
        historical, source_anchor=True, instances=[["head", "a", 1], ["tail", "b", 1]]
    )
    assert first == describe(
        historical, source_anchor=True, instances=[["tail", "other", 2], ["head", "new", 1]]
    )


@pytest.mark.parametrize("value", [None, 0, True, -1])
def test_missing_or_invalid_historical_event_quantity_fails(historical, value):
    with pytest.raises(ComposeLipidError):
        describe(historical, family="a3_amine_aldehyde_alkyne", metadata={"events": value})


def test_historical_event_quantity_affects_partition(historical):
    assert describe(
        historical, family="a3_amine_aldehyde_alkyne", metadata={"events": 1}
    ) != describe(historical, family="a3_amine_aldehyde_alkyne", metadata={"events": 2})


def test_missing_historical_family_fails(historical):
    with pytest.raises(ComposeLipidError):
        describe(historical, family="unknown")
