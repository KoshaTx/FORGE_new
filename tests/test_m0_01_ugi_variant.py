"""M0-01: the assembly chemistry is the AGILE-type Ugi three-component reaction.

Guards the settled decision (docs/DECISION_LOG.md, 2026-07-29) against silent drift. These assertions
read the vendored registry, which is the chemistry source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "data" / "vendor"
CONFIG = REPO / "configs" / "assembly" / "ugi_variant.yaml"

pytestmark = pytest.mark.needs_vendor


@pytest.fixture(scope="module")
def ugi_reaction() -> dict:
    path = VENDOR / "qualified_reactions_v1.json"
    if not path.exists():
        pytest.skip("run `make vendor` first")
    reactions = json.loads(path.read_text())["reactions"]
    match = [r for r in reactions if r["reaction_id"] == "ugi_3cr_agile"]
    assert match, "ugi_3cr_agile missing from the qualified reaction registry"
    return match[0]


def test_config_declares_agile_3cr_without_carboxylic_acid_component() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["variant"] == "ugi_3cr_agile"
    assert cfg["expected_reactant_count"] == 3
    assert cfg["ester_origin"] == "aldehyde_component"
    assert cfg["carboxylic_acid_reactant_component"] is False
    assert (
        cfg["compatibility"]["amine_head_site_multiplicity_semantics"]
        == "symmetry_distinct_required_handle_matches"
    )
    assert cfg["compatibility"]["raw_required_handle_match_count"] == "diagnostic_only"
    assert cfg["compatibility"]["deduplicate_forward_products"] is True
    assert (
        cfg["compatibility"][
            "multiple_unique_products_require_explicit_site_selection"
        ]
        is True
    )


def test_smarts_has_exactly_three_reactants(ugi_reaction: dict) -> None:
    """Three components, not four. A 4CR would add a carboxylic acid reactant."""
    smarts = ugi_reaction["atom_mapped_reaction_smarts"]
    reactants = smarts.split(">>")[0].split(".")
    assert len(reactants) == 3, f"expected 3 reactants, got {len(reactants)}: {reactants}"


def test_no_carboxylic_acid_reactant(ugi_reaction: dict) -> None:
    roles = {r["name"] for r in ugi_reaction["reactant_roles"]}
    assert roles == {"amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"}
    assert not any("acid" in r for r in roles), f"carboxylic acid role present: {roles}"


def test_selectivity_policy_says_three_component(ugi_reaction: dict) -> None:
    policy = ugi_reaction["selectivity_policy"].lower()
    assert "three-component" in policy


def test_product_is_alpha_amino_not_acylamino(ugi_reaction: dict) -> None:
    """Ugi-3CR gives an alpha-amino amide; Ugi-4CR gives an alpha-ACYLamino amide."""
    product = ugi_reaction["atom_mapped_reaction_smarts"].split(">>")[1]
    assert product == "[N:1][CH1:2][C+0:3](=O)[NH1+0:4]", product


@pytest.mark.parametrize("field", ["known_positive_examples", "known_negative_examples"])
def test_registry_carries_worked_examples(ugi_reaction: dict, field: str) -> None:
    """Positive AND negative examples are what make the registry a usable verifier."""
    assert ugi_reaction.get(field), f"{field} absent — cannot qualify the transform"
