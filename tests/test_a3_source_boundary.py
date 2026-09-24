"""A carbonyl H does not make a carboxylic-acid or amide carbonyl an aldehyde."""

import json
from pathlib import Path

import pytest

from forge.assembly.families import RegistryAssemblyAdapter
from forge.assembly.repeated_components import replay_repeated_components
from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def adapter(version):
    path = ROOT / f"data/vendor/qualified_a3_source_program_v{version}.json"
    return RegistryAssemblyAdapter.from_registry(
        path,
        reaction_id="a3_amine_aldehyde_terminal_alkyne",
        expected_sha256=str(sha256_file(path)),
    )


@pytest.mark.parametrize("carbonyl", ["O=CO", "O=COC", "O=CN", "O=CS"])
def test_non_aldehyde_carbonyls_are_explicitly_excluded(carbonyl):
    components = {"amine_head": "CNC", "aldehyde": carbonyl, "alkyne": "C#CC"}
    # Preserve the original failure as a reproducible negative control.
    assert adapter(1).forward_products(components).products
    current = adapter(2)
    assert not next(r for r in current.assess_roles(components) if r.role == "aldehyde").qualified
    assert not current.forward_products(components).products


@pytest.mark.parametrize("aldehyde", ["C=O", "CC=O", "O=Cc1ccccc1", "O=CCC(=O)O"])
def test_true_aldehydes_keep_their_original_programs(aldehyde):
    components = {"amine_head": "CNC", "aldehyde": aldehyde, "alkyne": "C#CC"}
    before = adapter(1).forward_products(components)
    after = adapter(2).forward_products(components)
    assert len(after.products) == 1
    assert before == after


@pytest.mark.parametrize("index", [0, 1])
def test_narrowed_query_preserves_both_independent_source_controls(index):
    source = ROOT / "results/phase1/compose_lipid_v8_a3_source_v1/adjudication-v2.json"
    control = json.loads(source.read_text())["source_controls"][index]
    registry = json.loads((ROOT / "data/vendor/qualified_a3_source_program_v2.json").read_text())
    result = replay_repeated_components(
        adapter(2),
        control["components"],
        control["expected_product"],
        accumulator_role="amine_head",
        events=control["events"],
        byproducts_per_event=registry["curation"]["byproducts_per_event"],
    )
    assert result["computed_consistency_pass"]
