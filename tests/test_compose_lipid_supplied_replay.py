"""Explicit precursor roles and quantities must survive source-conditioned replay."""

import json
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import RegistryAssemblyAdapter
from forge.assembly.repeated_components import RepeatBounds
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_supplied_replay import replay_supplied, resolve_components

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/vendor/qualified_michael_source_program_v1.json"


@pytest.fixture
def inputs():
    reaction = json.loads(REGISTRY.read_text())["reactions"][0]
    adapter = RegistryAssemblyAdapter.from_registry(
        REGISTRY, reaction_id=reaction["reaction_id"], expected_sha256=str(sha256_file(REGISTRY))
    )
    return (
        adapter,
        reaction["source_program"],
        {
            "registry_to_source_roles": {
                "amine_head": "amine_head",
                "acceptor_tail": "disulfide_acceptor",
            }
        },
    )


def item(product="CNC(CC(=O)OCCSSCC)=O", occupancy=1):
    # Product default is intentionally not the Michael product: target matching may not prune sites.
    return {
        "source": {"constitution": product, "primary_metadata": {"occupancy": occupancy}},
        "preparation": {
            "eligible_for_program_preparation": True,
            "component_instances": [
                ["amine_head", "head", 1],
                ["disulfide_acceptor", "tail", occupancy],
            ],
        },
    }


STRUCTURES = {"head": "CN", "tail": "C=CC(=O)OCCSSCC"}


def test_source_disulfide_acceptor_keeps_whole_structure_and_repeat_count(inputs):
    adapter, program, binding = inputs
    supplied = item("CN(CCC(=O)OCCSSCC)CCC(=O)OCCSSCC", 2)
    result = replay_supplied(adapter, supplied, STRUCTURES, binding, program, RepeatBounds())
    assert result["computed_consistency_pass"]
    assert result["events"] == 2
    assert result["checks"]["full_element_hydrogen_charge_balance"]
    assert not result["experimental_selectivity_qualified"]


def test_misassigned_sites_cannot_be_rescued_by_matching_target(inputs):
    adapter, program, binding = inputs
    result = replay_supplied(
        adapter,
        item("CN(CCC(=O)OCCSSCC)CCC(=O)OCCSSCC", 2),
        {**STRUCTURES, "head": "NCCN"},
        binding,
        program,
        RepeatBounds(),
    )
    assert not result["computed_consistency_pass"]
    assert len(result["forward_layers"][-1]) > 1


def test_disulfide_arms_cannot_be_shortened_to_a_simpler_acceptor(inputs):
    adapter, program, binding = inputs
    result = replay_supplied(
        adapter,
        item("CNCCC(=O)OCCSSCC"),
        {**STRUCTURES, "tail": "C=CC(=O)OCC"},
        binding,
        program,
        RepeatBounds(),
    )
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["full_element_hydrogen_charge_balance"]


def test_missing_occupancy_metadata_uses_explicit_quantities_not_product_inference(inputs):
    adapter, program, binding = inputs
    supplied = item("CNCCC(=O)OCCSSCC")
    supplied["source"]["primary_metadata"] = {}
    assert replay_supplied(adapter, supplied, STRUCTURES, binding, program, RepeatBounds())[
        "computed_consistency_pass"
    ]


@pytest.mark.parametrize("value", [0, 2, True, 1.0])
def test_metadata_cannot_override_instance_quantity(inputs, value):
    _, program, binding = inputs
    with pytest.raises(ComposeLipidError, match="event metadata"):
        resolve_components(
            item()["preparation"]["component_instances"],
            STRUCTURES,
            binding["registry_to_source_roles"],
            program,
            {"occupancy": value},
        )


@pytest.mark.parametrize(
    "instances",
    [
        [["amine_head", "head", 2], ["disulfide_acceptor", "tail", 1]],
        [["amine_head", "head", 1]],
        [
            ["amine_head", "head", 1],
            ["disulfide_acceptor", "tail", 1],
            ["disulfide_acceptor", "head", 1],
        ],
        [["amine_head", "head", 1], ["disulfide_acceptor", "unknown", 1]],
    ],
)
def test_incomplete_or_ambiguous_role_assignments_fail_loudly(inputs, instances):
    _, program, binding = inputs
    with pytest.raises(ComposeLipidError):
        resolve_components(instances, STRUCTURES, binding["registry_to_source_roles"], program, {})


def test_bound_saturation_cannot_admit_an_early_matching_target(inputs):
    adapter, program, binding = inputs
    result = replay_supplied(
        adapter,
        item("CNCCC(=O)OCCSSCC"),
        STRUCTURES,
        binding,
        program,
        RepeatBounds(maximum_outcomes=1),
    )
    assert result["bound_reasons"]
    assert not result["computed_consistency_pass"]


def test_protected_target_is_rejected_before_parsing_product(inputs):
    adapter, program, binding = inputs
    supplied = item("invalid product")
    supplied["preparation"]["eligible_for_program_preparation"] = False
    with pytest.raises(ComposeLipidError, match="Protected target"):
        replay_supplied(adapter, supplied, STRUCTURES, binding, program, RepeatBounds())


def test_excess_event_count_abstains_without_silently_truncating(inputs):
    adapter, program, binding = inputs
    result = replay_supplied(
        adapter,
        item("invalid product", program["maximum_events"] + 1),
        STRUCTURES,
        binding,
        program,
        RepeatBounds(),
    )
    assert result["disposition"] == "outside_registry_event_support"
    assert not result["computed_consistency_pass"]


@pytest.mark.parametrize(
    "instances",
    [
        [["amine_nucleophile", "head", 1], ["conjugated_carbonyl_acceptor", "tail", 2]],
        [["amine_head", "head", 1], ["amine_head", "other", 1], ["disulfide_acceptor", "tail", 2]],
    ],
)
def test_other_source_programs_abstain_without_guessing_role_aliases(inputs, instances):
    adapter, program, binding = inputs
    supplied = item("invalid product")
    supplied["preparation"]["component_instances"] = instances
    result = replay_supplied(adapter, supplied, STRUCTURES, binding, program, RepeatBounds())
    assert result["disposition"] == "unsupported_source_role_tuple"
    assert not result["computed_consistency_pass"]
