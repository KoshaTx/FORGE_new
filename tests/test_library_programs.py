from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.assembly.families import LibraryAssemblyError, load_assembly_libraries
from forge.assembly.library_programs import (
    LibraryProgramLimits,
    recover_library_programs,
    replay_library_program,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def libraries():
    config = json.loads(
        (REPO / "configs/multireaction/library_program_dataset_v1.json").read_text()
    )
    return load_assembly_libraries(
        [
            (REPO / config["inputs"][k]["path"], config["inputs"][k]["sha256"])
            for k in config["registries"]
        ],
        expected_families=config["programs"],
    )


@pytest.mark.parametrize("family", ["aza_michael_amine_acrylate", "amide_coupling_acid_amine"])
def test_repeated_recovery_and_every_step_replay_with_either_accumulator_position(
    libraries, family
):
    adapter = libraries[family]
    raw = next(
        r
        for r in json.loads(adapter.registry_path.read_text())["reactions"]
        if r["reaction_id"] == family
    )
    components = dict(zip(adapter.roles, raw["known_positive_examples"][0]["reactants"]))
    components["amine_head"] = "NCCN"
    first = adapter.forward_products(components).products[0]
    second = adapter.forward_products({**components, "amine_head": first}).products[0]
    result = recover_library_programs(
        adapter,
        components,
        [first, second],
        accumulator_role="amine_head",
        limits=LibraryProgramLimits(3),
    )
    assert sorted(t.minimum_steps for t in result.targets) == [1, 2]
    for target in result.targets:
        assert target.qualified_unique_program
        assert replay_library_program(
            adapter, components, target.policy_intermediate_products, accumulator_role="amine_head"
        )
    assert not replay_library_program(
        adapter, components, [first, "CC"], accumulator_role="amine_head"
    )


def test_transform_match_outside_registry_policy_never_admitted(libraries):
    adapter = libraries["aza_michael_amine_acrylate"]
    from forge.assembly.library_programs import _layer

    components = dict(zip(adapter.roles, ["NCCNCCN", "C=CC(=O)OCC"]))
    products, saturated, policy = _layer(adapter, components, 256)
    assert products and not saturated and not policy
    result = recover_library_programs(
        adapter, components, products, accumulator_role="amine_head", limits=LibraryProgramLimits(1)
    )
    assert all(t.exact and not t.qualified_unique_program for t in result.targets)
    assert all(t.policy_path_count_capped_at_two == 0 for t in result.targets)


def test_ambiguous_ordered_constitutional_paths_are_not_silently_chosen(libraries, monkeypatch):
    from forge.assembly import library_programs as module

    adapter = libraries["aza_michael_amine_acrylate"]
    # Two distinct intermediates converge. Reactive-site duplicates within a step are already
    # collapsed; this tests genuine sequence ambiguity across different intermediate graphs.
    edges = {"N": ("CN", "CCN"), "CN": ("CCCN",), "CCN": ("CCCN",)}
    monkeypatch.setattr(
        module, "_layer", lambda a, c, n: (edges.get(c["amine_head"], ()), False, True)
    )
    result = recover_library_programs(
        adapter,
        dict(zip(adapter.roles, ["N", "C=CC(=O)OCC"])),
        ["CCCN"],
        accumulator_role="amine_head",
        limits=LibraryProgramLimits(2),
        append_only_size_bound=False,
    )
    target = result.targets[0]
    assert target.exact and target.minimum_steps == 2
    assert target.minimum_path_count_capped_at_two == target.policy_path_count_capped_at_two == 2
    assert not target.qualified_unique_program


@pytest.mark.parametrize(
    "limit,reason",
    [
        ({"maximum_states": 1}, "state_limit"),
        ({"maximum_expansions": 1}, "expansion_limit"),
        ({"maximum_outcomes": 1}, "outcome_limit"),
    ],
)
def test_incomplete_layer_never_admits_a_target(libraries, monkeypatch, limit, reason):
    from forge.assembly import library_programs as module

    adapter = libraries["aza_michael_amine_acrylate"]

    def layer(a, c, n):
        if c["amine_head"] == "N":
            return ("CN",), False, True
        return ("CCN", "CCCN"), n == 1, True

    monkeypatch.setattr(module, "_layer", layer)
    result = recover_library_programs(
        adapter,
        dict(zip(adapter.roles, ["N", "C=CC(=O)OCC"])),
        ["CN", "CCN"],
        accumulator_role="amine_head",
        limits=LibraryProgramLimits(3, **limit),
        append_only_size_bound=False,
    )
    targets = {t.product_smiles: t for t in result.targets}
    assert targets["CN"].qualified_unique_program
    assert not targets["CCN"].exact
    assert result.completed_depth == 1 and result.termination == reason


def test_size_bound_preserves_recovered_paths_and_does_not_prune_sites(libraries):
    from dataclasses import asdict

    adapter = libraries["urea_amine_isocyanate"]
    components = dict(zip(adapter.roles, ["CN", "CCN=C=O"]))
    first = adapter.forward_products(components).products[0]
    targets = [first, "CC"]
    slow, fast = (
        recover_library_programs(
            adapter,
            components,
            targets,
            accumulator_role="amine_head",
            limits=LibraryProgramLimits(4),
            append_only_size_bound=flag,
        )
        for flag in (False, True)
    )

    def comparable(result):
        return [
            {k: v for k, v in asdict(t).items() if k != "unresolved_reason"} for t in result.targets
        ]

    assert comparable(slow) == comparable(fast)
    assert fast.expansions < slow.expansions
    assert fast.termination == "append_only_frontier_above_targets"


@pytest.mark.parametrize("invalid", [0, -1, True, 2.5])
def test_limits_must_be_positive_integers(invalid):
    with pytest.raises(LibraryAssemblyError):
        LibraryProgramLimits(invalid)


def test_fixed_arity_contract_and_wrong_accumulator_fail(libraries):
    adapter = libraries["ugi_3cr_agile"]
    components = dict(zip(adapter.roles, ["CN", "CCC=O", "CC[N+]#[C-]"]))
    for role, steps in [(None, 2), ("amine_head", 2)]:
        with pytest.raises(LibraryAssemblyError):
            recover_library_programs(
                adapter,
                components,
                ["CC"],
                accumulator_role=role,
                limits=LibraryProgramLimits(steps),
            )
