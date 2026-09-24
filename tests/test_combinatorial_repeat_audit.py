from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.phase1.multireaction.combinatorial_repeat_audit import summarize
from experiments.phase1.multireaction.combinatorial_repeat_audit_verify import verify
from forge.assembly.families import LibraryAssemblyError, load_assembly_libraries
from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_programs import LibraryProgramLimits
from forge.assembly.library_repeat_audit import (
    DiagnosticStep,
    audit_repeated_program,
    classify_repeat_attempt,
    replay_diagnostic_steps,
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


def _two_steps(adapter, *, mixed):
    # Small mechanical fixtures, not substitute corpus or source-execution evidence.
    first_inputs = dict(zip(adapter.roles, ["NCCN", "C=CC(=O)OCC"]))
    first = adapter.forward_products(first_inputs).products[0]
    tail_role = next(r for r in adapter.roles if r != "amine_head")
    second_inputs = {**first_inputs, "amine_head": first}
    if mixed:
        second_inputs[tail_role] = "C=CC(=O)OCCCC"
    second = adapter.forward_products(second_inputs).products[0]
    return second, (
        DiagnosticStep(tuple(sorted(first_inputs.items())), first),
        DiagnosticStep(tuple(sorted(second_inputs.items())), second),
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_mixed_diagnostic_is_separate_from_identical_repeat_gate(libraries, mixed):
    adapter = libraries["aza_michael_amine_acrylate"]
    product, _ = _two_steps(adapter, mixed=mixed)
    original = check_generated_program(
        adapter, product, depth=2, accumulator_role="amine_head", limits=LibraryProgramLimits(2)
    )
    audit = audit_repeated_program(
        adapter, product, depth=2, accumulator_role="amine_head", limits=LibraryProgramLimits(2)
    )
    assert audit["status"] == "complete"
    assert audit["requested_depth_programs"]
    assert original.exact is not mixed
    assert classify_repeat_attempt(original.status, audit) == (
        "different_coreactants_only_at_requested_depth" if mixed else "original_exact_program"
    )
    for program in audit["requested_depth_programs"]:
        assert program["admission_status"] == "diagnostic_only"
        steps = tuple(
            DiagnosticStep(tuple(map(tuple, s["components"])), s["product"])
            for s in program["steps"]
        )
        assert replay_diagnostic_steps(
            adapter, steps, accumulator_role="amine_head", maximum_outcomes=256
        )
    assert not audit["original_gate_changed"]
    assert (
        check_generated_program(
            adapter, product, depth=2, accumulator_role="amine_head", limits=LibraryProgramLimits(2)
        )
        == original
    )


def test_replay_rejects_a_wrong_intermediate_handoff(libraries):
    adapter = libraries["aza_michael_amine_acrylate"]
    _, steps = _two_steps(adapter, mixed=True)
    second = dict(steps[1].components)
    second["amine_head"] = "NCCN"
    corrupt = (steps[0], replace(steps[1], components=tuple(sorted(second.items()))))
    assert not replay_diagnostic_steps(
        adapter, corrupt, accumulator_role="amine_head", maximum_outcomes=256
    )


def test_shorter_path_does_not_satisfy_the_requested_depth(libraries):
    adapter = libraries["aza_michael_amine_acrylate"]
    product = adapter.forward_products(dict(zip(adapter.roles, ["NCCN", "C=CC(=O)OCC"]))).products[
        0
    ]
    audit = audit_repeated_program(
        adapter, product, depth=2, accumulator_role="amine_head", limits=LibraryProgramLimits(2)
    )
    assert audit["verified_path_counts_by_depth"]["1"] > 0
    assert not audit["requested_depth_programs"]
    assert classify_repeat_attempt("no_exact_program", audit) == "only_shorter_policy_valid_program"


def test_role_rejected_raw_transform_stays_rejected(libraries):
    from forge.assembly.library_programs import _layer

    adapter = libraries["aza_michael_amine_acrylate"]
    # The third amine violates the frozen registry multiplicity rule.
    products, _, qualified = _layer(
        adapter, dict(zip(adapter.roles, ["NCCNCCN", "C=CC(=O)OCC"])), 256
    )
    assert products and not qualified
    audit = audit_repeated_program(
        adapter, products[0], depth=1, accumulator_role="amine_head", limits=LibraryProgramLimits(1)
    )
    assert not audit["requested_depth_programs"]
    assert classify_repeat_attempt("no_exact_program", audit) == "role_policy_blocks_first_step"
    assert all(not x["accepted"] for x in audit["dead_end_examples"].values())


@pytest.mark.parametrize(
    "kwargs", [{"maximum_outcomes": 1}, {"maximum_expansions": 1}, {"maximum_states": 1}]
)
def test_incomplete_search_does_not_publish_full_programs(libraries, kwargs):
    adapter = libraries["aza_michael_amine_acrylate"]
    product, _ = _two_steps(adapter, mixed=True)
    audit = audit_repeated_program(
        adapter,
        product,
        depth=2,
        accumulator_role="amine_head",
        limits=LibraryProgramLimits(2, **kwargs),
    )
    assert audit["status"] == "abstain"
    assert not audit["requested_depth_programs"]
    assert classify_repeat_attempt("no_exact_program", audit) == "diagnostic_search_abstained"


@pytest.mark.parametrize("depth", [0, 3, True, 1.5])
def test_invalid_depth_rejected(libraries, depth):
    with pytest.raises(LibraryAssemblyError, match="depth"):
        audit_repeated_program(
            libraries["aza_michael_amine_acrylate"],
            "CC",
            depth=depth,
            accumulator_role="amine_head",
            limits=LibraryProgramLimits(2),
        )


def test_summary_retains_invalid_attempts_and_diagnostic_only_matches():
    def row(status, classification, diagnostic):
        return {
            "arm": "trained",
            "family": "example",
            "requested_depth": 2,
            "original_status": status,
            "classification": classification,
            "diagnostic": diagnostic,
        }

    rows = [
        row("invalid_graph", "invalid_graph", None),
        row(
            "no_exact_program",
            "different_coreactants_only_at_requested_depth",
            {"requested_depth_programs": [{}]},
        ),
    ]
    summary = summarize(rows)[0]
    assert summary["attempts"] == 2 and summary["valid_connected"] == 1
    assert summary["original_exact"] == 0
    assert summary["diagnostic_full_depth_attempts"] == 1
    assert summary["diagnostic_fraction_all_attempts"] == 0.5


def test_disagreement_with_original_gate_is_an_error():
    with pytest.raises(LibraryAssemblyError, match="disagree"):
        classify_repeat_attempt(
            "no_exact_program",
            {"status": "complete", "requested_depth_programs": [{"identical_coreactants": True}]},
        )


def test_saved_audit_recomputes_every_attempt_and_control():
    path = REPO / "results/phase1/combinatorial_repeat_audit_v1/result.json"
    if not path.exists():
        pytest.skip("saved repeat-audit artifact is absent")
    result = verify(REPO, path)
    assert result["attempts_recomputed"] == 896
    assert result["training_controls_recomputed"] > 0
    assert result["full_depth_programs_forward_replayed"] > 0


@pytest.mark.parametrize("mutation", ["summary", "gate", "input"])
def test_verifier_rejects_forged_audit_claims(tmp_path, mutation):
    path = REPO / "results/phase1/combinatorial_repeat_audit_v1/result.json"
    if not path.exists():
        pytest.skip("saved repeat-audit artifact is absent")
    result = json.loads(path.read_text())
    if mutation == "summary":
        result["classifications_by_arm"]["trained"]["original_exact_program"] += 1
    elif mutation == "gate":
        result["gates"]["original_results_reproduced_without_changes"] = False
    else:
        result["inputs"]["cache"] = result["inputs"]["attempts"]
    forged = tmp_path / "result.json"
    forged.write_text(json.dumps(result))
    with pytest.raises(LibraryAssemblyError, match="differ|substituted"):
        verify(REPO, forged)
