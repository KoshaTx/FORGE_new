from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.phase1.multireaction.combinatorial_generation import summarize
from experiments.phase1.multireaction.combinatorial_generation_verify import verify
from forge.assembly.families import LibraryAssemblyError, load_assembly_libraries
from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_programs import LibraryProgramLimits, replay_library_program
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_program_flow import (
    collate_synthesis_program_layouts,
    derive_role_morphology_states,
)
from forge.model.synthesis_program_sampling import (
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)

REPO = Path(__file__).resolve().parents[1]
DATASET = json.loads((REPO / "configs/multireaction/library_program_dataset_v1.json").read_text())


@pytest.fixture(scope="module")
def libraries():
    return load_assembly_libraries(
        [
            (REPO / DATASET["inputs"][k]["path"], DATASET["inputs"][k]["sha256"])
            for k in DATASET["registries"]
        ],
        expected_families=DATASET["programs"],
    )


@pytest.mark.parametrize("family", sorted(DATASET["programs"]))
def test_all_twelve_registry_positive_products_have_forward_replayed_witnesses(libraries, family):
    adapter = libraries[family]
    raw = next(
        r
        for r in json.loads(adapter.registry_path.read_text())["reactions"]
        if r["reaction_id"] == family
    )
    product = raw["known_positive_examples"][0]["expected"]
    check = check_generated_program(
        adapter, product, depth=1, accumulator_role=None, limits=LibraryProgramLimits(1)
    )
    assert check.exact
    for witness in check.programs:
        assert replay_library_program(
            adapter, witness["components"], witness["intermediate_products"], accumulator_role=None
        )


def test_repeated_check_uses_full_requested_depth_and_identical_reagent(libraries):
    adapter = libraries["aza_michael_amine_acrylate"]
    components = dict(zip(adapter.roles, ["NCCN", "C=CC(=O)OCC"]))
    first = adapter.forward_products(components).products[0]
    second = adapter.forward_products({**components, "amine_head": first}).products[0]
    check = check_generated_program(
        adapter, second, depth=2, accumulator_role="amine_head", limits=LibraryProgramLimits(2)
    )
    assert check.exact
    assert all(len(w["intermediate_products"]) == 2 for w in check.programs)
    too_deep = check_generated_program(
        adapter, first, depth=2, accumulator_role="amine_head", limits=LibraryProgramLimits(2)
    )
    assert not too_deep.exact


def test_saturated_and_incomplete_search_never_admits_partial_witness(libraries):
    adapter = libraries["aza_michael_amine_acrylate"]
    product = adapter.forward_products(dict(zip(adapter.roles, ["NCCN", "C=CC(=O)OCC"]))).products[
        0
    ]
    saturated = check_generated_program(
        adapter,
        product,
        depth=1,
        accumulator_role=None,
        limits=LibraryProgramLimits(1, maximum_outcomes=1),
    )
    assert saturated.status == "abstain" and not saturated.programs
    limited = check_generated_program(
        adapter,
        product,
        depth=2,
        accumulator_role="amine_head",
        limits=LibraryProgramLimits(2, maximum_expansions=1),
    )
    assert limited.status == "abstain" and not limited.programs
    with pytest.raises(LibraryAssemblyError, match="depth"):
        check_generated_program(
            adapter, product, depth=True, accumulator_role=None, limits=LibraryProgramLimits(1)
        )


def test_summary_keeps_invalid_attempts_and_duplicate_products_in_denominators():
    good = {
        "program_id": "x",
        "valid_connected": True,
        "canonical_smiles": "CC",
        "assembly": {"status": "exact_computed_program"},
        "novel_vs_train": True,
        "all_witnesses_have_novel_component": False,
        "failure_reason": None,
    }
    bad = {**good, "valid_connected": False, "failure_reason": "invalid_graph"}
    summary = summarize([good, good, bad], ["x"])["x"]
    assert summary["attempts"] == 3
    assert summary["exact_program_fraction_all_attempts"] == 2 / 3
    assert summary["unique_exact_products"] == summary["effective_product_count"] == 1
    assert summary["unique_novel_exact_products"] == 1


def test_twelve_family_sampler_does_not_copy_variable_graph_targets():
    checkpoint = REPO / "results/phase1/combinatorial_checkpoint_v1/checkpoint.json"
    if not checkpoint.exists():
        pytest.skip("retained diagnostic checkpoint is absent")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, vocabulary, atoms, nodes, bonds, _ = load_synthesis_program_checkpoint(
            checkpoint, device="cpu"
        )
        with SynthesisProgramProductionCache(
            REPO / "results/phase1/combinatorial_program_cache_v2/cache.npz"
        ) as cache:
            records = [
                cache.record(int(cache.indices(program_id=p, fold="train")[0]))
                for p in vocabulary.program_states[1:]
            ]
        # Hold the declared coarse semantics fixed while changing every variable target field.
        records = [
            replace(r, role_morphology_states=derive_role_morphology_states(r)) for r in records
        ]
        changed = []
        for record in records:
            states = record.graph.node_states.copy()
            states[~record.fixed_atom_mask] = (states[~record.fixed_atom_mask] + 1) % len(atoms)
            assert np.any(states != record.graph.node_states)
            parents = record.graph.parents.copy()
            bonds_changed = record.graph.parent_bonds.copy()
            for child in range(1, record.node_count):
                if not record.fixed_parent_bond_mask[child]:
                    parents[child] = child - 1
                    bonds_changed[child] = (bonds_changed[child] + 1) % 4
            left, right, closure_bonds = (
                a.copy()
                for a in (
                    record.graph.closure_left,
                    record.graph.closure_right,
                    record.graph.closure_bonds,
                )
            )
            for slot in range(record.graph.closure_count):
                if not record.fixed_closure_bond_mask[slot]:
                    left[slot], right[slot], closure_bonds[slot] = 0, record.node_count - 1, 2
            changed.append(
                replace(
                    record,
                    graph=replace(
                        record.graph,
                        node_states=states,
                        parents=parents,
                        parent_bonds=bonds_changed,
                        closure_left=left,
                        closure_right=right,
                        closure_bonds=closure_bonds,
                        canonical_smiles="CC",
                    ),
                )
            )
        layouts = [
            collate_synthesis_program_layouts(r, maximum_closures=1) for r in (records, changed)
        ]
        assert all(torch.equal(layouts[0][k], layouts[1][k]) for k in layouts[0])
        outputs = [
            sample_synthesis_program_products(
                model,
                r,
                atoms,
                nodes,
                bonds,
                samples_per_program=1,
                sample_steps=2,
                batch_size=12,
                seed=12,
                device="cpu",
                terminal_decode_policy="strict_valence_topology_argmax",
            )
            for r in (records, changed)
        ]
        assert [r["canonical_smiles"] for r in outputs[0][0]] == [
            r["canonical_smiles"] for r in outputs[1][0]
        ]
        assert outputs[0][1]["fixed_state_failures"] == outputs[1][1]["fixed_state_failures"] == 0
        assert len(outputs[0][0]) == 12
    finally:
        torch.set_num_threads(previous_threads)


def test_saved_generation_is_verified_from_attempts_and_forward_replay():
    result = REPO / "results/phase1/combinatorial_generation_v1/result.json"
    if not result.exists():
        pytest.skip("saved generation diagnostic is absent")
    checked = verify(REPO, result)
    assert checked["status"] == "verified"
    assert checked["attempts_verified"] == 2 * 12 * 64
    assert checked["witnesses_forward_replayed"] > 0


@pytest.mark.parametrize("mutation", ["summary", "inputs", "claim"])
def test_verifier_rejects_forged_result(tmp_path, mutation):
    source = REPO / "results/phase1/combinatorial_generation_v1/result.json"
    if not source.exists():
        pytest.skip("saved generation diagnostic is absent")
    result = json.loads(source.read_text())
    if mutation == "summary":
        family = next(iter(result["per_arm"]["trained"]))
        result["per_arm"]["trained"][family]["exact_program"] += 1
    elif mutation == "inputs":
        # Both pins still resolve; only the configured identity check catches substitution.
        result["inputs"]["checkpoint"] = result["inputs"]["cache"]
    else:
        result["generation_demonstrated_all_families"] = False
    path = tmp_path / "forged-result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="differs|substituted"):
        verify(REPO, path)
