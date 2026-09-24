"""Complete donor chemistry, source-head retention and bounded coupled changes."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from forge.assembly.generated_source import evaluate_executor
from forge.assembly.grouped_atom_origins import trace_grouped_program
from forge.assembly.grouped_program import RegistryGroupedProgram
from forge.core.hashing import sha256_file
from forge.corpus.qualified_program_cache import QualifiedProgramExample
from forge.model.compose_lipid_component_constraints import (
    _violations,
    propose_component_constraints,
)
from forge.model.compose_lipid_component_policy import compile_component_policies
from forge.model.compose_lipid_layout import build_layout, summarize_layout
from forge.model.precursor_reuse_projection import state_graph
from tests.test_compose_lipid_generation import gold_predictions
from tests.test_source_instance_coordinates import record


def layout_from_record(r, quantities):
    e = QualifiedProgramExample(
        r, "test", tuple((role, role, q) for role, q in quantities.items()), ()
    )
    bundle, choices, rings = summarize_layout(e)
    return build_layout(bundle, choices, rings, identity="component-test")


def state_from_record(r):
    return {
        k: getattr(r.graph, "node_states" if k == "nodes" else k).tolist()
        for k in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        )
    }


@pytest.fixture(scope="module")
def ren():
    path = Path("data/vendor/qualified_ren_love_source_program_v1.json")
    program = RegistryGroupedProgram.from_registry(
        path, program_id="source_ren_2_incorporated_arms", expected_sha256=str(sha256_file(path))
    )
    document = json.loads(
        Path(
            "results/phase1/compose_lipid_user_supplements_v1/control-transcriptions.json"
        ).read_text()
    )
    source = document["controls"]["ren"]
    mapping = {r: r for r in program.roles}
    traced = trace_grouped_program(program, source["components"], source_roles=mapping)
    assert traced.annotations is not None
    a = traced.annotations
    r, _, atoms = record(a.canonical_product_smiles, a.atom_roles, a.core_positions)
    layout = layout_from_record(r, program.quantities)
    executor = dict(kind="grouped", program=program, mapping=mapping, full_source_contract=True)
    domain = json.loads(
        Path("data/vendor/qualified_ester_thiol_yne_source_program_v1.json").read_text()
    )["ester_thiol_domain"]
    executor["proposal_retained_queries"] = {
        "bromoester_arm": [
            {**q, "whole_component": True} for q in domain["constraints"]["required_queries"]
        ]
    }
    return r, layout, atoms, executor


@pytest.mark.needs_vendor
@pytest.mark.parametrize("both_bad", [False, True])
def test_complete_tail_is_screened_then_shared_and_independently_replayed(ren, both_bad):
    r, layout, atoms, executor = ren
    state = state_from_record(r)
    original = copy.deepcopy(state)
    pred = {k: v[0].numpy() for k, v in gold_predictions(r, atoms).items()}
    carbon = next(i for i, a in enumerate(atoms) if a.symbol == "C")
    blocks = [b for b in r.component_blocks if b.role == "bromoester_arm"]
    assert len(blocks) == 2
    for b in blocks if both_bad else blocks[:1]:
        oxygen = next(
            i
            for i in range(b.start, b.stop)
            if atoms[state["nodes"][i]].symbol == "O" and not r.fixed_atom_mask[i]
        )
        state["nodes"][oxygen] = carbon
    policies = compile_component_policies(executor)
    assert next(p for p in policies if p["role"] == "bromoester_arm")["product_element_counts"] == {
        "Br": 0,
        "O": 2,
    }
    out = propose_component_constraints(layout, state, pred, atoms, policies)
    assert out == propose_component_constraints(layout, state, pred, atoms, policies)
    admitted = [
        p
        for p in out["proposals"]
        if p["smiles"]
        and evaluate_executor(executor, p["smiles"], events=2)["exact_registry_program_roundtrip"]
    ]
    assert admitted, out
    assert out["mutations_evaluated"] <= 4 * 128
    if both_bad:
        assert admitted[0]["policy"] == "coupled_component_repair"
        assert all(len(e["locations"]) == 2 for e in admitted[0]["edits"])
    else:
        assert admitted[0]["donors"] == [1]
    for p in admitted:
        assert len(p["nodes"]) == len(original["nodes"])
        assert np.count_nonzero(p["edges"]) == np.count_nonzero(state_graph(original)[1])
    assert state != original  # Inputs were deliberately corrupted, not silently rewritten.


def test_required_head_cannot_be_rescued_by_another_origin_or_an_amide():
    cfg = json.loads(
        Path("configs/multireaction/compose_lipid_v8_passerini_program_v1.json").read_text()
    )
    policy = cfg["source_contract"]["retained_amine_policy"]
    smiles = "CCN(CC)CCNC(=O)C"
    molecule = Chem.MolFromSmiles(smiles)
    # The only tertiary nonacyl nitrogen belongs to the other precursor.
    roles = ["other" if i <= 6 else "head" for i in range(molecule.GetNumAtoms())]
    core = [
        "map_1" if i == 6 else "map_2" if i == 7 else "exterior"
        for i in range(molecule.GetNumAtoms())
    ]
    r, _, atoms = record(smiles, roles, core)
    layout = layout_from_record(r, {"head": 1, "other": 1})
    failures = _violations(
        *state_graph(state_from_record(r)),
        atoms,
        layout,
        [dict(role="head", retained_amine_policy=policy)],
    )
    assert [f["kind"] for f in failures] == ["retained_amine"]


def test_extra_ketone_predicate_does_not_ban_esters():
    reaction = json.loads(
        Path("data/vendor/qualified_miao_cyclic_source_program_v2.json").read_text()
    )["reactions"][0]
    executor = dict(
        full_source_contract=True,
        kind="fixed",
        reaction=reaction,
        mapping={r["name"]: r["name"] for r in reaction["reactant_roles"]},
    )
    policy = next(p for p in compile_component_policies(executor) if p["role"] == "coupled_ketone")
    for smiles, expected in [("CCCC(=O)OCC", 0), ("CCC(=O)CC", 1)]:
        mol = Chem.MolFromSmiles(smiles)
        r, _, atoms = record(
            smiles,
            ["coupled_ketone"] * mol.GetNumAtoms(),
            ["map_1"] + ["exterior"] * (mol.GetNumAtoms() - 1),
        )
        layout = layout_from_record(r, {"coupled_ketone": 1})
        failures = _violations(*state_graph(state_from_record(r)), atoms, layout, [policy])
        assert len(failures) == expected


def test_complete_source_contract_required_and_forward_failure_cannot_be_pruned():
    with pytest.raises(ValueError, match="complete source contract"):
        evaluate_executor(dict(kind="program"), "CC", events=1)
    program = SimpleNamespace(
        infer=lambda _: dict(complete_search=True, candidate_components=[dict(a="C")]),
        replay=lambda *args: dict(
            computed_consistency_pass=False, checks=dict(unique_unfiltered_forward_exact=False)
        ),
    )
    e = dict(kind="program", program=program, full_source_contract=True, mapping={"a": "a"})
    result = evaluate_executor(e, "CC", events=1)
    assert not result["exact_registry_program_roundtrip"]
    assert len(result["candidates"]) == 1 and not result["accepted_components"]


@pytest.mark.needs_vendor
def test_nonfinite_predictions_and_excess_donor_budget_fail_closed(ren):
    r, layout, atoms, e = ren
    pred = {k: v[0].numpy() for k, v in gold_predictions(r, atoms).items()}
    out = propose_component_constraints(
        layout, state_from_record(r), pred, atoms, compile_component_policies(e), maximum_donors=1
    )
    assert out["status"] == "donor_budget_exceeded" and not out["proposals"]
    pred["nodes"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        propose_component_constraints(
            layout, state_from_record(r), pred, atoms, compile_component_policies(e)
        )


@pytest.mark.needs_vendor
def test_correct_oxygen_count_cannot_substitute_for_the_retained_ester(ren):
    *_, executor = ren
    policy = next(p for p in compile_component_policies(executor) if p["role"] == "bromoester_arm")
    for smiles, expected in [("CCOCCOC", 1), ("CCCC(=O)OCC", 0)]:
        count = Chem.MolFromSmiles(smiles).GetNumAtoms()
        r, _, atoms = record(
            smiles, ["bromoester_arm"] * count, ["map_1"] + ["exterior"] * (count - 1)
        )
        layout = layout_from_record(r, {"bromoester_arm": 1})
        failed = _violations(*state_graph(state_from_record(r)), atoms, layout, [policy])
        assert len(failed) == expected
        if expected:
            assert failed[0]["kind"] == "required_query"


@pytest.mark.needs_vendor
@pytest.mark.parametrize(
    "registry",
    [
        "qualified_thiol_yne_staged_source_program_v1.json",
        "qualified_ester_thiol_yne_source_program_v1.json",
    ],
)
def test_both_thiol_yne_programs_project_the_required_headgroup(registry):
    specification = json.loads((Path("data/vendor") / registry).read_text())["programs"][0]
    e = dict(
        full_source_contract=True,
        program=SimpleNamespace(specification=specification),
        mapping={role: role for role in specification["terminal_constraints"]},
    )
    policies = compile_component_policies(e)
    head = next(p for p in policies if p["role"] == "amine_head")
    assert head["required_queries"] == specification["product_constraints"]["required_queries"]


@pytest.mark.needs_vendor
def test_coupled_branch_repair_satisfies_basic_head_and_competing_amine_exclusion():
    registry = json.loads(
        Path("data/vendor/qualified_acid_epoxide_staged_source_program_v1.json").read_text()
    )
    role = registry["reactions"][1]["reactant_roles"][0]
    assert role["name"] == "amine_acid_head"
    policy = dict(
        role="head",
        **registry["programs"][0]["terminal_constraints"]["amine_acid_head"],
        remaining_handles=[
            dict(smarts=s, maximum_matches=0, exclude_core=True) for s in role["forbidden_smarts"]
        ],
    )
    r, _, atoms = record("CCCCCC", ["head"] * 6, ["map_1"] + ["exterior"] * 5)
    from forge.model.defog_feasibility import AtomState
    from forge.model.qualified_vocabulary import QualifiedAtomVocabulary

    atoms = QualifiedAtomVocabulary((*atoms.states, AtomState("N", 0, False, 0)))
    layout = layout_from_record(r, {"head": 1})
    state = state_from_record(r)
    predictions = {k: v[0].numpy() for k, v in gold_predictions(r, atoms).items()}
    without = propose_component_constraints(layout, state, predictions, atoms, [policy])
    assert not any(p["smiles"] for p in without["proposals"])
    with_branch = propose_component_constraints(
        layout, state, predictions, atoms, [policy], allow_branch_edits=True
    )
    changed = [p for p in with_branch["proposals"] if p["smiles"]]
    assert changed and changed[-1]["edits"][0]["kind"] == "branch_atom"
    assert not _violations(
        np.asarray(changed[-1]["nodes"]), np.asarray(changed[-1]["edges"]), atoms, layout, [policy]
    )
    assert len(changed[-1]["nodes"]) == 6 and np.count_nonzero(changed[-1]["edges"]) == 10
    assert with_branch["mutations_evaluated"] <= 4 * 2 * 128
