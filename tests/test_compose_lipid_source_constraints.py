"""Registered equality, symmetry, immutable cores and bounded checked completion."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from forge.model.compose_lipid_completion import admit_generated_repeat_completions
from forge.model.compose_lipid_generation import propose_generated_reuse
from forge.model.compose_lipid_source_constraints import (
    POLICY_ID,
    SourceConstraints,
    check_source_completion,
    compile_constraints,
    propose_source_completion,
    query_correspondences,
)
from forge.model.synthesis_program_graph import SynthesisProgramComponentBlock
from tests.test_compose_lipid_completion import checker, example


def different_stage_labels():
    layout, state, atoms = example()
    positions = layout.record.core_position_states.copy()
    tails = [b for b in layout.record.component_blocks if b.role == "tail"]
    for label, block in zip((101, 102), tails, strict=True):
        core = [i for i in range(block.start, block.stop) if positions[i] > 1]
        assert len(core) == 1
        positions[core] = label
    layout = replace(layout, record=replace(layout.record, core_position_states=positions))
    return layout, state, atoms


def test_registered_stage_alias_recovers_generated_repeat_without_reference_atoms():
    layout, state, atoms = different_stage_labels()
    assert not propose_generated_reuse(layout, state, atoms)["proposals"]
    original = copy.deepcopy(state)
    constraints = SourceConstraints(layout.record.program_id, ((102, 101),))
    proposed = propose_source_completion(layout, state, atoms, constraints)
    assert state == original
    assert len(proposed["proposals"]) == 1
    assert proposed["proposals"][0]["smiles"] in {"CCNCC", "OCNCO"}
    assert proposed["proposals"][0]["registered_repeat_groups"] == 1


def test_internal_arm_correspondence_copies_generated_connectivity():
    layout, state, atoms = different_stage_labels()
    block = SynthesisProgramComponentBlock("all", 1, 0, layout.record.node_count)
    layout = replace(
        layout,
        quantities={"all": 1},
        record=replace(
            layout.record,
            component_blocks=(block,),
            role_states=np.ones(layout.record.node_count, dtype=np.int64),
        ),
    )
    proposed = propose_source_completion(
        layout,
        state,
        atoms,
        SourceConstraints(layout.record.program_id, symmetric_anchors=((101, 102),)),
    )
    assert proposed["proposals"][0]["smiles"] in {"CCNCC", "OCNCO"}
    assert proposed["proposals"][0]["symmetric_arm_groups"] == 1


def test_wrong_program_or_changed_core_fails_loudly():
    layout, state, atoms = different_stage_labels()
    with pytest.raises(ValueError, match="another program"):
        propose_source_completion(layout, state, atoms, SourceConstraints("wrong"))
    index = int(np.flatnonzero(layout.record.fixed_atom_mask)[0])
    state["nodes"][index] += 1
    with pytest.raises(ValueError, match="immutable assembly"):
        propose_source_completion(layout, state, atoms, SourceConstraints(layout.record.program_id))


def test_single_proposal_rejection_never_changes_admission_or_tries_second_donor():
    layout, state, atoms = different_stage_labels()
    calls = []
    row = check_source_completion(
        layout,
        state,
        atoms,
        SourceConstraints(layout.record.program_id, ((102, 101),)),
        check_product=checker(calls, donor_exact=False),
    )
    selected = admit_generated_repeat_completions(
        [row], is_train_product=lambda _: False, policy_id=POLICY_ID
    )[0]
    assert len(calls) == 2 and not selected["accepted"] and not selected["changed"]
    assert selected["original_smiles"] == selected["selected_smiles"]


def test_exact_original_is_preserved_before_any_projection():
    layout, state, atoms = different_stage_labels()
    row = check_source_completion(
        layout,
        state,
        atoms,
        SourceConstraints(layout.record.program_id, ((102, 101),)),
        check_product=lambda *_: dict(status="evaluated", exact=True),
    )
    assert row["status"] == "original_exact" and row["proposals"] == []


def test_already_admitted_completion_is_immutable_under_new_constraints():
    from forge.model.compose_lipid_completion import check_generated_repeat_completion

    layout, state, atoms = example()
    baseline = check_generated_repeat_completion(layout, state, atoms, check_product=checker([]))
    baseline = admit_generated_repeat_completions([baseline], is_train_product=lambda _: False)[0]
    assert baseline["changed"] and baseline["accepted"]
    # This unresolved symmetric relation must never discard the successful baseline.
    constraints = SourceConstraints(layout.record.program_id, symmetric_anchors=((100, 101),))
    row = check_source_completion(
        layout, state, atoms, constraints, check_product=checker([]), baseline=baseline
    )
    selected = admit_generated_repeat_completions(
        [row], is_train_product=lambda _: False, policy_id=POLICY_ID
    )[0]
    assert selected["accepted"] and selected["selected_smiles"] == baseline["selected_smiles"]
    assert not selected["proposals"]
    with pytest.raises(ValueError, match="failed full-program replay"):
        check_source_completion(
            layout,
            state,
            atoms,
            constraints,
            check_product=checker([], donor_exact=False),
            baseline=baseline,
        )


def test_query_correspondence_respects_atom_and_bond_predicates_and_bounds():
    left, right = Chem.MolFromSmarts("C-O"), Chem.MolFromSmarts("O-C")
    assert query_correspondences(left, right) == [(1, 0)]
    assert query_correspondences(left, Chem.MolFromSmarts("O=C")) == []
    assert query_correspondences(Chem.MolFromSmarts("c:c"), Chem.MolFromSmarts("c-c")) == []
    with pytest.raises(ValueError, match="search bound"):
        query_correspondences(left, right, maximum_states=1)


def test_compiler_derives_stage_equality_from_vendored_registry():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "data/vendor/qualified_thiol_yne_staged_source_program_v1.json").read_text()
    )
    program = registry["programs"][0]
    vocabulary = ["unconditioned", "exterior"]
    for step, stage in enumerate(program["stages"], 1):
        reaction = next(
            r for r in registry["reactions"] if r["reaction_id"] == stage["reaction_id"]
        )
        from rdkit.Chem import rdChemReactions

        rxn = rdChemReactions.ReactionFromSmarts(reaction["atom_mapped_reaction_smarts"])
        vocabulary.extend(
            f"test:step_{step}:map_{a.GetAtomMapNum()}"
            for q in rxn.GetReactants()
            for a in q.GetAtoms()
            if a.GetAtomMapNum()
        )
    compiled = compile_constraints(
        registry,
        program_id="test",
        core_vocabulary=vocabulary,
        sequential_program=program["program_id"],
    )
    assert len(compiled.core_aliases) == 1 and not compiled.symmetric_anchors
    modified = copy.deepcopy(registry)
    modified["programs"][0]["equal_component_groups"] = []
    assert not compile_constraints(
        modified,
        program_id="test",
        core_vocabulary=vocabulary,
        sequential_program=program["program_id"],
    ).core_aliases
