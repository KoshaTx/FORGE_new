"""Exact selector observation, symmetry-aware target ranking and state-universe checks."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from forge.model.defog_feasibility import AtomState
from forge.model.sparse_topology_feasibility import SparseGraphRecord
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_semantic_placement import (
    UgiSemanticPlacementError,
    _record_smiles,
    _selector_geometry,
    baseline_head_placement_candidates,
    compare_candidate_universes,
    rank_candidate_scores,
)


class FixtureSupport:
    """Artificial permissive support isolates candidate-enumeration behavior, not chemistry evidence."""

    enforces_role_cycles = False

    def component_is_within_observed_support(self, *args, **kwargs):
        return True

    def allows_role_edge_for_any_bond(self, *args):
        return True

    def allows_role_triangle(self, *args):
        return True


def _arguments(*, multiple_states=False):
    atoms = (AtomState("C", 0, False, 0), AtomState("N", 0, False, 0))
    if multiple_states:
        atoms += (AtomState("N", 0, False, 1),)
    edges = np.zeros((5, 5), dtype=np.int64)
    for node in range(1, 5):
        edges[node, node - 1] = edges[node - 1, node] = 1
    graph = SparseGraphRecord(
        structure_id="fixture",
        canonical_smiles="NCCCN",
        node_states=np.array([1, 0, 0, 0, 1]),
        parents=np.array([0, 0, 1, 2, 3]),
        parent_bonds=np.zeros(5, dtype=np.int64),
        closure_left=np.empty(0, dtype=np.int64),
        closure_right=np.empty(0, dtype=np.int64),
        closure_bonds=np.empty(0, dtype=np.int64),
        edges=edges,
    )
    record = SynthesisProgramGraphRecord(
        graph=graph,
        canonical_atom_order=np.arange(5),
        program_id="fixture",
        program_state=1,
        program_depth=1,
        role_states=np.ones(5, dtype=np.int64),
        core_position_states=np.array([2, 1, 1, 1, 1]),
        component_blocks=(SynthesisProgramComponentBlock("head", 1, 0, 5),),
        fixed_atom_mask=np.array([True, False, False, False, False]),
        fixed_parent_bond_mask=np.array([False, True, False, False, False]),
        fixed_closure_bond_mask=np.empty(0, dtype=bool),
    )
    logits = np.zeros((5, len(atoms)), dtype=np.float64)
    logits[4, 1] = 4
    return {
        "record": record,
        "atom_vocabulary": atoms,
        "node_logits": logits,
        "local_chemistry_support": FixtureSupport(),
        "ester_policy": SimpleNamespace(reaction_id="fixture", amine_role="head"),
        "semantic_target": UgiAmineSemanticTarget(5, 3, 2, 0),
    }


def test_original_selector_candidates_and_correct_target_rank_are_exposed():
    args = _arguments()
    result = baseline_head_placement_candidates(**args)
    assert result["rank"]["candidate_count"] == 2
    assert result["rank"]["selected_is_correct"]
    assert result["rank"]["optimistic_rank"] == 1
    assert result["constitutional_evaluations"] == 2
    assert result["filter_stage_counts"]["cycle_support"] == 2
    assert result["minimum_used"] == [2, 4, 4, 4, 2]
    assert result["baseline_selector_calls"] == 1 and result["random_draws"] == 0
    assert result["candidates"][result["correct_indices"][0]]["canonical_smiles"] == "NCCCN"
    args["node_logits"][1, 1] = 8
    wrong = baseline_head_placement_candidates(**args, reference_candidates=result)
    assert not wrong["rank"]["selected_is_correct"]
    assert wrong["rank"]["optimistic_rank"] == 2
    assert wrong["constitutional_reuses"] == 2 and wrong["constitutional_evaluations"] == 0
    assert compare_candidate_universes(result, wrong)["same_candidate_states"]


def test_neutral_same_symbol_state_changes_are_not_hidden_by_symbol_universe():
    args = _arguments(multiple_states=True)
    first = baseline_head_placement_candidates(**args)
    args["node_logits"][:, 2] = 10
    second = baseline_head_placement_candidates(**args, reference_candidates=first)
    compared = compare_candidate_universes(first, second)
    assert first["multiple_neutral_states_by_symbol"] == {"N": [1, 2]}
    assert compared["same_legal_symbol_universe"]
    assert not compared["same_candidate_states"]
    assert compared["state_changed_pattern_count"] == 2
    assert second["constitutional_reuses"] == 0
    assert len(second["candidates"]) == 2  # No competing assignment is dropped.


def test_minimum_reservation_uses_fixed_bond_units_and_variable_single_minimum():
    args = _arguments()
    record = args["record"]
    bonds = record.graph.parent_bonds.copy()
    bonds[1] = 1  # Fixed double bond: four valence units.
    bonds[2] = 2  # Variable target triple: reserves only the decoder minimum.
    changed = replace(record, graph=replace(record.graph, parent_bonds=bonds))
    assert _selector_geometry(changed)["minimum_used"].tolist() == [4, 6, 4, 4, 2]


def test_tied_correct_set_rank_and_first_index_selection_are_distinct():
    result = rank_candidate_scores([5, 4, 4, 4, 1], [2, 3])
    assert result["optimistic_rank"] == 2
    assert result["pessimistic_rank"] == 3
    assert result["exact_score_rank"] == 2
    assert not result["selected_is_correct"]
    assert result["wrong_candidates_tied_with_best_correct"] == 1
    tie = rank_candidate_scores([4, 4], [1])
    assert tie["optimistic_rank"] == 1 and tie["pessimistic_rank"] == 2
    assert tie["selected_candidate_index"] == 0 and not tie["selected_is_correct"]


def test_singleton_target_absence_and_abstention_remain_visible():
    args = _arguments()
    args["semantic_target"] = UgiAmineSemanticTarget(5, 4, 1, 0)
    singleton = baseline_head_placement_candidates(**args)
    assert singleton["rank"]["singleton"] and not singleton["rank"]["target_present"]
    assert singleton["rank"]["optimistic_rank"] is None
    assert not singleton["rank"]["informative_ranking"]
    args["semantic_target"] = UgiAmineSemanticTarget(5, 99, 1, 0)
    empty = baseline_head_placement_candidates(**args)
    assert empty["selector_abstention"] == "ugi_amine_carbon_diameter_unavailable"
    assert empty["candidates"] == [] and empty["rank"]["candidate_count"] == 0


def test_reference_identity_mismatch_and_changed_baseline_contract_fail():
    args = _arguments()
    first = baseline_head_placement_candidates(**args)
    record = args["record"]
    args["record"] = replace(record, graph=replace(record.graph, structure_id="other"))
    with pytest.raises(UgiSemanticPlacementError, match="different full target"):
        baseline_head_placement_candidates(**args, reference_candidates=first)
    args = _arguments()
    args["semantic_target"] = UgiAmineSemanticTarget(5, 3, 2, 0, 2, 0)
    with pytest.raises(UgiSemanticPlacementError, match="four-coordinate"):
        baseline_head_placement_candidates(**args)


def test_nonfinite_logits_fail_before_selector():
    args = _arguments()
    args["node_logits"][0, 0] = np.nan
    with pytest.raises(UgiSemanticPlacementError, match="finite"):
        baseline_head_placement_candidates(**args)


def test_constitutional_symmetry_admits_multiple_atom_index_assignments():
    args = _arguments()
    old = args["record"]
    edges = np.zeros((4, 4), dtype=np.int64)
    for left, right in ((0, 1), (1, 2), (1, 3)):
        edges[left, right] = edges[right, left] = 1
    graph = replace(
        old.graph,
        canonical_smiles="CC(N)N",
        node_states=np.array([1, 0, 1, 0]),
        parents=np.array([0, 0, 1, 1]),
        parent_bonds=np.zeros(4, dtype=np.int64),
        edges=edges,
    )
    args["record"] = replace(
        old,
        graph=graph,
        canonical_atom_order=np.arange(4),
        role_states=np.ones(4, dtype=np.int64),
        core_position_states=np.array([2, 1, 1, 1]),
        component_blocks=(SynthesisProgramComponentBlock("head", 1, 0, 4),),
        fixed_atom_mask=np.array([True, False, False, False]),
        fixed_parent_bond_mask=np.array([False, True, False, False]),
    )
    args["node_logits"] = np.zeros((4, 2))
    args["semantic_target"] = UgiAmineSemanticTarget(3, 2, 2, 0)
    result = baseline_head_placement_candidates(**args)
    assert len(result["candidates"]) == 2
    assert result["correct_indices"] == [0, 1]
    assert result["rank"]["optimistic_rank"] == result["rank"]["pessimistic_rank"] == 1
    assert result["candidates"][0]["states"] != result["candidates"][1]["states"]
    assert (
        result["candidates"][0]["canonical_smiles"] == result["candidates"][1]["canonical_smiles"]
    )


def test_production_cache_empty_dense_edges_preserve_all_candidate_identities_and_ranks():
    args = _arguments()
    dense = baseline_head_placement_candidates(**args)
    record = args["record"]
    args["record"] = replace(
        record, graph=replace(record.graph, edges=np.empty((0, 0), dtype=np.int64))
    )
    sparse = baseline_head_placement_candidates(**args)
    assert sparse["target_canonical_smiles"] == dense["target_canonical_smiles"]
    assert sparse["candidates"] == dense["candidates"]
    assert sparse["correct_indices"] == dense["correct_indices"]
    assert sparse["rank"] == dense["rank"]
    assert sparse["selected_candidate_index"] == dense["selected_candidate_index"]


def test_sparse_reconstruction_keeps_exact_parent_and_closure_bond_states():
    args = _arguments()
    original = args["record"]
    # Existing sparse bond indices represent the exact recorded double/single ring;
    # this exercises reconstruction directly without claiming Ugi candidate applicability.
    graph = replace(
        original.graph,
        node_states=np.zeros(4, dtype=np.int64),
        parents=np.array([0, 0, 1, 2]),
        parent_bonds=np.array([0, 1, 0, 0]),
        closure_left=np.array([0]),
        closure_right=np.array([3]),
        closure_bonds=np.array([0]),
        edges=np.empty((0, 0), dtype=np.int64),
    )
    record = replace(
        original,
        graph=graph,
        canonical_atom_order=np.arange(4),
        role_states=np.ones(4, dtype=np.int64),
        core_position_states=np.array([2, 1, 1, 1]),
        component_blocks=(SynthesisProgramComponentBlock("head", 1, 0, 4),),
        fixed_atom_mask=np.array([True, False, False, False]),
        fixed_parent_bond_mask=np.array([False, True, False, False]),
        fixed_closure_bond_mask=np.array([False]),
    )
    smiles = _record_smiles(record, graph.node_states, args["atom_vocabulary"])
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    assert molecule.GetNumAtoms() == 4 and molecule.GetNumBonds() == 4
    assert sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in molecule.GetBonds()) == 1
    assert molecule.GetRingInfo().NumRings() == 1
