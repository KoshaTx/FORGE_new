"""Read-only exact baseline amine-candidate enumeration and constitutional ranking.

The existing selector owns the candidate universe. This module observes its return,
then compares candidate head assignments on the unchanged clean target graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from rdkit import Chem, rdBase

from forge.core.hashing import sha256_json
from forge.model.defog_feasibility import AtomState
from forge.model.local_chemistry_support import LocalChemistrySupport, tree_path_indices
from forge.model.sparse_topology_feasibility import SPARSE_BOND_TO_INDEX
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord
from forge.model.synthesis_program_sampling import (
    _atom_capacity_table,
    _bond_unit_table,
    _distances_to_reaction_core,
    _generated_ring_support,
    _select_ugi_amine_atom_states,
    _terminal_smiles,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_sampling_trace import SamplingTrace, _json_safe


class UgiSemanticPlacementError(ValueError):
    """The exact candidate observation or constitutional comparison is inconsistent."""


def rank_candidate_scores(
    scores: Sequence[float], correct_indices: Sequence[int], *, tie_atol: float = 1e-8
) -> dict[str, Any]:
    """Rank the best correct assignment, retaining wrong-score ties and first-index argmax."""

    values = np.asarray(scores, dtype=np.float64)
    correct = tuple(correct_indices)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or not np.isfinite(tie_atol)
        or tie_atol < 0
        or len(set(correct)) != len(correct)
        or any(type(index) is not int or not 0 <= index < len(values) for index in correct)
    ):
        raise UgiSemanticPlacementError(
            "invalid candidate scores, correct indices or tie tolerance"
        )
    selected = int(np.argmax(values)) if len(values) else None
    result = {
        "candidate_count": len(values),
        "correct_candidate_count": len(correct),
        "target_present": bool(correct),
        "singleton": len(values) == 1,
        "informative_ranking": len(values) > 1 and bool(correct),
        "selected_candidate_index": selected,
        "selected_is_correct": selected in correct,
        "tie_atol": tie_atol,
        "optimistic_rank": None,
        "pessimistic_rank": None,
        "exact_score_rank": None,
        "wrong_candidates_tied_with_best_correct": None,
    }
    if correct:
        best = float(values[list(correct)].max())
        higher = int(np.sum(values > best + tie_atol))
        wrong = np.ones(len(values), dtype=bool)
        wrong[list(correct)] = False
        ties = int(np.sum(wrong & (np.abs(values - best) <= tie_atol)))
        result.update(
            {
                "best_correct_score": best,
                "optimistic_rank": 1 + higher,
                "pessimistic_rank": 1 + higher + ties,
                "exact_score_rank": 1 + int(np.sum(values > best)),
                "wrong_candidates_tied_with_best_correct": ties,
            }
        )
    return result


def _selector_geometry(record: SynthesisProgramGraphRecord) -> dict[str, Any]:
    """Build the same reservation and graph metadata used immediately before atom selection."""

    graph, count = record.graph, record.node_count
    neighbors = [set() for _ in range(count)]
    minimum_used = np.zeros(count, dtype=np.int64)
    bond_units = _bond_unit_table(len(SPARSE_BOND_TO_INDEX))
    minimum_bond_units = int(bond_units[SPARSE_BOND_TO_INDEX[Chem.BondType.SINGLE]])
    occupied = set()
    edges = [
        (
            int(graph.parents[child]),
            child,
            int(graph.parent_bonds[child]),
            bool(record.fixed_parent_bond_mask[child]),
        )
        for child in range(1, count)
    ]
    edges += [
        (int(left), int(right), int(bond), bool(record.fixed_closure_bond_mask[slot]))
        for slot, (left, right, bond) in enumerate(
            zip(graph.closure_left, graph.closure_right, graph.closure_bonds, strict=True)
        )
    ]
    for left, right, bond, fixed in edges:
        pair = tuple(sorted((left, right)))
        if left == right or pair in occupied or not 0 <= left < count or not 0 <= right < count:
            raise UgiSemanticPlacementError("target graph has invalid or duplicate edges")
        occupied.add(pair)
        neighbors[left].add(right)
        neighbors[right].add(left)
        minimum_used[[left, right]] += int(bond_units[bond]) if fixed else minimum_bond_units
    triangles = [
        (left, middle, right)
        for left in range(count)
        for middle in sorted(value for value in neighbors[left] if value > left)
        for right in sorted(
            value for value in neighbors[left].intersection(neighbors[middle]) if value > middle
        )
    ]
    cycles = [
        tree_path_indices(
            graph.parents, int(graph.closure_left[slot]), int(graph.closure_right[slot])
        )
        for slot in range(graph.closure_count)
        if not record.fixed_closure_bond_mask[slot]
    ]
    role_by_state = {block.role_state: block.role for block in record.component_blocks}
    return {
        "minimum_used": minimum_used,
        "neighbors": neighbors,
        "triangles": triangles,
        "generated_cycles": cycles,
        "role_names": tuple(role_by_state[int(state)] for state in record.role_states),
        "distances_to_core": _distances_to_reaction_core(neighbors, record.core_position_states),
        "ring_nodes": _generated_ring_support(cycles)[0],
        "ring_edges": _generated_ring_support(cycles)[1],
    }


def _record_digest(record: SynthesisProgramGraphRecord, atoms: Sequence[AtomState]) -> str:
    return str(sha256_json(_json_safe({"record": record, "atom_vocabulary": atoms})))


def _record_smiles(
    record: SynthesisProgramGraphRecord, nodes: np.ndarray, atoms: Sequence[AtomState]
) -> str | None:
    """Use the sampler's exact sparse reconstruction; cache records need no dense edges."""

    state = {
        field: torch.from_numpy(np.asarray(value, dtype=np.int64).copy())[None, ...]
        for field, value in {
            "nodes": nodes,
            "parents": record.graph.parents,
            "parent_bonds": record.graph.parent_bonds,
            "closure_left": record.graph.closure_left,
            "closure_right": record.graph.closure_right,
            "closure_bonds": record.graph.closure_bonds,
        }.items()
    }
    with rdBase.BlockLogs():
        return _terminal_smiles(state, 0, record.node_count, record.graph.closure_count, atoms)


def baseline_head_placement_candidates(
    *,
    record: SynthesisProgramGraphRecord,
    atom_vocabulary: Sequence[AtomState],
    node_logits: np.ndarray,
    local_chemistry_support: LocalChemistrySupport,
    ester_policy: UgiEsterChemotypePolicy,
    semantic_target: UgiAmineSemanticTarget,
    max_candidates: int = 50000,
    max_events: int = 100000,
    tie_atol: float = 1e-8,
    reference_candidates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Observe exactly one unchanged baseline selector call on a supplied clean topology.

    ``node_logits`` has shape (record.node_count, len(atom_vocabulary)). Correctness is
    constitutional whole-product equality after substituting only the candidate head atoms;
    all target bonds and every other atom remain fixed. Sanitization failures stay in the
    candidate universe. Reference reuse requires the exact full record/vocabulary digest and
    identical candidate node-state assignment; it never substitutes an old candidate list.
    """

    logits = np.asarray(node_logits)
    if logits.shape != (record.node_count, len(atom_vocabulary)) or not np.isfinite(logits).all():
        raise UgiSemanticPlacementError("node logits must be finite and match record/vocabulary")
    if (
        semantic_target.hydrogen_bond_donors is not None
        or semantic_target.heavy_branch_atoms is not None
    ):
        raise UgiSemanticPlacementError("baseline probe requires the frozen four-coordinate target")
    if record.program_id != ester_policy.reaction_id:
        raise UgiSemanticPlacementError("record does not match the ester policy reaction")
    blocks = [block for block in record.component_blocks if block.role == ester_policy.amine_role]
    if len(blocks) != 1:
        raise UgiSemanticPlacementError("record must contain one amine block")
    block = blocks[0]
    nodes = tuple(
        node
        for node in range(block.start, block.stop)
        if int(record.core_position_states[node]) == 1 and not record.fixed_atom_mask[node]
    )
    record_digest = _record_digest(record, atom_vocabulary)
    prior = {}
    if reference_candidates is not None:
        if reference_candidates["record_vocabulary_sha256"] != record_digest:
            raise UgiSemanticPlacementError(
                "reference candidates belong to a different full target record"
            )
        if reference_candidates["exterior_nodes"] != list(nodes):
            raise UgiSemanticPlacementError("reference exterior coordinate order differs")
        prior = {tuple(row["states"]): row for row in reference_candidates["candidates"]}
    if reference_candidates is not None:
        target_smiles = reference_candidates["target_canonical_smiles"]
    else:
        target_smiles = _record_smiles(record, record.graph.node_states, atom_vocabulary)
        if target_smiles is None:
            raise UgiSemanticPlacementError("clean target sparse reconstruction is invalid")
    geometry = _selector_geometry(record)
    # Standalone selectors are deliberately represented as attempt 0 by the existing observer.
    with SamplingTrace(
        selected_attempts=(0,), max_candidates=max_candidates, max_events=max_events
    ) as trace:
        returned, reason = _select_ugi_amine_atom_states(
            {"nodes": logits[None, ...]},
            0,
            record,
            atom_vocabulary,
            _atom_capacity_table(atom_vocabulary),
            local_chemistry_support=local_chemistry_support,
            policy=ester_policy,
            semantic_target=semantic_target,
            **geometry,
        )
    decisions = [
        event
        for event in trace.events
        if event["kind"] == "terminal_assignment"
        and event["selector"] == "_select_ugi_amine_atom_states"
    ]
    outcomes = [
        event
        for event in trace.events
        if event["kind"] == "selector_outcome"
        and event["selector"] == "_select_ugi_amine_atom_states"
    ]
    if len(outcomes) != 1 or len(decisions) != int(returned is not None):
        raise UgiSemanticPlacementError(
            "selector trace did not expose exactly one complete outcome"
        )
    result = {
        "schema_version": "forge.ugi_baseline_head_placement.v1",
        "record_vocabulary_sha256": record_digest,
        "record_id": record.graph.structure_id,
        "target_canonical_smiles": target_smiles,
        "semantic_target": semantic_target.to_mapping(),
        "exterior_nodes": list(nodes),
        "selector_abstention": reason,
        "filter_stage_counts": outcomes[0].get("filter_stage_counts"),
        "candidates": [],
        "constitutional_evaluations": 0,
        "constitutional_reuses": 0,
        "baseline_selector_calls": 1,
        "random_draws": 0,
        "input_logits_sha256": str(sha256_json(logits.tolist())),
        "minimum_used": geometry["minimum_used"].tolist(),
        "equivalence": "stereo_free_complete_target_graph_with_candidate_head_states_only",
    }
    neutral = {}
    for state, atom in enumerate(atom_vocabulary):
        if atom.formal_charge == 0 and not atom.aromatic:
            neutral.setdefault(atom.symbol, []).append(state)
    result["multiple_neutral_states_by_symbol"] = {
        symbol: states for symbol, states in neutral.items() if len(states) > 1
    }
    if decisions:
        event = decisions[0]
        for observed in event["candidates"]:
            assignment = dict(observed["identity"]["entries"])
            if set(assignment) != set(nodes):
                raise UgiSemanticPlacementError(
                    "selector candidate does not assign every exterior node"
                )
            states = tuple(int(assignment[node]) for node in nodes)
            row = {
                "candidate_index": observed["candidate_index"],
                "states": list(states),
                "symbols": [atom_vocabulary[state].symbol for state in states],
                "neural_score": observed["neural_score"],
                "semantic_distance": observed["semantic_distance"],
                "identity_sha256": observed["identity_sha256"],
            }
            if states in prior:
                previous = prior[states]
                row.update(
                    {
                        key: previous[key]
                        for key in ("canonical_smiles", "constitutional_error", "correct")
                    }
                )
                result["constitutional_reuses"] += 1
            else:
                candidate = record.graph.node_states.copy()
                candidate[list(nodes)] = states
                smiles = _record_smiles(record, candidate, atom_vocabulary)
                row.update(
                    canonical_smiles=smiles,
                    constitutional_error=(
                        None if smiles is not None else "invalid_sparse_target_bond_completion"
                    ),
                    correct=smiles is not None and smiles == target_smiles,
                )
                result["constitutional_evaluations"] += 1
            result["candidates"].append(row)
        selected = int(event["selected_candidate_index"])
        if returned != dict(zip(nodes, result["candidates"][selected]["states"], strict=True)):
            raise UgiSemanticPlacementError(
                "selector return differs from observed selected candidate"
            )
    correct = [row["candidate_index"] for row in result["candidates"] if row["correct"]]
    result["correct_indices"] = correct
    result["rank"] = rank_candidate_scores(
        [row["neural_score"] for row in result["candidates"]], correct, tie_atol=tie_atol
    )
    result["selected_candidate_index"] = result["rank"]["selected_candidate_index"]
    result["symbol_universe_sha256"] = str(
        sha256_json(sorted(row["symbols"] for row in result["candidates"]))
    )
    result["candidate_assignments_sha256"] = str(
        sha256_json([row["states"] for row in result["candidates"]])
    )
    return result


def compare_candidate_universes(
    reference: Mapping[str, Any], comparison: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare symbols separately from state choices; never call changed states identical."""

    if (
        reference["record_vocabulary_sha256"] != comparison["record_vocabulary_sha256"]
        or reference["exterior_nodes"] != comparison["exterior_nodes"]
    ):
        raise UgiSemanticPlacementError(
            "candidate comparisons require the same record and coordinates"
        )

    def index(report):
        rows = {tuple(row["symbols"]): tuple(row["states"]) for row in report["candidates"]}
        if len(rows) != len(report["candidates"]):
            raise UgiSemanticPlacementError("selector repeated a symbol assignment")
        return rows

    left, right = index(reference), index(comparison)
    changed = sorted(
        pattern for pattern in left.keys() & right.keys() if left[pattern] != right[pattern]
    )
    return {
        "same_legal_symbol_universe": left.keys() == right.keys(),
        "same_candidate_states": left == right,
        "state_changed_symbol_patterns": [list(pattern) for pattern in changed],
        "state_changed_pattern_count": len(changed),
        "added_symbol_patterns": [list(pattern) for pattern in sorted(right.keys() - left.keys())],
        "removed_symbol_patterns": [
            list(pattern) for pattern in sorted(left.keys() - right.keys())
        ],
    }
