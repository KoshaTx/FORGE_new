"""Exact baseline Ugi topology candidates and anchored heavy-skeleton labels.

This module neither reads a corpus nor chooses a molecule. The caller owns TRAIN
membership and input pins. Existing baseline enumerators own support and order;
no semantic guidance or directional tail target is added. Target labels ignore
exterior chemistry and serialization, and are not constitutional-product labels.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from rdkit import Chem

from forge.core.hashing import sha256_json
from forge.model.defog_feasibility import AtomState
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_program_flow import derive_role_morphology_states
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_program import (
    ROLE_NAMES,
    UgiMorphologyProgramError,
    preorder_attached_forest_to_parents,
)
from forge.model.ugi_transformer_topology import (
    UGI_PROGRAM_ID,
    UgiTransformerTopologyError,
    UgiTransformerTopologyPolicy,
    _enumerate_constructive_ester_offspring,
    _role_targets,
    _root_aligned_permutation,
    enumerate_amine_semantic_topologies,
)


class UgiTopologyCandidatesError(ValueError):
    """A supplied record or candidate contract is malformed or unsupported."""


@dataclass(frozen=True)
class TopologyCandidate:
    offspring: tuple[int, ...]
    closures: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TopologyCandidateSet:
    role: str
    record_node_count: int
    maximum_children: int
    first_closure_slot: int
    exterior_nodes: tuple[int, ...]
    graph_node_indices: tuple[int, ...]
    node_colors: np.ndarray
    color_feature_names: tuple[str, ...]
    candidates: tuple[TopologyCandidate, ...]
    enumerated_candidates: tuple[TopologyCandidate, ...]
    enumerated_identity_sha256: str
    unavailable_reason: str | None
    failed_candidate_index: int | None
    adjacency: np.ndarray
    global_closures: np.ndarray
    positive_mask: np.ndarray
    graph_class_ids: tuple[int, ...]
    candidate_graph_keys: tuple[str, ...]
    target_graph_key: str
    support_key_sha256: str
    ordered_identity_sha256: str
    target_record_sha256: str
    status: str
    reason: str | None

    @property
    def graph_class_count(self) -> int:
        return len(set(self.graph_class_ids))

    @property
    def target_present(self) -> bool:
        return bool(self.positive_mask.any())

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def enumerated_candidate_count(self) -> int:
        return len(self.enumerated_candidates)


# Bounded support cache; keys hash the complete policies, never object identity.
_SUPPORT_CACHE: OrderedDict[str, tuple[TopologyCandidate, ...]] = OrderedDict()
_MAX_SUPPORT_KEYS = 256


def clear_topology_candidate_caches() -> None:
    """Clear bounded pure-data caches, e.g. before measuring cold-path performance."""
    _SUPPORT_CACHE.clear()
    _colored_graph_key.cache_clear()


def _readonly(value: Any, dtype: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if shape is not None:
        array = array.reshape(shape)
    array.setflags(write=False)
    return array


@lru_cache(maxsize=8192)
def _colored_graph_key(
    colors: tuple[tuple[str, int, int], ...], edges: tuple[tuple[int, int], ...]
) -> str:
    """Canonical colored unweighted graph; dummy atoms encode no target chemistry.

    Isotopes represent colors, so isomericSmiles must stay True to retain them.
    These synthetic graph keys are never chemical SMILES or assessed products.
    """
    palette = {color: index + 1 for index, color in enumerate(sorted(set(colors)))}
    molecule = Chem.RWMol()
    for color in colors:
        atom = Chem.Atom(0)
        atom.SetIsotope(palette[color])
        molecule.AddAtom(atom)
    for left, right in edges:
        molecule.AddBond(left, right, Chem.BondType.SINGLE)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    # Include the palette itself: equally shaped differently colored graphs must differ.
    return str(sha256_json({"palette": sorted(palette), "graph": canonical}))


def _edges(record: SynthesisProgramGraphRecord) -> tuple[tuple[int, int], ...]:
    """Read authoritative sparse edges, including caches with no dense edge matrix."""
    graph, count = record.graph, record.node_count
    parents = np.asarray(graph.parents)
    left, right = np.asarray(graph.closure_left), np.asarray(graph.closure_right)
    if (
        parents.shape != (count,)
        or not np.issubdtype(parents.dtype, np.integer)
        or int(parents[0]) != 0
        or left.shape != right.shape
        or left.ndim != 1
        or not np.issubdtype(left.dtype, np.integer)
        or not np.issubdtype(right.dtype, np.integer)
        or any(not 0 <= int(parents[child]) < child for child in range(1, count))
    ):
        raise UgiTopologyCandidatesError("target sparse parents/closures are malformed")
    edges = [(int(parents[child]), child) for child in range(1, count)]
    for a, b in zip(left, right, strict=True):
        if not 0 <= int(a) < count or not 0 <= int(b) < count or a == b:
            raise UgiTopologyCandidatesError("target sparse closure endpoint is invalid")
        edges.append(tuple(sorted((int(a), int(b)))))
    if len(set(edges)) != len(edges):
        raise UgiTopologyCandidatesError("target sparse graph contains duplicate edges")
    return tuple(sorted(edges))


def _support_candidates(
    *,
    key: str,
    role: str,
    target: UgiAmineSemanticTarget | None,
    morphology: tuple[int, int, int, int],
    maximum_children: int,
    topology_policy: UgiTransformerTopologyPolicy,
    ester_policy: UgiEsterChemotypePolicy,
    local_chemistry_support: LocalChemistrySupport | None,
) -> tuple[TopologyCandidate, ...]:
    if key in _SUPPORT_CACHE:
        _SUPPORT_CACHE.move_to_end(key)
        return _SUPPORT_CACHE[key]
    count, junctions, cycles, attachments = morphology
    try:
        if role == ester_policy.amine_role:
            raw = enumerate_amine_semantic_topologies(
                node_count=count,
                junction_budget=junctions,
                cycle_rank=cycles,
                attachment_count=attachments,
                target=target,
                maximum_children=maximum_children,
                policy=topology_policy,
                allowed_ring_sizes=(
                    ester_policy.allowed_amine_cycle_sizes(count) if cycles else None
                ),
                local_chemistry_support=local_chemistry_support,
            )
            # This is precisely the baseline helper's by_key deduplication/sorted order.
            identities = {(tuple(map(int, item.offspring)), tuple(item.closures)) for item in raw}
            result = tuple(TopologyCandidate(*item) for item in sorted(identities))
        else:
            raw = _enumerate_constructive_ester_offspring(
                node_count=count,
                minimum_side_carbons=ester_policy.minimum_ester_side_carbons,
                minimum_long_side_carbons=ester_policy.minimum_ester_long_side_carbons,
                exact_full_side_carbons=None,
                exact_alkoxy_handle_and_acyl_side_carbons=None,
            )
            # Ester helper retains enumeration order and serialization multiplicity.
            result = tuple(TopologyCandidate(tuple(map(int, item)), ()) for item in raw)
    except (UgiMorphologyProgramError, UgiTransformerTopologyError) as error:
        raise UgiTopologyCandidatesError(f"unsupported {role} enumeration: {error}") from error
    _SUPPORT_CACHE[key] = result
    if len(_SUPPORT_CACHE) > _MAX_SUPPORT_KEYS:
        _SUPPORT_CACHE.popitem(last=False)
    return result


def topology_candidates(
    record: SynthesisProgramGraphRecord,
    role: str,
    atom_vocabulary: Sequence[AtomState],
    topology_policy: UgiTransformerTopologyPolicy,
    ester_policy: UgiEsterChemotypePolicy,
    local_chemistry_support: LocalChemistrySupport | None,
    amine_target: UgiAmineSemanticTarget | None,
    *,
    maximum_children: int,
    core_position_classes: int,
) -> TopologyCandidateSet:
    """Return the original ordered candidate universe with topology-only target labels.

    Explicit dimensions come from the frozen checkpoint/program vocabulary, not
    a fitted statistic. Node colors have shape [M, core_position_classes + 1].
    Every candidate adjacency is unweighted [M,M], ordered exterior first then
    all fixed reaction-core nodes. Outside-role exterior nodes are not included.
    Unsupported size/ester morphology yields an explicit unavailable outcome.
    If the amine helper cannot align even one enumerated topology, it fails
    before drawing from the entire set. Retain that whole-case unavailability,
    including every raw identity and the first failing index; never prune just
    the offending candidate. Enumeration limits and malformed records raise.
    """
    if (
        record.program_id != ester_policy.reaction_id
        or record.program_id != UGI_PROGRAM_ID
        or role not in (ester_policy.amine_role, ester_policy.aldehyde_role)
        or role not in ROLE_NAMES
    ):
        raise UgiTopologyCandidatesError("record/role is outside the baseline Ugi contract")
    if (
        type(maximum_children) is not int
        or maximum_children < 2
        or type(core_position_classes) is not int
        or core_position_classes < 3
    ):
        raise UgiTopologyCandidatesError("explicit child/core vocabulary dimensions are invalid")
    if (
        not atom_vocabulary
        or not np.issubdtype(record.graph.node_states.dtype, np.integer)
        or np.any(record.graph.node_states < 0)
        or np.any(record.graph.node_states >= len(atom_vocabulary))
        or np.any(record.core_position_states < 1)
        or np.any(record.core_position_states >= core_position_classes)
    ):
        raise UgiTopologyCandidatesError("record atom/core states exceed the declared vocabulary")
    if role == ester_policy.amine_role and (
        not isinstance(amine_target, UgiAmineSemanticTarget)
        or amine_target.hydrogen_bond_donors is not None
        or amine_target.heavy_branch_atoms is not None
    ):
        raise UgiTopologyCandidatesError("baseline head requires exactly four amine coordinates")
    if record.role_morphology_states is None:
        record = replace(record, role_morphology_states=derive_role_morphology_states(record))
    try:
        targets = _role_targets(record)
    except UgiTransformerTopologyError as error:
        raise UgiTopologyCandidatesError(str(error)) from error
    blocks = [block for block in record.component_blocks if block.role == role]
    if len(blocks) != 1:
        raise UgiTopologyCandidatesError("one role must have exactly one component block")
    block = blocks[0]
    exterior = tuple(
        node for node in range(block.start, block.stop) if record.core_position_states[node] == 1
    )
    morphology = targets[role]
    count, junctions, cycles, attachments = morphology
    if len(exterior) != count or count < 1:
        raise UgiTopologyCandidatesError("exterior size differs from the declared morphology")
    core = tuple(int(node) for node in np.flatnonzero(record.core_position_states > 1))
    if not core or not np.all(record.fixed_atom_mask[list(core)]):
        raise UgiTopologyCandidatesError("reaction-core nodes must be adapter-fixed")
    if np.any(record.fixed_atom_mask[list(exterior)]):
        raise UgiTopologyCandidatesError("fixed exterior atoms need an explicit candidate contract")
    target_edges = _edges(record)
    roots = tuple(
        local
        for local, node in enumerate(exterior)
        if record.fixed_parent_bond_mask[node]
        and record.core_position_states[int(record.graph.parents[node])] > 1
    )
    if len(roots) != attachments:
        raise UgiTopologyCandidatesError("fixed core attachments differ from morphology")
    nodes = (*exterior, *core)
    node_to_local = {node: local for local, node in enumerate(nodes)}
    exterior_set, core_set = set(exterior), set(core)
    fixed_edges = tuple((a, b) for a, b in target_edges if a in core_set and b in core_set) + tuple(
        (int(record.graph.parents[exterior[root]]), exterior[root]) for root in roots
    )
    fixed_pairs = {tuple(sorted(edge)) for edge in fixed_edges}
    for a, b in target_edges:
        if (a in exterior_set) != (b in exterior_set) and (a in exterior_set or b in exterior_set):
            if tuple(sorted((a, b))) not in fixed_pairs:
                raise UgiTopologyCandidatesError("role exterior has an unsupported boundary edge")
    # Only root-to-core edges define the anchor flag, not all core-core bonds.
    anchor_nodes = {
        node
        for root in roots
        for node in (exterior[root], int(record.graph.parents[exterior[root]]))
    }
    role_by_state = {block.role_state: block.role for block in record.component_blocks}
    colors = tuple(
        (
            role_by_state[int(record.role_states[node])],
            int(record.core_position_states[node]),
            int(node in anchor_nodes),
        )
        for node in nodes
    )
    numeric_colors = np.zeros((len(nodes), core_position_classes + 1), dtype=np.float32)
    for local, node in enumerate(nodes):
        numeric_colors[local, int(record.core_position_states[node])] = 1
        numeric_colors[local, -1] = int(node in anchor_nodes)
    local_target_edges = tuple(
        sorted(
            tuple(sorted((node_to_local[a], node_to_local[b])))
            for a, b in target_edges
            if a in node_to_local and b in node_to_local
        )
    )
    target_key = _colored_graph_key(colors, local_target_edges)
    support_key = str(
        sha256_json(
            {
                "role": role,
                "morphology": morphology,
                "maximum_children": maximum_children,
                "topology_policy": topology_policy.to_mapping(),
                "ester_policy": ester_policy.to_mapping(),
                "local_chemistry_support": (
                    None
                    if local_chemistry_support is None
                    else local_chemistry_support.to_mapping()
                ),
                "amine_target": (
                    amine_target.to_mapping() if role == ester_policy.amine_role else None
                ),
                "semantic_guidance": None,
                "exact_directional_tail_targets": None,
            }
        )
    )
    unavailable = None
    if not (
        ester_policy.minimum_topology_exterior_atoms(role)
        <= count
        <= ester_policy.maximum_exterior_atoms(role)
    ):
        unavailable = "role_exterior_size_outside_frozen_chemotype_support"
    elif role == ester_policy.aldehyde_role and (junctions, cycles, attachments) != (1, 0, 1):
        unavailable = "aldehyde_morphology_incompatible_with_constructive_ester"
    candidates = (
        ()
        if unavailable is not None
        else _support_candidates(
            key=support_key,
            role=role,
            target=amine_target,
            morphology=morphology,
            maximum_children=maximum_children,
            topology_policy=topology_policy,
            ester_policy=ester_policy,
            local_chemistry_support=local_chemistry_support,
        )
    )
    enumerated_candidates = candidates
    failed_candidate_index = None
    adjacency = np.zeros((len(candidates), len(nodes), len(nodes)), dtype=bool)
    global_closures = np.zeros((len(candidates), cycles, 2), dtype=np.int64)
    graph_keys = []
    for index, candidate in enumerate(candidates):
        parents = preorder_attached_forest_to_parents(
            np.asarray(candidate.offspring), attachment_count=attachments
        )
        try:
            permutation = _root_aligned_permutation(parents, roots)
        except UgiTransformerTopologyError as error:
            if (
                role == ester_policy.amine_role
                and str(error) == "root alignment would violate sparse parent ordering"
            ):
                # The unchanged amine helper visits ALL candidates before its single
                # multinomial draw. One alignment failure makes the whole law
                # unavailable; retaining only alignable candidates would change it.
                unavailable = "root_alignment_would_violate_sparse_parent_ordering"
                failed_candidate_index = index
                candidates = ()
                adjacency = np.zeros((0, len(nodes), len(nodes)), dtype=bool)
                global_closures = np.zeros((0, cycles, 2), dtype=np.int64)
                graph_keys.clear()
                break
            raise UgiTopologyCandidatesError(str(error)) from error
        edges = list(fixed_edges)
        for child, parent in enumerate(parents):
            if parent >= 0:
                edges.append(
                    (exterior[int(permutation[parent])], exterior[int(permutation[child])])
                )
        if len(candidate.closures) != cycles:
            raise UgiTopologyCandidatesError("candidate closure count changed")
        for slot, (left, right) in enumerate(candidate.closures):
            pair = (exterior[int(permutation[left])], exterior[int(permutation[right])])
            global_closures[index, slot] = pair
            edges.append(pair)
        local_edges = tuple(
            sorted(tuple(sorted((node_to_local[a], node_to_local[b]))) for a, b in edges)
        )
        if len(set(local_edges)) != len(local_edges):
            raise UgiTopologyCandidatesError("candidate contains duplicate edges")
        for left, right in local_edges:
            adjacency[index, left, right] = adjacency[index, right, left] = True
        graph_keys.append(_colored_graph_key(colors, local_edges))
    class_by_key: dict[str, int] = {}
    class_ids = tuple(class_by_key.setdefault(key, len(class_by_key)) for key in graph_keys)
    positive = np.asarray([key == target_key for key in graph_keys], dtype=bool)
    reason = unavailable
    if not candidates:
        reason = reason or "no_legal_baseline_candidates"
    elif not positive.any():
        reason = "target_heavy_skeleton_absent"
    elif len(class_by_key) == 1:
        reason = "single_heavy_skeleton_class"
    return TopologyCandidateSet(
        role=role,
        record_node_count=record.node_count,
        maximum_children=maximum_children,
        first_closure_slot=sum(targets[name][2] for name in ROLE_NAMES[: ROLE_NAMES.index(role)]),
        exterior_nodes=exterior,
        graph_node_indices=nodes,
        node_colors=_readonly(numeric_colors, np.float32),
        color_feature_names=tuple(
            [f"core_position_state_{index}" for index in range(core_position_classes)]
            + ["fixed_core_attachment"]
        ),
        candidates=candidates,
        enumerated_candidates=enumerated_candidates,
        enumerated_identity_sha256=str(
            sha256_json([(item.offspring, item.closures) for item in enumerated_candidates])
        ),
        unavailable_reason=unavailable,
        failed_candidate_index=failed_candidate_index,
        adjacency=_readonly(adjacency, bool),
        global_closures=_readonly(global_closures, np.int64),
        positive_mask=_readonly(positive, bool),
        graph_class_ids=class_ids,
        candidate_graph_keys=tuple(graph_keys),
        target_graph_key=target_key,
        support_key_sha256=support_key,
        ordered_identity_sha256=str(
            sha256_json([(item.offspring, item.closures) for item in candidates])
        ),
        target_record_sha256=str(
            sha256_json(
                {
                    "record_id": record.graph.structure_id,
                    "sparse_edges": target_edges,
                    "nodes": nodes,
                    "colors": colors,
                    "node_states": record.graph.node_states.tolist(),
                    "atom_vocabulary": [
                        [atom.symbol, atom.formal_charge, atom.aromatic, atom.explicit_hydrogens]
                        for atom in atom_vocabulary
                    ],
                }
            )
        ),
        status=(
            "unavailable" if unavailable is not None else "empty" if not candidates else "complete"
        ),
        reason=reason,
    )


def score_candidates(
    candidate_set: TopologyCandidateSet, predictions: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Exact float64 baseline scores for ONE record, with detached CPU predictions.

    Shapes follow the model: offspring [1,padded_nodes,maximum_children+1] and
    closure endpoints [1,slots,padded_nodes]. Structured fields take precedence
    individually as in decode_ugi_exact_topology. Closure endpoint addition occurs
    in the original numpy dtype before conversion to float64; orientation uses
    reverse > direct (ties retain direct). No probabilities or RNG draws occur.
    """
    chosen = {
        name: f"structured_{name}" if f"structured_{name}" in predictions else name
        for name in ("offspring", "closure_left", "closure_right")
    }
    values = {}
    for name, key in chosen.items():
        value = predictions.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or value.shape[0] != 1
            or value.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(value).all())
        ):
            raise UgiTopologyCandidatesError(f"{key} must be finite floating one-record logits")
        values[name] = value.detach().cpu()
    offspring = values["offspring"]
    padded_nodes = offspring.shape[1]
    if (
        padded_nodes < candidate_set.record_node_count
        or offspring.shape[2] != candidate_set.maximum_children + 1
        or values["closure_left"].shape != values["closure_right"].shape
        or values["closure_left"].shape[2] != padded_nodes
        or values["closure_left"].dtype != values["closure_right"].dtype
        or values["closure_left"].shape[1]
        < candidate_set.first_closure_slot + candidate_set.global_closures.shape[1]
    ):
        raise UgiTopologyCandidatesError("offspring/closure dimensions differ from candidate set")
    if not candidate_set.candidates:
        return torch.empty(0, dtype=torch.float64)
    logits = offspring[0, list(candidate_set.exterior_nodes)].to(torch.float64)
    positions = torch.arange(len(candidate_set.exterior_nodes))
    # Keep the original per-vector reduction: a batched sum can change reduction order.
    scores = torch.stack(
        [logits[positions, torch.tensor(item.offspring)].sum() for item in candidate_set.candidates]
    )
    left_logits = values["closure_left"][0].numpy()
    right_logits = values["closure_right"][0].numpy()
    for local_slot in range(candidate_set.global_closures.shape[1]):
        slot = candidate_set.first_closure_slot + local_slot
        left, right = candidate_set.global_closures[:, local_slot].T
        direct = left_logits[slot, left] + right_logits[slot, right]
        reverse = left_logits[slot, right] + right_logits[slot, left]
        contribution = np.where(reverse > direct, reverse, direct).astype(np.float64)
        scores = scores + torch.from_numpy(contribution)
    if not bool(torch.isfinite(scores).all()):
        raise UgiTopologyCandidatesError("baseline candidate score overflowed")
    return scores
