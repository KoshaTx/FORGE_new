"""Factorized count-only semantic layouts for synthesis-program sampling.

The prior learns only program depth, closure count, origin-component multiplicities and sizes,
reaction-core coordinate counts, and adapter-fixed reaction-core states.  It never stores or copies a
training product's variable atom states, parent pointers, bonds, component identities, or SMILES.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.sparse_topology_feasibility import SparseGraphRecord
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)


class SynthesisProgramLayoutError(ValueError):
    """A factorized layout cannot satisfy the declared program support."""


@dataclass(frozen=True)
class _RecordSummary:
    depth: int
    closure_count: int
    blocks: tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]
    fixed_signature: tuple[tuple[Any, ...], ...]
    weight: float


@dataclass(frozen=True)
class _WeightedSupport:
    """One immutable categorical support compiled once from weighted source rows."""

    values: tuple[Any, ...]
    probabilities: np.ndarray

    @classmethod
    def build(cls, values: Sequence[Any], weights: Sequence[float]) -> _WeightedSupport:
        grouped: dict[Any, float] = {}
        for value, weight in zip(values, weights, strict=True):
            grouped[value] = grouped.get(value, 0.0) + float(weight)
        support = tuple(sorted(grouped))
        probabilities = np.asarray([grouped[value] for value in support], dtype=np.float64)
        if not support or not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
            raise SynthesisProgramLayoutError("factorized layout field has no positive measure")
        probabilities /= probabilities.sum()
        probabilities.setflags(write=False)
        return cls(values=support, probabilities=probabilities)

    def sample(self, rng: np.random.Generator) -> Any:
        return self.values[int(rng.choice(len(self.values), p=self.probabilities))]


@dataclass(frozen=True)
class _ProgramDistribution:
    depth: _WeightedSupport
    closure_count: _WeightedSupport
    semantic_bundle_by_depth: Mapping[int, _WeightedSupport]
    component_size_by_bundle: Mapping[
        tuple[int, tuple[Any, ...], int, tuple[tuple[int, int], ...]],
        _WeightedSupport,
    ]


def _semantic_bundle(summary: _RecordSummary) -> tuple[Any, ...]:
    """Return the joint count-only contract that determines structural compatibility."""

    # Core coordinates belong to component occurrences, not merely to a global role.  Losing that
    # association permits a repeated-program layout to put every core atom in one precursor copy
    # and none in another.  The signature is count-only: it retains no atom state, bond, component
    # identity or stored fragment.
    multiplicity_pattern = tuple(
        sorted(
            Counter((role, core_signature) for role, _, core_signature in summary.blocks).items()
        )
    )
    return multiplicity_pattern, summary.fixed_signature


def _compile_program_distribution(summaries: Sequence[_RecordSummary]) -> _ProgramDistribution:
    """Compile O(1)-per-field categorical draws without retaining source identities."""

    depth_values = [summary.depth for summary in summaries]
    closure_values = [summary.closure_count for summary in summaries]
    weights = [summary.weight for summary in summaries]
    bundle_values: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    bundle_weights: dict[int, list[float]] = defaultdict(list)
    size_values: dict[tuple[int, tuple[Any, ...], int, tuple[tuple[int, int], ...]], list[int]] = (
        defaultdict(list)
    )
    size_weights: dict[
        tuple[int, tuple[Any, ...], int, tuple[tuple[int, int], ...]], list[float]
    ] = defaultdict(list)
    for summary in summaries:
        bundle = _semantic_bundle(summary)
        bundle_values[summary.depth].append(bundle)
        bundle_weights[summary.depth].append(summary.weight)
        by_component_signature: dict[tuple[int, tuple[tuple[int, int], ...]], list[int]] = (
            defaultdict(list)
        )
        for role_state, size, core_signature in summary.blocks:
            by_component_signature[(int(role_state), core_signature)].append(int(size))
        for (role_state, core_signature), local_sizes in by_component_signature.items():
            if len(set(local_sizes)) != 1:
                raise SynthesisProgramLayoutError(
                    "identical repeated-component signatures have unequal atom counts"
                )
            key = (summary.depth, bundle, role_state, core_signature)
            size_values[key].append(local_sizes[0])
            size_weights[key].append(summary.weight)
    return _ProgramDistribution(
        depth=_WeightedSupport.build(depth_values, weights),
        closure_count=_WeightedSupport.build(closure_values, weights),
        semantic_bundle_by_depth={
            depth: _WeightedSupport.build(values, bundle_weights[depth])
            for depth, values in bundle_values.items()
        },
        component_size_by_bundle={
            key: _WeightedSupport.build(values, size_weights[key])
            for key, values in size_values.items()
        },
    )


def _fixed_signature(record: SynthesisProgramGraphRecord) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for child in range(1, record.node_count):
        if not record.fixed_parent_bond_mask[child]:
            continue
        parent = int(record.graph.parents[child])
        endpoints = tuple(
            sorted(
                (
                    int(record.core_position_states[index]),
                    int(record.role_states[index]),
                    bool(record.fixed_atom_mask[index]),
                )
                for index in (child, parent)
            )
        )
        rows.append((*endpoints, int(record.graph.parent_bonds[child])))
    for slot in range(record.graph.closure_count):
        if not record.fixed_closure_bond_mask[slot]:
            continue
        left = int(record.graph.closure_left[slot])
        right = int(record.graph.closure_right[slot])
        endpoints = tuple(
            sorted(
                (
                    int(record.core_position_states[index]),
                    int(record.role_states[index]),
                    bool(record.fixed_atom_mask[index]),
                )
                for index in (left, right)
            )
        )
        rows.append((*endpoints, int(record.graph.closure_bonds[slot])))
    return tuple(sorted(rows))


def _summarize_program(
    cache: SynthesisProgramProductionCache,
    program_id: str,
) -> tuple[tuple[_RecordSummary, ...], dict[int, int]]:
    """Collapse training records to count-only semantics and adapter-fixed atom states."""

    indices = cache.indices(program_id=program_id, fold="train")
    source = cache.arrays["source_weights"][indices].astype(np.float64)
    source /= source.sum()
    summaries: list[_RecordSummary] = []
    fixed_node_states: dict[int, int] = {}
    for cache_index, weight in zip(indices, source, strict=True):
        record = cache.record(int(cache_index))
        for position in np.flatnonzero(record.fixed_atom_mask):
            core_state = int(record.core_position_states[position])
            node_state = int(record.graph.node_states[position])
            previous = fixed_node_states.setdefault(core_state, node_state)
            if previous != node_state:
                raise SynthesisProgramLayoutError(
                    f"adapter-fixed node state varies for {program_id}:{core_state}"
                )
        summaries.append(
            _RecordSummary(
                depth=record.program_depth,
                closure_count=record.graph.closure_count,
                blocks=tuple(
                    (
                        block.role_state,
                        block.atom_count,
                        tuple(
                            sorted(
                                Counter(
                                    int(value)
                                    for value in record.core_position_states[
                                        block.start : block.stop
                                    ]
                                    if int(value) > 1
                                ).items()
                            )
                        ),
                    )
                    for block in record.component_blocks
                ),
                fixed_signature=_fixed_signature(record),
                weight=float(weight),
            )
        )
    if not summaries:
        raise SynthesisProgramLayoutError(f"count prior has no training records for {program_id}")
    return tuple(summaries), fixed_node_states


class SynthesisProgramLayoutPrior:
    """Program-specific factorized count prior fitted to the weighted training fold."""

    def __init__(self, cache: SynthesisProgramProductionCache) -> None:
        self.vocabulary = cache.vocabulary
        self.maximum_heavy_atoms = int(cache.metadata["support"]["maximum_heavy_atoms"])
        self.maximum_closures = int(cache.metadata["support"]["maximum_closures"])
        self._distributions: dict[str, _ProgramDistribution] = {}
        self._fixed_node_states: dict[str, dict[int, int]] = {}
        for program_id in cache.vocabulary.program_states[1:]:
            summaries, fixed_node_states = _summarize_program(cache, program_id)
            self._distributions[program_id] = _compile_program_distribution(summaries)
            self._fixed_node_states[program_id] = fixed_node_states
        self.validate_support()

    def _sample_fields(self, program_id: str, rng: np.random.Generator) -> tuple[
        int,
        int,
        list[tuple[int, int, tuple[tuple[int, int], ...]]],
        tuple[tuple[Any, ...], ...],
    ]:
        distribution = self._distributions[program_id]
        depth = int(distribution.depth.sample(rng))
        closure_count = int(distribution.closure_count.sample(rng))
        bundle = distribution.semantic_bundle_by_depth[depth].sample(rng)
        multiplicity_pattern, fixed_signature = bundle
        blocks: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
        for (role_state, core_signature), multiplicity in multiplicity_pattern:
            size = int(
                distribution.component_size_by_bundle[
                    (depth, bundle, int(role_state), core_signature)
                ].sample(rng)
            )
            blocks.extend((int(role_state), size, core_signature) for _ in range(int(multiplicity)))
        return depth, closure_count, blocks, fixed_signature

    def validate_support(self) -> dict[str, int]:
        """Exhaust every finite semantic bundle at its smallest admitted role sizes."""

        bundle_count = 0
        size_support_count = 0
        for program_id, distribution in sorted(self._distributions.items()):
            for depth, bundle_support in sorted(distribution.semantic_bundle_by_depth.items()):
                for bundle_index, bundle in enumerate(bundle_support.values):
                    multiplicity_pattern, fixed_signature = bundle
                    blocks: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
                    for (role_state, core_signature), multiplicity in multiplicity_pattern:
                        sizes = distribution.component_size_by_bundle[
                            (depth, bundle, int(role_state), core_signature)
                        ].values
                        size_support_count += len(sizes)
                        blocks.extend(
                            (int(role_state), int(min(sizes)), core_signature)
                            for _ in range(int(multiplicity))
                        )
                    closure_count = int(max(distribution.closure_count.values))
                    if self._fixed_node_states[program_id]:
                        self._fixed_core_layout(
                            program_id=program_id,
                            depth=depth,
                            closure_count=closure_count,
                            blocks=blocks,
                            fixed_signature=fixed_signature,
                            sample_index=bundle_index,
                        )
                    else:
                        self._generic_layout(
                            program_id=program_id,
                            depth=depth,
                            closure_count=closure_count,
                            blocks=blocks,
                            sample_index=bundle_index,
                            rng=np.random.default_rng(0),
                        )
                    bundle_count += 1
        return {
            "programs": len(self._distributions),
            "semantic_bundles": bundle_count,
            "component_size_support_cells": size_support_count,
        }

    @staticmethod
    def _assign_core_positions(
        blocks: Sequence[tuple[int, int, tuple[tuple[int, int], ...]]],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        total = sum(size for _, size, _ in blocks)
        roles = np.zeros(total, dtype=np.int64)
        core = np.ones(total, dtype=np.int64)
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        for role_state, size, core_signature in blocks:
            roles[cursor : cursor + size] = role_state
            spans.append((role_state, cursor, cursor + size))
            labels = [state for state, count in core_signature for _ in range(count)]
            if len(labels) > size:
                raise SynthesisProgramLayoutError(
                    "factorized component size cannot hold its reaction-core coordinates"
                )
            if labels:
                selected = rng.choice(
                    np.arange(cursor, cursor + size), size=len(labels), replace=False
                )
                for position, label in zip(sorted(selected.tolist()), sorted(labels), strict=True):
                    core[position] = label
            cursor += size
        return core, spans

    def _generic_layout(
        self,
        *,
        program_id: str,
        depth: int,
        closure_count: int,
        blocks: Sequence[tuple[int, int, tuple[tuple[int, int], ...]]],
        sample_index: int,
        rng: np.random.Generator,
    ) -> SynthesisProgramGraphRecord:
        core_positions, spans = self._assign_core_positions(blocks, rng)
        node_count = len(core_positions)
        role_states = np.zeros(node_count, dtype=np.int64)
        component_blocks: list[SynthesisProgramComponentBlock] = []
        for role_state, start, stop in spans:
            role_states[start:stop] = role_state
            component_blocks.append(
                SynthesisProgramComponentBlock(
                    role=self.vocabulary.role_states[role_state],
                    role_state=role_state,
                    start=start,
                    stop=stop,
                )
            )
        return self._record(
            program_id=program_id,
            depth=depth,
            closure_count=closure_count,
            role_states=role_states,
            core_positions=core_positions,
            component_blocks=component_blocks,
            fixed_node_states={},
            fixed_edges=(),
            sample_index=sample_index,
        )

    def _ugi_layout(
        self,
        *,
        program_id: str,
        depth: int,
        closure_count: int,
        blocks: Sequence[tuple[int, int, tuple[tuple[int, int], ...]]],
        fixed_signature: Sequence[tuple[Any, ...]],
        sample_index: int,
    ) -> SynthesisProgramGraphRecord:
        core_by_role: dict[int, list[int]] = defaultdict(list)
        for role_state, _, core_signature in blocks:
            for core_state, count in core_signature:
                core_by_role[int(role_state)].extend([int(core_state)] * int(count))
        internal_edges: list[tuple[int, int, int]] = []
        attachment_counts: Counter[tuple[int, int, int]] = Counter()
        for left, right, bond in fixed_signature:
            left_core, left_role, left_fixed = left
            right_core, right_role, right_fixed = right
            if left_fixed and right_fixed:
                internal_edges.append((int(left_core), int(right_core), int(bond)))
            else:
                fixed = left if left_fixed else right
                variable = right if left_fixed else left
                attachment_counts[(int(fixed[0]), int(variable[1]), int(bond))] += 1
        if not internal_edges or not self._fixed_node_states[program_id]:
            raise SynthesisProgramLayoutError("Ugi layout lacks an adapter-fixed core")
        role_graph: dict[int, set[int]] = defaultdict(set)
        core_role = {
            core_state: role_state
            for role_state, values in core_by_role.items()
            for core_state in values
        }
        for left, right, _ in internal_edges:
            role_graph[core_role[left]].add(core_role[right])
            role_graph[core_role[right]].add(core_role[left])
        root_role = min(role_graph)
        role_order: list[int] = []
        queue = deque([root_role])
        seen = {root_role}
        while queue:
            role = queue.popleft()
            role_order.append(role)
            for neighbor in sorted(role_graph[role]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        role_sizes = {role: size for role, size, _ in blocks}
        if set(role_sizes) != set(core_by_role) or len(role_sizes) != len(blocks):
            raise SynthesisProgramLayoutError(
                "Ugi layout requires one origin component for every role"
            )
        # Order core positions within each role by their fixed-core graph, then append exterior
        # slots.  The graph-level BFS below makes every fixed pointer parent precede its child.
        ordered_core_by_role = {role: sorted(values) for role, values in core_by_role.items()}
        node_specs: list[tuple[int, int]] = []
        component_blocks: list[SynthesisProgramComponentBlock] = []
        for role in role_order:
            start = len(node_specs)
            values = ordered_core_by_role[role]
            size = role_sizes[role]
            if size < len(values):
                raise SynthesisProgramLayoutError("Ugi role size cannot hold its fixed core")
            node_specs.extend((role, value) for value in values)
            node_specs.extend((role, 1) for _ in range(size - len(values)))
            component_blocks.append(
                SynthesisProgramComponentBlock(
                    role=self.vocabulary.role_states[role],
                    role_state=role,
                    start=start,
                    stop=len(node_specs),
                )
            )
        index_by_core = {core: index for index, (_, core) in enumerate(node_specs) if core > 1}
        # Reorder complete role blocks by a BFS of the core graph rooted in the first block's
        # first core.  With the qualified acyclic Ugi core, the role order above already ensures
        # parents precede children; orient every internal edge from lower to higher index.
        fixed_edges = [
            (
                min(index_by_core[left], index_by_core[right]),
                max(index_by_core[left], index_by_core[right]),
                bond,
            )
            for left, right, bond in internal_edges
        ]
        attachment_offsets: Counter[int] = Counter()
        for (core_state, role_state, bond), count in sorted(attachment_counts.items()):
            core_index = index_by_core[core_state]
            exterior = [
                index
                for index, (role, core) in enumerate(node_specs)
                if role == role_state and core == 1
            ]
            offset = attachment_offsets[role_state]
            if offset + count > len(exterior):
                raise SynthesisProgramLayoutError(
                    "factorized Ugi role size cannot hold fixed precursor attachments"
                )
            fixed_edges.extend(
                (core_index, exterior[offset + local_offset], bond) for local_offset in range(count)
            )
            attachment_offsets[role_state] += count
        return self._record(
            program_id=program_id,
            depth=depth,
            closure_count=closure_count,
            role_states=np.asarray([role for role, _ in node_specs], dtype=np.int64),
            core_positions=np.asarray([core for _, core in node_specs], dtype=np.int64),
            component_blocks=component_blocks,
            fixed_node_states=self._fixed_node_states[program_id],
            fixed_edges=fixed_edges,
            sample_index=sample_index,
        )

    @staticmethod
    def _select_fixed_edges(
        *,
        node_specs: Sequence[tuple[int, int]],
        block_indices: Sequence[int],
        fixed_signature: Sequence[tuple[Any, ...]],
    ) -> list[tuple[int, int, int]]:
        """Instantiate an anonymous fixed-edge multiset without retaining component identity."""

        selected: list[tuple[int, int, int]] = []
        used_pairs: set[tuple[int, int]] = set()
        degree: Counter[int] = Counter()
        for (left, right, bond), count in sorted(Counter(fixed_signature).items()):
            left_core, left_role, left_fixed = left
            right_core, right_role, right_fixed = right
            left_nodes = [
                index
                for index, (role, core) in enumerate(node_specs)
                if role == int(left_role)
                and core == int(left_core)
                and (core > 1) == bool(left_fixed)
            ]
            right_nodes = [
                index
                for index, (role, core) in enumerate(node_specs)
                if role == int(right_role)
                and core == int(right_core)
                and (core > 1) == bool(right_fixed)
            ]
            if not left_nodes or not right_nodes:
                raise SynthesisProgramLayoutError(
                    "fixed reaction-core signature has no compatible layout endpoints"
                )
            candidates = {
                (min(left_index, right_index), max(left_index, right_index))
                for left_index in left_nodes
                for right_index in right_nodes
                if left_index != right_index
            }
            candidates.difference_update(used_pairs)
            if len(candidates) < count:
                raise SynthesisProgramLayoutError(
                    "fixed reaction-core signature exceeds compatible layout edges"
                )

            def edge_key(pair: tuple[int, int]) -> tuple[int, int, int, int, int]:
                first, second = pair
                same_role = int(left_role) == int(right_role)
                same_block = block_indices[first] == block_indices[second]
                preferred = same_block if same_role else not same_block
                return (
                    int(not preferred),
                    degree[first] + degree[second],
                    max(degree[first], degree[second]),
                    first,
                    second,
                )

            # When one endpoint population equals the edge multiplicity, cover every such
            # reaction coordinate exactly once.  This is what binds each repeated BL tail to a
            # head junction without copying a particular head or tail identity.
            cover: list[int] | None = None
            if count == len(left_nodes):
                cover = left_nodes
            elif count == len(right_nodes):
                cover = right_nodes
            chosen: list[tuple[int, int]] = []
            if cover is not None:
                for endpoint in cover:
                    compatible = [pair for pair in candidates if endpoint in pair]
                    if not compatible:
                        raise SynthesisProgramLayoutError(
                            "fixed reaction-core coordinates cannot be matched one-to-one"
                        )
                    pair = min(compatible, key=edge_key)
                    chosen.append(pair)
                    candidates.remove(pair)
                    degree.update(pair)
            else:
                for _ in range(count):
                    pair = min(candidates, key=edge_key)
                    chosen.append(pair)
                    candidates.remove(pair)
                    degree.update(pair)
            for first, second in chosen:
                used_pairs.add((first, second))
                selected.append((first, second, int(bond)))
        return selected

    def _repeated_fixed_core_layout(
        self,
        *,
        program_id: str,
        depth: int,
        closure_count: int,
        blocks: Sequence[tuple[int, int, tuple[tuple[int, int], ...]]],
        fixed_signature: Sequence[tuple[Any, ...]],
        sample_index: int,
    ) -> SynthesisProgramGraphRecord:
        """Lay out one two-role repeated program with an adapter-fixed reaction junction."""

        by_role: dict[int, list[tuple[int, int, tuple[tuple[int, int], ...]]]] = defaultdict(list)
        for block in blocks:
            by_role[int(block[0])].append(block)
        if len(by_role) != 2:
            raise SynthesisProgramLayoutError(
                "repeated fixed-core layout currently requires exactly two precursor roles"
            )
        singleton_roles = [role for role, values in by_role.items() if len(values) == 1]
        repeated_roles = [role for role, values in by_role.items() if len(values) == depth]
        if depth == 1:
            # At depth one both roles are singletons.  The repeat component is the role with the
            # richer per-step core signature; this criterion is semantic and count-only.
            repeated_role = max(
                by_role,
                key=lambda role: sum(count for _, count in by_role[role][0][2]),
            )
            singleton_role = next(role for role in by_role if role != repeated_role)
        elif len(singleton_roles) == 1 and len(repeated_roles) == 1:
            singleton_role = singleton_roles[0]
            repeated_role = repeated_roles[0]
        else:
            raise SynthesisProgramLayoutError(
                "repeated fixed-core component multiplicity disagrees with program depth"
            )

        ordered_blocks = [*by_role[singleton_role], *by_role[repeated_role]]
        node_specs: list[tuple[int, int]] = []
        block_indices: list[int] = []
        component_blocks: list[SynthesisProgramComponentBlock] = []
        for block_index, (role, size, core_signature) in enumerate(ordered_blocks):
            labels = [state for state, count in core_signature for _ in range(count)]
            if size < len(labels):
                raise SynthesisProgramLayoutError(
                    "repeated component size cannot hold its fixed reaction core"
                )
            start = len(node_specs)
            node_specs.extend((role, state) for state in sorted(labels))
            node_specs.extend((role, 1) for _ in range(size - len(labels)))
            block_indices.extend([block_index] * size)
            component_blocks.append(
                SynthesisProgramComponentBlock(
                    role=self.vocabulary.role_states[role],
                    role_state=role,
                    start=start,
                    stop=len(node_specs),
                )
            )
        fixed_edges = self._select_fixed_edges(
            node_specs=node_specs,
            block_indices=block_indices,
            fixed_signature=fixed_signature,
        )
        return self._record(
            program_id=program_id,
            depth=depth,
            closure_count=closure_count,
            role_states=np.asarray([role for role, _ in node_specs], dtype=np.int64),
            core_positions=np.asarray([core for _, core in node_specs], dtype=np.int64),
            component_blocks=component_blocks,
            fixed_node_states=self._fixed_node_states[program_id],
            fixed_edges=fixed_edges,
            sample_index=sample_index,
        )

    def _fixed_core_layout(
        self,
        *,
        program_id: str,
        depth: int,
        closure_count: int,
        blocks: Sequence[tuple[int, int, tuple[tuple[int, int], ...]]],
        fixed_signature: Sequence[tuple[Any, ...]],
        sample_index: int,
    ) -> SynthesisProgramGraphRecord:
        role_multiplicities = Counter(role for role, _, _ in blocks)
        if all(count == 1 for count in role_multiplicities.values()):
            return self._ugi_layout(
                program_id=program_id,
                depth=depth,
                closure_count=closure_count,
                blocks=blocks,
                fixed_signature=fixed_signature,
                sample_index=sample_index,
            )
        return self._repeated_fixed_core_layout(
            program_id=program_id,
            depth=depth,
            closure_count=closure_count,
            blocks=blocks,
            fixed_signature=fixed_signature,
            sample_index=sample_index,
        )

    def _record(
        self,
        *,
        program_id: str,
        depth: int,
        closure_count: int,
        role_states: np.ndarray,
        core_positions: np.ndarray,
        component_blocks: Sequence[SynthesisProgramComponentBlock],
        fixed_node_states: Mapping[int, int],
        fixed_edges: Sequence[tuple[int, int, int]],
        sample_index: int,
    ) -> SynthesisProgramGraphRecord:
        node_count = len(role_states)
        if node_count > self.maximum_heavy_atoms or closure_count > self.maximum_closures:
            raise SynthesisProgramLayoutError("factorized layout exceeds declared support")
        nodes = np.zeros(node_count, dtype=np.int64)
        fixed_atoms = np.zeros(node_count, dtype=np.bool_)
        for index, core_state in enumerate(core_positions):
            if int(core_state) in fixed_node_states:
                nodes[index] = fixed_node_states[int(core_state)]
                fixed_atoms[index] = True
        parents = np.zeros(node_count, dtype=np.int64)
        parent_bonds = np.zeros(node_count, dtype=np.int64)
        fixed_parents = np.zeros(node_count, dtype=np.bool_)
        for left, right, bond in fixed_edges:
            parent, child = (left, right) if left < right else (right, left)
            if child == 0 or fixed_parents[child]:
                raise SynthesisProgramLayoutError(
                    "adapter-fixed layout does not form a rooted sparse tree"
                )
            parents[child] = parent
            parent_bonds[child] = bond
            fixed_parents[child] = True
        graph = SparseGraphRecord(
            structure_id=f"factorized-layout-{program_id}-{sample_index:08d}",
            canonical_smiles="",
            node_states=nodes,
            parents=parents,
            parent_bonds=parent_bonds,
            closure_left=np.zeros(closure_count, dtype=np.int64),
            closure_right=np.zeros(closure_count, dtype=np.int64),
            closure_bonds=np.zeros(closure_count, dtype=np.int64),
            edges=np.empty((0, 0), dtype=np.int8),
        )
        return SynthesisProgramGraphRecord(
            graph=graph,
            canonical_atom_order=np.arange(node_count, dtype=np.int64),
            program_id=program_id,
            program_state=self.vocabulary.program_to_index[program_id],
            program_depth=depth,
            role_states=role_states,
            core_position_states=core_positions,
            component_blocks=tuple(component_blocks),
            fixed_atom_mask=fixed_atoms,
            fixed_parent_bond_mask=fixed_parents,
            fixed_closure_bond_mask=np.zeros(closure_count, dtype=np.bool_),
        )

    def sample(
        self,
        program_id: str,
        *,
        sample_count: int,
        seed: int,
    ) -> tuple[SynthesisProgramGraphRecord, ...]:
        """Draw count-only layouts for one requested program without target-graph reuse."""

        if program_id not in self._distributions or sample_count < 1:
            raise SynthesisProgramLayoutError("invalid factorized layout request")
        rng = np.random.default_rng(seed)
        output: list[SynthesisProgramGraphRecord] = []
        for sample_index in range(sample_count):
            depth, closures, blocks, fixed = self._sample_fields(program_id, rng)
            if self._fixed_node_states[program_id]:
                record = self._fixed_core_layout(
                    program_id=program_id,
                    depth=depth,
                    closure_count=closures,
                    blocks=blocks,
                    fixed_signature=fixed,
                    sample_index=sample_index,
                )
            else:
                record = self._generic_layout(
                    program_id=program_id,
                    depth=depth,
                    closure_count=closures,
                    blocks=blocks,
                    sample_index=sample_index,
                    rng=rng,
                )
            output.append(record)
        return tuple(output)


__all__ = [
    "SynthesisProgramLayoutError",
    "SynthesisProgramLayoutPrior",
]
