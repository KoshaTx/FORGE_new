"""Exact sparse closure feasibility for the Phase 1 V5 morphology model.

The V5 representation generates a connected tree first and then adds a small
number of non-tree edges.  This module answers the bounded decision problem:
can a lexicographically ordered closure prefix be completed without exceeding
per-node coarse heavy-degree capacity?

The implementation deliberately stores no dense pairwise matrix.  Candidate
edges are scanned lazily.  A fast constructive pass handles the dense,
low-cycle-rank regime expected for lipids; an exhaustive memoized fallback
preserves exactness for greedy traps and infeasible states.  There is no search
cutoff, so a returned ``False`` is a proof within the declared coarse topology
model, not a timeout or heuristic rejection.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from numbers import Integral

Edge = tuple[int, int]


class ClosureFeasibilityError(ValueError):
    """Raised when a closure-feasibility problem is malformed."""


@dataclass(frozen=True)
class ClosureCompletionResult:
    """Exact feasibility decision and one canonical completion when feasible."""

    feasible: bool
    completion: tuple[Edge, ...]
    greedy_succeeded: bool
    exact_states_visited: int


def _normalize_edge(edge: Sequence[int], node_count: int, *, label: str) -> Edge:
    if len(edge) != 2:
        raise ClosureFeasibilityError(f"{label} edge must contain exactly two endpoints")
    left, right = edge
    if isinstance(left, bool) or isinstance(right, bool):
        raise ClosureFeasibilityError(f"{label} endpoints must be integer node indices")
    if not isinstance(left, Integral) or not isinstance(right, Integral):
        raise ClosureFeasibilityError(f"{label} endpoints must be integer node indices")
    left, right = int(left), int(right)
    if not 0 <= left < right < node_count:
        raise ClosureFeasibilityError(f"{label} edge must satisfy 0 <= left < right < node_count")
    return left, right


def _normalize_problem(
    *,
    node_count: int,
    tree_edges: Iterable[Sequence[int]],
    residual_capacities: Sequence[int],
    remaining_closures: int,
    prefix: Iterable[Sequence[int]],
) -> tuple[frozenset[Edge], tuple[int, ...], tuple[Edge, ...]]:
    if isinstance(node_count, bool) or not isinstance(node_count, Integral) or node_count < 1:
        raise ClosureFeasibilityError("node_count must be a positive integer")
    node_count = int(node_count)
    if (
        isinstance(remaining_closures, bool)
        or not isinstance(remaining_closures, Integral)
        or remaining_closures < 0
    ):
        raise ClosureFeasibilityError("remaining_closures must be a nonnegative integer")
    if len(residual_capacities) != node_count:
        raise ClosureFeasibilityError("residual_capacities must have length node_count")
    capacities: list[int] = []
    for capacity in residual_capacities:
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 0:
            raise ClosureFeasibilityError("residual capacities must be nonnegative integers")
        capacities.append(int(capacity))

    normalized_tree = frozenset(
        _normalize_edge(edge, node_count, label="tree") for edge in tree_edges
    )
    if node_count > 1 and len(normalized_tree) != node_count - 1:
        raise ClosureFeasibilityError("tree_edges must contain node_count - 1 unique edges")
    if node_count == 1 and normalized_tree:
        raise ClosureFeasibilityError("a one-node tree cannot contain edges")
    parents = list(range(node_count))

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for left, right in normalized_tree:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            raise ClosureFeasibilityError("tree_edges must be acyclic")
        parents[right_root] = left_root
    if node_count > 1 and len({find(node) for node in range(node_count)}) != 1:
        raise ClosureFeasibilityError("tree_edges must be connected")

    normalized_prefix = tuple(_normalize_edge(edge, node_count, label="prefix") for edge in prefix)
    if normalized_prefix != tuple(sorted(set(normalized_prefix))):
        raise ClosureFeasibilityError(
            "prefix closure edges must be unique and strictly lexicographically ordered"
        )
    if any(edge in normalized_tree for edge in normalized_prefix):
        raise ClosureFeasibilityError("prefix closure edge cannot duplicate a tree edge")
    return normalized_tree, tuple(capacities), normalized_prefix


def _iter_candidate_edges(
    node_count: int,
    tree_edges: frozenset[Edge],
    capacities: tuple[int, ...],
    after: Edge | None,
) -> Iterator[Edge]:
    """Yield eligible non-tree pairs in strict lexicographic order."""

    for left in range(node_count - 1):
        if capacities[left] <= 0:
            continue
        for right in range(left + 1, node_count):
            edge = (left, right)
            if after is not None and edge <= after:
                continue
            if capacities[right] <= 0 or edge in tree_edges:
                continue
            yield edge


def _upper_bound(
    node_count: int,
    tree_edges: frozenset[Edge],
    capacities: tuple[int, ...],
    after: Edge | None,
) -> int:
    """Return a safe upper bound on additional feasible closure cardinality."""

    total_capacity_bound = sum(capacities) // 2
    if total_capacity_bound == 0:
        return 0
    incident = [0] * node_count
    candidate_count = 0
    for left, right in _iter_candidate_edges(
        node_count,
        tree_edges,
        capacities,
        after,
    ):
        candidate_count += 1
        incident[left] += 1
        incident[right] += 1
    incidence_bound = (
        sum(min(capacity, degree) for capacity, degree in zip(capacities, incident, strict=True))
        // 2
    )
    return min(total_capacity_bound, candidate_count, incidence_bound)


def _greedy_completion(
    node_count: int,
    tree_edges: frozenset[Edge],
    capacities: tuple[int, ...],
    after: Edge | None,
    need: int,
) -> tuple[Edge, ...] | None:
    """Construct a fast lexicographic completion; failure is not a proof."""

    remaining = list(capacities)
    selected: list[Edge] = []
    cursor = after
    while len(selected) < need:
        edge = next(
            _iter_candidate_edges(
                node_count,
                tree_edges,
                tuple(remaining),
                cursor,
            ),
            None,
        )
        if edge is None:
            return None
        left, right = edge
        remaining[left] -= 1
        remaining[right] -= 1
        selected.append(edge)
        cursor = edge
    return tuple(selected)


def exact_closure_completion(
    *,
    node_count: int,
    tree_edges: Iterable[Sequence[int]],
    residual_capacities: Sequence[int],
    remaining_closures: int,
    prefix: Iterable[Sequence[int]] = (),
) -> ClosureCompletionResult:
    """Find an exact residual completion for a sparse closure prefix.

    ``residual_capacities`` are capacities *after* the supplied prefix has been
    consumed.  Future closure pairs must be lexicographically greater than the
    last prefix pair.  The guarantee concerns only topology and coarse degree
    capacity; exact bond-order, aromaticity, and sanitization are later gates.
    """

    normalized_tree, capacities, normalized_prefix = _normalize_problem(
        node_count=node_count,
        tree_edges=tree_edges,
        residual_capacities=residual_capacities,
        remaining_closures=remaining_closures,
        prefix=prefix,
    )
    after = normalized_prefix[-1] if normalized_prefix else None
    if remaining_closures == 0:
        return ClosureCompletionResult(True, (), True, 0)
    if _upper_bound(node_count, normalized_tree, capacities, after) < remaining_closures:
        return ClosureCompletionResult(False, (), False, 0)

    greedy = _greedy_completion(
        node_count,
        normalized_tree,
        capacities,
        after,
        remaining_closures,
    )
    if greedy is not None:
        return ClosureCompletionResult(True, greedy, True, 0)

    states_visited = 0

    @cache
    def search(
        state_capacities: tuple[int, ...],
        state_after: Edge | None,
        need: int,
    ) -> tuple[Edge, ...] | None:
        nonlocal states_visited
        states_visited += 1
        if need == 0:
            return ()
        if (
            _upper_bound(
                node_count,
                normalized_tree,
                state_capacities,
                state_after,
            )
            < need
        ):
            return None
        greedy_tail = _greedy_completion(
            node_count,
            normalized_tree,
            state_capacities,
            state_after,
            need,
        )
        if greedy_tail is not None:
            return greedy_tail
        for edge in _iter_candidate_edges(
            node_count,
            normalized_tree,
            state_capacities,
            state_after,
        ):
            left, right = edge
            reduced = list(state_capacities)
            reduced[left] -= 1
            reduced[right] -= 1
            tail = search(tuple(reduced), edge, need - 1)
            if tail is not None:
                return (edge, *tail)
        return None

    completion = search(capacities, after, remaining_closures)
    return ClosureCompletionResult(
        feasible=completion is not None,
        completion=completion or (),
        greedy_succeeded=False,
        exact_states_visited=states_visited,
    )


def closure_edge_retains_exact_completion(
    *,
    candidate: Sequence[int],
    node_count: int,
    tree_edges: Iterable[Sequence[int]],
    residual_capacities: Sequence[int],
    remaining_closures_including_candidate: int,
    prefix: Iterable[Sequence[int]] = (),
) -> ClosureCompletionResult:
    """Test whether one next edge leaves an exact residual completion."""

    normalized_tree, capacities, normalized_prefix = _normalize_problem(
        node_count=node_count,
        tree_edges=tree_edges,
        residual_capacities=residual_capacities,
        remaining_closures=remaining_closures_including_candidate,
        prefix=prefix,
    )
    if remaining_closures_including_candidate < 1:
        raise ClosureFeasibilityError("remaining_closures_including_candidate must be at least one")
    edge = _normalize_edge(candidate, node_count, label="candidate")
    if normalized_prefix and edge <= normalized_prefix[-1]:
        return ClosureCompletionResult(False, (), False, 0)
    if edge in normalized_tree:
        return ClosureCompletionResult(False, (), False, 0)
    left, right = edge
    if capacities[left] <= 0 or capacities[right] <= 0:
        return ClosureCompletionResult(False, (), False, 0)
    reduced = list(capacities)
    reduced[left] -= 1
    reduced[right] -= 1
    residual = exact_closure_completion(
        node_count=node_count,
        tree_edges=normalized_tree,
        residual_capacities=reduced,
        remaining_closures=remaining_closures_including_candidate - 1,
        prefix=(*normalized_prefix, edge),
    )
    if not residual.feasible:
        return residual
    return ClosureCompletionResult(
        feasible=True,
        completion=(edge, *residual.completion),
        greedy_succeeded=residual.greedy_succeeded,
        exact_states_visited=residual.exact_states_visited,
    )
