from __future__ import annotations

import itertools
import random

import pytest

from forge.product.v5_closure_feasibility import (
    ClosureFeasibilityError,
    closure_edge_retains_exact_completion,
    exact_closure_completion,
)


def _path_tree(node_count: int) -> tuple[tuple[int, int], ...]:
    return tuple((index, index + 1) for index in range(node_count - 1))


def _brute_force_feasible(
    node_count: int,
    tree_edges: tuple[tuple[int, int], ...],
    capacities: tuple[int, ...],
    closure_count: int,
    prefix: tuple[tuple[int, int], ...] = (),
) -> bool:
    forbidden = set(tree_edges) | set(prefix)
    after = prefix[-1] if prefix else None
    candidates = [
        (left, right)
        for left in range(node_count - 1)
        for right in range(left + 1, node_count)
        if (left, right) not in forbidden and (after is None or (left, right) > after)
    ]
    for chosen in itertools.combinations(candidates, closure_count):
        used = [0] * node_count
        for left, right in chosen:
            used[left] += 1
            used[right] += 1
        if all(count <= capacity for count, capacity in zip(used, capacities, strict=True)):
            return True
    return False


def test_exact_completion_constructs_strictly_ordered_non_tree_edges() -> None:
    result = exact_closure_completion(
        node_count=6,
        tree_edges=_path_tree(6),
        residual_capacities=(1, 1, 1, 1, 1, 1),
        remaining_closures=3,
    )

    assert result.feasible
    assert result.completion == tuple(sorted(result.completion))
    assert len(set(result.completion)) == 3
    assert not (set(result.completion) & set(_path_tree(6)))


def test_global_capacity_infeasibility_is_rejected() -> None:
    result = exact_closure_completion(
        node_count=4,
        tree_edges=_path_tree(4),
        residual_capacities=(1, 1, 1, 0),
        remaining_closures=2,
    )

    assert not result.feasible
    assert result.completion == ()


def test_exact_fallback_resolves_a_lexicographic_greedy_trap() -> None:
    result = exact_closure_completion(
        node_count=4,
        tree_edges=((0, 1), (0, 2), (0, 3)),
        residual_capacities=(0, 1, 1, 2),
        remaining_closures=2,
    )

    assert result.feasible
    assert not result.greedy_succeeded
    assert result.exact_states_visited > 0
    assert result.completion == ((1, 3), (2, 3))


def test_candidate_mask_rejects_edge_that_strands_residual_problem() -> None:
    tree = ((0, 1), (0, 2), (0, 3), (0, 4))
    capacities = (0, 1, 1, 1, 1)
    # Choosing (1, 2) first leaves (3, 4), so this candidate is feasible.
    feasible = closure_edge_retains_exact_completion(
        candidate=(1, 2),
        node_count=5,
        tree_edges=tree,
        residual_capacities=capacities,
        remaining_closures_including_candidate=2,
    )
    # Choosing (2, 4) leaves only the earlier disjoint edge (1, 3).
    stranded = closure_edge_retains_exact_completion(
        candidate=(2, 4),
        node_count=5,
        tree_edges=tree,
        residual_capacities=capacities,
        remaining_closures_including_candidate=2,
    )

    assert feasible.feasible
    assert feasible.completion == ((1, 2), (3, 4))
    assert not stranded.feasible


def test_exact_solver_agrees_with_brute_force_on_random_small_problems() -> None:
    generator = random.Random(20260731)
    for node_count in range(2, 8):
        tree = _path_tree(node_count)
        for _ in range(40):
            capacities = tuple(generator.randrange(3) for _ in range(node_count))
            closure_count = generator.randrange(4)
            expected = _brute_force_feasible(
                node_count,
                tree,
                capacities,
                closure_count,
            )
            observed = exact_closure_completion(
                node_count=node_count,
                tree_edges=tree,
                residual_capacities=capacities,
                remaining_closures=closure_count,
            )
            assert observed.feasible is expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "node_count": 3,
                "tree_edges": ((0, 1), (1, 2)),
                "residual_capacities": (1, -1, 1),
                "remaining_closures": 1,
            },
            "nonnegative integers",
        ),
        (
            {
                "node_count": 3,
                "tree_edges": ((0, 1), (1, 2)),
                "residual_capacities": (1, 1, 1),
                "remaining_closures": 1,
                "prefix": ((0, 2), (0, 2)),
            },
            "strictly lexicographically ordered",
        ),
        (
            {
                "node_count": 3,
                "tree_edges": ((0, 1),),
                "residual_capacities": (1, 1, 1),
                "remaining_closures": 1,
            },
            "node_count - 1",
        ),
        (
            {
                "node_count": 4,
                "tree_edges": ((0, 1), (1, 2), (0, 2)),
                "residual_capacities": (1, 1, 1, 1),
                "remaining_closures": 1,
            },
            "acyclic",
        ),
    ],
)
def test_malformed_problems_fail_cleanly(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ClosureFeasibilityError, match=message):
        exact_closure_completion(**kwargs)  # type: ignore[arg-type]
