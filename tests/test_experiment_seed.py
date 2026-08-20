from __future__ import annotations

from experiments._runtime.seed import SeedPlan


def test_keyed_seeds_are_stable_and_distinct() -> None:
    plan = SeedPlan(root_seed=20260819, namespace="paper-model/train")
    assert plan.derive("batch", 4) == plan.derive("batch", 4)
    assert plan.derive("batch", 4) != plan.derive("batch", 5)
    assert plan.derive("batch", 4) != SeedPlan(20260820, "paper-model/train").derive("batch", 4)


def test_python_stream_is_independent_of_call_order() -> None:
    plan = SeedPlan(root_seed=11, namespace="matched-arms")
    left = plan.python("particle", "left", 7)
    _ = [plan.python("unrelated", index).random() for index in range(20)]
    right = plan.python("particle", "left", 7)
    assert [left.random() for _ in range(5)] == [right.random() for _ in range(5)]
