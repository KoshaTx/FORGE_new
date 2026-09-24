"""Protect family normalization and logical-shard ordering for eight-GPU work."""

import copy

import pytest
import torch

from forge.model import compose_lipid_workers as workers
from tests.test_compose_lipid_run import run_args  # noqa: F401
from tests.test_compose_lipid_sharding import fixture
from tests.test_compose_lipid_training_data import prepared  # noqa: F401


def test_tasks_assign_four_shards_to_largest_family_without_omissions():
    groups = [dict(nodes=torch.zeros(8, n), node_mask=torch.ones(8, n)) for n in (7, 25, 10)]
    serial = workers.adaptive_family_tasks(groups, "adaptive_serial")
    parallel = workers.adaptive_family_tasks(groups, "adaptive_parallel")
    assert serial == [(0, (0, 1), 2), (1, (0, 1, 2, 3), 4), (2, (0, 1), 2)]
    assert len(parallel) == 8 and parallel[0] == (0, (0,), 2)
    for family, parts, divisions in serial:
        assert [p for f, ps, _ in parallel if f == family for p in ps] == list(parts)
        assert all(n == divisions for f, _, n in parallel if f == family)


def test_task_plan_rejects_indivisible_family():
    groups = [dict(nodes=torch.zeros(7, 10), node_mask=torch.ones(7, 10))] * 3
    with pytest.raises(ValueError, match="divisible"):
        workers.adaptive_family_tasks(groups, "adaptive_parallel")


def test_independent_four_shards_equal_serial_four_shards(run_args):  # noqa: F811
    model, clean, config, node, bond = fixture(run_args)
    options = dict(seed=810, device="cpu", divisions=4)
    serial = workers.split_gradient(model, clean, config, node, bond, parts=(0, 1, 2, 3), **options)
    parallel = workers.add_results(
        [
            workers.split_gradient(model, clean, config, node, bond, parts=(i,), **options)
            for i in range(4)
        ]
    )
    for i in (0, 2):
        torch.testing.assert_close(serial[i], parallel[i], atol=0, rtol=0)
    assert serial[1] == parallel[1]
    for key in serial[3]:
        torch.testing.assert_close(serial[3][key], parallel[3][key], atol=0, rtol=0)


def test_partition_count_preserves_objective_without_dropout(run_args):  # noqa: F811
    model, clean, config, node, bond = fixture(run_args)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "dropout") and isinstance(module.dropout, float):
            module.dropout = 0.0
    reference = workers.split_gradient(
        model,
        clean,
        config,
        node,
        bond,
        seed=811,
        device="cpu",
        parts=(0, 1),
        divisions=2,
    )
    candidate = workers.split_gradient(
        copy.deepcopy(model),
        clean,
        config,
        node,
        bond,
        seed=811,
        device="cpu",
        parts=(0, 1, 2, 3),
        divisions=4,
    )
    torch.testing.assert_close(reference[0], candidate[0], atol=2e-6, rtol=2e-4)
    torch.testing.assert_close(reference[2], candidate[2], atol=2e-6, rtol=2e-5)
    assert reference[1] == candidate[1]


def test_shared_family_buffers_own_input_and_signal_growth():
    pool = {}
    source = torch.arange(8, dtype=torch.float32)
    assert workers.share_batch(pool, {source.dtype: source})
    previous = pool[source.dtype]
    assert previous.is_shared()
    source.add_(100)
    torch.testing.assert_close(previous, torch.arange(8, dtype=torch.float32))
    assert not workers.share_batch(pool, {source.dtype: source[:4]})
    assert pool[source.dtype] is previous
    torch.testing.assert_close(previous[:4], source[:4])
    assert workers.share_batch(pool, {source.dtype: torch.zeros(16)})
    assert pool[source.dtype] is not previous
    assert pool[source.dtype].is_shared()
    torch.testing.assert_close(previous[:4], source[:4])
