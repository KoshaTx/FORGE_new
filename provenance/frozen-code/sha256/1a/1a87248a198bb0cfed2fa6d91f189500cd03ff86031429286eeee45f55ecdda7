"""Lookahead does not advance committed sampling or change restart batches."""

import copy

import numpy as np
import pytest
import torch

from forge.corpus.compose_lipid_source_view import pin
from forge.corpus.compose_lipid_tensor_cache import PreparedTensorCache
from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
from forge.model.compose_lipid_prefetch import PreparedBatches
from forge.model.family_exposure import balanced_families
from tests.test_compose_lipid_tensor_cache import cached  # noqa: F401
from tests.test_compose_lipid_training_data import prepared  # noqa: F401


def test_prefetch_keeps_draws_tensors_and_committed_restart_exact(cached, prepared):  # noqa: F811
    root, manifest, inputs, policy, *_ = cached
    cache = PreparedTensorCache(root, pin(root, manifest), inputs=inputs, policy=policy)
    seed = 174
    rng = np.random.default_rng(seed)
    initial = copy.deepcopy(rng.bit_generator.state)
    with ComposeLipidTrainingData(**prepared[0]) as data:
        families = len(cache.metadata["by_family"])
        options = dict(
            seed=seed,
            stop=8,
            families=families,
            families_per_batch=3,
            batch_size=6,
            depth=3,
            workers=2,
        )
        with PreparedBatches(data, cache, sampler_state=initial, start=0, **options) as stream:
            continuous = list(stream)
        assert rng.bit_generator.state == initial
        for step, ticket in enumerate(continuous):
            selected = balanced_families(families=families, per_batch=3, step=step, seed=seed)
            indices = data.sample_indices(6, rng, families_per_batch=3, family_selection=selected)
            np.testing.assert_array_equal(ticket.indices, indices)
            assert ticket.sampler_state == rng.bit_generator.state
            expected = data.batch(indices, **policy)
            assert all(torch.equal(value, ticket.tensors[key]) for key, value in expected.items())
        with PreparedBatches(data, cache, sampler_state=initial, start=0, **options) as stream:
            first = [next(stream) for _ in range(3)]
            checkpoint_state = copy.deepcopy(first[-1].sampler_state)
        with PreparedBatches(
            data, cache, sampler_state=checkpoint_state, start=3, **options
        ) as stream:
            resumed = list(stream)
        for a, b in zip(continuous, first + resumed, strict=True):
            assert a.step == b.step and a.sampler_state == b.sampler_state
            np.testing.assert_array_equal(a.indices, b.indices)
            assert all(torch.equal(value, b.tensors[key]) for key, value in a.tensors.items())


def test_worker_errors_propagate_without_changing_committed_rng(cached, prepared):  # noqa: F811
    class BrokenCache:
        def batch(self, indices):
            raise ValueError("bad prepared batch")

    state = copy.deepcopy(np.random.default_rng(9).bit_generator.state)
    with ComposeLipidTrainingData(**prepared[0]) as data:
        with PreparedBatches(
            data,
            BrokenCache(),
            sampler_state=state,
            seed=9,
            start=0,
            stop=3,
            families=len(cached[6]),
            families_per_batch=3,
            batch_size=6,
        ) as stream:
            with pytest.raises(ValueError, match="bad prepared batch"):
                next(stream)
    assert state == np.random.default_rng(9).bit_generator.state
