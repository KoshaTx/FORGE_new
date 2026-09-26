"""Lookahead does not advance committed sampling or change restart batches."""

import copy

import numpy as np
import pytest
import torch

from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin
from forge.corpus.compose_lipid_tensor_cache import PreparedTensorCache
from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
from forge.model.compose_lipid_prefetch import TAPE_SCHEMA, PreparedBatches, load_presentation_tape
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


def test_explicit_tape_prefetch_never_draws_or_commits_lookahead(
    cached, prepared, monkeypatch  # noqa: F811
):
    root, manifest, inputs, policy, *_ = cached
    cache = PreparedTensorCache(root, pin(root, manifest), inputs=inputs, policy=policy)
    state = copy.deepcopy(np.random.default_rng(23).bit_generator.state)
    with ComposeLipidTrainingData(**prepared[0]) as data:
        rng = np.random.default_rng(17)
        tape = np.stack([data.sample_indices(6, rng, families_per_batch=3) for _ in range(7)])

        def forbid(*args, **kwargs):
            raise AssertionError("Explicit tape must not make implicit new draws")

        monkeypatch.setattr(data, "sample_indices", forbid)
        options = dict(
            sampler_state=state,
            seed=23,
            stop=7,
            families=len(cached[6]),
            families_per_batch=3,
            batch_size=6,
            depth=4,
            workers=2,
            presentation_indices=tape,
        )
        with PreparedBatches(data, cache, start=0, **options) as stream:
            continuous = list(stream)
        with PreparedBatches(data, cache, start=0, **options) as stream:
            committed = [next(stream) for _ in range(2)]
        with PreparedBatches(data, cache, start=2, **options) as stream:
            resumed = list(stream)
        for expected, actual in zip(continuous, committed + resumed, strict=True):
            assert expected.step == actual.step
            assert expected.sampler_state == actual.sampler_state == state
            np.testing.assert_array_equal(actual.indices, tape[actual.step])
            assert all(torch.equal(v, actual.tensors[k]) for k, v in expected.tensors.items())


@pytest.mark.parametrize(
    "defect", ["negative", "fractional", "outside", "wrong_family", "excluded", "population"]
)
def test_authenticated_tape_rejects_source_and_index_drift(prepared, defect):  # noqa: F811
    with ComposeLipidTrainingData(**prepared[0]) as data:
        root = prepared[0]["repo"]
        families = [
            r[0]
            for r in data._database.execute("SELECT DISTINCT family FROM weights ORDER BY family")
        ]
        groups = np.array([[0, 1, 2]])
        indices = data.sample_indices(
            6, np.random.default_rng(8), families_per_batch=3, family_selection=groups[0]
        )[None]
        inputs = {k: prepared[0][k] for k in ("population", "verification", "measure")}
        config = {
            "inputs": inputs,
            "runtime": {"optimizer_steps": 1, "batch_size": 6, "families_per_batch": 3},
        }
        doc = {
            "schema_version": TAPE_SCHEMA,
            "inputs": inputs,
            "families": families,
            "optimizer_steps": 1,
            "batch_size": 6,
            "families_per_batch": 3,
            "indices_key": "indices",
            "families_key": "groups",
            "excluded_target_ids": [],
            "presentations_by_family": {f: 2 if i < 3 else 0 for i, f in enumerate(families)},
        }
        if defect == "negative":
            indices[0, 0] = -1
        elif defect == "fractional":
            indices = indices.astype(float)
            indices[0, 0] += 0.5
        elif defect == "outside":
            indices[0, 0] = len(data)
        elif defect == "wrong_family":
            groups[0, 2] = 3
            doc["presentations_by_family"] = {
                f: 2 if i in (0, 1, 3) else 0 for i, f in enumerate(families)
            }
        elif defect == "excluded":
            doc["excluded_target_ids"] = [
                data._database.execute(
                    "SELECT target_id FROM weights WHERE record_index=?", (int(indices[0, 0]),)
                ).fetchone()[0]
            ]
        else:
            doc["inputs"] = {}
        np.savez(root / "tape.npz", indices=indices, groups=groups)
        doc["artifact"] = pin(root, root / "tape.npz")
        write_json(root / "tape.json", doc)
        with pytest.raises(ValueError, match="[Pp]resentation tape"):
            load_presentation_tape(root, pin(root, root / "tape.json"), config=config, data=data)
