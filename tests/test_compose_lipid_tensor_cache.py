"""Prepared storage preserves whole graphs, duplicate draws and complete cohort identity."""

import copy
import json

import pytest
import torch

from forge.core.hashing import PinError
from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin
from forge.corpus.compose_lipid_tensor_cache import (
    PreparedTensorCache,
    finish_tensor_cache,
    pack_tensors,
    write_tensor_shard,
)
from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
from tests.test_compose_lipid_training_data import prepared  # noqa: F401


@pytest.fixture
def cached(prepared):  # noqa: F811
    args, _, _ = prepared
    with ComposeLipidTrainingData(**args) as data:
        policy = dict(
            maximum_nodes=128,
            maximum_closures=12,
            node_padding="batch",
            repeat_supervision="exact_fragment",
            core_conditioning="qualified_core",
        )
        clean = data.batch(list(range(len(data))), **policy)
        n = len(data)
    root = args["repo"]
    output = root / "prepared-tensors"
    shards = [
        write_tensor_shard(root, output, start, {k: v[start : start + 3] for k, v in clean.items()})
        for start in range(0, n, 3)
    ]
    census = {
        int(f): int((clean["family_states"] == f).sum()) for f in clean["family_states"].unique()
    }
    inputs = {k: args[k] for k in ("population", "verification", "measure")}
    manifest = finish_tensor_cache(
        root,
        output,
        shards=shards,
        records=n,
        inputs=inputs,
        policy=policy,
        expected_by_family=census,
        implementation=pin(root, root / "forge/model/compose_lipid_training.py"),
    )
    return root, manifest, inputs, policy, clean, shards, census


def test_prepared_gather_matches_every_tensor_and_retains_large_graphs(cached):
    root, manifest, inputs, policy, clean, _, _ = cached
    cache = PreparedTensorCache(root, pin(root, manifest), inputs=inputs, policy=policy)
    indices = [len(clean["nodes"]) - 1, 0, 2, 0]
    actual = cache.batch(indices)
    width = int(clean["node_mask"][indices].sum(1).max())
    from forge.corpus.compose_lipid_tensor_cache import NODE_FIELDS

    for key, value in clean.items():
        expected = value[indices, :width] if key in NODE_FIELDS else value[indices]
        assert torch.equal(actual[key], expected), key
    assert int(actual["node_mask"].sum(1).max()) == 100
    actual["nodes"].zero_()
    assert torch.equal(cache.batch(indices)["nodes"], clean["nodes"][indices, :width])


def test_shard_receipt_accepts_a_volume_mount_alias(cached):
    root, _, _, _, clean, *_ = cached
    alias = root.parent / "volume-alias"
    alias.symlink_to(root, target_is_directory=True)
    row = write_tensor_shard(alias, alias / "alias-shard", 0, clean)
    assert row["artifact"] == pin(root, root / "alias-shard/shard-000000000.npz")


@pytest.mark.parametrize("bad", [[-1], [100000], [True], [1.5], []])
def test_prepared_cache_rejects_invalid_indices(cached, bad):
    root, manifest, inputs, policy, *_ = cached
    cache = PreparedTensorCache(root, pin(root, manifest), inputs=inputs, policy=policy)
    with pytest.raises(IndexError):
        cache.batch(bad)


def test_cache_rejects_missing_or_overlapping_shards(cached):
    root, _, inputs, policy, clean, shards, census = cached
    for rows in (shards[:-1], [shards[0], *shards]):
        with pytest.raises(ValueError, match="missing|overlapping|complete"):
            finish_tensor_cache(
                root,
                root / "other",
                shards=rows,
                records=len(clean["nodes"]),
                inputs=inputs,
                policy=policy,
                expected_by_family=census,
                implementation={},
            )


def test_cache_rejects_policy_drift_corruption_and_memory_overflow(cached):
    root, manifest, inputs, policy, *_ = cached
    with pytest.raises(ValueError, match="policy"):
        PreparedTensorCache(
            root, pin(root, manifest), inputs=inputs, policy=policy | {"maximum_nodes": 96}
        )
    with pytest.raises(ValueError, match="RAM budget"):
        PreparedTensorCache(
            root, pin(root, manifest), inputs=inputs, policy=policy, maximum_bytes=1
        )
    doc = json.loads(manifest.read_text())
    with (root / doc["shards"][0]["artifact"]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(PinError):
        PreparedTensorCache(root, pin(root, manifest), inputs=inputs, policy=policy)


def test_storage_overflow_and_rehashed_incomplete_cache_fail(cached):
    root, manifest, inputs, policy, clean, *_ = cached
    bad = copy.deepcopy(clean)
    bad["repeat_atom_groups"][0, 0] = 2**40
    with pytest.raises(ValueError, match="overflow"):
        pack_tensors(bad)
    doc = json.loads(manifest.read_text())
    doc["shards"].pop()
    write_json(manifest, doc)
    with pytest.raises(ValueError, match="Incomplete"):
        PreparedTensorCache(root, pin(root, manifest), inputs=inputs, policy=policy)
