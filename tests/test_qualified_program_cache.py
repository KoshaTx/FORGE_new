"""Qualified cache provenance, vocabulary remapping and source-aware model batching."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.core.hashing import PinError
from forge.corpus.compose_lipid_source_view import pin
from forge.corpus.qualified_program_cache import (
    QualifiedProgramCache,
    QualifiedProgramCacheError,
    QualifiedProgramExample,
    collate_qualified_program_examples,
)
from forge.model.constitutional_program_graph import tensorize_constitutional_program_product
from forge.model.reaction_program_flow import collate_synthesis_program_records
from forge.model.synthesis_program_graph import SynthesisProgramGraphError

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results/phase1/compose_lipid_unified_preparation_v1"
pytestmark = pytest.mark.needs_vendor


@pytest.fixture(scope="module")
def source_pins():
    if not (ASSETS / "verification.json").exists():
        pytest.skip("requires the documented compose_lipid_unified_preparation_v1 artifacts")
    return {
        name: pin(ROOT, ASSETS / file)
        for name, file in (("population", "result.json"), ("verification", "verification.json"))
    }


@pytest.fixture(scope="module")
def examples(source_pins):
    with QualifiedProgramCache(ROOT, **source_pins, maximum_cached_shards=2) as cache:
        identities = [
            row[0]
            for row in cache._database.execute(
                "SELECT min(target_id) FROM records GROUP BY program_key"
            )
        ]
        yield cache, cache.records(identities)


def source_row(cache, identity):
    shard, index = cache._database.execute(
        "SELECT shard_id,row_index FROM records WHERE target_id=?", (identity,)
    ).fetchone()
    return cache._shard(shard)[1][index]


def test_every_binding_decodes_like_fresh_full_graph_tensorization(examples):
    cache, records = examples
    assert {e.record.program_id for e in records} == set(cache.vocabulary.program_states[1:])
    assert any(e.record.node_count > 96 for e in records)
    fresh = []
    for example in records:
        row = source_row(cache, example.record.graph.structure_id)
        a = row["semantic_replay"]["annotations"]
        graph = tensorize_constitutional_program_product(
            record_id=row["target_id"],
            program_id=row["program_key"],
            canonical_product_smiles=a["canonical_product_smiles"],
            atom_roles=a["atom_roles"],
            atom_core_positions=[
                "exterior" if core == "exterior" else row["program_key"] + ":" + core
                for core in a["core_positions"]
            ],
            program_depth=example.record.program_depth,
            vocabulary=cache.vocabulary,
            atom_vocabulary=cache.atom_vocabulary,
        )
        assert example.component_instances == tuple(tuple(v) for v in row["component_instances"])
        assert example.record.graph.edges.shape == (0, 0)
        for field in (
            "node_states",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        ):
            np.testing.assert_array_equal(
                getattr(example.record.graph, field), getattr(graph.graph, field)
            )
        fresh.append(replace(example, record=graph))
    actual = collate_qualified_program_examples(records, maximum_nodes=254, maximum_closures=12)
    reference = collate_qualified_program_examples(fresh, maximum_nodes=254, maximum_closures=12)
    assert set(actual) == set(reference)
    assert all(torch.equal(actual[key], reference[key]) for key in actual)
    assert all(isinstance(v, torch.Tensor) for v in actual.values())
    assert int(actual["node_mask"].sum()) == sum(e.record.node_count for e in records)


def test_introduced_and_fragmented_origins_keep_true_source_counts(examples):
    _, records = examples
    batch = collate_qualified_program_examples(records, maximum_closures=12)
    introduced, disconnected, repeated = False, False, False
    for i, example in enumerate(records):
        instances = batch["component_instance_states"][i, : example.record.node_count]
        assert len(set(instances.tolist()) - {0}) == sum(example.source_quantities.values())
        for block in example.record.component_blocks:
            values = instances[block.start : block.stop]
            if block.role in example.introduced_roles:
                introduced = True
                assert not values.any()
                assert not batch["component_position_states"][i, block.start : block.stop].any()
                assert not batch["repeat_group_states"][i, block.start : block.stop].any()
                assert torch.all(batch["core_position_states"][i, block.start : block.stop] > 1)
            else:
                assert torch.all(values > 0)
        for role, quantity in example.source_quantities.items():
            blocks = [b for b in example.record.component_blocks if b.role == role]
            if quantity == 1 and len(blocks) > 1:
                disconnected = True
                assert len({int(instances[b.start]) for b in blocks}) == 1
            if quantity > 1:
                repeated = True
                assert len({int(instances[b.start]) for b in blocks}) == quantity
    assert introduced and disconnected and repeated


def test_null_conditioning_matches_shared_graph_collator(examples):
    _, records = examples
    kwargs = dict(maximum_nodes=254, maximum_closures=12, conditioning_mode="null")
    expected = collate_synthesis_program_records([e.record for e in records], **kwargs)
    actual = collate_qualified_program_examples(records, **kwargs)
    assert all(torch.equal(actual[key], expected[key]) for key in expected)


def test_returned_arrays_are_owned_and_cache_bound_is_enforced(examples):
    cache, records = examples
    identity = records[0].record.graph.structure_id
    first = cache.record(identity)
    first.record.graph.node_states[:] = 0
    first.record.role_states[:] = 0
    second = cache.record(identity)
    assert np.any(second.record.graph.node_states)
    assert np.all(second.record.role_states > 0)
    for example in records:
        cache.record(example.record.graph.structure_id)
        assert len(cache._cached) <= cache.maximum_cached_shards


def test_random_access_preserves_requested_order_and_repeated_draws(examples):
    cache, records = examples
    ids = [records[-1].record.graph.structure_id, records[0].record.graph.structure_id]
    ids.append(ids[0])
    assert [e.record.graph.structure_id for e in cache.records(ids)] == ids
    with pytest.raises(KeyError):
        cache.record("not-an-eligible-record")
    with pytest.raises(QualifiedProgramCacheError, match="Unknown qualified family"):
        next(cache.iter_records(family="not-qualified"))


def test_preparation_cannot_supply_a_training_measure(examples):
    with pytest.raises(QualifiedProgramCacheError, match="no admitted"):
        examples[0].training_measure({})


@pytest.mark.parametrize("bound", [0, -1, True, 1.5])
def test_cache_memory_bound_is_explicit(source_pins, bound):
    with pytest.raises(QualifiedProgramCacheError, match="positive integer"):
        QualifiedProgramCache(ROOT, **source_pins, maximum_cached_shards=bound)


def test_pinned_verification_cannot_be_substituted(source_pins):
    altered = {**source_pins["verification"], "sha256": "0" * 64}
    with pytest.raises(PinError):
        QualifiedProgramCache(ROOT, population=source_pins["population"], verification=altered)


@pytest.mark.parametrize(
    "field,value",
    [
        ("old_projection", "heldout"),
        ("corrected_projection", "heldout"),
        ("eligible_for_program_preparation", False),
        ("training_admitted", True),
    ],
)
def test_source_protection_is_checked_before_exposing_a_record(
    source_pins, monkeypatch, field, value
):
    with QualifiedProgramCache(ROOT, **source_pins) as cache:
        lookup = cache._database.execute("SELECT * FROM records LIMIT 1").fetchone()
        arrays, source, shard = cache._shard(lookup[4])
        source = copy.deepcopy(source)
        source[lookup[5]][field] = value
        monkeypatch.setattr(cache, "_shard", lambda identity: (arrays, source, shard))
        with pytest.raises(QualifiedProgramCacheError, match="protected, unresolved or nonexact"):
            cache.record(lookup[0])


def test_saved_source_instance_corruption_is_rejected(source_pins, monkeypatch):
    with QualifiedProgramCache(ROOT, **source_pins) as cache:
        lookup = cache._database.execute(
            "SELECT * FROM records WHERE family='aldehyde_ugi3' LIMIT 1"
        ).fetchone()
        arrays, source, shard = cache._shard(lookup[4])
        arrays = {k: v.copy() for k, v in arrays.items()}
        lo, hi = arrays["node_offsets"][lookup[5] : lookup[5] + 2]
        arrays["source_instance_states"][lo:hi] = 1
        monkeypatch.setattr(cache, "_shard", lambda identity: (arrays, source, shard))
        with pytest.raises(QualifiedProgramCacheError, match="Saved source coordinates changed"):
            cache.record(lookup[0])


def test_global_component_identity_never_changes_model_inputs(examples):
    _, records = examples
    changed = [
        replace(
            e,
            component_instances=tuple(
                (r, "another-nonneural-id", q) for r, _, q in e.component_instances
            ),
        )
        for e in records
    ]
    first = collate_qualified_program_examples(records, maximum_closures=12)
    second = collate_qualified_program_examples(changed, maximum_closures=12)
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_incomplete_quantities_are_rejected_even_when_conditioning_is_null(examples):
    example = examples[1][0]
    invalid = QualifiedProgramExample(example.record, example.family, (), ())
    with pytest.raises(SynthesisProgramGraphError, match="cover every product origin"):
        collate_qualified_program_examples([invalid], maximum_closures=12, conditioning_mode="null")
