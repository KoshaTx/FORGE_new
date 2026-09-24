"""Lossless compiled storage, preparation boundaries and malformed-cache rejection."""

import json
import shutil
import sqlite3
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.core.hashing import PinError
from forge.corpus.compose_lipid_source_view import pin
from forge.corpus.mapped_program_cache import (
    MappedProgramCache,
    _pack,
    compile_qualified_program_cache,
)
from forge.corpus.qualified_program_cache import (
    QualifiedProgramCache,
    QualifiedProgramCacheError,
    collate_qualified_program_examples,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phase1/compose_lipid_unified_preparation_v1"
pytestmark = pytest.mark.needs_vendor


@pytest.fixture(scope="module")
def prepared():
    if not (BASE / "verification.json").exists():
        pytest.skip("requires the documented compose_lipid_unified_preparation_v1 artifacts")
    inputs = {
        name: pin(ROOT, BASE / file)
        for name, file in (("population", "result.json"), ("verification", "verification.json"))
    }
    with tempfile.TemporaryDirectory(
        prefix="mapped-test-", dir=ROOT / "results/phase1"
    ) as directory:
        with QualifiedProgramCache(ROOT, **inputs) as source:
            ids = [
                row[0]
                for row in source._database.execute(
                    "SELECT min(target_id) FROM records GROUP BY program_key"
                )
            ]
            ids.append(
                source._database.execute(
                    "SELECT target_id FROM records ORDER BY atoms DESC LIMIT 1"
                ).fetchone()[0]
            )
            ids = list(dict.fromkeys(ids))
            expected = source.records(ids)
        output = Path(directory) / "compiled"
        manifest = compile_qualified_program_cache(
            ROOT, output, **inputs, target_ids=ids, chunk_size=3
        )
        yield dict(
            directory=Path(directory),
            output=output,
            manifest=manifest,
            inputs=inputs,
            ids=ids,
            expected=expected,
        )


def assert_examples_equal(actual, expected):
    assert actual.family == expected.family
    assert actual.component_instances == expected.component_instances
    assert actual.introduced_roles == expected.introduced_roles
    assert actual.record.program_id == expected.record.program_id
    assert actual.record.program_state == expected.record.program_state
    assert actual.record.program_depth == expected.record.program_depth
    assert actual.record.component_blocks == expected.record.component_blocks
    assert actual.record.graph.structure_id == expected.record.graph.structure_id
    assert actual.record.graph.canonical_smiles == expected.record.graph.canonical_smiles
    for name in (
        "node_states",
        "parents",
        "parent_bonds",
        "closure_left",
        "closure_right",
        "closure_bonds",
    ):
        np.testing.assert_array_equal(
            getattr(actual.record.graph, name), getattr(expected.record.graph, name)
        )
    for name in (
        "canonical_atom_order",
        "role_states",
        "core_position_states",
        "fixed_atom_mask",
        "fixed_parent_bond_mask",
        "fixed_closure_bond_mask",
    ):
        np.testing.assert_array_equal(getattr(actual.record, name), getattr(expected.record, name))


def test_every_binding_and_maximum_size_survive_chunk_boundaries(prepared):
    with MappedProgramCache(ROOT, manifest=pin(ROOT, prepared["manifest"])) as cache:
        assert cache.metadata["selection"] == "diagnostic"
        assert len(cache) == len(prepared["expected"])
        assert max(e.record.node_count for e in prepared["expected"]) == 250
        assert any(e.introduced_roles for e in prepared["expected"])
        actual = cache.records(list(range(len(cache))))
        for a, b in zip(actual, prepared["expected"], strict=True):
            assert_examples_equal(a, b)
        for mode in ("program", "null"):
            kwargs = dict(maximum_nodes=254, maximum_closures=12, conditioning_mode=mode)
            a = collate_qualified_program_examples(actual, **kwargs)
            b = collate_qualified_program_examples(prepared["expected"], **kwargs)
            assert set(a) == set(b)
            assert all(torch.equal(a[name], b[name]) for name in a)


def test_random_indices_ids_repeated_draws_and_owned_arrays(prepared):
    with MappedProgramCache(ROOT, manifest=pin(ROOT, prepared["manifest"])) as cache:
        assert all(
            isinstance(a, np.memmap) and not a.flags.writeable for a in cache.arrays.values()
        )
        indices = [len(cache) - 1, 0, len(cache) - 1]
        for index, value in zip(indices, cache.records(indices), strict=True):
            assert_examples_equal(value, prepared["expected"][index])
        first = cache.record(0)
        assert type(first.record.graph.node_states) is np.ndarray
        assert type(first.record.role_states) is np.ndarray
        assert first.record.graph.node_states.flags.owndata
        first.record.graph.node_states[:] = 0
        first.record.role_states[:] = 0
        assert_examples_equal(cache.record_by_id(prepared["ids"][0]), prepared["expected"][0])
        with pytest.raises(KeyError):
            cache.record_by_id("absent")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            cache._database.execute("DELETE FROM sources")
    with pytest.raises(QualifiedProgramCacheError, match="closed"):
        cache.record(0)


@pytest.mark.parametrize("index", [-1, 10**9, True, 1.5])
def test_invalid_index_is_rejected(prepared, index):
    with MappedProgramCache(ROOT, manifest=pin(ROOT, prepared["manifest"])) as cache:
        with pytest.raises(IndexError):
            cache.record(index)


def test_preparation_never_supplies_training_weights(prepared):
    with MappedProgramCache(ROOT, manifest=pin(ROOT, prepared["manifest"])) as cache:
        assert cache.metadata["training_admitted"] is False
        assert cache.metadata["sampling_weights_fitted"] is False
        assert not {"source_weights", "fold_states"} & cache.arrays.keys()
        with pytest.raises(QualifiedProgramCacheError, match="no admitted"):
            cache.training_measure({})


def test_compiler_refuses_overwrite_or_duplicate_graphs_and_cleans_partial_output(prepared):
    with pytest.raises(FileExistsError):
        compile_qualified_program_cache(ROOT, prepared["output"], **prepared["inputs"])
    destination = prepared["directory"] / "duplicates"
    with pytest.raises(sqlite3.IntegrityError):
        compile_qualified_program_cache(
            ROOT,
            destination,
            **prepared["inputs"],
            target_ids=[prepared["ids"][0]] * 2,
            chunk_size=1,
        )
    assert not destination.exists()
    assert not list(prepared["directory"].glob(".mapped-*"))


def test_numeric_narrowing_cannot_silently_truncate(prepared):
    example = prepared["expected"][0]
    states = example.record.graph.node_states.copy()
    states[0] = 65536
    bad = replace(
        example,
        record=replace(example.record, graph=replace(example.record.graph, node_states=states)),
    )
    with pytest.raises(QualifiedProgramCacheError, match="overflow: node_states"):
        _pack([bad])


@pytest.mark.parametrize("damage", ["checksum", "offsets", "semantic", "index", "admission"])
def test_corrupted_cache_is_rejected(prepared, damage):
    directory = prepared["directory"] / damage
    shutil.copytree(prepared["output"], directory)
    path = directory / "manifest.json"
    doc = json.loads(path.read_text())
    if damage == "checksum":
        with (directory / "node_states.npy").open("ab") as handle:
            handle.write(b"changed")
    elif damage in {"offsets", "semantic"}:
        field = "node_offsets" if damage == "offsets" else "role_states"
        array = np.load(directory / (field + ".npy"), allow_pickle=False)
        array[1] = 0 if damage == "offsets" else 65535
        np.save(directory / (field + ".npy"), array, allow_pickle=False)
    elif damage == "index":
        with sqlite3.connect(directory / "sources.sqlite") as database:
            database.execute("DELETE FROM sources WHERE idx=1")
    else:
        doc["training_admitted"] = True
    for name, value in doc["artifacts"].items():
        replacement = pin(ROOT, directory / name)
        if damage == "checksum":
            replacement["sha256"] = value["sha256"]
        doc["artifacts"][name] = replacement
    path.write_text(json.dumps(doc))
    with pytest.raises((PinError, QualifiedProgramCacheError)):
        MappedProgramCache(ROOT, manifest=pin(ROOT, path))
