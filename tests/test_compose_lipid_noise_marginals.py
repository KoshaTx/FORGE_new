"""Weighted sparse marginals using a real measure/admission and synthetic shard reader."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from experiments.phase1.multireaction.compose_lipid_run import training_marginals
from forge.corpus import compose_lipid_noise_marginals as noise
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_training_data import (
    ComposeLipidTrainingData,
    ComposeLipidTrainingDataError,
)
from tests.test_compose_lipid_training_data import (
    prepared,  # noqa: F401
    rewrite_admission,
    rewrite_weights,
)
from tests.test_source_instance_coordinates import record

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def complete(prepared):  # noqa: F811
    args, rows, examples = prepared
    repo = args["repo"]
    snapshot = json.loads((repo / "source-snapshot.json").read_text())
    for name in (
        "forge/corpus/compose_lipid_noise_marginals.py",
        "forge/model/sparse_topology_feasibility.py",
    ):
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        snapshot[name] = pin(repo, destination)["sha256"]
    dump(repo / "source-snapshot.json", snapshot)
    validation = json.loads((repo / "validation.json").read_text())
    validation["inputs"]["source-snapshot.json"] = pin(repo, repo / "source-snapshot.json")
    dump(repo / "validation.json", validation)
    rewrite_admission(
        args, lambda d: d["inputs"].update(validation=pin(repo, repo / "validation.json"))
    )
    # All ordinary records have six C and one N. Add real triple/double/closure
    # bonds while preserving that inventory and the seven-atom graph size.
    for index, smiles in ((2, "C#CN(C=C)CC"), (3, "C1CCNCCC1")):
        mol = Chem.MolFromSmiles(smiles)
        roles = ["head" if a.GetSymbol() == "N" else "tail" for a in mol.GetAtoms()]
        cores = ["core" if r == "head" else "exterior" for r in roles]
        value = record(smiles, roles, cores)[0]
        identity = rows[index][0]
        value = replace(value, graph=replace(value.graph, structure_id=identity))
        examples[identity] = replace(examples[identity], record=value)
    return args, rows, examples


def test_weighted_counts_use_graph_measure_and_exclude_root_bond(complete):
    args, _, _ = complete
    output = noise.compile_noise_marginals(**args, output=args["repo"] / "noise.json")
    result = json.loads(output.read_text())
    # Family 0 has a 100-C chain plus two 6-C/1-N graphs, each mass 1/69.
    # Each of the other 22 families has one 6-C/1-N graph, mass 1/23.
    np.testing.assert_allclose(result["node"], [127 / 144, 17 / 144], rtol=0, atol=1e-15)
    # Exactly 508:1:1 weighted bond units: each ring closure counts once,
    # and the 25 serialization root sentinels contribute zero bonds.
    np.testing.assert_allclose(result["bond"], [508 / 510, 1 / 510, 1 / 510], rtol=0, atol=1e-15)
    assert result["records"] == 25 and result["maximum_heavy_atoms"] == 100
    assert result["training_calls"] == 0 and result["model_allocations"] == 0
    assert result["inputs"] == {k: args[k] for k in ("population", "measure")}
    assert result["qualification_inputs"] == {k: args[k] for k in ("verification", "admission")}
    assert result["by_family"]["family_00"] == 3


def test_compilation_is_byte_deterministic_and_accepted_by_existing_runner(complete):
    args, _, _ = complete
    first = noise.compile_noise_marginals(**args, output=args["repo"] / "first.json")
    second = noise.compile_noise_marginals(**args, output=args["repo"] / "second.json")
    assert first.read_bytes() == second.read_bytes()
    with ComposeLipidTrainingData(**args) as data:
        cfg = {
            "inputs": {"noise_marginals": pin(args["repo"], first)},
            "model": {"bond_classes": 3},
        }
        node, bond = training_marginals(args["repo"], cfg, data, torch.device("cpu"))
        assert node.dtype == bond.dtype == torch.float32
        torch.testing.assert_close(node, torch.tensor([127 / 144, 17 / 144]))
        torch.testing.assert_close(bond, torch.tensor([508 / 510, 1 / 510, 1 / 510]))


def test_stream_preserves_compiled_identity_order_and_probabilities(complete):
    args, rows, _ = complete
    with ComposeLipidTrainingData(**args) as data:
        records = list(data.iter_weighted_examples())
        assert [e.record.graph.structure_id for e, _ in records] == [r[0] for r in rows]
        assert [p for _, p in records] == [1 / 69] * 3 + [1 / 23] * 22
    with pytest.raises(ComposeLipidTrainingDataError, match="closed"):
        next(data.iter_weighted_examples())


def test_incomplete_admission_fails_before_graph_loading_or_output(complete, monkeypatch):
    args, _, _ = complete
    rewrite_admission(args, lambda d: d.update(training_ready=False))
    monkeypatch.setattr(
        ComposeLipidTrainingData,
        "iter_weighted_examples",
        lambda self: pytest.fail("graphs must remain unread"),
    )
    output = args["repo"] / "not-created" / "noise.json"
    with pytest.raises(ComposeLipidTrainingDataError, match="Final training"):
        noise.compile_noise_marginals(**args, output=output)
    assert not output.parent.exists()


def test_rehashed_probability_tampering_cannot_change_noise_sources(complete):
    args, _, _ = complete
    rewrite_weights(args, lambda db: db.execute("UPDATE weights SET probability=0.04"))
    output = args["repo"] / "noise.json"
    with pytest.raises(ComposeLipidTrainingDataError, match="probability changed"):
        noise.compile_noise_marginals(**args, output=output)
    assert not output.exists()


def test_absent_bond_states_are_not_filled_with_pseudocounts(complete):
    args, rows, examples = complete
    single = examples[rows[1][0]].record
    for index in (2, 3):
        identity = rows[index][0]
        examples[identity] = replace(
            examples[identity],
            record=replace(single, graph=replace(single.graph, structure_id=identity)),
        )
    with pytest.raises(ValueError, match="Every declared bond state"):
        noise.compile_noise_marginals(**args, output=args["repo"] / "noise.json")
    assert not (args["repo"] / "noise.json").exists()


@pytest.mark.parametrize("omit", [0, 24])
def test_an_incomplete_graph_stream_cannot_publish_marginals(complete, monkeypatch, omit):
    args, _, _ = complete
    original = ComposeLipidTrainingData.iter_weighted_examples

    def incomplete(self):
        yield from (row for index, row in enumerate(original(self)) if index != omit)

    monkeypatch.setattr(ComposeLipidTrainingData, "iter_weighted_examples", incomplete)
    with pytest.raises(ValueError, match="omitted admitted records"):
        noise.compile_noise_marginals(**args, output=args["repo"] / "noise.json")
    assert not (args["repo"] / "noise.json").exists()


def test_failed_serialization_leaves_no_published_artifact(complete, monkeypatch):
    args, _, _ = complete
    output = args["repo"] / "noise.json"

    def fail(*args):
        raise OSError("injected failed write")

    monkeypatch.setattr(noise, "dump", fail)
    with pytest.raises(OSError, match="failed write"):
        noise.compile_noise_marginals(**args, output=output)
    assert not output.exists()
    assert not list(output.parent.glob(".compose-noise-*"))


def test_existing_output_is_preserved(complete):
    args, _, _ = complete
    output = args["repo"] / "noise.json"
    output.write_text("existing artifact")
    with pytest.raises(FileExistsError):
        noise.compile_noise_marginals(**args, output=output)
    assert output.read_text() == "existing artifact"
