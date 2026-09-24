"""Training adapter contracts using synthetic metadata and synthetic graph fixtures."""

import copy
import json
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.core.hashing import PinError
from forge.corpus import compose_lipid_training_data as data_module
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_training_measure import compile_training_measure
from forge.corpus.qualified_program_cache import (
    QualifiedProgramExample,
    collate_qualified_program_examples,
)
from forge.model.compose_lipid_training import (
    compose_lipid_forward_loss,
    compose_lipid_training_step,
)
from forge.model.reaction_program_flow import (
    noise_synthesis_program_batch,
    synthesis_program_flow_loss,
)
from forge.model.reaction_program_transformer import reaction_program_transformer_loss
from forge.model.synthesis_program_training import (
    _synthesis_program_predict,
    build_synthesis_program_flow,
    synthesis_program_forward,
)
from tests.test_compose_lipid_training_measure import add_qualified_cohort, inputs
from tests.test_source_instance_coordinates import record, repeated

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    """The SQL producer is real; the graph reader stands in for synthetic shards.

    No real corpus is assigned an admission flag or sampled for optimizer updates.
    The existing qualified-cache tests separately authenticate real source shards.
    """
    options = getattr(request, "param", {})
    args, rows = inputs(tmp_path, **options)
    with sqlite3.connect(tmp_path / "preparation.sqlite") as db:
        db.execute("ALTER TABLE records ADD COLUMN atoms INTEGER NOT NULL DEFAULT 7")
        db.execute("UPDATE records SET atoms=100 WHERE target_id=?", (rows[0][0],))
    p = tmp_path / "population.json"
    population = json.loads(p.read_text())
    population["artifacts"]["preparation.sqlite"] = pin(tmp_path, tmp_path / "preparation.sqlite")
    dump(p, population)
    args["population"] = pin(tmp_path, p)
    p = tmp_path / "verification.json"
    verified = json.loads(p.read_text())
    verified["inputs"]["population"] = args["population"]
    dump(p, verified)
    args["verification"] = pin(tmp_path, p)
    if options:
        args = add_qualified_cohort(tmp_path, args)
    weighted = compile_training_measure(tmp_path, tmp_path / "weighted", **args)
    weights_pin = pin(tmp_path, weighted)
    snapshots = {}
    for name in data_module.TRAINING_TEST_FILES + (
        "forge/corpus/compose_lipid_training_data.py",
        "forge/corpus/compose_lipid_training_measure.py",
        "forge/corpus/qualified_program_cache.py",
        "forge/model/compose_lipid_training.py",
    ):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(ROOT / name, destination)
        snapshots[name] = pin(tmp_path, destination)["sha256"]
    snapshot_path = tmp_path / "source-snapshot.json"
    dump(snapshot_path, snapshots)
    validation_path = tmp_path / "validation.json"
    dump(
        validation_path,
        {
            "schema_version": data_module.TRAINING_VALIDATION_SCHEMA,
            "training_checks_passed": True,
            "test_files": list(data_module.TRAINING_TEST_FILES),
            "all_test_files_executed": True,
            "pytest_exit_code": 0,
            "source_snapshot_unchanged": True,
            "vendor_verify_exit_code": 0,
            "tests": {"failures": 0, "errors": 0, "passed": 1, "skipped": 0},
            "inputs": {"source-snapshot.json": pin(tmp_path, snapshot_path)},
        },
    )
    admission_path = tmp_path / "admission.json"
    dump(
        admission_path,
        {
            "schema_version": "forge.compose_lipid_training_admission.v1",
            "training_admitted": True,
            "training_ready": True,
            "inputs": {
                "population": args["population"],
                "verification": args["verification"],
                "measure": weights_pin,
                **({"cohort": args["cohort"]} if options else {}),
                "validation": pin(tmp_path, validation_path),
            },
        },
    )
    graph, vocabulary, atoms = repeated()
    long_graph = record("C" * 100, ["tail"] * 99 + ["head"], ["exterior"] * 99 + ["core"])[0]
    examples = {}
    for i, row in enumerate(rows):
        value = long_graph if i == 0 else graph
        value = replace(value, graph=replace(value.graph, structure_id=row[0]))
        examples[row[0]] = QualifiedProgramExample(
            value,
            row[2],
            (("head", "synthetic-head", 1), ("tail", "synthetic-tail", 1 if i == 0 else 3)),
            (),
        )

    class SyntheticCache:
        def __init__(self, *args, **kwargs):
            self.vocabulary, self.atom_vocabulary = vocabulary, atoms
            self.closed = False

        def records(self, identities):
            return tuple(examples[identity] for identity in identities)

        def close(self):
            self.closed = True

    monkeypatch.setattr(data_module, "QualifiedProgramCache", SyntheticCache)
    return (
        {
            "repo": tmp_path,
            "population": args["population"],
            "verification": args["verification"],
            "measure": weights_pin,
            "admission": pin(tmp_path, admission_path),
        },
        rows,
        examples,
    )


def rewrite_admission(args, mutate):
    p = args["repo"] / "admission.json"
    d = json.loads(p.read_text())
    mutate(d)
    dump(p, d)
    args["admission"] = pin(args["repo"], p)


def rewrite_weights(args, mutation):
    root = args["repo"]
    with sqlite3.connect(root / "weighted/weights.sqlite") as db:
        mutation(db)
    p = root / "weighted/result.json"
    d = json.loads(p.read_text())
    d["artifact"] = pin(root, root / "weighted/weights.sqlite")
    dump(p, d)
    args["measure"] = pin(root, p)
    rewrite_admission(args, lambda d: d["inputs"].update(measure=args["measure"]))


def test_compiled_family_probability_is_not_rebalanced_by_source_aliases(prepared):
    args, rows, _ = prepared
    with data_module.ComposeLipidTrainingData(**args) as data:
        expected = np.array([1 / 69] * 3 + [1 / 23] * 22)
        actual = np.diff(np.concatenate(([0.0], data._cumulative)))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)

        # Check every probability interval with a deterministic midpoint, not a flaky census.
        class Midpoints:
            def random(self, size):
                assert size == len(rows)
                return data._cumulative - expected / 2

        np.testing.assert_array_equal(data.sample_indices(len(rows), Midpoints()), np.arange(25))


@pytest.mark.parametrize("node_padding", ["model", "batch"])
def test_source_aware_batch_preserves_draw_order_repeats_and_large_graphs(prepared, node_padding):
    args, rows, examples = prepared
    with data_module.ComposeLipidTrainingData(**args) as data:
        selected = [0, 3, 0]
        batch = data.batch(
            selected, maximum_nodes=128, maximum_closures=2, node_padding=node_padding
        )
        reference = collate_qualified_program_examples(
            [examples[rows[i][0]] for i in selected],
            maximum_nodes=100 if node_padding == "batch" else 128,
            maximum_closures=2,
        )
        assert all(torch.equal(batch[k], reference[k]) for k in reference)
        assert batch["node_mask"].sum(dim=1).tolist() == [100, 7, 100]
        assert batch["component_instance_states"].max(dim=1).values.tolist() == [2, 4, 2]
        assert all(isinstance(v, torch.Tensor) for v in batch.values())
        with pytest.raises(data_module.ComposeLipidTrainingDataError, match="large molecules"):
            data.batch([3], maximum_nodes=96, maximum_closures=2, node_padding=node_padding)
        small = data.batch([3], maximum_nodes=128, maximum_closures=2, node_padding=node_padding)
        assert small["nodes"].shape == (1, 7 if node_padding == "batch" else 128)
        with pytest.raises(ValueError, match="node_padding"):
            data.batch([3], maximum_nodes=128, maximum_closures=2, node_padding="truncate")
        with pytest.raises(IndexError):
            data.batch([True], maximum_nodes=128, maximum_closures=2)
        with pytest.raises(IndexError):
            data.batch([25], maximum_nodes=128, maximum_closures=2)
    with pytest.raises(data_module.ComposeLipidTrainingDataError, match="closed"):
        data.sample_indices(1, np.random.default_rng(0))


def test_caller_rng_state_reproduces_draws_exactly(prepared):
    args, _, _ = prepared
    with data_module.ComposeLipidTrainingData(**args) as data:
        rng = np.random.default_rng(34)
        data.sample_indices(11, rng)
        saved = copy.deepcopy(rng.bit_generator.state)
        expected = data.sample_indices(31, rng)
        restored = np.random.default_rng(0)
        restored.bit_generator.state = saved
        np.testing.assert_array_equal(data.sample_indices(31, restored), expected)


def test_modified_exact_evidence_database_is_rejected(prepared):
    args, _, _ = prepared
    with sqlite3.connect(args["repo"] / "evidence.sqlite") as db:
        db.execute("DELETE FROM exact WHERE target_id='target-0-0'")
    with pytest.raises(PinError):
        data_module.ComposeLipidTrainingData(**args)


@pytest.mark.parametrize("field", ["training_admitted", "training_ready"])
def test_preparation_flags_do_not_admit_training(prepared, field):
    args, _, _ = prepared
    rewrite_admission(args, lambda d: d.update({field: False}))
    with pytest.raises(data_module.ComposeLipidTrainingDataError, match="admission is required"):
        data_module.ComposeLipidTrainingData(**args)


def test_an_admission_cannot_be_reused_for_different_weights(prepared):
    args, _, _ = prepared
    args["measure"] = {**args["measure"], "sha256": "0" * 64}
    with pytest.raises(data_module.ComposeLipidTrainingDataError, match="different measure"):
        data_module.ComposeLipidTrainingData(**args)


@pytest.mark.parametrize(
    "reason",
    ["failure", "changed_source", "missing_adapter", "incomplete_suite", "skip", "old_schema"],
)
def test_training_validation_is_required_for_the_current_source(prepared, reason):
    args, _, _ = prepared
    root = args["repo"]
    if reason == "changed_source":
        (root / "forge/model/compose_lipid_training.py").write_text("# changed\n")
    else:
        p = root / "validation.json"
        d = json.loads(p.read_text())
        if reason == "failure":
            d["tests"]["failures"] = 1
        elif reason == "incomplete_suite":
            d["test_files"].pop()
        elif reason == "skip":
            d["tests"]["skipped"] = 1
        elif reason == "old_schema":
            d["schema_version"] = "forge.compose_lipid_all_family_validation.v1"
        else:
            snapshot = root / "source-snapshot.json"
            dump(snapshot, {})
            d["inputs"]["source-snapshot.json"] = pin(root, snapshot)
        dump(p, d)
        rewrite_admission(args, lambda a: a["inputs"].update(validation=pin(root, p)))
    with pytest.raises((data_module.ComposeLipidTrainingDataError, PinError)):
        data_module.ComposeLipidTrainingData(**args)


def test_missing_historical_outputs_and_old_failures_do_not_block_training_checks(prepared):
    args, _, _ = prepared
    root = args["repo"]
    assert not (
        root / "results/phase1/ugi_architecture_selection_v3/full_step1000/result.json"
    ).exists()
    p = root / "validation.json"
    d = json.loads(p.read_text())
    d["historical_full_suite"] = {"failures": 131, "errors": 17}
    dump(p, d)
    rewrite_admission(args, lambda a: a["inputs"].update(validation=pin(root, p)))
    with data_module.ComposeLipidTrainingData(**args) as data:
        assert len(data) == 25


@pytest.mark.parametrize("mutation", ["probability", "order", "missing_row"])
def test_rehashed_probability_or_graph_identity_tampering_is_rejected(prepared, mutation):
    args, _, _ = prepared

    def change(db):
        if mutation == "probability":
            db.execute("UPDATE weights SET probability=0.1 WHERE record_index=0")
        elif mutation == "missing_row":
            db.execute("DELETE FROM weights WHERE record_index=0")
        else:
            db.execute("UPDATE weights SET record_index=100 WHERE record_index=0")
            db.execute("UPDATE weights SET record_index=0 WHERE record_index=3")
            db.execute("UPDATE weights SET record_index=3 WHERE record_index=100")

    rewrite_weights(args, change)
    with pytest.raises(data_module.ComposeLipidTrainingDataError):
        data_module.ComposeLipidTrainingData(**args)


@pytest.mark.parametrize("architecture", ["sparse_mpnn", "reaction_program_graph_transformer"])
def test_both_model_paths_match_existing_loss_and_produce_finite_updates(prepared, architecture):
    args, _, _ = prepared
    torch.set_num_threads(1)
    torch.manual_seed(0)
    weights = {"role_weight": 1.0, "core_weight": 1.0, "repeat_consistency_weight": 1.0}
    if architecture == "sparse_mpnn":
        weights = dict.fromkeys(weights, 0.0)
    with data_module.ComposeLipidTrainingData(**args) as data:
        model = build_synthesis_program_flow(
            vocabulary=data.vocabulary,
            node_classes=len(data.atom_vocabulary),
            device="cpu",
            model_config={
                "architecture": architecture,
                "hidden_dim": 16,
                "layers": 1,
                "attention_heads": 2,
                "expert_count": 2,
                "adapter_dim": 4,
                "maximum_heavy_atoms": 128,
                "maximum_closures": 2,
                "dropout": 0.0,
                "bond_classes": 3,
                "repeat_group_conditioning": True,
            },
        )
        clean = data.batch([3, 0], maximum_nodes=128, maximum_closures=2)
        p0 = torch.full((len(data.atom_vocabulary),), 1 / len(data.atom_vocabulary))
        b0 = torch.full((3,), 1 / 3)
        times = torch.tensor([0.2, 0.7])
        loss, _ = compose_lipid_forward_loss(
            model,
            clean,
            architecture=architecture,
            node_marginal=p0,
            bond_marginal=b0,
            times=times,
            generator=torch.Generator().manual_seed(7),
            semantic_weights=weights,
        )
        actual_gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        predictions, _ = synthesis_program_forward(
            model, clean, p0, b0, times, torch.Generator().manual_seed(7)
        )
        expected, _ = (
            reaction_program_transformer_loss(predictions, clean, **weights)
            if architecture == "reaction_program_graph_transformer"
            else synthesis_program_flow_loss(predictions, clean)
        )
        expected_gradients = torch.autograd.grad(
            expected, tuple(model.parameters()), allow_unused=True
        )
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
            if reference is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                assert torch.isfinite(actual).all()
        # Shape changes consume random numbers differently: compare identical corrupted
        # active states, rather than incorrectly expecting equal seeds to give equal noise.
        short = data.batch([3, 0], maximum_nodes=128, maximum_closures=2, node_padding="batch")
        noisy = noise_synthesis_program_batch(
            clean, p0, b0, times, torch.Generator().manual_seed(7)
        )
        short_noisy = {
            k: v[:, :100] if k in {"nodes", "parents", "parent_bonds"} else v
            for k, v in noisy.items()
        }
        short_predictions = _synthesis_program_predict(model, short, short_noisy, times)
        short_loss, _ = (
            reaction_program_transformer_loss(short_predictions, short, **weights)
            if architecture == "reaction_program_graph_transformer"
            else synthesis_program_flow_loss(short_predictions, short)
        )
        short_gradients = torch.autograd.grad(
            short_loss, tuple(model.parameters()), allow_unused=True
        )
        torch.testing.assert_close(short_loss, expected, atol=1e-5, rtol=1e-5)
        for actual, reference in zip(short_gradients, expected_gradients, strict=True):
            if reference is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, reference, atol=1e-5, rtol=2e-4)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = compose_lipid_training_step(
            data,
            model,
            optimizer,
            batch_size=2,
            maximum_nodes=128,
            maximum_closures=2,
            architecture=architecture,
            device=torch.device("cpu"),
            rng=np.random.default_rng(1),
            generator=torch.Generator().manual_seed(2),
            node_marginal=p0,
            bond_marginal=b0,
            semantic_weights=weights,
            gradient_clip_norm=1.0,
            node_padding="batch",
        )
        assert np.isfinite(list(result.values())).all()
        assert result["gradient_norm"] > 0
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())


@pytest.mark.parametrize("damage", [None, "selection", "population", "family", "order"])
def test_mapped_training_preserves_admission_measure_and_draws(prepared, monkeypatch, damage):
    args, rows, examples = prepared
    repo = args["repo"]
    population = json.loads((repo / "population.json").read_text())
    with sqlite3.connect(repo / "mapped.sqlite") as db:
        db.execute(
            "CREATE TABLE sources(idx INTEGER, target_id TEXT, constitution_id TEXT, family TEXT)"
        )
        db.executemany(
            "INSERT INTO sources VALUES (?,?,?,?)", [(i, *row[:3]) for i, row in enumerate(rows)]
        )
        if damage == "order":
            db.execute("UPDATE sources SET idx=24-idx")

    class SyntheticMappedCache:
        def __init__(self, *a, **kw):
            self.vocabulary, self.atom_vocabulary = repeated()[1:]
            self.metadata = {
                "selection": (
                    "diagnostic" if damage == "selection" else "complete_qualified_population"
                ),
                "inputs": {name: args[name] for name in ("population", "verification")},
                "by_family": population["by_family"],
                "artifacts": {"sources.sqlite": pin(repo, repo / "mapped.sqlite")},
            }
            if damage == "population":
                self.metadata["inputs"]["population"] = {}
            if damage == "family":
                self.metadata["by_family"] = {}

        def __len__(self):
            return len(rows)

        def record(self, index):
            return examples[rows[index][0]]

        def records(self, indices):
            return tuple(self.record(index) for index in indices)

        def close(self):
            pass

    monkeypatch.setattr(data_module, "MappedProgramCache", SyntheticMappedCache)
    if damage:
        with pytest.raises(data_module.ComposeLipidTrainingDataError, match="Mapped cache"):
            data_module.ComposeLipidTrainingData(**args, mapped_cache={"path": "synthetic"})
        return
    with (
        data_module.ComposeLipidTrainingData(**args) as original,
        data_module.ComposeLipidTrainingData(**args, mapped_cache={"path": "synthetic"}) as mapped,
    ):
        indices = [0, 3, 0, 24]
        a = original.batch(indices, maximum_nodes=128, maximum_closures=2)
        b = mapped.batch(indices, maximum_nodes=128, maximum_closures=2)
        assert all(torch.equal(a[k], b[k]) for k in a)
        np.testing.assert_array_equal(
            original.sample_indices(100, np.random.default_rng(1)),
            mapped.sample_indices(100, np.random.default_rng(1)),
        )
        assert [(e.record.graph.structure_id, p) for e, p in original.iter_weighted_examples()] == [
            (e.record.graph.structure_id, p) for e, p in mapped.iter_weighted_examples()
        ]


@pytest.mark.parametrize("prepared", [{"missing": True, "pending": 7}], indirect=True)
def test_authorized_qualified_cohort_loader_preserves_sampling_and_exclusions(prepared):
    args, rows, _ = prepared
    with data_module.ComposeLipidTrainingData(**args) as data:
        assert len(data) == len(rows)
        assert data.maximum_heavy_atoms == 100
        examples = list(data.iter_weighted_examples())
        assert len(examples) == len(rows)
        assert sum(p for _, p in examples) == pytest.approx(1.0)
        assert all(
            p == pytest.approx(1 / (22 * (3 if e.family == "family_00" else 1)))
            for e, p in examples
        )


@pytest.mark.parametrize("prepared", [{"missing": True, "pending": 7}], indirect=True)
def test_subset_admission_requires_the_matching_cohort_pin(prepared):
    args, _, _ = prepared
    rewrite_admission(args, lambda d: d["inputs"].pop("cohort"))
    with pytest.raises(data_module.ComposeLipidTrainingDataError, match="cohort"):
        data_module.ComposeLipidTrainingData(**args)
