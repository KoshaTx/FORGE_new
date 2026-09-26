"""Local synthetic qualification of restart and detached submission boundaries."""

import copy
import json
import random
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments._runtime.backends.modal import ModalRuntimeBackend
from experiments._runtime.modal import modal_call_receipt_path
from experiments._runtime.seed import SeedPlan
from experiments._runtime.stage import RunContext
from experiments.phase1.multireaction import compose_lipid_run as runner
from experiments.phase1.multireaction import compose_lipid_training as launch
from forge.core.hashing import PinError, sha256_file
from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin
from forge.model.training_restart import TrainingRestartError
from tests.test_compose_lipid_training_data import (
    ROOT,
    prepared,  # noqa: F401
    rewrite_admission,
)
from tests.test_source_instance_coordinates import repeated


@pytest.fixture
def run_args(prepared):  # noqa: F811
    args, _, _ = prepared
    repo = args["repo"]
    snapshot = json.loads((repo / "source-snapshot.json").read_text())
    for name in runner.RUN_SOURCES:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        snapshot[name] = str(sha256_file(destination))
    write_json(repo / "source-snapshot.json", snapshot)
    validation = json.loads((repo / "validation.json").read_text())
    validation["inputs"]["source-snapshot.json"] = pin(repo, repo / "source-snapshot.json")
    write_json(repo / "validation.json", validation)
    rewrite_admission(
        args, lambda doc: doc["inputs"].update(validation=pin(repo, repo / "validation.json"))
    )
    size = len(repeated()[2])
    write_json(
        repo / "noise.json",
        {
            "schema_version": "forge.compose_lipid_noise_marginals.v1",
            "inputs": {key: args[key] for key in ("population", "measure")},
            "node": [1 / size] * size,
            "bond": [1 / 3] * 3,
        },
    )
    config = {
        "schema_version": runner.CONFIG_SCHEMA,
        "inputs": {key: args[key] for key in runner.DATA_INPUTS}
        | {"noise_marginals": pin(repo, repo / "noise.json")},
        "model": {
            "architecture": "reaction_program_graph_transformer",
            "hidden_dim": 16,
            "layers": 1,
            "attention_heads": 2,
            "expert_count": 2,
            "adapter_dim": 4,
            "maximum_heavy_atoms": 128,
            "maximum_closures": 2,
            "dropout": 0.2,
            "bond_classes": 3,
            "repeat_group_conditioning": True,
        },
        "runtime": {
            "optimizer_steps": 4,
            "checkpoint_interval": 2,
            "batch_size": 2,
            "cpu_threads": 1,
            "maximum_cached_shards": 2,
            "workers": 0,
            "precision": "float32",
            "deterministic": True,
        },
        "optimizer": {"lr": 1e-3, "eps": 1e-8, "weight_decay": 0.01, "betas": [0.9, 0.999]},
        "semantic_weights": {
            "role_weight": 1.0,
            "core_weight": 1.0,
            "repeat_consistency_weight": 1.0,
        },
        "gradient_clip_norm": 1.0,
    }
    write_json(repo / "train.json", config)
    return {"repo": repo, "config_path": repo / "train.json", "device": "cpu", "seed": 19}


def execute(args, name, *, resume=False, commit=lambda: None):
    return runner.run_training(
        **args,
        work_dir=args["repo"] / name,
        output_dir=args["repo"] / (name + "-output"),
        resume=resume,
        commit_progress=commit,
    )


def saved(args, name):
    work = args["repo"] / name
    pointer = json.loads((work / "latest.json").read_text())
    return torch.load(work / pointer["checkpoint"]["path"], weights_only=False)


@pytest.mark.parametrize("node_padding", [None, True, "truncate"])
def test_invalid_padding_policy_is_rejected_before_training(run_args, node_padding):
    config = json.loads(run_args["config_path"].read_text())
    config["runtime"]["node_padding"] = node_padding
    write_json(run_args["config_path"], config)
    with pytest.raises(TrainingRestartError, match="runtime.node_padding"):
        execute(run_args, "invalid-padding")


def assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_exact(a, b)
    else:
        assert left == right


def branch_tape_case(run_args):
    """Use only existing tiny admitted fixture graphs; no research checkpoint."""
    from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
    from forge.model.compose_lipid_prefetch import TAPE_SCHEMA
    from tests.test_compose_lipid_prepared_run import prepared_run

    args, config, cache = prepared_run.__wrapped__(run_args)
    root = args["repo"]
    config["inputs"]["prepared_tensors"] = cache
    config["runtime"].update(prepared_loading="prefetch", prefetch_depth=3, prefetch_workers=2)
    original_path = root / "source-config.json"
    write_json(original_path, config)
    source_args = dict(args, config_path=original_path)
    execute(source_args, "branch-source")
    source = saved(source_args, "branch-source")
    source_pin = pin(root, root / "branch-source-output/checkpoint.pt")
    initialization = {
        "schema_version": runner.INITIALIZATION_SCHEMA,
        "origin": {
            "configuration": pin(root, original_path),
            "checkpoint": source_pin,
            "completed_steps": source["completed_steps"],
            "examples_seen": source["examples_seen"],
        },
        "model_config": config["model"],
        "optimizer_config": config["optimizer"],
        "model": source["model"],
        "optimizer": source["optimizer"],
        "new_branch_counters": {"completed_steps": 0, "examples_seen": 0},
        "random_policy": {
            "seed": args["seed"],
            "reset_all_training_streams": True,
            "original_random_state_restored": False,
        },
    }
    torch.save(initialization, root / "initialization.pt")
    config["inputs"]["initialization"] = pin(root, root / "initialization.pt")
    with ComposeLipidTrainingData(
        root, **{k: config["inputs"][k] for k in runner.DATA_INPUTS}
    ) as data:
        families = [
            r[0]
            for r in data._database.execute("SELECT DISTINCT family FROM weights ORDER BY family")
        ]
        groups = np.tile(np.array([0, 1, 2]), (4, 1))
        rng = np.random.default_rng(38)
        indices = np.stack(
            [data.sample_indices(6, rng, families_per_batch=3, family_selection=g) for g in groups]
        )
    np.savez(root / "presentation.npz", indices=indices, groups=groups)
    tape = {
        "schema_version": TAPE_SCHEMA,
        "inputs": {k: config["inputs"][k] for k in ("population", "verification", "measure")},
        "families": families,
        "optimizer_steps": 4,
        "batch_size": 6,
        "families_per_batch": 3,
        "artifact": pin(root, root / "presentation.npz"),
        "indices_key": "indices",
        "families_key": "groups",
        "excluded_target_ids": [],
        "presentations_by_family": {f: 8 if i < 3 else 0 for i, f in enumerate(families)},
    }
    write_json(root / "presentation.json", tape)
    config["inputs"]["presentation_tape"] = pin(root, root / "presentation.json")
    config["runtime"]["family_schedule"] = "presentation_tape_v1"
    write_json(args["config_path"], config)
    return args, config, source, indices


@pytest.mark.parametrize("group_weight", [None, 0.0, 1.0])
def test_declared_branch_tape_restores_zero_progress_and_exact_restart(run_args, group_weight):
    args, config, source, indices = branch_tape_case(run_args)
    if group_weight is not None:
        config["objective"] = dict(config["objective"], parent_group_loss_weight=group_weight)
        write_json(args["config_path"], config)
    initial = []

    def inspect():
        state = saved(args, "branch-full")
        if state["completed_steps"] == 0:
            initial.append(state)

    result = execute(args, "branch-full", commit=inspect)
    assert len(initial) == 1
    assert initial[0]["examples_seen"] == initial[0]["presentation_cursor"] == 0
    assert set(initial[0]["family_presentations"].values()) == {0}
    assert initial[0]["last_metrics"] == {}
    assert initial[0]["branch"]["origin"]["completed_steps"] == source["completed_steps"]
    for key in ("model", "optimizer"):
        assert_exact(initial[0][key], source[key])
    assert not torch.equal(
        initial[0]["random"]["torch_cpu_rng_state"], source["random"]["torch_cpu_rng_state"]
    )
    assert result["examples_seen"] == indices.size
    assert result["presentation_cursor"] == 4

    def interrupt():
        if saved(args, "branch-resume")["completed_steps"] == 2:
            raise ConnectionError("interrupt with uncommitted prefetched tape entries")

    with pytest.raises(ConnectionError, match="prefetched"):
        execute(args, "branch-resume", commit=interrupt)
    execute(args, "branch-resume", resume=True)
    assert_exact(saved(args, "branch-full"), saved(args, "branch-resume"))
    with pytest.raises(TrainingRestartError, match="fresh runs forbid"):
        execute(args, "branch-full")


def test_fresh_branch_only_permits_declared_parent_group_objective_change(run_args):
    args, config, _, _ = branch_tape_case(run_args)
    config["objective"] = dict(config["objective"], parent_group_loss_weight=1.0)
    runner.read_initialization(args["repo"], config)
    config["objective"]["offspring_weight"] += 1.0
    with pytest.raises(TrainingRestartError, match="objective policy differs"):
        runner.read_initialization(args["repo"], config)


def test_normal_resume_rejects_parent_group_objective_drift(run_args):
    args, config, _, _ = branch_tape_case(run_args)
    execute(args, "group-resume")
    config["objective"] = dict(config["objective"], parent_group_loss_weight=1.0)
    write_json(args["config_path"], config)
    with pytest.raises(TrainingRestartError):
        execute(args, "group-resume", resume=True)


@pytest.mark.parametrize(
    "defect", ["weight", "optimizer", "lr", "seed", "stale_tape", "stale_origin"]
)
def test_branch_and_tape_drift_rejected_before_model_allocation(run_args, monkeypatch, defect):
    args, config, _, _ = branch_tape_case(run_args)
    root = args["repo"]
    if defect in ("weight", "optimizer"):
        state = torch.load(root / "initialization.pt", weights_only=False)
        if defect == "weight":
            next(iter(state["model"].values())).add_(1)
        else:
            next(iter(state["optimizer"]["state"].values()))["exp_avg"].add_(1)
        torch.save(state, root / "initialization.pt")
        config["inputs"]["initialization"] = pin(root, root / "initialization.pt")
    elif defect == "lr":
        config["optimizer"]["lr"] *= 2
    elif defect == "seed":
        args = dict(args, seed=args["seed"] + 1)
    elif defect == "stale_tape":
        with (root / "presentation.npz").open("ab") as handle:
            handle.write(b"drift")
    else:
        with (root / "branch-source-output/checkpoint.pt").open("ab") as handle:
            handle.write(b"drift")
    write_json(args["config_path"], config)

    def forbid(**kwargs):
        raise AssertionError("Invalid branch must fail before model allocation")

    monkeypatch.setattr(runner, "build_synthesis_program_flow", forbid)
    with pytest.raises((TrainingRestartError, PinError)):
        execute(args, "bad-branch")


def test_tape_cursor_cannot_skip_a_prefetched_update(run_args):
    args, _, _, _ = branch_tape_case(run_args)

    def interrupt():
        if saved(args, "cursor")["completed_steps"] == 2:
            raise ConnectionError("stop")

    with pytest.raises(ConnectionError):
        execute(args, "cursor", commit=interrupt)
    work = args["repo"] / "cursor"
    pointer = json.loads((work / "latest.json").read_text())
    checkpoint = work / pointer["checkpoint"]["path"]
    state = torch.load(checkpoint, weights_only=False)
    state["presentation_cursor"] += 1
    torch.save(state, checkpoint)
    pointer["checkpoint"] = pin(work, checkpoint)
    write_json(work / "latest.json", pointer)
    with pytest.raises(TrainingRestartError, match="presentation cursor"):
        execute(args, "cursor", resume=True)


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_invalid_memory_fill_policy_fails_before_training(run_args, value):
    config = json.loads(run_args["config_path"].read_text())
    config["runtime"]["fill_uninitialized_memory"] = value
    write_json(run_args["config_path"], config)
    with pytest.raises(TrainingRestartError, match="fill_uninitialized_memory"):
        execute(run_args, "invalid-fill")


def test_memory_fill_policy_is_scoped_exact_and_restartable(run_args):
    previous = torch.utils.deterministic.fill_uninitialized_memory
    observed = []

    def observe():
        observed.append(torch.utils.deterministic.fill_uninitialized_memory)

    execute(run_args, "fill-default", commit=observe)
    assert observed and all(observed)
    assert torch.utils.deterministic.fill_uninitialized_memory == previous
    observed.clear()
    config = json.loads(run_args["config_path"].read_text())
    config["runtime"]["fill_uninitialized_memory"] = False
    write_json(run_args["config_path"], config)
    result = execute(run_args, "no-fill", commit=observe)
    assert observed and not any(observed)
    assert result["identity"]["runtime"]["fill_uninitialized_memory"] is False
    assert torch.utils.deterministic.fill_uninitialized_memory == previous
    left, right = saved(run_args, "fill-default"), saved(run_args, "no-fill")
    for key in ("model", "optimizer", "random", "completed_steps", "last_metrics"):
        assert_exact(left[key], right[key])

    def interrupt():
        assert torch.utils.deterministic.fill_uninitialized_memory is False
        pointer = json.loads((run_args["repo"] / "no-fill-resume/latest.json").read_text())
        if pointer["completed_steps"] == 2:
            raise ConnectionError("interrupted allocation-policy test")

    with pytest.raises(ConnectionError):
        execute(run_args, "no-fill-resume", commit=interrupt)
    assert torch.utils.deterministic.fill_uninitialized_memory == previous
    execute(run_args, "no-fill-resume", resume=True)
    assert_exact(saved(run_args, "no-fill"), saved(run_args, "no-fill-resume"))
    assert torch.utils.deterministic.fill_uninitialized_memory == previous


@pytest.mark.parametrize(
    "architecture,repeat_mode",
    [
        ("sparse_mpnn", "serialization"),
        ("reaction_program_graph_transformer", "serialization"),
        ("reaction_program_graph_transformer", "exact_fragment"),
        ("reaction_program_graph_transformer", "restored"),
    ],
)
@pytest.mark.parametrize("node_padding", [None, "batch"])
def test_interrupted_run_matches_all_uninterrupted_state(
    run_args, monkeypatch, architecture, node_padding, repeat_mode
):
    config = json.loads(run_args["config_path"].read_text())
    config["model"]["architecture"] = architecture
    config["runtime"]["repeat_supervision"] = (
        repeat_mode if repeat_mode != "restored" else "exact_fragment"
    )
    if repeat_mode == "restored":
        from tests.test_compose_lipid_restoration import OBJECTIVE

        config["objective"] = OBJECTIVE
        config["model"].update(
            maximum_children=127,
            role_morphology_conditioning=True,
            program_routed_output_heads=True,
        )
        config["runtime"]["core_conditioning"] = "qualified_core"
    if node_padding is not None:
        config["runtime"]["node_padding"] = node_padding
    if architecture == "sparse_mpnn":
        config["semantic_weights"] = dict.fromkeys(config["semantic_weights"], 0.0)
    write_json(run_args["config_path"], config)
    draws = []
    sample = runner.ComposeLipidTrainingData.sample_indices

    def sampled(self, batch_size, rng):
        indices = sample(self, batch_size, rng)
        draws.append(indices.tolist())
        return indices

    monkeypatch.setattr(runner.ComposeLipidTrainingData, "sample_indices", sampled)

    def transport_randomness():
        random.random()
        np.random.random()
        torch.rand(3)

    execute(run_args, "continuous", commit=transport_randomness)
    expected_draws = copy.deepcopy(draws)
    draws.clear()
    commits = []

    def disconnect():
        pointer = json.loads((run_args["repo"] / "interrupted/latest.json").read_text())
        commits.append(pointer["completed_steps"])
        if pointer["completed_steps"] == 2:
            raise ConnectionError("simulated disconnect after checkpoint publication")

    with pytest.raises(ConnectionError):
        execute(run_args, "interrupted", commit=disconnect)
    assert commits == [0, 2]
    assert not (run_args["repo"] / "interrupted-output/result.json").exists()
    random.seed(100)
    np.random.seed(101)
    torch.manual_seed(102)
    result = execute(run_args, "interrupted", resume=True)
    assert result["completed_steps"] == 4 and result["examples_seen"] == 8
    assert draws == expected_draws
    assert_exact(saved(run_args, "continuous"), saved(run_args, "interrupted"))
    assert len(list((run_args["repo"] / "interrupted").glob("checkpoint_*.pt"))) == 2


@pytest.mark.parametrize("point", ["serialization", "pointer_publication"])
def test_failed_checkpoint_keeps_previous_generation_restartable(run_args, monkeypatch, point):
    execute(run_args, "reference")
    original_save = torch.save
    original_write = runner.write_json

    def fail_save(value, path, *args, **kwargs):
        if value.get("completed_steps") == 2:
            if hasattr(path, "write"):
                path.write(b"interrupted write")
            else:
                Path(path).write_bytes(b"interrupted write")
            raise OSError("disk failure")
        return original_save(value, path, *args, **kwargs)

    def fail_pointer(path, value, **kwargs):
        if path.name == "latest.json" and value["completed_steps"] == 2:
            raise OSError("pointer failure")
        return original_write(path, value, **kwargs)

    with monkeypatch.context() as patch:
        if point == "serialization":
            patch.setattr(torch, "save", fail_save)
        else:
            patch.setattr(runner, "write_json", fail_pointer)
        with pytest.raises(OSError):
            execute(run_args, "broken")
    assert saved(run_args, "broken")["completed_steps"] == 0
    execute(run_args, "broken", resume=True)
    assert_exact(saved(run_args, "reference"), saved(run_args, "broken"))


@pytest.mark.parametrize("change", ["seed", "source", "config", "runtime"])
def test_resume_rejects_changed_identity_before_updates(run_args, monkeypatch, change):
    execute(run_args, "run")
    if change == "seed":
        run_args["seed"] += 1
    elif change == "source":
        monkeypatch.setattr(runner, "source_fingerprint", lambda repo: "0" * 64)
    elif change == "runtime":
        monkeypatch.setattr(runner, "_runtime_identity", lambda *a: {"torch": "different"})
    else:
        cfg = json.loads(run_args["config_path"].read_text())
        cfg["optimizer"]["lr"] *= 2
        write_json(run_args["config_path"], cfg)
    with pytest.raises(TrainingRestartError, match="changed"):
        execute(run_args, "run", resume=True)


def test_corrupt_checkpoint_is_rejected_before_deserialization(run_args, monkeypatch):
    execute(run_args, "run")
    path = run_args["repo"] / "run/checkpoint_000000004.pt"
    with path.open("ab") as handle:
        handle.write(b"corruption")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("unverified pickle loaded"))
    with pytest.raises(PinError):
        execute(run_args, "run", resume=True)


def test_explicit_resume_and_completed_run_do_not_repeat_updates(run_args, monkeypatch):
    with pytest.raises(TrainingRestartError, match="Explicit resume"):
        execute(run_args, "run", resume=True)
    execute(run_args, "run")
    with pytest.raises(TrainingRestartError, match="fresh runs"):
        execute(run_args, "run")
    monkeypatch.setattr(
        runner, "compose_lipid_training_step", lambda *a, **k: pytest.fail("extra step")
    )
    assert execute(run_args, "run", resume=True)["completed_steps"] == 4


def test_unadmitted_data_cannot_reach_model_or_submission(run_args, monkeypatch):
    cfg = json.loads(run_args["config_path"].read_text())
    doc = json.loads((run_args["repo"] / "admission.json").read_text())
    doc["training_admitted"] = False
    write_json(run_args["repo"] / "admission.json", doc)
    cfg["inputs"]["admission"] = pin(run_args["repo"], run_args["repo"] / "admission.json")
    write_json(run_args["config_path"], cfg)
    monkeypatch.setattr(
        runner, "build_synthesis_program_flow", lambda **k: pytest.fail("model allocated")
    )
    with pytest.raises(ValueError, match="admission is required"):
        execute(run_args, "rejected")
    assert not (run_args["repo"] / "rejected").exists()
    item = SimpleNamespace(
        implementation=launch.IMPLEMENTATION,
        config=SimpleNamespace(resolve=lambda repo: run_args["config_path"]),
    )
    monkeypatch.setattr(launch.ExperimentSpec, "load", lambda path: SimpleNamespace(stages=[item]))
    monkeypatch.setattr(launch, "launch_modal", lambda *a, **k: pytest.fail("compute submitted"))
    with pytest.raises(ValueError, match="admission is required"):
        launch.submit_detached(run_args["repo"], Path("spec.json"), profile="full", replicate=0)
    assert not (run_args["repo"] / "runs").exists()


def test_stage_uses_backend_volume_committer(run_args):
    repo = run_args["repo"]
    cfg = json.loads(run_args["config_path"].read_text())
    spec = json.loads((ROOT / "experiments/installation_smoke/experiment.json").read_text())
    item = spec["stages"][0]
    item.update(
        implementation=launch.IMPLEMENTATION,
        config=pin(repo, run_args["config_path"]),
        inputs=cfg["inputs"],
        outputs={
            "checkpoint": {"path": "checkpoint.pt", "schema_version": runner.CHECKPOINT_SCHEMA},
            "result": {"path": "result.json", "schema_version": runner.RESULT_SCHEMA},
        },
    )
    write_json(repo / "spec.json", spec)
    stage_spec = launch.ExperimentSpec.load(repo / "spec.json").stages[0]
    context = RunContext(
        repo=repo,
        experiment_id="fixture",
        run_id="fixture",
        profile="smoke",
        replicate=0,
        backend="modal",
        stage=stage_spec,
        resources=stage_spec.resources,
        work_dir=repo / "stage-work",
        output_dir=repo / "stage-output",
        config_path=run_args["config_path"],
        inputs={key: repo / value["path"] for key, value in cfg["inputs"].items()},
        dependencies={},
        seed_plan=SeedPlan(root_seed=19, namespace="fixture"),
    )
    committed = []

    def commit():
        pointer = json.loads((context.work_dir / "latest.json").read_text())
        assert (
            sha256_file(context.work_dir / pointer["checkpoint"]["path"])
            == pointer["checkpoint"]["sha256"]
        )
        committed.append(pointer["completed_steps"])

    result = ModalRuntimeBackend(progress_commit=commit).execute(launch.train, context)
    assert committed == [0, 2, 4, 4]
    assert result.metrics["completed_steps"] == 4
    assert {value.label for value in result.artifacts} == {"checkpoint", "result"}


@pytest.mark.parametrize(
    "prepared_tensors,tape_defect",
    [(False, None), (True, None), (True, "range"), (True, "family"), (True, "excluded")],
)
def test_inventory_includes_tensor_and_source_recipe_closure(
    run_args, prepared_tensors, tape_defect
):
    repo = run_args["repo"]

    def save(name, value):
        write_json(repo / name, value)
        return pin(repo, repo / name)

    cfg = json.loads(run_args["config_path"].read_text())
    atoms = save("atoms.json", {"fixture": True})
    stage_request = save("stage-request.json", {"inputs": {"atom_vocabulary": atoms}})
    remaps = save(
        "remaps.json",
        {"atom_vocabulary": atoms, "source_to_union_state_indices": {"fixture_stage": {}}},
    )
    request = json.loads((repo / "request.json").read_text())
    request["inputs"]["fixture_stage"] = {"request": stage_request}
    request_pin = save("request.json", request)
    recipe = save("recipes.jsonl", {"fixture": "source quantities and identities"})
    origin = save(
        "origin.json",
        {
            "artifact": recipe,
            "historical_reference": {"path": "unrelated-missing-history.json", "sha256": "0" * 64},
        },
    )
    (repo / "tensor.npz").write_bytes(b"transport fixture; no tensor is decoded by this test")
    shard = save("shard.json", {"artifact": pin(repo, repo / "tensor.npz"), "origin_shard": origin})
    with sqlite3.connect(repo / "preparation.sqlite") as db:
        db.execute("CREATE TABLE shards(id INTEGER,receipt TEXT)")
        db.execute("INSERT INTO shards VALUES (?,?)", (0, json.dumps(shard)))
    pop = json.loads((repo / "population.json").read_text())
    pop["artifacts"].update(
        {
            "preparation.sqlite": pin(repo, repo / "preparation.sqlite"),
            "vocabulary-remaps.json": remaps,
        }
    )
    pop["request"] = request_pin
    population = save("population.json", pop)
    verified = json.loads((repo / "verification.json").read_text())
    verified["inputs"]["population"] = population
    verification = save("verification.json", verified)
    weighted = json.loads((repo / "weighted/result.json").read_text())
    weighted["inputs"].update(population=population, verification=verification)
    measure = save("weighted/result.json", weighted)
    admission = json.loads((repo / "admission.json").read_text())
    admission["inputs"].update(population=population, verification=verification, measure=measure)
    cfg["inputs"].update(
        population=population,
        verification=verification,
        measure=measure,
        admission=save("admission.json", admission),
    )
    noise = json.loads((repo / "noise.json").read_text())
    noise["inputs"].update(population=population, measure=measure)
    cfg["inputs"]["noise_marginals"] = save("noise.json", noise)
    if prepared_tensors:
        from collections import Counter

        from forge.corpus.compose_lipid_tensor_cache import finish_tensor_cache, write_tensor_shard
        from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData

        policy = dict(
            maximum_nodes=128,
            maximum_closures=2,
            node_padding="batch",
            repeat_supervision="exact_fragment",
            core_conditioning="qualified_core",
        )
        cfg["runtime"].update(
            prepared_loading="synchronous",
            node_padding="batch",
            repeat_supervision="exact_fragment",
            core_conditioning="qualified_core",
        )
        with ComposeLipidTrainingData(
            repo, **{k: cfg["inputs"][k] for k in runner.DATA_INPUTS}
        ) as data:
            clean = data.batch(list(range(len(data))), **policy)
        out = repo / "prepared"
        row = write_tensor_shard(repo, out, 0, clean)
        manifest = finish_tensor_cache(
            repo,
            out,
            shards=[row],
            records=len(clean["nodes"]),
            policy=policy,
            inputs={k: cfg["inputs"][k] for k in ("population", "verification", "measure")},
            expected_by_family=Counter(map(int, clean["family_states"])),
            implementation=pin(repo, repo / "forge/model/compose_lipid_training.py"),
        )
        cfg["inputs"]["prepared_tensors"] = pin(repo, manifest)
        if tape_defect is not None:
            from forge.model.compose_lipid_prefetch import TAPE_SCHEMA

            cfg["runtime"].update(
                prepared_loading="prefetch",
                family_schedule="presentation_tape_v1",
                families_per_batch=1,
                prefetch_depth=2,
                prefetch_workers=1,
            )
            with ComposeLipidTrainingData(
                repo, **{k: cfg["inputs"][k] for k in runner.DATA_INPUTS}
            ) as data:
                families = [
                    row[0]
                    for row in data._database.execute(
                        "SELECT DISTINCT family FROM weights ORDER BY family"
                    )
                ]
                rows = list(
                    data._database.execute("SELECT record_index,target_id,family FROM weights")
                )
                first = next(row for row in rows if row[2] == families[0])
                other = next(row for row in rows if row[2] != families[0])
                indices = np.full((4, 2), first[0], dtype=np.int64)
                if tape_defect == "range":
                    indices[0, 0] = len(data)
                elif tape_defect == "family":
                    indices[0, 0] = other[0]
            np.savez(repo / "tape.npz", indices=indices, groups=np.zeros((4, 1), dtype=np.int64))
            cfg["inputs"]["presentation_tape"] = save(
                "tape.json",
                {
                    "schema_version": TAPE_SCHEMA,
                    "inputs": {
                        k: cfg["inputs"][k] for k in ("population", "verification", "measure")
                    },
                    "families": families,
                    "optimizer_steps": 4,
                    "batch_size": 2,
                    "families_per_batch": 1,
                    "artifact": pin(repo, repo / "tape.npz"),
                    "indices_key": "indices",
                    "families_key": "groups",
                    "excluded_target_ids": [first[1]] if tape_defect == "excluded" else [],
                    "presentations_by_family": {
                        f: 8 if i == 0 else 0 for i, f in enumerate(families)
                    },
                },
            )
    write_json(run_args["config_path"], cfg)
    if tape_defect is not None:
        # Schema and artifact pins pass; staging must also authenticate actual record identities.
        runner.read_admitted_config(repo, run_args["config_path"])
        with pytest.raises(TrainingRestartError, match="Presentation tape"):
            launch.training_files(repo, run_args["config_path"])
        return
    inventory = launch.training_files(repo, run_args["config_path"])
    assert {
        "tensor.npz",
        "recipes.jsonl",
        "shard.json",
        "origin.json",
        "atoms.json",
        "remaps.json",
        "stage-request.json",
        "preparation.sqlite",
    } <= inventory.keys()
    assert "unrelated-missing-history.json" not in inventory
    if prepared_tensors:
        assert {"prepared/manifest.json", "prepared/shard-000000000.npz"} <= inventory.keys()
    (repo / "recipes.jsonl").write_text("tampered")
    with pytest.raises(PinError):
        launch.training_files(repo, run_args["config_path"])


@pytest.fixture
def plan():
    return {
        "request_id": "a" * 64,
        "source_sha256": "b" * 64,
        "spec_sha256": "c" * 64,
        "uploads": {"train.json": {"sha256": "d" * 64, "bytes": 2}},
        "resource_envelope": {"gpu_type": "fixture-only"},
    }


def test_submission_detaches_persists_receipt_and_refuses_duplicates(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(launch, "detached_plan", lambda *a, **k: plan)
    calls = []

    def fake_launch(repo, spec, **options):
        calls.append(options)
        assert options["detached"] is True and options["resume"] is False
        assert (repo / "runs/_compose_submissions" / plan["request_id"] / "request.json").is_file()
        write_json(
            modal_call_receipt_path(repo, plan["request_id"]),
            plan | {"function_call_id": "fc-fixture", "status": "launched"},
        )
        return 0

    monkeypatch.setattr(launch, "launch_modal", fake_launch)
    receipt = launch.submit_detached(tmp_path, tmp_path / "spec.json", profile="full", replicate=0)
    assert json.loads(receipt.read_text())["function_call_id"] == "fc-fixture"
    with pytest.raises(TrainingRestartError, match="Existing call"):
        launch.submit_detached(tmp_path, tmp_path / "spec.json", profile="full", replicate=0)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["disconnect", "exit", "missing_receipt", "changed_receipt"])
def test_uncertain_submission_never_automatically_retries(tmp_path, monkeypatch, plan, failure):
    monkeypatch.setattr(launch, "detached_plan", lambda *a, **k: plan)
    calls = []

    def submit(repo, *args, **kwargs):
        calls.append(1)
        if failure == "disconnect":
            raise ConnectionError("disconnected after possible remote spawn")
        if failure == "changed_receipt":
            write_json(
                modal_call_receipt_path(repo, plan["request_id"]),
                plan | {"source_sha256": "e" * 64},
            )
        return 1 if failure == "exit" else 0

    monkeypatch.setattr(launch, "launch_modal", submit)
    with pytest.raises((TrainingRestartError, ConnectionError)):
        launch.submit_detached(tmp_path, tmp_path / "spec.json", profile="full", replicate=0)
    with pytest.raises(TrainingRestartError):
        launch.submit_detached(tmp_path, tmp_path / "spec.json", profile="full", replicate=0)
    assert len(calls) == 1
    failure_record = tmp_path / "runs/_compose_submissions" / plan["request_id"] / "failure.json"
    assert json.loads(failure_record.read_text())["automatic_retry"] is False


def test_incomplete_upload_inventory_is_rejected(run_args, monkeypatch, plan):
    cfg = json.loads(run_args["config_path"].read_text())
    item = SimpleNamespace(
        implementation=launch.IMPLEMENTATION,
        config=SimpleNamespace(resolve=lambda repo: run_args["config_path"]),
        resources=SimpleNamespace(precision="float32", workers=0, cpus=1),
        determinism=SimpleNamespace(mode="strict"),
        outputs={
            name: SimpleNamespace(to_mapping=lambda value=value: value)
            for name, value in {
                "checkpoint": {"path": "checkpoint.pt", "schema_version": runner.CHECKPOINT_SCHEMA},
                "result": {"path": "result.json", "schema_version": runner.RESULT_SCHEMA},
            }.items()
        },
        inputs={
            name: SimpleNamespace(to_mapping=lambda value=value: value)
            for name, value in cfg["inputs"].items()
        },
    )
    monkeypatch.setattr(launch.ExperimentSpec, "load", lambda path: SimpleNamespace(stages=[item]))
    monkeypatch.setattr(
        launch,
        "training_files",
        lambda *a: {"shard.npz": {"path": "shard.npz", "sha256": "e" * 64}},
    )
    monkeypatch.setattr(launch, "modal_request_plan", lambda *a, **k: plan)
    with pytest.raises(TrainingRestartError, match="shard.npz"):
        launch.detached_plan(run_args["repo"], Path("spec.json"), profile="full", replicate=0)
    plan["uploads"]["shard.npz"] = {"sha256": "e" * 64}
    assert (
        launch.detached_plan(run_args["repo"], Path("spec.json"), profile="full", replicate=0)
        == plan
    )


def test_catalog_registers_the_training_stage():
    from experiments import load_catalog
    from experiments._runtime.registry import registry

    load_catalog()
    assert registry.resolve(launch.IMPLEMENTATION) is launch.train


def test_branch_submission_checks_derived_seed_before_staging(run_args, monkeypatch):
    args, config, _, _ = branch_tape_case(run_args)
    repo = args["repo"]
    spec = json.loads((ROOT / "experiments/installation_smoke/experiment.json").read_text())
    spec["replicates"]["smoke"] = 3
    item = spec["stages"][0]
    item.update(
        implementation=launch.IMPLEMENTATION,
        config=pin(repo, args["config_path"]),
        inputs=config["inputs"],
        outputs={
            "checkpoint": {"path": "checkpoint.pt", "schema_version": runner.CHECKPOINT_SCHEMA},
            "result": {"path": "result.json", "schema_version": runner.RESULT_SCHEMA},
        },
    )
    item["determinism"]["stream"] = "branch-fixture"
    write_json(repo / "spec.json", spec)
    staged = []
    monkeypatch.setattr(launch, "training_files", lambda *a: staged.append(True) or {})
    monkeypatch.setattr(launch, "modal_request_plan", lambda *a, **k: {"uploads": {}})
    with pytest.raises(TrainingRestartError, match="stage seed"):
        launch.detached_plan(repo, repo / "spec.json", profile="smoke", replicate=2)
    assert not staged
    parsed = launch.ExperimentSpec.load(repo / "spec.json")
    root_seed = SeedPlan(parsed.root_seed, f"{parsed.experiment_id}/smoke/replicates").derive(
        "replicate", 2
    )
    expected = (
        SeedPlan(root_seed, f"{parsed.experiment_id}/smoke/branch-fixture").derive(
            "compose-training"
        )
        % 2**32
    )
    initializer = torch.load(repo / "initialization.pt", weights_only=False)
    initializer["random_policy"]["seed"] = expected
    torch.save(initializer, repo / "initialization.pt")
    config["inputs"]["initialization"] = pin(repo, repo / "initialization.pt")
    write_json(args["config_path"], config)
    item.update(config=pin(repo, args["config_path"]), inputs=config["inputs"])
    write_json(repo / "spec.json", spec)
    assert launch.detached_plan(repo, repo / "spec.json", profile="smoke", replicate=2) == {
        "uploads": {}
    }
    assert staged == [True]


def test_package_rejects_unadmitted_data_before_any_fitting(run_args):
    from experiments.phase1.multireaction.compose_lipid_package import build_package
    from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingDataError

    root = run_args["repo"]
    path = root / "admission.json"
    doc = json.loads(path.read_text())
    doc["training_admitted"] = False
    write_json(path, doc)
    with pytest.raises(ComposeLipidTrainingDataError, match="Final training"):
        build_package(
            root,
            root / "package",
            recipe=root / "absent",
            admission=path,
            preflight=root / "absent",
        )
    assert not (root / "package").exists()


def test_package_binds_validated_recipe_and_requires_real_cuda_receipt(run_args, monkeypatch):
    from experiments._runtime.spec import ExperimentSpec
    from experiments.phase1.multireaction import compose_lipid_package as package

    root = run_args["repo"]
    config = json.loads(run_args["config_path"].read_text())
    mapping = pin(root, root / "source.bin")
    recipe = {
        k: config[k]
        for k in ("model", "runtime", "optimizer", "semantic_weights", "gradient_clip_norm")
    }
    recipe.update(
        schema_version="forge.compose_lipid_training_recipe.v1",
        seed=1,
        execution={
            "gpu_type": "L4",
            "gpu_count": 1,
            "detached": True,
            "automatic_retries": 0,
            "timeout_seconds": 86400,
        },
        inputs_prepared={k: config["inputs"][k] for k in ("population", "verification")}
        | {"mapped_cache": mapping},
    )
    write_json(root / "recipe.json", recipe)
    write_json(
        root / "gpu-payload.json",
        {"inputs": {"mapped_cache": mapping}, "batch_size": recipe["runtime"]["batch_size"]},
    )
    gpu = {
        "schema_version": "forge.compose_lipid_gpu_preflight.v1",
        "passed": True,
        "device": "cpu",
        "gpu": "NVIDIA L4",
        "model": config["model"],
        "optimizer_steps": 0,
        "model_state_unchanged": True,
        "inputs": {"payload": pin(root, root / "gpu-payload.json"), "gpu": "L4"},
    }
    write_json(root / "gpu.json", gpu)
    args = {
        "repo": root,
        "output": root / "package",
        "recipe": root / "recipe.json",
        "admission": root / "admission.json",
        "preflight": root / "gpu.json",
    }
    with pytest.raises(ValueError, match="CUDA diagnostic"):
        package.build_package(**args)
    assert not args["output"].exists()
    gpu["device"] = "cuda"
    write_json(root / "gpu.json", gpu)

    def compile_noise(repo, output, **inputs):
        assert inputs["mapped_cache"] == mapping
        shutil.copyfile(root / "noise.json", output)
        return output

    monkeypatch.setattr(package, "compile_noise_marginals", compile_noise)
    monkeypatch.setattr(package, "training_files", lambda repo, path: {"source": mapping})

    def plan(repo, path, **kwargs):
        spec = ExperimentSpec.load(path)
        assert spec.stages[0].inputs["mapped_cache"].to_mapping() == mapping
        assert spec.stages[0].resources.device == "cuda"
        return {"synthetic_plan": True}

    monkeypatch.setattr(package, "detached_plan", plan)
    spec = package.build_package(**args)
    result = json.loads((args["output"] / "result.json").read_text())
    assert spec.exists() and result["ready"] and not result["training_launched"]
