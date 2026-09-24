"""Prepared execution preserves the actual trainer's update-boundary restart state."""

import json
from collections import Counter

import pytest

from experiments.phase1.multireaction.compose_lipid_run import read_admitted_config
from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin
from forge.corpus.compose_lipid_tensor_cache import finish_tensor_cache, write_tensor_shard
from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
from forge.model.training_restart import TrainingRestartError
from tests.test_compose_lipid_restoration import OBJECTIVE
from tests.test_compose_lipid_run import assert_exact, execute, run_args, saved  # noqa: F401
from tests.test_compose_lipid_training_data import prepared  # noqa: F401


@pytest.fixture
def prepared_run(run_args):  # noqa: F811
    root = run_args["repo"]
    config = json.loads(run_args["config_path"].read_text())
    config["objective"] = OBJECTIVE
    config["model"].update(
        maximum_children=127, role_morphology_conditioning=True, program_routed_output_heads=True
    )
    config["runtime"].update(
        batch_size=6,
        families_per_batch=3,
        family_schedule="balanced_cycles",
        node_padding="family",
        repeat_supervision="exact_fragment",
        core_conditioning="qualified_core",
    )
    policy = dict(
        maximum_nodes=128,
        maximum_closures=2,
        node_padding="batch",
        repeat_supervision="exact_fragment",
        core_conditioning="qualified_core",
    )
    with ComposeLipidTrainingData(
        root,
        **{k: config["inputs"][k] for k in ("population", "verification", "measure", "admission")},
    ) as data:
        clean = data.batch(list(range(len(data))), **policy)
    output = root / "tensor-cache"
    row = write_tensor_shard(root, output, 0, clean)
    inputs = {k: config["inputs"][k] for k in ("population", "verification", "measure")}
    manifest = finish_tensor_cache(
        root,
        output,
        shards=[row],
        records=len(clean["nodes"]),
        inputs=inputs,
        policy=policy,
        expected_by_family=Counter(map(int, clean["family_states"])),
        implementation=pin(root, root / "forge/model/compose_lipid_training.py"),
    )
    write_json(run_args["config_path"], config)
    return run_args, config, pin(root, manifest)


@pytest.mark.parametrize("loading", ["synchronous", "prefetch"])
def test_prepared_trainer_matches_live_and_resumes_every_state(prepared_run, loading):
    args, config, manifest = prepared_run
    execute(args, "live")
    reference = saved(args, "live")
    config["inputs"]["prepared_tensors"] = manifest
    config["runtime"].update(prepared_loading=loading, prefetch_depth=3, prefetch_workers=2)
    write_json(args["config_path"], config)
    execute(args, "cached")
    actual = saved(args, "cached")
    for key in (
        "model",
        "optimizer",
        "random",
        "completed_steps",
        "last_metrics",
        "family_presentations",
    ):
        assert_exact(actual[key], reference[key])

    def interrupt():
        pointer = json.loads((args["repo"] / "interrupted/latest.json").read_text())
        if pointer["completed_steps"] == 2:
            raise ConnectionError("interrupt prepared update boundary")

    with pytest.raises(ConnectionError, match="update boundary"):
        execute(args, "interrupted", commit=interrupt)
    execute(args, "interrupted", resume=True)
    assert_exact(actual, saved(args, "interrupted"))


@pytest.mark.parametrize(
    "change",
    [
        {"prepared_loading": "prefetch"},
        {"prepared_loading": "wrong"},
        {"node_padding": "model"},
        {"prefetch_depth": 0},
    ],
)
def test_prepared_configuration_fails_before_model_allocation(prepared_run, change):
    args, config, manifest = prepared_run
    config["inputs"]["prepared_tensors"] = manifest
    config["runtime"].update(prepared_loading="prefetch", prefetch_depth=2, prefetch_workers=1)
    config["runtime"].update(change)
    if change == {"prepared_loading": "prefetch"}:
        del config["inputs"]["prepared_tensors"]
    write_json(args["config_path"], config)
    with pytest.raises(TrainingRestartError):
        read_admitted_config(args["repo"], args["config_path"])


@pytest.mark.parametrize(
    "field,value", [("complete", False), ("inputs", {}), ("policy", {}), ("records", 0)]
)
def test_prepared_manifest_is_checked_before_submission(prepared_run, field, value):
    args, config, manifest = prepared_run
    path = args["repo"] / manifest["path"]
    doc = json.loads(path.read_text())
    doc[field] = value
    write_json(path, doc)
    config["inputs"]["prepared_tensors"] = pin(args["repo"], path)
    config["runtime"]["prepared_loading"] = "synchronous"
    write_json(args["config_path"], config)
    with pytest.raises(TrainingRestartError, match="identity or completeness"):
        read_admitted_config(args["repo"], args["config_path"])


@pytest.mark.parametrize(
    "change",
    [
        {"parallel_backend": "unknown"},
        {"batch_size": 9},
        {"families_per_batch": 2},
        {"prepared_loading": "synchronous"},
        {"node_padding": "batch"},
    ],
)
def test_six_gpu_contract_rejects_incompatible_batches(prepared_run, change):
    args, config, manifest = prepared_run
    config["inputs"]["prepared_tensors"] = manifest
    config["runtime"].update(
        parallel_backend="six_gpu_shards_v1",
        prepared_loading="prefetch",
        prefetch_depth=2,
        prefetch_workers=1,
    )
    config["runtime"].update(change)
    write_json(args["config_path"], config)
    with pytest.raises(TrainingRestartError):
        read_admitted_config(args["repo"], args["config_path"])


def test_six_gpu_contract_cannot_silently_use_cpu(prepared_run):
    args, config, manifest = prepared_run
    config["inputs"]["prepared_tensors"] = manifest
    config["runtime"].update(
        parallel_backend="six_gpu_shards_v1",
        prepared_loading="prefetch",
        prefetch_depth=2,
        prefetch_workers=1,
    )
    write_json(args["config_path"], config)
    read_admitted_config(args["repo"], args["config_path"])
    with pytest.raises(TrainingRestartError, match="exactly six CUDA"):
        execute(args, "bad-allocation")


@pytest.mark.parametrize("change", [{"batch_size": 18}, {"relation_embedding_backend": None}])
def test_adaptive_backend_requires_four_way_divisibility_and_ordered_kernel(prepared_run, change):
    args, config, manifest = prepared_run
    config["inputs"]["prepared_tensors"] = manifest
    config["model"]["attention_heads"] = 8
    config["runtime"].update(
        parallel_backend="adaptive_eight_gpu_shards_v1",
        relation_embedding_backend="ordered_fp32_v1",
        prepared_loading="prefetch",
        prefetch_depth=2,
        prefetch_workers=1,
    )
    config["runtime"].update(change)
    write_json(args["config_path"], config)
    with pytest.raises(TrainingRestartError):
        read_admitted_config(args["repo"], args["config_path"])


def test_adaptive_backend_cannot_silently_use_cpu(prepared_run):
    args, config, manifest = prepared_run
    config["inputs"]["prepared_tensors"] = manifest
    config["model"]["attention_heads"] = 8
    config["runtime"].update(
        batch_size=12,
        parallel_backend="adaptive_eight_gpu_shards_v1",
        relation_embedding_backend="ordered_fp32_v1",
        prepared_loading="prefetch",
        prefetch_depth=2,
        prefetch_workers=1,
    )
    write_json(args["config_path"], config)
    read_admitted_config(args["repo"], args["config_path"])
    with pytest.raises(TrainingRestartError, match="exactly eight CUDA"):
        execute(args, "bad-eight-gpu-allocation")
