from __future__ import annotations

import ast
import json
from pathlib import Path

from forge_experiment import registry
from forge_experiment.spec import ExperimentSpec, StageSpec

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _stage(spec: ExperimentSpec, stage_id: str) -> StageSpec:
    return next(stage for stage in spec.stages if stage.stage_id == stage_id)


def test_training_specs_use_registered_stages_and_one_frozen_contract() -> None:
    smoke = ExperimentSpec.load(EXPERIMENTS / "phase1-training-smoke.json")
    production = ExperimentSpec.load(EXPERIMENTS / "phase1-training-production.json")
    assert [stage.implementation for stage in smoke.stages] == [
        "corpus.ugi.training-cache-verify.v1",
        "generate.ugi.joint-train.v1",
        "generate.ugi.closure-train.v1",
    ]
    for stage in (*smoke.stages, *production.stages):
        registry.resolve(stage.implementation)

    for stage_id in ("cache", "joint", "closure"):
        smoke_stage = _stage(smoke, stage_id)
        production_stage = _stage(production, stage_id)
        assert smoke_stage.config == production_stage.config
        assert dict(smoke_stage.inputs) == dict(production_stage.inputs)
    assert _stage(smoke, "joint").resources.device == "cpu"
    assert _stage(production, "joint").resources.device == "cuda"
    assert _stage(production, "joint").resources.gpu_type == "L4"


def test_production_training_is_all_fold_fixed_step_and_uses_every_cached_record() -> None:
    config = _json(
        REPO / "configs" / "model" / "phase1_ugi_joint_sparse_balanced_v2_production_refit.json"
    )
    counts = config["expected_fold_counts"]
    assert isinstance(counts, dict)
    assert counts == {"train": 66_464, "calibration": 15_800, "heldout": 30_122}
    assert sum(counts.values()) == 112_386
    partition = config["training_partition"]
    assert isinstance(partition, dict)
    assert partition["training_folds"] == ["train", "calibration", "heldout"]
    assert partition["selection_mode"] == "fixed_final_step"
    assert partition["diagnostic_use"] == "in_sample_monitoring_only_not_checkpoint_selection"
    full = config["full"]
    assert isinstance(full, dict)
    assert full["steps"] == 1_700
    assert full["early_stopping"]["patience"] == 0


def test_sampling_contract_is_sharded_exact_l1_and_has_no_guidance_or_retries() -> None:
    spec = ExperimentSpec.load(EXPERIMENTS / "phase1-sampling.json")
    stage = _stage(spec, "sample")
    registry.resolve(stage.implementation)
    config = _json(EXPERIMENTS / "stages" / "phase1_sampling_v1.json")
    assert config["inputs"] == {label: pin.to_mapping() for label, pin in stage.inputs.items()}
    profiles = config["profiles"]
    assert isinstance(profiles, dict)
    assert profiles["smoke"]["program_count"] == 4
    assert profiles["full"]["program_count"] == 3_072
    assert profiles["full"]["shard_size"] == 1_024
    assert spec.replicates == {"smoke": 1, "full": 3}
    assert config["policy"] == {
        "contiguous_program_blocks": True,
        "exact_l1_terminal_admission": True,
        "reference_comparison": "deferred",
        "retries_or_repairs": False,
        "route_calls": 0,
        "oracle_calls": 0,
        "candidate_selection": False,
    }


def test_importing_the_runner_does_not_pull_in_the_training_stack() -> None:
    """The stage bodies must keep their `forge.design` imports function-local.

    `model_stages` lives inside the runner, so nothing structural stops someone hoisting those
    imports to module level for tidiness. Doing so costs ~550ms and drags in torch, which is an
    optional extra -- `forge experiment list` and `forge doctor` would then fail outright for anyone
    who installed without it. This is the property that used to be protected by keeping the stages
    in a separate package; it is now protected by this test.
    """
    source = Path(__file__).resolve().parents[1] / "src" / "forge_experiment" / "model_stages.py"
    tree = ast.parse(source.read_text())

    module_level = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module
    }
    domain = {name for name in module_level if name.startswith(("forge.design", "forge.corpus"))}

    assert not domain, (
        "model_stages imports the science at module level, which makes torch a hard requirement "
        f"for every runner command: {sorted(domain)}"
    )
