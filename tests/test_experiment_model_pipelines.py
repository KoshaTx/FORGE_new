from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments import load_catalog
from experiments._runtime import registry
from experiments._runtime.spec import ExperimentSpec, StageSpec

REPO = Path(__file__).resolve().parents[1]
PRODUCT_L1 = REPO / "experiments" / "phase1" / "product_l1"
MULTIREACTION = REPO / "experiments" / "phase1" / "multireaction"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _stage(spec: ExperimentSpec, stage_id: str) -> StageSpec:
    return next(stage for stage in spec.stages if stage.stage_id == stage_id)


def test_training_specs_use_registered_stages_and_one_frozen_contract() -> None:
    load_catalog()
    smoke = ExperimentSpec.load(PRODUCT_L1 / "training_smoke.json")
    production = ExperimentSpec.load(PRODUCT_L1 / "training_production.json")
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
    load_catalog()
    spec = ExperimentSpec.load(PRODUCT_L1 / "sampling.json")
    stage = _stage(spec, "sample")
    registry.resolve(stage.implementation)
    config = _json(PRODUCT_L1 / "configs" / "sampling_v1.json")
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
    """The application adapters must keep heavyweight scientific imports function-local.

    Loading the catalog must not drag in torch, which is an optional extra. The application module
    may import lightweight core serialization at module scope; model, corpus, training, and sampling
    implementations stay inside their stage functions.
    """
    module_level = {
        node.module
        for source in (*PRODUCT_L1.glob("*stages.py"), PRODUCT_L1 / "_stage_support.py")
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    domain = {
        name
        for name in module_level
        if name.startswith(
            (
                "forge.corpus",
                "forge.model",
                "experiments.phase1.product_l1.sampling.",
                "experiments.phase1.product_l1.training.",
            )
        )
        or name
        in {
            "experiments.phase1.product_l1.sampling",
            "experiments.phase1.product_l1.training",
        }
    }

    assert not domain, (
        "product_l1 stages import the science at module level, making torch a hard requirement "
        f"for every runner command: {sorted(domain)}"
    )


def test_multireaction_smoke_has_an_authenticated_checkpoint_boundary() -> None:
    load_catalog()
    spec = ExperimentSpec.load(MULTIREACTION / "training_smoke.json")
    assert [stage.implementation for stage in spec.stages] == [
        "model.multireaction.training.v2",
        "model.multireaction.sampling.v2",
    ]
    assert _stage(spec, "sampling").needs == ("training",)
    for stage in spec.stages:
        registry.resolve(stage.implementation)
    training = _json(REPO / "configs" / "multireaction" / "training_smoke_v1.json")
    assert training["training"]["arms"] == [
        "program",
        "null",
        "program_id_shuffled",
    ]
    assert training["model"]["maximum_heavy_atoms"] == 194
    assert training["expected"]["maxima"]["heavy_atoms"] == 194
    assert _stage(spec, "training").outputs["checkpoint"].path == "checkpoint.json"


def test_multireaction_overfit_gate_is_separate_and_cannot_authorize_production() -> None:
    load_catalog()
    spec = ExperimentSpec.load(MULTIREACTION / "overfit.json")
    assert [stage.stage_id for stage in spec.topological_stages()] == [
        "training",
        "sampling",
        "qualification",
    ]
    assert _stage(spec, "qualification").needs == ("training", "sampling")
    for stage in spec.stages:
        registry.resolve(stage.implementation)
    training = _json(REPO / "configs" / "multireaction" / "training_overfit_v1.json")
    sampling = _json(REPO / "configs" / "multireaction" / "sampling_overfit_v1.json")
    assert training["selection"] == {
        "mode": "deterministic_records_per_program",
        "records_per_program": 1,
    }
    assert training["training"]["arms"] == ["program"]
    assert sampling["sampling"]["layout_source"] == "checkpoint_training_semantics"
    assert "production launch" in " ".join(spec.nonclaims).lower()


def test_shared_representation_is_a_full_census_before_production_training() -> None:
    load_catalog()
    spec = ExperimentSpec.load(MULTIREACTION / "shared_representation.json")
    assert len(spec.stages) == 1
    stage = spec.stages[0]
    assert stage.implementation == "model.shared-synthesis-program-representation.v1"
    registry.resolve(stage.implementation)
    assert stage.resources.device == "cpu"
    assert stage.resources.timeout_seconds == 1800
    config = _json(
        REPO / "configs" / "multireaction" / "shared_synthesis_program_representation_v1.json"
    )
    assert config["support_bounds"] == {
        "maximum_heavy_atoms": 194,
        "maximum_closures": 3,
        "atom_vocabulary": "phase1_product_v3_explicit_aromaticity",
    }
    assert sum(program["expected_records"] for program in config["programs"]) == 113_150
    assert "production" in " ".join(spec.nonclaims).lower()


def test_shared_program_integration_keeps_cache_training_sampling_separate() -> None:
    load_catalog()
    spec = ExperimentSpec.load(MULTIREACTION / "shared_integration.json")
    assert [stage.stage_id for stage in spec.topological_stages()] == [
        "cache",
        "training",
        "sampling",
        "qualification",
    ]
    assert _stage(spec, "training").needs == ("cache",)
    assert _stage(spec, "sampling").needs == ("cache", "training")
    assert _stage(spec, "qualification").needs == ("cache", "training", "sampling")
    for stage in spec.stages:
        registry.resolve(stage.implementation)
    cache = _json(REPO / "configs/multireaction/shared_training_cache_overfit_v1.json")
    assert cache["selection"] == {
        "fold": "train",
        "records_per_program": 1,
        "program_prior": "equal_total_mass",
    }
    assert "production" in " ".join(spec.nonclaims).lower()


def test_shared_production_design_is_a_nonlaunching_preflight() -> None:
    load_catalog()
    spec = ExperimentSpec.load(MULTIREACTION / "shared_production_design.json")
    assert len(spec.stages) == 1
    stage = spec.stages[0]
    assert stage.implementation == "model.shared-synthesis-program-production-design.v1"
    registry.resolve(stage.implementation)
    assert stage.resources.device == "cpu"
    config = _json(REPO / "configs/multireaction/shared_production_comparison_design_v1.json")
    training = config["training"]
    assert isinstance(training, dict)
    assert training["folds"] == ["train"]
    assert training["checkpoint_selection"] == "fixed_final_step"
    execution = config["execution"]
    assert isinstance(execution, dict)
    assert execution["production_launch_authorized"] is False
    assert "does not launch" in " ".join(spec.nonclaims).lower()
