from __future__ import annotations

from pathlib import Path

from experiments import load_catalog
from experiments._runtime.registry import registry
from experiments._runtime.spec import ExperimentSpec
from experiments.phase1.multireaction.ugi_morphology_diversity_program_draw import (
    run_ugi_morphology_diversity_program_draw,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/ugi_morphology_diversity_program_draw_seed0_v1.json"
EXPERIMENT = (
    REPO / "experiments/phase1/multireaction/ugi_morphology_diversity_seed0_h100_preflight_v1.json"
)


def test_program_draw_flattens_only_the_same_measured_train_support(tmp_path: Path) -> None:
    result = run_ugi_morphology_diversity_program_draw(
        CONFIG,
        REPO,
        tmp_path / "program_draw.json",
    )

    assert result["status"] == "pass"
    assert result["temperature_selection"]["temperature"] == 2.0
    assert result["temperature_selection"]["expected_unique_ratio"] >= 1.15
    assert result["temperature_selection"]["kl_from_source"] <= 0.15
    assert result["calibration_draw_summary"]["unique_modes"] == 147
    assert result["prior_audit"]["joint_support_size"] == 231
    assert result["component_identity_conditioning"] is False
    assert result["training_generation_repair_retry_route_or_oracle_calls"] == 0
    assert all(
        set(row)
        == {
            "all_role_semantic_target",
            "baseline_amine_semantic_target",
            "program",
            "sample_index",
        }
        for row in result["samples"]
    )


def test_preflight_descriptor_is_valid_and_uses_registered_stage() -> None:
    spec = ExperimentSpec.load(EXPERIMENT)
    load_catalog()

    assert spec.experiment_id == "phase1-ugi-morphology-diversity-seed0-h100-preflight-v1"
    assert len(spec.stages) == 1
    assert spec.stages[0].resources.gpu_type == "H100!"
    registry.resolve(spec.stages[0].implementation)
