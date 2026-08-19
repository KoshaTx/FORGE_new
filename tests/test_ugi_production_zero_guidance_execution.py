from __future__ import annotations

from pathlib import Path

import pytest

from scripts.phase1_run_ugi_production_zero_guidance_rehearsal import (
    EXECUTION_ORCHESTRATION_SOURCE_PATHS,
    ProductionZeroGuidanceExecutionError,
    _build_orchestration_implementation,
    _require_orchestration_implementation_unchanged,
    run_production_zero_guidance_rehearsal,
)

REPO = Path(__file__).resolve().parents[1]


def test_orchestration_implementation_is_explicit_stable_and_self_including() -> None:
    first = _build_orchestration_implementation(REPO)
    second = _build_orchestration_implementation(REPO)

    assert first == second
    assert [source["path"] for source in first["sources"]] == list(
        EXECUTION_ORCHESTRATION_SOURCE_PATHS
    )
    assert "scripts/phase1_run_ugi_production_zero_guidance_rehearsal.py" in (
        EXECUTION_ORCHESTRATION_SOURCE_PATHS
    )
    assert "src/forge/product/ugi_zero_guidance_rehearsal.py" in (
        EXECUTION_ORCHESTRATION_SOURCE_PATHS
    )
    assert "src/forge/product/ugi_matched_planner_cache_binding.py" in (
        EXECUTION_ORCHESTRATION_SOURCE_PATHS
    )
    _require_orchestration_implementation_unchanged(REPO, first)


def test_external_config_is_rejected_before_output_or_model_work(tmp_path: Path) -> None:
    config = tmp_path / "external-config.json"
    config.write_text("{}\n")
    output = tmp_path / "must-not-exist"

    with pytest.raises(
        ProductionZeroGuidanceExecutionError,
        match="inside the authenticated repository",
    ):
        run_production_zero_guidance_rehearsal(
            repo_root=REPO,
            config_path=config,
            output_dir=output,
        )

    assert not output.exists()
