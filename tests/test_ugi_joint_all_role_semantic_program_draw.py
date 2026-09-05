from __future__ import annotations

from pathlib import Path

from experiments.phase1.multireaction.ugi_joint_all_role_semantic_program_draw import (
    run_ugi_joint_all_role_semantic_program_draw,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/ugi_joint_all_role_semantic_program_draw_seed0_v2.json"


def test_joint_draw_pairs_identical_head_targets_and_directional_aldehydes(
    tmp_path: Path,
) -> None:
    result = run_ugi_joint_all_role_semantic_program_draw(
        CONFIG,
        REPO,
        tmp_path / "program_draw.json",
    )

    assert result["status"] == "pass"
    assert result["rows"] == 3072
    assert result["prior_audit"]["measured_training_rows"] == 480
    assert result["prior_audit"]["eligible_rows"] < 480
    assert result["paired_amine_targets_identical"] is True
    for row in result["samples"]:
        target = row["all_role_semantic_target"]
        assert row["baseline_amine_semantic_target"] == target["amine"]
        assert target["tail_pair"]["aldehyde_alkoxy_handle_side_carbons"] == 6
        assert target["tail_pair"]["aldehyde_acyl_side_carbons"] >= 10
