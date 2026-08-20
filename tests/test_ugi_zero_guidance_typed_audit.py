from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.design.audit.ugi_zero_guidance_typed_audit import (
    UgiZeroGuidanceTypedAuditError,
    build_support_audit_artifact,
    build_zero_guidance_run_artifact,
    derive_typed_counts,
)

REPO = Path(__file__).resolve().parents[1]
V2_RUN = REPO / "results/phase1/ugi_production_zero_guidance_seam_v2/run.json"


def _support_record(ordinal: int, arm: str, phase: str, utility: float | None):
    reason = (
        "strict_exact_route_complete"
        if utility == 1.0
        else "budget_exhausted" if utility is None else "missing_route_knowledge"
    )
    return {
        "ordinal": ordinal,
        "arm": arm,
        "assessment_phase": phase,
        "terminal_id": f"terminal-{ordinal}",
        "potential": {
            "disposition": (
                "support_bonus" if utility == 1.0 else "censor" if utility is None else "neutral"
            ),
            "reasons": [reason],
            "strict_complete_role_count": 3 if utility == 1.0 else 0,
            "route_completion_utility": utility,
            "true_planner_censor": utility is None,
        },
        "scalar_value": None,
        "success_probability": None,
    }


def _records_for_existing_run(run: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for arm_name in ("guided", "post_hoc"):
        arm = run[arm_name]
        assert isinstance(arm, dict)
        for group in arm["checkpoint_groups"]:
            for utility, receipt in zip(
                group["route_completion_utilities"],
                group["source_assessment_receipt_sha256s"],
                strict=True,
            ):
                if receipt is not None:
                    records.append(
                        _support_record(len(records), arm_name, "checkpoint_shadow", utility)
                    )
        for assessment in arm["productive_assessments"]:
            if assessment["disposition"] == "assessed_representative":
                records.append(
                    _support_record(
                        len(records),
                        arm_name,
                        "productive_final",
                        assessment["route_completion_utility"],
                    )
                )
    return records


def test_existing_zero_seam_is_partitioned_without_false_censors() -> None:
    historical = json.loads(V2_RUN.read_text())
    run = historical["run"]
    run_artifact = build_zero_guidance_run_artifact(run)
    records = _records_for_existing_run(run)
    support = build_support_audit_artifact(records)
    typed = derive_typed_counts(run_artifact, support)

    assert support["record_count"] == 476
    assert run_artifact["scope"]["candidate_selection"] is False
    assert "development_endpoint_subset_selected" not in run_artifact["scope"]
    for arm_name in ("guided", "post_hoc"):
        counts = typed["by_arm"][arm_name]
        assert counts["checkpoint_shadow"] == {
            "attempts": 192,
            "completion_error": 0,
            "invalid_terminal": 10,
            "nonexact_l1": 0,
            "budget_exhausted_before_planner": 0,
            "route_complete": 1,
            "route_incomplete": 181,
            "true_planner_censor": 0,
            "duplicate_not_assessed": 0,
        }
        assert counts["productive_final"]["invalid_terminal"] == 2
        assert counts["productive_final"]["canonical_representatives"] == 56
        assert counts["productive_final"]["duplicate_not_assessed"] == 6
        assert counts["productive_final"]["route_complete"] == 1
        assert counts["productive_final"]["route_incomplete"] == 55
        assert counts["productive_final"]["true_planner_censor"] == 0
        assert counts["all_attempts"] == {
            "attempts": 256,
            "completion_error": 0,
            "invalid_terminal": 12,
            "nonexact_l1": 0,
            "budget_exhausted_before_planner": 0,
            "route_complete": 2,
            "route_incomplete": 236,
            "true_planner_censor": 0,
            "duplicate_not_assessed": 6,
        }
    assert typed["planner_censor_reasons"] == {
        "budget_exhausted_after_planner_invocation": 0,
        "invalid_input": 0,
        "execution_error": 0,
    }


def test_missing_support_record_fails_closed() -> None:
    historical = json.loads(V2_RUN.read_text())
    run_artifact = build_zero_guidance_run_artifact(historical["run"])
    records = _records_for_existing_run(historical["run"])
    support = build_support_audit_artifact(records[:-1])
    with pytest.raises(
        UgiZeroGuidanceTypedAuditError,
        match="post_hoc productive support records",
    ):
        derive_typed_counts(run_artifact, support)


def test_support_record_cannot_mislabel_null_utility() -> None:
    record = _support_record(0, "guided", "checkpoint_shadow", None)
    record["potential"]["true_planner_censor"] = False
    with pytest.raises(
        UgiZeroGuidanceTypedAuditError,
        match="censor flag disagrees",
    ):
        build_support_audit_artifact([record])
