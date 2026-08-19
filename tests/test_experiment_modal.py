from __future__ import annotations

from pathlib import Path

import pytest

from forge.experiment.errors import BackendError
from forge.experiment.modal import modal_request_plan

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "configs" / "experiments" / "installation-smoke.json"


def test_modal_plan_is_a_hash_pinned_dry_run() -> None:
    plan = modal_request_plan(
        REPO,
        SPEC,
        profile="smoke",
        replicate=0,
        device=None,
    )
    assert plan["backend"] == "modal"
    assert plan["run_id"] is None
    assert plan["resource_envelope"]["gpu_type"] is None
    assert set(plan["uploads"]) == {
        "configs/experiments/installation-smoke.json",
        "configs/experiments/stages/installation_snapshot_v1.json",
        "pyproject.toml",
        "uv.lock",
    }
    assert len(plan["request_id"]) == 64


def test_modal_plan_refuses_mps_without_launching() -> None:
    with pytest.raises(BackendError, match="MPS"):
        modal_request_plan(
            REPO,
            SPEC,
            profile="smoke",
            replicate=0,
            device="mps",
        )
