from __future__ import annotations

from pathlib import Path

import pytest

from experiments._runtime.errors import BackendError
from experiments._runtime.modal import modal_request_plan

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "experiments" / "installation_smoke" / "experiment.json"


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
        "experiments/installation_smoke/experiment.json",
        "experiments/installation_smoke/configs/snapshot_v1.json",
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
