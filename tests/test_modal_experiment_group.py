from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import experiments._runtime.modal_group as modal_group_module
from experiments._runtime.errors import BackendError
from experiments._runtime.modal_group import (
    ModalExperimentGroup,
    launch_modal_group,
    modal_group_plan,
)
from experiments._runtime.spec import ExperimentSpec

REPO = Path(__file__).resolve().parents[1]
GROUP_PATH = (
    REPO / "experiments/phase1/multireaction/reaction_specialists_seed0_h100.json"
)


def test_reaction_specialist_group_allocates_independent_exact_h100_specs() -> None:
    group = ModalExperimentGroup.load(GROUP_PATH)

    assert group.group_id == "phase1-reaction-specialists-seed0-h100"
    assert [member.member_id for member in group.parallel] == ["ugi", "bl", "lx"]
    for member in (group.preflight, *group.parallel):
        spec = ExperimentSpec.load(member.resolve(REPO))
        assert all(stage.resources.gpu_type == "H100!" for stage in spec.stages)
        assert all(stage.resources.device == "cuda" for stage in spec.stages)
    ugi = ExperimentSpec.load(group.parallel[0].resolve(REPO))
    assert [stage.stage_id for stage in ugi.stages] == ["specialize_ugi", "compare_ugi_v0"]
    assert ugi.stages[0].needs == ()
    assert ugi.stages[1].needs == ("specialize_ugi",)


def test_modal_group_plan_requires_one_source_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_plan(
        repo: Path,
        spec_path: Path,
        *,
        profile: str,
        replicate: int,
        device: str | None,
    ) -> dict[str, object]:
        del repo, profile, replicate, device
        calls.append(spec_path.name)
        return {
            "experiment_id": ExperimentSpec.load(spec_path).experiment_id,
            "request_id": spec_path.stem,
            "source_sha256": "a" * 64,
            "resource_envelope": {"gpu_type": "H100!"},
        }

    monkeypatch.setattr(modal_group_module, "modal_request_plan", fake_plan)
    plan = modal_group_plan(REPO, GROUP_PATH)

    assert len(calls) == 4
    assert plan["source_sha256"] == "a" * 64
    assert set(plan["parallel"]) == {"ugi", "bl", "lx"}


def test_modal_group_never_launches_production_after_failed_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modal_group_module,
        "modal_group_plan",
        lambda repo, group_path: {
            "group_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "parallel": {},
        },
    )
    launched: list[str] = []

    def fail_preflight(repo: Path, spec_path: Path, **kwargs: object) -> int:
        del repo, kwargs
        launched.append(spec_path.name)
        return 7

    with pytest.raises(BackendError, match="no production member was launched"):
        launch_modal_group(REPO, GROUP_PATH, launch=fail_preflight)
    assert launched == ["reaction_specialist_preflight_seed0_h100.json"]


def test_modal_group_launches_three_production_requests_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_ids = {member_id: member_id * 8 for member_id in ("ugi", "bl", "lx")}
    monkeypatch.setattr(
        modal_group_module,
        "modal_group_plan",
        lambda repo, group_path: {
            "group_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "parallel": {
                member_id: {"request_id": request_id}
                for member_id, request_id in request_ids.items()
            },
        },
    )
    monkeypatch.setattr(
        modal_group_module,
        "modal_request_plan",
        lambda repo, spec_path, **kwargs: {
            "request_id": request_ids[
                next(
                    member_id
                    for member_id in request_ids
                    if f"_{member_id}_" in spec_path.name
                )
            ]
        },
    )
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def successful_launch(repo: Path, spec_path: Path, **kwargs: object) -> int:
        nonlocal active, maximum_active
        del repo, kwargs
        if "preflight" in spec_path.name:
            return 0
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return 0

    result = launch_modal_group(REPO, GROUP_PATH, launch=successful_launch)

    assert maximum_active == 3
    assert result["parallel_exit_codes"] == {"bl": 0, "lx": 0, "ugi": 0}
    assert result["status"] == "complete"


def test_modal_group_refuses_source_drift_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modal_group_module,
        "modal_group_plan",
        lambda repo, group_path: {
            "group_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "parallel": {
                member_id: {"request_id": f"expected-{member_id}"}
                for member_id in ("ugi", "bl", "lx")
            },
        },
    )
    monkeypatch.setattr(
        modal_group_module,
        "modal_request_plan",
        lambda repo, spec_path, **kwargs: {"request_id": "changed"},
    )
    launched: list[str] = []

    def preflight_only(repo: Path, spec_path: Path, **kwargs: object) -> int:
        del repo, kwargs
        launched.append(spec_path.name)
        return 0

    with pytest.raises(BackendError, match="changed after preflight"):
        launch_modal_group(REPO, GROUP_PATH, launch=preflight_only)
    assert launched == ["reaction_specialist_preflight_seed0_h100.json"]
