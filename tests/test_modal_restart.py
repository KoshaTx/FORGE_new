"""Explicit recovery preserves prior receipts and rejects ambiguous/live attempts."""

# ruff: noqa: F811

import json
from types import SimpleNamespace

import pytest

from experiments._runtime.errors import BackendError
from experiments._runtime.modal import (
    modal_call_receipt_path,
    modal_restart_receipt_path,
    require_terminal_modal_call,
)
from experiments.phase1.multireaction import compose_lipid_training as launch
from forge.core.io import write_json
from forge.model.training_restart import TrainingRestartError
from tests.test_compose_lipid_run import plan  # noqa: F401


@pytest.mark.parametrize("status", ["PENDING", "SUCCESS", "UNKNOWN", None])
def test_restart_refuses_live_successful_or_unknown_calls(status):
    roots = [] if status is None else [SimpleNamespace(status=SimpleNamespace(name=status))]
    with pytest.raises(BackendError, match="not confirmed"):
        require_terminal_modal_call(SimpleNamespace(get_call_graph=lambda: roots))


@pytest.mark.parametrize("status", ["FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT"])
def test_restart_accepts_confirmed_terminal_failure(status):
    roots = [SimpleNamespace(status=SimpleNamespace(name=status))]
    assert require_terminal_modal_call(SimpleNamespace(get_call_graph=lambda: roots)) == status


def test_restart_preserves_parent_and_uses_same_request_workspace(
    tmp_path, monkeypatch, plan
):  # noqa: F811
    parent = modal_call_receipt_path(tmp_path, plan["request_id"])
    write_json(parent, plan | dict(function_call_id="fc-first", status="launched"))
    before = parent.read_bytes()
    target = modal_restart_receipt_path(tmp_path, plan, parent)
    monkeypatch.setattr(launch, "detached_plan", lambda *a, **k: plan)
    calls = []

    def submit(repo, spec, **options):
        calls.append(options)
        assert options == dict(
            profile="full",
            replicate=0,
            device=None,
            resume=True,
            detached=True,
            restart_from=parent,
            diagnosis="Confirmed interruption; checkpoint verified",
        )
        write_json(target, plan | dict(function_call_id="fc-second", status="launched"))
        return 0

    monkeypatch.setattr(launch, "launch_modal", submit)
    options = dict(
        profile="full",
        replicate=0,
        restart_from=parent,
        diagnosis="Confirmed interruption; checkpoint verified",
    )
    assert launch.submit_detached(tmp_path, tmp_path / "spec.json", **options) == target
    assert parent.read_bytes() == before
    assert json.loads(target.read_text())["request_id"] == plan["request_id"]
    with pytest.raises(TrainingRestartError, match="Existing call"):
        launch.submit_detached(tmp_path, tmp_path / "spec.json", **options)
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["source_sha256", "spec_sha256", "uploads", "request_id"])
def test_restart_rejects_changed_identity(tmp_path, plan, field):  # noqa: F811
    parent = modal_call_receipt_path(tmp_path, plan["request_id"])
    write_json(parent, plan | dict(function_call_id="fc-first", status="launched"))
    with pytest.raises(BackendError, match="Restart changed"):
        modal_restart_receipt_path(tmp_path, plan | {field: "changed"}, parent)
