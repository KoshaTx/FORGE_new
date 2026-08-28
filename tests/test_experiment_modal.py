from __future__ import annotations

from pathlib import Path

import pytest

from experiments._runtime.errors import BackendError
from experiments._runtime.modal import (
    _config_dependency_uploads,
    modal_request_plan,
    modal_run_staging_paths,
    modal_volume_relative_path,
)
from forge.core.hashing import sha256_file

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


def test_modal_config_dependency_closure_uploads_nested_pins(tmp_path: Path) -> None:
    data = tmp_path / "data.bin"
    data.write_bytes(b"assessment reference")
    child = tmp_path / "child.json"
    child.write_text(
        '{"inputs":{"data":{"path":"data.bin","sha256":"'
        + sha256_file(data)
        + '"}}}'
    )
    root = tmp_path / "root.json"
    root.write_text(
        '{"inputs":{"child":{"path":"child.json","sha256":"'
        + sha256_file(child)
        + '"}}}'
    )

    uploads = _config_dependency_uploads(tmp_path, root)

    assert uploads == {"child.json": child, "data.bin": data}


def test_modal_config_dependency_closure_rejects_nested_drift(tmp_path: Path) -> None:
    data = tmp_path / "data.bin"
    data.write_bytes(b"changed")
    root = tmp_path / "root.json"
    root.write_text('{"inputs":{"data":{"path":"data.bin","sha256":"' + "0" * 64 + '"}}}')

    with pytest.raises(BackendError, match="pinned input.*changed"):
        _config_dependency_uploads(tmp_path, root)


def test_modal_image_syncs_from_the_local_uv_project() -> None:
    modal_app_source = (REPO / "experiments" / "_runtime" / "modal_app.py").read_text()

    assert (
        "LOCAL_REPO = Path(__file__).resolve().parents[2] if modal.is_local() else IMAGE_PROJECT"
        in modal_app_source
    )
    assert ".uv_sync(\n        str(LOCAL_REPO)," in modal_app_source
    assert ".uv_sync(\n        str(IMAGE_PROJECT)," not in modal_app_source
    assert '.env({"PYTHONPATH": str(IMAGE_PROJECT)})' in modal_app_source
    assert "experiment_volume.reload()" not in modal_app_source


def test_modal_volume_path_accepts_the_resolved_mount_target(tmp_path: Path) -> None:
    physical_volume = tmp_path / "physical-volume"
    physical_run = physical_volume / "jobs" / "request" / "run"
    physical_run.mkdir(parents=True)
    mounted_volume = tmp_path / "mounted-volume"
    mounted_volume.symlink_to(physical_volume, target_is_directory=True)

    assert modal_volume_relative_path(physical_run.resolve(), mounted_volume) == "jobs/request/run"


def test_modal_volume_path_rejects_a_run_outside_the_volume(tmp_path: Path) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(BackendError, match="outside Modal volume"):
        modal_volume_relative_path(outside, volume)


def test_modal_run_staging_preserves_the_content_addressed_run_name(tmp_path: Path) -> None:
    run_id = "d97c3dd86b4c2383382a4ed91055135bda0235aaab2db6006d7c1a741c69a2a7"
    final_run = tmp_path / run_id

    staging_root, staged_run = modal_run_staging_paths(final_run)
    try:
        assert staged_run.parent == staging_root
        assert staged_run.name == run_id
        assert staged_run.is_dir()
        assert not final_run.exists()
    finally:
        staged_run.rmdir()
        staging_root.rmdir()
