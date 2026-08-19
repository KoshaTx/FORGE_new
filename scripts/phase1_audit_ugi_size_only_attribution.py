#!/usr/bin/env python3
"""Reconcile the completed size-only cloud run with its frozen launch inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]


class AttributionAuditError(RuntimeError):
    """Raised when a downloaded cloud artifact cannot be attributed exactly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*.py") if candidate.is_file())
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(candidate)))
    return digest.hexdigest()


def _run_fingerprint(config: Path, source: Path, launcher: Path) -> str:
    digest = hashlib.sha256()
    for value in (_sha256(config), _python_tree_sha256(source), _sha256(launcher)):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AttributionAuditError(f"expected a JSON object: {path}")
    return value


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AttributionAuditError(f"{label} mismatch: {observed!r} != {expected!r}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def audit(
    *,
    preflight_path: Path,
    receipt_path: Path,
    run_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    preflight = _load_json(preflight_path)
    receipt = _load_json(receipt_path)
    result_path = run_dir / "result.json"
    progress_path = run_dir / "progress.json"
    result = _load_json(result_path)

    immutable = preflight["immutable_inputs"]
    config_path = REPO / immutable["training_config"]["path"]
    cache_path = REPO / immutable["training_cache"]["path"]
    launcher_path = REPO / immutable["launcher"]["path"]
    source_path = REPO / "src/forge"
    hashes = {
        "training_config_sha256": _sha256(config_path),
        "training_cache_sha256": _sha256(cache_path),
        "launcher_sha256": _sha256(launcher_path),
        "python_source_tree_sha256": _python_tree_sha256(source_path),
    }
    fingerprint = _run_fingerprint(config_path, source_path, launcher_path)
    run_name = f"ugi_joint_sparse_size_only_v1_{fingerprint[:12]}"

    _require_equal(
        hashes["training_config_sha256"],
        immutable["training_config"]["sha256"],
        "preflight config hash",
    )
    _require_equal(
        hashes["training_cache_sha256"],
        immutable["training_cache"]["sha256"],
        "preflight cache hash",
    )
    _require_equal(
        hashes["launcher_sha256"], immutable["launcher"]["sha256"], "preflight launcher hash"
    )
    _require_equal(
        hashes["python_source_tree_sha256"],
        immutable["python_source_tree_sha256"],
        "preflight source hash",
    )
    _require_equal(run_name, preflight["run_name"], "preflight run name")

    receipt_execution = receipt["immutable_execution"]
    _require_equal(receipt["run_name"], run_name, "receipt run name")
    _require_equal(receipt_execution["run_fingerprint_sha256"], fingerprint, "receipt fingerprint")
    for key, value in hashes.items():
        _require_equal(receipt_execution[key], value, f"receipt {key}")

    cloud = result["cloud_execution"]
    _require_equal(cloud["run_name"], run_name, "result run name")
    _require_equal(cloud["source"]["run_fingerprint_sha256"], fingerprint, "result fingerprint")
    _require_equal(
        cloud["source"]["python_tree_sha256"],
        hashes["python_source_tree_sha256"],
        "result source hash",
    )
    _require_equal(
        cloud["source"]["launcher_sha256"], hashes["launcher_sha256"], "result launcher hash"
    )
    _require_equal(
        cloud["mounted_inputs"][immutable["training_config"]["path"]],
        hashes["training_config_sha256"],
        "mounted config",
    )
    _require_equal(
        cloud["mounted_inputs"][immutable["training_cache"]["path"]],
        hashes["training_cache_sha256"],
        "mounted cache",
    )
    _require_equal(result["status"], "complete", "training status")
    _require_equal(result["selection"]["stop_reason"], "calibration_early_stopping", "stop reason")

    checkpoint_rows = []
    checkpoint_payloads = []
    for snapshot in result["checkpoint_snapshots"]:
        step = int(snapshot["step"])
        local_path = run_dir / f"checkpoint_step_{step}.pt"
        observed_hash = _sha256(local_path)
        _require_equal(observed_hash, snapshot["sha256"], f"checkpoint {step} hash")
        payload = torch.load(local_path, map_location="cpu", weights_only=False)
        _require_equal(
            payload["schema_version"],
            "phase1_ugi_joint_sparse_checkpoint.v1",
            f"checkpoint {step} schema",
        )
        _require_equal(int(payload["step"]), step, f"checkpoint {step} embedded step")
        _require_equal(
            payload["model_config"]["conditioning_mode"],
            "size_only",
            f"checkpoint {step} conditioning",
        )
        _require_equal(
            payload["inputs"]["prepared_cache"]["sha256"],
            hashes["training_cache_sha256"],
            f"checkpoint {step} cache",
        )
        checkpoint_payloads.append(payload)
        checkpoint_rows.append(
            {"step": step, "path": str(local_path.relative_to(REPO)), "sha256": observed_hash}
        )

    expected_steps = [250, 500, 1000, 2000, 4000]
    _require_equal(
        [row["step"] for row in checkpoint_rows], expected_steps, "serial checkpoint set"
    )
    reference_payload = checkpoint_payloads[0]
    for payload in checkpoint_payloads[1:]:
        _require_equal(payload["inputs"], reference_payload["inputs"], "checkpoint training inputs")
        _require_equal(
            payload["model_config"], reference_payload["model_config"], "checkpoint model config"
        )
        _require_equal(
            set(payload["source_marginals"]),
            set(reference_payload["source_marginals"]),
            "checkpoint source keys",
        )

    special = {}
    for label, result_key, filename in (
        ("best", "checkpoint", "checkpoint_best.pt"),
        ("latest", "checkpoint_latest", "checkpoint_latest.pt"),
    ):
        path = run_dir / filename
        observed_hash = _sha256(path)
        _require_equal(observed_hash, result[result_key]["sha256"], f"{label} checkpoint hash")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        special[label] = {
            "path": str(path.relative_to(REPO)),
            "sha256": observed_hash,
            "embedded_step": int(payload["step"]),
        }
    _require_equal(
        special["best"]["embedded_step"],
        int(result["selection"]["best_step"]),
        "best checkpoint step",
    )
    _require_equal(
        special["latest"]["embedded_step"],
        int(result["selection"]["completed_steps"]),
        "latest checkpoint step",
    )

    output = {
        "schema_version": "phase1_ugi_size_only_attribution_audit.v1",
        "status": "pass",
        "run_name": run_name,
        "run_name_drift": False,
        "run_name_drift_resolution": "The immutable preflight, launch receipt, recomputed source fingerprint and completed cloud result all name the same run.",
        "cloud_app_id": receipt["cloud_execution"]["app_id"],
        "run_fingerprint_sha256": fingerprint,
        "immutable_execution": hashes,
        "inputs": {
            "preflight": {
                "path": str(preflight_path.relative_to(REPO)),
                "sha256": _sha256(preflight_path),
            },
            "launch_receipt": {
                "path": str(receipt_path.relative_to(REPO)),
                "sha256": _sha256(receipt_path),
            },
            "result": {"path": str(result_path.relative_to(REPO)), "sha256": _sha256(result_path)},
            "progress": {
                "path": str(progress_path.relative_to(REPO)),
                "sha256": _sha256(progress_path),
            },
        },
        "training_completion": {
            "completed_steps": int(result["selection"]["completed_steps"]),
            "best_step": int(result["selection"]["best_step"]),
            "best_calibration_loss": float(result["selection"]["best_calibration_loss"]),
            "stop_reason": result["selection"]["stop_reason"],
        },
        "checkpoint_snapshots": checkpoint_rows,
        "special_checkpoints": special,
        "scientific_evaluation_authorized": True,
    }
    _atomic_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        type=Path,
        default=REPO / "results/phase1/ugi_joint_sparse_size_only_v1_launch_preflight.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=REPO / "results/phase1/ugi_joint_sparse_size_only_v1_launch_receipt.json",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=REPO / "results/phase1/ugi_joint_sparse_size_only_v1_full"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_joint_sparse_size_only_v1_attribution_audit.json",
    )
    args = parser.parse_args()
    result = audit(
        preflight_path=args.preflight.resolve(),
        receipt_path=args.receipt.resolve(),
        run_dir=args.run_dir.resolve(),
        output_path=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_name": result["run_name"],
                "fingerprint": result["run_fingerprint_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
