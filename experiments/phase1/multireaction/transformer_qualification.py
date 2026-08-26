"""Freeze one verified reaction-program Transformer qualification run as a result receipt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._runtime import verify_run_directory
from forge.core.hashing import pin_record, sha256_file
from forge.core.io import read_json_object, write_json

SCHEMA = "forge.reaction_program_transformer_qualification.v1"


class TransformerQualificationError(ValueError):
    """A run cannot support the bounded Transformer architecture qualification."""


def freeze_transformer_qualification(
    run_dir: Path, repo: Path, output_path: Path
) -> dict[str, Any]:
    repo = repo.resolve()
    run_dir = run_dir.resolve()
    verify_run_directory(run_dir)
    run_path = run_dir / "run.json"
    run = read_json_object(run_path, error=TransformerQualificationError, label="run manifest")
    if (
        run.get("experiment_id")
        not in {
            "phase1-reaction-program-transformer-overfit",
            "phase1-reaction-program-transformer-overfit-v2",
            "phase1-reaction-program-transformer-overfit-v3",
        }
        or run.get("profile") != "smoke"
        or run.get("status") != "complete"
        or set(run.get("stages", {})) != {"cache", "training"}
    ):
        raise TransformerQualificationError("run is not the complete Transformer overfit gate")
    manifest_path = run_dir / "stages/training/manifest.json"
    manifest = read_json_object(
        manifest_path, error=TransformerQualificationError, label="training manifest"
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"checkpoint", "result"}:
        raise TransformerQualificationError("training artifacts changed")
    paths: dict[str, Path] = {}
    for label, record in artifacts.items():
        path = run_dir / "stages/training" / str(record["path"])
        if not path.is_file() or str(sha256_file(path)) != record.get("sha256"):
            raise TransformerQualificationError(f"training {label} is missing or changed")
        paths[label] = path
    training = read_json_object(
        paths["result"], error=TransformerQualificationError, label="training result"
    )
    gates = training.get("gates")
    details = training.get("training")
    if (
        training.get("status") not in {"pass", "fail"}
        or not isinstance(gates, dict)
        or not gates
        or not isinstance(details, dict)
        or details.get("architecture") != "reaction_program_graph_transformer"
        or details.get("gradient_balancing", {}).get("method")
        not in {
            "per_family_norm_balanced_deterministic_pcgrad",
            "equal_family_mass_deterministic_pcgrad",
        }
    ):
        raise TransformerQualificationError("run does not implement the Transformer contract")
    qualified = training["status"] == "pass" and all(gates.values())
    if training["status"] != ("pass" if all(gates.values()) else "fail"):
        raise TransformerQualificationError("training status and qualification gates disagree")
    result = {
        "schema_version": SCHEMA,
        "status": (
            "qualified_for_frozen_followup_preflight"
            if qualified
            else "not_qualified_for_followup_preflight"
        ),
        "run_id": run["run_id"],
        "source_sha256": run["source_sha256"],
        "run_manifest": pin_record(run_path, repo),
        "training_manifest": pin_record(manifest_path, repo),
        "training_result": pin_record(paths["result"], repo),
        "checkpoint": pin_record(paths["checkpoint"], repo),
        "architecture": details["architecture"],
        "semantic_objective": details["semantic_objective"],
        "gradient_balancing": details["gradient_balancing"],
        "initial_total_loss": details["initial_total_loss"],
        "final_total_loss": details["final_total_loss"],
        "final_to_initial_loss_ratio": details["final_to_initial_loss_ratio"],
        "exact_tensor_reconstruction_by_program": {
            program: values["reconstruction"]["exact_tensor_fraction"]
            for program, values in training["validation"].items()
        },
        "gates": gates,
        "calls": {"route": 0, "oracle": 0},
        "candidate_selection": False,
        "decision": (
            "The architecture is qualified for a separately frozen accelerator preflight. "
            "This overfit result is not evidence that multi-reaction generalization is fixed."
            if qualified
            else "The architecture did not pass its bounded overfit gate and cannot advance to "
            "an accelerator preflight from this run."
        ),
    }
    write_json(output_path.resolve(), result)
    return result


__all__ = ["SCHEMA", "TransformerQualificationError", "freeze_transformer_qualification"]
