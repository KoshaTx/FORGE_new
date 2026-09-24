"""Read-only verifier for the bounded twelve-family overfit gate."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from forge.core.hashing import resolve_pin, sha256_file

RESULT_SCHEMA = "forge.combinatorial_overfit_gate.v1"
SCIENTIFIC_FIELDS = (
    "scientific_question",
    "hypothesis",
    "alternative_explanation",
    "inputs",
    "selected_records",
    "model",
    "training",
    "per_program",
    "programs_with_improved_loss",
    "exact_tensor_records",
    "gates",
    "optimizer_steps_executed",
    "generator_sampling_calls",
    "checkpoint_retained",
    "remote_compute",
    "decision",
    "nonclaims",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _junit(path: Path) -> dict[str, Any]:
    cases = ET.parse(path).getroot().findall(".//testcase")
    failures = {
        f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    }
    failure_count = sum(case.find("failure") is not None for case in cases)
    error_count = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    return {
        "tests": len(cases),
        "passed": len(cases) - failure_count - error_count - skipped,
        "failures": failure_count,
        "errors": error_count,
        "skipped": skipped,
        "failure_error_ids": failures,
        "duration_seconds": sum(float(case.attrib.get("time", 0.0)) for case in cases),
    }


def verify(repo_root: Path) -> dict[str, Any]:
    repo = repo_root.resolve()
    result_path = repo / "results/phase1/combinatorial_overfit_gate_v1/result.json"
    replay_path = repo / "results/phase1/combinatorial_overfit_gate_v1/replay_result.json"
    focused_path = (
        repo / "results/phase1/combinatorial_overfit_gate_validation_v1/focused-tests.xml"
    )
    full_path = repo / "results/phase1/combinatorial_overfit_gate_validation_v1/full-tests.xml"
    baseline_path = repo / "results/phase1/combinatorial_shared_model_validation_v1/full-tests.xml"
    result = _read(result_path)
    replay = _read(replay_path)
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or replay.get("schema_version") != RESULT_SCHEMA
    ):
        raise ValueError("overfit result schema changed")
    config_pin = result.get("config")
    if not isinstance(config_pin, dict):
        raise ValueError("overfit result has no config pin")
    config_path = resolve_pin(config_pin, repo, label="overfit config")
    config = _read(config_path)
    focused = _junit(focused_path)
    full = _junit(full_path)
    baseline = _junit(baseline_path)
    checks = {
        "result_status_pass": result.get("status") == "pass",
        "all_result_gates_pass": bool(result.get("gates")) and all(result["gates"].values()),
        "config_inputs_exact": result.get("inputs") == config.get("inputs"),
        "all_input_pins_resolve": True,
        "all_source_hashes_resolve": True,
        "twelve_distinct_programs": len(result.get("selected_records", ())) == 12
        and len({row["program_id"] for row in result["selected_records"]}) == 12,
        "all_program_losses_improve": len(result.get("programs_with_improved_loss", ())) == 12,
        "all_records_reconstruct_exactly": result.get("exact_tensor_records") == 12,
        "fixed_states_preserved": result.get("gates", {}).get("fixed_states_never_noised") is True,
        "no_generation_remote_compute_or_checkpoint": result.get("generator_sampling_calls") == 0
        and result.get("remote_compute") is False
        and result.get("checkpoint_retained") is False,
        "scientific_replay_exact": all(
            result.get(field) == replay.get(field) for field in SCIENTIFIC_FIELDS
        ),
        "focused_tests_pass": focused["failures"] == focused["errors"] == 0,
        "full_suite_failure_error_ids_equal_prior_baseline": full["failure_error_ids"]
        == baseline["failure_error_ids"],
    }
    for label, pin in sorted(config["inputs"].items()):
        resolve_pin(pin, repo, label=label)
    for path, digest in sorted(result["sources"].items()):
        checks["all_source_hashes_resolve"] = (
            checks["all_source_hashes_resolve"] and str(sha256_file(repo / path)) == digest
        )
    ratio = float(result["training"]["final_to_initial_loss_ratio"])
    checks["loss_ratio_meets_frozen_threshold"] = ratio <= float(
        config["acceptance"]["maximum_final_to_initial_loss_ratio"]
    )
    checks["all_per_program_payloads_present"] = set(result["per_program"]) == {
        row["program_id"] for row in result["selected_records"]
    }
    if not all(checks.values()):
        raise ValueError(f"overfit artifact verification failed: {checks}")
    return {
        "schema_version": "forge.combinatorial_overfit_gate_artifact_verification.v1",
        "status": "pass",
        "inputs": {
            "config": {
                "path": str(config_path.relative_to(repo)),
                "sha256": str(sha256_file(config_path)),
            },
            "result": {
                "path": str(result_path.relative_to(repo)),
                "sha256": str(sha256_file(result_path)),
            },
            "replay_result": {
                "path": str(replay_path.relative_to(repo)),
                "sha256": str(sha256_file(replay_path)),
            },
            "focused_tests": {
                "path": str(focused_path.relative_to(repo)),
                "sha256": str(sha256_file(focused_path)),
            },
            "full_tests": {
                "path": str(full_path.relative_to(repo)),
                "sha256": str(sha256_file(full_path)),
            },
            "prior_full_tests": {
                "path": str(baseline_path.relative_to(repo)),
                "sha256": str(sha256_file(baseline_path)),
            },
        },
        "checks": checks,
        "observations": {
            "initial_total_loss": result["training"]["initial_total_loss"],
            "final_total_loss": result["training"]["final_total_loss"],
            "final_to_initial_loss_ratio": ratio,
            "programs_improved": len(result["programs_with_improved_loss"]),
            "exact_tensor_records": result["exact_tensor_records"],
        },
        "excluded_replay_metadata": ["created_at_utc", "duration_seconds", "environment"],
        "test_reports": {
            "focused": {key: value for key, value in focused.items() if key != "failure_error_ids"},
            "full": {key: value for key, value in full.items() if key != "failure_error_ids"},
            "prior_full_failure_error_count": len(baseline["failure_error_ids"]),
            "current_full_failure_error_count": len(full["failure_error_ids"]),
            "new_full_failure_error_ids": sorted(
                full["failure_error_ids"] - baseline["failure_error_ids"]
            ),
            "resolved_prior_failure_error_ids": sorted(
                baseline["failure_error_ids"] - full["failure_error_ids"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.repo_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = (args.repo_root.resolve() / args.output).resolve()
        if not output.is_relative_to(args.repo_root.resolve()):
            raise ValueError("verification output must remain inside the repository")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
