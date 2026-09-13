"""Restartable, local-only orchestration for the Ugi realism attribution study.

Each scientific audit remains independently runnable. This module only authenticates their
configs and artifacts, journals progress, and makes reuse explicit; it never promotes a model.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_json
from forge.core.io import read_json_object, write_json

CONFIG_SCHEMA = "forge.ugi_realism_diagnostic_suite_config.v1"
RESULT_SCHEMA = "forge.ugi_realism_diagnostic_suite.v1"
RECEIPT_SCHEMA = "forge.ugi_realism_diagnostic_stage_receipt.v1"
STAGES = {
    "support": "ugi_realism_support_audit",
    "model": "ugi_realism_model_attribution",
    "evaluator": "ugi_realism_evaluator_audit",
}
STAGE_STATUSES = {
    "support": "measured",
    "model": "completed_diagnostic_no_promotion",
    "evaluator": "numerical_complete_reviewer_pending",
}
POLICY = {
    "training_calls": 0,
    "generation_calls": 0,
    "remote_compute": False,
    "heldout_structure_access": False,
    "candidate_selection": False,
    "promotion_gate_changes": False,
}


class UgiRealismDiagnosticSuiteError(ValueError):
    """A diagnostic request or persisted result cannot be authenticated."""


def _read(path: Path) -> dict[str, Any]:
    return read_json_object(path, error=UgiRealismDiagnosticSuiteError)


def _pin(path: Path, repo: Path) -> dict[str, str]:
    record = pin_record(path, repo)
    return {key: record[key] for key in ("path", "sha256")}


def _inside(repo: Path, raw: str | Path, *, label: str) -> Path:
    candidate = Path(raw)
    candidate = candidate if candidate.is_absolute() else repo / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(repo) or resolved == repo:
        raise UgiRealismDiagnosticSuiteError(f"{label} must be a path within the repository")
    current = candidate.absolute()
    while current != repo and current.is_relative_to(repo):
        if current.is_symlink():
            raise UgiRealismDiagnosticSuiteError(f"{label} contains a symlink: {current}")
        current = current.parent
    return resolved


def implementation_snapshot(repo: Path) -> dict[str, dict[str, str]]:
    """Conservatively cover Python dependencies, including experiment helper imports."""

    files = [
        path
        for directory in (repo / "forge", repo / "experiments")
        for path in sorted(directory.rglob("*.py"))
        if path.is_file()
    ]
    files.extend(path for path in (repo / "pyproject.toml", repo / "uv.lock") if path.is_file())
    return {
        str(path.relative_to(repo)): _pin(_inside(repo, path, label="source"), repo)
        for path in sorted(files)
    }


def _provenance_pin(record: Any) -> dict[str, str]:
    if not isinstance(record, Mapping) or not {"path", "sha256"} <= set(record):
        raise UgiRealismDiagnosticSuiteError("invalid provenance pin")
    if set(record) - {"path", "sha256", "bytes"}:
        raise UgiRealismDiagnosticSuiteError("unexpected provenance pin fields")
    return {key: record[key] for key in ("path", "sha256")}


def _verify_records(repo: Path, records: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(records, Mapping) or not records:
        raise UgiRealismDiagnosticSuiteError(f"{label} contains no provenance records")
    for key, record in records.items():
        try:
            normalized = _provenance_pin(record)
            _inside(repo, normalized["path"], label=f"{label}/{key}")
            path = resolve_pin(normalized, repo, label=f"{label}/{key}")
            if "bytes" in record and record["bytes"] != path.stat().st_size:
                raise UgiRealismDiagnosticSuiteError(f"{label}/{key}: byte count differs")
        except (OSError, ValueError) as error:
            raise UgiRealismDiagnosticSuiteError(str(error)) from error


def _stage_inputs(repo: Path, config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    inputs = config.get("inputs")
    _verify_records(repo, inputs, label="stage inputs")
    return dict(inputs)


def _artifacts(repo: Path, output: Path) -> dict[str, dict[str, str]]:
    artifacts = {}
    for path in sorted(output.rglob("*")):
        _inside(repo, path, label="artifact")
        if path.is_file():
            artifacts[str(path.relative_to(output))] = _pin(path, repo)
    return artifacts


def _validate_stage_result(
    repo: Path,
    result_path: Path,
    *,
    name: str,
    config_pin: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    result = _read(result_path)
    if (
        result.get("status") != STAGE_STATUSES[name]
        or result.get("schema_version") != f"forge.{STAGES[name]}.v1"
    ):
        raise UgiRealismDiagnosticSuiteError(f"audit is not complete: {result_path}")
    if name == "evaluator" and (
        result.get("numerical_complete") is not True or result.get("reviewer_status") != "pending"
    ):
        raise UgiRealismDiagnosticSuiteError("evaluator numerical/reviewer completion differs")
    if _provenance_pin(result.get("config")) != config_pin:
        raise UgiRealismDiagnosticSuiteError(f"{name}: collected result uses a different config")
    declared_inputs = result.get("inputs")
    _verify_records(repo, declared_inputs, label=f"{name} recorded inputs")
    if {key: _provenance_pin(pin) for key, pin in declared_inputs.items()} != inputs:
        raise UgiRealismDiagnosticSuiteError(f"{name}: recorded inputs differ from config")
    stage_config = _read(resolve_pin(config_pin, repo, label=f"{name} config"))
    if result.get("policy") != stage_config.get("policy"):
        raise UgiRealismDiagnosticSuiteError(f"{name}: recorded policy differs from config")
    sources = result.get("sources")
    _verify_records(repo, sources, label=f"{name} recorded sources")
    for key, pin in stage_config.get("sources", {}).items():
        if _provenance_pin(sources.get(key)) != pin:
            raise UgiRealismDiagnosticSuiteError(f"{name}: recorded source differs: {key}")
    if "implementation_manifest" in result:
        manifest = result["implementation_manifest"]
        _verify_records(repo, manifest, label=f"{name} implementation")
        if str(sha256_json(manifest)) != result.get("implementation_manifest_sha256"):
            raise UgiRealismDiagnosticSuiteError(f"{name}: implementation digest differs")
    return result


def _load_config(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _read(config_path)
    if (
        set(config) != {"schema_version", "scientific_question", "stages", "policy"}
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != POLICY
        or not isinstance(config.get("scientific_question"), str)
        or not config["scientific_question"].strip()
        or not isinstance(config.get("stages"), dict)
        or set(config["stages"]) != set(STAGES)
    ):
        raise UgiRealismDiagnosticSuiteError("suite schema, stages or local-only policy changed")
    config_paths: dict[str, Path] = {}
    for name, stage in config["stages"].items():
        if not isinstance(stage, dict) or set(stage) != {"config", "collected_result"}:
            raise UgiRealismDiagnosticSuiteError(f"stage declaration changed: {name}")
        config_paths[name] = resolve_pin(stage["config"], repo, label=f"{name} config")
        _stage_inputs(repo, _read(config_paths[name]))
    return config, config_paths


def _runner(name: str) -> Callable[..., dict[str, Any]]:
    module_name = STAGES[name]
    module = importlib.import_module(f"experiments.phase1.multireaction.{module_name}")
    return getattr(module, f"run_{module_name}")


def _new_attempt(output: Path, name: str) -> Path:
    """Preserve interrupted partial output and allocate a fresh local stage attempt."""

    parent = output / "stages" / name
    parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        candidate = parent / f"attempt_{attempt:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            attempt += 1
            continue
        return candidate / "output"


def _validate_receipt(
    repo: Path,
    receipt: Mapping[str, Any],
    *,
    name: str,
    config_pin: Mapping[str, Any],
    source_digest: str,
    inputs: Mapping[str, Any],
    collected_result: Mapping[str, Any],
) -> None:
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("stage") != name
        or receipt.get("config") != config_pin
        or receipt.get("source_snapshot_sha256") != source_digest
        or receipt.get("inputs") != inputs
        or receipt.get("execution") not in {"collected_pinned_result", "local_cpu"}
    ):
        raise UgiRealismDiagnosticSuiteError(
            f"{name}: receipt/config/source differs; preserve this run and use a new output directory"
        )
    _verify_records(repo, receipt.get("artifacts"), label=f"{name} artifacts")
    if (
        receipt["execution"] == "collected_pinned_result"
        and receipt["artifacts"].get("result.json") != collected_result
    ):
        raise UgiRealismDiagnosticSuiteError(f"{name}: collected result differs from declaration")
    result_path = resolve_pin(receipt["artifacts"]["result.json"], repo, label=f"{name} result")
    if _artifacts(repo, result_path.parent) != receipt["artifacts"]:
        raise UgiRealismDiagnosticSuiteError(f"{name}: artifact inventory changed")
    _validate_stage_result(repo, result_path, name=name, config_pin=config_pin, inputs=inputs)


def run_ugi_realism_diagnostic_suite(
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    collect: bool = False,
) -> dict[str, Any]:
    """Run missing stages or collect pinned completed stages; reuse only verified receipts."""

    repo = repo_root.resolve()
    config_path = _inside(repo, config_path, label="suite config")
    output = _inside(repo, output_dir, label="suite output")
    config, stage_configs = _load_config(repo, config_path)
    sources = implementation_snapshot(repo)
    if not sources:
        raise UgiRealismDiagnosticSuiteError("suite has no implementation provenance")
    source_digest = str(sha256_json(sources))
    config_pin = _pin(config_path, repo)
    source_path = output / "source_snapshot.json"
    existing = {}
    if source_path.exists() and _read(source_path) != sources:
        raise UgiRealismDiagnosticSuiteError(
            "implementation changed; preserve this run and use a new output directory"
        )
    if (output / "result.json").exists():
        existing = _read(output / "result.json")
        if existing.get("config") != config_pin:
            raise UgiRealismDiagnosticSuiteError("existing suite uses a different configuration")
    write_json(source_path, sources)
    receipts: dict[str, dict[str, str]] = {}
    for name in STAGES:
        declared = config["stages"][name]
        inputs = _stage_inputs(repo, _read(stage_configs[name]))
        receipt_path = output / "receipts" / f"{name}.json"
        if receipt_path.exists():
            receipt = _read(receipt_path)
            _validate_receipt(
                repo,
                receipt,
                name=name,
                config_pin=declared["config"],
                source_digest=source_digest,
                inputs=inputs,
                collected_result=declared["collected_result"],
            )
        else:
            progress = {
                "schema_version": RESULT_SCHEMA,
                "status": "running",
                "config": config_pin,
                "source_snapshot": _pin(source_path, repo),
                "completed_stages": receipts,
                "active_stage": name,
                "policy": POLICY,
            }
            write_json(output / "progress.json", progress)
            try:
                if collect:
                    result_path = resolve_pin(
                        declared["collected_result"], repo, label=f"{name} collected result"
                    )
                    stage_output = result_path.parent
                else:
                    stage_output = _new_attempt(output, name)
                    _runner(name)(repo, stage_configs[name], stage_output)
                    result_path = stage_output / "result.json"
                _validate_stage_result(
                    repo, result_path, name=name, config_pin=declared["config"], inputs=inputs
                )
                _verify_records(repo, inputs, label=f"{name} post-run inputs")
                if implementation_snapshot(repo) != sources:
                    raise UgiRealismDiagnosticSuiteError("implementation changed during audit")
                artifacts = _artifacts(repo, stage_output)
                if "result.json" not in artifacts:
                    raise UgiRealismDiagnosticSuiteError(f"{name}: result.json is missing")
                receipt = {
                    "schema_version": RECEIPT_SCHEMA,
                    "status": "complete",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "stage": name,
                    "config": declared["config"],
                    "inputs": inputs,
                    "source_snapshot_sha256": source_digest,
                    "execution": "collected_pinned_result" if collect else "local_cpu",
                    "artifacts": artifacts,
                }
                write_json(receipt_path, receipt)
            except Exception as error:
                write_json(output / "progress.json", {**progress, "status": "failed"})
                write_json(
                    output / "failures" / f"{name}.json",
                    {
                        **progress,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                raise
        receipts[name] = _pin(receipt_path, repo)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "created_at_utc": existing.get("created_at_utc", datetime.now(timezone.utc).isoformat()),
        "config": config_pin,
        "source_snapshot": _pin(source_path, repo),
        "source_snapshot_sha256": source_digest,
        "stages": receipts,
        "policy": POLICY,
        "model_quality_improvement_established": False,
        "visual_reviewer_calibration": "packet_prepared_judgments_pending",
        "nonclaims": [
            "Completion authenticates diagnostic execution, not any model promotion.",
            "Teacher-forced probability probes do not establish on-policy sampling improvements.",
            "Admitted measured examples do not bound all possible generated distributions.",
            "Supplementary graph metrics and calibration packets do not replace frozen gates.",
        ],
    }
    write_json(output / "result.json", result)
    write_json(
        output / "progress.json",
        {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "stages": receipts,
        },
    )
    return result


def verify_ugi_realism_diagnostic_suite(repo_root: Path, result_path: Path) -> dict[str, Any]:
    """Read-only authentication of source, config, stage inputs, receipts and all outputs."""

    repo = repo_root.resolve()
    result_path = _inside(repo, result_path, label="suite result")
    result = _read(result_path)
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("policy") != POLICY
        or result.get("model_quality_improvement_established") is not False
        or result.get("visual_reviewer_calibration") != "packet_prepared_judgments_pending"
        or not isinstance(result.get("stages"), dict)
        or set(result["stages"]) != set(STAGES)
    ):
        raise UgiRealismDiagnosticSuiteError("suite result is incomplete or its scope changed")
    config_path = resolve_pin(result["config"], repo, label="suite config")
    config, stage_configs = _load_config(repo, config_path)
    source_path = resolve_pin(result["source_snapshot"], repo, label="source snapshot")
    sources = _read(source_path)
    _verify_records(repo, sources, label="source snapshot")
    source_digest = str(sha256_json(sources))
    if source_digest != result.get("source_snapshot_sha256"):
        raise UgiRealismDiagnosticSuiteError("source snapshot digest changed")
    if implementation_snapshot(repo) != sources:
        raise UgiRealismDiagnosticSuiteError("implementation inventory changed since execution")
    for name in STAGES:
        receipt_path = resolve_pin(result["stages"][name], repo, label=f"{name} receipt")
        _validate_receipt(
            repo,
            _read(receipt_path),
            name=name,
            config_pin=config["stages"][name]["config"],
            source_digest=source_digest,
            inputs=_stage_inputs(repo, _read(stage_configs[name])),
            collected_result=config["stages"][name]["collected_result"],
        )
    return {"status": "verified", "stages": list(STAGES), "result": artifact_record(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--collect", action="store_true", help="Collect pinned completed audits")
    action.add_argument("--verify", type=Path, metavar="RESULT_JSON")
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_ugi_realism_diagnostic_suite(args.repo_root, args.verify)
    else:
        if args.config is None or args.output_dir is None:
            parser.error("--config and --output-dir are required for run/collect")
        result = run_ugi_realism_diagnostic_suite(
            args.repo_root, args.config, args.output_dir, collect=args.collect
        )
    print(result["status"])


if __name__ == "__main__":
    main()
