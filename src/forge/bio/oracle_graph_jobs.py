"""Deterministic, resumable job manifests for the M0-07 graph matrix."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "m0_07_oracle_graph_jobs_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_graph_jobs.v1"
JOB_FIELDS = (
    "job_id",
    "ensemble_id",
    "architecture",
    "endpoint",
    "scheme",
    "fold",
    "seed",
    "selection_eligible",
    "train_rows",
    "calibration_rows",
    "test_rows",
    "train_labels_sha256",
    "calibration_labels_sha256",
    "test_labels_sha256",
    "output_relative_path",
    "status",
)


class OracleGraphJobsError(ValueError):
    """Raised when the graph job matrix violates its frozen contract."""


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleGraphJobsError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleGraphJobsError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise OracleGraphJobsError(f"{label} must contain an object")
    return value


def _verify_input(
    repo_root: Path,
    specification: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise OracleGraphJobsError(f"{label} input specification is incomplete")
    path = repo_root / relative
    if not path.is_file():
        raise OracleGraphJobsError(f"{label} not found: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise OracleGraphJobsError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}


def _read_assignments(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"scheme", "fold", "label", "stage"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise OracleGraphJobsError(f"split assignments lack fields: {sorted(missing)}")
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise OracleGraphJobsError(f"split assignments not found: {path}") from exc


def _labels_digest(labels: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(labels)).encode()).hexdigest()


def _identifier(parts: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _gzip_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=JOB_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda value: str(value["job_id"])):
        writer.writerow({field: row[field] for field in JOB_FIELDS})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as archive:
        archive.write(text.getvalue().encode())
    return output.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
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


def build_graph_job_rows(
    assignments: Sequence[Mapping[str, str]],
    *,
    architectures: Sequence[str],
    endpoints: Sequence[str],
    seeds: Sequence[int],
    selection_schemes: set[str],
    fit_output_root: str,
) -> list[dict[str, Any]]:
    """Expand every architecture, endpoint, partition, and seed exactly once."""

    partitions: dict[tuple[str, int], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in assignments:
        scheme = str(row["scheme"])
        fold = int(row["fold"])
        stage = str(row["stage"])
        if stage not in {"train", "calibration", "test"}:
            raise OracleGraphJobsError(f"unsupported split stage: {stage}")
        partitions[(scheme, fold)][stage].append(str(row["label"]))
    if not partitions:
        raise OracleGraphJobsError("split assignments contain no partitions")
    jobs = []
    for architecture in sorted(architectures):
        for endpoint in sorted(endpoints):
            for (scheme, fold), stages in sorted(partitions.items()):
                if set(stages) != {"train", "calibration", "test"}:
                    raise OracleGraphJobsError(f"{scheme} fold {fold} does not contain all stages")
                flattened = [
                    label for stage in ("train", "calibration", "test") for label in stages[stage]
                ]
                if len(flattened) != len(set(flattened)):
                    raise OracleGraphJobsError(f"{scheme} fold {fold} overlaps split labels")
                ensemble_id = _identifier((architecture, endpoint, scheme, str(fold)))
                for seed in sorted(seeds):
                    job_id = _identifier(
                        (
                            architecture,
                            endpoint,
                            scheme,
                            str(fold),
                            str(seed),
                        )
                    )
                    relative = (
                        f"{fit_output_root}/{architecture}/{endpoint}/"
                        f"{scheme}/fold-{fold}/seed-{seed}.json"
                    )
                    jobs.append(
                        {
                            "job_id": job_id,
                            "ensemble_id": ensemble_id,
                            "architecture": architecture,
                            "endpoint": endpoint,
                            "scheme": scheme,
                            "fold": fold,
                            "seed": seed,
                            "selection_eligible": str(scheme in selection_schemes).lower(),
                            "train_rows": len(stages["train"]),
                            "calibration_rows": len(stages["calibration"]),
                            "test_rows": len(stages["test"]),
                            "train_labels_sha256": _labels_digest(stages["train"]),
                            "calibration_labels_sha256": _labels_digest(stages["calibration"]),
                            "test_labels_sha256": _labels_digest(stages["test"]),
                            "output_relative_path": relative,
                            "status": "pending",
                        }
                    )
    if len({row["job_id"] for row in jobs}) != len(jobs):
        raise OracleGraphJobsError("graph job IDs are not unique")
    return jobs


def run_oracle_graph_job_manifest(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Freeze the resumable random-initialized supervised graph job matrix."""

    config = _load_json(config_path, "graph jobs config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleGraphJobsError("unsupported graph-jobs config schema")
    execution = config.get("execution")
    if (
        not isinstance(execution, dict)
        or execution.get("maximum_parallel_jobs") != 4
        or execution.get("torch_threads_per_job") != 2
        or execution.get("resume_policy") != "validate_completed_output_then_skip"
    ):
        raise OracleGraphJobsError("graph execution policy changed")
    matrix = config.get("matrix")
    if not isinstance(matrix, dict):
        raise OracleGraphJobsError("graph jobs config lacks the matrix")
    architectures = matrix.get("architectures")
    endpoints = matrix.get("endpoints")
    seeds = matrix.get("seeds")
    if (
        not isinstance(architectures, list)
        or not isinstance(endpoints, list)
        or not isinstance(seeds, list)
        or not all(isinstance(seed, int) for seed in seeds)
    ):
        raise OracleGraphJobsError("graph job matrix is malformed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise OracleGraphJobsError("graph jobs config lacks inputs")
    verified = {
        name: _verify_input(repo_root, specification, name)
        for name, specification in sorted(inputs.items())
        if isinstance(specification, dict)
    }
    required_inputs = {
        "curated_oracle_data",
        "graph_cache_result",
        "graph_profile_result",
        "oracle_split_assignments",
        "oracle_split_manifest",
    }
    if set(verified) != required_inputs:
        raise OracleGraphJobsError("graph job inputs are incomplete")
    split_manifest = _load_json(
        repo_root / verified["oracle_split_manifest"]["path"],
        "oracle split manifest",
    )
    selection_schemes = split_manifest.get("evaluation_contract", {}).get(
        "selection_eligible_schemes"
    )
    if not isinstance(selection_schemes, list) or "lantern_random" in selection_schemes:
        raise OracleGraphJobsError("selection-eligible split schemes are invalid")
    assignments = _read_assignments(repo_root / verified["oracle_split_assignments"]["path"])
    rows = build_graph_job_rows(
        assignments,
        architectures=[str(value) for value in architectures],
        endpoints=[str(value) for value in endpoints],
        seeds=[int(value) for value in seeds],
        selection_schemes={str(value) for value in selection_schemes},
        fit_output_root=str(execution["fit_output_root"]),
    )
    expected = config.get("expected")
    if not isinstance(expected, dict):
        raise OracleGraphJobsError("graph jobs config lacks expected counts")
    observed = {
        "partitions": len({(str(row["scheme"]), int(row["fold"])) for row in rows}),
        "jobs": len(rows),
        "ensembles": len({str(row["ensemble_id"]) for row in rows}),
        "selection_eligible_jobs": sum(row["selection_eligible"] == "true" for row in rows),
    }
    for field, value in observed.items():
        if value != expected.get(field):
            raise OracleGraphJobsError(
                f"{field} changed: expected {expected.get(field)}, observed {value}"
            )
    payload = _gzip_csv(rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "frozen_pending_graph_job_matrix",
        "inputs": {
            "config": {
                "path": str(config_path.resolve().relative_to(repo_root.resolve())),
                "sha256": _sha256_file(config_path),
                "bytes": config_path.stat().st_size,
            },
            **verified,
        },
        "matrix": matrix,
        "execution": execution,
        "summary": {
            **observed,
            "architectures": len(architectures),
            "endpoints": len(endpoints),
            "seeds": len(seeds),
            "job_status_counts": dict(sorted(Counter(str(row["status"]) for row in rows).items())),
        },
        "decision": {
            "jobs_authorized_to_run": True,
            "oracle_model_frozen": False,
            "runtime_used_for_model_selection": False,
            "random_diagnostic_used_for_selection": False,
        },
        "artifacts": {
            "oracle_graph_jobs.csv.gz": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        },
    }
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(output_dir / "oracle_graph_jobs.csv.gz", payload)
    _atomic_write(output_dir / "oracle_graph_jobs_result.json", result_payload)
    return result
