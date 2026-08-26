"""Fail-closed readiness diagnosis for the experiments named by the v1 manuscript."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments._runtime import diagnose_experiment
from experiments._runtime.source import source_fingerprint
from experiments._runtime.spec import ExperimentSpec
from experiments.catalog import SPECIFICATIONS
from forge.core.hashing import is_sha256, sha256_file
from forge_paper.contract import PaperContractError, PaperPin

EXPERIMENT_MATRIX_SCHEMA = "forge.paper_experiment_matrix.v1"
ENTRY_FIELDS = {
    "id",
    "title",
    "requirement",
    "kind",
    "declared_state",
    "manuscript_targets",
    "experiment_id",
    "profile",
    "replicates",
    "entrypoint",
    "evidence",
    "dependencies",
    "commands",
    "blockers",
    "notes",
    "source_change_required",
}
REQUIREMENTS = {"core", "row_retained", "context", "optional"}
KINDS = {
    "experiment",
    "aggregate",
    "frozen_result",
    "implementation",
    "external_integration",
    "optional",
}
DECLARED_STATES = {
    "completed",
    "launch_ready",
    "waiting_for_upstream",
    "not_implemented",
    "optional_unfrozen",
}


class ExperimentMatrixError(PaperContractError):
    """The paper experiment matrix is malformed or internally inconsistent."""


def _strings(value: object, *, label: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "non-empty " if not allow_empty else ""
        raise ExperimentMatrixError(f"{label} must be a {qualifier}string array")
    if not all(isinstance(item, str) and item for item in value):
        raise ExperimentMatrixError(f"{label} must contain non-empty strings")
    return tuple(value)


def _optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ExperimentMatrixError(f"{label} must be null or a non-empty string")
    return value


@dataclass(frozen=True)
class ExperimentMatrixEntry:
    entry_id: str
    title: str
    requirement: str
    kind: str
    declared_state: str
    manuscript_targets: tuple[str, ...]
    experiment_id: str | None
    profile: str | None
    replicates: tuple[int, ...]
    entrypoint: str | None
    evidence: tuple[PaperPin, ...]
    dependencies: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]
    blockers: tuple[str, ...]
    notes: tuple[str, ...]
    source_change_required: bool

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> ExperimentMatrixEntry:
        if not isinstance(value, dict) or set(value) != ENTRY_FIELDS:
            raise ExperimentMatrixError(f"{label} must define exactly {sorted(ENTRY_FIELDS)}")
        entry_id = value["id"]
        title = value["title"]
        if not isinstance(entry_id, str) or not entry_id or not entry_id.replace("_", "").isalnum():
            raise ExperimentMatrixError(f"{label}.id must use letters, numbers and underscores")
        if not isinstance(title, str) or not title:
            raise ExperimentMatrixError(f"{label}.title must be a non-empty string")
        requirement = value["requirement"]
        kind = value["kind"]
        declared_state = value["declared_state"]
        if requirement not in REQUIREMENTS:
            raise ExperimentMatrixError(f"{label}.requirement is unsupported: {requirement!r}")
        if kind not in KINDS:
            raise ExperimentMatrixError(f"{label}.kind is unsupported: {kind!r}")
        if declared_state not in DECLARED_STATES:
            raise ExperimentMatrixError(
                f"{label}.declared_state is unsupported: {declared_state!r}"
            )
        replicates_raw = value["replicates"]
        if not isinstance(replicates_raw, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in replicates_raw
        ):
            raise ExperimentMatrixError(f"{label}.replicates must be non-negative integers")
        if len(replicates_raw) != len(set(replicates_raw)):
            raise ExperimentMatrixError(f"{label}.replicates must be unique")
        evidence_raw = value["evidence"]
        if not isinstance(evidence_raw, list):
            raise ExperimentMatrixError(f"{label}.evidence must be an array")
        commands_raw = value["commands"]
        if not isinstance(commands_raw, list) or not all(
            isinstance(command, list)
            and command
            and all(isinstance(token, str) and token for token in command)
            for command in commands_raw
        ):
            raise ExperimentMatrixError(f"{label}.commands must be an array of command arrays")
        source_change = value["source_change_required"]
        if not isinstance(source_change, bool):
            raise ExperimentMatrixError(f"{label}.source_change_required must be boolean")
        experiment_id = _optional_string(value["experiment_id"], label=f"{label}.experiment_id")
        profile = _optional_string(value["profile"], label=f"{label}.profile")
        entrypoint = _optional_string(value["entrypoint"], label=f"{label}.entrypoint")
        if kind == "experiment" and (
            experiment_id is None or profile not in {"smoke", "full"} or not replicates_raw
        ):
            raise ExperimentMatrixError(
                f"{label} experiment entries require an id, profile and replicates"
            )
        if kind != "experiment" and experiment_id is not None:
            raise ExperimentMatrixError(f"{label} non-experiment entry cannot name an experiment")
        blockers = _strings(value["blockers"], label=f"{label}.blockers")
        if declared_state in {"not_implemented", "optional_unfrozen"} and not blockers:
            raise ExperimentMatrixError(f"{label} must state why it is not runnable")
        return cls(
            entry_id=entry_id,
            title=title,
            requirement=str(requirement),
            kind=str(kind),
            declared_state=str(declared_state),
            manuscript_targets=_strings(
                value["manuscript_targets"],
                label=f"{label}.manuscript_targets",
                allow_empty=False,
            ),
            experiment_id=experiment_id,
            profile=profile,
            replicates=tuple(replicates_raw),
            entrypoint=entrypoint,
            evidence=tuple(
                PaperPin.from_mapping(item, label=f"{label}.evidence[{index}]")
                for index, item in enumerate(evidence_raw)
            ),
            dependencies=_strings(value["dependencies"], label=f"{label}.dependencies"),
            commands=tuple(tuple(command) for command in commands_raw),
            blockers=blockers,
            notes=_strings(value["notes"], label=f"{label}.notes"),
            source_change_required=source_change,
        )


@dataclass(frozen=True)
class SourceFreeze:
    qualified_source_sha256: str
    preflight_experiment_id: str
    preflight_run_id: str
    preflight_result: PaperPin

    @classmethod
    def from_mapping(cls, value: object) -> SourceFreeze:
        fields = {
            "qualified_source_sha256",
            "preflight_experiment_id",
            "preflight_run_id",
            "preflight_result",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ExperimentMatrixError(f"source_freeze must define exactly {sorted(fields)}")
        digest = value["qualified_source_sha256"]
        experiment = value["preflight_experiment_id"]
        run_id = value["preflight_run_id"]
        if not isinstance(digest, str) or not is_sha256(digest):
            raise ExperimentMatrixError("source_freeze qualified source must be a SHA-256 digest")
        if not isinstance(experiment, str) or not experiment:
            raise ExperimentMatrixError("source_freeze preflight experiment must be named")
        if not isinstance(run_id, str) or not is_sha256(run_id):
            raise ExperimentMatrixError("source_freeze preflight run id must be a SHA-256 digest")
        return cls(
            qualified_source_sha256=digest,
            preflight_experiment_id=experiment,
            preflight_run_id=run_id,
            preflight_result=PaperPin.from_mapping(
                value["preflight_result"], label="source_freeze.preflight_result"
            ),
        )


@dataclass(frozen=True)
class ExperimentMatrix:
    paper_id: str
    as_of: str
    manuscript: str
    source_freeze: SourceFreeze
    entries: tuple[ExperimentMatrixEntry, ...]

    @classmethod
    def load(cls, path: Path) -> ExperimentMatrix:
        try:
            value: Any = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ExperimentMatrixError(
                f"experiment matrix could not be read: {path}: {error}"
            ) from error
        fields = {"schema_version", "paper_id", "as_of", "manuscript", "source_freeze", "entries"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ExperimentMatrixError(f"experiment matrix must define exactly {sorted(fields)}")
        if value["schema_version"] != EXPERIMENT_MATRIX_SCHEMA:
            raise ExperimentMatrixError("unsupported paper experiment matrix schema")
        for field in ("paper_id", "as_of", "manuscript"):
            if not isinstance(value[field], str) or not value[field]:
                raise ExperimentMatrixError(f"{field} must be a non-empty string")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ExperimentMatrixError("entries must be a non-empty array")
        entries = tuple(
            ExperimentMatrixEntry.from_mapping(item, label=f"entries[{index}]")
            for index, item in enumerate(raw_entries)
        )
        ids = {entry.entry_id for entry in entries}
        if len(ids) != len(entries):
            raise ExperimentMatrixError("entry ids must be unique")
        for entry in entries:
            unknown = set(entry.dependencies) - ids
            if unknown or entry.entry_id in entry.dependencies:
                raise ExperimentMatrixError(
                    f"entry {entry.entry_id!r} has invalid dependencies: {sorted(unknown)}"
                )
        return cls(
            paper_id=value["paper_id"],
            as_of=value["as_of"],
            manuscript=value["manuscript"],
            source_freeze=SourceFreeze.from_mapping(value["source_freeze"]),
            entries=entries,
        )


def _pin_status(repo: Path, pin: PaperPin) -> dict[str, Any]:
    path = repo / pin.path
    observed = str(sha256_file(path)) if path.is_file() and not path.is_symlink() else None
    return {
        "path": pin.path,
        "expected_sha256": pin.sha256,
        "observed_sha256": observed,
        "status": (
            "verified" if observed == pin.sha256 else "missing" if observed is None else "drift"
        ),
    }


def _entrypoint_exists(value: str | None) -> bool:
    if value is None or ":" not in value:
        return False
    module_name, attribute = value.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, RuntimeError):
        return False
    return callable(getattr(module, attribute, None))


def _matching_runs(
    repo: Path,
    *,
    experiment_id: str,
    profile: str,
    replicates: tuple[int, ...],
    source_sha256: str,
    spec_sha256: str,
) -> dict[int, str]:
    matches: dict[int, str] = {}
    for path in sorted((repo / "runs" / experiment_id).glob("*/run.json")):
        try:
            run = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        replicate = run.get("replicate")
        if (
            run.get("status") == "complete"
            and run.get("profile") == profile
            and run.get("source_sha256") == source_sha256
            and run.get("spec_sha256") == spec_sha256
            and isinstance(replicate, int)
            and replicate in replicates
        ):
            matches[replicate] = path.parent.name
    return matches


def diagnose_experiment_matrix(repo: Path, matrix_path: Path) -> dict[str, Any]:
    """Report what can run now and what still lacks a real implementation."""

    repo = repo.resolve()
    matrix = ExperimentMatrix.load(matrix_path)
    current_source = source_fingerprint(repo)
    preflight_pin = _pin_status(repo, matrix.source_freeze.preflight_result)
    preflight_run_path = (
        repo
        / "runs"
        / matrix.source_freeze.preflight_experiment_id
        / matrix.source_freeze.preflight_run_id
        / "run.json"
    )
    try:
        preflight_run = json.loads(preflight_run_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        preflight_run = {}
    source_freeze_intact = (
        current_source == matrix.source_freeze.qualified_source_sha256
        and preflight_pin["status"] == "verified"
        and preflight_run.get("status") == "complete"
        and preflight_run.get("source_sha256") == current_source
    )

    rows: list[dict[str, Any]] = []
    observed_by_id: dict[str, str] = {}
    for entry in matrix.entries:
        evidence = [_pin_status(repo, pin) for pin in entry.evidence]
        evidence_ok = all(item["status"] == "verified" for item in evidence)
        row: dict[str, Any] = {
            "id": entry.entry_id,
            "title": entry.title,
            "requirement": entry.requirement,
            "kind": entry.kind,
            "declared_state": entry.declared_state,
            "manuscript_targets": list(entry.manuscript_targets),
            "source_change_required": entry.source_change_required,
            "evidence": evidence,
            "dependencies": list(entry.dependencies),
            "commands": [list(command) for command in entry.commands],
            "blockers": list(entry.blockers),
            "notes": list(entry.notes),
        }
        if entry.kind == "experiment":
            assert entry.experiment_id is not None and entry.profile is not None
            relative_spec = SPECIFICATIONS.get(entry.experiment_id)
            if relative_spec is None:
                observed = "not_registered"
                row["experiment_ready"] = False
                row["completed_replicates"] = {}
            else:
                spec_path = repo / relative_spec
                diagnosis = diagnose_experiment(repo, spec_path)
                spec = ExperimentSpec.load(spec_path)
                if entry.profile not in spec.profiles or spec.replicates[entry.profile] <= max(
                    entry.replicates
                ):
                    raise ExperimentMatrixError(
                        f"{entry.entry_id} replicates exceed its registered experiment contract"
                    )
                matches = _matching_runs(
                    repo,
                    experiment_id=entry.experiment_id,
                    profile=entry.profile,
                    replicates=entry.replicates,
                    source_sha256=current_source,
                    spec_sha256=str(sha256_file(spec_path)),
                )
                row["experiment_ready"] = bool(diagnosis["ready"] and evidence_ok)
                row["completed_replicates"] = {
                    str(replicate): matches.get(replicate) for replicate in entry.replicates
                }
                observed = (
                    "completed"
                    if len(matches) == len(entry.replicates)
                    else "launch_ready" if diagnosis["ready"] and evidence_ok else "blocked"
                )
        elif entry.kind == "frozen_result":
            observed = "completed" if evidence and evidence_ok else "blocked"
        elif entry.kind == "aggregate":
            implementation_ready = _entrypoint_exists(entry.entrypoint)
            row["implementation_ready"] = implementation_ready
            dependencies_complete = all(
                observed_by_id.get(dependency) == "completed" for dependency in entry.dependencies
            )
            observed = (
                "ready_to_aggregate"
                if implementation_ready and dependencies_complete
                else "waiting_for_upstream" if implementation_ready else "not_implemented"
            )
        elif entry.kind == "optional":
            observed = "optional_unfrozen"
        elif entry.kind == "implementation":
            implementation_ready = _entrypoint_exists(entry.entrypoint) and evidence_ok
            row["implementation_ready"] = implementation_ready
            observed = "launch_ready" if implementation_ready else "not_implemented"
        elif entry.kind == "external_integration":
            adapter_ready = _entrypoint_exists(entry.entrypoint) and evidence_ok
            row["adapter_ready"] = adapter_ready
            native_port_ready = adapter_ready and bool(entry.commands) and not entry.blockers
            row["native_port_ready"] = native_port_ready
            observed = "launch_ready" if native_port_ready else (
                "waiting_for_upstream" if adapter_ready else "not_implemented"
            )
        else:
            observed = "not_implemented"
        observed_by_id[entry.entry_id] = observed
        row["observed_state"] = observed
        acceptable = {
            "completed": {"completed"},
            "launch_ready": {"launch_ready", "completed"},
            "waiting_for_upstream": {"waiting_for_upstream", "ready_to_aggregate"},
            "not_implemented": {"not_implemented"},
            "optional_unfrozen": {"optional_unfrozen"},
        }
        row["declaration_consistent"] = observed in acceptable[entry.declared_state]
        rows.append(row)

    row_by_id = {row["id"]: row for row in rows}
    immediate_ids = {"transformer_four_arm_production", "finite_catalogue_oracle"}
    immediate_production_ready = source_freeze_intact and all(
        row_by_id[entry_id]["observed_state"] in {"launch_ready", "completed"}
        for entry_id in immediate_ids
    )
    required_rows = [row for row in rows if row["requirement"] in {"core", "row_retained"}]
    setup_complete = all(
        row["observed_state"] in {"launch_ready", "completed", "ready_to_aggregate"}
        or (row["kind"] == "aggregate" and row["observed_state"] == "waiting_for_upstream")
        for row in required_rows
    )
    paper_results_complete = all(row["observed_state"] == "completed" for row in required_rows)
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row["observed_state"])
        counts[state] = counts.get(state, 0) + 1
    blockers = [
        {
            "id": row["id"],
            "state": row["observed_state"],
            "blockers": row["blockers"],
        }
        for row in required_rows
        if row["observed_state"]
        in {"blocked", "not_implemented", "not_registered", "waiting_for_upstream"}
        and row["kind"] != "aggregate"
    ]
    return {
        "schema_version": "forge.paper_experiment_diagnosis.v1",
        "paper_id": matrix.paper_id,
        "matrix": str(matrix_path.resolve().relative_to(repo)),
        "manuscript": matrix.manuscript,
        "as_of": matrix.as_of,
        "source_freeze": {
            "current_source_sha256": current_source,
            "qualified_source_sha256": matrix.source_freeze.qualified_source_sha256,
            "intact": source_freeze_intact,
            "preflight_result": preflight_pin,
        },
        "immediate_production_ready": immediate_production_ready,
        "setup_complete_for_all_retained_rows": setup_complete,
        "paper_results_complete": paper_results_complete,
        "counts": dict(sorted(counts.items())),
        "blockers": blockers,
        "entries": rows,
    }


__all__ = [
    "EXPERIMENT_MATRIX_SCHEMA",
    "ExperimentMatrix",
    "ExperimentMatrixEntry",
    "ExperimentMatrixError",
    "diagnose_experiment_matrix",
]
