"""Read-only evidence validator for the frozen dynamic terminal census.

The shard producer is already frozen.  This module does not import its CLI or
change its artifacts; it independently checks the completed shard lattice and
the aggregate assembled from it.  In particular, it closes validation gaps
that a count-only aggregate could miss: logical row hashes, exact coordinate
membership, global identities/seeds, and deterministic reconstruction of the
layout/source tensors intentionally omitted from compact partial-state files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.model.defog_feasibility import sha256_file
from experiments.phase1.product_l1.sampling.terminal_census import (
    RESULT_SCHEMA_VERSION as CENSUS_RESULT_SCHEMA_VERSION,
)
from experiments.phase1.product_l1.sampling.terminal_census import (
    SHARD_SCHEMA_VERSION,
    CensusContract,
    SelectedProgramManifest,
    UgiDynamicTerminalCensusError,
    _validate_completed_shard,
    load_census_contract,
    load_selected_program_manifest,
    particle_seed,
    restore_partial_state,
    rollout_seed,
)
from experiments.phase1.synthesis_guidance.adapters.selected_v1 import SelectedGuidanceState
from experiments.phase1.synthesis_guidance.adapters.selected_v3 import (
    build_selected_model_restartable_guidance_lane_v3,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - productive validation requires torch
    torch = None


CONFIG_SCHEMA_VERSION = "phase1_ugi_dynamic_terminal_census_validator_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_dynamic_terminal_census_validation.v1"
EXPECTED_INPUT_KEYS = frozenset(
    {
        "census_config",
        "census_source",
        "validator_source",
        "validator_runner",
        "validator_tests",
    }
)
EXPECTED_SCOPE = {
    "read_only": True,
    "candidate_selection": False,
    "guidance": False,
    "oracle_calls": 0,
    "potency_predictions": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "proposal_calls": 0,
    "sealed_holdout_access": False,
    "prospective_candidate_lock": False,
}


class UgiDynamicTerminalCensusValidationError(RuntimeError):
    """Raised when completed census evidence is incomplete or inconsistent."""


def _stable_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise UgiDynamicTerminalCensusValidationError(
            "validation value is not canonically serializable"
        ) from error


def _producer_sha256_payload(value: Any) -> str:
    """Mirror the already-frozen producer's logical-hash serialization."""

    try:
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise UgiDynamicTerminalCensusValidationError(
            "producer evidence is not canonically serializable"
        ) from error
    return hashlib.sha256(serialized).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UgiDynamicTerminalCensusValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiDynamicTerminalCensusValidationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiDynamicTerminalCensusValidationError(f"{label} must be a JSON object")
    return value


def _repository_file(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiDynamicTerminalCensusValidationError(f"{label} pin is malformed")
    expected = _require_sha256(record["sha256"], label=f"{label} sha256")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiDynamicTerminalCensusValidationError(
            f"{label} path escapes the repository"
        ) from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
        raise UgiDynamicTerminalCensusValidationError(f"{label} pin changed")
    return path


@dataclass(frozen=True)
class ValidationContract:
    """Hash-pinned validator inputs and the frozen producer contract."""

    repo: Path
    config_path: Path
    config_sha256: str
    census_contract: CensusContract
    inputs: Mapping[str, Path]
    scope: Mapping[str, Any]


def load_validation_contract(repo: Path, config_path: Path) -> ValidationContract:
    """Load the validator and independently authenticate the frozen producer."""

    root = repo.resolve()
    resolved_config = config_path.resolve()
    try:
        resolved_config.relative_to(root)
    except ValueError as error:
        raise UgiDynamicTerminalCensusValidationError(
            "validator config escapes the repository"
        ) from error
    config = _read_json(resolved_config, label="validator config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiDynamicTerminalCensusValidationError("unsupported validator config schema")
    if config.get("status") != "frozen_before_post_aggregation_validation":
        raise UgiDynamicTerminalCensusValidationError("validator config is not frozen")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiDynamicTerminalCensusValidationError("validator scope changed")
    records = config.get("inputs")
    if not isinstance(records, dict) or frozenset(records) != EXPECTED_INPUT_KEYS:
        raise UgiDynamicTerminalCensusValidationError("validator input set changed")
    inputs = {
        label: _repository_file(root, record, label=label) for label, record in records.items()
    }
    census_contract = load_census_contract(root, inputs["census_config"])
    if census_contract.inputs["census_source"] != inputs["census_source"]:
        raise UgiDynamicTerminalCensusValidationError(
            "validator and producer disagree on the census source"
        )
    return ValidationContract(
        repo=root,
        config_path=resolved_config,
        config_sha256=sha256_file(resolved_config),
        census_contract=census_contract,
        inputs=inputs,
        scope=dict(config["scope"]),
    )


def _tensor_mapping_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if torch is None or set(left) != set(right):
        return False
    return all(
        torch.is_tensor(left[key])
        and torch.is_tensor(right[key])
        and left[key].dtype == right[key].dtype
        and tuple(left[key].shape) == tuple(right[key].shape)
        and torch.equal(left[key].detach().cpu(), right[key].detach().cpu())
        for key in left
    )


def _tensor_mapping_sha256(value: Mapping[str, Any]) -> str:
    if torch is None:
        raise UgiDynamicTerminalCensusValidationError("state validation requires torch")
    digest = hashlib.sha256()
    for key in sorted(value):
        tensor = value[key]
        if not torch.is_tensor(tensor):
            raise UgiDynamicTerminalCensusValidationError(
                f"derived state field {key} is not a tensor"
            )
        local = tensor.detach().cpu().contiguous()
        digest.update(_stable_json_bytes([key, str(local.dtype), list(local.shape)]))
        digest.update(local.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_packed_state(path: Path) -> dict[str, Any]:
    if torch is None:
        raise UgiDynamicTerminalCensusValidationError("state validation requires torch")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise UgiDynamicTerminalCensusValidationError(
            f"packed state cannot be loaded: {path}"
        ) from error
    if not isinstance(value, dict):
        raise UgiDynamicTerminalCensusValidationError("packed state is not a mapping")
    return value


def _read_jsonl_gzip(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with gzip.open(path, "rt") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise UgiDynamicTerminalCensusValidationError(
                        f"{label} row {line_number} is not an object"
                    )
                rows.append(row)
    except UgiDynamicTerminalCensusValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiDynamicTerminalCensusValidationError(f"invalid {label}: {path}") from error
    return rows


def _expected_programs_for_shard(
    manifest: SelectedProgramManifest,
    contract: CensusContract,
    shard_index: int,
) -> tuple[Any, ...]:
    start = shard_index * contract.design.shard_programs
    return manifest.programs[start : start + contract.design.shard_programs]


def _expanded_shard_design(
    programs: Sequence[Any],
    contract: CensusContract,
) -> tuple[tuple[bytes, ...], tuple[int, ...], tuple[dict[str, Any], ...]]:
    program_bytes = []
    seeds = []
    metadata = []
    for program in programs:
        for replicate in range(contract.design.states_per_program):
            program_bytes.append(program.program_bytes)
            seeds.append(
                particle_seed(
                    contract.design,
                    population_index=program.population_index,
                    state_replicate=replicate,
                )
            )
            metadata.append(
                {
                    **program.manifest_record(),
                    "state_replicate": replicate,
                }
            )
    return tuple(program_bytes), tuple(seeds), tuple(metadata)


def _completion_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "adapter_identity_sha256": row.get("adapter_identity_sha256"),
        "particle_state_sha256": row.get("particle_state_sha256"),
        "particle_index": row.get("particle_index"),
        "checkpoint_index": row.get("checkpoint_index"),
        "rollout_index": row.get("rollout_index"),
        "seed": row.get("seed"),
    }


def _require_row_identity(
    row: Mapping[str, Any],
    *,
    shard_index: int,
    metadata: Mapping[str, Any],
    particle_index: int,
    checkpoint: int,
    rollout_index: int,
    expected_seed: int,
    expected_particle_sha256: str,
    adapter_identity_sha256: str,
    sample_steps: int,
) -> None:
    expected_fields = {
        "shard_index": shard_index,
        **metadata,
        "particle_index": particle_index,
        "checkpoint_index": checkpoint,
        "rollout_index": rollout_index,
        "seed": expected_seed,
        "particle_state_sha256": expected_particle_sha256,
        "adapter_identity_sha256": adapter_identity_sha256,
        "source_state_unchanged": True,
        "product_transition_calls": sample_steps - checkpoint,
        "status": "terminal",
        "error_detail": None,
    }
    for key, expected in expected_fields.items():
        if row.get(key) != expected:
            raise UgiDynamicTerminalCensusValidationError(
                f"terminal coordinate field {key} changed in shard {shard_index}"
            )
    if not isinstance(row.get("native_terminal"), dict):
        raise UgiDynamicTerminalCensusValidationError(
            f"terminal coordinate lacks a native row in shard {shard_index}"
        )
    expected_completion_id = _producer_sha256_payload(_completion_identity(row))
    if row.get("completion_id") != expected_completion_id:
        raise UgiDynamicTerminalCensusValidationError(
            f"terminal completion identity changed in shard {shard_index}"
        )


def _require_aggregate_result_hash(result: Mapping[str, Any]) -> str:
    claimed = _require_sha256(result.get("result_sha256"), label="aggregate result")
    content = {key: value for key, value in result.items() if key != "result_sha256"}
    if _producer_sha256_payload(content) != claimed:
        raise UgiDynamicTerminalCensusValidationError("aggregate logical result hash changed")
    return claimed


def _artifact_path(root: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiDynamicTerminalCensusValidationError(f"aggregate {label} artifact is malformed")
    expected = _require_sha256(record["sha256"], label=f"aggregate {label}")
    path = (root / str(record["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise UgiDynamicTerminalCensusValidationError(
            f"aggregate {label} artifact escapes the census root"
        ) from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
        raise UgiDynamicTerminalCensusValidationError(f"aggregate {label} artifact changed")
    return path


def validate_completed_census(
    contract: CensusContract,
    manifest: SelectedProgramManifest,
    census_dir: Path,
    *,
    lane: Any,
) -> dict[str, Any]:
    """Validate all shard and aggregate evidence without changing census files."""

    if census_dir.is_symlink():
        raise UgiDynamicTerminalCensusValidationError("census directory is missing or symbolic")
    root = census_dir.resolve()
    if not root.is_dir():
        raise UgiDynamicTerminalCensusValidationError("census directory is missing or symbolic")
    if getattr(lane, "adapter_identity_sha256", None) is None:
        raise UgiDynamicTerminalCensusValidationError("validation lane lacks an identity")

    terminal_rows: list[dict[str, Any]] = []
    expected_state_manifest = []
    receipt_hashes = []
    all_completion_ids = []
    all_rollout_seeds = []
    all_particle_seeds = []
    all_coordinates = []
    state_reconstruction_records = []

    for shard_index in range(contract.design.shard_count):
        shard_dir = root / "shards" / f"shard_{shard_index:04d}"
        if not shard_dir.is_dir() or shard_dir.is_symlink():
            raise UgiDynamicTerminalCensusValidationError(
                f"census shard {shard_index} is incomplete"
            )
        try:
            receipt = _validate_completed_shard(
                shard_dir,
                contract=contract,
                manifest=manifest,
                shard_index=shard_index,
            )
        except UgiDynamicTerminalCensusError as error:
            raise UgiDynamicTerminalCensusValidationError(
                f"census shard {shard_index} failed producer validation"
            ) from error
        receipt_hashes.append(receipt["result_sha256"])
        if receipt.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise UgiDynamicTerminalCensusValidationError("shard schema changed")
        if receipt.get("adapter_identity_sha256") != lane.adapter_identity_sha256:
            raise UgiDynamicTerminalCensusValidationError(
                f"census shard {shard_index} used another adapter"
            )
        local_programs = _expected_programs_for_shard(manifest, contract, shard_index)
        programs, seeds, particle_metadata = _expanded_shard_design(local_programs, contract)
        if len(set(seeds)) != len(seeds):
            raise UgiDynamicTerminalCensusValidationError("particle seed collision within shard")
        all_particle_seeds.extend(seeds)
        if receipt.get("particle_seed_manifest_sha256") != _producer_sha256_payload(seeds):
            raise UgiDynamicTerminalCensusValidationError(
                f"particle seed manifest changed in shard {shard_index}"
            )

        initialized = lane.initialize(
            programs,
            seed=contract.design.particle_seed_base,
            particle_seeds=seeds,
            device=contract.design.device,
        ).state
        if not isinstance(initialized, SelectedGuidanceState):
            raise UgiDynamicTerminalCensusValidationError(
                "validation lane returned an invalid reference state"
            )
        states_by_checkpoint = {}
        for checkpoint in contract.design.checkpoints:
            label = f"partial_state_step_{checkpoint:02d}"
            artifact = receipt["artifacts"][label]
            path = shard_dir / artifact["path"]
            payload = _load_packed_state(path)
            try:
                restored = restore_partial_state(lane, payload)
            except UgiDynamicTerminalCensusError as error:
                raise UgiDynamicTerminalCensusValidationError(
                    f"packed state failed logical replay in shard {shard_index} step {checkpoint}"
                ) from error
            if restored.step != checkpoint or restored.state_sha256 != artifact["state_sha256"]:
                raise UgiDynamicTerminalCensusValidationError(
                    f"packed state boundary changed in shard {shard_index} step {checkpoint}"
                )
            for particle_index, (observed, reference) in enumerate(
                zip(restored.particles, initialized.particles, strict=True)
            ):
                if observed.program_bytes != reference.program_bytes:
                    raise UgiDynamicTerminalCensusValidationError(
                        "reconstructed state changed its morphology program"
                    )
                if not _tensor_mapping_equal(
                    observed.trajectory.layout,
                    reference.trajectory.layout,
                ):
                    raise UgiDynamicTerminalCensusValidationError(
                        f"layout reconstruction is nondeterministic for particle {particle_index}"
                    )
                if not _tensor_mapping_equal(
                    observed.trajectory.sources,
                    reference.trajectory.sources,
                ):
                    raise UgiDynamicTerminalCensusValidationError(
                        f"source reconstruction is nondeterministic for particle {particle_index}"
                    )
            states_by_checkpoint[checkpoint] = restored
            state_reconstruction_records.append(
                {
                    "shard_index": shard_index,
                    "checkpoint": checkpoint,
                    "state_sha256": restored.state_sha256,
                    "layout_sha256": _sha256_payload(
                        [
                            _tensor_mapping_sha256(particle.trajectory.layout)
                            for particle in restored.particles
                        ]
                    ),
                    "sources_sha256": _sha256_payload(
                        [
                            _tensor_mapping_sha256(particle.trajectory.sources)
                            for particle in restored.particles
                        ]
                    ),
                }
            )
            expected_state_manifest.append(
                {
                    "shard_index": shard_index,
                    "label": label,
                    "path": str(Path("shards") / shard_dir.name / artifact["path"]),
                    "sha256": artifact["sha256"],
                    "state_sha256": artifact["state_sha256"],
                    "step": artifact["step"],
                    "particles": artifact["particles"],
                }
            )

        terminal_artifact = receipt["artifacts"]["terminals"]
        local_rows = _read_jsonl_gzip(
            shard_dir / terminal_artifact["path"],
            label=f"shard {shard_index} terminal ledger",
        )
        if len(local_rows) != terminal_artifact["rows"]:
            raise UgiDynamicTerminalCensusValidationError(
                f"shard {shard_index} terminal row count changed"
            )
        if _producer_sha256_payload(local_rows) != terminal_artifact["logical_sha256"]:
            raise UgiDynamicTerminalCensusValidationError(
                f"shard {shard_index} terminal logical hash changed"
            )
        rows_by_coordinate: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in local_rows:
            coordinate = (
                row.get("population_index"),
                row.get("state_replicate"),
                row.get("checkpoint_index"),
                row.get("rollout_index"),
            )
            if coordinate in rows_by_coordinate:
                raise UgiDynamicTerminalCensusValidationError(
                    f"shard {shard_index} repeats a terminal coordinate"
                )
            rows_by_coordinate[coordinate] = row
        expected_coordinates = set()
        for particle_index, metadata in enumerate(particle_metadata):
            for checkpoint in contract.design.checkpoints:
                state = states_by_checkpoint[checkpoint]
                particle_sha256 = state.particles[particle_index].state_sha256
                for rollout_index in range(contract.design.rollouts_per_state_checkpoint):
                    expected_coordinates.add(
                        (
                            metadata["population_index"],
                            metadata["state_replicate"],
                            checkpoint,
                            rollout_index,
                        )
                    )
                    expected_seed = rollout_seed(
                        contract.design,
                        population_index=metadata["population_index"],
                        state_replicate=metadata["state_replicate"],
                        checkpoint_index=checkpoint,
                        rollout_index=rollout_index,
                    )
                    coordinate = (
                        metadata["population_index"],
                        metadata["state_replicate"],
                        checkpoint,
                        rollout_index,
                    )
                    row = rows_by_coordinate.get(coordinate)
                    if row is None:
                        raise UgiDynamicTerminalCensusValidationError(
                            f"shard {shard_index} does not contain each expected coordinate once"
                        )
                    _require_row_identity(
                        row,
                        shard_index=shard_index,
                        metadata=metadata,
                        particle_index=particle_index,
                        checkpoint=checkpoint,
                        rollout_index=rollout_index,
                        expected_seed=expected_seed,
                        expected_particle_sha256=particle_sha256,
                        adapter_identity_sha256=lane.adapter_identity_sha256,
                        sample_steps=contract.design.sample_steps,
                    )
        observed_coordinates = {
            (
                row.get("population_index"),
                row.get("state_replicate"),
                row.get("checkpoint_index"),
                row.get("rollout_index"),
            )
            for row in local_rows
        }
        if observed_coordinates != expected_coordinates:
            raise UgiDynamicTerminalCensusValidationError(
                f"shard {shard_index} coordinate membership changed"
            )
        terminal_rows.extend(local_rows)
        all_coordinates.extend(observed_coordinates)
        all_completion_ids.extend(row["completion_id"] for row in local_rows)
        all_rollout_seeds.extend(row["seed"] for row in local_rows)

    expected_attempts = contract.design.terminal_attempts
    if len(terminal_rows) != expected_attempts or len(set(all_coordinates)) != expected_attempts:
        raise UgiDynamicTerminalCensusValidationError(
            "global terminal coordinate lattice is incomplete"
        )
    if len(set(all_completion_ids)) != expected_attempts:
        raise UgiDynamicTerminalCensusValidationError("global completion IDs are not unique")
    if len(set(all_rollout_seeds)) != expected_attempts:
        raise UgiDynamicTerminalCensusValidationError("global rollout seeds are not unique")
    if len(set(all_particle_seeds)) != len(all_particle_seeds):
        raise UgiDynamicTerminalCensusValidationError("global particle seeds are not unique")
    if set(all_particle_seeds).intersection(all_rollout_seeds):
        raise UgiDynamicTerminalCensusValidationError(
            "particle and rollout seed namespaces collide"
        )

    aggregate = _read_json(root / "result.json", label="aggregate census result")
    aggregate_sha256 = _require_aggregate_result_hash(aggregate)
    if (
        aggregate.get("schema_version") != CENSUS_RESULT_SCHEMA_VERSION
        or aggregate.get("status") != "complete_calibration_only_dynamic_terminal_census"
        or aggregate.get("config")
        != {
            "path": str(contract.config_path.relative_to(contract.repo)),
            "sha256": contract.config_sha256,
        }
        or aggregate.get("design") != contract.design.to_dict()
        or aggregate.get("selected_program_manifest_sha256") != manifest.manifest_sha256
        or aggregate.get("population_manifest_sha256") != manifest.population_manifest_sha256
        or aggregate.get("scope") != contract.scope
    ):
        raise UgiDynamicTerminalCensusValidationError("aggregate census contract changed")
    artifacts = aggregate.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "terminal_ledger",
        "partial_state_manifest",
        "shard_receipts_sha256",
    }:
        raise UgiDynamicTerminalCensusValidationError("aggregate artifact set changed")
    if artifacts["shard_receipts_sha256"] != _producer_sha256_payload(receipt_hashes):
        raise UgiDynamicTerminalCensusValidationError("aggregate shard receipt binding changed")
    aggregate_terminal_path = _artifact_path(
        root,
        artifacts["terminal_ledger"],
        label="terminal ledger",
    )
    aggregate_rows = _read_jsonl_gzip(
        aggregate_terminal_path,
        label="aggregate terminal ledger",
    )
    if aggregate_rows != terminal_rows:
        raise UgiDynamicTerminalCensusValidationError(
            "aggregate terminal ledger differs from ordered shard evidence"
        )
    state_manifest_path = _artifact_path(
        root,
        artifacts["partial_state_manifest"],
        label="partial-state manifest",
    )
    try:
        state_manifest = json.loads(state_manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiDynamicTerminalCensusValidationError(
            "aggregate partial-state manifest is invalid"
        ) from error
    if state_manifest != expected_state_manifest:
        raise UgiDynamicTerminalCensusValidationError(
            "aggregate partial-state manifest differs from shard evidence"
        )
    statuses: dict[str, int] = {}
    raw_valid = 0
    exact_l1 = 0
    native_rows = 0
    for row in terminal_rows:
        status = str(row.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        native = row.get("native_terminal")
        if isinstance(native, dict):
            native_rows += 1
            raw_valid += int(native.get("raw_molecule_valid") is True)
            exact_l1 += int(native.get("terminal_valid") is True)
    expected_counts = {
        "shards": contract.design.shard_count,
        "partial_states": contract.design.selected_partial_states,
        "terminal_attempts": expected_attempts,
        "native_rows": native_rows,
        "raw_molecule_valid": raw_valid,
        "valid_exact_l1": exact_l1,
        "statuses": statuses,
    }
    if aggregate.get("counts") != expected_counts:
        raise UgiDynamicTerminalCensusValidationError("aggregate scientific counts changed")

    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "validated_complete_dynamic_terminal_census_read_only",
        "validated_at_utc": _utc_now(),
        "census_config_sha256": contract.config_sha256,
        "aggregate_result_sha256": aggregate_sha256,
        "selected_program_manifest_sha256": manifest.manifest_sha256,
        "population_manifest_sha256": manifest.population_manifest_sha256,
        "adapter_identity_sha256": lane.adapter_identity_sha256,
        "counts": expected_counts,
        "global_identities": {
            "particle_seeds": len(all_particle_seeds),
            "unique_particle_seeds": len(set(all_particle_seeds)),
            "rollout_seeds": len(all_rollout_seeds),
            "unique_rollout_seeds": len(set(all_rollout_seeds)),
            "completion_ids": len(all_completion_ids),
            "unique_completion_ids": len(set(all_completion_ids)),
            "coordinates": len(all_coordinates),
            "unique_coordinates": len(set(all_coordinates)),
        },
        "state_reconstruction_manifest_sha256": _sha256_payload(state_reconstruction_records),
        "terminal_ledger_logical_sha256": _sha256_payload(terminal_rows),
        "scope": dict(EXPECTED_SCOPE),
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def validate_completed_census_from_config(
    repo: Path,
    config_path: Path,
    census_dir: Path,
) -> dict[str, Any]:
    """Build the authenticated lane and validate a completed aggregate."""

    validation = load_validation_contract(repo, config_path)
    manifest = load_selected_program_manifest(validation.census_contract)
    lane = build_selected_model_restartable_guidance_lane_v3(validation.repo)
    result = validate_completed_census(
        validation.census_contract,
        manifest,
        census_dir,
        lane=lane,
    )
    content = {
        **{key: value for key, value in result.items() if key != "result_sha256"},
        "validator_config": {
            "path": str(validation.config_path.relative_to(validation.repo)),
            "sha256": validation.config_sha256,
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def write_validation_receipt(path: Path, result: Mapping[str, Any], *, census_dir: Path) -> None:
    """Write one receipt outside the immutable census evidence directory."""

    resolved = path.resolve()
    census_root = census_dir.resolve()
    try:
        resolved.relative_to(census_root)
    except ValueError:
        pass
    else:
        raise UgiDynamicTerminalCensusValidationError(
            "validation receipt must be written outside the census evidence directory"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(dict(result), indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "EXPECTED_SCOPE",
    "RESULT_SCHEMA_VERSION",
    "UgiDynamicTerminalCensusValidationError",
    "ValidationContract",
    "load_validation_contract",
    "validate_completed_census",
    "validate_completed_census_from_config",
    "write_validation_receipt",
]
