"""Fresh native-terminal confirmation on unused morphology programs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.io import atomic_write as _atomic_write
from forge.core.io import write_json as _atomic_json
from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_dynamic_frozen_prior_terminal_census import (
    _jsonl_gzip_bytes,
    _sha256_payload,
    complete_native_no_route_terminal,
    load_census_contract,
)
from forge.product.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.product.ugi_selected_guidance_adapter import SelectedGuidanceState
from forge.product.ugi_selected_guidance_adapter_v3 import (
    build_selected_model_restartable_guidance_lane_v3,
)
from forge.product.ugi_selected_restartable_generator import SAMPLE_STEPS

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_confirmation_config.v1"
SHARD_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_confirmation_shard.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_confirmation.v1"
EXPECTED_SCOPE = {
    "partition": "unused_program_confirmation",
    "fresh_native_terminals": True,
    "potency_predictions": 0,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "proposal_calls": 0,
    "nonzero_guidance": False,
    "candidate_selection": False,
    "sealed_holdout_access": False,
    "retries_or_repairs": False,
}
EXPECTED_INPUTS = {
    "base_census_config",
    "proposal_result",
    "proposal_schedule",
    "runner",
    "source",
    "tests",
}


class UgiMorphologyProposalConfirmationError(RuntimeError):
    """Raised when the unused-program confirmation contract changes."""


@dataclass(frozen=True)
class ConfirmationDesign:
    programs: int
    shard_programs: int
    particle_seed_base: int
    terminal_seed_base: int
    device: str

    @property
    def shard_count(self) -> int:
        if self.programs % self.shard_programs:
            raise UgiMorphologyProposalConfirmationError("programs do not divide into shards")
        return self.programs // self.shard_programs


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyProposalConfirmationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiMorphologyProposalConfirmationError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyProposalConfirmationError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiMorphologyProposalConfirmationError(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiMorphologyProposalConfirmationError(f"pin changed: {label}")
    return path


def _keyed_seed(base: int, *, purpose: str, population_index: int) -> int:
    payload = json.dumps(
        {"base": base, "purpose": purpose, "population_index": population_index},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _design(value: Any) -> ConfirmationDesign:
    expected = {
        "programs",
        "shard_programs",
        "particle_seed_base",
        "terminal_seed_base",
        "device",
        "sample_steps",
        "terminals_per_program",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise UgiMorphologyProposalConfirmationError("confirmation design changed")
    if int(value["sample_steps"]) != SAMPLE_STEPS or int(value["terminals_per_program"]) != 1:
        raise UgiMorphologyProposalConfirmationError("confirmation terminal design changed")
    design = ConfirmationDesign(
        programs=int(value["programs"]),
        shard_programs=int(value["shard_programs"]),
        particle_seed_base=int(value["particle_seed_base"]),
        terminal_seed_base=int(value["terminal_seed_base"]),
        device=str(value["device"]),
    )
    if design.programs != 3072 or design.shard_programs < 1:
        raise UgiMorphologyProposalConfirmationError("confirmation population changed")
    return design


@dataclass(frozen=True)
class ConfirmationContract:
    repo: Path
    config_path: Path
    config_sha256: str
    inputs: Mapping[str, Path]
    design: ConfirmationDesign
    schedule: tuple[Mapping[str, Any], ...]
    schedule_sha256: str


def load_confirmation_contract(repo: Path, config_path: Path) -> ConfirmationContract:
    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="confirmation config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyProposalConfirmationError("unsupported confirmation schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiMorphologyProposalConfirmationError("confirmation scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiMorphologyProposalConfirmationError("confirmation input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    proposal_result = _load_json(paths["proposal_result"], label="proposal result")
    schedule_value = _load_json(paths["proposal_schedule"], label="proposal schedule")
    records = schedule_value.get("records")
    if (
        proposal_result.get("proposal", {}).get("schedule_sha256")
        != schedule_value.get("schedule_sha256")
        or not isinstance(records, list)
        or len(records) != 3072
    ):
        raise UgiMorphologyProposalConfirmationError("proposal schedule identity changed")
    population_indices = [int(record["population_index"]) for record in records]
    if len(set(population_indices)) != len(population_indices):
        raise UgiMorphologyProposalConfirmationError("proposal schedule has duplicate rows")
    base_contract = load_census_contract(repo, paths["base_census_config"])
    if base_contract.design.sample_steps != SAMPLE_STEPS:
        raise UgiMorphologyProposalConfirmationError("base generator sample steps changed")
    return ConfirmationContract(
        repo=repo,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        inputs=paths,
        design=_design(config["design"]),
        schedule=tuple(sorted(records, key=lambda record: int(record["population_index"]))),
        schedule_sha256=str(schedule_value["schedule_sha256"]),
    )


def confirmation_plan(repo: Path, config_path: Path) -> dict[str, Any]:
    contract = load_confirmation_contract(repo, config_path)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "planned_not_executed",
        "config_sha256": contract.config_sha256,
        "schedule_sha256": contract.schedule_sha256,
        "programs": contract.design.programs,
        "shards": contract.design.shard_count,
        "scope": dict(EXPECTED_SCOPE),
    }


def _shard_records(
    contract: ConfirmationContract, shard_index: int
) -> tuple[Mapping[str, Any], ...]:
    if not 0 <= shard_index < contract.design.shard_count:
        raise UgiMorphologyProposalConfirmationError("shard index is outside the design")
    start = shard_index * contract.design.shard_programs
    return contract.schedule[start : start + contract.design.shard_programs]


def _validate_shard(
    contract: ConfirmationContract, output_dir: Path, *, shard_index: int
) -> dict[str, Any]:
    path = output_dir / "shards" / f"shard_{shard_index:04d}" / "receipt.json"
    receipt = _load_json(path, label="confirmation shard receipt")
    logical = {key: value for key, value in receipt.items() if key != "result_sha256"}
    if receipt.get("result_sha256") != _sha256_payload(logical):
        raise UgiMorphologyProposalConfirmationError("confirmation shard hash changed")
    expected = _shard_records(contract, shard_index)
    if (
        receipt.get("schema_version") != SHARD_SCHEMA_VERSION
        or receipt.get("status") != "complete_unused_program_confirmation_shard"
        or receipt.get("config_sha256") != contract.config_sha256
        or receipt.get("schedule_sha256") != contract.schedule_sha256
        or receipt.get("shard_index") != shard_index
        or receipt.get("population_indices")
        != [int(record["population_index"]) for record in expected]
        or receipt.get("scope") != EXPECTED_SCOPE
    ):
        raise UgiMorphologyProposalConfirmationError("confirmation shard contract changed")
    terminal = receipt.get("terminal_ledger")
    if not isinstance(terminal, Mapping) or set(terminal) != {
        "path",
        "sha256",
        "logical_sha256",
        "rows",
    }:
        raise UgiMorphologyProposalConfirmationError("confirmation shard artifact changed")
    artifact = path.parent / str(terminal["path"])
    if (
        artifact.is_symlink()
        or not artifact.is_file()
        or sha256_file(artifact) != terminal["sha256"]
        or int(terminal["rows"]) != len(expected)
    ):
        raise UgiMorphologyProposalConfirmationError("confirmation terminal artifact changed")
    return receipt


def execute_confirmation_shard(
    contract: ConfirmationContract,
    output_dir: Path,
    *,
    shard_index: int,
    lane: Any,
) -> dict[str, Any]:
    final_dir = output_dir.resolve() / "shards" / f"shard_{shard_index:04d}"
    if final_dir.exists():
        return {
            **_validate_shard(contract, output_dir.resolve(), shard_index=shard_index),
            "resumed": True,
        }
    records = _shard_records(contract, shard_index)
    program_bytes = tuple(
        canonical_morphology_program_bytes(dict(record["program"])) for record in records
    )
    particle_seeds = tuple(
        _keyed_seed(
            contract.design.particle_seed_base,
            purpose="unused_program_confirmation_particle",
            population_index=int(record["population_index"]),
        )
        for record in records
    )
    terminal_seeds = tuple(
        _keyed_seed(
            contract.design.terminal_seed_base,
            purpose="unused_program_confirmation_terminal",
            population_index=int(record["population_index"]),
        )
        for record in records
    )
    if len(set((*particle_seeds, *terminal_seeds))) != 2 * len(records):
        raise UgiMorphologyProposalConfirmationError("confirmation seed collision")
    initialized = lane.initialize(
        program_bytes,
        seed=contract.design.particle_seed_base,
        particle_seeds=particle_seeds,
        device=contract.design.device,
    )
    state = initialized.state
    if not isinstance(state, SelectedGuidanceState):
        raise UgiMorphologyProposalConfirmationError("lane initialization changed")
    confirmation_checkpoint = 6
    advanced = lane.advance(state, target_step=confirmation_checkpoint)
    if advanced.product_transition_calls != len(records) * confirmation_checkpoint:
        raise UgiMorphologyProposalConfirmationError("transition accounting changed")
    state = advanced.state
    rows = []
    for particle_index, (record, seed) in enumerate(zip(records, terminal_seeds, strict=True)):
        completion = complete_native_no_route_terminal(
            lane,
            state,
            particle_index=particle_index,
            seed=seed,
            checkpoint_index=confirmation_checkpoint,
            rollout_index=0,
        )
        rows.append(
            {
                "shard_index": shard_index,
                "particle_index": particle_index,
                "population_index": int(record["population_index"]),
                "product_id": str(record["product_id"]),
                "program_sha256": str(record["program_sha256"]),
                "proposal_score": float(record["proposal_score"]),
                "proposal_probability": float(record["proposal_probability"]),
                "prior_probability": float(record["prior_probability"]),
                "importance_ratio_prior_over_proposal": float(
                    record["importance_ratio_prior_over_proposal"]
                ),
                "score_rank": int(record["score_rank"]),
                "score_quartile": int(record["score_quartile"]),
                **completion,
            }
        )
    work_root = output_dir.resolve() / "shards"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".shard_{shard_index:04d}.", dir=work_root))
    try:
        terminal_path = work_dir / "terminals.jsonl.gz"
        _atomic_write(terminal_path, _jsonl_gzip_bytes(rows))
        content = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "status": "complete_unused_program_confirmation_shard",
            "config_sha256": contract.config_sha256,
            "schedule_sha256": contract.schedule_sha256,
            "shard_index": shard_index,
            "population_indices": [int(record["population_index"]) for record in records],
            "particle_seed_manifest_sha256": _sha256_payload(particle_seeds),
            "terminal_seed_manifest_sha256": _sha256_payload(terminal_seeds),
            "terminal_ledger": {
                "path": terminal_path.name,
                "sha256": sha256_file(terminal_path),
                "logical_sha256": _sha256_payload(rows),
                "rows": len(rows),
            },
            "scope": dict(EXPECTED_SCOPE),
        }
        receipt = {**content, "result_sha256": _sha256_payload(content)}
        _atomic_json(work_dir / "receipt.json", receipt)
        os.replace(work_dir, final_dir)
        return {**receipt, "resumed": False}
    except Exception:
        raise


def run_all_confirmation_shards(
    repo: Path, config_path: Path, output_dir: Path
) -> list[dict[str, Any]]:
    contract = load_confirmation_contract(repo, config_path)
    lane = build_selected_model_restartable_guidance_lane_v3(contract.repo)
    return [
        execute_confirmation_shard(
            contract,
            output_dir,
            shard_index=shard_index,
            lane=lane,
        )
        for shard_index in range(contract.design.shard_count)
    ]


def aggregate_confirmation(repo: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    contract = load_confirmation_contract(repo, config_path)
    output_dir = output_dir.resolve()
    receipts = []
    rows: list[dict[str, Any]] = []
    for shard_index in range(contract.design.shard_count):
        receipt = _validate_shard(contract, output_dir, shard_index=shard_index)
        receipts.append(receipt)
        path = (
            output_dir / "shards" / f"shard_{shard_index:04d}" / receipt["terminal_ledger"]["path"]
        )
        with gzip.open(path, "rt") as handle:
            rows.extend(json.loads(line) for line in handle)
    if len(rows) != contract.design.programs:
        raise UgiMorphologyProposalConfirmationError("confirmation aggregate is incomplete")
    if len({int(row["population_index"]) for row in rows}) != len(rows):
        raise UgiMorphologyProposalConfirmationError("confirmation aggregate duplicates programs")
    ledger_path = output_dir / "terminal_ledger.jsonl.gz"
    _atomic_write(ledger_path, _jsonl_gzip_bytes(rows))
    valid = sum(bool(row.get("native_terminal", {}).get("terminal_valid")) for row in rows)
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_unused_program_morphology_confirmation_census",
        "config": {
            "path": str(contract.config_path.relative_to(contract.repo)),
            "sha256": contract.config_sha256,
        },
        "schedule_sha256": contract.schedule_sha256,
        "counts": {
            "shards": len(receipts),
            "programs": len(rows),
            "terminal_attempts": len(rows),
            "valid_exact_l1": valid,
        },
        "artifacts": {
            "terminal_ledger": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "logical_sha256": _sha256_payload(rows),
                "rows": len(rows),
            },
            "shard_receipts_sha256": _sha256_payload(
                [receipt["result_sha256"] for receipt in receipts]
            ),
        },
        "scope": dict(EXPECTED_SCOPE),
        "nonclaims": [
            "no potency, route or synthesis value was evaluated",
            "no nonzero guidance or candidate selection was performed",
            "terminal structural support remains to be scored read-only",
        ],
    }
    result = {**content, "result_sha256": _sha256_payload(content)}
    _atomic_json(output_dir / "result.json", result)
    return result


__all__ = [
    "ConfirmationContract",
    "ConfirmationDesign",
    "UgiMorphologyProposalConfirmationError",
    "aggregate_confirmation",
    "confirmation_plan",
    "execute_confirmation_shard",
    "load_confirmation_contract",
    "run_all_confirmation_shards",
]
