"""Restartable selected-v3 generation for the frozen production candidate arms."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.io import atomic_write as _atomic_write
from forge.core.io import write_json as _atomic_json
from forge.data.r1_prime_audit import sha256_file
from forge.design.flow.ugi_dynamic_frozen_prior_terminal_census import (
    _jsonl_gzip_bytes,
    _sha256_payload,
    complete_native_no_route_terminal,
    load_census_contract,
)
from forge.design.flow.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.design.flow.ugi_selected_guidance_adapter import SelectedGuidanceState
from forge.design.flow.ugi_selected_guidance_adapter_v3 import (
    build_selected_model_restartable_guidance_lane_v3,
)
from forge.design.sampling.ugi_selected_restartable_generator import SAMPLE_STEPS
from forge.design.schedule.ugi_production_candidate_schedule import ARM_IDS

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_candidate_generation_config.v1"
SHARD_SCHEMA_VERSION = "phase1_ugi_production_candidate_generation_shard.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_candidate_generation.v1"
TERMINAL_LEDGER_SCHEMA_VERSION = "forge.ugi_production_candidate_terminal_ledger.v1"
EXPECTED_SCOPE = {
    "matched_production_candidate_pool_only": True,
    "native_selected_v3_terminal_completion": True,
    "common_particle_and_terminal_randomness_across_arms": True,
    "terminal_scoring": False,
    "potency_model_calls": 0,
    "applicability_model_calls": 0,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "nonzero_trajectory_guidance": False,
    "candidate_selection": False,
    "sealed_holdout_access": False,
    "retries_or_repairs": False,
}
EXPECTED_INPUTS = {
    "base_census_config",
    "runner",
    "schedule",
    "schedule_result",
    "source",
    "tests",
}
TERMINAL_LEDGER_REQUIRED_FIELDS = {
    "arm_id",
    "draw_index",
    "common_uniform",
    "support_index",
    "program_sha256",
    "program",
    "selected_probability",
    "broad_prior_probability",
    "support_proposal_probability",
    "importance_ratio_broad_over_arm",
    "support_score",
    "particle_seed",
    "terminal_seed",
    "shard_index",
    "particle_index",
    "native_terminal",
}


class UgiProductionCandidateGenerationError(RuntimeError):
    """Raised when production candidate generation is not exact and restartable."""


@dataclass(frozen=True)
class ProductionCandidateDesign:
    draws_per_arm: int
    shard_draws: int
    particle_seed_base: int
    terminal_seed_base: int
    device: str
    checkpoint: int

    @property
    def shard_count(self) -> int:
        if self.shard_draws < 1 or self.draws_per_arm % self.shard_draws:
            raise UgiProductionCandidateGenerationError(
                "draws_per_arm must divide exactly into positive shards"
            )
        return self.draws_per_arm // self.shard_draws

    @property
    def terminal_attempts(self) -> int:
        return self.draws_per_arm * len(ARM_IDS)


@dataclass(frozen=True)
class ProductionCandidateContract:
    repo: Path
    config_path: Path
    config_sha256: str
    inputs: Mapping[str, Path]
    design: ProductionCandidateDesign
    schedule: tuple[Mapping[str, Any], ...]
    schedule_sha256: str


NativeCompleter = Callable[..., dict[str, Any]]


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiProductionCandidateGenerationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiProductionCandidateGenerationError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiProductionCandidateGenerationError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiProductionCandidateGenerationError(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiProductionCandidateGenerationError(f"pin changed: {label}")
    return path


def production_draw_seed(base: int, *, purpose: str, draw_index: int) -> int:
    if base < 0 or draw_index < 0 or not purpose:
        raise UgiProductionCandidateGenerationError("invalid production seed coordinate")
    payload = json.dumps(
        {"base": base, "purpose": purpose, "draw_index": draw_index},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _design(value: Any) -> ProductionCandidateDesign:
    expected = {
        "draws_per_arm",
        "shard_draws",
        "particle_seed_base",
        "terminal_seed_base",
        "device",
        "sample_steps",
        "checkpoint",
        "terminals_per_draw_per_arm",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise UgiProductionCandidateGenerationError("production generation design changed")
    if (
        int(value["draws_per_arm"]) != 16384
        or int(value["shard_draws"]) != 128
        or int(value["sample_steps"]) != SAMPLE_STEPS
        or int(value["checkpoint"]) != 6
        or int(value["terminals_per_draw_per_arm"]) != 1
        or str(value["device"]) != "cpu"
    ):
        raise UgiProductionCandidateGenerationError(
            "production generation differs from the frozen selected-v3 design"
        )
    design = ProductionCandidateDesign(
        draws_per_arm=int(value["draws_per_arm"]),
        shard_draws=int(value["shard_draws"]),
        particle_seed_base=int(value["particle_seed_base"]),
        terminal_seed_base=int(value["terminal_seed_base"]),
        device=str(value["device"]),
        checkpoint=int(value["checkpoint"]),
    )
    if design.particle_seed_base == design.terminal_seed_base:
        raise UgiProductionCandidateGenerationError("particle and terminal seeds overlap")
    _ = design.shard_count
    return design


def _validate_schedule_records(records: Any, *, draws: int) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(records, list) or len(records) != draws:
        raise UgiProductionCandidateGenerationError("production schedule draw count changed")
    required = {
        "support_index",
        "program_sha256",
        "program",
        "selected_probability",
        "broad_prior_probability",
        "support_proposal_probability",
        "importance_ratio_broad_over_arm",
        "support_score",
    }
    for draw_index, row in enumerate(records):
        if (
            not isinstance(row, Mapping)
            or int(row.get("draw_index", -1)) != draw_index
            or not 0.0 <= float(row.get("common_uniform", -1.0)) < 1.0
            or not isinstance(row.get("arms"), Mapping)
            or tuple(row["arms"]) != ARM_IDS
        ):
            raise UgiProductionCandidateGenerationError("production schedule order changed")
        for arm in ARM_IDS:
            arm_record = row["arms"][arm]
            if not isinstance(arm_record, Mapping) or set(arm_record) != required:
                raise UgiProductionCandidateGenerationError(f"{arm} schedule schema changed")
            canonical = canonical_morphology_program_bytes(dict(arm_record["program"]))
            if (
                hashlib.sha256(canonical).hexdigest() != arm_record["program_sha256"]
                or float(arm_record["selected_probability"]) <= 0.0
                or float(arm_record["broad_prior_probability"]) <= 0.0
                or float(arm_record["support_proposal_probability"]) <= 0.0
            ):
                raise UgiProductionCandidateGenerationError(
                    f"{arm} morphology identity or support changed"
                )
    return tuple(records)


def load_production_candidate_contract(
    repo: Path, config_path: Path
) -> ProductionCandidateContract:
    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="production generation config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiProductionCandidateGenerationError("unsupported production generation schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiProductionCandidateGenerationError("production generation scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiProductionCandidateGenerationError("production generation input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    design = _design(config["design"])
    schedule_result = _load_json(paths["schedule_result"], label="production schedule result")
    schedule = _load_json(paths["schedule"], label="production schedule")
    decision = schedule_result.get("decision", {})
    if (
        decision.get("schedule_frozen_before_generation") is not True
        or decision.get("applicability_proposal_promoted") is not True
        or decision.get("potency_tilting_promoted") is not False
        or decision.get("synthesis_tilting_promoted") is not False
        or schedule_result.get("artifacts", {}).get("schedule.json", {}).get("schedule_sha256")
        != schedule.get("schedule_sha256")
        or schedule.get("status") != "frozen_before_production_candidate_generation"
        or schedule.get("design", {}).get("draws_per_arm") != design.draws_per_arm
        or tuple(schedule.get("design", {}).get("arms", ())) != ARM_IDS
        or schedule.get("population", {}).get("qualified_programs") != 57190
        or schedule.get("population", {}).get("all_arms_positive_on_all_programs") is not True
    ):
        raise UgiProductionCandidateGenerationError("frozen production schedule changed")
    records = _validate_schedule_records(schedule.get("records"), draws=design.draws_per_arm)
    base = load_census_contract(repo, paths["base_census_config"])
    if base.design.sample_steps != SAMPLE_STEPS:
        raise UgiProductionCandidateGenerationError("selected-v3 generator schedule changed")
    return ProductionCandidateContract(
        repo=repo,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        inputs=paths,
        design=design,
        schedule=records,
        schedule_sha256=str(schedule["schedule_sha256"]),
    )


def production_candidate_plan(repo: Path, config_path: Path) -> dict[str, Any]:
    contract = load_production_candidate_contract(repo, config_path)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "planned_not_executed",
        "config_sha256": contract.config_sha256,
        "schedule_sha256": contract.schedule_sha256,
        "arms": list(ARM_IDS),
        "draws_per_arm": contract.design.draws_per_arm,
        "terminal_attempts": contract.design.terminal_attempts,
        "shards": contract.design.shard_count,
        "terminal_ledger_schema_version": TERMINAL_LEDGER_SCHEMA_VERSION,
        "terminal_ledger_required_fields": sorted(TERMINAL_LEDGER_REQUIRED_FIELDS),
        "scope": dict(EXPECTED_SCOPE),
    }


def _shard_records(
    contract: ProductionCandidateContract, shard_index: int
) -> tuple[Mapping[str, Any], ...]:
    if not 0 <= shard_index < contract.design.shard_count:
        raise UgiProductionCandidateGenerationError("shard index is outside the design")
    start = shard_index * contract.design.shard_draws
    return contract.schedule[start : start + contract.design.shard_draws]


def _validate_shard(
    contract: ProductionCandidateContract, output_dir: Path, *, shard_index: int
) -> dict[str, Any]:
    shard_dir = output_dir / "shards" / f"shard_{shard_index:04d}"
    receipt = _load_json(shard_dir / "receipt.json", label="production generation shard receipt")
    logical = {key: value for key, value in receipt.items() if key != "result_sha256"}
    expected_draws = _shard_records(contract, shard_index)
    expected_indices = [int(row["draw_index"]) for row in expected_draws]
    expected_counts = {
        "draws": len(expected_draws),
        "terminal_attempts": len(expected_draws) * len(ARM_IDS),
        "terminal_attempts_per_arm": {arm: len(expected_draws) for arm in ARM_IDS},
        "product_transition_calls": len(expected_draws) * len(ARM_IDS) * SAMPLE_STEPS,
    }
    if (
        receipt.get("result_sha256") != _sha256_payload(logical)
        or receipt.get("schema_version") != SHARD_SCHEMA_VERSION
        or receipt.get("status") != "complete_production_candidate_generation_shard"
        or receipt.get("config_sha256") != contract.config_sha256
        or receipt.get("schedule_sha256") != contract.schedule_sha256
        or receipt.get("shard_index") != shard_index
        or receipt.get("draw_indices") != expected_indices
        or receipt.get("arms") != list(ARM_IDS)
        or receipt.get("scope") != EXPECTED_SCOPE
        or receipt.get("counts") != expected_counts
    ):
        raise UgiProductionCandidateGenerationError("production shard contract changed")
    artifact = receipt.get("terminal_ledger")
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "path",
        "sha256",
        "logical_sha256",
        "rows",
        "schema_version",
    }:
        raise UgiProductionCandidateGenerationError("production shard artifact changed")
    ledger_path = shard_dir / str(artifact["path"])
    if (
        artifact.get("schema_version") != TERMINAL_LEDGER_SCHEMA_VERSION
        or not ledger_path.is_file()
        or ledger_path.is_symlink()
        or sha256_file(ledger_path) != artifact["sha256"]
        or int(artifact["rows"]) != len(expected_draws) * len(ARM_IDS)
    ):
        raise UgiProductionCandidateGenerationError("production shard ledger changed")
    return receipt


def execute_production_candidate_shard(
    contract: ProductionCandidateContract,
    output_dir: Path,
    *,
    shard_index: int,
    lane: Any,
    native_completer: NativeCompleter = complete_native_no_route_terminal,
) -> dict[str, Any]:
    """Execute or resume one atomic production draw shard."""

    output_dir = output_dir.resolve()
    final_dir = output_dir / "shards" / f"shard_{shard_index:04d}"
    if final_dir.exists():
        return {**_validate_shard(contract, output_dir, shard_index=shard_index), "resumed": True}
    draws = _shard_records(contract, shard_index)
    draw_indices = tuple(int(row["draw_index"]) for row in draws)
    particle_seeds = tuple(
        production_draw_seed(
            contract.design.particle_seed_base,
            purpose="production_morphology_particle",
            draw_index=draw_index,
        )
        for draw_index in draw_indices
    )
    terminal_seeds = tuple(
        production_draw_seed(
            contract.design.terminal_seed_base,
            purpose="production_morphology_terminal",
            draw_index=draw_index,
        )
        for draw_index in draw_indices
    )
    if (
        len(set(particle_seeds)) != len(draws)
        or len(set(terminal_seeds)) != len(draws)
        or set(particle_seeds) & set(terminal_seeds)
    ):
        raise UgiProductionCandidateGenerationError("production seed collision")

    rows = []
    adapter_identity: str | None = None
    for arm in ARM_IDS:
        arm_programs = tuple(
            canonical_morphology_program_bytes(dict(row["arms"][arm]["program"])) for row in draws
        )
        initialized = lane.initialize(
            arm_programs,
            seed=contract.design.particle_seed_base,
            particle_seeds=particle_seeds,
            device=contract.design.device,
        )
        state = initialized.state
        if not isinstance(state, SelectedGuidanceState):
            raise UgiProductionCandidateGenerationError(
                "selected-v3 initialization returned an invalid state"
            )
        if adapter_identity is None:
            adapter_identity = state.adapter_identity_sha256
        elif adapter_identity != state.adapter_identity_sha256:
            raise UgiProductionCandidateGenerationError(
                "selected-v3 adapter identity changed between arms"
            )
        advanced = lane.advance(state, target_step=contract.design.checkpoint)
        if advanced.product_transition_calls != len(draws) * contract.design.checkpoint:
            raise UgiProductionCandidateGenerationError("selected-v3 transition accounting changed")
        state = advanced.state
        for particle_index, (draw, particle_seed, terminal_seed) in enumerate(
            zip(draws, particle_seeds, terminal_seeds, strict=True)
        ):
            completion = native_completer(
                lane,
                state,
                particle_index=particle_index,
                seed=terminal_seed,
                checkpoint_index=contract.design.checkpoint,
                rollout_index=0,
            )
            arm_record = draw["arms"][arm]
            row = {
                "arm_id": arm,
                "draw_index": int(draw["draw_index"]),
                "common_uniform": float(draw["common_uniform"]),
                "support_index": int(arm_record["support_index"]),
                "program_sha256": str(arm_record["program_sha256"]),
                "program": dict(arm_record["program"]),
                "selected_probability": float(arm_record["selected_probability"]),
                "broad_prior_probability": float(arm_record["broad_prior_probability"]),
                "support_proposal_probability": float(arm_record["support_proposal_probability"]),
                "importance_ratio_broad_over_arm": float(
                    arm_record["importance_ratio_broad_over_arm"]
                ),
                "support_score": float(arm_record["support_score"]),
                "particle_seed": particle_seed,
                "terminal_seed": terminal_seed,
                "shard_index": shard_index,
                "particle_index": particle_index,
                **completion,
            }
            if not TERMINAL_LEDGER_REQUIRED_FIELDS.issubset(row):
                raise UgiProductionCandidateGenerationError(
                    "native terminal output lacks a required production field"
                )
            rows.append(row)
    arm_order = {arm: index for index, arm in enumerate(ARM_IDS)}
    rows.sort(key=lambda row: (int(row["draw_index"]), arm_order[str(row["arm_id"])]))

    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".shard_{shard_index:04d}.", dir=shard_root))
    try:
        ledger_path = work_dir / "terminals.jsonl.gz"
        _atomic_write(ledger_path, _jsonl_gzip_bytes(rows))
        content = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "status": "complete_production_candidate_generation_shard",
            "config_sha256": contract.config_sha256,
            "schedule_sha256": contract.schedule_sha256,
            "shard_index": shard_index,
            "draw_indices": list(draw_indices),
            "arms": list(ARM_IDS),
            "adapter_identity_sha256": adapter_identity,
            "particle_seed_manifest_sha256": _sha256_payload(particle_seeds),
            "terminal_seed_manifest_sha256": _sha256_payload(terminal_seeds),
            "counts": {
                "draws": len(draws),
                "terminal_attempts": len(rows),
                "terminal_attempts_per_arm": {arm: len(draws) for arm in ARM_IDS},
                "product_transition_calls": len(rows) * SAMPLE_STEPS,
            },
            "terminal_ledger": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "logical_sha256": _sha256_payload(rows),
                "rows": len(rows),
                "schema_version": TERMINAL_LEDGER_SCHEMA_VERSION,
            },
            "scope": dict(EXPECTED_SCOPE),
        }
        receipt = {**content, "result_sha256": _sha256_payload(content)}
        _atomic_json(work_dir / "receipt.json", receipt)
        os.replace(work_dir, final_dir)
        return {**receipt, "resumed": False}
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def run_all_production_candidate_shards(
    repo: Path, config_path: Path, output_dir: Path
) -> list[dict[str, Any]]:
    contract = load_production_candidate_contract(repo, config_path)
    lane = build_selected_model_restartable_guidance_lane_v3(contract.repo)
    return [
        execute_production_candidate_shard(contract, output_dir, shard_index=shard_index, lane=lane)
        for shard_index in range(contract.design.shard_count)
    ]


def run_one_production_candidate_shard(
    repo: Path, config_path: Path, output_dir: Path, *, shard_index: int
) -> dict[str, Any]:
    contract = load_production_candidate_contract(repo, config_path)
    lane = build_selected_model_restartable_guidance_lane_v3(contract.repo)
    return execute_production_candidate_shard(
        contract, output_dir, shard_index=shard_index, lane=lane
    )


def aggregate_production_candidate_generation(
    repo: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    contract = load_production_candidate_contract(repo, config_path)
    output_dir = output_dir.resolve()
    receipts = []
    rows: list[dict[str, Any]] = []
    for shard_index in range(contract.design.shard_count):
        receipt = _validate_shard(contract, output_dir, shard_index=shard_index)
        receipts.append(receipt)
        ledger_path = (
            output_dir
            / "shards"
            / f"shard_{shard_index:04d}"
            / str(receipt["terminal_ledger"]["path"])
        )
        with gzip.open(ledger_path, "rt") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if len(rows) != contract.design.terminal_attempts:
        raise UgiProductionCandidateGenerationError("production terminal aggregate is incomplete")
    arm_order = {arm: index for index, arm in enumerate(ARM_IDS)}
    rows.sort(key=lambda row: (int(row["draw_index"]), arm_order[str(row["arm_id"])]))
    by_draw: dict[int, list[dict[str, Any]]] = defaultdict(list)
    arm_counts = defaultdict(int)
    valid_counts = defaultdict(int)
    for row in rows:
        if not TERMINAL_LEDGER_REQUIRED_FIELDS.issubset(row):
            raise UgiProductionCandidateGenerationError("aggregate row lacks production fields")
        by_draw[int(row["draw_index"])].append(row)
        arm = str(row["arm_id"])
        arm_counts[arm] += 1
        valid_counts[arm] += int(bool(row["native_terminal"].get("terminal_valid")))
    if set(by_draw) != set(range(contract.design.draws_per_arm)):
        raise UgiProductionCandidateGenerationError("aggregate draw support changed")
    for draw_index, grouped in by_draw.items():
        if [row["arm_id"] for row in grouped] != list(ARM_IDS):
            raise UgiProductionCandidateGenerationError(f"draw {draw_index} lacks one arm")
        if (
            len({float(row["common_uniform"]) for row in grouped}) != 1
            or len({int(row["particle_seed"]) for row in grouped}) != 1
            or len({int(row["terminal_seed"]) for row in grouped}) != 1
        ):
            raise UgiProductionCandidateGenerationError(
                f"draw {draw_index} lost common random numbers"
            )
    if dict(arm_counts) != {arm: contract.design.draws_per_arm for arm in ARM_IDS}:
        raise UgiProductionCandidateGenerationError("production arm budgets are not matched")
    adapter_identities = {receipt["adapter_identity_sha256"] for receipt in receipts}
    if len(adapter_identities) != 1:
        raise UgiProductionCandidateGenerationError(
            "selected-v3 adapter changed across production shards"
        )

    ledger_path = output_dir / "terminal_ledger.jsonl.gz"
    _atomic_write(ledger_path, _jsonl_gzip_bytes(rows))
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_matched_production_candidate_generation",
        "config": {
            "path": str(contract.config_path.relative_to(contract.repo)),
            "sha256": contract.config_sha256,
        },
        "schedule_sha256": contract.schedule_sha256,
        "adapter_identity_sha256": next(iter(adapter_identities)),
        "counts": {
            "shards": len(receipts),
            "draws_per_arm": contract.design.draws_per_arm,
            "terminal_attempts": len(rows),
            "terminal_attempts_per_arm": dict(arm_counts),
            "valid_exact_l1_per_arm": dict(valid_counts),
            "product_transition_calls": len(rows) * SAMPLE_STEPS,
        },
        "common_random_numbers": {
            "common_uniforms_across_arms": True,
            "common_particle_seeds_across_arms": True,
            "common_terminal_seeds_across_arms": True,
        },
        "artifacts": {
            "terminal_ledger.jsonl.gz": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "logical_sha256": _sha256_payload(rows),
                "rows": len(rows),
                "schema_version": TERMINAL_LEDGER_SCHEMA_VERSION,
                "required_fields": sorted(TERMINAL_LEDGER_REQUIRED_FIELDS),
            },
            "shard_receipts_sha256": _sha256_payload(
                [receipt["result_sha256"] for receipt in receipts]
            ),
        },
        "scope": dict(EXPECTED_SCOPE),
        "next_gate": "frozen_terminal_applicability_ranking_and_route_funnel",
        "nonclaims": [
            "No terminal was scored, ranked, filtered, routed, repaired or retried during generation.",
            "The support-enriched arm changes only the frozen morphology allocation.",
            "Neither potency nor synthesis trajectory tilting is present.",
            "This result does not select or lock prospective candidates.",
        ],
    }
    result = {**content, "result_sha256": _sha256_payload(content)}
    _atomic_json(output_dir / "result.json", result)
    return result


__all__ = [
    "ProductionCandidateContract",
    "ProductionCandidateDesign",
    "TERMINAL_LEDGER_REQUIRED_FIELDS",
    "UgiProductionCandidateGenerationError",
    "aggregate_production_candidate_generation",
    "execute_production_candidate_shard",
    "load_production_candidate_contract",
    "production_candidate_plan",
    "production_draw_seed",
    "run_all_production_candidate_shards",
    "run_one_production_candidate_shard",
]
