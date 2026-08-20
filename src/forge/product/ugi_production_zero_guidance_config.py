"""Frozen identity and schedule loader for a production zero-guidance rehearsal.

This module authenticates the selected generator/closure artifacts and the
restartable-equivalence receipt, then builds a deterministic matched schedule.
It deliberately does not load a model, complete a terminal, execute a route
source, compute a synthesis scalar, invoke biology or select candidates.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import read_json_object
from forge.product.ugi_matched_budget_orchestration import (
    MatchedBudgetLimits,
    MatchedScheduleEntry,
    RouteComputeUsage,
)
from forge.product.ugi_morphology_program import UgiMorphologyProgram
from forge.product.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.product.ugi_zero_guidance_rehearsal import RestartableGeneratorClosureIdentity
from forge.route.planner_cache import PlannerCacheContext, PlannerCacheError
from forge.route.planner_cache_snapshot import (
    planner_cache_context_sha256,
    require_current_l3_context,
)
from forge.route.terminal_assessment import (
    required_three_role_route_reservation,
)

PRODUCTION_ZERO_GUIDANCE_CONFIG_SCHEMA_VERSION = (
    "phase1_ugi_production_zero_guidance_rehearsal_config.v1"
)
PRODUCTION_ZERO_GUIDANCE_PLAN_SCHEMA_VERSION = (
    "forge.ugi_production_zero_guidance_rehearsal_plan.v1"
)
_EXPECTED_SCOPE = "production_zero_guidance_rehearsal_schedule_and_identity_only"
_EXPECTED_SCOPE_GUARDS = {
    "production_generator_executed": False,
    "production_route_source_executed": False,
    "nonzero_synthesis_guidance": False,
    "synthesis_scalar_defined": False,
    "biological_guidance": False,
    "candidate_selection": False,
    "private_holdout_accessed": False,
}
_EXPECTED_ARTIFACT_NAMES = {
    "closure_checkpoint",
    "generator_checkpoint",
    "production_generator_manifest",
    "program_draw",
    "restartable_equivalence_config",
    "restartable_equivalence_result",
}
_EXPECTED_L3_INTERVAL_SEMANTICS = "accessed_inclusive_expires_exclusive"
_EXPECTED_TERMINAL_DECODER_ID = "ugi_joint_terminal_completion:argmax:v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UgiProductionZeroGuidanceConfigError(RuntimeError):
    """Raised when the frozen rehearsal plan cannot be authenticated exactly."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise UgiProductionZeroGuidanceConfigError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=UgiProductionZeroGuidanceConfigError, label=label)


def _require_exact_fields(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise UgiProductionZeroGuidanceConfigError(f"{label} has an unsupported field set")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise UgiProductionZeroGuidanceConfigError(
            f"{label} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _require_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise UgiProductionZeroGuidanceConfigError(
            f"{label} must be an integer no smaller than {minimum}"
        )
    return value


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise UgiProductionZeroGuidanceConfigError(f"{label} must be a nonempty string")
    return value


def _route_usage_to_dict(value: RouteComputeUsage) -> dict[str, int]:
    return {
        "logical_planner_calls": value.logical_planner_calls,
        "physical_planner_calls": value.physical_planner_calls,
        "logical_verifier_calls": value.logical_verifier_calls,
        "physical_verifier_calls": value.physical_verifier_calls,
    }


def _program_to_dict(value: UgiMorphologyProgram) -> dict[str, list[int]]:
    return {
        "node_counts": list(value.node_counts),
        "junction_budgets": list(value.junction_budgets),
        "cycle_ranks": list(value.cycle_ranks),
        "attachment_counts": list(value.attachment_counts),
    }


@dataclass(frozen=True)
class AuthenticatedProductionArtifact:
    """One repository-owned artifact that matched its frozen digest."""

    name: str
    relative_path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ProductionZeroGuidanceRehearsalPlan:
    """Nonexecuting, typed identity and matched schedule for one rehearsal."""

    config_path: str
    config_sha256: str
    identity: RestartableGeneratorClosureIdentity
    artifacts: tuple[AuthenticatedProductionArtifact, ...]
    programs: tuple[UgiMorphologyProgram, ...]
    schedule: tuple[MatchedScheduleEntry, ...]
    budget_limits: MatchedBudgetLimits
    per_unit_route_reservation: RouteComputeUsage
    planner_context_sha256: str
    cumulative_source_inputs_sha256: str
    assessment_as_of_utc: str
    l3_accessed_at_utc: str
    l3_expires_at_utc: str
    l3_interval_semantics: str
    seed: int
    sample_steps: int
    terminal_checkpoint_index: int

    @property
    def schedule_sha256(self) -> str:
        return _sha256_payload(
            [
                {
                    "unit_id": entry.unit_id,
                    "morphology_program_sha256": entry.morphology_program_sha256,
                    "program_index": entry.program_index,
                    "particle_index": entry.particle_index,
                    "checkpoint_index": entry.checkpoint_index,
                    "generator_checkpoint_sha256": entry.generator_checkpoint_sha256,
                    "closure_checkpoint_sha256": entry.closure_checkpoint_sha256,
                    "rollout_index": entry.rollout_index,
                    "productive_generation_calls": entry.productive_generation_calls,
                    "route_reservation": _route_usage_to_dict(entry.route_reservation),
                }
                for entry in self.schedule
            ]
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCTION_ZERO_GUIDANCE_PLAN_SCHEMA_VERSION,
            "scope": _EXPECTED_SCOPE,
            "config": {"path": self.config_path, "sha256": self.config_sha256},
            "identity": self.identity.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "planner_context_sha256": self.planner_context_sha256,
            "cumulative_source_inputs_sha256": self.cumulative_source_inputs_sha256,
            "assessment": {
                "assessment_as_of_utc": self.assessment_as_of_utc,
                "l3_accessed_at_utc": self.l3_accessed_at_utc,
                "l3_expires_at_utc": self.l3_expires_at_utc,
                "l3_interval_semantics": self.l3_interval_semantics,
            },
            "execution": {
                "guidance_strength": 0,
                "program_count": len(self.programs),
                "sample_steps": self.sample_steps,
                "seed": self.seed,
                "terminal_checkpoint_index": self.terminal_checkpoint_index,
                "production_generator_executed": False,
                "production_route_source_executed": False,
            },
            "programs": [_program_to_dict(program) for program in self.programs],
            "schedule_sha256": self.schedule_sha256,
            "per_unit_route_reservation": _route_usage_to_dict(self.per_unit_route_reservation),
            "budget_limits": {
                "productive_generation_calls": (self.budget_limits.productive_generation_calls),
                "terminal_completions": self.budget_limits.terminal_completions,
                "final_candidates": self.budget_limits.final_candidates,
                "route": _route_usage_to_dict(self.budget_limits.route),
            },
            "scope_guards": dict(_EXPECTED_SCOPE_GUARDS),
        }

    @property
    def plan_sha256(self) -> str:
        return _sha256_payload(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "plan_sha256": self.plan_sha256}


def _authenticated_artifacts(
    repo_root: Path,
    value: Any,
) -> tuple[tuple[AuthenticatedProductionArtifact, ...], dict[str, Path]]:
    artifacts = _require_exact_fields(value, _EXPECTED_ARTIFACT_NAMES, label="artifacts")
    root = repo_root.resolve()
    records = []
    paths: dict[str, Path] = {}
    for name in sorted(_EXPECTED_ARTIFACT_NAMES):
        specification = _require_exact_fields(
            artifacts[name], {"path", "sha256"}, label=f"artifact {name}"
        )
        relative = _require_nonempty(specification["path"], label=f"artifact {name} path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise UgiProductionZeroGuidanceConfigError(
                f"artifact {name} path must be repository-relative"
            )
        path = root / relative_path
        if path.is_symlink():
            raise UgiProductionZeroGuidanceConfigError(f"artifact {name} cannot be a symbolic link")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise UgiProductionZeroGuidanceConfigError(
                f"artifact {name} is missing or leaves the repository"
            ) from error
        if not resolved.is_file():
            raise UgiProductionZeroGuidanceConfigError(f"artifact {name} is not a file")
        expected_sha256 = _require_sha256(specification["sha256"], label=f"artifact {name} sha256")
        observed_sha256 = _sha256_bytes(resolved.read_bytes())
        if observed_sha256 != expected_sha256:
            raise UgiProductionZeroGuidanceConfigError(f"artifact {name} SHA-256 mismatch")
        records.append(AuthenticatedProductionArtifact(name, relative, observed_sha256))
        paths[name] = resolved
    return tuple(records), paths


def _integer_triplet(value: Any, *, label: str, minimum: int) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < minimum for item in value
        )
    ):
        raise UgiProductionZeroGuidanceConfigError(
            f"{label} must contain three integers no smaller than {minimum}"
        )
    return tuple(value)  # type: ignore[return-value]


def _load_programs(path: Path, *, program_count: int) -> tuple[UgiMorphologyProgram, ...]:
    value = _read_json(path, label="program draw")
    if value.get("schema_version") != "phase1_ugi_program_probe.v1":
        raise UgiProductionZeroGuidanceConfigError("program draw schema changed")
    if value.get("fold") != "training_program_prior":
        raise UgiProductionZeroGuidanceConfigError(
            "production rehearsal programs must come from the training-program prior"
        )
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) < program_count:
        raise UgiProductionZeroGuidanceConfigError(
            "program draw has fewer samples than the frozen rehearsal count"
        )
    programs = []
    for index, row in enumerate(samples[:program_count]):
        if not isinstance(row, dict) or "program" not in row:
            raise UgiProductionZeroGuidanceConfigError(f"program draw row {index} is malformed")
        program = _require_exact_fields(
            row["program"],
            {"node_counts", "junction_budgets", "cycle_ranks", "attachment_counts"},
            label=f"program draw row {index} program",
        )
        programs.append(
            UgiMorphologyProgram(
                node_counts=_integer_triplet(
                    program["node_counts"], label=f"program {index} node_counts", minimum=1
                ),
                junction_budgets=_integer_triplet(
                    program["junction_budgets"],
                    label=f"program {index} junction_budgets",
                    minimum=0,
                ),
                cycle_ranks=_integer_triplet(
                    program["cycle_ranks"], label=f"program {index} cycle_ranks", minimum=0
                ),
                attachment_counts=_integer_triplet(
                    program["attachment_counts"],
                    label=f"program {index} attachment_counts",
                    minimum=1,
                ),
            )
        )
    return tuple(programs)


def _require_manifest_identity(
    manifest_path: Path,
    *,
    generator_sha256: str,
    closure_sha256: str,
    program_draw_sha256: str,
) -> None:
    value = _read_json(manifest_path, label="production generator manifest")
    if (
        value.get("schema_version") != "phase1_ugi_product_l1_production_generator.v1"
        or value.get("status") != "frozen"
    ):
        raise UgiProductionZeroGuidanceConfigError(
            "production generator manifest is not the frozen v1 manifest"
        )
    try:
        manifest_generator = value["model"]["checkpoint"]["sha256"]
        manifest_closure = value["sampling_dependencies"]["closure_checkpoint"]["sha256"]
        manifest_program_draw = value["selection"]["fresh_program_draw"]["sha256"]
        checkpoint_step = value["identity"]["checkpoint_step"]
    except (KeyError, TypeError) as error:
        raise UgiProductionZeroGuidanceConfigError(
            "production generator manifest identity is malformed"
        ) from error
    if (
        manifest_generator != generator_sha256
        or manifest_closure != closure_sha256
        or manifest_program_draw != program_draw_sha256
    ):
        raise UgiProductionZeroGuidanceConfigError(
            "production manifest artifact identity differs from the rehearsal config"
        )
    if checkpoint_step != 1000:
        raise UgiProductionZeroGuidanceConfigError(
            "production manifest selected checkpoint step changed"
        )


def _require_restartable_equivalence(
    config_path: Path,
    result_path: Path,
    *,
    generator_sha256: str,
    program_draw_sha256: str,
    program_count: int,
    sample_steps: int,
    seed: int,
    generator_implementation_sha256: str,
) -> None:
    config = _read_json(config_path, label="restartable-equivalence config")
    expected_config_fields = {
        "schema_version",
        "joint_checkpoint",
        "program_draw",
        "program_count",
        "generator_implementation_sha256",
        "sample_steps",
        "seed",
        "batch_sizes",
        "device",
        "output_dir",
        "scope",
        "production_synthesis_guidance",
        "biological_guidance",
    }
    _require_exact_fields(config, expected_config_fields, label="restartable-equivalence config")
    if config.get("schema_version") != "phase1_ugi_restartable_sampler_equivalence_config.v1":
        raise UgiProductionZeroGuidanceConfigError("restartable-equivalence config schema changed")
    joint_specification = _require_exact_fields(
        config.get("joint_checkpoint"),
        {"path", "sha256"},
        label="restartable-equivalence joint checkpoint",
    )
    draw_specification = _require_exact_fields(
        config.get("program_draw"),
        {"path", "sha256"},
        label="restartable-equivalence program draw",
    )
    if (
        config.get("program_count") != program_count
        or config.get("sample_steps") != sample_steps
        or config.get("seed") != seed
        or config.get("generator_implementation_sha256") != generator_implementation_sha256
        or config.get("production_synthesis_guidance") is not False
        or config.get("biological_guidance") is not False
        or config.get("scope") != "zero_guidance_equivalence_only"
    ):
        raise UgiProductionZeroGuidanceConfigError(
            "restartable-equivalence config differs from the frozen zero-guidance schedule"
        )
    if (
        joint_specification.get("sha256") != generator_sha256
        or draw_specification.get("sha256") != program_draw_sha256
    ):
        raise UgiProductionZeroGuidanceConfigError(
            "restartable-equivalence config artifact identity changed"
        )

    result = _read_json(result_path, label="restartable-equivalence result")
    if (
        result.get("schema_version") != "phase1_ugi_restartable_sampler_equivalence.v1"
        or result.get("status") != "complete"
        or result.get("decision") != "restartable_zero_guidance_schedule_is_bitwise_equivalent"
        or result.get("scope") != "zero_guidance_equivalence_only"
        or result.get("production_synthesis_guidance") is not False
        or result.get("biological_guidance") is not False
        or result.get("program_count") != program_count
        or result.get("sample_steps") != sample_steps
        or result.get("seed") != seed
    ):
        raise UgiProductionZeroGuidanceConfigError(
            "restartable-equivalence result does not admit this frozen schedule"
        )
    comparisons = result.get("comparisons")
    valid_comparisons = isinstance(comparisons, list) and bool(comparisons)
    if valid_comparisons:
        for row in comparisons:
            metadata = row.get("metadata") if isinstance(row, dict) else None
            if (
                not isinstance(row, dict)
                or row.get("bitwise_equal") is not True
                or row.get("terminals") != program_count
                or not isinstance(metadata, dict)
                or metadata.get("sample_steps") != sample_steps
            ):
                valid_comparisons = False
                break
    if not valid_comparisons:
        raise UgiProductionZeroGuidanceConfigError(
            "restartable-equivalence comparisons are incomplete or nonidentical"
        )
    inputs = result.get("inputs")
    input_config = inputs.get("config") if isinstance(inputs, dict) else None
    input_checkpoint = inputs.get("joint_checkpoint") if isinstance(inputs, dict) else None
    input_draw = inputs.get("program_draw") if isinstance(inputs, dict) else None
    input_implementation = (
        inputs.get("generator_implementation") if isinstance(inputs, dict) else None
    )
    if (
        not isinstance(inputs, dict)
        or not isinstance(input_config, dict)
        or not isinstance(input_checkpoint, dict)
        or not isinstance(input_draw, dict)
        or not isinstance(input_implementation, dict)
        or input_config.get("sha256") != _sha256_bytes(config_path.read_bytes())
        or input_checkpoint.get("sha256") != generator_sha256
        or input_draw.get("sha256") != program_draw_sha256
        or input_implementation.get("implementation_sha256") != generator_implementation_sha256
    ):
        raise UgiProductionZeroGuidanceConfigError(
            "restartable-equivalence result input pins changed"
        )


def _multiply_route_usage(value: RouteComputeUsage, count: int) -> RouteComputeUsage:
    output = RouteComputeUsage()
    for _ in range(count):
        output = output.plus(value)
    return output


def load_production_zero_guidance_rehearsal_plan(
    repo_root: Path,
    config_path: Path,
    *,
    planner_context: PlannerCacheContext,
) -> ProductionZeroGuidanceRehearsalPlan:
    """Authenticate frozen inputs and return a deterministic, nonexecuting plan."""

    if not isinstance(repo_root, Path) or not isinstance(config_path, Path):
        raise UgiProductionZeroGuidanceConfigError("repo_root and config_path must be Paths")
    if not isinstance(planner_context, PlannerCacheContext):
        raise UgiProductionZeroGuidanceConfigError(
            "planner_context must be a typed PlannerCacheContext"
        )
    config = _read_json(config_path, label="production zero-guidance config")
    _require_exact_fields(
        config,
        {"schema_version", "scope", "execution", "assessment", "artifacts", "scope_guards"},
        label="production zero-guidance config",
    )
    if config.get("schema_version") != PRODUCTION_ZERO_GUIDANCE_CONFIG_SCHEMA_VERSION:
        raise UgiProductionZeroGuidanceConfigError(
            "unsupported production zero-guidance config schema"
        )
    if config.get("scope") != _EXPECTED_SCOPE:
        raise UgiProductionZeroGuidanceConfigError("production zero-guidance scope changed")
    if config.get("scope_guards") != _EXPECTED_SCOPE_GUARDS:
        raise UgiProductionZeroGuidanceConfigError("production zero-guidance scope guards changed")

    execution = _require_exact_fields(
        config["execution"],
        {
            "guidance_strength",
            "program_count",
            "sample_steps",
            "seed",
            "terminal_checkpoint_index",
            "generator_implementation_sha256",
            "terminal_decoder_id",
        },
        label="execution",
    )
    if isinstance(execution["guidance_strength"], bool) or execution["guidance_strength"] != 0:
        raise UgiProductionZeroGuidanceConfigError("only guidance_strength=0 is permitted")
    program_count = _require_integer(execution["program_count"], label="program_count", minimum=1)
    sample_steps = _require_integer(execution["sample_steps"], label="sample_steps", minimum=1)
    seed = _require_integer(execution["seed"], label="seed")
    terminal_checkpoint_index = _require_integer(
        execution["terminal_checkpoint_index"], label="terminal_checkpoint_index", minimum=1
    )
    if terminal_checkpoint_index != sample_steps:
        raise UgiProductionZeroGuidanceConfigError(
            "terminal checkpoint must be the terminal of the frozen sampler schedule"
        )
    terminal_decoder_id = _require_nonempty(
        execution["terminal_decoder_id"], label="terminal_decoder_id"
    )
    generator_implementation_sha256 = _require_sha256(
        execution["generator_implementation_sha256"],
        label="generator_implementation_sha256",
    )
    if terminal_decoder_id != _EXPECTED_TERMINAL_DECODER_ID:
        raise UgiProductionZeroGuidanceConfigError(
            "terminal decoder identity differs from the selected generator lane"
        )

    assessment = _require_exact_fields(
        config["assessment"],
        {"assessment_as_of_utc", "cumulative_source_inputs_sha256", "l3_window"},
        label="assessment",
    )
    assessment_as_of_utc = _require_nonempty(
        assessment["assessment_as_of_utc"], label="assessment_as_of_utc"
    )
    source_inputs_sha256 = _require_sha256(
        assessment["cumulative_source_inputs_sha256"],
        label="cumulative_source_inputs_sha256",
    )
    l3_window = _require_exact_fields(
        assessment["l3_window"],
        {"accessed_utc", "expires_utc", "interval_semantics"},
        label="l3_window",
    )
    accessed_at = _require_nonempty(l3_window["accessed_utc"], label="l3 accessed_utc")
    expires_at = _require_nonempty(l3_window["expires_utc"], label="l3 expires_utc")
    if l3_window["interval_semantics"] != _EXPECTED_L3_INTERVAL_SEMANTICS:
        raise UgiProductionZeroGuidanceConfigError("L3 interval semantics changed")
    if (
        planner_context.l3_snapshot_sha256 != source_inputs_sha256
        or planner_context.l3_accessed_at_utc != accessed_at
        or planner_context.l3_expires_at_utc != expires_at
    ):
        raise UgiProductionZeroGuidanceConfigError(
            "planner context does not match the frozen cumulative-source L3 identity"
        )
    if planner_context.budget_limits.maximum_logical_planner_calls != 1:
        raise UgiProductionZeroGuidanceConfigError(
            "each isolated component planner must own exactly one logical planner call"
        )
    try:
        require_current_l3_context(planner_context, assessment_at_utc=assessment_as_of_utc)
    except PlannerCacheError as error:
        raise UgiProductionZeroGuidanceConfigError(
            "planner context is not current at the frozen assessment time"
        ) from error

    artifacts, artifact_paths = _authenticated_artifacts(repo_root, config["artifacts"])
    artifact_by_name = {record.name: record for record in artifacts}
    generator_sha256 = artifact_by_name["generator_checkpoint"].sha256
    closure_sha256 = artifact_by_name["closure_checkpoint"].sha256
    program_draw_sha256 = artifact_by_name["program_draw"].sha256
    _require_manifest_identity(
        artifact_paths["production_generator_manifest"],
        generator_sha256=generator_sha256,
        closure_sha256=closure_sha256,
        program_draw_sha256=program_draw_sha256,
    )
    _require_restartable_equivalence(
        artifact_paths["restartable_equivalence_config"],
        artifact_paths["restartable_equivalence_result"],
        generator_sha256=generator_sha256,
        program_draw_sha256=program_draw_sha256,
        program_count=program_count,
        sample_steps=sample_steps,
        seed=seed,
        generator_implementation_sha256=generator_implementation_sha256,
    )
    programs = _load_programs(artifact_paths["program_draw"], program_count=program_count)

    identity = RestartableGeneratorClosureIdentity(
        generator_checkpoint_sha256=generator_sha256,
        closure_checkpoint_sha256=closure_sha256,
        production_generator_manifest_sha256=artifact_by_name[
            "production_generator_manifest"
        ].sha256,
        restartable_equivalence_receipt_sha256=artifact_by_name[
            "restartable_equivalence_result"
        ].sha256,
        generator_implementation_sha256=generator_implementation_sha256,
        terminal_decoder_id=terminal_decoder_id,
    )
    reservation = required_three_role_route_reservation(planner_context.budget_limits)
    schedule = tuple(
        MatchedScheduleEntry(
            unit_id=f"production-zero-guidance-{index:04d}",
            morphology_program=canonical_morphology_program_bytes(program),
            program_index=index,
            particle_index=index,
            checkpoint_index=terminal_checkpoint_index,
            generator_checkpoint_sha256=generator_sha256,
            closure_checkpoint_sha256=closure_sha256,
            rollout_index=0,
            productive_generation_calls=1,
            route_reservation=reservation,
        )
        for index, program in enumerate(programs)
    )
    budget_limits = MatchedBudgetLimits(
        productive_generation_calls=program_count,
        terminal_completions=program_count,
        final_candidates=program_count,
        route=_multiply_route_usage(reservation, program_count),
    )
    try:
        relative_config_path = config_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        relative_config_path = str(config_path.resolve())
    return ProductionZeroGuidanceRehearsalPlan(
        config_path=relative_config_path,
        config_sha256=_sha256_bytes(config_path.read_bytes()),
        identity=identity,
        artifacts=artifacts,
        programs=programs,
        schedule=schedule,
        budget_limits=budget_limits,
        per_unit_route_reservation=reservation,
        planner_context_sha256=planner_cache_context_sha256(planner_context),
        cumulative_source_inputs_sha256=source_inputs_sha256,
        assessment_as_of_utc=assessment_as_of_utc,
        l3_accessed_at_utc=accessed_at,
        l3_expires_at_utc=expires_at,
        l3_interval_semantics=_EXPECTED_L3_INTERVAL_SEMANTICS,
        seed=seed,
        sample_steps=sample_steps,
        terminal_checkpoint_index=terminal_checkpoint_index,
    )


__all__ = [
    "AuthenticatedProductionArtifact",
    "PRODUCTION_ZERO_GUIDANCE_CONFIG_SCHEMA_VERSION",
    "PRODUCTION_ZERO_GUIDANCE_PLAN_SCHEMA_VERSION",
    "ProductionZeroGuidanceRehearsalPlan",
    "UgiProductionZeroGuidanceConfigError",
    "load_production_zero_guidance_rehearsal_plan",
]
