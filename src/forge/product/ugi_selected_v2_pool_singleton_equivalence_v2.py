"""Production-runtime pooled/singleton equivalence for selected Ugi v2.

This additive audit supersedes, without modifying, the first pooled/singleton
receipt.  It binds the qualified numerical runtime, the prior selected-v2
equivalence chain, initialization seed manifests and source-state nonmutation
for both checkpoint and productive completions.  It executes no routing,
guidance, biology, selection or holdout access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_nonzero_guidance_runner import (
    GuidanceTerminalCompletionReceipt,
    ParticleGroupDesign,
    load_grouped_smc_schedule_qualification,
)
from forge.product.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.product.ugi_selected_generator_implementation import (
    build_selected_generator_implementation_qualification,
)
from forge.product.ugi_selected_guidance_adapter import (
    _particle_seed_manifest_sha256,
)
from forge.product.ugi_selected_guidance_adapter_v2 import (
    build_selected_model_restartable_guidance_lane_v2,
)
from forge.product.ugi_selected_restartable_generator import SAMPLE_STEPS
from forge.product.ugi_selected_restartable_generator_v2 import (
    GENERATOR_CHECKPOINT_SHA256,
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    PRODUCTION_GENERATOR_MANIFEST_SHA256,
    TERMINAL_DECODER_ID,
)
from forge.product.ugi_synthesis_guidance import keyed_random_seed

CONFIG_SCHEMA_VERSION = "phase1_ugi_selected_v2_pool_singleton_equivalence_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi_selected_v2_pool_singleton_equivalence.v2"
EXPECTED_IMPLEMENTATION_SHA256 = "52df6be6878da34e1205a307332173763c801a82026573afd7857e876fe2b193"
EXPECTED_RUNTIME_VERSIONS = (
    ("numpy", "2.5.1"),
    ("python", "3.14.2"),
    ("rdkit", "2026.03.4"),
    ("torch", "2.13.0"),
)
EXPECTED_BASE_ADAPTER_IDENTITY_SHA256 = (
    "ea06a5b6300657757ddb70364ecc4944424d4fb9e69b7cdb6aed9f07f83d574a"
)
EXPECTED_ASSIGNMENT_SHA256 = "c0b30c929cca59e08e9746050856761d2304934f883c06f1fb3379376c5daa27"
EXPECTED_EXPANDED_PROGRAM_MANIFEST_SHA256 = (
    "9affa5ff2598ffc99376800b78269f8cf714670e06be30901c6e1364ab1716d4"
)
EXPECTED_PARTICLE_SEED_MANIFEST_SHA256 = (
    "09fbd24a9c035a049055d1f3df299a4a67c2f1bb1669da8d6ec8e6ae9a97dfa1"
)
EXPECTED_INPUT_KEYS = frozenset(
    {
        "audit_source",
        "audit_runner",
        "audit_tests",
        "grouped_schedule",
        "production_generator_manifest",
        "selected_restartable_generator_v2",
        "selected_guidance_adapter_v2",
        "prior_selected_v2_equivalence",
        "qualified_runtime_receipt",
    }
)


class UgiSelectedV2PoolSingletonEquivalenceV2Error(RuntimeError):
    """Raised when the production-runtime equivalence contract is violated."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(f"{label} must be a JSON object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            f"{label} path escapes repository"
        ) from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(f"{label} hash changed")
    return path


def _require_self_hash(value: dict[str, Any], *, label: str) -> None:
    claimed = value.get("result_sha256")
    content = {key: item for key, item in value.items() if key != "result_sha256"}
    if not isinstance(claimed, str) or _sha256_payload(content) != claimed:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(f"{label} logical result hash changed")


def _validate_prior_receipt(path: Path) -> dict[str, Any]:
    prior = _load(path, label="prior selected-v2 equivalence receipt")
    _require_self_hash(prior, label="prior selected-v2 equivalence receipt")
    if (
        prior.get("schema_version") != "phase1_ugi_selected_v2_full_equivalence.v1"
        or prior.get("status") != "selected_v2_grouped_restartable_and_direct_callback_equivalent"
        or prior.get("execution", {}).get("all_equal") is not True
        or prior.get("execution", {}).get("particles") != 64
        or prior.get("execution", {}).get("checkpoint_comparisons") != 192
        or prior.get("execution", {}).get("productive_direct_callback_comparisons") != 64
        or prior.get("scope")
        != {
            "guidance": False,
            "routing": False,
            "biology": False,
            "candidate_selection": False,
            "holdout_access": False,
        }
    ):
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "prior selected-v2 equivalence receipt is not qualified"
        )
    selected = prior.get("selected_generator", {})
    if selected != {
        "adapter_identity_sha256": EXPECTED_BASE_ADAPTER_IDENTITY_SHA256,
        "checkpoint_sha256": GENERATOR_CHECKPOINT_SHA256,
        "maximum_adjacent_branch_runs": list(MAXIMUM_ADJACENT_BRANCH_RUNS),
        "terminal_decoder_id": TERMINAL_DECODER_ID,
    }:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "prior selected-v2 identity differs from the production lane"
        )
    artifact = prior.get("comparison_rows_artifact")
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "prior comparison-row artifact is missing"
        )
    return prior


def _validate_runtime_receipt(path: Path, current: dict[str, Any]) -> None:
    receipt = _load(path, label="qualified runtime receipt")
    implementation = receipt.get("inputs", {}).get("generator_implementation")
    if (
        receipt.get("decision") != "restartable_zero_guidance_schedule_is_bitwise_equivalent"
        or implementation != current
        or implementation.get("implementation_sha256") != EXPECTED_IMPLEMENTATION_SHA256
        or tuple(tuple(item) for item in implementation.get("runtime_versions", []))
        != EXPECTED_RUNTIME_VERSIONS
    ):
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "current runtime is not the qualified production runtime"
        )


def _validate_config(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any], dict[str, Any]]:
    config = _load(config_path, label="pool-singleton equivalence v2 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "unsupported pool-singleton v2 config schema"
        )
    if config.get("scope") != {
        "guidance": False,
        "routing": False,
        "biology": False,
        "candidate_selection": False,
        "holdout_access": False,
        "nonzero_guidance": False,
    }:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error("pool-singleton v2 scope changed")
    if config.get("design") != {
        "assignment_seed": 20260821,
        "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
        "morphology_program_count": 16,
        "particles_per_program": 4,
        "sample_steps": 8,
        "state_steps": [0, 2, 4, 6, 8],
        "completion_checkpoints": [2, 4, 6],
        "expanded_program_manifest_sha256": (EXPECTED_EXPANDED_PROGRAM_MANIFEST_SHA256),
        "particle_seed_manifest_sha256": EXPECTED_PARTICLE_SEED_MANIFEST_SHA256,
    }:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error("pool-singleton v2 design changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or frozenset(inputs) != EXPECTED_INPUT_KEYS:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error("pool-singleton v2 input set changed")
    paths = {label: _pin(repo, value, label=label) for label, value in inputs.items()}
    manifest = _load(paths["production_generator_manifest"], label="production manifest")
    if (
        sha256_file(paths["production_generator_manifest"]) != PRODUCTION_GENERATOR_MANIFEST_SHA256
        or manifest.get("status") != "frozen_after_independent_decoder_confirmation"
        or manifest.get("identity", {}).get("checkpoint_step") != 2000
        or manifest.get("identity", {}).get("terminal_decoder") != "bond_stochastic"
        or tuple(manifest.get("identity", {}).get("maximum_adjacent_branch_runs_by_role", []))
        != MAXIMUM_ADJACENT_BRANCH_RUNS
    ):
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "production generator manifest changed semantically"
        )
    prior = _validate_prior_receipt(paths["prior_selected_v2_equivalence"])
    qualification = build_selected_generator_implementation_qualification(repo).to_dict()
    if (
        qualification.get("implementation_sha256") != EXPECTED_IMPLEMENTATION_SHA256
        or tuple(tuple(item) for item in qualification.get("runtime_versions", []))
        != EXPECTED_RUNTIME_VERSIONS
    ):
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "audit is not running in the qualified production environment"
        )
    _validate_runtime_receipt(paths["qualified_runtime_receipt"], qualification)
    return config, paths, prior, qualification


def _normalized_completion(
    receipt: GuidanceTerminalCompletionReceipt,
) -> dict[str, Any]:
    if receipt.terminal is None:
        return {
            "status": "completion_error",
            "error_detail": receipt.error_detail,
            "product_transition_calls": receipt.product_transition_calls,
        }
    terminal = receipt.terminal
    candidate = native_completion_record_from_locked_terminal(terminal)
    return {
        "status": "terminal",
        "terminal_valid": terminal.terminal_valid,
        "exact_l1": terminal.exact_l1,
        "terminal_bytes_sha256": hashlib.sha256(terminal.terminal_bytes).hexdigest(),
        "candidate_record_sha256": _sha256_payload(candidate),
        "product_transition_calls": receipt.product_transition_calls,
    }


def _exact_completion(receipt: GuidanceTerminalCompletionReceipt) -> dict[str, Any]:
    value = _normalized_completion(receipt)
    if receipt.terminal is None:
        return value
    terminal = receipt.terminal
    return {
        **value,
        "unit_id": terminal.unit_id,
        "terminal_id": terminal.terminal_id,
        "morphology_program_sha256": terminal.morphology_program_sha256,
        "checkpoint_index": terminal.checkpoint_index,
        "generator_checkpoint_sha256": terminal.generator_checkpoint_sha256,
        "closure_checkpoint_sha256": terminal.closure_checkpoint_sha256,
        "generation_trace_sha256": hashlib.sha256(terminal.generation_trace_bytes).hexdigest(),
    }


def _particle_state_record(particle: Any) -> dict[str, Any]:
    return {
        "program_sha256": particle.program_sha256,
        "categorical_state_sha256": particle.categorical_state_sha256,
        "rng_state_sha256": particle.rng_state_sha256,
        "step": particle.trajectory.step,
    }


def _program_manifest(programs: tuple[bytes, ...]) -> str:
    return _sha256_payload(tuple(hashlib.sha256(item).hexdigest() for item in programs))


def build_selected_v2_pool_singleton_equivalence_v2(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute the production-runtime selected-v2 equivalence contract."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config, paths, prior, qualification = _validate_config(repo, config_path)
    design_value = config["design"]
    qualification_schedule = load_grouped_smc_schedule_qualification(paths["grouped_schedule"])
    assignment = qualification_schedule.by_seed()[design_value["assignment_seed"]]
    design = ParticleGroupDesign(
        morphology_program_count=design_value["morphology_program_count"],
        particles_per_program=design_value["particles_per_program"],
    )
    expanded_programs = design.expanded_programs(assignment.programs)
    if (
        assignment.assignment_sha256 != EXPECTED_ASSIGNMENT_SHA256
        or len(assignment.programs) != 16
        or len(set(assignment.programs)) != 16
        or len(assignment.stochastic_particle_seeds) != 64
        or len(set(assignment.stochastic_particle_seeds)) != 64
        or _program_manifest(expanded_programs) != EXPECTED_EXPANDED_PROGRAM_MANIFEST_SHA256
        or _particle_seed_manifest_sha256(assignment.stochastic_particle_seeds)
        != EXPECTED_PARTICLE_SEED_MANIFEST_SHA256
    ):
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "grouped assignment differs from the frozen production design"
        )

    lane = build_selected_model_restartable_guidance_lane_v2(repo)
    if lane.adapter_identity_sha256 != EXPECTED_BASE_ADAPTER_IDENTITY_SHA256:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "current adapter identity differs from the prior equivalence chain"
        )
    if prior["selected_generator"]["adapter_identity_sha256"] != lane.adapter_identity_sha256:
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "prior and current selected-v2 adapter identities differ"
        )

    pooled_receipt = lane.initialize(
        expanded_programs,
        seed=assignment.seed,
        particle_seeds=assignment.stochastic_particle_seeds,
        device="cpu",
    )
    if (
        pooled_receipt.consumed_particle_seed_manifest_sha256
        != EXPECTED_PARTICLE_SEED_MANIFEST_SHA256
    ):
        raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
            "pooled initialization seed manifest changed"
        )
    pooled_state = pooled_receipt.state
    singleton_states = []
    singleton_seed_manifests = []
    for program, particle_seed in zip(
        expanded_programs, assignment.stochastic_particle_seeds, strict=True
    ):
        receipt = lane.initialize(
            (program,),
            seed=assignment.seed,
            particle_seeds=(particle_seed,),
            device="cpu",
        )
        expected_manifest = _particle_seed_manifest_sha256((particle_seed,))
        if receipt.consumed_particle_seed_manifest_sha256 != expected_manifest:
            raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
                "singleton initialization seed manifest changed"
            )
        singleton_states.append(receipt.state)
        singleton_seed_manifests.append(expected_manifest)

    rows: list[dict[str, Any]] = []

    def compare_states(step: int) -> None:
        for particle_index, singleton in enumerate(singleton_states):
            pooled_record = _particle_state_record(pooled_state.particles[particle_index])
            singleton_record = _particle_state_record(singleton.particles[0])
            if pooled_record != singleton_record:
                raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
                    f"pooled/singleton state differs at step {step}, particle {particle_index}"
                )
            rows.append(
                {
                    "comparison": "state",
                    "step": step,
                    "particle_index": particle_index,
                    "program_index": particle_index // design.particles_per_program,
                    "particle_seed": assignment.stochastic_particle_seeds[particle_index],
                    "pooled": pooled_record,
                    "singleton": singleton_record,
                    "equal": True,
                }
            )

    compare_states(0)
    previous = 0
    completion_checkpoints = set(design_value["completion_checkpoints"])
    for step in design_value["state_steps"][1:]:
        pooled_advanced = lane.advance(pooled_state, target_step=step)
        if pooled_advanced.product_transition_calls != design.particle_count * (step - previous):
            raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
                "pooled transition accounting changed"
            )
        pooled_state = pooled_advanced.state
        advanced_singletons = []
        for singleton in singleton_states:
            advanced = lane.advance(singleton, target_step=step)
            if advanced.product_transition_calls != step - previous:
                raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
                    "singleton transition accounting changed"
                )
            advanced_singletons.append(advanced.state)
        singleton_states = advanced_singletons
        compare_states(step)

        if step in completion_checkpoints:
            for particle_index, singleton in enumerate(singleton_states):
                program_index = particle_index // design.particles_per_program
                local_index = particle_index % design.particles_per_program
                seed = keyed_random_seed(
                    assignment.seed,
                    arm="matched_checkpoint_completion",
                    program_index=program_index,
                    particle_index=local_index,
                    checkpoint_index=step,
                    rollout_index=0,
                )
                pooled_before = pooled_state.state_sha256
                singleton_before = singleton.state_sha256
                pooled_completion = lane.complete_terminal(
                    pooled_state,
                    particle_index=particle_index,
                    seed=seed,
                    checkpoint_index=step,
                )
                singleton_completion = lane.complete_terminal(
                    singleton,
                    particle_index=0,
                    seed=seed,
                    checkpoint_index=step,
                )
                pooled_after = pooled_state.state_sha256
                singleton_after = singleton.state_sha256
                pooled_value = _normalized_completion(pooled_completion)
                singleton_value = _normalized_completion(singleton_completion)
                if (
                    pooled_value != singleton_value
                    or pooled_before != pooled_after
                    or singleton_before != singleton_after
                ):
                    raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
                        f"checkpoint completion differs or mutates state at step {step}, "
                        f"particle {particle_index}"
                    )
                rows.append(
                    {
                        "comparison": "checkpoint_completion",
                        "step": step,
                        "particle_index": particle_index,
                        "program_index": program_index,
                        "within_program_particle_index": local_index,
                        "program_sha256": pooled_state.particles[particle_index].program_sha256,
                        "particle_seed": assignment.stochastic_particle_seeds[particle_index],
                        "rollout_seed": seed,
                        "pooled": pooled_value,
                        "singleton": singleton_value,
                        "pooled_source_before_sha256": pooled_before,
                        "pooled_source_after_sha256": pooled_after,
                        "singleton_source_before_sha256": singleton_before,
                        "singleton_source_after_sha256": singleton_after,
                        "source_states_unchanged": True,
                        "equal": True,
                    }
                )
        previous = step

    for particle_index, (particle_seed, singleton) in enumerate(
        zip(assignment.stochastic_particle_seeds, singleton_states, strict=True)
    ):
        invocation_seed = particle_seed + 1
        pooled_before = pooled_state.state_sha256
        singleton_before = singleton.state_sha256
        pooled_completion = lane.complete_terminal(
            pooled_state,
            particle_index=particle_index,
            seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        singleton_completion = lane.complete_terminal(
            singleton,
            particle_index=0,
            seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        pooled_after = pooled_state.state_sha256
        singleton_after = singleton.state_sha256
        pooled_value = _normalized_completion(pooled_completion)
        singleton_value = _normalized_completion(singleton_completion)
        if (
            pooled_value != singleton_value
            or pooled_before != pooled_after
            or singleton_before != singleton_after
        ):
            raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
                f"productive completion differs or mutates state at particle {particle_index}"
            )
        request = lane.completion_generation_request(
            pooled_state,
            particle_index=particle_index,
            invocation_seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        direct_completion = GuidanceTerminalCompletionReceipt(
            terminal=lane.callback(request), product_transition_calls=0
        )
        pooled_exact = _exact_completion(pooled_completion)
        direct_exact = _exact_completion(direct_completion)
        if pooled_exact != direct_exact:
            raise UgiSelectedV2PoolSingletonEquivalenceV2Error(
                f"productive direct callback differs at particle {particle_index}"
            )
        rows.append(
            {
                "comparison": "productive_completion",
                "step": SAMPLE_STEPS,
                "particle_index": particle_index,
                "program_index": particle_index // design.particles_per_program,
                "within_program_particle_index": (particle_index % design.particles_per_program),
                "program_sha256": pooled_state.particles[particle_index].program_sha256,
                "particle_seed": particle_seed,
                "invocation_seed": invocation_seed,
                "pooled": pooled_value,
                "singleton": singleton_value,
                "pooled_exact": pooled_exact,
                "direct": direct_exact,
                "pooled_source_before_sha256": pooled_before,
                "pooled_source_after_sha256": pooled_after,
                "singleton_source_before_sha256": singleton_before,
                "singleton_source_after_sha256": singleton_after,
                "source_states_unchanged": True,
                "equal": True,
            }
        )

    result_content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "selected_v2_production_runtime_pool_singleton_equivalent",
        "decision": "qualified_for_equivalence_binding_only",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "runtime_qualification": qualification,
        "prior_equivalence": {
            "file_sha256": sha256_file(paths["prior_selected_v2_equivalence"]),
            "result_sha256": prior["result_sha256"],
            "adapter_identity_sha256": prior["selected_generator"]["adapter_identity_sha256"],
        },
        "selected_generator": {
            "base_adapter_identity_sha256": lane.adapter_identity_sha256,
            "checkpoint_sha256": GENERATOR_CHECKPOINT_SHA256,
            "terminal_decoder_id": TERMINAL_DECODER_ID,
            "maximum_adjacent_branch_runs": list(MAXIMUM_ADJACENT_BRANCH_RUNS),
        },
        "initialization": {
            "expanded_program_manifest_sha256": _program_manifest(expanded_programs),
            "pooled_particle_seed_manifest_sha256": (
                pooled_receipt.consumed_particle_seed_manifest_sha256
            ),
            "singleton_particle_seed_manifests_sha256": _sha256_payload(
                tuple(singleton_seed_manifests)
            ),
        },
        "execution": {
            "assignment_seed": assignment.seed,
            "assignment_sha256": assignment.assignment_sha256,
            "particles": design.particle_count,
            "state_comparisons": design.particle_count * len(design_value["state_steps"]),
            "checkpoint_completion_comparisons": design.particle_count
            * len(design_value["completion_checkpoints"]),
            "productive_singleton_comparisons": design.particle_count,
            "productive_direct_callback_comparisons": design.particle_count,
            "source_state_mutations": 0,
            "all_equal": True,
            "comparison_rows_sha256": _sha256_payload(rows),
        },
        "scope": {
            **config["scope"],
            "production_sources_exercised": True,
            "production_execution": False,
            "nonzero_guidance_authorized": False,
        },
    }
    return {**result_content, "result_sha256": _sha256_payload(result_content)}, rows


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "EXPECTED_BASE_ADAPTER_IDENTITY_SHA256",
    "EXPECTED_IMPLEMENTATION_SHA256",
    "EXPECTED_INPUT_KEYS",
    "EXPECTED_RUNTIME_VERSIONS",
    "RESULT_SCHEMA_VERSION",
    "UgiSelectedV2PoolSingletonEquivalenceV2Error",
    "_validate_config",
    "build_selected_v2_pool_singleton_equivalence_v2",
]
