"""Calibration-only dynamic terminal census for the frozen selected-v3 Ugi generator.

This module is additive.  It does not modify the selected-v2/v3 generator or
guidance adapters and it deliberately stops at native exact-L1 terminal
completion.  It never evaluates potency, routes, synthesis values, proposals,
or candidate selection.

The census has a nested design: frozen morphology programs, independent noisy
states within each program, and independent terminal continuations from exact
flow checkpoints.  Partial states are saved as primitive/tensor-only payloads
that can be loaded with ``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.product.defog_feasibility import sha256_file
from forge.product.ugi_joint_end_to_end_sampling import complete_ugi_joint_terminals
from forge.product.ugi_joint_sparse_sampling import (
    UgiJointSparseTrajectoryState,
    advance_ugi_joint_sparse_state,
    finalize_ugi_joint_sparse_state,
)
from forge.product.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.product.ugi_selected_guidance_adapter import (
    SelectedGuidanceParticleState,
    SelectedGuidanceState,
)
from forge.product.ugi_selected_guidance_adapter_v3 import (
    build_selected_model_restartable_guidance_lane_v3,
)
from forge.product.ugi_selected_restartable_generator import SAMPLE_STEPS
from forge.product.ugi_selected_restartable_generator_v2 import (
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    TERMINAL_TEMPERATURE,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - productive census requires torch
    torch = None


CONFIG_SCHEMA_VERSION = "phase1_ugi_dynamic_frozen_prior_terminal_census_config.v1"
STATE_SCHEMA_VERSION = "forge.ugi_dynamic_frozen_prior_partial_state.v1"
SHARD_SCHEMA_VERSION = "phase1_ugi_dynamic_frozen_prior_terminal_census_shard.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_dynamic_frozen_prior_terminal_census.v1"

EXPECTED_INPUT_KEYS = frozenset(
    {
        "fresh_pool_config",
        "frozen_programs",
        "prior_terminal_reference",
        "production_generator_manifest",
        "generator_checkpoint",
        "closure_checkpoint",
        "selected_v3_adapter_source",
        "selected_v2_equivalence_result",
        "selected_v2_equivalence_rows",
        "census_source",
        "census_runner",
        "census_tests",
    }
)
EXPECTED_SCOPE = {
    "partition": "calibration",
    "candidate_selection": False,
    "guidance": False,
    "nonzero_guidance": False,
    "oracle_calls": 0,
    "potency_predictions": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "proposal_calls": 0,
    "sealed_holdout_access": False,
    "prospective_candidate_lock": False,
    "retries_or_repairs": False,
}


class UgiDynamicTerminalCensusError(RuntimeError):
    """Raised when the frozen-prior census contract is violated."""


def _stable_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise UgiDynamicTerminalCensusError("value is not canonically serializable") from error


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _lower_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UgiDynamicTerminalCensusError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UgiDynamicTerminalCensusError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UgiDynamicTerminalCensusError(f"{label} must be a nonnegative integer")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    if torch is None:
        raise UgiDynamicTerminalCensusError("partial-state persistence requires torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _jsonl_gzip_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        for row in rows:
            handle.write(_stable_json_bytes(dict(row)) + b"\n")
    return buffer.getvalue()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiDynamicTerminalCensusError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiDynamicTerminalCensusError(f"{label} must be a JSON object")
    return value


def _resolve_pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiDynamicTerminalCensusError(f"{label} pin is malformed")
    expected = _lower_sha256(record["sha256"], label=f"{label} sha256")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiDynamicTerminalCensusError(f"{label} path escapes the repository") from error
    if not path.is_file() or path.is_symlink():
        raise UgiDynamicTerminalCensusError(f"{label} is missing or is not a real file")
    if sha256_file(path) != expected:
        raise UgiDynamicTerminalCensusError(f"{label} file hash changed")
    return path


@dataclass(frozen=True)
class CensusDesign:
    """Frozen nested sampling design."""

    population_programs: int
    selected_programs: int
    states_per_program: int
    checkpoints: tuple[int, ...]
    rollouts_per_state_checkpoint: int
    sample_steps: int
    shard_programs: int
    program_selection_seed: int
    particle_seed_base: int
    rollout_seed_base: int
    device: str

    def __post_init__(self) -> None:
        for label in (
            "population_programs",
            "selected_programs",
            "states_per_program",
            "rollouts_per_state_checkpoint",
            "sample_steps",
            "shard_programs",
        ):
            _positive_integer(getattr(self, label), label=label)
        for label in ("program_selection_seed", "particle_seed_base", "rollout_seed_base"):
            _nonnegative_integer(getattr(self, label), label=label)
        if self.selected_programs > self.population_programs:
            raise UgiDynamicTerminalCensusError(
                "selected_programs cannot exceed population_programs"
            )
        if self.device != "cpu":
            raise UgiDynamicTerminalCensusError("selected-v3 census is CPU-qualified only")
        if (
            not self.checkpoints
            or tuple(sorted(set(self.checkpoints))) != self.checkpoints
            or any(step <= 0 or step >= self.sample_steps for step in self.checkpoints)
        ):
            raise UgiDynamicTerminalCensusError(
                "checkpoints must be unique increasing interior flow steps"
            )

    @property
    def selected_partial_states(self) -> int:
        return self.selected_programs * self.states_per_program * len(self.checkpoints)

    @property
    def terminal_attempts(self) -> int:
        return self.selected_partial_states * self.rollouts_per_state_checkpoint

    @property
    def shard_count(self) -> int:
        return math.ceil(self.selected_programs / self.shard_programs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_programs": self.population_programs,
            "selected_programs": self.selected_programs,
            "states_per_program": self.states_per_program,
            "checkpoints": list(self.checkpoints),
            "rollouts_per_state_checkpoint": self.rollouts_per_state_checkpoint,
            "sample_steps": self.sample_steps,
            "shard_programs": self.shard_programs,
            "program_selection_seed": self.program_selection_seed,
            "particle_seed_base": self.particle_seed_base,
            "rollout_seed_base": self.rollout_seed_base,
            "device": self.device,
            "selected_partial_states": self.selected_partial_states,
            "terminal_attempts": self.terminal_attempts,
            "shard_count": self.shard_count,
        }


@dataclass(frozen=True)
class CensusContract:
    """Validated config and immutable input paths."""

    repo: Path
    config_path: Path
    config_sha256: str
    design: CensusDesign
    inputs: Mapping[str, Path]
    scope: Mapping[str, Any]


@dataclass(frozen=True)
class SelectedProgram:
    """One deterministic member of the 4,096-program prior draw."""

    population_index: int
    selection_rank: int
    product_id: str
    program_bytes: bytes
    program_sha256: str
    metadata: Mapping[str, Any]
    selection_priority_sha256: str

    def manifest_record(self) -> dict[str, Any]:
        return {
            "population_index": self.population_index,
            "selection_rank": self.selection_rank,
            "product_id": self.product_id,
            "program_sha256": self.program_sha256,
            "selection_priority_sha256": self.selection_priority_sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SelectedProgramManifest:
    """Selected programs plus the logical design hash."""

    programs: tuple[SelectedProgram, ...]
    manifest_sha256: str
    population_manifest_sha256: str


def _design_from_config(value: Any) -> CensusDesign:
    if not isinstance(value, dict):
        raise UgiDynamicTerminalCensusError("design must be a JSON object")
    expected = {
        "population_programs",
        "selected_programs",
        "states_per_program",
        "checkpoints",
        "rollouts_per_state_checkpoint",
        "sample_steps",
        "shard_programs",
        "program_selection_seed",
        "particle_seed_base",
        "rollout_seed_base",
        "device",
        "selection_method",
        "terminal_decoder_mode",
        "terminal_temperature",
        "maximum_adjacent_branch_runs",
        "expected_partial_states",
        "expected_terminal_attempts",
    }
    if set(value) != expected:
        raise UgiDynamicTerminalCensusError("design fields changed")
    if value["selection_method"] != "sha256_priority_without_replacement_v1":
        raise UgiDynamicTerminalCensusError("selection method changed")
    if value["terminal_decoder_mode"] != "bond_stochastic":
        raise UgiDynamicTerminalCensusError("terminal decoder changed")
    if float(value["terminal_temperature"]) != TERMINAL_TEMPERATURE:
        raise UgiDynamicTerminalCensusError("terminal temperature changed")
    if tuple(value["maximum_adjacent_branch_runs"]) != MAXIMUM_ADJACENT_BRANCH_RUNS:
        raise UgiDynamicTerminalCensusError("adjacent branch-run policy changed")
    design = CensusDesign(
        population_programs=value["population_programs"],
        selected_programs=value["selected_programs"],
        states_per_program=value["states_per_program"],
        checkpoints=tuple(value["checkpoints"]),
        rollouts_per_state_checkpoint=value["rollouts_per_state_checkpoint"],
        sample_steps=value["sample_steps"],
        shard_programs=value["shard_programs"],
        program_selection_seed=value["program_selection_seed"],
        particle_seed_base=value["particle_seed_base"],
        rollout_seed_base=value["rollout_seed_base"],
        device=value["device"],
    )
    if design.sample_steps != SAMPLE_STEPS:
        raise UgiDynamicTerminalCensusError("selected-v3 sample-step count changed")
    if value["expected_partial_states"] != design.selected_partial_states:
        raise UgiDynamicTerminalCensusError("expected partial-state count is inconsistent")
    if value["expected_terminal_attempts"] != design.terminal_attempts:
        raise UgiDynamicTerminalCensusError("expected terminal-attempt count is inconsistent")
    return design


def load_census_contract(repo: Path, config_path: Path) -> CensusContract:
    """Validate the frozen census config without constructing the model."""

    root = repo.resolve()
    resolved_config = config_path.resolve()
    try:
        resolved_config.relative_to(root)
    except ValueError as error:
        raise UgiDynamicTerminalCensusError("config path escapes the repository") from error
    config = _read_json(resolved_config, label="census config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiDynamicTerminalCensusError("unsupported census config schema")
    if config.get("status") != "frozen_before_calibration_only_census":
        raise UgiDynamicTerminalCensusError("census config is not frozen")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiDynamicTerminalCensusError("census scope changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or frozenset(inputs) != EXPECTED_INPUT_KEYS:
        raise UgiDynamicTerminalCensusError("census input set changed")
    paths = {label: _resolve_pin(root, pin, label=label) for label, pin in inputs.items()}
    design = _design_from_config(config.get("design"))
    return CensusContract(
        repo=root,
        config_path=resolved_config,
        config_sha256=sha256_file(resolved_config),
        design=design,
        inputs=paths,
        scope=dict(config["scope"]),
    )


def _selection_priority(seed: int, *, product_id: str, program_sha256: str) -> str:
    return _sha256_payload(
        {
            "method": "sha256_priority_without_replacement_v1",
            "seed": seed,
            "product_id": product_id,
            "program_sha256": program_sha256,
        }
    )


def select_frozen_prior_programs(
    program_rows: Sequence[Mapping[str, Any]],
    *,
    selected_count: int,
    selection_seed: int,
) -> SelectedProgramManifest:
    """Select programs by stable content priority without consulting terminals."""

    _positive_integer(selected_count, label="selected_count")
    _nonnegative_integer(selection_seed, label="selection_seed")
    if selected_count > len(program_rows):
        raise UgiDynamicTerminalCensusError("selected_count exceeds the program population")
    candidates = []
    product_ids = set()
    population_records = []
    metadata_fields = (
        "source_stratum",
        "branch_class",
        "component_novelty_class",
        "held_role_class",
    )
    for index, row in enumerate(program_rows):
        if not isinstance(row, Mapping):
            raise UgiDynamicTerminalCensusError(f"program row {index} is malformed")
        product_id = row.get("product_id")
        program = row.get("program")
        if not isinstance(product_id, str) or not product_id or not isinstance(program, Mapping):
            raise UgiDynamicTerminalCensusError(f"program row {index} lacks identity or program")
        if product_id in product_ids:
            raise UgiDynamicTerminalCensusError("frozen program product IDs are not unique")
        product_ids.add(product_id)
        try:
            program_bytes = canonical_morphology_program_bytes(dict(program))
        except Exception as error:
            raise UgiDynamicTerminalCensusError(
                f"program row {index} is not a canonical morphology program"
            ) from error
        program_sha256 = hashlib.sha256(program_bytes).hexdigest()
        priority = _selection_priority(
            selection_seed,
            product_id=product_id,
            program_sha256=program_sha256,
        )
        metadata = {key: row[key] for key in metadata_fields if key in row}
        candidates.append(
            {
                "population_index": index,
                "product_id": product_id,
                "program_bytes": program_bytes,
                "program_sha256": program_sha256,
                "metadata": metadata,
                "selection_priority_sha256": priority,
            }
        )
        population_records.append(
            {
                "product_id": product_id,
                "program_sha256": program_sha256,
            }
        )
    ordered = sorted(
        candidates,
        key=lambda row: (
            row["selection_priority_sha256"],
            row["product_id"],
            row["program_sha256"],
        ),
    )
    selected = tuple(
        SelectedProgram(selection_rank=rank, **row)
        for rank, row in enumerate(ordered[:selected_count])
    )
    records = [value.manifest_record() for value in selected]
    return SelectedProgramManifest(
        programs=selected,
        manifest_sha256=_sha256_payload(records),
        population_manifest_sha256=_sha256_payload(
            sorted(population_records, key=lambda row: row["product_id"])
        ),
    )


def load_selected_program_manifest(contract: CensusContract) -> SelectedProgramManifest:
    """Load and deterministically select the frozen prior programs."""

    value = _read_json(contract.inputs["frozen_programs"], label="frozen programs")
    rows = value.get("samples")
    if not isinstance(rows, list) or len(rows) != contract.design.population_programs:
        raise UgiDynamicTerminalCensusError("frozen program population count changed")
    manifest = select_frozen_prior_programs(
        rows,
        selected_count=contract.design.selected_programs,
        selection_seed=contract.design.program_selection_seed,
    )
    return manifest


def _keyed_seed(base_seed: int, **coordinates: Any) -> int:
    _nonnegative_integer(base_seed, label="base_seed")
    payload = _stable_json_bytes({"base_seed": base_seed, "coordinates": coordinates})
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def particle_seed(
    design: CensusDesign,
    *,
    population_index: int,
    state_replicate: int,
) -> int:
    return _keyed_seed(
        design.particle_seed_base,
        purpose="dynamic_frozen_prior_partial_state",
        population_index=population_index,
        state_replicate=state_replicate,
    )


def rollout_seed(
    design: CensusDesign,
    *,
    population_index: int,
    state_replicate: int,
    checkpoint_index: int,
    rollout_index: int,
) -> int:
    return _keyed_seed(
        design.rollout_seed_base,
        purpose="dynamic_frozen_prior_native_completion",
        population_index=population_index,
        state_replicate=state_replicate,
        checkpoint_index=checkpoint_index,
        rollout_index=rollout_index,
    )


def _tensor_cpu_clone(value: Any, *, label: str) -> Any:
    if torch is None or not torch.is_tensor(value):
        raise UgiDynamicTerminalCensusError(f"{label} is not a tensor")
    return value.detach().cpu().clone()


def pack_partial_state(state: SelectedGuidanceState) -> dict[str, Any]:
    """Pack a selected state without repeated layout/source tensors."""

    if not isinstance(state, SelectedGuidanceState):
        raise UgiDynamicTerminalCensusError("partial state must be SelectedGuidanceState")
    particles = []
    for particle in state.particles:
        particles.append(
            {
                "global_particle_index": particle.global_particle_index,
                "program_index": particle.program_index,
                "program_b64": base64.b64encode(particle.program_bytes).decode("ascii"),
                "original_particle_seed": particle.original_particle_seed,
                "founder_particle_index": particle.founder_particle_index,
                "ancestry_path": list(particle.ancestry_path),
                "step": particle.trajectory.step,
                "sample_steps": particle.trajectory.sample_steps,
                "device": particle.trajectory.device,
                "channels": {
                    key: _tensor_cpu_clone(value, label=f"channel {key}")
                    for key, value in sorted(particle.trajectory.channels.items())
                },
                "generator_state": _tensor_cpu_clone(
                    particle.trajectory.generator_state,
                    label="generator_state",
                ),
                "program_sha256": particle.program_sha256,
                "categorical_state_sha256": particle.categorical_state_sha256,
                "rng_state_sha256": particle.rng_state_sha256,
                "provenance_sha256": particle.provenance_sha256,
                "lineage_sha256": particle.lineage_sha256,
                "particle_state_sha256": particle.state_sha256,
            }
        )
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "base_seed": state.base_seed,
        "adapter_identity_sha256": state.adapter_identity_sha256,
        "step": state.step,
        "state_sha256": state.state_sha256,
        "categorical_state_sha256": state.categorical_state_sha256,
        "provenance_sha256": state.provenance_sha256,
        "lineage_sha256": state.lineage_sha256,
        "particles": particles,
    }


def _decode_program_b64(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise UgiDynamicTerminalCensusError("packed program is malformed")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise UgiDynamicTerminalCensusError("packed program is not valid base64") from error
    if not decoded:
        raise UgiDynamicTerminalCensusError("packed program is empty")
    return decoded


def restore_partial_state(lane: Any, payload: Mapping[str, Any]) -> SelectedGuidanceState:
    """Reconstruct a selected state and require exact logical hashes."""

    if torch is None:
        raise UgiDynamicTerminalCensusError("partial-state restoration requires torch")
    if not isinstance(payload, Mapping) or payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise UgiDynamicTerminalCensusError("unsupported partial-state schema")
    records = payload.get("particles")
    if not isinstance(records, list) or not records:
        raise UgiDynamicTerminalCensusError("partial state has no particles")
    adapter_identity = _lower_sha256(
        payload.get("adapter_identity_sha256"),
        label="adapter identity",
    )
    if getattr(lane, "adapter_identity_sha256", None) != adapter_identity:
        raise UgiDynamicTerminalCensusError("partial state belongs to another adapter")
    programs = tuple(_decode_program_b64(record.get("program_b64")) for record in records)
    seeds = tuple(
        _nonnegative_integer(record.get("original_particle_seed"), label="particle seed")
        for record in records
    )
    base_seed = _nonnegative_integer(payload.get("base_seed"), label="base seed")
    initialized = lane.initialize(
        programs,
        seed=base_seed,
        particle_seeds=seeds,
        device="cpu",
    ).state
    if not isinstance(initialized, SelectedGuidanceState):
        raise UgiDynamicTerminalCensusError("lane did not initialize a selected guidance state")
    particles = []
    for index, (record, reference) in enumerate(zip(records, initialized.particles, strict=True)):
        if not isinstance(record, Mapping):
            raise UgiDynamicTerminalCensusError("packed particle is malformed")
        channels = record.get("channels")
        if not isinstance(channels, Mapping) or set(channels) != set(reference.trajectory.channels):
            raise UgiDynamicTerminalCensusError("packed channel set changed")
        restored_channels = {
            key: _tensor_cpu_clone(channels[key], label=f"packed channel {key}").to(
                reference.trajectory.device
            )
            for key in sorted(channels)
        }
        generator_state = _tensor_cpu_clone(
            record.get("generator_state"), label="packed generator_state"
        ).to(reference.trajectory.device)
        step = _nonnegative_integer(record.get("step"), label="packed step")
        sample_steps = _positive_integer(record.get("sample_steps"), label="packed sample_steps")
        if sample_steps != SAMPLE_STEPS or step > sample_steps:
            raise UgiDynamicTerminalCensusError("packed flow boundary is invalid")
        trajectory = UgiJointSparseTrajectoryState(
            programs=reference.trajectory.programs,
            layout={key: value.clone() for key, value in reference.trajectory.layout.items()},
            sources={key: value.clone() for key, value in reference.trajectory.sources.items()},
            channels=restored_channels,
            sample_steps=sample_steps,
            step=step,
            generator_state=generator_state,
            device=reference.trajectory.device,
        )
        particle = SelectedGuidanceParticleState(
            global_particle_index=_nonnegative_integer(
                record.get("global_particle_index"), label="global_particle_index"
            ),
            program_index=_nonnegative_integer(record.get("program_index"), label="program_index"),
            program_bytes=programs[index],
            original_particle_seed=seeds[index],
            trajectory=trajectory,
            founder_particle_index=_nonnegative_integer(
                record.get("founder_particle_index"), label="founder_particle_index"
            ),
            ancestry_path=tuple(record.get("ancestry_path", ())),
        )
        for key, observed in (
            ("program_sha256", particle.program_sha256),
            ("categorical_state_sha256", particle.categorical_state_sha256),
            ("rng_state_sha256", particle.rng_state_sha256),
            ("provenance_sha256", particle.provenance_sha256),
            ("lineage_sha256", particle.lineage_sha256),
            ("particle_state_sha256", particle.state_sha256),
        ):
            if record.get(key) != observed:
                raise UgiDynamicTerminalCensusError(
                    f"restored particle {index} {key} differs from its packed hash"
                )
        particles.append(particle)
    restored = SelectedGuidanceState(
        base_seed=base_seed,
        adapter_identity_sha256=adapter_identity,
        particles=tuple(particles),
    )
    for key, observed in (
        ("state_sha256", restored.state_sha256),
        ("categorical_state_sha256", restored.categorical_state_sha256),
        ("provenance_sha256", restored.provenance_sha256),
        ("lineage_sha256", restored.lineage_sha256),
    ):
        if payload.get(key) != observed:
            raise UgiDynamicTerminalCensusError(f"restored state {key} differs")
    if payload.get("step") != restored.step:
        raise UgiDynamicTerminalCensusError("restored state step differs")
    return restored


def save_partial_state(path: Path, state: SelectedGuidanceState) -> dict[str, Any]:
    """Persist and immediately weights-only reload one partial state."""

    if torch is None:
        raise UgiDynamicTerminalCensusError("partial-state persistence requires torch")
    payload = pack_partial_state(state)
    _atomic_torch_save(path, payload)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or loaded.get("state_sha256") != state.state_sha256:
        raise UgiDynamicTerminalCensusError("weights-only partial-state reload failed")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "state_sha256": state.state_sha256,
        "step": state.step,
        "particles": len(state.particles),
    }


def load_partial_state(path: Path, lane: Any) -> SelectedGuidanceState:
    """Load a trusted task artifact with the weights-only unpickler."""

    if torch is None or not path.is_file() or path.is_symlink():
        raise UgiDynamicTerminalCensusError("partial-state artifact is missing")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise UgiDynamicTerminalCensusError("partial-state artifact cannot be loaded") from error
    return restore_partial_state(lane, payload)


def _replace_trajectory_generator_state(
    trajectory: UgiJointSparseTrajectoryState,
    generator_state: Any,
) -> UgiJointSparseTrajectoryState:
    return UgiJointSparseTrajectoryState(
        programs=trajectory.programs,
        layout={key: value.clone() for key, value in trajectory.layout.items()},
        sources={key: value.clone() for key, value in trajectory.sources.items()},
        channels={key: value.clone() for key, value in trajectory.channels.items()},
        sample_steps=trajectory.sample_steps,
        step=trajectory.step,
        generator_state=generator_state.clone(),
        device=trajectory.device,
    )


def complete_native_no_route_terminal(
    lane: Any,
    state: SelectedGuidanceState,
    *,
    particle_index: int,
    seed: int,
    checkpoint_index: int,
    rollout_index: int,
) -> dict[str, Any]:
    """Complete one clone to a native L1-annotated row with no value calls."""

    if torch is None:
        raise UgiDynamicTerminalCensusError("native terminal completion requires torch")
    _nonnegative_integer(particle_index, label="particle_index")
    _nonnegative_integer(seed, label="seed")
    _nonnegative_integer(rollout_index, label="rollout_index")
    if checkpoint_index not in (2, 4, 6) or checkpoint_index != state.step:
        raise UgiDynamicTerminalCensusError("native continuation requires checkpoint 2, 4 or 6")
    if particle_index >= len(state.particles):
        raise UgiDynamicTerminalCensusError("particle index exceeds partial state")
    if state.adapter_identity_sha256 != getattr(lane, "adapter_identity_sha256", None):
        raise UgiDynamicTerminalCensusError("lane and partial-state identities differ")
    particle = state.particles[particle_index]
    source_before = particle.state_sha256
    trajectory = particle.trajectory.clone()
    flow_state = torch.Generator(device=trajectory.device).manual_seed(seed).get_state()
    trajectory = _replace_trajectory_generator_state(trajectory, flow_state)
    trajectory = advance_ugi_joint_sparse_state(
        lane.callback.model,
        trajectory,
        target_step=SAMPLE_STEPS,
    )
    finalization = finalize_ugi_joint_sparse_state(
        lane.callback.model,
        trajectory,
        tree_generator_state=torch.Generator().manual_seed(seed + 1).get_state(),
        allowed_ring_sizes=lane.callback.allowed_ring_sizes,
        maximum_heavy_degree=lane.callback.maximum_heavy_degree,
        maximum_adjacent_branch_runs=MAXIMUM_ADJACENT_BRANCH_RUNS,
    )
    if len(finalization.terminals) != 1:
        raise UgiDynamicTerminalCensusError(
            "one partial particle did not finalize to exactly one terminal"
        )
    completion = complete_ugi_joint_terminals(
        lane.callback.model,
        lane.callback.closure_model,
        finalization.terminals,
        lane.callback.corpus,
        program_metadata=({},),
        closure_generator_state=torch.Generator().manual_seed(seed + 1).get_state(),
        allowed_ring_sizes=lane.callback.allowed_ring_sizes,
        maximum_heavy_degree=lane.callback.maximum_heavy_degree,
        l1_reaction=lane.callback.reaction,
        terminal_decoder_mode="bond_stochastic",
        terminal_generator_state=torch.Generator().manual_seed(seed + 2).get_state(),
        terminal_temperature=TERMINAL_TEMPERATURE,
    )
    if len(completion.rows) != 1:
        raise UgiDynamicTerminalCensusError(
            "one partial particle did not complete to exactly one native row"
        )
    native = dict(completion.rows[0])
    source_after = particle.state_sha256
    if source_after != source_before:
        raise UgiDynamicTerminalCensusError("native continuation mutated its partial state")
    identity = {
        "adapter_identity_sha256": state.adapter_identity_sha256,
        "particle_state_sha256": source_before,
        "particle_index": particle_index,
        "checkpoint_index": checkpoint_index,
        "rollout_index": rollout_index,
        "seed": seed,
    }
    return {
        "completion_id": _sha256_payload(identity),
        "status": "terminal",
        **identity,
        "source_state_unchanged": True,
        "product_transition_calls": SAMPLE_STEPS - checkpoint_index,
        "error_detail": None,
        "native_terminal": native,
    }


def _program_shard(
    manifest: SelectedProgramManifest,
    design: CensusDesign,
    shard_index: int,
) -> tuple[SelectedProgram, ...]:
    _nonnegative_integer(shard_index, label="shard_index")
    if shard_index >= design.shard_count:
        raise UgiDynamicTerminalCensusError("shard_index exceeds the design")
    start = shard_index * design.shard_programs
    return manifest.programs[start : start + design.shard_programs]


def _expanded_particle_design(
    programs: Sequence[SelectedProgram],
    design: CensusDesign,
) -> tuple[tuple[bytes, ...], tuple[int, ...], tuple[dict[str, Any], ...]]:
    expanded_programs = []
    seeds = []
    metadata = []
    for program in programs:
        for replicate in range(design.states_per_program):
            expanded_programs.append(program.program_bytes)
            seeds.append(
                particle_seed(
                    design,
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
    if len(set(seeds)) != len(seeds):
        raise UgiDynamicTerminalCensusError("particle seed collision")
    return tuple(expanded_programs), tuple(seeds), tuple(metadata)


def _shard_expected_counts(program_count: int, design: CensusDesign) -> dict[str, int]:
    particles = program_count * design.states_per_program
    partial_states = particles * len(design.checkpoints)
    return {
        "programs": program_count,
        "particles": particles,
        "partial_states": partial_states,
        "terminal_attempts": partial_states * design.rollouts_per_state_checkpoint,
    }


def _receipt_artifact(path: Path, *, root: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def _validate_completed_shard(
    shard_dir: Path,
    *,
    contract: CensusContract,
    manifest: SelectedProgramManifest,
    shard_index: int,
) -> dict[str, Any]:
    receipt_path = shard_dir / "receipt.json"
    receipt = _read_json(receipt_path, label="completed shard receipt")
    claimed_result_sha256 = _lower_sha256(
        receipt.get("result_sha256"), label="completed shard result"
    )
    receipt_content = {key: value for key, value in receipt.items() if key != "result_sha256"}
    if _sha256_payload(receipt_content) != claimed_result_sha256:
        raise UgiDynamicTerminalCensusError(f"completed shard {shard_index} logical hash changed")
    expected_programs = _program_shard(manifest, contract.design, shard_index)
    expected_counts = _shard_expected_counts(len(expected_programs), contract.design)
    expected_population_indices = [value.population_index for value in expected_programs]
    valid = (
        receipt.get("schema_version") == SHARD_SCHEMA_VERSION
        and receipt.get("status") == "complete_calibration_only_native_terminal_census_shard"
        and receipt.get("config_sha256") == contract.config_sha256
        and receipt.get("selected_program_manifest_sha256") == manifest.manifest_sha256
        and receipt.get("population_manifest_sha256") == manifest.population_manifest_sha256
        and receipt.get("shard_index") == shard_index
        and receipt.get("selected_program_population_indices") == expected_population_indices
        and receipt.get("counts") == expected_counts
        and receipt.get("scope") == EXPECTED_SCOPE
    )
    if not valid:
        raise UgiDynamicTerminalCensusError(f"completed shard {shard_index} changed")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise UgiDynamicTerminalCensusError("completed shard lacks artifacts")
    expected_artifacts = {
        *(f"partial_state_step_{checkpoint:02d}" for checkpoint in contract.design.checkpoints),
        "terminals",
    }
    if set(artifacts) != expected_artifacts:
        raise UgiDynamicTerminalCensusError("completed shard artifact set changed")
    for label, record in artifacts.items():
        if not isinstance(record, dict) or not {"path", "sha256"}.issubset(record):
            raise UgiDynamicTerminalCensusError(f"shard artifact {label} is malformed")
        path = (shard_dir / str(record["path"])).resolve()
        try:
            path.relative_to(shard_dir.resolve())
        except ValueError as error:
            raise UgiDynamicTerminalCensusError("shard artifact escapes its directory") from error
        if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
            raise UgiDynamicTerminalCensusError(f"shard artifact {label} changed")
        if label == "terminals":
            if (
                set(record) != {"path", "sha256", "rows", "logical_sha256"}
                or record.get("rows") != expected_counts["terminal_attempts"]
            ):
                raise UgiDynamicTerminalCensusError("terminal artifact receipt changed")
            _lower_sha256(record.get("logical_sha256"), label="terminal logical hash")
        else:
            checkpoint = int(label.rsplit("_", 1)[1])
            if (
                set(record) != {"path", "sha256", "state_sha256", "step", "particles"}
                or record.get("step") != checkpoint
                or record.get("particles") != expected_counts["particles"]
            ):
                raise UgiDynamicTerminalCensusError("partial-state artifact receipt changed")
            _lower_sha256(record.get("state_sha256"), label="partial-state logical hash")
    return receipt


NativeCompleter = Callable[..., dict[str, Any]]


def execute_census_shard(
    contract: CensusContract,
    manifest: SelectedProgramManifest,
    output_dir: Path,
    *,
    shard_index: int,
    lane: Any,
    native_completer: NativeCompleter = complete_native_no_route_terminal,
) -> dict[str, Any]:
    """Execute one independently resumable calibration-only shard."""

    shard_root = output_dir.resolve() / "shards"
    final_dir = shard_root / f"shard_{shard_index:04d}"
    if final_dir.exists():
        receipt = _validate_completed_shard(
            final_dir,
            contract=contract,
            manifest=manifest,
            shard_index=shard_index,
        )
        return {**receipt, "resumed_completed_shard": True}
    shard_root.mkdir(parents=True, exist_ok=True)
    local_programs = _program_shard(manifest, contract.design, shard_index)
    programs, seeds, particle_metadata = _expanded_particle_design(local_programs, contract.design)
    expected_counts = _shard_expected_counts(len(local_programs), contract.design)
    work_dir = Path(tempfile.mkdtemp(prefix=f".shard_{shard_index:04d}.", dir=shard_root))
    start = time.perf_counter()
    try:
        initialized = lane.initialize(
            programs,
            seed=contract.design.particle_seed_base,
            particle_seeds=seeds,
            device=contract.design.device,
        )
        state = initialized.state
        if not isinstance(state, SelectedGuidanceState):
            raise UgiDynamicTerminalCensusError("lane initialization returned an invalid state")
        state_artifacts = {}
        terminal_rows = []
        previous_step = 0
        for checkpoint in contract.design.checkpoints:
            advanced = lane.advance(state, target_step=checkpoint)
            expected_calls = len(programs) * (checkpoint - previous_step)
            if advanced.product_transition_calls != expected_calls:
                raise UgiDynamicTerminalCensusError(
                    "productive transition accounting changed during census"
                )
            state = advanced.state
            state_path = work_dir / f"partial_state_step_{checkpoint:02d}.pt"
            artifact = save_partial_state(state_path, state)
            restored = load_partial_state(state_path, lane)
            if restored.state_sha256 != state.state_sha256:
                raise UgiDynamicTerminalCensusError("saved partial-state replay differs")
            state = restored
            state_artifacts[f"partial_state_step_{checkpoint:02d}"] = {
                **artifact,
                "path": state_path.name,
            }
            source_pool_before = state.state_sha256
            for particle_index, metadata in enumerate(particle_metadata):
                for rollout_index in range(contract.design.rollouts_per_state_checkpoint):
                    seed = rollout_seed(
                        contract.design,
                        population_index=metadata["population_index"],
                        state_replicate=metadata["state_replicate"],
                        checkpoint_index=checkpoint,
                        rollout_index=rollout_index,
                    )
                    completion = native_completer(
                        lane,
                        state,
                        particle_index=particle_index,
                        seed=seed,
                        checkpoint_index=checkpoint,
                        rollout_index=rollout_index,
                    )
                    terminal_rows.append(
                        {
                            "shard_index": shard_index,
                            **metadata,
                            **completion,
                        }
                    )
            if state.state_sha256 != source_pool_before:
                raise UgiDynamicTerminalCensusError("checkpoint rollouts mutated the source pool")
            previous_step = checkpoint
        if len(terminal_rows) != expected_counts["terminal_attempts"]:
            raise UgiDynamicTerminalCensusError("terminal ledger count differs from the design")
        completion_ids = [row.get("completion_id") for row in terminal_rows]
        if len(set(completion_ids)) != len(completion_ids):
            raise UgiDynamicTerminalCensusError("terminal completion IDs are not unique")
        terminal_path = work_dir / "terminals.jsonl.gz"
        _atomic_write(terminal_path, _jsonl_gzip_bytes(terminal_rows))
        artifacts = {
            **state_artifacts,
            "terminals": {
                "path": terminal_path.name,
                "sha256": sha256_file(terminal_path),
                "rows": len(terminal_rows),
                "logical_sha256": _sha256_payload(terminal_rows),
            },
        }
        content = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "status": "complete_calibration_only_native_terminal_census_shard",
            "completed_at_utc": _utc_now(),
            "config_sha256": contract.config_sha256,
            "selected_program_manifest_sha256": manifest.manifest_sha256,
            "population_manifest_sha256": manifest.population_manifest_sha256,
            "adapter_identity_sha256": lane.adapter_identity_sha256,
            "shard_index": shard_index,
            "selected_program_population_indices": [
                value.population_index for value in local_programs
            ],
            "particle_seed_manifest_sha256": _sha256_payload(seeds),
            "counts": expected_counts,
            "artifacts": artifacts,
            "elapsed_seconds": time.perf_counter() - start,
            "scope": dict(EXPECTED_SCOPE),
        }
        receipt = {**content, "result_sha256": _sha256_payload(content)}
        _atomic_json(work_dir / "receipt.json", receipt)
        os.replace(work_dir, final_dir)
        return {**receipt, "resumed_completed_shard": False}
    except Exception:
        # A failed work directory is intentionally retained for diagnosis.  It
        # is never treated as resumable evidence because it has no final name.
        raise


def run_census_shard(
    repo: Path,
    config_path: Path,
    output_dir: Path,
    *,
    shard_index: int,
) -> dict[str, Any]:
    """Build the authenticated selected-v3 lane and execute one shard."""

    contract = load_census_contract(repo, config_path)
    manifest = load_selected_program_manifest(contract)
    lane = build_selected_model_restartable_guidance_lane_v3(contract.repo)
    return execute_census_shard(
        contract,
        manifest,
        output_dir,
        shard_index=shard_index,
        lane=lane,
    )


def census_plan(repo: Path, config_path: Path) -> dict[str, Any]:
    """Return the complete frozen design without constructing the model."""

    contract = load_census_contract(repo, config_path)
    manifest = load_selected_program_manifest(contract)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "planned_not_executed",
        "config": {
            "path": str(contract.config_path.relative_to(contract.repo)),
            "sha256": contract.config_sha256,
        },
        "design": contract.design.to_dict(),
        "selected_program_manifest_sha256": manifest.manifest_sha256,
        "population_manifest_sha256": manifest.population_manifest_sha256,
        "selected_program_population_indices": [
            value.population_index for value in manifest.programs
        ],
        "scope": dict(contract.scope),
    }


def aggregate_completed_census(
    repo: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate every shard and compose deterministic terminal/state ledgers."""

    contract = load_census_contract(repo, config_path)
    manifest = load_selected_program_manifest(contract)
    root = output_dir.resolve()
    receipts = []
    terminal_rows = []
    state_manifest = []
    for shard_index in range(contract.design.shard_count):
        shard_dir = root / "shards" / f"shard_{shard_index:04d}"
        if not shard_dir.is_dir() or shard_dir.is_symlink():
            raise UgiDynamicTerminalCensusError(f"census shard {shard_index} is incomplete")
        receipt = _validate_completed_shard(
            shard_dir,
            contract=contract,
            manifest=manifest,
            shard_index=shard_index,
        )
        receipts.append(receipt)
        terminal_path = shard_dir / receipt["artifacts"]["terminals"]["path"]
        with gzip.open(terminal_path, "rt") as handle:
            for line in handle:
                terminal_rows.append(json.loads(line))
        for label, artifact in sorted(receipt["artifacts"].items()):
            if label.startswith("partial_state_step_"):
                state_manifest.append(
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
    if len(terminal_rows) != contract.design.terminal_attempts:
        raise UgiDynamicTerminalCensusError("aggregate terminal count differs from design")
    terminal_path = root / "terminal_ledger.jsonl.gz"
    state_path = root / "partial_state_manifest.json"
    _atomic_write(terminal_path, _jsonl_gzip_bytes(terminal_rows))
    _atomic_json(state_path, state_manifest)
    statuses = {}
    exact_l1 = 0
    valid = 0
    for row in terminal_rows:
        status = str(row.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        native = row.get("native_terminal")
        if isinstance(native, dict):
            valid += int(native.get("raw_molecule_valid") is True)
            exact_l1 += int(native.get("terminal_valid") is True)
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_calibration_only_dynamic_terminal_census",
        "completed_at_utc": _utc_now(),
        "config": {
            "path": str(contract.config_path.relative_to(contract.repo)),
            "sha256": contract.config_sha256,
        },
        "design": contract.design.to_dict(),
        "selected_program_manifest_sha256": manifest.manifest_sha256,
        "population_manifest_sha256": manifest.population_manifest_sha256,
        "counts": {
            "shards": len(receipts),
            "partial_states": len(manifest.programs)
            * contract.design.states_per_program
            * len(contract.design.checkpoints),
            "terminal_attempts": len(terminal_rows),
            "native_rows": sum(
                isinstance(row.get("native_terminal"), dict) for row in terminal_rows
            ),
            "raw_molecule_valid": valid,
            "valid_exact_l1": exact_l1,
            "statuses": statuses,
        },
        "artifacts": {
            "terminal_ledger": _receipt_artifact(terminal_path, root=root),
            "partial_state_manifest": _receipt_artifact(state_path, root=root),
            "shard_receipts_sha256": _sha256_payload(
                [receipt["result_sha256"] for receipt in receipts]
            ),
        },
        "scope": dict(EXPECTED_SCOPE),
        "nonclaims": [
            "no oracle or potency value was evaluated",
            "no L2/L3 route or synthesis value was evaluated",
            "no proposal, guidance, candidate selection or prospective lock was performed",
        ],
    }
    result = {**content, "result_sha256": _sha256_payload(content)}
    _atomic_json(root / "result.json", result)
    return result


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CensusContract",
    "CensusDesign",
    "EXPECTED_SCOPE",
    "RESULT_SCHEMA_VERSION",
    "SHARD_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "SelectedProgram",
    "SelectedProgramManifest",
    "UgiDynamicTerminalCensusError",
    "aggregate_completed_census",
    "census_plan",
    "complete_native_no_route_terminal",
    "execute_census_shard",
    "load_census_contract",
    "load_partial_state",
    "load_selected_program_manifest",
    "pack_partial_state",
    "particle_seed",
    "restore_partial_state",
    "rollout_seed",
    "run_census_shard",
    "save_partial_state",
    "select_frozen_prior_programs",
]
