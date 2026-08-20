"""One-shot blinded execution of the paired Ugi-3 route-saturation holdout.

This module is intentionally separate from the ordinary molecular-sampling CLI.
It revalidates the prereveal registry binding, claims the holdout exactly once,
generates the frozen molecular sample without rendering it, evaluates the same
exact-L1-eligible products against R0 and R1, and atomically seals private
identity ledgers plus a de-identified public aggregate.

The public entry point never returns molecular identities.  A failed attempt
leaves an immutable claim and an in-progress directory so that it cannot be
silently retried.
"""

from __future__ import annotations

import ctypes
import errno
import gzip
import hashlib
import importlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rdkit import rdBase

from forge.data.r1_prime_audit import sha256_file
from forge.design.sampling.ugi_blinded_headless_sampling import (
    HeadlessRuntime,
    preflight_headless_runtime,
    sample_headless_runtime,
)
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES
from forge.route.assessment.ugi3_route_registry_pair_contract import (
    PRIVATE_COMPONENT_TRANSITION_FIELDS,
    PRIVATE_ELIGIBILITY_FIELDS,
    PRIVATE_PRODUCT_TRANSITION_FIELDS,
    RECORD_LEDGER_SCHEMA_VERSION,
    TARGET_KEY_SHA256,
    Ugi3RouteRegistryPairContractError,
    component_key_sha256,
    exact_l1_eligible,
    validate_binding,
    validate_public_aggregate,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI environment gate
    torch = None


CONFIG_SCHEMA_VERSION = "phase1_ugi3_route_saturation_blinded_execution_config.v1"
EXECUTABLE_STATUS = "frozen_one_shot_execution_unrun"
PENDING_STATUS = "awaiting_reconciled_immutable_binding"
SAMPLE_SCHEMA_VERSION = "phase1_ugi3_route_saturation_private_sample.v1"
ELIGIBILITY_SCHEMA_VERSION = "phase1_ugi3_route_saturation_eligibility_ledger.v1"
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_route_saturation_component_transitions.v1"
PRODUCT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_route_saturation_product_transitions.v1"
MANIFEST_SCHEMA_VERSION = "phase1_ugi3_route_saturation_execution_manifest.v1"
SEAL_SCHEMA_VERSION = "phase1_ugi3_route_saturation_execution_seal.v1"
BINDING_VALIDATION_SCHEMA_VERSION = "phase1_ugi3_route_saturation_binding_validation.v1"
PUBLIC_AGGREGATE_SCHEMA_VERSION = "phase1_ugi3_route_saturation_public_aggregate.v1"
CLAIM_SCHEMA_VERSION = "phase1_ugi3_route_saturation_one_shot_claim.v1"
CLAIM_STATUS = "irreversible_one_shot_claim"
MANIFEST_STATUS = "complete_private_outputs_ready_for_atomic_publication"
SEALED_STATUS = "one_shot_sample_evaluated_and_atomically_sealed"
CONFIDENCE_INTERVAL_METHOD = "two_sided_clopper_pearson_95pct"

REQUIRED_POLICY = {
    "sample_exactly_once": True,
    "permit_retry": False,
    "permit_resampling": False,
    "permit_overwrite": False,
    "render_molecules": False,
    "inspect_molecules_during_execution": False,
    "same_products_for_r0_and_r1": True,
    "exact_role_and_constitutional_identity_only": True,
    "permit_similarity_or_family_route_closure": False,
    "permit_post_reveal_registry_or_threshold_tuning": False,
    "biological_guidance_enabled": False,
    "write_private_identity_ledgers": True,
    "write_public_aggregate_only": True,
}

REQUIRED_DECISION = {
    "binding_reconciled": True,
    "one_shot_execution_run": False,
    "holdout_molecules_inspected": False,
    "biological_guidance_enabled": False,
}

INPUT_NAMES = {
    "protocol",
    "binding",
    "holdout_contract",
    "holdout_seal",
    "program_draw",
    "joint_checkpoint",
    "closure_checkpoint",
    "prepared_training_cache",
    "qualified_reactions",
}
IMPLEMENTATION_NAMES = {
    "execution_module",
    "execution_cli",
    "headless_sampling_module",
    "headless_dependency_manifest",
    "joint_flow_module",
    "registry_pair_contract_module",
}
OUTPUT_NAMES = {"final_directory", "attempt_directory", "one_shot_claim"}
ARTIFACT_FILES = {
    "binding_validation": "binding_validation.json",
    "private_sample": "private_sample.json.gz",
    "private_eligibility": "private_exact_l1_eligibility.json.gz",
    "private_components": "private_component_transitions.json.gz",
    "private_products": "private_product_transitions.json.gz",
    "public_aggregate": "public_aggregate.json",
    "manifest": "manifest.json",
    "seal": "seal.json",
}
EXPECTED_OUTPUT_FILES = {"claim.json", *ARTIFACT_FILES.values()}
FORBIDDEN_RENDER_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".pdf",
}
EVIDENCE_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
PUBLIC_EVIDENCE_FAMILIES = {
    "exact_identity",
    "family_projected",
    "not_applicable",
    "provenance_only",
    "unregistered_exact_identity",
    "declared_record_without_family",
}
PREFLIGHT_MANIFEST_FIELDS = (
    "status",
    "joint_checkpoint_schema",
    "closure_checkpoint_schema",
    "prepared_cache_sha256",
    "prepared_cache_schema",
    "atom_vocabulary_size",
    "program_rows",
    "qualified_reaction_loaded",
    "models_constructed_and_state_loaded",
    "molecular_sampling_performed",
    "rdkit_molecular_renderer_imported",
)
RUNTIME_DEPENDENCY_MANIFEST_SCHEMA_VERSION = "phase1_ugi3_blinded_headless_dependency_manifest.v1"
RUNTIME_DEPENDENCY_MODULES = (
    "forge.chem.reactive_sites",
    "forge.data",
    "forge.data.r0_splits",
    "forge.data.r1_prime_audit",
    "forge.potency",
    "forge.potency.audit.ugi_semantic_annotations",
    "forge.design",
    "forge.design.flow.adapter_node_conditioning",
    "forge.design.audit.canonical_representation_audit",
    "forge.design.flow.defog_feasibility",
    "forge.design.flow.lipid_context",
    "forge.design.flow.lipid_support_skeleton",
    "forge.design.flow.phase1_flow",
    "forge.design.flow.phase1_tree_topology_flow",
    "forge.design.flow.sparse_topology_feasibility",
    "forge.design.flow.ugi_adapter_features",
    "forge.design.sampling.ugi_blinded_headless_sampling",
    "forge.design.corpus.ugi_chemistry_corpus",
    "forge.design.flow.ugi_chemistry_flow",
    "forge.design.flow.ugi_chemistry_interface",
    "forge.design.flow.ugi_closure_placement",
    "forge.design.corpus.ugi_component_expansion",
    "forge.design.corpus.ugi_generated_components",
    "forge.design.corpus.ugi_held_component_gate",
    "forge.design.flow.ugi_joint_sparse_flow",
    "forge.design.flow.ugi_morphology_flow",
    "forge.design.flow.ugi_morphology_program",
    "forge.design.training.ugi_training_cache",
    "forge.design.flow.v5_sparse_representation",
    "forge.route",
    "forge.route.engine.planner",
    "forge.route.engine.qualified_forward",
    "forge.route.evidence.ugi3_exact_c18_route",
    "forge.route.assessment.ugi3_route_registry_pair_contract",
    "forge.route.terminals.ugi3_route_saturation_blinded_execution",
    "forge.value",
    "forge.value.synthesis.synthesis",
)


class Ugi3BlindedHoldoutExecutionError(RuntimeError):
    """Raised when one-shot execution cannot proceed without violating the contract."""


@dataclass(frozen=True)
class PreparedExecution:
    """Validated, hash-owned one-shot execution inputs."""

    repo: Path
    config_path: Path
    config: dict[str, Any]
    config_sha256: str
    input_paths: dict[str, Path]
    input_hashes: dict[str, str]
    implementation_paths: dict[str, Path]
    implementation_hashes: dict[str, str]
    final_directory: Path
    attempt_directory: Path
    claim_path: Path
    program_rows: int
    sampling: dict[str, Any]


@dataclass(frozen=True)
class PairedEvaluation:
    """Private ledgers and the public paired-registry estimand."""

    eligibility_rows: tuple[dict[str, Any], ...]
    component_rows: tuple[dict[str, Any], ...]
    product_rows: tuple[dict[str, Any], ...]
    public_aggregate: dict[str, Any]


def _stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _gzip_json_bytes(value: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(_stable_json_bytes(value))
    return buffer.getvalue()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise Ugi3BlindedHoldoutExecutionError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3BlindedHoldoutExecutionError(f"{label} must be a JSON object")
    return value


def _load_gzip_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (FileNotFoundError, gzip.BadGzipFile, json.JSONDecodeError, OSError) as exc:
        raise Ugi3BlindedHoldoutExecutionError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3BlindedHoldoutExecutionError(f"{label} must be a JSON object")
    return value


def _repo_path(repo: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise Ugi3BlindedHoldoutExecutionError(f"{label} path is missing")
    path = Path(raw)
    return path if path.is_absolute() else repo / path


def _portable(path: Path, *, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_pinned(
    repo: Path,
    specification: Any,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(specification, dict):
        raise Ugi3BlindedHoldoutExecutionError(f"{label} specification is malformed")
    path = _repo_path(repo, specification.get("path"), label=label)
    expected = specification.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise Ugi3BlindedHoldoutExecutionError(f"{label} SHA-256 is malformed")
    try:
        observed = sha256_file(path)
    except OSError as exc:
        raise Ugi3BlindedHoldoutExecutionError(f"{label} is missing: {path}") from exc
    if observed != expected:
        raise Ugi3BlindedHoldoutExecutionError(f"{label} hash changed")
    return path, observed


def _validate_runtime_dependency_manifest(
    prepared: PreparedExecution,
    *,
    reject_unpinned_loaded_modules: bool,
) -> None:
    manifest = _load_json(
        prepared.implementation_paths["headless_dependency_manifest"],
        label="headless dependency manifest",
    )
    if (
        set(manifest) != {"schema_version", "status", "software", "modules"}
        or manifest.get("schema_version") != RUNTIME_DEPENDENCY_MANIFEST_SCHEMA_VERSION
    ):
        raise Ugi3BlindedHoldoutExecutionError("headless dependency manifest schema changed")
    if manifest.get("status") != "frozen_renderer_free_runtime_dependency_closure":
        raise Ugi3BlindedHoldoutExecutionError("headless dependency manifest is not frozen")
    expected_software = {
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", None),
        "numpy": np.__version__,
        "rdkit": rdBase.rdkitVersion,
        "pyyaml": yaml.__version__,
    }
    if manifest.get("software") != expected_software:
        raise Ugi3BlindedHoldoutExecutionError("blinded runtime software versions changed")
    modules = manifest.get("modules")
    if not isinstance(modules, dict) or set(modules) != set(RUNTIME_DEPENDENCY_MODULES):
        raise Ugi3BlindedHoldoutExecutionError("headless dependency module closure changed")
    for module_name in RUNTIME_DEPENDENCY_MODULES:
        specification = modules[module_name]
        path, observed = _validate_pinned(
            prepared.repo,
            specification,
            label=f"runtime dependency {module_name}",
        )
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != path.resolve():
            raise Ugi3BlindedHoldoutExecutionError(
                f"runtime dependency {module_name} resolved to an unpinned module"
            )
        if observed != specification["sha256"]:
            raise Ugi3BlindedHoldoutExecutionError(f"runtime dependency {module_name} hash changed")
    if reject_unpinned_loaded_modules:
        loaded_local: set[str] = set()
        source_root = (prepared.repo / "src").resolve()
        for module_name, module in sys.modules.items():
            module_file = getattr(module, "__file__", None)
            if not module_name.startswith("forge.") or module_file is None:
                continue
            resolved = Path(module_file).resolve()
            if resolved.is_relative_to(source_root):
                loaded_local.add(module_name)
        unpinned = loaded_local - set(RUNTIME_DEPENDENCY_MODULES)
        if unpinned:
            raise Ugi3BlindedHoldoutExecutionError(
                "runtime loaded unpinned local modules before claim: " + ", ".join(sorted(unpinned))
            )
        if any(
            module_name == "rdkit.Chem.Draw" or module_name.startswith("rdkit.Chem.Draw.")
            for module_name in sys.modules
        ):
            raise Ugi3BlindedHoldoutExecutionError("molecular renderer was imported before claim")


def _sampling_from_anchor(anchor: Mapping[str, Any]) -> dict[str, Any]:
    seeds = anchor.get("seeds")
    frozen = anchor.get("sampling")
    program = anchor.get("program_draw")
    if not isinstance(seeds, dict) or not isinstance(frozen, dict) or not isinstance(program, dict):
        raise Ugi3BlindedHoldoutExecutionError("protocol holdout anchor is malformed")
    return {
        "program_rows": program.get("rows"),
        "program_seed": seeds.get("program"),
        "flow_seed": seeds.get("flow"),
        "closure_seed": seeds.get("terminal"),
        "terminal_seed": seeds.get("terminal"),
        "sample_steps": frozen.get("sample_steps"),
        "batch_size": frozen.get("batch_size"),
        "maximum_adjacent_branch_runs": frozen.get("maximum_adjacent_branch_runs"),
        "terminal_decoder_mode": frozen.get("terminal_decoder_mode"),
        "terminal_temperature": frozen.get("terminal_temperature"),
        "device": "cpu",
        "evaluate_exact_l1_terminal_admission": True,
        "render_molecules": False,
        "biological_guidance_enabled": False,
    }


def prepare_execution(repo: Path, config_path: Path) -> PreparedExecution:
    """Validate every prereveal input without claiming or sampling the holdout."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="blinded execution config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3BlindedHoldoutExecutionError("unsupported blinded execution config schema")
    status = config.get("status")
    if status == PENDING_STATUS:
        raise Ugi3BlindedHoldoutExecutionError(
            "execution config is non-executable until the immutable binding is reconciled"
        )
    if status != EXECUTABLE_STATUS:
        raise Ugi3BlindedHoldoutExecutionError("blinded execution config is not executable")
    if config.get("policy") != REQUIRED_POLICY:
        raise Ugi3BlindedHoldoutExecutionError("one-shot execution policy changed")
    if config.get("decision") != REQUIRED_DECISION:
        raise Ugi3BlindedHoldoutExecutionError("one-shot decision state is unsafe")

    inputs = config.get("inputs")
    implementation = config.get("implementation")
    outputs = config.get("outputs")
    if not isinstance(inputs, dict) or set(inputs) != INPUT_NAMES:
        raise Ugi3BlindedHoldoutExecutionError("blinded execution inputs changed")
    if not isinstance(implementation, dict) or set(implementation) != IMPLEMENTATION_NAMES:
        raise Ugi3BlindedHoldoutExecutionError("blinded implementation pins changed")
    if not isinstance(outputs, dict) or set(outputs) != OUTPUT_NAMES:
        raise Ugi3BlindedHoldoutExecutionError("blinded execution outputs changed")

    input_paths: dict[str, Path] = {}
    input_hashes: dict[str, str] = {}
    for name in sorted(INPUT_NAMES):
        input_paths[name], input_hashes[name] = _validate_pinned(
            repo, inputs[name], label=name.replace("_", " ")
        )
    implementation_paths: dict[str, Path] = {}
    implementation_hashes: dict[str, str] = {}
    for name in sorted(IMPLEMENTATION_NAMES):
        implementation_paths[name], implementation_hashes[name] = _validate_pinned(
            repo, implementation[name], label=name.replace("_", " ")
        )

    protocol = _load_json(input_paths["protocol"], label="paired protocol")
    anchor = protocol.get("sealed_holdout_anchor")
    if not isinstance(anchor, dict):
        raise Ugi3BlindedHoldoutExecutionError("paired protocol lacks its holdout anchor")
    for name, anchor_name in (
        ("holdout_contract", "contract"),
        ("holdout_seal", "seal"),
        ("program_draw", "program_draw"),
    ):
        anchored = anchor.get(anchor_name)
        specification = inputs[name]
        if not isinstance(anchored, dict) or (
            anchored.get("path") != specification.get("path")
            or anchored.get("sha256") != specification.get("sha256")
        ):
            raise Ugi3BlindedHoldoutExecutionError(f"{name} is not the protocol-owned input")

    contract = _load_json(input_paths["holdout_contract"], label="holdout contract")
    contract_inputs = contract.get("inputs")
    if not isinstance(contract_inputs, dict):
        raise Ugi3BlindedHoldoutExecutionError("holdout contract inputs are malformed")
    for name in ("joint_checkpoint", "closure_checkpoint", "qualified_reactions"):
        frozen = contract_inputs.get(name)
        specification = inputs[name]
        if not isinstance(frozen, dict) or (
            frozen.get("path") != specification.get("path")
            or frozen.get("sha256") != specification.get("sha256")
        ):
            raise Ugi3BlindedHoldoutExecutionError(f"{name} is not holdout-contract-owned")

    binding = _load_json(input_paths["binding"], label="registry-pair binding")
    bound_protocol = binding.get("protocol")
    if not isinstance(bound_protocol, dict) or bound_protocol != {
        "path": inputs["protocol"]["path"],
        "sha256": input_hashes["protocol"],
    }:
        raise Ugi3BlindedHoldoutExecutionError(
            "registry-pair binding does not own the pinned protocol"
        )

    expected_sampling = _sampling_from_anchor(anchor)
    if config.get("sampling") != expected_sampling:
        raise Ugi3BlindedHoldoutExecutionError("sampling settings differ from the sealed protocol")
    if expected_sampling["closure_seed"] != expected_sampling["flow_seed"] + 1:
        raise Ugi3BlindedHoldoutExecutionError(
            "frozen closure seed no longer matches the qualified sampler schedule"
        )
    if contract.get("holdout_sampling", {}).get("sealed_sample_output") != anchor.get(
        "sealed_sample_output"
    ):
        raise Ugi3BlindedHoldoutExecutionError("holdout output path changed")

    final_directory = _repo_path(
        repo, outputs.get("final_directory"), label="final output directory"
    )
    attempt_directory = _repo_path(
        repo, outputs.get("attempt_directory"), label="attempt directory"
    )
    claim_path = _repo_path(repo, outputs.get("one_shot_claim"), label="one-shot claim")
    anchored_final = _repo_path(
        repo, anchor.get("sealed_sample_output"), label="protocol sample output"
    )
    if final_directory.resolve() != anchored_final.resolve():
        raise Ugi3BlindedHoldoutExecutionError("final directory is not the sealed output path")
    if len({final_directory.resolve(), attempt_directory.resolve(), claim_path.resolve()}) != 3:
        raise Ugi3BlindedHoldoutExecutionError("one-shot output paths are not distinct")
    if not (
        final_directory.parent.resolve()
        == attempt_directory.parent.resolve()
        == claim_path.parent.resolve()
    ):
        raise Ugi3BlindedHoldoutExecutionError("one-shot outputs must share one parent filesystem")

    program_rows = expected_sampling["program_rows"]
    if isinstance(program_rows, bool) or not isinstance(program_rows, int) or program_rows <= 0:
        raise Ugi3BlindedHoldoutExecutionError("sealed program count is invalid")
    prepared = PreparedExecution(
        repo=repo,
        config_path=config_path,
        config=config,
        config_sha256=sha256_file(config_path),
        input_paths=input_paths,
        input_hashes=input_hashes,
        implementation_paths=implementation_paths,
        implementation_hashes=implementation_hashes,
        final_directory=final_directory,
        attempt_directory=attempt_directory,
        claim_path=claim_path,
        program_rows=program_rows,
        sampling=expected_sampling,
    )
    _validate_runtime_dependency_manifest(
        prepared,
        reject_unpinned_loaded_modules=False,
    )
    return prepared


def _preflight_runtime(prepared: PreparedExecution) -> HeadlessRuntime:
    """Load and validate every sampling dependency before the irreversible claim."""

    try:
        return preflight_headless_runtime(
            prepared.repo,
            joint_checkpoint_path=prepared.input_paths["joint_checkpoint"],
            closure_checkpoint_path=prepared.input_paths["closure_checkpoint"],
            prepared_cache_path=prepared.input_paths["prepared_training_cache"],
            prepared_cache_sha256=prepared.input_hashes["prepared_training_cache"],
            qualified_reactions_path=prepared.input_paths["qualified_reactions"],
            program_draw_path=prepared.input_paths["program_draw"],
            expected_program_rows=prepared.program_rows,
        )
    except Exception as exc:
        raise Ugi3BlindedHoldoutExecutionError(
            f"renderer-free runtime preflight failed: {exc}"
        ) from exc


def _revalidate_prepared_hashes(prepared: PreparedExecution) -> None:
    """Close the preflight-to-claim TOCTOU window for every frozen byte stream."""

    if sha256_file(prepared.config_path) != prepared.config_sha256:
        raise Ugi3BlindedHoldoutExecutionError("execution config changed during preflight")
    for label, path, expected in (
        *(
            (f"input {name}", prepared.input_paths[name], prepared.input_hashes[name])
            for name in sorted(prepared.input_paths)
        ),
        *(
            (
                f"implementation {name}",
                prepared.implementation_paths[name],
                prepared.implementation_hashes[name],
            )
            for name in sorted(prepared.implementation_paths)
        ),
    ):
        try:
            observed = sha256_file(path)
        except OSError as exc:
            raise Ugi3BlindedHoldoutExecutionError(f"{label} disappeared during preflight") from exc
        if observed != expected:
            raise Ugi3BlindedHoldoutExecutionError(f"{label} changed during preflight")


def _headless_sample(
    prepared: PreparedExecution,
    runtime: HeadlessRuntime,
) -> dict[str, Any]:
    """Run the frozen sampler using only resources loaded before the claim."""

    pinned_inputs = {
        name: {
            "path": _portable(prepared.input_paths[name], repo=prepared.repo),
            "sha256": prepared.input_hashes[name],
        }
        for name in (
            "program_draw",
            "joint_checkpoint",
            "closure_checkpoint",
            "prepared_training_cache",
            "qualified_reactions",
        )
    }
    try:
        result = sample_headless_runtime(
            runtime,
            settings=prepared.sampling,
            pinned_inputs=pinned_inputs,
        )
    except Exception as exc:
        raise Ugi3BlindedHoldoutExecutionError("headless sampling failed after claim") from exc
    if len(result.get("samples", ())) != prepared.program_rows:
        raise Ugi3BlindedHoldoutExecutionError("sampler did not return one row per sealed program")
    return result


def _registry_records_from_manifest(
    repo: Path,
    specification: Mapping[str, Any],
    *,
    label: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_path, _ = _validate_pinned(repo, specification, label=f"{label} manifest")
    manifest = _load_json(manifest_path, label=f"{label} manifest")
    records_path, _ = _validate_pinned(
        repo, manifest.get("records"), label=f"{label} record ledger"
    )
    ledger = _load_gzip_json(records_path, label=f"{label} record ledger")
    if ledger.get("schema_version") != RECORD_LEDGER_SCHEMA_VERSION:
        raise Ugi3BlindedHoldoutExecutionError(f"{label} record schema changed")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise Ugi3BlindedHoldoutExecutionError(f"{label} records are malformed")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise Ugi3BlindedHoldoutExecutionError(f"{label} contains a malformed record")
        role = record.get("role")
        smiles = record.get("canonical_smiles")
        if not isinstance(role, str) or not isinstance(smiles, str):
            raise Ugi3BlindedHoldoutExecutionError(f"{label} record lacks exact identity")
        key = (role, smiles)
        if key in indexed:
            raise Ugi3BlindedHoldoutExecutionError(f"{label} contains duplicate identities")
        indexed[key] = record
    return indexed


def load_registry_pair(
    repo: Path,
    binding_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Load the already validator-qualified R0/R1 exact-identity records."""

    binding = _load_json(binding_path, label="registry-pair binding")
    r0 = _registry_records_from_manifest(repo, binding.get("r0_manifest"), label="R0/without-C18")
    r1 = _registry_records_from_manifest(repo, binding.get("r1_manifest"), label="R1/with-C18")
    if set(r0) != set(r1):
        raise Ugi3BlindedHoldoutExecutionError("R0/R1 registry identity sets changed")
    return r0, r1


def _sample_key(experiment_id: str, index: int, sampling: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": experiment_id,
        "sample_index": index,
        "program_seed": sampling["program_seed"],
        "flow_seed": sampling["flow_seed"],
        "closure_seed": sampling["closure_seed"],
        "terminal_seed": sampling["terminal_seed"],
    }
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def _transition(before: bool, after: bool, *, label: str) -> str:
    if before and not after:
        raise Ugi3BlindedHoldoutExecutionError(f"{label} regressed from complete to noncomplete")
    if before:
        return "complete_in_both"
    if after:
        return "noncomplete_to_complete"
    return "noncomplete_in_both"


def _evidence_family(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return "unregistered_exact_identity"
    value = record.get("value")
    if isinstance(value, dict) and isinstance(value.get("evidence_support"), str):
        family = value["evidence_support"]
        if family not in PUBLIC_EVIDENCE_FAMILIES:
            raise Ugi3BlindedHoldoutExecutionError(
                "registry evidence family is not approved for public aggregation"
            )
        return family
    source = record.get("source_class")
    if source is None:
        return "declared_record_without_family"
    raise Ugi3BlindedHoldoutExecutionError(
        "registry record lacks an approved public evidence-family label"
    )


def _log_binomial_sum(start: int, stop: int, n: int, p: float) -> float:
    """Return a stable finite binomial probability sum over [start, stop]."""

    if start > stop:
        return 0.0
    if p <= 0.0:
        return 1.0 if start <= 0 <= stop else 0.0
    if p >= 1.0:
        return 1.0 if start <= n <= stop else 0.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(index + 1)
        - math.lgamma(n - index + 1)
        + index * log_p
        + (n - index) * log_q
        for index in range(start, stop + 1)
    ]
    maximum = max(terms)
    return math.exp(maximum) * math.fsum(math.exp(value - maximum) for value in terms)


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute an exact two-sided binomial interval without extra dependencies."""

    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials < 0
        or successes < 0
        or successes > trials
    ):
        raise Ugi3BlindedHoldoutExecutionError("invalid binomial counts")
    if not 0.0 < confidence < 1.0:
        raise Ugi3BlindedHoldoutExecutionError("confidence must be between zero and one")
    if trials == 0:
        return 0.0, 1.0
    alpha_tail = (1.0 - confidence) / 2.0

    if successes == 0:
        lower = 0.0
    else:
        low, high = 0.0, 1.0
        for _ in range(64):
            midpoint = (low + high) / 2.0
            survival = _log_binomial_sum(successes, trials, trials, midpoint)
            if survival < alpha_tail:
                low = midpoint
            else:
                high = midpoint
        lower = (low + high) / 2.0

    if successes == trials:
        upper = 1.0
    else:
        low, high = 0.0, 1.0
        for _ in range(64):
            midpoint = (low + high) / 2.0
            cdf = _log_binomial_sum(0, successes, trials, midpoint)
            if cdf > alpha_tail:
                low = midpoint
            else:
                high = midpoint
        upper = (low + high) / 2.0
    return lower, upper


def evaluate_registry_pair(
    sample_rows: Sequence[Mapping[str, Any]],
    r0_records: Mapping[tuple[str, str], Mapping[str, Any]],
    r1_records: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    experiment_id: str,
    sampling: Mapping[str, Any],
    sealed_program_rows: int,
) -> PairedEvaluation:
    """Evaluate one sealed molecular sample against both exact registries."""

    if len(sample_rows) != sealed_program_rows:
        raise Ugi3BlindedHoldoutExecutionError("sample count differs from sealed program count")
    if set(r0_records) != set(r1_records):
        raise Ugi3BlindedHoldoutExecutionError("paired registry identity sets differ")

    eligibility_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    evidence_counts: dict[str, Counter[str]] = {"r0": Counter(), "r1": Counter()}
    evidence_complete: dict[str, Counter[str]] = {"r0": Counter(), "r1": Counter()}
    eligible_count = 0
    complete_r0_count = 0
    complete_r1_count = 0
    marginal_count = 0
    target_occurrences = 0
    target_flip_occurrences = 0

    for index, row in enumerate(sample_rows):
        if not isinstance(row, Mapping):
            raise Ugi3BlindedHoldoutExecutionError(f"sample row {index} is malformed")
        sample_key = _sample_key(experiment_id, index, sampling)
        forward = row.get("l1_forward_verification")
        exact_forward = (
            forward.get("exact_product_reconstructed") if isinstance(forward, Mapping) else None
        )
        eligible = exact_l1_eligible(row)
        eligibility = {
            "sample_key_sha256": sample_key,
            "valid": row.get("valid") is True,
            "component_reconstruction_valid": row.get("component_reconstruction_valid") is True,
            "exact_product_reconstructed": exact_forward is True,
            "eligible": eligible,
        }
        if tuple(eligibility) != PRIVATE_ELIGIBILITY_FIELDS:
            raise Ugi3BlindedHoldoutExecutionError("eligibility ledger field order changed")
        eligibility_rows.append(eligibility)

        if not eligible:
            product = {
                "sample_key_sha256": sample_key,
                "eligible": False,
                "r0_route_complete": False,
                "r1_route_complete": False,
                "transition": "ineligible_exact_l1",
            }
            if tuple(product) != PRIVATE_PRODUCT_TRANSITION_FIELDS:
                raise Ugi3BlindedHoldoutExecutionError("product ledger field order changed")
            product_rows.append(product)
            continue

        components = row.get("component_smiles_by_role")
        if not isinstance(components, Mapping) or set(components) != set(ROLE_NAMES):
            raise Ugi3BlindedHoldoutExecutionError(
                f"eligible sample row {index} lacks the exact three Ugi components"
            )
        eligible_count += 1
        r0_states: list[bool] = []
        r1_states: list[bool] = []
        target_in_product = 0
        for position, role in enumerate(ROLE_NAMES):
            smiles = components.get(role)
            if not isinstance(smiles, str) or not smiles:
                raise Ugi3BlindedHoldoutExecutionError(
                    f"eligible sample row {index} has a malformed {role} identity"
                )
            key = (role, smiles)
            key_hash = component_key_sha256(role, smiles)
            left = r0_records.get(key)
            right = r1_records.get(key)
            left_complete = bool(left is not None and left.get("route_complete") is True)
            right_complete = bool(right is not None and right.get("route_complete") is True)
            transition = _transition(left_complete, right_complete, label="component")
            component = {
                "sample_key_sha256": sample_key,
                "component_position": position,
                "component_key_sha256": key_hash,
                "role": role,
                "r0_route_complete": left_complete,
                "r1_route_complete": right_complete,
                "transition": transition,
            }
            if tuple(component) != PRIVATE_COMPONENT_TRANSITION_FIELDS:
                raise Ugi3BlindedHoldoutExecutionError("component ledger field order changed")
            component_rows.append(component)
            r0_states.append(left_complete)
            r1_states.append(right_complete)
            for registry, record, complete in (
                ("r0", left, left_complete),
                ("r1", right, right_complete),
            ):
                family = _evidence_family(record)
                evidence_counts[registry][family] += 1
                if complete:
                    evidence_complete[registry][family] += 1
            if key_hash == TARGET_KEY_SHA256:
                target_in_product += 1

        r0_complete = all(r0_states)
        r1_complete = all(r1_states)
        transition = _transition(r0_complete, r1_complete, label="product")
        product = {
            "sample_key_sha256": sample_key,
            "eligible": True,
            "r0_route_complete": r0_complete,
            "r1_route_complete": r1_complete,
            "transition": transition,
        }
        if tuple(product) != PRIVATE_PRODUCT_TRANSITION_FIELDS:
            raise Ugi3BlindedHoldoutExecutionError("product ledger field order changed")
        product_rows.append(product)
        complete_r0_count += int(r0_complete)
        complete_r1_count += int(r1_complete)
        flipped = transition == "noncomplete_to_complete"
        marginal_count += int(flipped)
        target_occurrences += target_in_product
        if flipped:
            target_flip_occurrences += target_in_product

    lower, upper = clopper_pearson_interval(marginal_count, eligible_count)
    by_family = {
        registry: {
            family: {
                "component_occurrences": evidence_counts[registry][family],
                "route_complete_occurrences": evidence_complete[registry][family],
            }
            for family in sorted(evidence_counts[registry])
        }
        for registry in ("r0", "r1")
    }
    public = {
        "sealed_program_rows": sealed_program_rows,
        "sampled_products": len(sample_rows),
        "exact_l1_eligible_products": eligible_count,
        "eligible_products_complete_in_r0": complete_r0_count,
        "eligible_products_complete_in_r1": complete_r1_count,
        "marginal_completion_count": marginal_count,
        "marginal_completion_fraction": (
            marginal_count / eligible_count if eligible_count else None
        ),
        "confidence_interval_method": CONFIDENCE_INTERVAL_METHOD,
        "confidence_interval_95pct": [lower, upper],
        "target_component_occurrences": target_occurrences,
        "target_component_occurrences_that_flip_product_completion": target_flip_occurrences,
        "aggregate_counts_by_evidence_family": by_family,
    }
    _validate_public_aggregate_strict(public)
    return PairedEvaluation(
        eligibility_rows=tuple(eligibility_rows),
        component_rows=tuple(component_rows),
        product_rows=tuple(product_rows),
        public_aggregate=public,
    )


def _require_nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Ugi3BlindedHoldoutExecutionError(f"{label} must be a nonnegative integer")
    return value


def _validate_public_aggregate_strict(payload: Mapping[str, Any]) -> None:
    """Validate the public estimand and recursively constrain its only free keys."""

    try:
        validate_public_aggregate(payload)
    except Ugi3RouteRegistryPairContractError as exc:
        raise Ugi3BlindedHoldoutExecutionError(str(exc)) from exc
    sealed = _require_nonnegative_integer(
        payload["sealed_program_rows"], label="sealed program rows"
    )
    sampled = _require_nonnegative_integer(payload["sampled_products"], label="sampled products")
    eligible = _require_nonnegative_integer(
        payload["exact_l1_eligible_products"], label="exact-L1 eligible products"
    )
    complete_r0 = _require_nonnegative_integer(
        payload["eligible_products_complete_in_r0"], label="R0-complete products"
    )
    complete_r1 = _require_nonnegative_integer(
        payload["eligible_products_complete_in_r1"], label="R1-complete products"
    )
    marginal = _require_nonnegative_integer(
        payload["marginal_completion_count"], label="marginal completion count"
    )
    target = _require_nonnegative_integer(
        payload["target_component_occurrences"], label="target occurrences"
    )
    target_flips = _require_nonnegative_integer(
        payload["target_component_occurrences_that_flip_product_completion"],
        label="target flip occurrences",
    )
    if not (sampled == sealed and eligible <= sampled):
        raise Ugi3BlindedHoldoutExecutionError("public sample denominators are inconsistent")
    if complete_r0 > eligible or complete_r1 > eligible or marginal > eligible:
        raise Ugi3BlindedHoldoutExecutionError("public route-completion counts exceed denominator")
    if complete_r1 != complete_r0 + marginal:
        raise Ugi3BlindedHoldoutExecutionError(
            "public paired completion transition is inconsistent"
        )
    if target > eligible or target_flips > target or target_flips > marginal:
        raise Ugi3BlindedHoldoutExecutionError("public target flip count exceeds occurrences")

    fraction = payload["marginal_completion_fraction"]
    if eligible == 0:
        if fraction is not None:
            raise Ugi3BlindedHoldoutExecutionError("zero-denominator fraction must be null")
    elif (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not math.isclose(float(fraction), marginal / eligible, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise Ugi3BlindedHoldoutExecutionError(
            "public marginal completion fraction is inconsistent"
        )
    if payload["confidence_interval_method"] != CONFIDENCE_INTERVAL_METHOD:
        raise Ugi3BlindedHoldoutExecutionError("public confidence-interval method changed")
    interval = payload["confidence_interval_95pct"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise Ugi3BlindedHoldoutExecutionError("public confidence interval is malformed")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in interval
    ):
        raise Ugi3BlindedHoldoutExecutionError("public confidence interval is malformed")
    lower, upper = map(float, interval)
    if not 0.0 <= lower <= upper <= 1.0:
        raise Ugi3BlindedHoldoutExecutionError("public confidence interval is out of bounds")
    if eligible and not lower <= float(fraction) <= upper:
        raise Ugi3BlindedHoldoutExecutionError("public confidence interval excludes its estimate")
    expected_interval = clopper_pearson_interval(marginal, eligible)
    if not all(
        math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15)
        for observed, expected in zip((lower, upper), expected_interval, strict=True)
    ):
        raise Ugi3BlindedHoldoutExecutionError(
            "public confidence interval was not recomputed exactly"
        )

    by_family = payload["aggregate_counts_by_evidence_family"]
    if not isinstance(by_family, dict) or set(by_family) != {"r0", "r1"}:
        raise Ugi3BlindedHoldoutExecutionError("evidence-family aggregate must contain R0 and R1")
    for registry in ("r0", "r1"):
        families = by_family[registry]
        if not isinstance(families, dict):
            raise Ugi3BlindedHoldoutExecutionError("evidence-family aggregate is malformed")
        total_occurrences = 0
        for family, counts in families.items():
            if (
                not isinstance(family, str)
                or EVIDENCE_FAMILY_RE.fullmatch(family) is None
                or family not in PUBLIC_EVIDENCE_FAMILIES
            ):
                raise Ugi3BlindedHoldoutExecutionError(
                    "evidence-family aggregate contains an unsafe free-form key"
                )
            if not isinstance(counts, dict) or set(counts) != {
                "component_occurrences",
                "route_complete_occurrences",
            }:
                raise Ugi3BlindedHoldoutExecutionError("evidence-family count record is malformed")
            occurrences = _require_nonnegative_integer(
                counts["component_occurrences"], label="component occurrences"
            )
            route_complete = _require_nonnegative_integer(
                counts["route_complete_occurrences"], label="route-complete occurrences"
            )
            if route_complete > occurrences:
                raise Ugi3BlindedHoldoutExecutionError(
                    "route-complete occurrences exceed component occurrences"
                )
            total_occurrences += occurrences
        if total_occurrences != len(ROLE_NAMES) * eligible:
            raise Ugi3BlindedHoldoutExecutionError(
                f"{registry.upper()} evidence-family counts do not cover all eligible components"
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _exclusive_claim(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise Ugi3BlindedHoldoutExecutionError(
            f"one-shot holdout was already claimed: {path}"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one directory while refusing every overwrite race."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    system = platform.system()
    if system == "Darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
    elif system == "Linux" and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)  # RENAME_NOREPLACE
    else:  # pragma: no cover - production is constrained to Linux/macOS
        raise Ugi3BlindedHoldoutExecutionError(
            "platform lacks an atomic no-replace directory publication primitive"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise Ugi3BlindedHoldoutExecutionError(
            f"sealed output appeared during atomic publication: {destination}"
        )
    raise Ugi3BlindedHoldoutExecutionError(
        f"atomic no-replace publication failed: {os.strerror(error_number)}"
    )


def _artifact_spec(
    payload: bytes,
    *,
    schema_version: str,
    visibility: str,
    rows: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": _sha256_bytes(payload),
        "schema_version": schema_version,
        "visibility": visibility,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _safe_preflight_summary(
    summary: Mapping[str, Any], prepared: PreparedExecution
) -> dict[str, Any]:
    safe = {field: summary.get(field) for field in PREFLIGHT_MANIFEST_FIELDS}
    required = {
        "status": "runtime_dependencies_preflighted_without_sampling",
        "joint_checkpoint_schema": "phase1_ugi_joint_sparse_checkpoint.v1",
        "prepared_cache_sha256": prepared.input_hashes["prepared_training_cache"],
        "prepared_cache_schema": "phase1_ugi_training_cache.v1",
        "program_rows": prepared.program_rows,
        "qualified_reaction_loaded": True,
        "models_constructed_and_state_loaded": True,
        "molecular_sampling_performed": False,
        "rdkit_molecular_renderer_imported": False,
    }
    if any(safe.get(key) != value for key, value in required.items()):
        raise Ugi3BlindedHoldoutExecutionError("runtime preflight summary is unsafe")
    if safe["closure_checkpoint_schema"] not in {
        "phase1_ugi_sparse_closure_checkpoint.v1",
        "phase1_ugi_sparse_closure_checkpoint.v2",
    }:
        raise Ugi3BlindedHoldoutExecutionError("runtime closure-checkpoint schema is unsafe")
    atom_vocabulary_size = safe["atom_vocabulary_size"]
    if (
        isinstance(atom_vocabulary_size, bool)
        or not isinstance(atom_vocabulary_size, int)
        or atom_vocabulary_size <= 0
    ):
        raise Ugi3BlindedHoldoutExecutionError("runtime atom vocabulary is malformed")
    return safe


def _preflight_registry_records(
    r0_records: Mapping[tuple[str, str], Mapping[str, Any]],
    r1_records: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    """Reject every registry-shape/evidence failure before consuming the claim."""

    if set(r0_records) != set(r1_records):
        raise Ugi3BlindedHoldoutExecutionError("R0/R1 registry identity sets changed")
    for registry_name, records in (("R0", r0_records), ("R1", r1_records)):
        for key, record in records.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or key[0] not in ROLE_NAMES
                or not isinstance(key[1], str)
                or not key[1]
                or not isinstance(record, Mapping)
                or record.get("role") != key[0]
                or record.get("canonical_smiles") != key[1]
                or not isinstance(record.get("route_complete"), bool)
            ):
                raise Ugi3BlindedHoldoutExecutionError(
                    f"{registry_name} registry record shape is incompatible with paired evaluation"
                )
            _evidence_family(record)
    for key in r0_records:
        if r0_records[key]["route_complete"] and not r1_records[key]["route_complete"]:
            raise Ugi3BlindedHoldoutExecutionError("R1 registry contains a completion regression")


def _binding_ready(result: Mapping[str, Any], prepared: PreparedExecution) -> None:
    required = {
        "registry_pair_bound": True,
        "holdout_reveal_authorized": True,
        "readiness": "immutable_registry_pair_valid_holdout_reveal_authorized",
        "authorized_delta_count": 1,
        "binding_sha256": prepared.input_hashes["binding"],
    }
    for field, expected in required.items():
        if result.get(field) != expected:
            raise Ugi3BlindedHoldoutExecutionError(
                f"binding validation did not authorize execution: {field}"
            )


def _validate_private_sample_payload(
    payload: Mapping[str, Any], prepared: PreparedExecution
) -> list[Mapping[str, Any]]:
    expected_fields = {
        "schema_version",
        "status",
        "visibility",
        "render",
        "rendered_molecules",
        "rendering_disabled",
        "samples",
        "statistics",
        "sampling",
        "pinned_inputs",
    }
    if set(payload) != expected_fields:
        raise Ugi3BlindedHoldoutExecutionError("private sample payload fields changed")
    required = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "status": "sampled_once_headless",
        "visibility": "private_identity_bearing_hash_pinned",
        "render": None,
        "rendered_molecules": 0,
        "rendering_disabled": True,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise Ugi3BlindedHoldoutExecutionError("private sample declaration is unsafe")
    rows = payload.get("samples")
    if not isinstance(rows, list) or len(rows) != prepared.program_rows:
        raise Ugi3BlindedHoldoutExecutionError(
            "headless sampler did not return the sealed sample count"
        )
    if not isinstance(payload.get("statistics"), Mapping) or not isinstance(
        payload.get("sampling"), Mapping
    ):
        raise Ugi3BlindedHoldoutExecutionError("private sample metadata is malformed")
    expected_pins = {
        name: {
            "path": _portable(prepared.input_paths[name], repo=prepared.repo),
            "sha256": prepared.input_hashes[name],
        }
        for name in (
            "program_draw",
            "joint_checkpoint",
            "closure_checkpoint",
            "prepared_training_cache",
            "qualified_reactions",
        )
    }
    if payload.get("pinned_inputs") != expected_pins:
        raise Ugi3BlindedHoldoutExecutionError("private sample input pins changed")
    return rows


def run_one_shot_blinded_holdout(
    repo: Path,
    config_path: Path,
    *,
    _sample_fn: Callable[[PreparedExecution, Any], Mapping[str, Any]] | None = None,
    _preflight_fn: Callable[[PreparedExecution], Any] | None = None,
    _binding_validator: Callable[[Path, Path, Path], Mapping[str, Any]] | None = None,
    _registry_loader: (
        Callable[
            [Path, Path],
            tuple[
                Mapping[tuple[str, str], Mapping[str, Any]],
                Mapping[tuple[str, str], Mapping[str, Any]],
            ],
        ]
        | None
    ) = None,
) -> dict[str, Any]:
    """Execute, evaluate and seal one blinded holdout exactly once.

    Test-only dependency injection is keyword-private.  The CLI always uses the
    real binding validator, headless sampler and immutable registry loader.
    """

    prepared = prepare_execution(repo, config_path)
    validator = _binding_validator or validate_binding
    try:
        binding_result = dict(
            validator(
                prepared.repo,
                prepared.input_paths["protocol"],
                prepared.input_paths["binding"],
            )
        )
    except Ugi3RouteRegistryPairContractError as exc:
        raise Ugi3BlindedHoldoutExecutionError(
            f"immutable registry binding did not validate: {exc}"
        ) from exc
    _binding_ready(binding_result, prepared)
    registry_loader = _registry_loader or load_registry_pair
    r0_records, r1_records = registry_loader(prepared.repo, prepared.input_paths["binding"])
    _preflight_registry_records(r0_records, r1_records)

    if os.path.lexists(prepared.final_directory):
        raise Ugi3BlindedHoldoutExecutionError(
            f"sealed holdout output already exists: {prepared.final_directory}"
        )
    if os.path.lexists(prepared.attempt_directory):
        raise Ugi3BlindedHoldoutExecutionError(
            f"prior one-shot attempt exists: {prepared.attempt_directory}"
        )
    if os.path.lexists(prepared.claim_path):
        raise Ugi3BlindedHoldoutExecutionError(
            f"one-shot holdout was already claimed: {prepared.claim_path}"
        )

    preflight = _preflight_fn or _preflight_runtime
    try:
        runtime = preflight(prepared)
    except Ugi3BlindedHoldoutExecutionError:
        raise
    except Exception as exc:
        raise Ugi3BlindedHoldoutExecutionError("runtime preflight failed before claim") from exc
    raw_preflight_summary = getattr(runtime, "preflight_summary", None)
    if not isinstance(raw_preflight_summary, Mapping):
        raise Ugi3BlindedHoldoutExecutionError("runtime preflight lacks its non-sampling summary")
    preflight_summary = _safe_preflight_summary(raw_preflight_summary, prepared)
    if _preflight_fn is None:
        _validate_runtime_dependency_manifest(
            prepared,
            reject_unpinned_loaded_modules=True,
        )
    _revalidate_prepared_hashes(prepared)
    if any(
        os.path.lexists(path)
        for path in (prepared.final_directory, prepared.attempt_directory, prepared.claim_path)
    ):
        raise Ugi3BlindedHoldoutExecutionError(
            "one-shot output, attempt or claim appeared during runtime preflight"
        )

    claim = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "status": CLAIM_STATUS,
        "claimed_at_utc": _utc_now(),
        "config_sha256": prepared.config_sha256,
        "binding_sha256": prepared.input_hashes["binding"],
        "permit_retry": False,
        "permit_resampling": False,
    }
    _exclusive_claim(prepared.claim_path, _pretty_json_bytes(claim))
    try:
        prepared.attempt_directory.mkdir(parents=False, exist_ok=False, mode=0o700)
        _atomic_write(
            prepared.attempt_directory / "claim.json",
            _pretty_json_bytes(claim),
        )
        sampler = _sample_fn or _headless_sample
        sample_payload = dict(sampler(prepared, runtime))
        rows = _validate_private_sample_payload(sample_payload, prepared)

        sample_bytes = _gzip_json_bytes(sample_payload)
        experiment_id = prepared.config.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise Ugi3BlindedHoldoutExecutionError("execution experiment_id is missing")
        evaluation = evaluate_registry_pair(
            rows,
            r0_records,
            r1_records,
            experiment_id=experiment_id,
            sampling=prepared.sampling,
            sealed_program_rows=prepared.program_rows,
        )

        binding_payload = {
            "schema_version": BINDING_VALIDATION_SCHEMA_VERSION,
            "status": "validated_before_one_shot_claim",
            "validation": binding_result,
        }
        eligibility_payload = {
            "schema_version": ELIGIBILITY_SCHEMA_VERSION,
            "visibility": "private_hash_pinned",
            "rows": list(evaluation.eligibility_rows),
        }
        component_payload = {
            "schema_version": COMPONENT_LEDGER_SCHEMA_VERSION,
            "visibility": "private_hash_pinned",
            "rows": list(evaluation.component_rows),
        }
        product_payload = {
            "schema_version": PRODUCT_LEDGER_SCHEMA_VERSION,
            "visibility": "private_hash_pinned",
            "rows": list(evaluation.product_rows),
        }
        payloads = {
            "binding_validation": _pretty_json_bytes(binding_payload),
            "private_sample": sample_bytes,
            "private_eligibility": _gzip_json_bytes(eligibility_payload),
            "private_components": _gzip_json_bytes(component_payload),
            "private_products": _gzip_json_bytes(product_payload),
            "public_aggregate": _pretty_json_bytes(evaluation.public_aggregate),
        }
        for name, payload in payloads.items():
            _atomic_write(prepared.attempt_directory / ARTIFACT_FILES[name], payload)

        artifact_specs = {
            "binding_validation": _artifact_spec(
                payloads["binding_validation"],
                schema_version=BINDING_VALIDATION_SCHEMA_VERSION,
                visibility="private_hash_pinned",
            ),
            "private_sample": _artifact_spec(
                payloads["private_sample"],
                schema_version=SAMPLE_SCHEMA_VERSION,
                visibility="private_identity_bearing_hash_pinned",
                rows=len(rows),
            ),
            "private_eligibility": _artifact_spec(
                payloads["private_eligibility"],
                schema_version=ELIGIBILITY_SCHEMA_VERSION,
                visibility="private_hash_pinned",
                rows=len(evaluation.eligibility_rows),
            ),
            "private_components": _artifact_spec(
                payloads["private_components"],
                schema_version=COMPONENT_LEDGER_SCHEMA_VERSION,
                visibility="private_hash_pinned",
                rows=len(evaluation.component_rows),
            ),
            "private_products": _artifact_spec(
                payloads["private_products"],
                schema_version=PRODUCT_LEDGER_SCHEMA_VERSION,
                visibility="private_hash_pinned",
                rows=len(evaluation.product_rows),
            ),
            "public_aggregate": _artifact_spec(
                payloads["public_aggregate"],
                schema_version=PUBLIC_AGGREGATE_SCHEMA_VERSION,
                visibility="public_aggregate_only",
            ),
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": MANIFEST_STATUS,
            "experiment_id": experiment_id,
            "completed_at_utc": _utc_now(),
            "config": {
                "path": _portable(prepared.config_path, repo=prepared.repo),
                "sha256": prepared.config_sha256,
            },
            "binding_validation": binding_result,
            "runtime_preflight": dict(preflight_summary),
            "pinned_inputs": {
                name: {
                    "path": _portable(prepared.input_paths[name], repo=prepared.repo),
                    "sha256": prepared.input_hashes[name],
                }
                for name in sorted(prepared.input_paths)
            },
            "pinned_implementation": {
                name: {
                    "path": _portable(prepared.implementation_paths[name], repo=prepared.repo),
                    "sha256": prepared.implementation_hashes[name],
                }
                for name in sorted(prepared.implementation_paths)
            },
            "sampling": prepared.sampling,
            "software": {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "torch": getattr(torch, "__version__", None),
            },
            "artifacts": {
                ARTIFACT_FILES[name]: specification
                for name, specification in sorted(artifact_specs.items())
            },
            "privacy": {
                "molecular_rendering_disabled": True,
                "identity_bearing_artifacts_private": True,
                "public_result_contains_only_validated_aggregate": True,
            },
        }
        manifest_bytes = _pretty_json_bytes(manifest)
        _atomic_write(prepared.attempt_directory / ARTIFACT_FILES["manifest"], manifest_bytes)
        seal = {
            "schema_version": SEAL_SCHEMA_VERSION,
            "status": SEALED_STATUS,
            "sealed_at_utc": _utc_now(),
            "manifest": {
                "path": ARTIFACT_FILES["manifest"],
                "sha256": _sha256_bytes(manifest_bytes),
            },
            "public_aggregate": {
                "path": ARTIFACT_FILES["public_aggregate"],
                "sha256": artifact_specs["public_aggregate"]["sha256"],
            },
            "one_shot_claim": {
                "path": _portable(prepared.claim_path, repo=prepared.repo),
                "sha256": sha256_file(prepared.claim_path),
            },
            "rendering_performed": False,
            "retry_permitted": False,
            "resampling_permitted": False,
            "biological_guidance_used": False,
        }
        _atomic_write(
            prepared.attempt_directory / ARTIFACT_FILES["seal"],
            _pretty_json_bytes(seal),
        )
        if any(
            path.suffix.lower() in FORBIDDEN_RENDER_SUFFIXES
            for path in prepared.attempt_directory.rglob("*")
        ):
            raise Ugi3BlindedHoldoutExecutionError("molecular rendering artifact was created")
        _fsync_directory(prepared.attempt_directory)
        _rename_directory_noreplace(prepared.attempt_directory, prepared.final_directory)
        _fsync_directory(prepared.final_directory.parent)
    except Exception as exc:
        if prepared.attempt_directory.exists():
            failure = {
                "schema_version": "phase1_ugi3_route_saturation_failed_attempt.v1",
                "status": "failed_attempt_retry_prohibited",
                "failed_at_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "failure_code": "post_claim_execution_failed_private_identities_withheld",
                "one_shot_claim_sha256": sha256_file(prepared.claim_path),
            }
            try:
                _atomic_write(
                    prepared.attempt_directory / "failure.json", _pretty_json_bytes(failure)
                )
            except OSError:
                pass
        raise Ugi3BlindedHoldoutExecutionError(
            "one-shot holdout failed after its irreversible claim; retry is prohibited"
        ) from None

    return {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": SEALED_STATUS,
        "output_directory": _portable(prepared.final_directory, repo=prepared.repo),
        "seal_sha256": sha256_file(prepared.final_directory / ARTIFACT_FILES["seal"]),
        "public_aggregate": evaluation.public_aggregate,
    }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_private_ledger_envelope(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    expected_rows: int,
    fields: tuple[str, ...],
    label: str,
) -> list[Mapping[str, Any]]:
    if set(payload) != {"schema_version", "visibility", "rows"}:
        raise Ugi3BlindedHoldoutExecutionError(f"{label} envelope fields changed")
    if payload.get("schema_version") != schema_version or payload.get("visibility") != (
        "private_hash_pinned"
    ):
        raise Ugi3BlindedHoldoutExecutionError(f"{label} declaration changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise Ugi3BlindedHoldoutExecutionError(f"{label} row count changed")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != set(fields):
            raise Ugi3BlindedHoldoutExecutionError(f"{label} row {index} fields changed")
        if not _is_sha256(row.get("sample_key_sha256")):
            raise Ugi3BlindedHoldoutExecutionError(f"{label} row {index} sample key is malformed")
    return rows


def validate_completed_blinded_holdout(
    repo: Path,
    output_directory: Path,
    *,
    config_path: Path | None = None,
    _registry_loader: (
        Callable[
            [Path, Path],
            tuple[
                Mapping[tuple[str, str], Mapping[str, Any]],
                Mapping[tuple[str, str], Mapping[str, Any]],
            ],
        ]
        | None
    ) = None,
) -> dict[str, Any]:
    """Fully validate a completed private seal and return only its public aggregate."""

    repo = repo.resolve()
    trusted_config_path = (
        config_path
        if config_path is not None
        else repo / "configs/route/phase1_ugi3_route_saturation_blinded_execution_v1.json"
    )
    prepared = prepare_execution(repo, trusted_config_path)
    output_argument = Path(output_directory)
    if output_argument.is_symlink() or not output_argument.is_dir():
        raise Ugi3BlindedHoldoutExecutionError("sealed output is not one regular directory")
    output_directory = output_argument.resolve()
    if prepared.final_directory.resolve() != output_directory:
        raise Ugi3BlindedHoldoutExecutionError(
            "seal is not at the trusted config-owned output path"
        )
    descendants = list(output_directory.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise Ugi3BlindedHoldoutExecutionError("sealed output contains a symbolic link")
    if any(path.suffix.lower() in FORBIDDEN_RENDER_SUFFIXES for path in descendants):
        raise Ugi3BlindedHoldoutExecutionError("sealed output contains a rendering artifact")
    entries = {path.name for path in output_directory.iterdir()}
    if entries != EXPECTED_OUTPUT_FILES or any(
        not (output_directory / name).is_file() for name in EXPECTED_OUTPUT_FILES
    ):
        raise Ugi3BlindedHoldoutExecutionError("sealed output file set changed")

    seal_path = output_directory / ARTIFACT_FILES["seal"]
    seal = _load_json(seal_path, label="execution seal")
    expected_seal_fields = {
        "schema_version",
        "status",
        "sealed_at_utc",
        "manifest",
        "public_aggregate",
        "one_shot_claim",
        "rendering_performed",
        "retry_permitted",
        "resampling_permitted",
        "biological_guidance_used",
    }
    if set(seal) != expected_seal_fields or seal.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise Ugi3BlindedHoldoutExecutionError("execution seal schema changed")
    if seal.get("status") != SEALED_STATUS or {
        "rendering_performed": seal.get("rendering_performed"),
        "retry_permitted": seal.get("retry_permitted"),
        "resampling_permitted": seal.get("resampling_permitted"),
        "biological_guidance_used": seal.get("biological_guidance_used"),
    } != {
        "rendering_performed": False,
        "retry_permitted": False,
        "resampling_permitted": False,
        "biological_guidance_used": False,
    }:
        raise Ugi3BlindedHoldoutExecutionError("execution seal policy changed")
    manifest_spec = seal.get("manifest")
    public_spec = seal.get("public_aggregate")
    if manifest_spec != {
        "path": ARTIFACT_FILES["manifest"],
        "sha256": manifest_spec.get("sha256") if isinstance(manifest_spec, dict) else None,
    } or not _is_sha256(manifest_spec.get("sha256")):
        raise Ugi3BlindedHoldoutExecutionError("execution seal manifest reference is malformed")
    if public_spec != {
        "path": ARTIFACT_FILES["public_aggregate"],
        "sha256": public_spec.get("sha256") if isinstance(public_spec, dict) else None,
    } or not _is_sha256(public_spec.get("sha256")):
        raise Ugi3BlindedHoldoutExecutionError("execution seal public reference is malformed")
    manifest_path = output_directory / ARTIFACT_FILES["manifest"]
    public_path = output_directory / ARTIFACT_FILES["public_aggregate"]
    if sha256_file(manifest_path) != manifest_spec["sha256"]:
        raise Ugi3BlindedHoldoutExecutionError("execution manifest hash changed")
    if sha256_file(public_path) != public_spec["sha256"]:
        raise Ugi3BlindedHoldoutExecutionError("public aggregate hash changed")

    manifest = _load_json(manifest_path, label="execution manifest")
    expected_manifest_fields = {
        "schema_version",
        "status",
        "experiment_id",
        "completed_at_utc",
        "config",
        "binding_validation",
        "runtime_preflight",
        "pinned_inputs",
        "pinned_implementation",
        "sampling",
        "software",
        "artifacts",
        "privacy",
    }
    if set(manifest) != expected_manifest_fields or manifest.get("schema_version") != (
        MANIFEST_SCHEMA_VERSION
    ):
        raise Ugi3BlindedHoldoutExecutionError("execution manifest schema changed")
    if manifest.get("status") != MANIFEST_STATUS:
        raise Ugi3BlindedHoldoutExecutionError("execution manifest status changed")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise Ugi3BlindedHoldoutExecutionError("execution experiment identifier is malformed")
    if manifest.get("privacy") != {
        "molecular_rendering_disabled": True,
        "identity_bearing_artifacts_private": True,
        "public_result_contains_only_validated_aggregate": True,
    }:
        raise Ugi3BlindedHoldoutExecutionError("execution privacy declaration changed")
    if not isinstance(manifest.get("software"), dict) or set(manifest["software"]) != {
        "python",
        "rdkit",
        "torch",
    }:
        raise Ugi3BlindedHoldoutExecutionError("execution software declaration changed")

    config_spec = manifest.get("config")
    if (
        not isinstance(config_spec, dict)
        or set(config_spec) != {"path", "sha256"}
        or not (_is_sha256(config_spec.get("sha256")))
    ):
        raise Ugi3BlindedHoldoutExecutionError("execution config reference is malformed")
    if config_spec != {
        "path": _portable(prepared.config_path, repo=repo),
        "sha256": prepared.config_sha256,
    }:
        raise Ugi3BlindedHoldoutExecutionError("manifest config pin changed")
    expected_inputs = {
        name: {
            "path": _portable(prepared.input_paths[name], repo=repo),
            "sha256": prepared.input_hashes[name],
        }
        for name in sorted(prepared.input_paths)
    }
    expected_implementation = {
        name: {
            "path": _portable(prepared.implementation_paths[name], repo=repo),
            "sha256": prepared.implementation_hashes[name],
        }
        for name in sorted(prepared.implementation_paths)
    }
    if (
        manifest.get("pinned_inputs") != expected_inputs
        or manifest.get("pinned_implementation") != expected_implementation
    ):
        raise Ugi3BlindedHoldoutExecutionError("manifest frozen pins changed")
    if manifest.get("sampling") != prepared.sampling:
        raise Ugi3BlindedHoldoutExecutionError("manifest sampling settings changed")
    runtime_summary = manifest.get("runtime_preflight")
    if not isinstance(runtime_summary, dict) or set(runtime_summary) != set(
        PREFLIGHT_MANIFEST_FIELDS
    ):
        raise Ugi3BlindedHoldoutExecutionError("manifest runtime preflight is unsafe")
    _safe_preflight_summary(runtime_summary, prepared)

    binding_result = manifest.get("binding_validation")
    if not isinstance(binding_result, dict):
        raise Ugi3BlindedHoldoutExecutionError("manifest binding validation is malformed")
    _binding_ready(binding_result, prepared)

    external_claim_path = prepared.claim_path
    if external_claim_path.is_symlink() or not external_claim_path.is_file():
        raise Ugi3BlindedHoldoutExecutionError("external one-shot claim is missing")
    claim_hash = sha256_file(external_claim_path)
    expected_claim_spec = {
        "path": _portable(external_claim_path, repo=repo),
        "sha256": claim_hash,
    }
    if seal.get("one_shot_claim") != expected_claim_spec:
        raise Ugi3BlindedHoldoutExecutionError("execution seal claim reference changed")
    external_claim = _load_json(external_claim_path, label="external one-shot claim")
    internal_claim = _load_json(output_directory / "claim.json", label="internal one-shot claim")
    if external_claim != internal_claim or set(external_claim) != {
        "schema_version",
        "status",
        "claimed_at_utc",
        "config_sha256",
        "binding_sha256",
        "permit_retry",
        "permit_resampling",
    }:
        raise Ugi3BlindedHoldoutExecutionError("one-shot claim copies disagree")
    if external_claim != {
        **external_claim,
        "schema_version": CLAIM_SCHEMA_VERSION,
        "status": CLAIM_STATUS,
        "config_sha256": prepared.config_sha256,
        "binding_sha256": prepared.input_hashes["binding"],
        "permit_retry": False,
        "permit_resampling": False,
    }:
        raise Ugi3BlindedHoldoutExecutionError("one-shot claim declaration changed")

    public = _load_json(public_path, label="public aggregate")
    _validate_public_aggregate_strict(public)
    if public["sealed_program_rows"] != prepared.program_rows:
        raise Ugi3BlindedHoldoutExecutionError("public sealed-program count changed")

    artifacts = manifest.get("artifacts")
    expected_artifact_files = {
        ARTIFACT_FILES[name]
        for name in (
            "binding_validation",
            "private_sample",
            "private_eligibility",
            "private_components",
            "private_products",
            "public_aggregate",
        )
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifact_files:
        raise Ugi3BlindedHoldoutExecutionError("execution artifact manifest changed")
    expected_artifact_declarations = {
        ARTIFACT_FILES["binding_validation"]: (
            BINDING_VALIDATION_SCHEMA_VERSION,
            "private_hash_pinned",
            None,
        ),
        ARTIFACT_FILES["private_sample"]: (
            SAMPLE_SCHEMA_VERSION,
            "private_identity_bearing_hash_pinned",
            prepared.program_rows,
        ),
        ARTIFACT_FILES["private_eligibility"]: (
            ELIGIBILITY_SCHEMA_VERSION,
            "private_hash_pinned",
            prepared.program_rows,
        ),
        ARTIFACT_FILES["private_components"]: (
            COMPONENT_LEDGER_SCHEMA_VERSION,
            "private_hash_pinned",
            len(ROLE_NAMES) * public["exact_l1_eligible_products"],
        ),
        ARTIFACT_FILES["private_products"]: (
            PRODUCT_LEDGER_SCHEMA_VERSION,
            "private_hash_pinned",
            prepared.program_rows,
        ),
        ARTIFACT_FILES["public_aggregate"]: (
            PUBLIC_AGGREGATE_SCHEMA_VERSION,
            "public_aggregate_only",
            None,
        ),
    }
    for name, (schema_version, visibility, rows) in expected_artifact_declarations.items():
        specification = artifacts.get(name)
        expected_keys = {"sha256", "schema_version", "visibility"}
        if rows is not None:
            expected_keys.add("rows")
        if (
            not isinstance(specification, dict)
            or set(specification) != expected_keys
            or not _is_sha256(specification.get("sha256"))
            or specification.get("schema_version") != schema_version
            or specification.get("visibility") != visibility
            or (rows is not None and specification.get("rows") != rows)
        ):
            raise Ugi3BlindedHoldoutExecutionError(f"artifact declaration changed: {name}")
        if sha256_file(output_directory / name) != specification["sha256"]:
            raise Ugi3BlindedHoldoutExecutionError(f"sealed artifact hash changed: {name}")
    if artifacts[ARTIFACT_FILES["public_aggregate"]]["sha256"] != public_spec["sha256"]:
        raise Ugi3BlindedHoldoutExecutionError("public aggregate seals disagree")

    binding_payload = _load_json(
        output_directory / ARTIFACT_FILES["binding_validation"],
        label="binding-validation artifact",
    )
    if binding_payload != {
        "schema_version": BINDING_VALIDATION_SCHEMA_VERSION,
        "status": "validated_before_one_shot_claim",
        "validation": binding_result,
    }:
        raise Ugi3BlindedHoldoutExecutionError("binding-validation artifact changed")
    sample_payload = _load_gzip_json(
        output_directory / ARTIFACT_FILES["private_sample"], label="private sample"
    )
    sample_rows = _validate_private_sample_payload(sample_payload, prepared)
    eligibility_payload = _load_gzip_json(
        output_directory / ARTIFACT_FILES["private_eligibility"],
        label="private eligibility ledger",
    )
    component_payload = _load_gzip_json(
        output_directory / ARTIFACT_FILES["private_components"],
        label="private component ledger",
    )
    product_payload = _load_gzip_json(
        output_directory / ARTIFACT_FILES["private_products"],
        label="private product ledger",
    )
    eligibility_rows = _validate_private_ledger_envelope(
        eligibility_payload,
        schema_version=ELIGIBILITY_SCHEMA_VERSION,
        expected_rows=prepared.program_rows,
        fields=PRIVATE_ELIGIBILITY_FIELDS,
        label="private eligibility ledger",
    )
    component_rows = _validate_private_ledger_envelope(
        component_payload,
        schema_version=COMPONENT_LEDGER_SCHEMA_VERSION,
        expected_rows=len(ROLE_NAMES) * public["exact_l1_eligible_products"],
        fields=PRIVATE_COMPONENT_TRANSITION_FIELDS,
        label="private component ledger",
    )
    product_rows = _validate_private_ledger_envelope(
        product_payload,
        schema_version=PRODUCT_LEDGER_SCHEMA_VERSION,
        expected_rows=prepared.program_rows,
        fields=PRIVATE_PRODUCT_TRANSITION_FIELDS,
        label="private product ledger",
    )
    for index, row in enumerate(eligibility_rows):
        if any(not isinstance(row[field], bool) for field in PRIVATE_ELIGIBILITY_FIELDS[1:]):
            raise Ugi3BlindedHoldoutExecutionError(
                f"private eligibility ledger row {index} has malformed states"
            )
    for index, row in enumerate(component_rows):
        position = row["component_position"]
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position not in range(len(ROLE_NAMES))
            or row["role"] != ROLE_NAMES[position]
            or not _is_sha256(row["component_key_sha256"])
            or not isinstance(row["r0_route_complete"], bool)
            or not isinstance(row["r1_route_complete"], bool)
            or row["transition"]
            not in {"complete_in_both", "noncomplete_to_complete", "noncomplete_in_both"}
        ):
            raise Ugi3BlindedHoldoutExecutionError(
                f"private component ledger row {index} is malformed"
            )
    for index, row in enumerate(product_rows):
        if (
            not isinstance(row["eligible"], bool)
            or not isinstance(row["r0_route_complete"], bool)
            or not isinstance(row["r1_route_complete"], bool)
            or row["transition"]
            not in {
                "ineligible_exact_l1",
                "complete_in_both",
                "noncomplete_to_complete",
                "noncomplete_in_both",
            }
        ):
            raise Ugi3BlindedHoldoutExecutionError(
                f"private product ledger row {index} is malformed"
            )

    registry_loader = _registry_loader or load_registry_pair
    r0_records, r1_records = registry_loader(repo, prepared.input_paths["binding"])
    recomputed = evaluate_registry_pair(
        sample_rows,
        r0_records,
        r1_records,
        experiment_id=experiment_id,
        sampling=prepared.sampling,
        sealed_program_rows=prepared.program_rows,
    )
    if (
        list(recomputed.eligibility_rows) != eligibility_rows
        or list(recomputed.component_rows) != component_rows
        or list(recomputed.product_rows) != product_rows
        or recomputed.public_aggregate != public
    ):
        raise Ugi3BlindedHoldoutExecutionError(
            "private ledgers or public aggregate disagree with exact recomputation"
        )
    return {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "completed_blinded_holdout_valid",
        "output_directory": _portable(output_directory, repo=repo),
        "seal_sha256": sha256_file(seal_path),
        "public_aggregate": public,
    }
