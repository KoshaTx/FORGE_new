"""Audit measured joint Ugi morphology-program support without guidance.

The bounded structural mode defined here is a distribution over complete
``UgiMorphologyProgram`` tuples.  It never composes precursor-role marginals
independently, never selects component identities, and never assigns a
biological or synthesis utility.  Open mode is an exact identity operation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.product.ugi_morphology_program import UgiMorphologyProgram
from forge.product.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
    decode_canonical_morphology_program_bytes,
)
from forge.product.ugi_training_cache import load_ugi_training_cache

CONFIG_SCHEMA_VERSION = "phase1_ugi_joint_program_support_audit_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi_joint_program_support_audit.v2"
LEDGER_SCHEMA_VERSION = "phase1_ugi_joint_program_support_ledger.v2"
SUPERSESSION_SCHEMA_VERSION = "phase1_ugi_bounded_structural_envelope_supersession.v1"

ROLE_ORDER = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
PROGRAM_COORDINATES = tuple(
    f"{field}.{role}"
    for field in (
        "node_count",
        "junction_budget",
        "cycle_rank",
        "attachment_count",
    )
    for role in ROLE_ORDER
)
EXPECTED_SCOPE = {
    "read_only": True,
    "biological_targets_read": False,
    "whole_programs_sampled_jointly": True,
    "role_programs_sampled_independently": False,
    "component_ids_enter_program_state": False,
    "component_graphs_remain_generated": True,
    "all_three_new_action_unchanged": "abstain",
    "oracle_calls": 0,
    "biological_guidance": False,
    "synthesis_calls": 0,
    "synthesis_guidance": False,
    "proposal_calls": 0,
    "generator_trajectories_advanced": False,
    "candidate_selection": False,
    "prospective_candidate_lock": False,
    "nonzero_guidance_execution": False,
}
EXPECTED_INPUTS = {
    "applicability_census_ledger",
    "applicability_census_result",
    "fresh_pool_sample",
    "frozen_program_schedule",
    "prepared_cache",
    "prepared_cache_result",
    "runner",
    "source",
    "superseded_envelope",
    "superseded_envelope_census",
    "tests",
}
LEDGER_FIELDS = (
    "program_sha256",
    "occurrences",
    "probability",
    "node_counts_json",
    "junction_budgets_json",
    "cycle_ranks_json",
    "attachment_counts_json",
)


class UgiJointProgramSupportAuditError(RuntimeError):
    """Raised when the measured joint-program audit violates its contract."""


@dataclass(frozen=True)
class EmpiricalJointProgramPrior:
    """Occurrence-weighted support over complete Ugi morphology programs."""

    programs: tuple[UgiMorphologyProgram, ...]
    counts: tuple[int, ...]
    probabilities: np.ndarray


@dataclass(frozen=True)
class JointProgramKernel:
    """Robust joint-space neighborhood around complete measured programs."""

    prior: EmpiricalJointProgramPrior
    scales: np.ndarray
    bandwidth: float
    local_radius: float


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiJointProgramSupportAuditError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiJointProgramSupportAuditError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiJointProgramSupportAuditError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiJointProgramSupportAuditError(f"{label} path escapes repository") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiJointProgramSupportAuditError(f"{label} hash changed")
    return path


def _program_from_mapping(value: Any) -> UgiMorphologyProgram:
    if not isinstance(value, Mapping):
        raise UgiJointProgramSupportAuditError("morphology program must be an object")
    required = {
        "node_counts",
        "junction_budgets",
        "cycle_ranks",
        "attachment_counts",
    }
    if set(value) != required:
        raise UgiJointProgramSupportAuditError("morphology program fields changed")
    try:
        program = UgiMorphologyProgram(
            node_counts=tuple(int(item) for item in value["node_counts"]),
            junction_budgets=tuple(int(item) for item in value["junction_budgets"]),
            cycle_ranks=tuple(int(item) for item in value["cycle_ranks"]),
            attachment_counts=tuple(int(item) for item in value["attachment_counts"]),
        )
    except (TypeError, ValueError) as error:
        raise UgiJointProgramSupportAuditError("invalid morphology program values") from error
    if not all(
        len(field) == len(ROLE_ORDER)
        for field in (
            program.node_counts,
            program.junction_budgets,
            program.cycle_ranks,
            program.attachment_counts,
        )
    ):
        raise UgiJointProgramSupportAuditError("morphology program role count changed")
    if (
        any(item < 1 for item in program.node_counts)
        or any(item < 0 for item in program.junction_budgets)
        or any(item < 0 for item in program.cycle_ranks)
        or any(item < 1 for item in program.attachment_counts)
    ):
        raise UgiJointProgramSupportAuditError("morphology program has invalid counts")
    return program


def _program_mapping(value: UgiMorphologyProgram) -> dict[str, list[int]]:
    return {
        "node_counts": list(value.node_counts),
        "junction_budgets": list(value.junction_budgets),
        "cycle_ranks": list(value.cycle_ranks),
        "attachment_counts": list(value.attachment_counts),
    }


def build_empirical_joint_program_prior(
    programs: Sequence[UgiMorphologyProgram],
) -> EmpiricalJointProgramPrior:
    """Aggregate complete program tuples without constructing role marginals."""

    if not programs:
        raise UgiJointProgramSupportAuditError("joint-program prior is empty")
    by_payload: Counter[bytes] = Counter()
    for program in programs:
        by_payload[canonical_morphology_program_bytes(program)] += 1
    payloads = tuple(sorted(by_payload))
    ordered = tuple(decode_canonical_morphology_program_bytes(payload) for payload in payloads)
    counts = tuple(by_payload[payload] for payload in payloads)
    total = sum(counts)
    probabilities = np.asarray(counts, dtype=np.float64) / total
    if (
        len(ordered) != len(counts)
        or np.any(probabilities <= 0.0)
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise UgiJointProgramSupportAuditError("joint-program prior is invalid")
    return EmpiricalJointProgramPrior(
        programs=ordered,
        counts=counts,
        probabilities=probabilities,
    )


def sample_empirical_joint_program_prior(
    prior: EmpiricalJointProgramPrior,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[UgiMorphologyProgram, ...]:
    """Sample whole observed tuples; precursor roles are never recombined."""

    if count < 1 or not prior.programs:
        raise UgiJointProgramSupportAuditError("invalid joint-program sample request")
    if prior.probabilities.shape != (len(prior.programs),):
        raise UgiJointProgramSupportAuditError("joint-program probability shape changed")
    indices = rng.choice(len(prior.programs), size=count, p=prior.probabilities)
    return tuple(prior.programs[int(index)] for index in indices)


def _program_vector(program: UgiMorphologyProgram) -> np.ndarray:
    return np.asarray(
        (
            *program.node_counts,
            *program.junction_budgets,
            *program.cycle_ranks,
            *program.attachment_counts,
        ),
        dtype=np.float64,
    )


def build_joint_program_kernel(
    programs: Sequence[UgiMorphologyProgram],
) -> JointProgramKernel:
    """Calibrate a small joint neighborhood without biological labels.

    Feature scales are occurrence-weighted interquartile ranges with an integer
    floor of one. The local radius and Laplace bandwidth are the higher 95th
    percentile of leave-one-unique-program-out nearest-neighbor distances.
    This preserves complete-program correlations instead of defining a
    feature-wise Cartesian box.
    """

    prior = build_empirical_joint_program_prior(programs)
    observed = np.stack([_program_vector(program) for program in programs])
    scales = np.maximum(
        np.quantile(observed, 0.75, axis=0) - np.quantile(observed, 0.25, axis=0),
        1.0,
    )
    unique = np.stack([_program_vector(program) for program in prior.programs])
    if unique.shape[0] < 2:
        raise UgiJointProgramSupportAuditError(
            "joint-program kernel requires at least two unique programs"
        )
    pairwise = np.mean(
        np.abs(unique[:, None, :] - unique[None, :, :]) / scales,
        axis=2,
    )
    np.fill_diagonal(pairwise, np.inf)
    nearest = pairwise.min(axis=1)
    radius = float(np.quantile(nearest, 0.95, method="higher"))
    if not np.isfinite(radius) or radius <= 0.0:
        raise UgiJointProgramSupportAuditError("joint-program kernel radius is invalid")
    return JointProgramKernel(
        prior=prior,
        scales=scales,
        bandwidth=radius,
        local_radius=radius,
    )


def joint_program_distances(
    kernel: JointProgramKernel,
    programs: Sequence[UgiMorphologyProgram],
) -> np.ndarray:
    """Return robust mean-L1 distance to the nearest complete anchor tuple."""

    if not programs:
        raise UgiJointProgramSupportAuditError("joint-program distance request is empty")
    anchors = np.stack([_program_vector(item) for item in kernel.prior.programs])
    queries = np.stack([_program_vector(item) for item in programs])
    output = np.empty(queries.shape[0], dtype=np.float64)
    for start in range(0, queries.shape[0], 256):
        batch = queries[start : start + 256]
        distances = np.mean(
            np.abs(batch[:, None, :] - anchors[None, :, :]) / kernel.scales,
            axis=2,
        )
        output[start : start + batch.shape[0]] = distances.min(axis=1)
    return output


def joint_program_distance(
    kernel: JointProgramKernel,
    program: UgiMorphologyProgram,
) -> float:
    """Return one nearest-anchor distance."""

    return float(joint_program_distances(kernel, (program,))[0])


def joint_program_affinities(
    kernel: JointProgramKernel,
    programs: Sequence[UgiMorphologyProgram],
) -> np.ndarray:
    """Return occurrence-weighted Laplace-kernel affinities in joint space."""

    if not programs:
        raise UgiJointProgramSupportAuditError("joint-program affinity request is empty")
    anchors = np.stack([_program_vector(item) for item in kernel.prior.programs])
    queries = np.stack([_program_vector(item) for item in programs])
    output = np.empty(queries.shape[0], dtype=np.float64)
    for start in range(0, queries.shape[0], 256):
        batch = queries[start : start + 256]
        distances = np.mean(
            np.abs(batch[:, None, :] - anchors[None, :, :]) / kernel.scales,
            axis=2,
        )
        output[start : start + batch.shape[0]] = (
            np.exp(-distances / kernel.bandwidth) @ kernel.prior.probabilities
        )
    if np.any(~np.isfinite(output)) or np.any(output <= 0.0) or np.any(output > 1.0):
        raise UgiJointProgramSupportAuditError("joint-program affinity is invalid")
    return output


def joint_program_affinity(
    kernel: JointProgramKernel,
    program: UgiMorphologyProgram,
) -> float:
    """Return one occurrence-weighted joint-program affinity."""

    return float(joint_program_affinities(kernel, (program,))[0])


def joint_program_neighborhood(
    kernel: JointProgramKernel,
    program: UgiMorphologyProgram,
) -> str:
    """Classify exact anchors, calibrated local neighbors and outside programs."""

    support = {canonical_morphology_program_bytes(item) for item in kernel.prior.programs}
    if canonical_morphology_program_bytes(program) in support:
        return "exact_anchor"
    if joint_program_distance(kernel, program) <= kernel.local_radius + 1e-12:
        return "local_smoothed_neighborhood"
    return "outside_local_neighborhood"


def joint_candidate_pool_probabilities(
    kernel: JointProgramKernel,
    programs: Sequence[UgiMorphologyProgram],
) -> np.ndarray:
    """Normalize affinities over a complete-program exact-plus-local pool."""

    if not programs:
        raise UgiJointProgramSupportAuditError("joint candidate pool is empty")
    neighborhoods = [joint_program_neighborhood(kernel, program) for program in programs]
    if any(value == "outside_local_neighborhood" for value in neighborhoods):
        raise UgiJointProgramSupportAuditError(
            "bounded joint candidate pool contains an outside-neighborhood program"
        )
    affinities = joint_program_affinities(kernel, programs)
    probabilities = affinities / affinities.sum()
    if np.any(probabilities <= 0.0) or not np.isclose(probabilities.sum(), 1.0):
        raise UgiJointProgramSupportAuditError("joint candidate probabilities are invalid")
    return probabilities


def evaluate_program_mode(
    kernel: JointProgramKernel,
    program: UgiMorphologyProgram,
    *,
    mode: str,
) -> dict[str, Any]:
    """Return schedule support while keeping every guidance multiplier neutral."""

    if mode not in {"open", "bounded_joint_program"}:
        raise UgiJointProgramSupportAuditError(f"unsupported program mode: {mode}")
    neighborhood = joint_program_neighborhood(kernel, program)
    return {
        "mode": mode,
        "neighborhood": neighborhood,
        "exact_anchor_membership": neighborhood == "exact_anchor",
        "schedule_admitted": (
            True if mode == "open" else neighborhood != "outside_local_neighborhood"
        ),
        "joint_distance": joint_program_distance(kernel, program),
        "joint_affinity": joint_program_affinity(kernel, program),
        "identity_multiplier": 1.0,
        "incremental_potential": 0.0,
        "biological_guidance_active": False,
        "synthesis_guidance_active": False,
    }


def _effective_count(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total < 1:
        return 0.0
    return 1.0 / sum((count / total) ** 2 for count in counts)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size < 1 or np.any(~np.isfinite(values)):
        raise UgiJointProgramSupportAuditError("invalid joint-program distance values")
    return {
        "minimum": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _ledger_bytes(prior: EmpiricalJointProgramPrior) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=LEDGER_FIELDS, lineterminator="\n")
    writer.writeheader()
    total = sum(prior.counts)
    for program, count in zip(prior.programs, prior.counts, strict=True):
        payload = canonical_morphology_program_bytes(program)
        writer.writerow(
            {
                "program_sha256": hashlib.sha256(payload).hexdigest(),
                "occurrences": count,
                "probability": format(count / total, ".17g"),
                "node_counts_json": _stable_json(list(program.node_counts)),
                "junction_budgets_json": _stable_json(list(program.junction_budgets)),
                "cycle_ranks_json": _stable_json(list(program.cycle_ranks)),
                "attachment_counts_json": _stable_json(list(program.attachment_counts)),
            }
        )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as handle:
        handle.write(output.getvalue().encode())
    return buffer.getvalue()


def _load_measured_programs(
    cache_path: Path,
) -> tuple[list[UgiMorphologyProgram], Counter[str]]:
    corpus, records_by_fold = load_ugi_training_cache(cache_path)
    measured: list[UgiMorphologyProgram] = []
    fold_counts: Counter[str] = Counter()
    product_ids: set[str] = set()
    if set(records_by_fold) != set(corpus.assignments_by_fold):
        raise UgiJointProgramSupportAuditError("cache folds are misaligned")
    for fold in sorted(records_by_fold):
        records = records_by_fold[fold]
        assignments = corpus.assignments_by_fold[fold]
        if len(records) != len(assignments):
            raise UgiJointProgramSupportAuditError(f"cache fold is misaligned: {fold}")
        for assignment, record in zip(assignments, records, strict=True):
            if str(assignment.get("is_source_adjudicated_measured_product", "")).lower() != "true":
                continue
            if assignment.get("product_id") != record.product_id:
                raise UgiJointProgramSupportAuditError("measured product ID is misaligned")
            if record.product_id in product_ids:
                raise UgiJointProgramSupportAuditError("measured product ID is duplicated")
            product_ids.add(record.product_id)
            measured.append(record.program)
            fold_counts[fold] += 1
    if len(measured) != 1_100:
        raise UgiJointProgramSupportAuditError(
            f"measured AGILE program count changed: {len(measured)}"
        )
    return measured, fold_counts


def _load_schedule(path: Path) -> tuple[list[str], list[UgiMorphologyProgram]]:
    value = _load_json(path, label="frozen program schedule")
    if value.get("schema_version") != "phase1_ugi_program_probe.v1":
        raise UgiJointProgramSupportAuditError("frozen schedule schema changed")
    rows = value.get("samples")
    if not isinstance(rows, list) or len(rows) != 4_096:
        raise UgiJointProgramSupportAuditError("frozen schedule count changed")
    identifiers: list[str] = []
    programs: list[UgiMorphologyProgram] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise UgiJointProgramSupportAuditError("frozen schedule row is malformed")
        identifiers.append(str(row.get("product_id")))
        programs.append(_program_from_mapping(row.get("program")))
    if len(set(identifiers)) != len(identifiers):
        raise UgiJointProgramSupportAuditError("frozen schedule product IDs are duplicated")
    return identifiers, programs


def _valid_terminal_ids(
    sample_path: Path,
    schedule: Mapping[str, bytes],
) -> set[str]:
    value = _load_json(sample_path, label="fresh terminal pool")
    if value.get("schema_version") != "phase1_ugi_joint_end_to_end_sampling.v1":
        raise UgiJointProgramSupportAuditError("fresh terminal schema changed")
    rows = value.get("samples")
    if not isinstance(rows, list) or len(rows) != len(schedule):
        raise UgiJointProgramSupportAuditError("fresh terminal count changed")
    valid: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise UgiJointProgramSupportAuditError("fresh terminal row is malformed")
        product_id = str(row.get("product_id"))
        if product_id in seen or product_id not in schedule:
            raise UgiJointProgramSupportAuditError("fresh terminal product IDs changed")
        seen.add(product_id)
        program = _program_from_mapping(row.get("program"))
        if canonical_morphology_program_bytes(program) != schedule[product_id]:
            raise UgiJointProgramSupportAuditError("fresh terminal program changed")
        forward = row.get("l1_forward_verification")
        exact_forward = (
            isinstance(forward, Mapping) and forward.get("exact_product_reconstructed") is True
        )
        is_valid = (
            row.get("valid") is True
            and row.get("raw_molecule_valid") is True
            and row.get("terminal_valid") is True
            and row.get("component_reconstruction_valid") is True
            and exact_forward
        )
        if is_valid:
            valid.add(product_id)
    if seen != set(schedule) or len(valid) != 3_975:
        raise UgiJointProgramSupportAuditError(
            f"fresh terminal support changed: {len(seen)} rows, {len(valid)} valid"
        )
    return valid


def _applicability_rows(path: Path, valid_ids: set[str]) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiJointProgramSupportAuditError("invalid applicability census ledger") from error
    required = {
        "product_id",
        "current_policy_action",
        "current_policy_reason",
        "exact_unseen_roles_json",
    }
    if not rows or not required.issubset(rows[0]):
        raise UgiJointProgramSupportAuditError("applicability census fields changed")
    ids = [row["product_id"] for row in rows]
    if len(ids) != 3_975 or len(set(ids)) != len(ids) or set(ids) != valid_ids:
        raise UgiJointProgramSupportAuditError("applicability census IDs changed")
    return rows


def build_joint_program_support_audit(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Build the measured joint-program and frozen-schedule support audit."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="joint-program audit config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "frozen_before_joint_program_support_audit"
        or config.get("scope") != EXPECTED_SCOPE
        or set(config.get("inputs", {})) != EXPECTED_INPUTS
    ):
        raise UgiJointProgramSupportAuditError("joint-program audit config changed")
    paths = {label: _pin(repo, record, label=label) for label, record in config["inputs"].items()}
    cache_result = _load_json(paths["prepared_cache_result"], label="prepared cache result")
    cache_artifact = cache_result.get("artifact")
    if (
        cache_result.get("schema_version") != "phase1_ugi_training_cache_result.v1"
        or not isinstance(cache_artifact, Mapping)
        or cache_artifact.get("sha256") != config["inputs"]["prepared_cache"]["sha256"]
    ):
        raise UgiJointProgramSupportAuditError("prepared cache result does not bind cache")

    measured_programs, fold_counts = _load_measured_programs(paths["prepared_cache"])
    kernel = build_joint_program_kernel(measured_programs)
    prior = kernel.prior
    support_payloads = {canonical_morphology_program_bytes(program) for program in prior.programs}
    identifiers, scheduled_programs = _load_schedule(paths["frozen_program_schedule"])
    schedule = {
        product_id: canonical_morphology_program_bytes(program)
        for product_id, program in zip(identifiers, scheduled_programs, strict=True)
    }
    valid_ids = _valid_terminal_ids(paths["fresh_pool_sample"], schedule)
    applicability = _applicability_rows(paths["applicability_census_ledger"], valid_ids)

    schedule_payloads = list(schedule.values())
    unique_schedule = set(schedule_payloads)
    overlap = unique_schedule & support_payloads
    measured_counts = {
        canonical_morphology_program_bytes(program): count
        for program, count in zip(prior.programs, prior.counts, strict=True)
    }
    schedule_distances = joint_program_distances(kernel, scheduled_programs)
    schedule_neighborhoods = [
        (
            "exact_anchor"
            if payload in support_payloads
            else (
                "local_smoothed_neighborhood"
                if distance <= kernel.local_radius + 1e-12
                else "outside_local_neighborhood"
            )
        )
        for payload, distance in zip(schedule_payloads, schedule_distances, strict=True)
    ]
    neighborhood_by_id = dict(zip(identifiers, schedule_neighborhoods, strict=True))
    schedule_neighborhood_counts = Counter(schedule_neighborhoods)
    valid_neighborhood_counts = Counter(neighborhood_by_id[product_id] for product_id in valid_ids)
    action_cross_tab: dict[str, Counter[str]] = {
        neighborhood: Counter()
        for neighborhood in (
            "exact_anchor",
            "local_smoothed_neighborhood",
            "outside_local_neighborhood",
        )
    }
    reason_cross_tab: dict[str, Counter[str]] = {
        neighborhood: Counter() for neighborhood in action_cross_tab
    }
    for row in applicability:
        group = neighborhood_by_id[row["product_id"]]
        action_cross_tab[group][row["current_policy_action"]] += 1
        reason_cross_tab[group][row["current_policy_reason"]] += 1

    active_action = "applicability_supported_no_score"
    all_three_reason = "unsupported_all_three_new"
    active_total = sum(row["current_policy_action"] == active_action for row in applicability)
    active_by_neighborhood = Counter(
        neighborhood_by_id[row["product_id"]]
        for row in applicability
        if row["current_policy_action"] == active_action
    )
    all_three_total = sum(row["current_policy_reason"] == all_three_reason for row in applicability)
    all_three_by_neighborhood = Counter(
        neighborhood_by_id[row["product_id"]]
        for row in applicability
        if row["current_policy_reason"] == all_three_reason
    )
    bounded_neighborhood_active = sum(
        row["current_policy_action"] == active_action
        and neighborhood_by_id[row["product_id"]] in {"exact_anchor", "local_smoothed_neighborhood"}
        for row in applicability
    )
    bounded_neighborhood_all_three = sum(
        row["current_policy_reason"] == all_three_reason
        and neighborhood_by_id[row["product_id"]] in {"exact_anchor", "local_smoothed_neighborhood"}
        for row in applicability
    )
    if active_total != 47 or all_three_total != 2_134:
        raise UgiJointProgramSupportAuditError("frozen applicability policy counts changed")

    envelope = _load_json(paths["superseded_envelope"], label="superseded envelope")
    envelope_census = _load_json(
        paths["superseded_envelope_census"], label="superseded envelope census"
    )
    if (
        envelope.get("reference", {}).get("measured_rows") != 1_100
        or envelope.get("reference", {}).get("measured_rows_inside_all_views") != 640
        or envelope_census.get("schema_version")
        != "phase1_ugi_bounded_structural_envelope_census.v1"
    ):
        raise UgiJointProgramSupportAuditError("superseded envelope facts changed")

    ledger = _ledger_bytes(prior)
    program_frequencies = Counter(prior.counts)
    local_program = next(
        program
        for program, neighborhood in zip(scheduled_programs, schedule_neighborhoods, strict=True)
        if neighborhood == "local_smoothed_neighborhood"
    )
    outside_program = next(
        program
        for program, neighborhood in zip(scheduled_programs, schedule_neighborhoods, strict=True)
        if neighborhood == "outside_local_neighborhood"
    )
    open_example = evaluate_program_mode(kernel, outside_program, mode="open")
    bounded_exact_example = evaluate_program_mode(
        kernel, prior.programs[0], mode="bounded_joint_program"
    )
    bounded_local_example = evaluate_program_mode(
        kernel, local_program, mode="bounded_joint_program"
    )
    bounded_outside_example = evaluate_program_mode(
        kernel, outside_program, mode="bounded_joint_program"
    )
    if (
        open_example["identity_multiplier"] != 1.0
        or open_example["incremental_potential"] != 0.0
        or not open_example["schedule_admitted"]
    ):
        raise UgiJointProgramSupportAuditError("open program mode is not identity")

    inputs = {
        label: {
            "path": str(path.relative_to(repo)),
            "sha256": config["inputs"][label]["sha256"],
        }
        for label, path in sorted(paths.items())
    }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_joint_measured_program_audit_nonzero_blocked",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": inputs,
        "scope": dict(EXPECTED_SCOPE),
        "program_definition": {
            "role_order": list(ROLE_ORDER),
            "fields": [
                "node_counts",
                "junction_budgets",
                "cycle_ranks",
                "attachment_counts",
            ],
            "sampling_unit": "complete_joint_ugi_morphology_program",
            "role_marginal_recombination": False,
            "component_identity_conditioning": False,
            "atom_and_bond_generation_unchanged": True,
            "primary_bounded_support": "joint_anchor_kernel_with_local_smoothing",
            "exact_tuple_membership_role": "strict_diagnostic_ablation",
            "hard_biological_gate": "unchanged_v3_multiview_chemical_applicability",
        },
        "measured_joint_prior": {
            "measured_rows": len(measured_programs),
            "fold_counts": dict(sorted(fold_counts.items())),
            "unique_joint_programs": len(prior.programs),
            "effective_joint_program_count_inverse_simpson": _effective_count(prior.counts),
            "largest_program_occurrences": max(prior.counts),
            "largest_program_fraction": max(prior.counts) / len(measured_programs),
            "program_occurrence_frequency": {
                str(count): programs for count, programs in sorted(program_frequencies.items())
            },
            "measured_rows_inside_support": len(measured_programs),
            "measured_rows_outside_support": 0,
        },
        "joint_kernel_calibration": {
            "distance": "mean absolute coordinate difference divided by weighted-IQR scale",
            "coordinate_order": list(PROGRAM_COORDINATES),
            "coordinate_scales": {
                name: float(scale)
                for name, scale in zip(PROGRAM_COORDINATES, kernel.scales, strict=True)
            },
            "scale_floor": 1.0,
            "bandwidth": kernel.bandwidth,
            "local_radius": kernel.local_radius,
            "radius_rule": (
                "higher 95th percentile of leave-one-unique-program-out nearest-anchor distances"
            ),
            "kernel": "occurrence-weighted Laplace mixture over complete measured programs",
            "biological_labels_used": False,
            "hard_biological_applicability_gate": False,
        },
        "frozen_schedule_coverage": {
            "scheduled_attempts": len(schedule_payloads),
            "neighborhood_counts": dict(sorted(schedule_neighborhood_counts.items())),
            "exact_anchor_fraction": schedule_neighborhood_counts["exact_anchor"]
            / len(schedule_payloads),
            "exact_plus_local_fraction": (
                schedule_neighborhood_counts["exact_anchor"]
                + schedule_neighborhood_counts["local_smoothed_neighborhood"]
            )
            / len(schedule_payloads),
            "nearest_anchor_distance_quantiles": _quantiles(schedule_distances),
            "unique_scheduled_programs": len(unique_schedule),
            "strict_exact_anchor_unique_overlap": len(overlap),
            "measured_joint_program_type_recall": len(overlap) / len(support_payloads),
            "measured_occurrence_mass_represented_by_schedule": sum(
                measured_counts[payload] for payload in overlap
            )
            / len(measured_programs),
            "interpretation": (
                "The frozen schedule was drawn from independent role marginals and is diagnostic "
                "only. The exact-anchor count is a strict ablation; the primary bounded structural "
                "prior also preserves calibrated nearby unseen whole programs."
            ),
        },
        "valid_terminal_cross_tab": {
            "valid_exact_l1_terminals": len(valid_ids),
            "neighborhood_counts": dict(sorted(valid_neighborhood_counts.items())),
            "current_policy_actions": {
                group: dict(sorted(counts.items()))
                for group, counts in sorted(action_cross_tab.items())
            },
            "current_policy_reasons": {
                group: dict(sorted(counts.items()))
                for group, counts in sorted(reason_cross_tab.items())
            },
            "active_contrast_overall": active_total,
            "active_contrast_by_neighborhood": dict(sorted(active_by_neighborhood.items())),
            "active_contrast_exact_plus_local": bounded_neighborhood_active,
            "all_three_new_overall": all_three_total,
            "all_three_new_by_neighborhood": dict(sorted(all_three_by_neighborhood.items())),
            "all_three_new_exact_plus_local": bounded_neighborhood_all_three,
            "all_three_new_action": "abstain",
        },
        "mode_contract": {
            "open_mode": open_example,
            "bounded_exact_example": bounded_exact_example,
            "bounded_local_example": bounded_local_example,
            "bounded_outside_example": bounded_outside_example,
            "morphology_mode_is_not_biological_applicability": True,
            "nonzero_guidance_authorized": False,
        },
        "supersession": {
            "superseded_artifact": "phase1_ugi_bounded_structural_envelope.v1",
            "reason": (
                "The feature-wise Cartesian envelope excluded 460 of 1,100 measured rows and "
                "did not preserve the empirical joint morphology distribution."
            ),
            "historical_artifacts_retained_write_once": True,
            "new_primary_bounded_structure": "empirical_joint_anchor_kernel_with_local_smoothing",
        },
        "adjudication": {
            "joint_anchor_kernel_schedule_justified": True,
            "exact_tuple_only_primary_bound_rejected": True,
            "exact_tuple_only_retained_as_diagnostic_ablation": True,
            "nearby_unobserved_joint_programs_preserved": True,
            "existing_independent_role_schedule_suitable_for_bounded_run": False,
            "new_joint_schedule_required": True,
            "terminal_chemistry_still_requires_v3_applicability": True,
            "all_three_new_still_abstained": True,
            "oracle_or_synthesis_utility_evaluated": False,
            "nonzero_guidance_authorized": False,
            "next_gate": (
                "freeze a whole-program schedule from the empirical anchor kernel over exact and "
                "calibrated-local tuples, then run lambda-zero identity before any bounded pilot"
            ),
        },
        "artifacts": {
            "program_ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(prior.programs),
                "sha256": sha256_bytes(ledger),
            }
        },
    }
    result["result_sha256"] = _sha256_payload(result)
    supersession = {
        "schema_version": SUPERSESSION_SCHEMA_VERSION,
        "status": "superseded_but_retained",
        "superseded": {
            "envelope": inputs["superseded_envelope"],
            "census": inputs["superseded_envelope_census"],
            "measured_rows": 1_100,
            "measured_rows_inside_all_views": 640,
            "measured_rows_excluded": 460,
        },
        "replacement": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "logical_sha256": result["result_sha256"],
            "definition": (
                "occurrence-weighted empirical kernel over complete programs with a calibrated "
                "local neighborhood"
            ),
            "measured_rows_inside_support": 1_100,
            "measured_rows_outside_support": 0,
        },
        "policy": {
            "old_artifacts_deleted_or_rewritten": False,
            "feature_envelope_remains_diagnostic_only": True,
            "exact_tuple_membership_is_diagnostic_only": True,
            "joint_anchor_kernel_is_primary_bounded_structural_prior": True,
            "v3_chemical_applicability_remains_hard_biological_gate": True,
            "open_mode_changed": False,
            "nonzero_guidance_authorized": False,
        },
    }
    supersession["supersession_sha256"] = _sha256_payload(supersession)
    return result, ledger, supersession


__all__ = [
    "EmpiricalJointProgramPrior",
    "JointProgramKernel",
    "UgiJointProgramSupportAuditError",
    "build_empirical_joint_program_prior",
    "build_joint_program_kernel",
    "build_joint_program_support_audit",
    "evaluate_program_mode",
    "joint_candidate_pool_probabilities",
    "joint_program_affinities",
    "joint_program_affinity",
    "joint_program_distance",
    "joint_program_distances",
    "joint_program_neighborhood",
    "sample_empirical_joint_program_prior",
]
