"""Fail-closed preflight and evaluation for the branch-spacing checkpoint challenge."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.io import read_json_object
from forge.data.r0_splits import sha256_file
from forge.design.ugi_morphology_program import maximum_adjacent_branch_graph_run
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES

CANDIDATE_SCHEMA_VERSION = "phase1_ugi_branch_spacing_promotion_candidate.v1"
PREFLIGHT_SCHEMA_VERSION = "phase1_ugi_branch_spacing_checkpoint_promotion_preflight.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_branch_spacing_checkpoint_promotion.v1"
ARCHITECTURE = "full_morphology_program_conditioning"
EXPECTED_STEPS = (1000, 2000)
UNCHANGED_POLICY_BLOCKS = (
    "reference_inverse_gate",
    "candidate_hard_gates",
    "per_role_component_collapse_safeguards",
    "descriptor_distribution_realism_floors",
    "primary_selection_statistic",
    "matched_uncertainty",
    "selection_algorithm",
)


class UgiBranchSpacingPromotionError(RuntimeError):
    """Raised when the promotion experiment diverges from its frozen contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=UgiBranchSpacingPromotionError, label=label)


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() and str(path).startswith("/root/forge_repo/"):
        return repository / path.relative_to("/root/forge_repo")
    return path if path.is_absolute() else repository / path


def _require_hash(repository: Path, specification: Mapping[str, Any], *, label: str) -> Path:
    if set(specification) < {"path", "sha256"}:
        raise UgiBranchSpacingPromotionError(f"{label} has no path/hash identity")
    path = _resolve(repository, str(specification["path"]))
    if not path.is_file():
        raise UgiBranchSpacingPromotionError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != specification["sha256"]:
        raise UgiBranchSpacingPromotionError(
            f"{label} hash mismatch: expected {specification['sha256']}, observed {observed}"
        )
    return path


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_resolved_policy(
    repository: Path,
    path: Path,
    *,
    seen: set[Path] | None = None,
) -> dict[str, Any]:
    resolved_path = path.resolve()
    visited = set() if seen is None else set(seen)
    if resolved_path in visited:
        raise UgiBranchSpacingPromotionError("cyclic promotion-policy inheritance")
    visited.add(resolved_path)
    revision = _load_json(resolved_path, label="selection policy")
    if "base_policy" not in revision:
        return revision
    base_path = _require_hash(repository, revision["base_policy"], label="base policy")
    resolved = _load_resolved_policy(repository, base_path, seen=visited)

    def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        output = dict(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(output.get(key), dict):
                output[key] = merge(output[key], value)
            else:
                output[key] = value
        return output

    overrides = revision.get("overrides")
    if not isinstance(overrides, dict):
        raise UgiBranchSpacingPromotionError("selection-policy revision lacks overrides")
    resolved = merge(resolved, overrides)
    resolved["schema_version"] = revision["schema_version"]
    return resolved


def _validate_candidate_config(
    repository: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _load_json(config_path, label="candidate config")
    if config.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise UgiBranchSpacingPromotionError("unsupported promotion-candidate schema")
    if config.get("status") != "frozen_before_independent_sampling":
        raise UgiBranchSpacingPromotionError("promotion candidate was not frozen before sampling")
    candidate = config.get("candidate", {})
    if candidate.get("architecture") != ARCHITECTURE:
        raise UgiBranchSpacingPromotionError("promotion candidate changed architecture")
    step = int(candidate.get("step", -1))
    if step not in EXPECTED_STEPS:
        raise UgiBranchSpacingPromotionError("promotion candidate changed checkpoint scope")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise UgiBranchSpacingPromotionError("promotion candidate lacks an input manifest")
    resolved_inputs = {
        name: _require_hash(repository, specification, label=f"candidate input {name}")
        for name, specification in inputs.items()
    }
    return config, resolved_inputs


def build_branch_spacing_promotion_preflight(
    *,
    repository: Path,
    v3_policy_path: Path,
    promotion_policy_path: Path,
    candidate_config_paths: Sequence[Path],
) -> dict[str, Any]:
    """Verify the independent experiment before either product sample is opened."""

    if len(candidate_config_paths) != 2:
        raise UgiBranchSpacingPromotionError("promotion requires exactly two candidates")
    v3 = _load_resolved_policy(repository, v3_policy_path)
    promotion = _load_resolved_policy(repository, promotion_policy_path)
    gate_hashes = {}
    for key in UNCHANGED_POLICY_BLOCKS:
        if promotion.get(key) != v3.get(key):
            raise UgiBranchSpacingPromotionError(f"frozen v3 policy block changed: {key}")
        gate_hashes[key] = _canonical_hash(v3[key])
    if promotion["scope"]["architectures"] != [ARCHITECTURE]:
        raise UgiBranchSpacingPromotionError("promotion policy reopens architecture selection")
    if tuple(promotion["scope"]["common_serial_checkpoint_steps"]) != EXPECTED_STEPS:
        raise UgiBranchSpacingPromotionError("promotion policy changed checkpoint scope")

    candidates = []
    for path in candidate_config_paths:
        config, inputs = _validate_candidate_config(repository, path)
        candidates.append((path, config, inputs))
    candidates.sort(key=lambda value: int(value[1]["candidate"]["step"]))
    if tuple(int(value[1]["candidate"]["step"]) for value in candidates) != EXPECTED_STEPS:
        raise UgiBranchSpacingPromotionError("promotion candidates are not steps 1000 and 2000")

    comparator = candidates[0][1]
    for _, config, _ in candidates[1:]:
        if config["sampling"] != comparator["sampling"]:
            raise UgiBranchSpacingPromotionError("candidate sampling contracts differ")
        left_inputs = {
            key: value for key, value in comparator["inputs"].items() if key != "joint_checkpoint"
        }
        right_inputs = {
            key: value for key, value in config["inputs"].items() if key != "joint_checkpoint"
        }
        if left_inputs != right_inputs:
            raise UgiBranchSpacingPromotionError("candidate non-checkpoint inputs differ")
    sampling = comparator["sampling"]
    policy_sampling = promotion["matched_sampling"]
    expected_limits = tuple(policy_sampling["maximum_adjacent_branch_graph_runs_by_role"].values())
    if tuple(sampling["maximum_adjacent_branch_graph_runs_by_role"]) != expected_limits:
        raise UgiBranchSpacingPromotionError("candidate branch-spacing contract changed")
    for key in ("seed", "attempted_draws", "sample_steps", "batch_size"):
        if sampling[key] != policy_sampling[key]:
            raise UgiBranchSpacingPromotionError(f"candidate sampling field differs: {key}")
    if not sampling["evaluate_exact_l1_terminal_admission"]:
        raise UgiBranchSpacingPromotionError("exact L1 terminal admission is disabled")
    if sampling["retry_or_resampling_allowed"]:
        raise UgiBranchSpacingPromotionError("promotion config permits retry or resampling")

    for _, config, _ in candidates:
        output_path = _resolve(repository, config["output"]["result"])
        if output_path.exists():
            raise UgiBranchSpacingPromotionError(
                f"candidate output existed before preflight: {output_path}"
            )
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "pass_frozen_before_candidate_sampling",
        "inputs": {
            "v3_policy": {
                "path": str(v3_policy_path.relative_to(repository)),
                "sha256": sha256_file(v3_policy_path),
            },
            "promotion_policy": {
                "path": str(promotion_policy_path.relative_to(repository)),
                "sha256": sha256_file(promotion_policy_path),
            },
            "candidate_configs": [
                {"path": str(path.relative_to(repository)), "sha256": sha256_file(path)}
                for path, _, _ in candidates
            ],
        },
        "unchanged_v3_policy_block_sha256": gate_hashes,
        "sampling": sampling,
        "candidates": [
            {
                "step": int(config["candidate"]["step"]),
                "joint_checkpoint": config["inputs"]["joint_checkpoint"],
                "output": config["output"],
            }
            for _, config, _ in candidates
        ],
        "outputs_absent_at_preflight": True,
    }


def _branch_contract(sample: Mapping[str, Any], limits: Sequence[int]) -> dict[str, Any]:
    rows = sample.get("samples")
    if not isinstance(rows, list):
        raise UgiBranchSpacingPromotionError("candidate result lacks samples")
    maxima = {role: 0 for role in ROLE_NAMES}
    violation_indices = []
    for index, row in enumerate(rows):
        program = row["program"]
        attachment_counts = tuple(int(value) for value in program["attachment_counts"])
        violated = False
        for role_index, role in enumerate(ROLE_NAMES):
            observed = maximum_adjacent_branch_graph_run(
                tuple(int(value) for value in row["offspring_by_role"][role]),
                attachment_count=attachment_counts[role_index],
            )
            maxima[role] = max(maxima[role], observed)
            violated = violated or observed > int(limits[role_index])
        if violated:
            violation_indices.append(index)
    return {
        "limits_by_role": dict(zip(ROLE_NAMES, (int(value) for value in limits), strict=True)),
        "maximum_observed_by_role": maxima,
        "violation_count": len(violation_indices),
        "violation_indices": violation_indices,
        "status": "pass" if not violation_indices else "fail",
    }


def evaluate_branch_spacing_promotion(
    *,
    repository: Path,
    preflight_path: Path,
    candidate_config_paths: Sequence[Path],
    selection_result_path: Path,
) -> dict[str, Any]:
    """Apply the frozen promotion rule after both one-shot samples exist."""

    preflight = _load_json(preflight_path, label="promotion preflight")
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or preflight.get("status") != "pass_frozen_before_candidate_sampling"
    ):
        raise UgiBranchSpacingPromotionError("promotion preflight did not pass")
    candidates = []
    for path in candidate_config_paths:
        config, _ = _validate_candidate_config(repository, path)
        if not any(
            row["sha256"] == sha256_file(path) for row in preflight["inputs"]["candidate_configs"]
        ):
            raise UgiBranchSpacingPromotionError("candidate config changed after preflight")
        sample_path = _resolve(repository, config["output"]["result"])
        sample = _load_json(sample_path, label="promotion candidate result")
        sampling = config["sampling"]
        if int(sample.get("seed", -1)) != int(sampling["seed"]):
            raise UgiBranchSpacingPromotionError("candidate sampling seed changed")
        if len(sample.get("samples", [])) != int(sampling["attempted_draws"]):
            raise UgiBranchSpacingPromotionError("candidate draw count changed")
        if int(sample.get("sampling", {}).get("sample_steps", -1)) != int(sampling["sample_steps"]):
            raise UgiBranchSpacingPromotionError("candidate flow-step count changed")
        if (
            sample.get("sampling", {}).get("maximum_adjacent_branch_runs")
            != sampling["maximum_adjacent_branch_graph_runs_by_role"]
        ):
            raise UgiBranchSpacingPromotionError("candidate branch limits changed")
        if not sample.get("sampling", {}).get("evaluate_exact_l1_terminal_admission"):
            raise UgiBranchSpacingPromotionError("candidate lacks exact L1 terminal annotation")
        for row in sample["samples"]:
            if bool(row.get("valid")) != bool(row.get("raw_molecule_valid")):
                raise UgiBranchSpacingPromotionError(
                    "terminal admission changed the raw validity denominator"
                )
            if "terminal_valid" not in row:
                raise UgiBranchSpacingPromotionError("candidate row lacks terminal admission state")
        branch = _branch_contract(sample, sampling["maximum_adjacent_branch_graph_runs_by_role"])
        candidates.append((int(config["candidate"]["step"]), sample_path, sample, branch))
    candidates.sort(key=lambda value: value[0])

    selection = _load_json(selection_result_path, label="selection result")
    expected_policy = preflight["inputs"]["promotion_policy"]
    if selection.get("policy", {}).get("sha256") != expected_policy["sha256"]:
        raise UgiBranchSpacingPromotionError("selector used another promotion policy")
    evaluated = selection.get("candidates_by_architecture", {}).get(ARCHITECTURE)
    if not isinstance(evaluated, list) or [int(row["step"]) for row in evaluated] != list(
        EXPECTED_STEPS
    ):
        raise UgiBranchSpacingPromotionError("selector did not evaluate both frozen checkpoints")
    by_step = {int(row["step"]): row for row in evaluated}
    within = selection.get("within_architecture_selection", {}).get(ARCHITECTURE, {})
    expected_selected = f"{ARCHITECTURE}:step_2000"
    requirements = {
        "both_preflights_passed": True,
        "step_2000_passes_every_unchanged_v3_gate": by_step[2000]["gate"]["status"] == "pass",
        "unchanged_within_architecture_selector_selects_step_2000": within.get("selected")
        == expected_selected,
        "zero_branch_contract_violations": all(
            branch["status"] == "pass" for _, _, _, branch in candidates
        ),
    }
    promoted = all(requirements.values())
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "decision": {
            "promote_step_2000": promoted,
            "selected_candidate": expected_selected if promoted else None,
            "retain_frozen_step_1000_production_generator": not promoted,
            "requirements": requirements,
        },
        "inputs": {
            "preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
            "selection_result": {
                "path": str(selection_result_path),
                "sha256": sha256_file(selection_result_path),
            },
            "candidate_results": {
                str(step): {"path": str(path), "sha256": sha256_file(path)}
                for step, path, _, _ in candidates
            },
        },
        "branch_contract": {str(step): branch for step, _, _, branch in candidates},
        "terminal_admission": {
            str(step): {
                "raw_valid": sum(bool(row["raw_molecule_valid"]) for row in sample["samples"]),
                "terminal_valid": sum(bool(row["terminal_valid"]) for row in sample["samples"]),
                "raw_valid_but_terminal_rejected": sum(
                    bool(row["raw_molecule_valid"]) and not bool(row["terminal_valid"])
                    for row in sample["samples"]
                ),
            }
            for step, _, sample, _ in candidates
        },
        "selector": {
            "within_architecture": within,
            "step_1000_gate": by_step[1000]["gate"],
            "step_2000_gate": by_step[2000]["gate"],
        },
        "limitations": [
            "This experiment compares two checkpoints of the already selected full-morphology architecture; it does not reopen architecture selection.",
            "Tail chemotype summaries are descriptive and did not influence promotion.",
            "Exact L1 terminal admission is not a synthesis-success probability and does not assess L2 or L3 closure.",
        ],
    }
