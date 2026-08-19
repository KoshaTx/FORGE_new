"""Nonselecting reuse of the frozen v3 architecture gate for one challenger.

The frozen selector is executed unchanged on the complete ten-candidate matrix,
with exactly one baseline sample replaced by a caller-specified diagnostic
sample.  The selector's decision is discarded.  Only the challenger and its
same-step baseline are retained after the other nine aggregate metric-and-gate
objects have been shown to reproduce exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from forge.data.r0_splits import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_v3_nonselecting_shadow_gate_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_v3_nonselecting_shadow_gate.v1"
CONDITIONING_ARCHITECTURES = {
    "full_morphology": "full_morphology_program_conditioning",
    "size_only": "size_only_conditioning",
}
NON_METRIC_CANDIDATE_FIELDS = {
    "architecture",
    "candidate_id",
    "checkpoint",
    "gate",
    "handle_failure_examples",
    "sampling_result",
    "step",
    "support_violation_examples",
    "usable_open_ended_success_indices",
    "usable_open_ended_vector_sha256",
}
REQUIRED_OUTPUT_CONTRACT = {
    "status": "diagnostic_only_cannot_replace_production",
    "retain_shadow_decision": False,
    "retain_only_challenger_and_same_step_baseline": True,
    "verify_all_unchanged_metric_gate_objects_exactly": True,
}

SelectorRunner = Callable[[list[str], Path, Path], None]


class UgiV3ShadowGateError(RuntimeError):
    """Raised when a shadow evaluation could diverge from the frozen v3 gate."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise UgiV3ShadowGateError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiV3ShadowGateError(f"{label} must be a JSON object")
    return value


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() and str(path).startswith("/root/forge_repo/"):
        return repository / path.relative_to("/root/forge_repo")
    return path if path.is_absolute() else repository / path


def _require_hash(repository: Path, specification: Mapping[str, Any], *, label: str) -> Path:
    if not {"path", "sha256"}.issubset(specification):
        raise UgiV3ShadowGateError(f"{label} has no path/hash identity")
    path = _resolve(repository, str(specification["path"]))
    if not path.is_file():
        raise UgiV3ShadowGateError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != specification["sha256"]:
        raise UgiV3ShadowGateError(
            f"{label} hash mismatch: expected {specification['sha256']}, observed {observed}"
        )
    return path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _candidate_objects(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for architecture, candidates in result["candidates_by_architecture"].items():
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            expected_id = f"{architecture}:step_{int(candidate['step'])}"
            if candidate_id != expected_id or candidate_id in output:
                raise UgiV3ShadowGateError(f"invalid candidate identity: {candidate_id}")
            output[candidate_id] = candidate
    return output


def _object_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _metric_gate_object(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return only aggregate candidate metrics and the unchanged gate object."""

    gate = candidate.get("gate")
    if not isinstance(gate, dict):
        raise UgiV3ShadowGateError("candidate has no gate object")
    return {
        "metrics": {
            key: value for key, value in candidate.items() if key not in NON_METRIC_CANDIDATE_FIELDS
        },
        "gate": gate,
    }


def _default_selector_runner(command: list[str], repository: Path, output_path: Path) -> None:
    environment = dict(os.environ)
    source_path = str(repository / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise UgiV3ShadowGateError(
            "frozen selector failed during shadow evaluation: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    if not output_path.is_file():
        raise UgiV3ShadowGateError("frozen selector did not write its shadow result")


def _validate_challenger(
    repository: Path,
    challenger_config: Mapping[str, Any],
    challenger_result: Mapping[str, Any],
    original_candidates: Mapping[str, Mapping[str, Any]],
) -> str:
    inputs = challenger_config.get("inputs")
    if not isinstance(inputs, dict):
        raise UgiV3ShadowGateError("challenger config has no input manifest")
    resolved_inputs = {
        name: _require_hash(repository, specification, label=f"challenger input {name}")
        for name, specification in inputs.items()
    }
    for required in ("joint_checkpoint", "closure_checkpoint", "program_draw"):
        if required not in resolved_inputs:
            raise UgiV3ShadowGateError(f"challenger config lacks {required}")

    sampling = challenger_config.get("sampling")
    if not isinstance(sampling, dict):
        raise UgiV3ShadowGateError("challenger config has no sampling contract")
    if int(challenger_result.get("seed", -1)) != int(sampling["seed"]):
        raise UgiV3ShadowGateError("challenger sampling seed changed")
    if len(challenger_result.get("samples", [])) != int(sampling["attempted_draws"]):
        raise UgiV3ShadowGateError("challenger attempted-draw count changed")
    result_sampling = challenger_result.get("sampling", {})
    if int(result_sampling.get("sample_steps", -1)) != int(sampling["sample_steps"]):
        raise UgiV3ShadowGateError("challenger sampling-step count changed")
    conditioning_mode = str(result_sampling.get("conditioning_mode"))
    try:
        architecture = CONDITIONING_ARCHITECTURES[conditioning_mode]
    except KeyError as error:
        raise UgiV3ShadowGateError(
            f"unsupported challenger conditioning mode: {conditioning_mode}"
        ) from error

    checkpoint_path = _resolve(repository, challenger_result["checkpoints"]["joint"])
    closure_path = _resolve(repository, challenger_result["checkpoints"]["closure"])
    program_path = _resolve(repository, challenger_result["matched_staged_result"])
    expected_paths = {
        "joint checkpoint": resolved_inputs["joint_checkpoint"],
        "closure checkpoint": resolved_inputs["closure_checkpoint"],
        "program draw": resolved_inputs["program_draw"],
    }
    observed_paths = {
        "joint checkpoint": checkpoint_path,
        "closure checkpoint": closure_path,
        "program draw": program_path,
    }
    for label, expected in expected_paths.items():
        if observed_paths[label].resolve() != expected.resolve():
            raise UgiV3ShadowGateError(f"challenger result used another {label}")

    checkpoint_hash = str(inputs["joint_checkpoint"]["sha256"])
    matches = [
        candidate_id
        for candidate_id, candidate in original_candidates.items()
        if candidate["architecture"] == architecture
        and candidate["checkpoint"]["sha256"] == checkpoint_hash
    ]
    if len(matches) != 1:
        raise UgiV3ShadowGateError(
            "challenger checkpoint does not identify exactly one frozen v3 candidate"
        )
    return matches[0]


def _selector_command(
    *,
    selector_path: Path,
    policy_path: Path,
    training_results: Mapping[str, Path],
    arm_samples: Mapping[str, Mapping[int, Path]],
    output_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(selector_path),
        "--policy",
        str(policy_path),
    ]
    for architecture, path in sorted(training_results.items()):
        command.extend(("--training-result", f"{architecture}:{path}"))
    for architecture, samples in sorted(arm_samples.items()):
        for step, path in sorted(samples.items()):
            command.extend(("--arm-sample", f"{architecture}:{step}:{path}"))
    command.extend(("--output", str(output_path)))
    return command


def evaluate_v3_shadow_gate(
    *,
    contract_path: Path,
    challenger_config_path: Path,
    challenger_config_sha256: str,
    challenger_result_path: Path,
    challenger_result_sha256: str,
    output_path: Path,
    repository: Path,
    selector_runner: SelectorRunner = _default_selector_runner,
) -> dict[str, Any]:
    """Evaluate one diagnostic challenger using the unchanged frozen v3 selector."""

    contract = _load_json(contract_path, label="shadow-gate contract")
    if contract.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiV3ShadowGateError("unsupported shadow-gate contract schema")
    if contract.get("status") != "frozen_baseline_contract":
        raise UgiV3ShadowGateError("shadow-gate baseline contract is not frozen")
    output_contract = contract.get("output_contract", {})
    if any(output_contract.get(key) != value for key, value in REQUIRED_OUTPUT_CONTRACT.items()):
        raise UgiV3ShadowGateError("shadow-gate output contract changed")

    baseline_inputs = {
        name: _require_hash(repository, specification, label=name)
        for name, specification in contract["inputs"].items()
    }
    training_results = {
        architecture: _require_hash(
            repository, specification, label=f"training result {architecture}"
        )
        for architecture, specification in contract["training_results"].items()
    }
    arm_samples = {
        architecture: {
            int(step): _require_hash(
                repository,
                specification,
                label=f"baseline sample {architecture}:{step}",
            )
            for step, specification in samples.items()
        }
        for architecture, samples in contract["baseline_arm_samples"].items()
    }

    challenger_config_spec = {
        "path": str(challenger_config_path),
        "sha256": challenger_config_sha256,
    }
    challenger_result_spec = {
        "path": str(challenger_result_path),
        "sha256": challenger_result_sha256,
    }
    challenger_config_path = _require_hash(
        repository, challenger_config_spec, label="challenger config"
    )
    challenger_result_path = _require_hash(
        repository, challenger_result_spec, label="challenger result"
    )
    challenger_config = _load_json(challenger_config_path, label="challenger config")
    challenger_result = _load_json(challenger_result_path, label="challenger result")
    original_result = _load_json(baseline_inputs["original_v3_result"], label="original v3 result")
    production_manifest = _load_json(
        baseline_inputs["production_manifest"], label="production manifest"
    )
    original_candidates = _candidate_objects(original_result)

    policy_spec = contract["inputs"]["policy"]
    selector_spec = contract["inputs"]["selector"]
    if original_result["policy"]["sha256"] != policy_spec["sha256"]:
        raise UgiV3ShadowGateError("original v3 result used another policy")
    if (
        production_manifest["selection"]["result"]["sha256"]
        != contract["inputs"]["original_v3_result"]["sha256"]
    ):
        raise UgiV3ShadowGateError("production manifest is not anchored to original v3")
    if production_manifest["selection"]["policy"]["sha256"] != policy_spec["sha256"]:
        raise UgiV3ShadowGateError("production manifest used another selection policy")

    for architecture, samples in arm_samples.items():
        for step, path in samples.items():
            candidate_id = f"{architecture}:step_{step}"
            candidate = original_candidates.get(candidate_id)
            if candidate is None:
                raise UgiV3ShadowGateError(f"baseline matrix has no {candidate_id}")
            if candidate["sampling_result"]["sha256"] != sha256_file(path):
                raise UgiV3ShadowGateError(
                    f"baseline sample differs from original v3: {candidate_id}"
                )

    target_id = _validate_challenger(
        repository,
        challenger_config,
        challenger_result,
        original_candidates,
    )
    target_architecture, step_text = target_id.split(":step_", 1)
    target_step = int(step_text)
    arm_samples[target_architecture][target_step] = challenger_result_path

    with tempfile.TemporaryDirectory(prefix="forge-v3-shadow-gate-") as temporary:
        shadow_path = Path(temporary) / "shadow_selector_result.json"
        command = _selector_command(
            selector_path=baseline_inputs["selector"],
            policy_path=baseline_inputs["policy"],
            training_results=training_results,
            arm_samples=arm_samples,
            output_path=shadow_path,
        )
        selector_runner(command, repository, shadow_path)
        shadow_result = _load_json(shadow_path, label="shadow selector result")

    if shadow_result.get("policy", {}).get("sha256") != policy_spec["sha256"]:
        raise UgiV3ShadowGateError("shadow selector output used another policy")
    shadow_candidates = _candidate_objects(shadow_result)
    if set(shadow_candidates) != set(original_candidates):
        raise UgiV3ShadowGateError("shadow selector candidate matrix changed")
    unchanged_ids = sorted(set(original_candidates) - {target_id})
    expected_unchanged = int(contract["expected_unchanged_candidates"])
    if len(unchanged_ids) != expected_unchanged:
        raise UgiV3ShadowGateError("unexpected unchanged-candidate count")
    mismatches = [
        candidate_id
        for candidate_id in unchanged_ids
        if _metric_gate_object(shadow_candidates[candidate_id])
        != _metric_gate_object(original_candidates[candidate_id])
    ]
    if mismatches:
        raise UgiV3ShadowGateError(
            "unchanged v3 metric-and-gate objects failed exact reproduction: " f"{mismatches}"
        )
    challenger_candidate = shadow_candidates[target_id]
    if challenger_candidate["sampling_result"]["sha256"] != challenger_result_sha256:
        raise UgiV3ShadowGateError("shadow selector did not evaluate the challenger result")

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "diagnostic_only_cannot_replace_production",
        "contract": {
            "path": str(contract_path.relative_to(repository)),
            "sha256": sha256_file(contract_path),
        },
        "inputs": {
            **{
                name: {
                    "path": contract["inputs"][name]["path"],
                    "sha256": sha256_file(path),
                }
                for name, path in sorted(baseline_inputs.items())
            },
            "challenger_config": {
                "path": str(challenger_config_path.relative_to(repository)),
                "sha256": challenger_config_sha256,
            },
            "challenger_result": {
                "path": str(challenger_result_path.relative_to(repository)),
                "sha256": challenger_result_sha256,
            },
            "training_results": {
                architecture: {
                    "path": contract["training_results"][architecture]["path"],
                    "sha256": sha256_file(path),
                }
                for architecture, path in sorted(training_results.items())
            },
            "baseline_arm_samples": {
                architecture: {
                    str(step): {
                        "path": contract["baseline_arm_samples"][architecture][str(step)]["path"],
                        "sha256": contract["baseline_arm_samples"][architecture][str(step)][
                            "sha256"
                        ],
                    }
                    for step in sorted(samples)
                }
                for architecture, samples in sorted(arm_samples.items())
            },
        },
        "frozen_evaluator": {
            "path": contract["inputs"]["selector"]["path"],
            "sha256": selector_spec["sha256"],
            "policy_path": contract["inputs"]["policy"]["path"],
            "policy_sha256": policy_spec["sha256"],
            "gate_logic_copied_or_reimplemented": False,
        },
        "target_candidate_id": target_id,
        "unchanged_candidate_reproduction": {
            "status": "pass",
            "expected_candidates": expected_unchanged,
            "exactly_reproduced_candidates": len(unchanged_ids),
            "metric_gate_object_sha256": {
                candidate_id: _object_hash(_metric_gate_object(original_candidates[candidate_id]))
                for candidate_id in unchanged_ids
            },
        },
        "same_step_comparison": {
            "original_v3_baseline": _metric_gate_object(original_candidates[target_id]),
            "challenger": _metric_gate_object(challenger_candidate),
        },
        "shadow_decision": {
            "retained": False,
            "usable": False,
            "reason": (
                "The frozen selector's decision is deliberately discarded. This diagnostic uses "
                "a previously inspected program draw and cannot replace the production generator."
            ),
        },
        "claim_boundary": {
            "may_report": (
                "The challenger passed or failed the unchanged v3 candidate gates on the "
                "previously inspected diagnostic draw."
            ),
            "may_not_report": [
                "a new production-generator selection",
                "an independent or pristine model-selection result",
                "route closure, synthesis success or biological activity",
            ],
            "new_independent_selection_draw_required_before_replacement": True,
        },
    }
    _atomic_json(output_path, result)
    return result
