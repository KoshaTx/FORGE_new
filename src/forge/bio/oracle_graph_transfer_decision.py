"""Freeze the scientific disposition of the completed R0 transfer lane.

This adjudicator does not fit or score a model. It reads the completed transfer
matrix and the already-frozen cross-representation selection artifact, verifies
their hashes and contracts, and records whether the transfer lane was selected.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "m0_07_oracle_graph_transfer_decision_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_graph_transfer_decision.v1"
TRANSFER_LANE = "label_free_r0_transfer"
TRANSFER_VARIANTS = frozenset(
    {
        "r0_pretrained_frozen_linear",
        "r0_pretrained_finetuned_dmpnn",
    }
)


class OracleGraphTransferDecisionError(ValueError):
    """Raised when the transfer-lane freeze is stale or scientifically invalid."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleGraphTransferDecisionError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleGraphTransferDecisionError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OracleGraphTransferDecisionError(f"{label} must contain a JSON object")
    return payload


def _verify_input(
    repo_root: Path,
    specification: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = specification.get("path")
    expected_hash = specification.get("sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise OracleGraphTransferDecisionError(f"{label} input specification is incomplete")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise OracleGraphTransferDecisionError(f"{label} path must remain repository-relative")
    path = (repo_root / relative_path).resolve()
    if not path.is_relative_to(repo_root.resolve()) or not path.is_file():
        raise OracleGraphTransferDecisionError(f"{label} not found inside repository: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise OracleGraphTransferDecisionError(
            f"{label} hash mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    return (
        _load_json(path, label),
        {
            "path": relative,
            "sha256": observed_hash,
            "bytes": path.stat().st_size,
        },
    )


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise OracleGraphTransferDecisionError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OracleGraphTransferDecisionError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise OracleGraphTransferDecisionError(f"{label} must be a finite number")
    return number


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleGraphTransferDecisionError("unsupported transfer-decision config schema")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "oracle_freeze_result",
        "transfer_matrix_result",
    }:
        raise OracleGraphTransferDecisionError("transfer-decision inputs changed")
    contract = config.get("contract")
    if not isinstance(contract, dict):
        raise OracleGraphTransferDecisionError("transfer-decision contract is missing")
    if (
        contract.get("lane") != TRANSFER_LANE
        or set(contract.get("variants", ())) != TRANSFER_VARIANTS
        or contract.get("selection_evidence")
        != "equal_endpoint_mean_equal_scheme_calibration_metrics"
        or contract.get("outer_test_used_for_selection") is not False
        or contract.get("fits_may_be_rerun") is not False
    ):
        raise OracleGraphTransferDecisionError("transfer-decision contract changed")


def adjudicate_transfer_lane(
    transfer_result: Mapping[str, Any],
    freeze_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen cross-representation policy without reading raw fits."""

    _validate_config(config)
    expected = config["expected"]
    if (
        transfer_result.get("schema_version") != "m0_07_oracle_graph_transfer_matrix.v1"
        or transfer_result.get("status")
        != "completed_pretrained_oracle_transfer_pending_scientific_freeze"
    ):
        raise OracleGraphTransferDecisionError("transfer matrix is incomplete or incompatible")
    summary = transfer_result.get("summary")
    if not isinstance(summary, dict):
        raise OracleGraphTransferDecisionError("transfer matrix lacks its summary")
    if (
        int(summary.get("fits", -1)) != int(expected["fits"])
        or int(summary.get("ensembles", -1)) != int(expected["ensembles"])
        or set(summary.get("variants", ())) != TRANSFER_VARIANTS
    ):
        raise OracleGraphTransferDecisionError("transfer matrix completeness changed")

    if (
        freeze_result.get("schema_version") != "m0_07_oracle_freeze.v1"
        or freeze_result.get("status") != "oracle_architecture_and_applicability_policy_frozen"
    ):
        raise OracleGraphTransferDecisionError("cross-representation oracle freeze is invalid")
    selection_contract = freeze_result.get("selection_contract")
    if not isinstance(selection_contract, dict):
        raise OracleGraphTransferDecisionError("oracle freeze lacks its selection contract")
    if (
        selection_contract.get("primary_metric")
        != "equal_endpoint_mean_equal_scheme_calibration_r2"
        or selection_contract.get("outer_test_metrics_used_for_architecture_selection") is not False
        or selection_contract.get("random_split_used") is not False
    ):
        raise OracleGraphTransferDecisionError("oracle selection policy changed")
    ranking = freeze_result.get("candidate_ranking")
    selected = freeze_result.get("selected_model")
    if not isinstance(ranking, list) or not ranking or not isinstance(selected, dict):
        raise OracleGraphTransferDecisionError("oracle freeze lacks candidate selection")
    ranked = sorted(ranking, key=lambda row: int(row["selection_rank"]))
    if ranked[0].get("candidate_id") != selected.get("candidate_id"):
        raise OracleGraphTransferDecisionError("selected model is not rank one")
    if selected.get("candidate_id") != expected["selected_candidate_id"]:
        raise OracleGraphTransferDecisionError("selected cross-representation model changed")

    transfer_candidates = [row for row in ranking if row.get("lane") == TRANSFER_LANE]
    if {row.get("representation") for row in transfer_candidates} != TRANSFER_VARIANTS:
        raise OracleGraphTransferDecisionError("transfer candidates are incomplete in the freeze")
    transfer_candidates.sort(key=lambda row: int(row["selection_rank"]))
    selected_transfer = selected.get("lane") == TRANSFER_LANE
    if selected_transfer:
        raise OracleGraphTransferDecisionError(
            "pinned result unexpectedly selects the transfer lane; create a new adjudication"
        )

    transfer_rows = []
    transfer_result_by_variant = {row["variant"]: row for row in summary.get("combined_rows", ())}
    for candidate in transfer_candidates:
        variant = str(candidate["representation"])
        matrix_row = transfer_result_by_variant.get(variant)
        if matrix_row is None:
            raise OracleGraphTransferDecisionError(
                f"transfer result lacks combined row for {variant}"
            )
        test_r2 = _finite_float(
            candidate["equal_endpoint_mean_equal_scheme_test_r2"],
            f"{variant} test R2",
        )
        if not math.isclose(
            test_r2,
            _finite_float(matrix_row["weighted_mean_test_r2"], f"{variant} matrix test R2"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise OracleGraphTransferDecisionError(
                f"{variant} test R2 differs between matrix and freeze"
            )
        transfer_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "variant": variant,
                "selection_rank": int(candidate["selection_rank"]),
                "equal_endpoint_calibration_r2": _finite_float(
                    candidate["equal_endpoint_mean_equal_scheme_calibration_r2"],
                    f"{variant} calibration R2",
                ),
                "equal_endpoint_test_r2": test_r2,
                "equal_endpoint_test_rmse": _finite_float(
                    candidate["equal_endpoint_mean_equal_scheme_test_rmse"],
                    f"{variant} test RMSE",
                ),
            }
        )

    policy = freeze_result.get("applicability_policy")
    if not isinstance(policy, dict):
        raise OracleGraphTransferDecisionError("oracle freeze lacks applicability policy")
    if policy.get("any_guidance_domain_authorized") is not False:
        raise OracleGraphTransferDecisionError(
            "pinned freeze unexpectedly authorizes a guidance domain"
        )
    winner_calibration_r2 = _finite_float(
        selected["equal_endpoint_mean_equal_scheme_calibration_r2"],
        "selected model calibration R2",
    )
    winner_test_r2 = _finite_float(
        selected["equal_endpoint_mean_equal_scheme_test_r2"],
        "selected model test R2",
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "label_free_r0_transfer_frozen_as_nonselected_diagnostic",
        "task": "M0-07 pretrained graph transfer scientific adjudication",
        "selection_contract": {
            "primary_metric": selection_contract["primary_metric"],
            "outer_test_metrics_used_for_selection": False,
            "random_split_used": False,
            "comparison_scope": "all_frozen_classical_supervised_graph_and_transfer_candidates",
        },
        "matrix_completeness": {
            "fits": int(summary["fits"]),
            "ensembles": int(summary["ensembles"]),
            "variants": sorted(TRANSFER_VARIANTS),
            "fits_rerun_for_adjudication": False,
        },
        "selected_cross_representation_model": {
            "candidate_id": selected["candidate_id"],
            "lane": selected["lane"],
            "selection_rank": int(selected["selection_rank"]),
            "equal_endpoint_calibration_r2": winner_calibration_r2,
            "equal_endpoint_test_r2": winner_test_r2,
        },
        "transfer_candidates": transfer_rows,
        "decision": {
            "lane_selected_by_frozen_policy": False,
            "lane_role": "nonselected_diagnostic_pretraining_ablation",
            "guidance_oracle_authorized": False,
            "production_checkpoint_authorized": False,
            "silent_promotion_prohibited": True,
            "negative_outer_test_r2_used_for_selection": False,
            "negative_outer_test_r2_interpretation": (
                "post-selection evidence that the transfer lane did not improve held-component "
                "generalization"
            ),
            "reopen_requires": (
                "a new hash-pinned cross-representation freeze with complete held-component "
                "evidence and an explicit applicability-policy decision"
            ),
        },
    }


def _atomic_write(path: Path, payload: bytes) -> None:
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


def run_transfer_decision(
    config_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify pinned artifacts and write the non-training transfer decision."""

    config = _load_json(config_path, "transfer-decision config")
    _validate_config(config)
    transfer_result, transfer_metadata = _verify_input(
        repo_root,
        config["inputs"]["transfer_matrix_result"],
        "transfer matrix result",
    )
    freeze_result, freeze_metadata = _verify_input(
        repo_root,
        config["inputs"]["oracle_freeze_result"],
        "oracle freeze result",
    )
    result = adjudicate_transfer_lane(transfer_result, freeze_result, config)
    result["configuration"] = {
        "path": str(config_path.resolve().relative_to(repo_root.resolve())),
        "sha256": sha256_file(config_path),
        "bytes": config_path.stat().st_size,
    }
    result["inputs"] = {
        "transfer_matrix_result": transfer_metadata,
        "oracle_freeze_result": freeze_metadata,
    }
    _atomic_write(
        output_path,
        (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return result
