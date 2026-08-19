"""Freeze endpoint-specific oracle selection without reopening the M0 matrix.

The M0-07 artifact selected one architecture using equal HeLa and RAW 264.7
weights. This module preserves that audit and derives a campaign-specific
ranking from the same candidates and the same calibration folds. Outer-test
metrics are copied only after selection for transparent descriptive reporting.
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

CONFIG_SCHEMA_VERSION = "phase1_oracle_campaign_selection_config.v1"
RESULT_SCHEMA_VERSION = "phase1_oracle_campaign_selection.v1"
FREEZE_SCHEMA_VERSION = "m0_07_oracle_freeze.v1"
FREEZE_STATUS = "oracle_architecture_and_applicability_policy_frozen"


class OracleCampaignSelectionError(ValueError):
    """Raised when endpoint-specific selection violates its frozen contract."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleCampaignSelectionError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleCampaignSelectionError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OracleCampaignSelectionError(f"{label} must contain a JSON object")
    return payload


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise OracleCampaignSelectionError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OracleCampaignSelectionError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise OracleCampaignSelectionError(f"{label} must be a finite number")
    return number


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleCampaignSelectionError("unsupported campaign-selection config schema")

    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"oracle_freeze_result"}:
        raise OracleCampaignSelectionError("campaign-selection inputs changed")
    input_spec = inputs["oracle_freeze_result"]
    if (
        not isinstance(input_spec, dict)
        or input_spec.get("schema_version") != FREEZE_SCHEMA_VERSION
        or input_spec.get("status") != FREEZE_STATUS
    ):
        raise OracleCampaignSelectionError("oracle freeze input contract changed")

    campaign = config.get("campaign")
    if not isinstance(campaign, dict):
        raise OracleCampaignSelectionError("campaign contract is missing")
    endpoint = campaign.get("endpoint")
    excluded = campaign.get("excluded_endpoints")
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or not isinstance(excluded, list)
        or endpoint in excluded
        or len(excluded) != len(set(excluded))
    ):
        raise OracleCampaignSelectionError("campaign endpoint contract is invalid")

    selection = config.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("candidate_universe") != "all_candidates_in_the_hash_pinned_m0_07_freeze"
        or selection.get("primary_metric") != "equal_scheme_mean_calibration_r2"
        or selection.get("tie_breakers")
        != [
            "lower_equal_scheme_mean_calibration_rmse",
            "higher_equal_scheme_mean_calibration_spearman_rho",
            "lexical_candidate_id",
        ]
        or selection.get("outer_test_metrics_used_for_selection") is not False
        or selection.get("outer_test_role") != "reporting_only_after_endpoint_specific_selection"
        or selection.get("refit_or_new_hyperparameter_search_allowed") is not False
    ):
        raise OracleCampaignSelectionError("endpoint-specific selection contract changed")

    immutability = config.get("immutability")
    if (
        not isinstance(immutability, dict)
        or immutability.get("m0_equal_endpoint_freeze_rewritten") is not False
    ):
        raise OracleCampaignSelectionError("the immutable M0 freeze cannot be rewritten")

    guidance = config.get("guidance")
    if not isinstance(guidance, dict) or guidance.get("authorized") is not False:
        raise OracleCampaignSelectionError(
            "campaign selection cannot silently authorize biological guidance"
        )


def _endpoint_row(candidate: Mapping[str, Any], endpoint: str) -> Mapping[str, Any]:
    endpoint_rows = candidate.get("endpoint_rows")
    if not isinstance(endpoint_rows, list):
        raise OracleCampaignSelectionError(
            f"{candidate.get('candidate_id', '<unknown>')} lacks endpoint rows"
        )
    matches = [
        row
        for row in endpoint_rows
        if isinstance(row, dict) and str(row.get("endpoint")) == endpoint
    ]
    if len(matches) != 1:
        raise OracleCampaignSelectionError(
            f"{candidate.get('candidate_id', '<unknown>')} must have exactly one {endpoint} row"
        )
    return matches[0]


def select_campaign_oracle(
    freeze_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Rank the frozen candidate universe using one endpoint's calibration evidence."""

    _validate_config(config)
    input_spec = config["inputs"]["oracle_freeze_result"]
    if (
        freeze_result.get("schema_version") != input_spec["schema_version"]
        or freeze_result.get("status") != input_spec["status"]
    ):
        raise OracleCampaignSelectionError("oracle freeze is incomplete or incompatible")

    freeze_contract = freeze_result.get("selection_contract")
    if not isinstance(freeze_contract, dict):
        raise OracleCampaignSelectionError("oracle freeze lacks its selection contract")
    if (
        freeze_contract.get("outer_test_metrics_used_for_architecture_selection") is not False
        or freeze_contract.get("random_split_used") is not False
    ):
        raise OracleCampaignSelectionError("upstream M0 selection used prohibited evidence")
    configured_schemes = config["selection"]["eligible_schemes"]
    if configured_schemes != freeze_contract.get("eligible_schemes"):
        raise OracleCampaignSelectionError("eligible scheme set differs from the frozen matrix")

    ranking = freeze_result.get("candidate_ranking")
    if not isinstance(ranking, list) or not ranking:
        raise OracleCampaignSelectionError("oracle freeze has no candidate ranking")
    candidate_ids = [row.get("candidate_id") for row in ranking if isinstance(row, dict)]
    if (
        len(candidate_ids) != len(ranking)
        or any(not isinstance(candidate_id, str) for candidate_id in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise OracleCampaignSelectionError("frozen candidate universe is invalid")

    endpoint = str(config["campaign"]["endpoint"])
    excluded_endpoints = set(config["campaign"]["excluded_endpoints"])
    frozen_endpoints = set(freeze_contract.get("endpoints", {}))
    if endpoint not in frozen_endpoints:
        raise OracleCampaignSelectionError(f"{endpoint} is absent from the frozen matrix")
    if excluded_endpoints != frozen_endpoints - {endpoint}:
        raise OracleCampaignSelectionError(
            "every noncampaign endpoint must be explicitly excluded from selection"
        )

    ranked: list[dict[str, Any]] = []
    for candidate in ranking:
        endpoint_metrics = _endpoint_row(candidate, endpoint)
        row = {
            "candidate_id": str(candidate["candidate_id"]),
            "lane": str(candidate["lane"]),
            "representation": str(candidate["representation"]),
            "model": str(candidate["model"]),
            "endpoint": endpoint,
            "calibration_r2": _finite_float(
                endpoint_metrics.get("equal_scheme_mean_calibration_r2"),
                f"{candidate['candidate_id']} calibration R2",
            ),
            "calibration_rmse": _finite_float(
                endpoint_metrics.get("equal_scheme_mean_calibration_rmse"),
                f"{candidate['candidate_id']} calibration RMSE",
            ),
            "calibration_spearman_rho": _finite_float(
                endpoint_metrics.get("equal_scheme_mean_calibration_spearman_rho"),
                f"{candidate['candidate_id']} calibration Spearman rho",
            ),
            "post_selection_test_r2": _finite_float(
                endpoint_metrics.get("equal_scheme_mean_test_r2"),
                f"{candidate['candidate_id']} test R2",
            ),
            "post_selection_test_rmse": _finite_float(
                endpoint_metrics.get("equal_scheme_mean_test_rmse"),
                f"{candidate['candidate_id']} test RMSE",
            ),
            "post_selection_test_spearman_rho": _finite_float(
                endpoint_metrics.get("equal_scheme_mean_test_spearman_rho"),
                f"{candidate['candidate_id']} test Spearman rho",
            ),
            "post_selection_absolute_90pct_coverage_gap": _finite_float(
                endpoint_metrics.get("equal_scheme_mean_absolute_90pct_coverage_gap"),
                f"{candidate['candidate_id']} coverage gap",
            ),
        }
        ranked.append(row)

    # Only the three calibration fields and a lexical final tie-breaker enter this key.
    # Test and coverage fields are deliberately absent.
    ranked.sort(
        key=lambda row: (
            -row["calibration_r2"],
            row["calibration_rmse"],
            -row["calibration_spearman_rho"],
            row["candidate_id"],
        )
    )
    for index, row in enumerate(ranked, start=1):
        row["campaign_selection_rank"] = index

    selected = ranked[0]
    shared_selected = freeze_result.get("selected_model")
    if not isinstance(shared_selected, dict):
        raise OracleCampaignSelectionError("oracle freeze lacks its M0 selected model")

    applicability = freeze_result.get("applicability_policy")
    if (
        not isinstance(applicability, dict)
        or applicability.get("any_guidance_domain_authorized") is not False
    ):
        raise OracleCampaignSelectionError(
            "this campaign artifact is only valid while every frozen domain abstains"
        )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "hela_campaign_oracle_selected_but_guidance_still_abstained",
        "task": str(config["task"]),
        "campaign": dict(config["campaign"]),
        "selection_contract": {
            "candidate_universe": config["selection"]["candidate_universe"],
            "eligible_schemes": list(configured_schemes),
            "selection_endpoint": endpoint,
            "excluded_endpoints": sorted(excluded_endpoints),
            "primary_metric": config["selection"]["primary_metric"],
            "tie_breakers": list(config["selection"]["tie_breakers"]),
            "outer_test_metrics_used_for_selection": False,
            "outer_test_role": config["selection"]["outer_test_role"],
            "models_refit_for_selection": False,
        },
        "selected_model": dict(selected),
        "candidate_ranking": ranked,
        "comparison_with_immutable_m0_freeze": {
            "m0_shared_candidate_id": str(shared_selected["candidate_id"]),
            "campaign_candidate_id": str(selected["candidate_id"]),
            "same_candidate": selected["candidate_id"] == shared_selected["candidate_id"],
            "m0_freeze_rewritten": False,
        },
        "guidance_policy": {
            "authorized": False,
            "reason": config["guidance"]["reason"],
            "reopen_requires": config["guidance"]["reopen_requires"],
        },
        "interpretation": {
            "primary_endpoint": "HeLa alone drives this campaign's model selection",
            "raw_264_7_role": "excluded; requires an independent future campaign freeze",
            "test_metrics": "descriptive post-selection evidence only",
        },
    }


def _resolve_pinned_input(
    repo_root: Path,
    specification: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    relative = specification.get("path")
    expected_hash = specification.get("sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise OracleCampaignSelectionError("oracle freeze input specification is incomplete")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise OracleCampaignSelectionError("oracle freeze path must remain repository-relative")
    path = (repo_root / relative_path).resolve()
    if not path.is_relative_to(repo_root.resolve()) or not path.is_file():
        raise OracleCampaignSelectionError(f"oracle freeze not found inside repository: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise OracleCampaignSelectionError(
            f"oracle freeze hash mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    return path, {
        "path": relative,
        "sha256": observed_hash,
        "bytes": path.stat().st_size,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def run_campaign_selection(
    config_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Load pinned inputs, select the endpoint-specific model, and persist the result."""

    config_path = config_path.resolve()
    repo_root = repo_root.resolve()
    config = _load_json(config_path, "campaign-selection config")
    _validate_config(config)
    freeze_path, freeze_metadata = _resolve_pinned_input(
        repo_root,
        config["inputs"]["oracle_freeze_result"],
    )
    freeze_result = _load_json(freeze_path, "oracle freeze result")
    result = select_campaign_oracle(freeze_result, config)
    result["inputs"] = {
        "config": {
            "path": str(config_path.relative_to(repo_root)),
            "sha256": sha256_file(config_path),
            "bytes": config_path.stat().st_size,
        },
        "oracle_freeze_result": freeze_metadata,
    }
    result["seed"] = int(config["seed"])
    _atomic_write_json(output_path.resolve(), result)
    return result
