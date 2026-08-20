"""Freeze the M0-07 oracle architecture and applicability policy.

The selection contract is intentionally separate from model fitting. It
compares every completed representation lane under the same scaffold and
held-component schemes, selects one representation and model across both
AGILE endpoints, and converts the selected model's scheme-level evidence into
explicit guidance or abstention decisions.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.core.io import atomic_write as _atomic_write
from forge.core.io import read_json_object

CONFIG_SCHEMA_VERSION = "m0_07_oracle_freeze_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_freeze.v1"
REQUIRED_METRIC_FIELDS = frozenset(
    {
        "representation",
        "model",
        "endpoint",
        "scheme",
        "fold",
        "calibration_r2",
        "calibration_rmse",
        "calibration_spearman_rho",
        "test_r2",
        "test_rmse",
        "test_spearman_rho",
        "test_coverage90",
    }
)


class OracleFreezeError(ValueError):
    """Raised when scientific oracle selection violates its frozen contract."""


def _stable_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(path, error=OracleFreezeError, label=label)


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OracleFreezeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise OracleFreezeError(f"{label} must be finite")
    return number


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise OracleFreezeError(f"{label} has no values")
    value = sum(values) / len(values)
    if not math.isfinite(value):
        raise OracleFreezeError(f"{label} mean is not finite")
    return value


def _collect_artifact_hashes(value: Any) -> set[str]:
    hashes: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "sha256" and isinstance(nested, str):
                hashes.add(nested)
            else:
                hashes.update(_collect_artifact_hashes(nested))
    elif isinstance(value, list):
        for nested in value:
            hashes.update(_collect_artifact_hashes(nested))
    return hashes


def _read_metric_rows(path: Path, lane: str) -> list[dict[str, Any]]:
    try:
        handle = gzip.open(path, "rt", newline="", encoding="utf-8")
    except FileNotFoundError as exc:
        raise OracleFreezeError(f"{lane} metrics not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        fieldnames = frozenset(reader.fieldnames or ())
        missing = REQUIRED_METRIC_FIELDS - fieldnames
        if missing:
            raise OracleFreezeError(f"{lane} metrics lack fields: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for row_index, source in enumerate(reader, start=2):
            rows.append(
                {
                    "lane": lane,
                    "representation": str(source["representation"]),
                    "model": str(source["model"]),
                    "endpoint": str(source["endpoint"]),
                    "scheme": str(source["scheme"]),
                    "fold": int(source["fold"]),
                    "calibration_r2": _finite_float(
                        source["calibration_r2"],
                        f"{lane} row {row_index} calibration_r2",
                    ),
                    "calibration_rmse": _finite_float(
                        source["calibration_rmse"],
                        f"{lane} row {row_index} calibration_rmse",
                    ),
                    "calibration_spearman_rho": _finite_float(
                        source["calibration_spearman_rho"],
                        f"{lane} row {row_index} calibration_spearman_rho",
                    ),
                    "test_r2": _finite_float(source["test_r2"], f"{lane} row {row_index} test_r2"),
                    "test_rmse": _finite_float(
                        source["test_rmse"], f"{lane} row {row_index} test_rmse"
                    ),
                    "test_spearman_rho": _finite_float(
                        source["test_spearman_rho"],
                        f"{lane} row {row_index} test_spearman_rho",
                    ),
                    "test_coverage90": _finite_float(
                        source["test_coverage90"],
                        f"{lane} row {row_index} test_coverage90",
                    ),
                }
            )
    if not rows:
        raise OracleFreezeError(f"{lane} metrics are empty")
    return rows


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleFreezeError(
            f"unsupported freeze config schema: {config.get('schema_version')!r}"
        )
    if config.get("seed") != 1729:
        raise OracleFreezeError("freeze seed changed")
    inputs = config.get("inputs")
    selection = config.get("selection")
    gate = config.get("guidance_gate")
    domains = config.get("applicability_domains")
    unsupported = config.get("unsupported_domains")
    if not all(
        isinstance(value, Mapping) for value in (inputs, selection, gate, domains, unsupported)
    ):
        raise OracleFreezeError("freeze config sections are incomplete")
    if len(inputs) != 3:
        raise OracleFreezeError("exactly three predeclared representation lanes are required")
    endpoints = selection.get("endpoints")
    schemes = selection.get("eligible_schemes")
    if not isinstance(endpoints, Mapping) or set(endpoints) != {"expt_Hela", "expt_Raw"}:
        raise OracleFreezeError("both AGILE endpoints must be equally adjudicated")
    weights = [_finite_float(value, "endpoint weight") for value in endpoints.values()]
    if any(value <= 0 for value in weights) or not math.isclose(sum(weights), 1.0):
        raise OracleFreezeError("endpoint weights must be positive and sum to one")
    if not isinstance(schemes, list) or len(schemes) != 7 or len(set(schemes)) != 7:
        raise OracleFreezeError("the seven selection-eligible schemes must be unique")
    if "lantern_random" in schemes or selection.get("random_split_used") is not False:
        raise OracleFreezeError("random-split evidence cannot select the oracle")
    if selection.get("one_representation_and_model_across_endpoints") is not True:
        raise OracleFreezeError("one representation and model must serve both endpoints")
    if selection.get("primary_metric") != "equal_endpoint_mean_equal_scheme_calibration_r2":
        raise OracleFreezeError("primary selection metric changed")
    expected_ties = [
        "lower_equal_endpoint_mean_equal_scheme_calibration_rmse",
        "higher_equal_endpoint_mean_equal_scheme_calibration_spearman_rho",
        "lexical_candidate_id",
    ]
    if selection.get("tie_breakers") != expected_ties:
        raise OracleFreezeError("selection tie breakers changed")
    if (
        selection.get("outer_test_metrics_used_for_architecture_selection") is not False
        or selection.get("outer_test_role")
        != "post_selection_evaluation_and_predeclared_domain_gate"
    ):
        raise OracleFreezeError("outer-test role changed")
    required_gate_keys = {
        "minimum_scheme_mean_test_r2",
        "minimum_scheme_mean_test_spearman_rho",
        "minimum_positive_r2_fold_fraction",
        "maximum_scheme_mean_absolute_90pct_coverage_gap",
    }
    if not required_gate_keys.issubset(gate):
        raise OracleFreezeError("guidance gate thresholds are incomplete")
    for key in required_gate_keys:
        _finite_float(gate[key], key)
    if gate.get("failed_gate_action") != "abstain":
        raise OracleFreezeError("failed applicability gates must abstain")
    eligible = set(schemes)
    for domain, required in domains.items():
        if not isinstance(required, list) or not required:
            raise OracleFreezeError(f"{domain} must require at least one scheme")
        if not set(required).issubset(eligible):
            raise OracleFreezeError(f"{domain} uses an ineligible evaluation scheme")
    if "three_unseen_components" not in unsupported or "in_vivo_endpoint" not in unsupported:
        raise OracleFreezeError("mandatory unsupported domains are missing")


def _load_lanes(
    config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, Any]] = {}
    for lane, specification in sorted(config["inputs"].items()):
        if not isinstance(specification, Mapping):
            raise OracleFreezeError(f"{lane} input specification is invalid")
        result_relative = specification.get("result_path")
        metrics_relative = specification.get("metrics_path")
        expected_schema = specification.get("result_schema")
        expected_status = specification.get("result_status")
        expected_candidates = specification.get("expected_candidates")
        if not all(
            isinstance(value, str)
            for value in (
                result_relative,
                metrics_relative,
                expected_schema,
                expected_status,
            )
        ):
            raise OracleFreezeError(f"{lane} input paths or schema are incomplete")
        if (
            not isinstance(expected_candidates, list)
            or not expected_candidates
            or any(
                not isinstance(candidate, list)
                or len(candidate) != 2
                or not all(isinstance(value, str) for value in candidate)
                for candidate in expected_candidates
            )
        ):
            raise OracleFreezeError(f"{lane} expected candidate roster is invalid")
        result_path = repo_root / result_relative
        metrics_path = repo_root / metrics_relative
        result = _load_json(result_path, f"{lane} result")
        if result.get("schema_version") != expected_schema:
            raise OracleFreezeError(f"{lane} result schema changed")
        status = str(result.get("status", ""))
        if status != expected_status:
            raise OracleFreezeError(f"{lane} result is not complete: {status!r}")
        result_hash = sha256_file(result_path)
        metrics_hash = sha256_file(metrics_path)
        if metrics_hash not in _collect_artifact_hashes(result.get("artifacts", {})):
            raise OracleFreezeError(f"{lane} metrics hash is not pinned by its result")
        lane_rows = _read_metric_rows(metrics_path, lane)
        observed_candidates = {(str(row["representation"]), str(row["model"])) for row in lane_rows}
        required_candidates = {
            (str(candidate[0]), str(candidate[1])) for candidate in expected_candidates
        }
        if observed_candidates != required_candidates:
            raise OracleFreezeError(
                f"{lane} candidate roster is {sorted(observed_candidates)}, "
                f"expected {sorted(required_candidates)}"
            )
        rows.extend(lane_rows)
        inputs[lane] = {
            "result": {
                "path": str(result_relative),
                "sha256": result_hash,
                "bytes": result_path.stat().st_size,
                "schema_version": expected_schema,
                "status": status,
            },
            "metrics": {
                "path": str(metrics_relative),
                "sha256": metrics_hash,
                "bytes": metrics_path.stat().st_size,
                "rows": len(lane_rows),
            },
        }
    return rows, inputs


def _candidate_id(row: Mapping[str, Any]) -> str:
    return "::".join(
        (
            str(row["lane"]),
            str(row["representation"]),
            str(row["model"]),
        )
    )


def _expected_folds(scheme: str) -> set[int]:
    """Return the frozen fold identifiers for one selection scheme."""

    if scheme == "lantern_scaffold_balanced":
        return {0}
    return set(range(5))


def aggregate_candidates(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Aggregate folds, schemes, and endpoints under the frozen contract."""

    selection = config["selection"]
    eligible_schemes = set(selection["eligible_schemes"])
    endpoint_weights = {
        str(endpoint): _finite_float(weight, f"{endpoint} weight")
        for endpoint, weight in selection["endpoints"].items()
    }
    filtered = [
        dict(row)
        for row in rows
        if str(row["scheme"]) in eligible_schemes and str(row["endpoint"]) in endpoint_weights
    ]
    if any(str(row["scheme"]) == "lantern_random" for row in filtered):
        raise OracleFreezeError("random split entered oracle selection")
    by_candidate_scheme: defaultdict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    identities: dict[str, dict[str, str]] = {}
    for row in filtered:
        candidate = _candidate_id(row)
        identities[candidate] = {
            "lane": str(row["lane"]),
            "representation": str(row["representation"]),
            "model": str(row["model"]),
        }
        key = (candidate, str(row["endpoint"]), str(row["scheme"]))
        by_candidate_scheme[key].append(row)
    scheme_metrics: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    seen_fold_keys: set[tuple[str, str, str, int]] = set()
    for (candidate, endpoint, scheme), group in sorted(by_candidate_scheme.items()):
        observed_folds: set[int] = set()
        for row in group:
            fold = int(row["fold"])
            fold_key = (candidate, endpoint, scheme, fold)
            if fold_key in seen_fold_keys:
                raise OracleFreezeError(f"duplicate metric fold: {fold_key}")
            seen_fold_keys.add(fold_key)
            observed_folds.add(fold)
        expected_folds = _expected_folds(scheme)
        if observed_folds != expected_folds:
            raise OracleFreezeError(
                f"{candidate}/{endpoint}/{scheme} has folds "
                f"{sorted(observed_folds)}, expected {sorted(expected_folds)}"
            )
        calibration_r2_values = [
            _finite_float(row["calibration_r2"], "calibration_r2") for row in group
        ]
        calibration_rmse_values = [
            _finite_float(row["calibration_rmse"], "calibration_rmse") for row in group
        ]
        calibration_spearman_values = [
            _finite_float(
                row["calibration_spearman_rho"],
                "calibration_spearman_rho",
            )
            for row in group
        ]
        r2_values = [_finite_float(row["test_r2"], "test_r2") for row in group]
        rmse_values = [_finite_float(row["test_rmse"], "test_rmse") for row in group]
        spearman_values = [
            _finite_float(row["test_spearman_rho"], "test_spearman_rho") for row in group
        ]
        coverage_gaps = [
            abs(_finite_float(row["test_coverage90"], "test_coverage90") - 0.9) for row in group
        ]
        scheme_metrics[candidate][endpoint][scheme] = {
            "folds": len(group),
            "mean_calibration_r2": _mean(
                calibration_r2_values,
                "scheme calibration_r2",
            ),
            "mean_calibration_rmse": _mean(
                calibration_rmse_values,
                "scheme calibration_rmse",
            ),
            "mean_calibration_spearman_rho": _mean(
                calibration_spearman_values,
                "scheme calibration_spearman_rho",
            ),
            "mean_test_r2": _mean(r2_values, "scheme test_r2"),
            "mean_test_rmse": _mean(rmse_values, "scheme test_rmse"),
            "mean_test_spearman_rho": _mean(spearman_values, "scheme test_spearman_rho"),
            "mean_absolute_90pct_coverage_gap": _mean(coverage_gaps, "scheme coverage gap"),
            "positive_r2_fold_fraction": sum(value > 0 for value in r2_values) / len(r2_values),
        }
    candidates: list[dict[str, Any]] = []
    for candidate, identity in sorted(identities.items()):
        observed_endpoints = set(scheme_metrics[candidate])
        if observed_endpoints != set(endpoint_weights):
            raise OracleFreezeError(
                f"{candidate} has endpoints {sorted(observed_endpoints)}, "
                f"expected {sorted(endpoint_weights)}"
            )
        endpoint_rows: list[dict[str, Any]] = []
        for endpoint in sorted(endpoint_weights):
            observed_schemes = set(scheme_metrics[candidate][endpoint])
            if observed_schemes != eligible_schemes:
                raise OracleFreezeError(
                    f"{candidate}/{endpoint} has schemes {sorted(observed_schemes)}, "
                    f"expected {sorted(eligible_schemes)}"
                )
            values = list(scheme_metrics[candidate][endpoint].values())
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "equal_scheme_mean_calibration_r2": _mean(
                        [value["mean_calibration_r2"] for value in values],
                        "endpoint calibration_r2",
                    ),
                    "equal_scheme_mean_calibration_rmse": _mean(
                        [value["mean_calibration_rmse"] for value in values],
                        "endpoint calibration_rmse",
                    ),
                    "equal_scheme_mean_calibration_spearman_rho": _mean(
                        [value["mean_calibration_spearman_rho"] for value in values],
                        "endpoint calibration_spearman_rho",
                    ),
                    "equal_scheme_mean_test_r2": _mean(
                        [value["mean_test_r2"] for value in values],
                        "endpoint test_r2",
                    ),
                    "equal_scheme_mean_test_rmse": _mean(
                        [value["mean_test_rmse"] for value in values],
                        "endpoint test_rmse",
                    ),
                    "equal_scheme_mean_absolute_90pct_coverage_gap": _mean(
                        [value["mean_absolute_90pct_coverage_gap"] for value in values],
                        "endpoint coverage gap",
                    ),
                    "equal_scheme_mean_test_spearman_rho": _mean(
                        [value["mean_test_spearman_rho"] for value in values],
                        "endpoint test_spearman_rho",
                    ),
                }
            )
        endpoint_by_name = {row["endpoint"]: row for row in endpoint_rows}
        candidates.append(
            {
                "candidate_id": candidate,
                **identity,
                "equal_endpoint_mean_equal_scheme_calibration_r2": sum(
                    endpoint_weights[endpoint]
                    * endpoint_by_name[endpoint]["equal_scheme_mean_calibration_r2"]
                    for endpoint in endpoint_weights
                ),
                "equal_endpoint_mean_equal_scheme_calibration_rmse": sum(
                    endpoint_weights[endpoint]
                    * endpoint_by_name[endpoint]["equal_scheme_mean_calibration_rmse"]
                    for endpoint in endpoint_weights
                ),
                "equal_endpoint_mean_equal_scheme_calibration_spearman_rho": sum(
                    endpoint_weights[endpoint]
                    * endpoint_by_name[endpoint]["equal_scheme_mean_calibration_spearman_rho"]
                    for endpoint in endpoint_weights
                ),
                "equal_endpoint_mean_equal_scheme_test_r2": sum(
                    endpoint_weights[endpoint]
                    * endpoint_by_name[endpoint]["equal_scheme_mean_test_r2"]
                    for endpoint in endpoint_weights
                ),
                "equal_endpoint_mean_equal_scheme_test_rmse": sum(
                    endpoint_weights[endpoint]
                    * endpoint_by_name[endpoint]["equal_scheme_mean_test_rmse"]
                    for endpoint in endpoint_weights
                ),
                "equal_endpoint_mean_absolute_90pct_coverage_gap": sum(
                    endpoint_weights[endpoint]
                    * endpoint_by_name[endpoint]["equal_scheme_mean_absolute_90pct_coverage_gap"]
                    for endpoint in endpoint_weights
                ),
                "endpoint_rows": endpoint_rows,
            }
        )
    if not candidates:
        raise OracleFreezeError("no complete oracle candidates were found")
    candidates.sort(
        key=lambda row: (
            -float(row["equal_endpoint_mean_equal_scheme_calibration_r2"]),
            float(row["equal_endpoint_mean_equal_scheme_calibration_rmse"]),
            -float(row["equal_endpoint_mean_equal_scheme_calibration_spearman_rho"]),
            str(row["candidate_id"]),
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["selection_rank"] = rank
    return candidates, {
        candidate: {endpoint: dict(schemes) for endpoint, schemes in endpoints.items()}
        for candidate, endpoints in scheme_metrics.items()
    }


def _scheme_passes(metric: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    return (
        float(metric["mean_test_r2"]) >= float(gate["minimum_scheme_mean_test_r2"])
        and float(metric["mean_test_spearman_rho"])
        >= float(gate["minimum_scheme_mean_test_spearman_rho"])
        and float(metric["positive_r2_fold_fraction"])
        >= float(gate["minimum_positive_r2_fold_fraction"])
        and float(metric["mean_absolute_90pct_coverage_gap"])
        <= float(gate["maximum_scheme_mean_absolute_90pct_coverage_gap"])
    )


def build_nested_selection_audit(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Select a candidate inside each outer fold, then evaluate its test rows."""

    selection = config["selection"]
    eligible_schemes = set(selection["eligible_schemes"])
    endpoint_weights = {
        str(endpoint): float(weight) for endpoint, weight in selection["endpoints"].items()
    }
    by_partition_candidate: defaultdict[
        tuple[str, int, str],
        dict[str, Mapping[str, Any]],
    ] = defaultdict(dict)
    for source in rows:
        scheme = str(source["scheme"])
        endpoint = str(source["endpoint"])
        if scheme not in eligible_schemes or endpoint not in endpoint_weights:
            continue
        candidate = _candidate_id(source)
        key = (scheme, int(source["fold"]), candidate)
        if endpoint in by_partition_candidate[key]:
            raise OracleFreezeError(
                f"nested audit has duplicate {scheme}/{source['fold']}/{candidate}/{endpoint}"
            )
        by_partition_candidate[key][endpoint] = source
    by_partition: defaultdict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for (scheme, fold, candidate), endpoint_rows in sorted(by_partition_candidate.items()):
        if set(endpoint_rows) != set(endpoint_weights):
            raise OracleFreezeError(f"nested audit {scheme}/{fold}/{candidate} lacks an endpoint")
        by_partition[(scheme, fold)].append(
            {
                "candidate_id": candidate,
                "equal_endpoint_calibration_r2": sum(
                    endpoint_weights[endpoint] * float(endpoint_rows[endpoint]["calibration_r2"])
                    for endpoint in endpoint_weights
                ),
                "equal_endpoint_calibration_rmse": sum(
                    endpoint_weights[endpoint] * float(endpoint_rows[endpoint]["calibration_rmse"])
                    for endpoint in endpoint_weights
                ),
                "equal_endpoint_calibration_spearman_rho": sum(
                    endpoint_weights[endpoint]
                    * float(endpoint_rows[endpoint]["calibration_spearman_rho"])
                    for endpoint in endpoint_weights
                ),
                "endpoint_rows": dict(endpoint_rows),
            }
        )
    fold_selections: list[dict[str, Any]] = []
    nested_test_rows: list[dict[str, Any]] = []
    for scheme in sorted(eligible_schemes):
        for fold in sorted(_expected_folds(scheme)):
            candidates = by_partition.get((scheme, fold), [])
            if not candidates:
                raise OracleFreezeError(f"nested audit lacks candidates for {scheme}/{fold}")
            candidates.sort(
                key=lambda row: (
                    -float(row["equal_endpoint_calibration_r2"]),
                    float(row["equal_endpoint_calibration_rmse"]),
                    -float(row["equal_endpoint_calibration_spearman_rho"]),
                    str(row["candidate_id"]),
                )
            )
            selected = candidates[0]
            fold_selections.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "selected_candidate_id": selected["candidate_id"],
                    "selection_calibration_r2": selected["equal_endpoint_calibration_r2"],
                    "selection_calibration_rmse": selected["equal_endpoint_calibration_rmse"],
                    "selection_calibration_spearman_rho": selected[
                        "equal_endpoint_calibration_spearman_rho"
                    ],
                    "outer_test_used_for_selection": False,
                }
            )
            for endpoint, source in sorted(selected["endpoint_rows"].items()):
                nested_test_rows.append(
                    {
                        "endpoint": endpoint,
                        "scheme": scheme,
                        "fold": fold,
                        "test_r2": float(source["test_r2"]),
                        "test_rmse": float(source["test_rmse"]),
                        "test_spearman_rho": float(source["test_spearman_rho"]),
                        "test_coverage90": float(source["test_coverage90"]),
                    }
                )
    grouped: defaultdict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in nested_test_rows:
        grouped[(str(row["endpoint"]), str(row["scheme"]))].append(row)
    scheme_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    for (endpoint, scheme), group in sorted(grouped.items()):
        observed_folds = {int(row["fold"]) for row in group}
        if observed_folds != _expected_folds(scheme):
            raise OracleFreezeError(
                f"nested audit {endpoint}/{scheme} has folds {sorted(observed_folds)}"
            )
        r2_values = [float(row["test_r2"]) for row in group]
        scheme_metrics[endpoint][scheme] = {
            "folds": len(group),
            "mean_test_r2": _mean(r2_values, "nested test_r2"),
            "mean_test_rmse": _mean(
                [float(row["test_rmse"]) for row in group],
                "nested test_rmse",
            ),
            "mean_test_spearman_rho": _mean(
                [float(row["test_spearman_rho"]) for row in group],
                "nested test_spearman_rho",
            ),
            "mean_absolute_90pct_coverage_gap": _mean(
                [abs(float(row["test_coverage90"]) - 0.9) for row in group],
                "nested coverage gap",
            ),
            "positive_r2_fold_fraction": sum(value > 0 for value in r2_values) / len(r2_values),
        }
    return {
        "selection_unit": "within_outer_scheme_fold",
        "selection_evidence": "calibration_only",
        "evaluation_evidence": "outer_test_after_fold_local_selection",
        "fold_selections": fold_selections,
        "scheme_metrics": {endpoint: dict(schemes) for endpoint, schemes in scheme_metrics.items()},
    }


def build_applicability_policy(
    selected_candidate_id: str,
    scheme_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    nested_scheme_metrics: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate held-out evidence into endpoint and domain-specific actions."""

    selected = scheme_metrics.get(selected_candidate_id)
    if selected is None:
        raise OracleFreezeError("selected candidate lacks scheme metrics")
    gate = config["guidance_gate"]
    endpoints: dict[str, Any] = {}
    for endpoint, metrics in sorted(selected.items()):
        if set(nested_scheme_metrics.get(endpoint, {})) != set(metrics):
            raise OracleFreezeError(f"nested applicability evidence lacks {endpoint} schemes")
        scheme_gates = {}
        for scheme, metric in sorted(metrics.items()):
            nested_metric = nested_scheme_metrics[endpoint][scheme]
            fixed_passes = _scheme_passes(metric, gate)
            nested_passes = _scheme_passes(nested_metric, gate)
            scheme_gates[scheme] = {
                **dict(metric),
                "fixed_candidate_passes_guidance_gate": fixed_passes,
                "nested_selection_audit": dict(nested_metric),
                "nested_selection_audit_passes_guidance_gate": nested_passes,
                "passes_guidance_gate": fixed_passes and nested_passes,
            }
        domains: dict[str, Any] = {}
        for domain, required_schemes in config["applicability_domains"].items():
            passed = all(
                scheme_gates[scheme]["passes_guidance_gate"] for scheme in required_schemes
            )
            domains[domain] = {
                "required_schemes": list(required_schemes),
                "passes": passed,
                "action": ("conformal_lower_confidence_guidance" if passed else "abstain"),
                "failed_schemes": [
                    scheme
                    for scheme in required_schemes
                    if not scheme_gates[scheme]["passes_guidance_gate"]
                ],
            }
        endpoints[endpoint] = {
            "scheme_gates": scheme_gates,
            "domains": domains,
        }
    unsupported = {
        domain: {
            "action": "abstain",
            "reason": reason,
        }
        for domain, reason in sorted(config["unsupported_domains"].items())
    }
    return {
        "thresholds": dict(gate),
        "endpoints": endpoints,
        "unsupported_domains": unsupported,
        "any_guidance_domain_authorized": any(
            domain["passes"]
            for endpoint in endpoints.values()
            for domain in endpoint["domains"].values()
        ),
    }


def freeze_oracle(
    config_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Select one oracle candidate and freeze evidence-bounded use."""

    config = _load_json(config_path, "oracle freeze config")
    _validate_config(config)
    metric_rows, input_metadata = _load_lanes(config, repo_root)
    candidates, scheme_metrics = aggregate_candidates(metric_rows, config)
    nested_audit = build_nested_selection_audit(metric_rows, config)
    selected = candidates[0]
    policy = build_applicability_policy(
        str(selected["candidate_id"]),
        scheme_metrics,
        nested_audit["scheme_metrics"],
        config,
    )
    config_hash = sha256_file(config_path)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "oracle_architecture_and_applicability_policy_frozen",
        "task": "M0-07 AGILE oracle scientific freeze",
        "configuration": {
            "path": str(config_path.resolve().relative_to(repo_root.resolve())),
            "sha256": config_hash,
            "bytes": config_path.stat().st_size,
        },
        "inputs": input_metadata,
        "selection_contract": dict(config["selection"]),
        "candidate_ranking": candidates,
        "selected_model": selected,
        "nested_selection_audit": nested_audit,
        "applicability_policy": policy,
        "decision": {
            "one_representation_and_model_frozen_across_endpoints": True,
            "production_checkpoint_frozen": False,
            "production_refit_required": True,
            "outer_test_metrics_used_for_architecture_selection": False,
            "random_split_used_for_selection": False,
            "lantern_released_checkpoint_used_for_selection": False,
            "virtual_candidates_used_as_labels": False,
            "agile_role": "predictive_general_in_vitro_transfection_oracle",
            "in_vivo_endpoint_oracle": False,
            "unavailable_properties_imputed": False,
        },
        "unavailable_properties_not_imputed": list(config["unavailable_properties_not_imputed"]),
    }
    _atomic_write(output_path, _stable_json(result))
    return result
