"""Bin-conditional conformal audit for the fixed HeLa oracle.

This additive audit does not refit or select an oracle.  It reuses the
calibration predictions written by the already-frozen graph-oracle fits,
classifies calibration structures with the version-3 distributional policy,
and estimates a fold-specific split-conformal radius from calibration rows in
the interpolative bin.  The radius is then evaluated once on outer-test rows
that the frozen version-3 audit also classified as interpolative.

The audit is deliberately nonauthorizing.  It was motivated after inspection
of the version-3 outer-test diagnostic and therefore cannot, by itself,
retroactively create a pristine guidance test.  It establishes whether
conditional calibration is technically coherent and whether a later bounded,
versioned authorization review is warranted.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.data.r1_prime_audit import sha256_file
from forge.potency.applicability import ugi_distributional_applicability as v1
from forge.potency.applicability import ugi_distributional_applicability_v2 as v2
from forge.potency.oracle.oracle_classical import conformal_radius

CONFIG_SCHEMA_VERSION = "phase1_ugi_interpolative_conformal_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_interpolative_conformal.v1"

_SCHEME_ROLES = {
    "held_head_5fold": ("amine",),
    "held_aldehyde_5fold": ("aldehyde",),
    "held_isocyanide_5fold": ("isocyanide",),
    "held_head_aldehyde_pair_5fold": ("amine", "aldehyde"),
    "held_head_isocyanide_pair_5fold": ("amine", "isocyanide"),
    "held_aldehyde_isocyanide_pair_5fold": ("aldehyde", "isocyanide"),
}


class UgiInterpolativeConformalError(ValueError):
    """Raised when conditional calibration violates its frozen contract."""


def _verified_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise UgiInterpolativeConformalError("conditional-calibration inputs are missing")
    output: dict[str, Path] = {}
    for name, record in inputs.items():
        if not isinstance(record, Mapping):
            raise UgiInterpolativeConformalError(f"invalid input record: {name}")
        path = repo / str(record.get("path"))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise UgiInterpolativeConformalError(f"input hash mismatch: {name}")
        output[str(name)] = path
    return output


def _product_reference(
    rows: Sequence[Mapping[str, str]],
    query: Mapping[str, str],
    held_roles: Sequence[str],
) -> v2.CountChemicalReference:
    """Exclude products sharing any exact held-role component identity."""

    retained = []
    for candidate in rows:
        shares_identity = any(
            v1._canonical(candidate[v1.ROLE_FIELDS[role]])
            == v1._canonical(query[v1.ROLE_FIELDS[role]])
            for role in held_roles
        )
        if not shares_identity:
            retained.append(candidate["product_smiles"])
    return v2.CountChemicalReference(retained)


def _calibration_bin(
    row: Mapping[str, str],
    training_rows: Sequence[Mapping[str, str]],
    held_roles: Sequence[str],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> str:
    references: dict[str, v2.CountChemicalReference] = {
        "product": _product_reference(training_rows, row, held_roles)
    }
    for role, field in v1.ROLE_FIELDS.items():
        values = [candidate[field] for candidate in training_rows]
        if role in held_roles:
            references[role] = v2._identity_excluded_reference(values, row[field])
        else:
            references[role] = v2.CountChemicalReference(values)
    distances = v1._distance_fields(references, row, generated=False)
    _, overall = v1._bins(distances, thresholds)
    return overall


def _calibration_bins(
    rows: Sequence[Mapping[str, str]],
    training_rows: Sequence[Mapping[str, str]],
    held_roles: Sequence[str],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    """Classify a fold with cached exact-identity-excluded references."""

    standard_components = {
        role: v2.CountChemicalReference(candidate[field] for candidate in training_rows)
        for role, field in v1.ROLE_FIELDS.items()
    }
    component_cache: dict[tuple[str, str], v2.CountChemicalReference] = {}
    product_cache: dict[tuple[str, ...], v2.CountChemicalReference] = {}
    output = []
    for row in rows:
        held_identity = tuple(v1._canonical(row[v1.ROLE_FIELDS[role]]) for role in held_roles)
        if held_identity not in product_cache:
            product_cache[held_identity] = _product_reference(training_rows, row, held_roles)
        references = {"product": product_cache[held_identity]}
        for role, field in v1.ROLE_FIELDS.items():
            if role not in held_roles:
                references[role] = standard_components[role]
                continue
            component = v1._canonical(row[field])
            key = (role, component)
            if key not in component_cache:
                component_cache[key] = v2._identity_excluded_reference(
                    (candidate[field] for candidate in training_rows), component
                )
            references[role] = component_cache[key]
        distances = v1._distance_fields(references, row, generated=False)
        _, overall = v1._bins(distances, thresholds)
        output.append(overall)
    return output


def _selected_calibration_ensembles(
    repo: Path,
    graph_result: Mapping[str, Any],
    *,
    endpoint: str,
    representation: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    entries = graph_result.get("fit_sources", {}).get("entries")
    if not isinstance(entries, list):
        raise UgiInterpolativeConformalError("graph fit-source registry is missing")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise UgiInterpolativeConformalError("invalid graph fit-source entry")
        relative = str(entry["path"])
        if f"/{representation}/{endpoint}/" not in relative or not any(
            f"/{scheme}/" in relative for scheme in _SCHEME_ROLES
        ):
            continue
        path = repo / relative
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise UgiInterpolativeConformalError("graph fit-source hash mismatch")
        fit = v1._read_json(path, "graph fit")
        job = fit.get("job", {})
        if (
            job.get("architecture") != representation
            or job.get("endpoint") != endpoint
            or job.get("scheme") not in _SCHEME_ROLES
        ):
            continue
        grouped[(str(job["scheme"]), int(job["fold"]))].append(fit)

    ensembles: dict[tuple[str, int], dict[str, Any]] = {}
    for key, fits in sorted(grouped.items()):
        fits.sort(key=lambda fit: int(fit["job"]["seed"]))
        if len(fits) != 3:
            raise UgiInterpolativeConformalError(f"selected ensemble lacks three seeds: {key}")
        labels = fits[0]["calibration"]["labels"]
        truth = fits[0]["calibration"]["truth"]
        if any(
            fit["calibration"]["labels"] != labels or fit["calibration"]["truth"] != truth
            for fit in fits[1:]
        ):
            raise UgiInterpolativeConformalError(f"calibration rows differ by seed: {key}")
        predictions = np.asarray(
            [fit["calibration"]["prediction"] for fit in fits], dtype=np.float64
        )
        ensembles[key] = {
            "labels": list(labels),
            "truth": np.asarray(truth, dtype=np.float64),
            "prediction": np.mean(predictions, axis=0),
            "seeds": [int(fit["job"]["seed"]) for fit in fits],
        }
    return ensembles


def _mean(values: Sequence[float], label: str) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise UgiInterpolativeConformalError(f"cannot average {label}")
    return float(sum(values) / len(values))


def _scheme_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_test_rows_per_fold: int,
) -> dict[str, Any]:
    by_fold: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["fold"])].append(row)
    fold_rows = []
    for fold, subset in sorted(by_fold.items()):
        if len(subset) < minimum_test_rows_per_fold:
            continue
        metric = v1._performance(subset)
        if metric["r2"] is None or metric["spearman_rho"] is None:
            continue
        fold_rows.append({"fold": fold, **metric})
    if not fold_rows:
        return {"eligible_folds": 0, "folds": [], "gate_passed": False}
    return {
        "eligible_folds": len(fold_rows),
        "folds": fold_rows,
        "mean_test_r2": _mean([float(row["r2"]) for row in fold_rows], "fold R2"),
        "mean_test_spearman_rho": _mean(
            [float(row["spearman_rho"]) for row in fold_rows], "fold Spearman"
        ),
        "positive_r2_fold_fraction": float(
            sum(float(row["r2"]) > 0 for row in fold_rows) / len(fold_rows)
        ),
        "mean_absolute_90pct_coverage_gap": _mean(
            [float(row["absolute_coverage90_gap"]) for row in fold_rows],
            "fold coverage gap",
        ),
        "pooled": v1._performance(rows),
    }


def build_interpolative_conformal_audit(
    repo: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Build a nonauthorizing conditional conformal audit."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = v1._read_json(config_path, "interpolative conformal config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiInterpolativeConformalError("unsupported conditional-calibration schema")
    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        raise UgiInterpolativeConformalError("conditional-calibration policy is missing")
    required = {
        "endpoint": v1.ENDPOINT,
        "representation": v1.SELECTED_REPRESENTATION,
        "distribution_bin": "interpolative",
        "coverage": 0.9,
        "oracle_refit_or_reselected": False,
        "biological_guidance_authorized": False,
        "candidate_selection_changed": False,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise UgiInterpolativeConformalError(f"conditional policy changed: {key}")
    minimum_calibration = int(policy.get("minimum_calibration_rows_per_fold"))
    minimum_test = int(policy.get("minimum_test_rows_per_fold"))
    if minimum_calibration < 20 or minimum_test < 20:
        raise UgiInterpolativeConformalError("minimum fold sizes must be at least 20")

    paths = _verified_inputs(config, repo)
    applicability = v1._read_json(paths["applicability_v3_result"], "v3 result")
    thresholds = applicability.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise UgiInterpolativeConformalError("v3 applicability thresholds are missing")
    curated_rows = v1._read_csv(paths["curated_agile"])
    curated = {row["label"]: {**row, "product_smiles": row["model_smiles"]} for row in curated_rows}
    if len(curated) != len(curated_rows):
        raise UgiInterpolativeConformalError("curated labels are duplicated")
    assignments = v1._split_index(v1._read_csv(paths["oracle_split_assignments"]))
    heldout_rows = v1._read_csv(paths["applicability_v3_heldout"])
    graph_result = v1._read_json(paths["oracle_graph_matrix_result"], "graph matrix result")
    ensembles = _selected_calibration_ensembles(
        repo,
        graph_result,
        endpoint=str(policy["endpoint"]),
        representation=str(policy["representation"]),
    )

    radii: dict[tuple[str, int], dict[str, Any]] = {}
    for key, ensemble in sorted(ensembles.items()):
        scheme, fold = key
        training = v1._training_rows(curated, assignments, scheme, fold, "train")
        held_roles = _SCHEME_ROLES[scheme]
        residuals = []
        bin_counts: dict[str, int] = defaultdict(int)
        calibration_rows = [curated[str(label)] for label in ensemble["labels"]]
        calibration_bins = _calibration_bins(calibration_rows, training, held_roles, thresholds)
        for bin_name, truth, prediction in zip(
            calibration_bins, ensemble["truth"], ensemble["prediction"], strict=True
        ):
            bin_counts[bin_name] += 1
            if bin_name == "interpolative":
                residuals.append(abs(float(truth) - float(prediction)))
        record: dict[str, Any] = {
            "scheme": scheme,
            "fold": fold,
            "calibration_bin_counts": dict(sorted(bin_counts.items())),
            "interpolative_calibration_rows": len(residuals),
            "seeds": ensemble["seeds"],
            "q90": None,
            "eligible": False,
        }
        if len(residuals) >= minimum_calibration:
            record["q90"] = conformal_radius(np.asarray(residuals), 0.9)
            record["eligible"] = True
        radii[key] = record

    evaluated: list[dict[str, Any]] = []
    for row in heldout_rows:
        if row["distribution_bin"] != "interpolative":
            continue
        key = (row["scheme"], int(row["fold"]))
        radius = radii.get(key)
        if not radius or not radius["eligible"]:
            continue
        absolute_error = float(row["absolute_error"])
        q90 = float(radius["q90"])
        evaluated.append(
            {
                "scheme": row["scheme"],
                "fold": int(row["fold"]),
                "label": row["label"],
                "distribution_bin": "interpolative",
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
                "absolute_error": absolute_error,
                "conformal_q90": q90,
                "covered90": absolute_error <= q90,
            }
        )

    gate = config.get("guidance_gate")
    if not isinstance(gate, Mapping):
        raise UgiInterpolativeConformalError("guidance gate is missing")
    by_scheme: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_scheme[str(row["scheme"])].append(row)
    scheme_results = {}
    for scheme in _SCHEME_ROLES:
        metric = _scheme_metrics(by_scheme.get(scheme, []), minimum_test_rows_per_fold=minimum_test)
        passed = (
            metric.get("eligible_folds", 0) >= int(policy["minimum_eligible_folds"])
            and metric.get("mean_test_r2", -math.inf) >= float(gate["minimum_scheme_mean_test_r2"])
            and metric.get("mean_test_spearman_rho", -math.inf)
            >= float(gate["minimum_scheme_mean_test_spearman_rho"])
            and metric.get("positive_r2_fold_fraction", -math.inf)
            >= float(gate["minimum_positive_r2_fold_fraction"])
            and metric.get("mean_absolute_90pct_coverage_gap", math.inf)
            <= float(gate["maximum_scheme_mean_absolute_90pct_coverage_gap"])
        )
        metric["gate_passed"] = bool(passed)
        scheme_results[scheme] = metric

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_interpolative_conditional_calibration_diagnostic",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            name: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "policy": dict(policy),
        "guidance_gate": dict(gate),
        "calibration_radii": [radii[key] for key in sorted(radii)],
        "evaluation": {
            "evaluated_interpolative_rows": len(evaluated),
            "scheme_results": scheme_results,
        },
        "adjudication": {
            "biological_guidance_authorized": False,
            "candidate_selection_changed": False,
            "oracle_refit_or_reselected": False,
            "reason": (
                "conditional calibration was specified after inspection of the v3 outer-test "
                "diagnostic and therefore requires a separate versioned authorization review"
            ),
            "next_gate": (
                "review role-specific scheme support and, if justified, freeze a bounded "
                "interpolative-domain pilot before any potency-guided sampling"
            ),
        },
    }
    logical = json.loads(json.dumps(result, sort_keys=True))
    result["result_sha256"] = hashlib.sha256(
        json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


__all__ = [
    "UgiInterpolativeConformalError",
    "_calibration_bin",
    "build_interpolative_conformal_audit",
]
