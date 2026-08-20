"""Nested recalibration of the Ugi biological-applicability radius.

This audit uses persisted out-of-fold HeLa errors.  It does not call the oracle,
score generated candidates, or change the molecular generator.  A threshold is
chosen inside each meta-training fold and evaluated only on held-out product
labels.  Ranking authority and absolute/LCB authority are evaluated separately.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import stable_json as _stable_json
from forge.potency.applicability.ugi_selective_risk_proposal_diagnosis import (
    DISTANCE_KINDS,
    VIEWS,
    _prediction_metrics,
    _spearman,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_applicability_recalibration_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi_applicability_recalibration.v2"
LEDGER_SCHEMA_VERSION = "phase1_ugi_applicability_recalibration_ledger.v2"
AUTHORITY_TYPES = ("ranking", "absolute")
LABEL_PATTERN = re.compile(r"^A(?P<amine>\d+)B(?P<aldehyde>\d+)C(?P<isocyanide>\d+)$")
SCHEME_GROUP_ROLES = {
    "held_head_5fold": ("amine",),
    "held_aldehyde_5fold": ("aldehyde",),
    "held_isocyanide_5fold": ("isocyanide",),
    "held_head_aldehyde_pair_5fold": ("amine", "aldehyde"),
    "held_head_isocyanide_pair_5fold": ("amine", "isocyanide"),
    "held_aldehyde_isocyanide_pair_5fold": ("aldehyde", "isocyanide"),
}


class UgiApplicabilityRecalibrationError(RuntimeError):
    """Raised when the nested applicability contract is malformed."""


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiApplicabilityRecalibrationError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiApplicabilityRecalibrationError(f"{label} path escapes repository") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiApplicabilityRecalibrationError(f"{label} hash changed")
    return path


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiApplicabilityRecalibrationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiApplicabilityRecalibrationError(f"{label} must contain one JSON object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiApplicabilityRecalibrationError(f"invalid {label}: {path}") from error
    if not rows:
        raise UgiApplicabilityRecalibrationError(f"{label} is empty")
    return rows


def _meta_fold(label: str, *, salt: str, folds: int) -> int:
    digest = hashlib.sha256(f"{salt}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _group_id(label: str, scheme: str) -> str:
    match = LABEL_PATTERN.fullmatch(label)
    if match is None or scheme not in SCHEME_GROUP_ROLES:
        raise UgiApplicabilityRecalibrationError(
            f"cannot derive held-component group: {scheme}/{label}"
        )
    return "|".join(f"{role}:{match.group(role)}" for role in SCHEME_GROUP_ROLES[scheme])


def _normalized_radius(row: Mapping[str, Any], thresholds: Mapping[str, Any]) -> float:
    ratios = []
    for view in VIEWS:
        for kind in DISTANCE_KINDS:
            denominator = float(thresholds[view][kind]["interpolative_max"])
            if denominator <= 0:
                raise UgiApplicabilityRecalibrationError(
                    f"nonpositive applicability radius: {view}/{kind}"
                )
            ratios.append(float(row[f"{view}_{kind}_distance"]) / denominator)
    value = float(max(ratios))
    if not math.isfinite(value):
        raise UgiApplicabilityRecalibrationError("nonfinite normalized applicability radius")
    return value


def _top_quartile_enrichment(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if len(rows) < 8:
        return None
    ordered = sorted(rows, key=lambda row: (-float(row["y_pred"]), str(row["label"])))
    count = max(1, math.ceil(0.25 * len(ordered)))
    overall = float(np.mean([float(row["y_true"]) for row in ordered]))
    selected = float(np.mean([float(row["y_true"]) for row in ordered[:count]]))
    return selected - overall


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prediction = _prediction_metrics(rows)
    absolute_errors = [float(row["absolute_error"]) for row in rows]
    prediction["mae"] = float(np.mean(absolute_errors)) if absolute_errors else None
    prediction["top_quartile_observed_enrichment"] = _top_quartile_enrichment(rows)
    if rows:
        prediction["coverage90"] = float(
            np.mean([float(row["absolute_error"]) <= float(row["conformal_q90"]) for row in rows])
        )
        prediction["coverage90_gap"] = abs(float(prediction["coverage90"]) - 0.9)
    else:
        prediction["coverage90"] = None
        prediction["coverage90_gap"] = None
    return prediction


def _threshold_summary(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    schemes: Sequence[str],
) -> dict[str, Any]:
    selected = [row for row in rows if float(row["normalized_radius"]) <= threshold]
    by_scheme = {}
    for scheme in schemes:
        full = [row for row in rows if row["scheme"] == scheme]
        retained = [row for row in selected if row["scheme"] == scheme]
        by_scheme[scheme] = {
            "coverage": len(retained) / len(full) if full else 0.0,
            "metrics": _metrics(retained),
            "full_metrics": _metrics(full),
        }
    full_mae = [record["full_metrics"]["mae"] for record in by_scheme.values()]
    retained_mae = [record["metrics"]["mae"] for record in by_scheme.values()]
    equal_scheme_mae_reduction = None
    if all(value is not None for value in full_mae + retained_mae):
        full_mean = float(np.mean(full_mae))
        retained_mean = float(np.mean(retained_mae))
        equal_scheme_mae_reduction = (full_mean - retained_mean) / full_mean
    return {
        "threshold": threshold,
        "records": len(selected),
        "coverage": len(selected) / len(rows) if rows else 0.0,
        "metrics": _metrics(selected),
        "by_scheme": by_scheme,
        "equal_scheme_mae_relative_reduction": equal_scheme_mae_reduction,
    }


def _passes(summary: Mapping[str, Any], criteria: Mapping[str, Any]) -> bool:
    if float(summary["coverage"]) < float(criteria["minimum_overall_coverage"]):
        return False
    reduction = summary["equal_scheme_mae_relative_reduction"]
    if reduction is None or float(reduction) < float(criteria["minimum_mae_relative_reduction"]):
        return False
    for record in summary["by_scheme"].values():
        if float(record["coverage"]) < float(criteria["minimum_per_scheme_coverage"]):
            return False
        metrics = record["metrics"]
        rho = metrics["spearman"]
        enrichment = metrics["top_quartile_observed_enrichment"]
        if rho is None or float(rho) < float(criteria["minimum_per_scheme_spearman"]):
            return False
        if enrichment is None or float(enrichment) < float(
            criteria["minimum_top_quartile_enrichment"]
        ):
            return False
        if "minimum_per_scheme_r2" in criteria:
            r2 = metrics["r2"]
            if r2 is None or float(r2) < float(criteria["minimum_per_scheme_r2"]):
                return False
        if "maximum_per_scheme_coverage90_gap" in criteria:
            gap = metrics["coverage90_gap"]
            if gap is None or float(gap) > float(criteria["maximum_per_scheme_coverage90_gap"]):
                return False
    return True


def _select_threshold(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    schemes: Sequence[str],
    criteria: Mapping[str, Any],
) -> tuple[float | None, list[dict[str, Any]]]:
    curve = [_threshold_summary(rows, threshold, schemes) for threshold in thresholds]
    passed = [record for record in curve if _passes(record, criteria)]
    return (float(passed[-1]["threshold"]) if passed else None), curve


def _nested_authority(
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float],
    schemes: Sequence[str],
    criteria: Mapping[str, Any],
    folds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outer = []
    retained: list[dict[str, Any]] = []
    for fold in range(folds):
        fit = [row for row in rows if int(row["meta_fold"]) != fold]
        test = [row for row in rows if int(row["meta_fold"]) == fold]
        selected, curve = _select_threshold(fit, thresholds, schemes, criteria)
        chosen = (
            [row for row in test if float(row["normalized_radius"]) <= selected]
            if selected is not None
            else []
        )
        retained.extend(dict(row, selected_threshold=selected) for row in chosen)
        outer.append(
            {
                "fold": fold,
                "fit_labels": len({str(row["label"]) for row in fit}),
                "test_labels": len({str(row["label"]) for row in test}),
                "selected_threshold": selected,
                "fit_curve": curve,
                "test_summary": _threshold_summary(
                    test,
                    selected if selected is not None else -1.0,
                    schemes,
                ),
            }
        )
    full_selected, full_curve = _select_threshold(rows, thresholds, schemes, criteria)
    retained_by_scheme = {
        scheme: [row for row in retained if row["scheme"] == scheme] for scheme in schemes
    }
    return (
        {
            "outer_folds": outer,
            "folds_with_selected_threshold": sum(
                record["selected_threshold"] is not None for record in outer
            ),
            "selected_thresholds": [record["selected_threshold"] for record in outer],
            "nested_retained_records": len(retained),
            "nested_retained_coverage": len(retained) / len(rows),
            "nested_metrics": _metrics(retained),
            "nested_by_scheme": {
                scheme: {
                    "records": len(values),
                    "coverage": len(values) / sum(row["scheme"] == scheme for row in rows),
                    "metrics": _metrics(values),
                }
                for scheme, values in retained_by_scheme.items()
            },
            "full_data_selected_threshold": full_selected,
            "full_data_curve": full_curve,
        },
        retained,
    )


def _bootstrap_confirmation(
    rows: Sequence[Mapping[str, Any]],
    schemes: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    collected = {scheme: {"spearman": [], "top_quartile_enrichment": []} for scheme in schemes}
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for scheme in schemes:
        local: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            if row["scheme"] == scheme:
                local.setdefault(str(row["group_id"]), []).append(row)
        grouped[scheme] = local
    for _ in range(replicates):
        for scheme in schemes:
            group_ids = sorted(grouped[scheme])
            sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
            local = [row for group_id in sampled for row in grouped[scheme][str(group_id)]]
            if len(local) < 8:
                continue
            truth = np.asarray([float(row["y_true"]) for row in local])
            prediction = np.asarray([float(row["y_pred"]) for row in local])
            rho = _spearman(truth, prediction)
            enrichment = _top_quartile_enrichment(local)
            if rho is not None and enrichment is not None:
                collected[scheme]["spearman"].append(float(rho))
                collected[scheme]["top_quartile_enrichment"].append(float(enrichment))
    return {
        scheme: {
            name: {
                "lower_95": float(np.quantile(values, 0.025)) if values else None,
                "median": float(np.quantile(values, 0.5)) if values else None,
                "upper_95": float(np.quantile(values, 0.975)) if values else None,
            }
            for name, values in metrics.items()
        }
        for scheme, metrics in collected.items()
    }


def _ledger_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "authority",
        "scheme",
        "fold",
        "label",
        "group_id",
        "meta_fold",
        "normalized_radius",
        "selected_threshold",
        "y_true",
        "y_pred",
        "absolute_error",
        "conformal_q90",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as handle:
        handle.write(buffer.getvalue().encode())
    return output.getvalue()


def build_applicability_recalibration(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Run the read-only nested threshold audit and return a deterministic ledger."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="applicability recalibration config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiApplicabilityRecalibrationError("unsupported recalibration config schema")
    inputs = {label: _pin(repo, record, label=label) for label, record in config["inputs"].items()}
    applicability = _load_json(inputs["applicability_result"], label="applicability result")
    rows = _read_csv(inputs["heldout_oof_ledger"], label="heldout OOF ledger")
    schemes = tuple(str(value) for value in config["policy"]["schemes"])
    external_schemes = tuple(
        str(value) for value in config["policy"].get("external_stress_schemes", ())
    )
    observed_schemes = {str(row["scheme"]) for row in rows}
    if observed_schemes != set(schemes) | set(external_schemes):
        raise UgiApplicabilityRecalibrationError("heldout schemes changed")
    # The external scaffold-balanced split is a useful stress test, but it is
    # not one of the role-structured interpolation regimes whose thresholds
    # this audit is authorized to select.  Preserve that distinction rather
    # than allowing the extra split to influence the fitted radius.
    rows = [row for row in rows if str(row["scheme"]) in set(schemes)]
    thresholds = tuple(float(value) for value in config["policy"]["candidate_thresholds"])
    if not thresholds or tuple(sorted(set(thresholds))) != thresholds:
        raise UgiApplicabilityRecalibrationError("candidate thresholds must be unique and sorted")
    folds = int(config["policy"]["outer_group_folds"])
    salt = str(config["policy"]["outer_group_salt"])
    augmented = []
    for row in rows:
        group_id = _group_id(str(row["label"]), str(row["scheme"]))
        augmented.append(
            {
                **row,
                "fold": int(row["fold"]),
                "group_id": group_id,
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
                "absolute_error": float(row["absolute_error"]),
                "conformal_q90": float(row["conformal_q90"]),
                "normalized_radius": _normalized_radius(row, applicability["thresholds"]),
                "meta_fold": _meta_fold(f"{row['scheme']}|{group_id}", salt=salt, folds=folds),
            }
        )
    authority = {}
    ledger_rows = []
    for index, name in enumerate(AUTHORITY_TYPES):
        nested, retained = _nested_authority(
            augmented,
            thresholds=thresholds,
            schemes=schemes,
            criteria=config["policy"]["criteria"][name],
            folds=folds,
        )
        bootstrap = _bootstrap_confirmation(
            retained,
            schemes,
            replicates=int(config["policy"]["bootstrap"]["replicates"]),
            seed=int(config["policy"]["bootstrap"]["seed"]) + index,
        )
        confirmation = config["policy"]["confirmation"]
        evidence_pass = nested["folds_with_selected_threshold"] >= int(
            confirmation["minimum_folds_with_threshold"]
        ) and all(
            bootstrap[scheme]["spearman"]["lower_95"] is not None
            and float(bootstrap[scheme]["spearman"]["lower_95"])
            > float(confirmation["minimum_bootstrap_spearman_lower"])
            and bootstrap[scheme]["top_quartile_enrichment"]["lower_95"] is not None
            and float(bootstrap[scheme]["top_quartile_enrichment"]["lower_95"])
            > float(confirmation["minimum_bootstrap_top_quartile_enrichment_lower"])
            for scheme in schemes
        )
        authority[name] = {
            **nested,
            "bootstrap": bootstrap,
            "evidence_pass": evidence_pass,
            "candidate_use": (
                "eligible_for_frozen_oracle_authority_after_final-generator census"
                if evidence_pass
                else "exploration_only_no_oracle_authority"
            ),
        }
        ledger_rows.extend(dict(row, authority=name) for row in retained)
    ledger_rows = sorted(
        ledger_rows,
        key=lambda row: (
            str(row["authority"]),
            str(row["scheme"]),
            int(row["fold"]),
            str(row["label"]),
        ),
    )
    ledger = _ledger_bytes(ledger_rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nested_nonselecting_recalibration",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in inputs.items()
        },
        "policy": config["policy"],
        "records": len(augmented),
        "unique_labels": len({str(row["label"]) for row in augmented}),
        "authority": authority,
        "decision_boundary": {
            "exact_novelty_is_not_a_veto": True,
            "outside_authorized_radius_is_retained_for_exploration": True,
            "thresholds_are_selected_from_persisted_oof_errors_not_generated_yield": True,
            "simultaneously_exact_new_role_claims_require_matching_structured_evidence": True,
            "final_generator_acceptance_rate_is_not_used_to_tune_thresholds": True,
        },
        "artifacts": {
            "ledger": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(ledger_rows),
                "sha256": sha256_bytes(ledger),
            }
        },
    }
    logical = json.loads(_stable_json(result))
    result["result_sha256"] = hashlib.sha256(_stable_json(logical).encode()).hexdigest()
    return result, ledger


__all__ = [
    "UgiApplicabilityRecalibrationError",
    "_group_id",
    "_meta_fold",
    "_normalized_radius",
    "_passes",
    "_select_threshold",
    "build_applicability_recalibration",
]
