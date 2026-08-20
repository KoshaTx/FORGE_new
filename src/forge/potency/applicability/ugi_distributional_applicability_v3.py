"""Component-shift-calibrated distributional applicability audit.

V2 corrected component distances but calibrated the whole-product view against
calibration combinations whose components were still represented elsewhere in
the training fold.  V3 excludes every training product sharing the calibration
component identity for the held role before measuring whole-product distance.
This makes both component and product thresholds model the same intended
question: chemical proximity under an exact-new component shift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.potency.applicability import ugi_distributional_applicability as v1
from forge.potency.applicability import ugi_distributional_applicability_v2 as v2

CONFIG_SCHEMA_VERSION = "phase1_ugi_distributional_applicability_config.v3"
RESULT_SCHEMA_VERSION = "phase1_ugi_distributional_applicability.v3"
GENERATED_LEDGER_SCHEMA_VERSION = "phase1_ugi_generated_distributional_applicability.v3"
HELDOUT_LEDGER_SCHEMA_VERSION = "phase1_ugi_heldout_distributional_applicability.v3"


def _product_reference_excluding_role_identity(
    rows: Sequence[Mapping[str, str]],
    *,
    role_field: str,
    query_component: str,
) -> v2.CountChemicalReference:
    query = v1._canonical(query_component)
    retained = [row["product_smiles"] for row in rows if v1._canonical(row[role_field]) != query]
    return v2.CountChemicalReference(retained)


def _calibration_thresholds(
    curated: Mapping[str, Mapping[str, str]],
    assignments: Mapping[tuple[str, int, str], str],
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, dict[str, dict[str, float]]]:
    collected: dict[str, dict[str, list[float]]] = {
        view: {kind: [] for kind in v1.DISTANCE_TYPES} for view in v1.VIEWS
    }
    for role, scheme in v1.ROLE_SCHEMES.items():
        folds = sorted(
            {fold for candidate_scheme, fold, _ in assignments if candidate_scheme == scheme}
        )
        role_field = v1.ROLE_FIELDS[role]
        for fold in folds:
            train = v1._training_rows(curated, assignments, scheme, fold, "train")
            calibration = v1._training_rows(curated, assignments, scheme, fold, "calibration")
            training_components = [row[role_field] for row in train]
            calibration_by_component: dict[str, list[Mapping[str, str]]] = {}
            for row in calibration:
                calibration_by_component.setdefault(v1._canonical(row[role_field]), []).append(row)
            for component, component_rows in sorted(calibration_by_component.items()):
                component_reference = v2._identity_excluded_reference(
                    training_components, component
                )
                component_distance = component_reference.distance(component)
                collected[role]["fingerprint"].append(component_distance.fingerprint)
                collected[role]["descriptor"].append(component_distance.descriptor)

                product_reference = _product_reference_excluding_role_identity(
                    train,
                    role_field=role_field,
                    query_component=component,
                )
                for row in component_rows:
                    product_distance = product_reference.distance(row["product_smiles"])
                    collected["product"]["fingerprint"].append(product_distance.fingerprint)
                    collected["product"]["descriptor"].append(product_distance.descriptor)

    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    for view in v1.VIEWS:
        thresholds[view] = {}
        for kind in v1.DISTANCE_TYPES:
            values = collected[view][kind]
            if not values:
                raise v1.UgiDistributionalApplicabilityError(
                    f"calibration produced no {view}/{kind} distances"
                )
            thresholds[view][kind] = {
                "interpolative_max": v1._quantile(values, lower_quantile),
                "boundary_max": v1._quantile(values, upper_quantile),
                "calibration_records": len(values),
                "calibration_unit": (
                    "unique_exact_identity_excluded_component_per_fold"
                    if view != "product"
                    else "product_after_held_role_component_identity_exclusion"
                ),
            }
    return thresholds


def build_distributional_applicability_audit_v3(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build the component-shift-calibrated nonselecting audit."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = v1._read_json(config_path, "distributional-applicability v3 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise v1.UgiDistributionalApplicabilityError("unsupported applicability v3 config schema")
    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        raise v1.UgiDistributionalApplicabilityError("applicability policy is missing")
    required_policy = {
        "endpoint": v1.ENDPOINT,
        "threshold_inputs": "structures_and_split_stage_only",
        "targets_predictions_and_errors_used_for_thresholds": False,
        "calibration_component_identity_excluded": True,
        "product_reference_excludes_calibration_component_identity": True,
        "morgan_fingerprint_counts_used": True,
        "biological_guidance_authorized": False,
        "candidate_selection_changed": False,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise v1.UgiDistributionalApplicabilityError(f"applicability v3 policy changed: {key}")
    lower = float(policy.get("interpolative_calibration_quantile"))
    upper = float(policy.get("boundary_calibration_quantile"))
    if not 0.0 < lower < upper < 1.0:
        raise v1.UgiDistributionalApplicabilityError("applicability quantiles are invalid")

    paths = v1._verified_inputs(config, repo)
    curated_rows = v1._read_csv(paths["curated_agile"])
    curated: dict[str, dict[str, str]] = {}
    for source in curated_rows:
        label = source["label"]
        if label in curated:
            raise v1.UgiDistributionalApplicabilityError("curated AGILE label is duplicated")
        curated[label] = {**source, "product_smiles": source["model_smiles"]}
    assignments = v1._split_index(v1._read_csv(paths["oracle_split_assignments"]))
    thresholds = _calibration_thresholds(
        curated,
        assignments,
        lower_quantile=lower,
        upper_quantile=upper,
    )
    metric_rows = v1._metric_rows(v1._read_csv(paths["oracle_graph_metrics"]))
    selected_predictions = v1._selected_predictions(v1._read_csv(paths["oracle_graph_predictions"]))
    heldout = v2._heldout_ledger(
        curated,
        assignments,
        selected_predictions,
        metric_rows,
        thresholds,
    )
    generated = v2._generated_ledger(
        v1._read_csv(paths["fresh_pool_oracle_predictions"]),
        v2._references(list(curated.values())),
        thresholds,
    )
    generated_bytes = v1._csv_bytes(generated, v1.GENERATED_FIELDS)
    heldout_bytes = v1._csv_bytes(heldout, v1.HELDOUT_FIELDS)

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_component_shift_calibrated_audit_guidance_still_abstained",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "correction_history": {
            "v1_and_v2_artifacts_rewritten": False,
            "v2_component_identity_exclusion_retained": True,
            "v3_product_reference_excludes_same_calibration_component": True,
            "reason": (
                "whole-product distance must be calibrated under the same exact-new "
                "component shift as the component distance"
            ),
        },
        "definitions": {
            "exact_identity_provenance": (
                "whether exact precursor graphs or the exact measured combination were observed"
            ),
            "distributional_applicability": (
                "multi-view chemical proximity calibrated under an identity-excluded component shift"
            ),
            "exact_new_does_not_imply_extrapolative": True,
            "interpolative_does_not_itself_authorize_guidance": True,
            "descriptor_names": list(v1.DESCRIPTOR_NAMES),
            "fingerprint": "Morgan radius 2 count fingerprint, 2048 dimensions",
        },
        "thresholds": thresholds,
        "heldout_oracle_evidence": v1._heldout_summary(heldout),
        "generated_pool": v1._generated_summary(generated),
        "artifacts": {
            "generated_applicability.csv.gz": {
                "schema_version": GENERATED_LEDGER_SCHEMA_VERSION,
                "records": len(generated),
                "sha256": sha256_bytes(generated_bytes),
            },
            "heldout_applicability.csv.gz": {
                "schema_version": HELDOUT_LEDGER_SCHEMA_VERSION,
                "records": len(heldout),
                "sha256": sha256_bytes(heldout_bytes),
            },
        },
        "adjudication": {
            "biological_guidance_authorized": False,
            "candidate_selection_changed": False,
            "raw_oracle_mean_used_for_ranking": False,
            "thresholds_used_targets_predictions_or_errors": False,
            "next_gate": (
                "independent fold-resolved review of interpolative-bin performance and a "
                "versioned guidance authorization decision"
            ),
        },
    }
    logical = json.loads(json.dumps(result, sort_keys=True))
    result["result_sha256"] = hashlib.sha256(
        json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result, generated_bytes, heldout_bytes


__all__ = [
    "_product_reference_excluding_role_identity",
    "build_distributional_applicability_audit_v3",
]
