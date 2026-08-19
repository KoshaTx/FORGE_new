"""Corrected distribution-aware applicability audit.

Version 1 correctly separated provenance from chemical distance but revealed a
calibration degeneracy: a calibration product's component commonly occurs in
other training-fold products, making its component distance exactly zero.  V2
calibrates the distance of each unique calibration component after excluding
that exact component identity from the fold reference.  It also uses Morgan
count fingerprints so long-chain homologues that share the same local binary
bits remain distinguishable.

The correction is additive.  V1 artifacts are retained as a diagnostic and are
not rewritten.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from forge.bio import ugi_distributional_applicability as v1
from forge.data.r1_prime_audit import sha256_bytes, sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_distributional_applicability_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi_distributional_applicability.v2"
GENERATED_LEDGER_SCHEMA_VERSION = "phase1_ugi_generated_distributional_applicability.v2"
HELDOUT_LEDGER_SCHEMA_VERSION = "phase1_ugi_heldout_distributional_applicability.v2"

_COUNT_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class CountChemicalReference:
    """V1 descriptor geometry with count-based Morgan similarity."""

    def __init__(self, smiles: Iterable[str]) -> None:
        canonical = tuple(sorted({v1._canonical(str(value)) for value in smiles}))
        if not canonical:
            raise v1.UgiDistributionalApplicabilityError("chemical reference is empty")
        self.canonical = canonical
        self.canonical_set = frozenset(canonical)
        self.fingerprints = tuple(
            _COUNT_MORGAN.GetCountFingerprint(v1._molecule(value)) for value in canonical
        )
        matrix = np.asarray([v1._descriptors(value) for value in canonical], dtype=np.float64)
        self.center = np.median(matrix, axis=0)
        q25 = np.quantile(matrix, 0.25, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        scale = q75 - q25
        standard = np.std(matrix, axis=0)
        self.scale = np.where(scale > 1e-12, scale, np.where(standard > 1e-12, standard, 1.0))
        self.standardized = (matrix - self.center) / self.scale

    def distance(self, smiles: str) -> v1.DistancePair:
        canonical = v1._canonical(smiles)
        fingerprint = _COUNT_MORGAN.GetCountFingerprint(v1._molecule(canonical))
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, list(self.fingerprints))
        fingerprint_distance = 1.0 - float(max(similarities))
        query = (
            np.asarray(v1._descriptors(canonical), dtype=np.float64) - self.center
        ) / self.scale
        descriptor_distance = float(
            np.min(np.linalg.norm(self.standardized - query, axis=1))
            / math.sqrt(len(v1.DESCRIPTOR_NAMES))
        )
        if not (math.isfinite(fingerprint_distance) and math.isfinite(descriptor_distance)):
            raise v1.UgiDistributionalApplicabilityError("chemical distance is not finite")
        return v1.DistancePair(
            fingerprint=fingerprint_distance,
            descriptor=descriptor_distance,
            exact_identity_seen=canonical in self.canonical_set,
        )


def _references(rows: Sequence[Mapping[str, str]]) -> dict[str, CountChemicalReference]:
    return {
        "product": CountChemicalReference(row["product_smiles"] for row in rows),
        **{
            role: CountChemicalReference(row[field] for row in rows)
            for role, field in v1.ROLE_FIELDS.items()
        },
    }


def _identity_excluded_reference(
    smiles: Iterable[str],
    query_smiles: str,
) -> CountChemicalReference:
    query = v1._canonical(query_smiles)
    retained = [value for value in smiles if v1._canonical(str(value)) != query]
    return CountChemicalReference(retained)


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
            product_reference = CountChemicalReference(row["product_smiles"] for row in train)
            for row in calibration:
                distance = product_reference.distance(row["product_smiles"])
                collected["product"]["fingerprint"].append(distance.fingerprint)
                collected["product"]["descriptor"].append(distance.descriptor)

            training_components = [row[role_field] for row in train]
            unique_calibration_components = sorted(
                {v1._canonical(row[role_field]) for row in calibration}
            )
            for component in unique_calibration_components:
                reference = _identity_excluded_reference(training_components, component)
                distance = reference.distance(component)
                collected[role]["fingerprint"].append(distance.fingerprint)
                collected[role]["descriptor"].append(distance.descriptor)

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
                    else "calibration_product"
                ),
            }
    return thresholds


def _heldout_ledger(
    curated: Mapping[str, Mapping[str, str]],
    assignments: Mapping[tuple[str, int, str], str],
    predictions: Sequence[Mapping[str, str]],
    metrics: Mapping[tuple[str, int], Mapping[str, str]],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, Any]]:
    references: dict[tuple[str, int], dict[str, CountChemicalReference]] = {}
    output = []
    for prediction in predictions:
        scheme = str(prediction["scheme"])
        fold = int(prediction["fold"])
        label = str(prediction["label"])
        if assignments.get((scheme, fold, label)) != "test":
            raise v1.UgiDistributionalApplicabilityError("prediction is not an outer-test record")
        key = (scheme, fold)
        if key not in references:
            references[key] = _references(
                v1._training_rows(curated, assignments, scheme, fold, "train")
            )
        row = curated[label]
        distances = v1._distance_fields(references[key], row, generated=False)
        _, overall = v1._bins(distances, thresholds)
        q90 = float(metrics[key]["conformal_q90"])
        absolute_error = float(prediction["absolute_error"])
        record: dict[str, Any] = {
            "scheme": scheme,
            "fold": fold,
            "label": label,
            "distribution_bin": overall,
            "y_true": float(prediction["y_true"]),
            "y_pred": float(prediction["ensemble_y_pred"]),
            "absolute_error": absolute_error,
            "conformal_q90": q90,
            "covered90": absolute_error <= q90,
        }
        for view in v1.VIEWS:
            record[f"{view}_fingerprint_distance"] = distances[view].fingerprint
            record[f"{view}_descriptor_distance"] = distances[view].descriptor
        output.append(record)
    return sorted(output, key=lambda row: (row["scheme"], row["fold"], row["label"]))


def _generated_ledger(
    rows: Sequence[Mapping[str, str]],
    full_reference: Mapping[str, CountChemicalReference],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("guidance_action") != "abstain" or row.get("guidance_score") not in (
            "",
            None,
        ):
            raise v1.UgiDistributionalApplicabilityError("frozen generated row was not abstained")
        distances = v1._distance_fields(full_reference, row, generated=True)
        view_bins, overall = v1._bins(distances, thresholds)
        unseen_roles = json.loads(row["unseen_component_roles_json"])
        if not isinstance(unseen_roles, list):
            raise v1.UgiDistributionalApplicabilityError("unseen role provenance is malformed")
        if row["combination_seen_in_measured_training"].lower() == "true":
            identity = "exact_measured_combination"
        elif unseen_roles:
            identity = "exact_new_component_identity"
        else:
            identity = "exact_seen_components_novel_combination"
        record: dict[str, Any] = {
            "sample_index": int(row["sample_index"]),
            "product_id": row["product_id"],
            "product_smiles": row["product_smiles"],
            "amine_smiles": row["amine_smiles"],
            "aldehyde_smiles": row["aldehyde_smiles"],
            "isocyanide_smiles": row["isocyanide_smiles"],
            "exact_identity_provenance": identity,
            "exact_unseen_roles_json": json.dumps(unseen_roles, separators=(",", ":")),
            "overall_distribution_bin": overall,
            "ensemble_mean_descriptive_only": float(row["ensemble_mean"]),
            "ensemble_standard_deviation": float(row["ensemble_standard_deviation"]),
            "guidance_action": "abstain",
        }
        for view in v1.VIEWS:
            record[f"{view}_fingerprint_distance"] = distances[view].fingerprint
            record[f"{view}_descriptor_distance"] = distances[view].descriptor
            record[f"{view}_distribution_bin"] = view_bins[view]
        output.append(record)
    return output


def build_distributional_applicability_audit_v2(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build the corrected nonselecting chemical-distribution audit."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = v1._read_json(config_path, "distributional-applicability v2 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise v1.UgiDistributionalApplicabilityError("unsupported applicability v2 config schema")
    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        raise v1.UgiDistributionalApplicabilityError("applicability policy is missing")
    required_policy = {
        "endpoint": v1.ENDPOINT,
        "threshold_inputs": "structures_and_split_stage_only",
        "targets_predictions_and_errors_used_for_thresholds": False,
        "calibration_component_identity_excluded": True,
        "morgan_fingerprint_counts_used": True,
        "biological_guidance_authorized": False,
        "candidate_selection_changed": False,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise v1.UgiDistributionalApplicabilityError(f"applicability v2 policy changed: {key}")
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
    heldout = _heldout_ledger(
        curated,
        assignments,
        selected_predictions,
        metric_rows,
        thresholds,
    )
    generated = _generated_ledger(
        v1._read_csv(paths["fresh_pool_oracle_predictions"]),
        _references(list(curated.values())),
        thresholds,
    )
    generated_bytes = v1._csv_bytes(generated, v1.GENERATED_FIELDS)
    heldout_bytes = v1._csv_bytes(heldout, v1.HELDOUT_FIELDS)

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_corrected_distributional_audit_guidance_still_abstained",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "correction_from_v1": {
            "v1_artifacts_rewritten": False,
            "calibration_component_exact_identity_excluded": True,
            "binary_morgan_replaced_by_count_morgan": True,
            "reason": (
                "V1 component calibration distances collapsed to zero because the same "
                "calibration component appeared in other training-fold products"
            ),
        },
        "definitions": {
            "exact_identity_provenance": (
                "whether exact precursor graphs or the exact measured combination were observed"
            ),
            "distributional_applicability": (
                "multi-view chemical proximity calibrated without potency targets, predictions or errors"
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
                "independent review of bin-specific outer-test performance and a versioned "
                "guidance authorization decision"
            ),
        },
    }
    logical = json.loads(json.dumps(result, sort_keys=True))
    result["result_sha256"] = hashlib.sha256(
        json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result, generated_bytes, heldout_bytes


__all__ = ["CountChemicalReference", "build_distributional_applicability_audit_v2"]
