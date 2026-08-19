"""Score every non-measured terminal inside frozen multiview support.

This versioned rescore separates technical oracle scoring from scientific
claim authority. Exact component novelty is recorded as provenance and selects
the strongest available structured-split calibration; it is not itself an
applicability veto.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import sha256_file
from forge.core.io import stable_json as _stable_json
from forge.potency import ugi_distributional_applicability as v1
from forge.potency.oracle_classical import conformal_radius
from forge.potency.ugi_hela_potency_diagnostic import (
    FrozenHeLaOracleWorker,
    HeLaBatchPredictor,
    HeLaPotencyDiagnosticPolicy,
    _sha256_payload,
)
from forge.potency.ugi_interpolative_conformal import (
    _SCHEME_ROLES,
    _calibration_bins,
    _selected_calibration_ensembles,
)
from forge.potency.ugi_morphology_potency_matched_adjudication import _candidate, _exact_l1
from forge.potency.ugi_production_terminal_ranking import (
    EXPECTED_ARMS,
    _gzip_csv,
    _load,
    _pin,
    _read_jsonl,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_full_support_rescoring_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_full_support_rescoring.v2"
LEDGER_SCHEMA_VERSION = "phase1_ugi_production_full_support_rescoring_ledger.v2"


class UgiProductionFullSupportRescoringError(RuntimeError):
    """Raised when the frozen full-support rescore contract changes."""


def pattern_id_for_roles(unseen_roles: Sequence[str]) -> str:
    """Return the frozen novelty-pattern identifier for an ordered role tuple."""

    roles = tuple(unseen_roles)
    known = {
        (): "known_components_novel_combination",
        ("amine",): "amine_only",
        ("aldehyde",): "aldehyde_only",
        ("isocyanide",): "isocyanide_only",
        ("amine", "aldehyde"): "amine_aldehyde",
        ("amine", "isocyanide"): "amine_isocyanide",
        ("aldehyde", "isocyanide"): "aldehyde_isocyanide",
        ("amine", "aldehyde", "isocyanide"): "amine_aldehyde_isocyanide",
    }
    try:
        return known[roles]
    except KeyError as error:
        raise UgiProductionFullSupportRescoringError(
            f"unsupported ordered novelty pattern: {roles}"
        ) from error


def _extended_scales(
    policy: HeLaPotencyDiagnosticPolicy, specifications: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    graph_result = _load(policy.paths["oracle_graph_matrix_result"], label="graph matrix")
    ensembles = _selected_calibration_ensembles(
        policy.repo,
        graph_result,
        endpoint="expt_Hela",
        representation=v1.SELECTED_REPRESENTATION,
    )
    assignments = v1._split_index(v1._read_csv(policy.paths["oracle_split_assignments"]))
    curated = {row["label"]: row for row in policy.curated_rows}
    output: dict[str, dict[str, Any]] = {}
    for scale_id, specification in specifications.items():
        if not isinstance(specification, Mapping):
            raise UgiProductionFullSupportRescoringError("scale specification is malformed")
        scheme = str(specification["scheme"])
        unseen_roles = tuple(str(value) for value in specification["unseen_roles"])
        eligible_folds = tuple(int(value) for value in specification["eligible_folds"])
        if unseen_roles != tuple(_SCHEME_ROLES.get(scheme, ())):
            raise UgiProductionFullSupportRescoringError("scale and held-role scheme disagree")
        fold_q90: list[tuple[int, float]] = []
        predictions: list[float] = []
        counts: dict[int, int] = {}
        for fold in eligible_folds:
            ensemble = ensembles[(scheme, fold)]
            training = v1._training_rows(curated, assignments, scheme, fold, "train")
            calibration_rows = [curated[str(label)] for label in ensemble["labels"]]
            bins = _calibration_bins(calibration_rows, training, unseen_roles, policy.thresholds)
            residuals: list[float] = []
            for bin_name, truth, prediction in zip(
                bins, ensemble["truth"], ensemble["prediction"], strict=True
            ):
                if bin_name == "interpolative":
                    residuals.append(abs(float(truth) - float(prediction)))
                    predictions.append(float(prediction))
            if len(residuals) < 20:
                raise UgiProductionFullSupportRescoringError(
                    "configured calibration fold has fewer than 20 interpolative rows"
                )
            counts[fold] = len(residuals)
            fold_q90.append((fold, conformal_radius(np.asarray(residuals), 0.9)))
        max_q90 = max(value for _, value in fold_q90)
        sorted_lcb90 = tuple(sorted(value - max_q90 for value in predictions))
        observed = {
            "fold_interpolative_rows": {str(key): value for key, value in counts.items()},
            "fold_q90": {str(key): value for key, value in fold_q90},
            "max_q90": max_q90,
            "calibration_records": len(sorted_lcb90),
            "minimum_lcb90": min(sorted_lcb90),
            "maximum_lcb90": max(sorted_lcb90),
            "sorted_lcb90_sha256": _sha256_payload(list(sorted_lcb90)),
        }
        if observed != specification["expected"]:
            raise UgiProductionFullSupportRescoringError(
                f"extended calibration changed for {scale_id}"
            )
        output[str(scale_id)] = {
            "scheme": scheme,
            "max_q90": max_q90,
            "sorted_lcb90": sorted_lcb90,
            "calibration_records": len(sorted_lcb90),
            "values_sha256": observed["sorted_lcb90_sha256"],
        }
    return output


def _classify(
    rows: Sequence[Mapping[str, Any]],
    policy: HeLaPotencyDiagnosticPolicy,
    pattern_policies: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        record: dict[str, Any] = {
            "arm_id": str(source.get("arm_id", "")),
            "draw_index": int(source.get("draw_index", -1)),
            "program_sha256": str(source.get("program_sha256", "")),
            "exact_l1": False,
            "reason": "invalid_or_nonexact_l1",
            "overall_bin": "",
            "view_bins": "",
            "exact_measured": False,
            "oracle_scored": False,
            "pattern_id": "",
            "unseen_roles": "",
            "authority_tier": "not_applicable",
            "calibration_scale": "",
            "canonical_product": "",
            "canonical_amine": "",
            "canonical_aldehyde": "",
            "canonical_isocyanide": "",
            "oracle_mean": "",
            "oracle_sd": "",
            "conformal_q90": "",
            "lcb90": "",
            "calibration_ecdf": "",
            "potency_utility": 0.0,
            "conservative_high_potency": False,
            "worker_receipt_sha256": "",
        }
        terminal = source.get("native_terminal")
        if not isinstance(terminal, Mapping) or not _exact_l1(terminal):
            output.append(record)
            continue
        candidate = _candidate(source)
        classification = policy.classify_candidate_mapping(candidate)
        canonical = classification["canonical"]
        unseen = tuple(str(role) for role in classification["unseen_roles"])
        exact_measured = bool(classification["exact_measured_combination"])
        record.update(
            {
                "exact_l1": True,
                "overall_bin": str(classification["overall_bin"]),
                "view_bins": ";".join(
                    f"{name}:{value}" for name, value in classification["view_bins"]
                ),
                "exact_measured": exact_measured,
                "unseen_roles": ",".join(unseen),
                "canonical_product": canonical["product"],
                "canonical_amine": canonical["amine"],
                "canonical_aldehyde": canonical["aldehyde"],
                "canonical_isocyanide": canonical["isocyanide"],
            }
        )
        if classification["overall_bin"] != "interpolative" or any(
            value != "interpolative" for _, value in classification["view_bins"]
        ):
            record["reason"] = f"distribution_{classification['overall_bin']}"
        elif exact_measured:
            record["reason"] = "exact_measured_product_neutral_control"
        else:
            pattern_id = pattern_id_for_roles(unseen)
            specification = pattern_policies[pattern_id]
            record.update(
                {
                    "reason": "interpolative_nonmeasured_oracle_scored",
                    "oracle_scored": True,
                    "pattern_id": pattern_id,
                    "authority_tier": str(specification["authority_tier"]),
                    "calibration_scale": specification["calibration_scale"] or "",
                    "candidate": candidate,
                }
            )
        output.append(record)
    return output


def _unique_scoring_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["oracle_scored"]:
            chosen.setdefault(str(row["canonical_product"]), row)
    return [chosen[key] for key in sorted(chosen)]


def _score_batch(
    rows: Sequence[dict[str, Any]],
    predictor: HeLaBatchPredictor,
) -> tuple[str, list[tuple[float, float]]]:
    response = predictor.predict([row["candidate"] for row in rows])
    prediction = response.get("prediction")
    receipt = response.get("receipt_sha256")
    classifications = response.get("classifications")
    if (
        response.get("status") != "complete"
        or not isinstance(receipt, str)
        or len(receipt) != 64
        or not isinstance(prediction, Mapping)
        or not isinstance(classifications, list)
        or len(classifications) != len(rows)
    ):
        raise UgiProductionFullSupportRescoringError("oracle response is malformed")
    means = prediction.get("ensemble_mean")
    deviations = prediction.get("ensemble_standard_deviation")
    if not isinstance(means, list) or not isinstance(deviations, list):
        raise UgiProductionFullSupportRescoringError("oracle predictions are absent")
    values: list[tuple[float, float]] = []
    for row, classification, raw_mean, raw_sd in zip(
        rows, classifications, means, deviations, strict=True
    ):
        if not isinstance(classification, Mapping):
            raise UgiProductionFullSupportRescoringError("oracle identity record is malformed")
        if classification.get("canonical", {}).get("product") != row["canonical_product"]:
            raise UgiProductionFullSupportRescoringError("oracle and generator identity differ")
        mean, deviation = float(raw_mean), float(raw_sd)
        if not math.isfinite(mean) or not math.isfinite(deviation) or deviation < 0:
            raise UgiProductionFullSupportRescoringError("oracle returned a nonfinite score")
        values.append((mean, deviation))
    return receipt, values


def _apply_scores(
    rows: list[dict[str, Any]],
    unique: Sequence[dict[str, Any]],
    predictor: HeLaBatchPredictor,
    scales: Mapping[str, Any],
    batch_size: int,
) -> set[str]:
    scored: dict[str, dict[str, Any]] = {}
    receipts: set[str] = set()
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        receipt, values = _score_batch(batch, predictor)
        receipts.add(receipt)
        for row, (mean, deviation) in zip(batch, values, strict=True):
            values_out: dict[str, Any] = {
                "oracle_mean": mean,
                "oracle_sd": deviation,
                "worker_receipt_sha256": receipt,
            }
            scale_id = str(row["calibration_scale"])
            if scale_id:
                scale = scales[scale_id]
                if hasattr(scale, "score"):
                    lcb90, cdf, utility = scale.score(mean)
                    q90 = float(scale.max_q90)
                else:
                    q90 = float(scale["max_q90"])
                    lcb90 = mean - q90
                    calibration = scale["sorted_lcb90"]
                    cdf = bisect_right(calibration, lcb90) / len(calibration)
                    utility = max(0.0, 2.0 * cdf - 1.0)
                values_out.update(
                    {
                        "conformal_q90": q90,
                        "lcb90": lcb90,
                        "calibration_ecdf": cdf,
                        "potency_utility": utility,
                        "conservative_high_potency": cdf > 0.5,
                    }
                )
            scored[str(row["canonical_product"])] = values_out
    for row in rows:
        if row["oracle_scored"]:
            row.update(scored[str(row["canonical_product"])])
    return receipts


def _arm_summary(rows: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm_id"] == arm]
    interpolative = [
        row for row in selected if row["exact_l1"] and row["overall_bin"] == "interpolative"
    ]
    scored = [row for row in selected if row["oracle_scored"]]
    calibrated = [row for row in scored if row["calibration_scale"]]
    return {
        "attempts": len(selected),
        "exact_l1": sum(bool(row["exact_l1"]) for row in selected),
        "raw_interpolative_rows": len(interpolative),
        "raw_interpolative_unique_products": len(
            {row["canonical_product"] for row in interpolative}
        ),
        "exact_measured_neutral_rows": sum(bool(row["exact_measured"]) for row in interpolative),
        "oracle_scored_rows": len(scored),
        "oracle_scored_unique_products": len({row["canonical_product"] for row in scored}),
        "calibrated_rows": len(calibrated),
        "raw_descriptive_only_rows": len(scored) - len(calibrated),
        "conservative_high_rows": sum(bool(row["conservative_high_potency"]) for row in scored),
        "patterns": dict(sorted(Counter(str(row["pattern_id"]) for row in scored).items())),
        "authority_tiers": dict(
            sorted(Counter(str(row["authority_tier"]) for row in scored).items())
        ),
    }


def build_full_support_rescoring(
    repo: Path,
    config_path: Path,
    *,
    predictor: HeLaBatchPredictor | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Build the versioned full-support scoring result and compressed ledger."""

    repo = repo.resolve()
    config = _load(config_path.resolve(), label="full-support rescoring config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "frozen_before_full_support_oracle_rescoring"
    ):
        raise UgiProductionFullSupportRescoringError("unsupported rescoring config")
    paths = {label: _pin(repo, record, label=label) for label, record in config["inputs"].items()}
    generation = _load(paths["generation_result"], label="production generation")
    if generation.get("status") != "complete_matched_production_candidate_generation":
        raise UgiProductionFullSupportRescoringError("production generation status changed")
    rows = _read_jsonl(paths["terminal_ledger"])
    policy = HeLaPotencyDiagnosticPolicy(repo, paths["hela_diagnostic_policy"])
    pattern_policies = config["pattern_policies"]
    if set(pattern_policies) != {
        pattern_id_for_roles(roles)
        for roles in (
            (),
            ("amine",),
            ("aldehyde",),
            ("isocyanide",),
            ("amine", "aldehyde"),
            ("amine", "isocyanide"),
            ("aldehyde", "isocyanide"),
            ("amine", "aldehyde", "isocyanide"),
        )
    }:
        raise UgiProductionFullSupportRescoringError("novelty pattern set changed")
    extended = _extended_scales(policy, config["extended_calibration_scales"])
    scales: dict[str, Any] = {**policy.scales, **extended}
    classified = _classify(rows, policy, pattern_policies)
    unique = _unique_scoring_rows(classified)
    batch_size = int(config["scoring"]["batch_size"])
    if predictor is None:
        with FrozenHeLaOracleWorker(policy) as worker:
            receipts = _apply_scores(classified, unique, worker, scales, batch_size)
            ready_receipt = worker.ready_receipt_sha256
    else:
        receipts = _apply_scores(classified, unique, predictor, scales, batch_size)
        ready_receipt = "test_predictor"
    fields = tuple(key for key in classified[0] if key != "candidate")
    ledger = _gzip_csv(classified, fields)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_full_interpolative_support_terminal_rescoring",
        "scope": dict(config["scope"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "policy": {
            "all_nonmeasured_interpolative_products_scored": True,
            "exact_novelty_is_provenance_not_applicability": True,
            "claim_authority_is_role_specific": True,
            "pattern_policies": pattern_policies,
        },
        "summary": {
            "arms": {arm: _arm_summary(classified, arm) for arm in EXPECTED_ARMS},
            "global_unique_oracle_calls": len(unique),
            "oracle_response_receipts": sorted(receipts),
        },
        "worker_ready_receipt_sha256": ready_receipt,
        "artifacts": {
            "terminal_rescoring.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(classified),
                "sha256": hashlib.sha256(ledger).hexdigest(),
            }
        },
        "next_gate": "diversity_balanced_route_shortlists_stratified_by_evidence_tier",
        "nonclaims": [
            "A raw oracle score is not a potency-generalization guarantee.",
            "Exploratory tiers do not authorize potency tilting.",
            "This rescore does not change the generator, applicability boundary or panel lock.",
        ],
    }
    result["result_sha256"] = hashlib.sha256(_stable_json(result).encode()).hexdigest()
    return result, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiProductionFullSupportRescoringError",
    "build_full_support_rescoring",
    "pattern_id_for_roles",
]
