"""Deterministic failure attribution for the bounded Ugi HeLa potency adapter."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class PotencyAdapterAttributionError(ValueError):
    """The potency-adapter attribution inputs violate the frozen data contract."""


_AGILE_LABEL = re.compile(r"^A(?P<head>\d+)_B(?P<aldehyde>\d+)_C(?P<isocyanide>\d+)$")


def _component_indices(label: str) -> tuple[int, int, int]:
    match = _AGILE_LABEL.fullmatch(label.strip())
    if match is None:
        raise PotencyAdapterAttributionError(f"invalid AGILE component label: {label!r}")
    return tuple(int(match.group(name)) for name in ("head", "aldehyde", "isocyanide"))


def factorial_dataset_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Describe the actual independent support behind a nominal product count."""

    if not rows:
        raise PotencyAdapterAttributionError("the YX HeLa dataset is empty")
    labels = [str(row["source_lipid_name"]) for row in rows]
    smiles = [str(row["model_smiles"]) for row in rows]
    components = [_component_indices(label) for label in labels]
    heads = {value[0] for value in components}
    aldehydes = {value[1] for value in components}
    isocyanides = {value[2] for value in components}
    tail_pairs = {(value[1], value[2]) for value in components}
    observed = {(value[0], value[1], value[2]) for value in components}
    expected = {
        (head, aldehyde, isocyanide)
        for head in heads
        for aldehyde in aldehydes
        for isocyanide in isocyanides
    }
    values = np.asarray([float(row["label_value"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise PotencyAdapterAttributionError("the YX HeLa labels are not finite")
    head_counts = [sum(value[0] == head for value in components) for head in sorted(heads)]
    pair_counts = [
        sum((value[1], value[2]) == pair for value in components) for pair in sorted(tail_pairs)
    ]
    return {
        "rows": len(rows),
        "unique_products": len(set(smiles)),
        "unique_source_labels": len(set(labels)),
        "unique_label_values": len(set(values.tolist())),
        "unique_heads": len(heads),
        "unique_aldehydes": len(aldehydes),
        "unique_isocyanides": len(isocyanides),
        "unique_aldehyde_isocyanide_pairs": len(tail_pairs),
        "complete_cartesian_library": observed == expected,
        "cartesian_product_count": len(expected),
        "products_per_head": sorted(set(head_counts)),
        "products_per_aldehyde_isocyanide_pair": sorted(set(pair_counts)),
        "label_mean": float(values.mean()),
        "label_sample_sd": float(values.std(ddof=1)),
        "label_minimum": float(values.min()),
        "label_maximum": float(values.max()),
    }


def label_variance_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    """Partition label variance into head, tail-pair and unreplicated interaction terms."""

    components = [_component_indices(str(row["source_lipid_name"])) for row in rows]
    values = np.asarray([float(row["label_value"]) for row in rows], dtype=np.float64)
    grand_mean = float(values.mean())
    head_values: dict[int, list[float]] = defaultdict(list)
    pair_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    for component, value in zip(components, values, strict=True):
        head_values[component[0]].append(float(value))
        pair_values[(component[1], component[2])].append(float(value))
    head_means = {key: float(np.mean(local)) for key, local in head_values.items()}
    pair_means = {key: float(np.mean(local)) for key, local in pair_values.items()}
    head_prediction = np.asarray([head_means[value[0]] for value in components])
    pair_prediction = np.asarray([pair_means[(value[1], value[2])] for value in components])
    additive_prediction = head_prediction + pair_prediction - grand_mean
    total = float(np.square(values - grand_mean).sum())
    if total <= 0.0:
        raise PotencyAdapterAttributionError("label variance is zero")

    def r_squared(prediction: np.ndarray) -> float:
        return 1.0 - float(np.square(values - prediction).sum()) / total

    additive = r_squared(additive_prediction)
    return {
        "head_mean_r2": r_squared(head_prediction),
        "aldehyde_isocyanide_pair_mean_r2": r_squared(pair_prediction),
        "additive_head_plus_pair_r2": additive,
        "unreplicated_interaction_or_noise_fraction": 1.0 - additive,
        "between_head_mean_sample_sd": float(np.std(list(head_means.values()), ddof=1)),
        "between_pair_mean_sample_sd": float(np.std(list(pair_means.values()), ddof=1)),
    }


def split_support_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Count product rows and held component groups in every frozen fold."""

    grouped: dict[tuple[str, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        scheme = str(row["scheme"])
        if scheme in {"held_head_5fold", "held_aldehyde_isocyanide_pair_5fold"}:
            grouped[(scheme, int(row["fold"]))].append(row)
    if len(grouped) != 10:
        raise PotencyAdapterAttributionError("the component-disjoint split must contain ten folds")
    folds: dict[str, Any] = {}
    for (scheme, fold), local in sorted(grouped.items()):
        stage_rows = {
            stage: sum(str(row["stage"]) == stage for row in local)
            for stage in ("train", "calibration", "test")
        }
        test_groups = {str(row["group_id"]) for row in local if row["stage"] == "test"}
        fit_groups = {
            str(row["group_id"]) for row in local if row["stage"] in {"train", "calibration"}
        }
        folds[f"{scheme}::{fold}"] = {
            "stage_rows": stage_rows,
            "fit_component_groups": len(fit_groups),
            "test_component_groups": len(test_groups),
            "fit_test_group_overlap": len(fit_groups & test_groups),
        }
    return {
        "folds": folds,
        "training_products_per_fold": sorted(
            {int(value["stage_rows"]["train"]) for value in folds.values()}
        ),
        "test_products_per_fold": sorted(
            {int(value["stage_rows"]["test"]) for value in folds.values()}
        ),
        "fit_groups_by_scheme": {
            scheme: sorted(
                {
                    int(value["fit_component_groups"])
                    for key, value in folds.items()
                    if key.startswith(f"{scheme}::")
                }
            )
            for scheme in ("held_head_5fold", "held_aldehyde_isocyanide_pair_5fold")
        },
        "test_groups_by_scheme": {
            scheme: sorted(
                {
                    int(value["test_component_groups"])
                    for key, value in folds.items()
                    if key.startswith(f"{scheme}::")
                }
            )
            for scheme in ("held_head_5fold", "held_aldehyde_isocyanide_pair_5fold")
        },
    }


def optimization_summary(crossfit: Mapping[str, Any]) -> dict[str, Any]:
    """Distinguish an optimizer failure from failure to learn label-specific held-out signal."""

    folds = crossfit.get("folds")
    if not isinstance(folds, list) or len(folds) != 10:
        raise PotencyAdapterAttributionError("cross-fit result does not contain ten folds")
    output: dict[str, Any] = {}
    for arm in ("real", "shuffled"):
        traces = [row[f"{arm}_losses"] for row in folds]
        first = np.asarray([float(trace[0]["denoising"]) for trace in traces])
        final = np.asarray([float(trace[-1]["denoising"]) for trace in traces])
        gradients = np.asarray([float(trace[-1]["gradient_norm"]) for trace in traces])
        retention = np.asarray([float(trace[-1]["retention_kl"]) for trace in traces])
        output[arm] = {
            "mean_initial_denoising_loss": float(first.mean()),
            "mean_final_denoising_loss": float(final.mean()),
            "mean_fractional_loss_reduction": float(1.0 - final.mean() / first.mean()),
            "mean_final_gradient_norm": float(gradients.mean()),
            "mean_final_retention_kl": float(retention.mean()),
            "all_final_values_finite": bool(
                np.isfinite(final).all()
                and np.isfinite(gradients).all()
                and np.isfinite(retention).all()
            ),
        }
    output["real_minus_shuffled_final_loss"] = float(
        output["real"]["mean_final_denoising_loss"]
        - output["shuffled"]["mean_final_denoising_loss"]
    )
    output["gross_optimization_failure"] = not (
        output["real"]["all_final_values_finite"]
        and output["real"]["mean_fractional_loss_reduction"] > 0.5
        and output["real"]["mean_final_gradient_norm"] > 0.0
    )
    return output


def signal_summary(crossfit: Mapping[str, Any]) -> dict[str, Any]:
    metrics = crossfit.get("time_bin_metrics")
    if not isinstance(metrics, Mapping) or len(metrics) != 6:
        raise PotencyAdapterAttributionError("cross-fit time-bin metrics are incomplete")
    bins: dict[str, Any] = {}
    for key, value in sorted(metrics.items()):
        real = float(value["real_auroc"])
        shuffled = float(value["shuffled_auroc"])
        bins[str(key)] = {
            "real_auroc": real,
            "shuffled_auroc": shuffled,
            "real_minus_shuffled_auroc": real - shuffled,
            "residual_spearman": float(value["residual_spearman"]),
            "passes": bool(value["gates"]["passes"]),
            "failed_checks": sorted(
                name for name, passed in value["gates"].items() if name != "passes" and not passed
            ),
        }
    eligible = {
        key: value
        for key, value in bins.items()
        if key.endswith("::middle") or key.endswith("::late")
    }
    return {
        "bins": bins,
        "maximum_eligible_real_minus_shuffled_auroc": max(
            float(value["real_minus_shuffled_auroc"]) for value in eligible.values()
        ),
        "active_time_bins": list(crossfit.get("active_time_bins", [])),
        "signal_gate_passed": crossfit.get("status") == "signal_gate_pass",
    }


def additional_study_support(
    observations: Sequence[Mapping[str, str]], assignments: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    """Count exact-Ugi supervision available elsewhere without pooling assay scales."""

    assignment_by_product = {str(row["canonical_product_smiles"]): row for row in assignments}
    if len(assignment_by_product) != len(assignments):
        raise PotencyAdapterAttributionError("Ugi assignments are not unique by product")
    matched: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in observations:
        if row["endpoint"] != "HeLa":
            continue
        assignment = assignment_by_product.get(str(row["model_smiles"]))
        if assignment is not None:
            matched[str(row["study_id"])].append(assignment)
    yx = matched.get("YX_2024", [])
    yx_sets = {
        "heads": {str(row["amine_head_smiles"]) for row in yx},
        "aldehydes": {str(row["oxoester_aldehyde_body_tail_smiles"]) for row in yx},
        "isocyanides": {str(row["isocyanide_tail_smiles"]) for row in yx},
    }
    studies: dict[str, Any] = {}
    for study_id in sorted({str(row["study_id"]) for row in observations}):
        all_rows = [
            row for row in observations if row["study_id"] == study_id and row["endpoint"] == "HeLa"
        ]
        local = matched.get(study_id, [])
        local_sets = {
            "heads": {str(row["amine_head_smiles"]) for row in local},
            "aldehydes": {str(row["oxoester_aldehyde_body_tail_smiles"]) for row in local},
            "isocyanides": {str(row["isocyanide_tail_smiles"]) for row in local},
        }
        studies[study_id] = {
            "hela_rows": len(all_rows),
            "matched_exact_ugi_rows": len(local),
            "unique_products": len({str(row["canonical_product_smiles"]) for row in local}),
            "unique_heads": len(local_sets["heads"]),
            "unique_aldehydes": len(local_sets["aldehydes"]),
            "unique_isocyanides": len(local_sets["isocyanides"]),
            "new_to_yx_heads": len(local_sets["heads"] - yx_sets["heads"]),
            "new_to_yx_aldehydes": len(local_sets["aldehydes"] - yx_sets["aldehydes"]),
            "new_to_yx_isocyanides": len(local_sets["isocyanides"] - yx_sets["isocyanides"]),
        }
    return {
        "studies": studies,
        "naive_cross_study_label_pooling_admissible": False,
        "reason": "label_value is normalized within each study and assay protocols are not exchangeable",
    }


__all__ = [
    "PotencyAdapterAttributionError",
    "additional_study_support",
    "factorial_dataset_summary",
    "label_variance_summary",
    "optimization_summary",
    "signal_summary",
    "split_support_summary",
]
