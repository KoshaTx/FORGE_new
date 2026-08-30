"""Produce the attributable diagnosis of the failed Ugi HeLa potency-adapter gate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_csv, read_json_object, write_json
from forge.potency.adapter_failure_attribution import (
    additional_study_support,
    factorial_dataset_summary,
    label_variance_summary,
    optimization_summary,
    signal_summary,
    split_support_summary,
)

CONFIG_SCHEMA = "forge.ugi_hela_potency_adapter_failure_attribution_config.v1"
RESULT_SCHEMA = "forge.ugi_hela_potency_adapter_failure_attribution.v1"


class PotencyAdapterFailureAttributionError(ValueError):
    """The frozen potency-adapter diagnosis cannot be attributed."""


def build_potency_adapter_failure_attribution(
    config_path: Path, repo: Path, output_path: Path
) -> dict[str, Any]:
    config = read_json_object(
        config_path,
        error=PotencyAdapterFailureAttributionError,
        label="potency-adapter failure-attribution config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PotencyAdapterFailureAttributionError("unsupported failure-attribution config")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise PotencyAdapterFailureAttributionError("failure-attribution inputs are missing")
    paths = {
        str(name): resolve_pin(pin, repo, label=str(name))
        for name, pin in sorted(raw_inputs.items())
    }
    observations = read_csv(paths["potency_observations"])
    yx = [row for row in observations if row["study_id"] == "YX_2024" and row["endpoint"] == "HeLa"]
    assignments = read_csv(paths["ugi_assignments"])
    splits = read_csv(paths["oracle_split_assignments"])
    crossfit = read_json_object(
        paths["crossfit_result"],
        error=PotencyAdapterFailureAttributionError,
        label="verified potency-adapter cross-fit result",
    )
    manifest = read_json_object(
        paths["crossfit_manifest"],
        error=PotencyAdapterFailureAttributionError,
        label="verified potency-adapter stage manifest",
    )
    oracle = read_json_object(
        paths["completed_molecule_oracle"],
        error=PotencyAdapterFailureAttributionError,
        label="frozen completed-molecule oracle selection",
    )
    if (
        crossfit.get("status") != "signal_gate_fail"
        or crossfit.get("checkpoints") != {}
        or manifest.get("summary", {}).get("guided_generation") is not False
        or manifest.get("resources", {}).get("gpu_type") != "H100!"
    ):
        raise PotencyAdapterFailureAttributionError(
            "the input is not the verified negative exact-H100 cross-fit"
        )
    selected = oracle.get("selected_model")
    if not isinstance(selected, Mapping):
        raise PotencyAdapterFailureAttributionError(
            "completed-molecule oracle selection is missing"
        )
    dataset = factorial_dataset_summary(yx)
    variance = label_variance_summary(yx)
    split_support = split_support_summary(splits)
    optimization = optimization_summary(crossfit)
    signal = signal_summary(crossfit)
    other_studies = additional_study_support(observations, assignments)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "failure_attributed",
        "config": pin_record(config_path, repo),
        "inputs": {name: pin_record(path, repo) for name, path in sorted(paths.items())},
        "dataset_geometry": dataset,
        "label_variance": variance,
        "component_disjoint_support": split_support,
        "optimization": optimization,
        "held_out_signal": signal,
        "completed_molecule_reference": {
            "candidate_id": selected.get("candidate_id"),
            "post_selection_test_spearman_rho": float(selected["post_selection_test_spearman_rho"]),
            "post_selection_test_r2": float(selected["post_selection_test_r2"]),
            "interpretation": (
                "The same endpoint contains useful terminal whole-molecule ranking signal; "
                "the failed adapter does not establish that the labels are globally unusable."
            ),
        },
        "additional_lnpdb_hela_support": other_studies,
        "attribution": {
            "headline_1100_is_independent_sample_count": False,
            "effective_component_support": {
                "heads_total": int(dataset["unique_heads"]),
                "aldehyde_isocyanide_pairs_total": int(dataset["unique_aldehyde_isocyanide_pairs"]),
                "heads_available_to_fit_per_fold": int(
                    split_support["fit_groups_by_scheme"]["held_head_5fold"][0]
                ),
                "pairs_available_to_fit_per_fold": int(
                    split_support["fit_groups_by_scheme"]["held_aldehyde_isocyanide_pair_5fold"][0]
                ),
            },
            "enough_for_completed_molecule_ranking": True,
            "enough_to_qualify_current_partial_state_adapter": False,
            "gross_optimizer_failure": bool(optimization["gross_optimization_failure"]),
            "dominant_observed_failure": (
                "The adapter learned a generic denoising/domain shift also learned from shuffled "
                "labels; its held-component label-specific AUROC increment never reached 0.05."
            ),
            "data_limitation": (
                "The complete 20-by-55 factorial adds repeated combinations, not 1,100 independent "
                "component chemotypes, and 48.6% of label variance is unreplicated interaction or "
                "measurement noise."
            ),
            "objective_limitation": (
                "Primary endpoint denoising can improve while ignoring target percentile because "
                "the noisy molecular state already identifies much of the clean molecule."
            ),
            "audit_limitation": (
                "The frozen result did not persist per-record real/shuffled scores or counterfactual "
                "quantile sweeps, so score correlation and quantile monotonicity are not auditable "
                "without a new prespecified run."
            ),
        },
        "recommended_next_experiment": {
            "blind_longer_training": False,
            "relax_signal_gate": False,
            "add_more_combinations_of_existing_20_by_55_components": False,
            "priority": (
                "Use an explicitly contrastive ordinal objective on paired counterfactual quantiles, "
                "pair real and shuffled arms on identical molecule batches and adapter initialization, "
                "and persist per-record scores plus q10/q50/q90 monotonicity."
            ),
            "data_priority": (
                "Add genuinely new independently measured head and tail chemotypes. JC_2023 can "
                "contribute limited study-aware auxiliary exact-Ugi supervision, but its normalized "
                "labels must not be naively pooled with YX_2024."
            ),
            "execution_authorized_by_this_attribution": False,
        },
        "nonclaims": [
            "This analysis does not authorize another potency-guidance run.",
            "The negative adapter result does not establish that molecular structure lacks HeLa signal.",
            "Within-study normalized LNPDB labels are not interchangeable absolute assay measurements.",
        ],
    }
    write_json(output_path, result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "PotencyAdapterFailureAttributionError",
    "build_potency_adapter_failure_attribution",
]
