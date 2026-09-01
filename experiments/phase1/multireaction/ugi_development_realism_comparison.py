"""Reassess a pinned Ugi sampler comparison with the train-only role-realism assay."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from experiments.phase1.multireaction.lipid_realism_assessment import (
    run_lipid_realism_assessment,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json

CONFIG_SCHEMA = "forge.ugi_development_realism_comparison_config.v1"
PANEL_CONFIG_SCHEMA = "forge.ugi_development_realism_comparison_config.v2"
RESULT_SCHEMA = "forge.ugi_development_realism_comparison.v1"
PANEL_RESULT_SCHEMA = "forge.ugi_development_realism_comparison.v2"

PANEL_PRIMARY_METRICS = (
    "role_normalized_wasserstein",
    "role_energy_distance",
    "role_rbf_mmd2",
)
PANEL_WEIGHTINGS = ("attempt_weighted", "unique_product_weighted")


class UgiDevelopmentRealismComparisonError(ValueError):
    """The pinned comparison archive or development contract is malformed."""


def _arm_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    global_assessment = result["assessment"]
    global_manifold = global_assessment["empirical_lipid_manifold"]
    role = result["ugi_role_realism"]
    attempt_role = role["attempt_weighted"]
    unique_role = role["unique_product_weighted"]
    c2st = unique_role["classifier_two_sample"]
    global_c2st = global_manifold["classifier_two_sample"]
    return {
        "connected_fraction_per_attempt": global_assessment["molecular_output"][
            "connected_fraction_per_attempt"
        ],
        "unique_connected_products": global_assessment["molecular_output"]["unique_connected"],
        "global_normalized_wasserstein": global_manifold["normalized_descriptor_wasserstein"][
            "mean_across_descriptors"
        ],
        "global_c2st_auc": global_c2st.get("auc_mean"),
        "exact_l1_unambiguous_fraction_per_attempt": role[
            "exact_l1_unambiguous_fraction_per_attempt"
        ],
        "unique_exact_l1_products": role["unique_exact_l1_products"],
        "attempt_role_normalized_wasserstein": attempt_role["normalized_wasserstein"][
            "mean_across_role_descriptors"
        ],
        "attempt_role_energy_distance": attempt_role["energy_distance"],
        "attempt_role_rbf_mmd2": attempt_role["rbf_mmd"]["mmd2"],
        "attempt_role_nearest_reference_distance_mean": attempt_role["nearest_reference_distance"][
            "mean"
        ],
        "unique_role_normalized_wasserstein": unique_role["normalized_wasserstein"][
            "mean_across_role_descriptors"
        ],
        "unique_role_energy_distance": unique_role["energy_distance"],
        "unique_role_rbf_mmd2": unique_role["rbf_mmd"]["mmd2"],
        "unique_role_nearest_reference_distance_mean": unique_role["nearest_reference_distance"][
            "mean"
        ],
        # Backward-compatible aliases for the original unique-product summary.
        "role_normalized_wasserstein": unique_role["normalized_wasserstein"][
            "mean_across_role_descriptors"
        ],
        "role_energy_distance": unique_role["energy_distance"],
        "role_rbf_mmd2": unique_role["rbf_mmd"]["mmd2"],
        "role_nearest_reference_distance_mean": unique_role["nearest_reference_distance"]["mean"],
        "role_c2st_auc": c2st.get("auc_mean"),
        "role_c2st_folds": c2st.get("folds"),
    }


def _validate_metric_panel(panel: Any) -> dict[str, Any]:
    if not isinstance(panel, Mapping) or set(panel) != {
        "primary_metrics",
        "weightings",
        "secondary_metrics",
        "decision_rule",
        "direction",
        "inference",
    }:
        raise UgiDevelopmentRealismComparisonError("metric panel fields changed")
    if tuple(panel["primary_metrics"]) != PANEL_PRIMARY_METRICS:
        raise UgiDevelopmentRealismComparisonError("primary metric panel changed")
    if tuple(panel["weightings"]) != PANEL_WEIGHTINGS:
        raise UgiDevelopmentRealismComparisonError("metric panel weightings changed")
    if panel["direction"] != "lower_is_better":
        raise UgiDevelopmentRealismComparisonError("metric panel direction changed")
    if panel["decision_rule"] != "descriptive_directional_consistency":
        raise UgiDevelopmentRealismComparisonError("metric panel decision rule changed")
    if panel["inference"] != "none_seed0_development_only":
        raise UgiDevelopmentRealismComparisonError("metric panel inference claim changed")
    secondary = panel["secondary_metrics"]
    if (
        not isinstance(secondary, list)
        or not secondary
        or any(not isinstance(metric, str) or not metric for metric in secondary)
    ):
        raise UgiDevelopmentRealismComparisonError("secondary metric panel is malformed")
    return dict(panel)


def _panel_comparison(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    baseline_arm: str,
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = summaries[baseline_arm]
    comparisons: dict[str, Any] = {}
    for arm_id, summary in sorted(summaries.items()):
        if arm_id == baseline_arm:
            continue
        by_weighting: dict[str, Any] = {}
        all_deltas: list[float] = []
        for weighting in panel["weightings"]:
            prefix = "attempt" if weighting == "attempt_weighted" else "unique"
            by_metric: dict[str, Any] = {}
            for metric in panel["primary_metrics"]:
                key = f"{prefix}_{metric}"
                baseline_value = float(baseline[key])
                arm_value = float(summary[key])
                delta = arm_value - baseline_value
                all_deltas.append(delta)
                by_metric[metric] = {
                    "baseline": baseline_value,
                    "arm": arm_value,
                    "arm_minus_baseline": delta,
                    "direction": (
                        "improved" if delta < 0.0 else "worsened" if delta > 0.0 else "tied"
                    ),
                }
            by_weighting[weighting] = by_metric
        if all(delta < 0.0 for delta in all_deltas):
            classification = "uniformly_improved"
        elif all(delta > 0.0 for delta in all_deltas):
            classification = "uniformly_worsened"
        else:
            classification = "mixed"
        comparisons[arm_id] = {
            "classification": classification,
            "by_weighting": by_weighting,
            "primary_metrics_improved": sum(delta < 0.0 for delta in all_deltas),
            "primary_metrics_total": len(all_deltas),
            "inference": panel["inference"],
        }
    return {
        "baseline_arm": baseline_arm,
        "direction": panel["direction"],
        "decision_rule": panel["decision_rule"],
        "comparisons": comparisons,
    }


def _validate_member(member: str) -> PurePosixPath:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise UgiDevelopmentRealismComparisonError(f"unsafe archive member: {member!r}")
    return path


def run_ugi_development_realism_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract only declared attempt ledgers and run the development assay without generation."""

    config = read_json_object(
        config_path,
        error=UgiDevelopmentRealismComparisonError,
        label="Ugi development realism comparison config",
    )
    schema = config.get("schema_version") if isinstance(config, Mapping) else None
    expected_fields = {
        "schema_version",
        "scientific_question",
        "inputs",
        "arms",
        "expected_attempts",
        "nonclaims",
    }
    if schema == PANEL_CONFIG_SCHEMA:
        expected_fields.update({"baseline_arm", "metric_panel"})
    if (
        not isinstance(config, Mapping)
        or schema not in {CONFIG_SCHEMA, PANEL_CONFIG_SCHEMA}
        or set(config) != expected_fields
    ):
        raise UgiDevelopmentRealismComparisonError("comparison config fields changed")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {
        "development_realism_config",
        "comparison_archive",
    }:
        raise UgiDevelopmentRealismComparisonError("comparison input pins changed")
    inputs = {key: resolve_pin(value, repo, label=key) for key, value in raw_inputs.items()}
    raw_arms = config["arms"]
    if not isinstance(raw_arms, Mapping) or not raw_arms:
        raise UgiDevelopmentRealismComparisonError("comparison arms are empty")
    expected_attempts = config["expected_attempts"]
    if isinstance(expected_attempts, bool) or not isinstance(expected_attempts, int):
        raise UgiDevelopmentRealismComparisonError("expected_attempts must be an integer")
    if expected_attempts <= 0:
        raise UgiDevelopmentRealismComparisonError("expected_attempts must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UgiDevelopmentRealismComparisonError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    arm_results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="forge-ugi-development-realism-") as temporary:
        temporary_dir = Path(temporary)
        with tarfile.open(inputs["comparison_archive"], mode="r:*") as archive:
            available = {member.name: member for member in archive.getmembers()}
            for arm_id, arm in sorted(raw_arms.items()):
                if (
                    not isinstance(arm_id, str)
                    or not isinstance(arm, Mapping)
                    or set(arm)
                    != {
                        "archive_member",
                        "method_id",
                        "seed",
                    }
                ):
                    raise UgiDevelopmentRealismComparisonError("comparison arm fields changed")
                member_name = str(arm["archive_member"])
                member_path = _validate_member(member_name)
                member = available.get(member_name)
                if member is None or not member.isfile():
                    raise UgiDevelopmentRealismComparisonError(
                        f"attempt ledger is absent from archive: {member_name}"
                    )
                extracted_path = temporary_dir.joinpath(*member_path.parts)
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise UgiDevelopmentRealismComparisonError(
                        f"could not read archive member: {member_name}"
                    )
                extracted_path.write_bytes(source.read())
                arm_results[arm_id] = run_lipid_realism_assessment(
                    inputs["development_realism_config"],
                    repo,
                    extracted_path,
                    output_dir / arm_id,
                    method_id=str(arm["method_id"]),
                    seed=int(arm["seed"]),
                    expected_attempts=expected_attempts,
                )

    summaries = {arm: _arm_summary(result) for arm, result in sorted(arm_results.items())}
    baseline_arm = str(config["baseline_arm"]) if schema == PANEL_CONFIG_SCHEMA else "argmax"
    baseline = summaries.get(baseline_arm)
    if baseline is None:
        raise UgiDevelopmentRealismComparisonError(
            f"comparison has no prespecified baseline arm: {baseline_arm}"
        )
    deltas = {
        arm: {
            key: float(value) - float(baseline[key])
            for key, value in summary.items()
            if arm != baseline_arm
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(baseline.get(key), (int, float))
            and not isinstance(baseline.get(key), bool)
        }
        for arm, summary in summaries.items()
        if arm != baseline_arm
    }
    metric_panel = (
        _validate_metric_panel(config["metric_panel"]) if schema == PANEL_CONFIG_SCHEMA else None
    )
    result = {
        "schema_version": PANEL_RESULT_SCHEMA if schema == PANEL_CONFIG_SCHEMA else RESULT_SCHEMA,
        "status": "pass"
        if all(value["status"] == "pass" for value in arm_results.values())
        else "fail",
        "scientific_question": config["scientific_question"],
        "expected_attempts_per_arm": expected_attempts,
        "baseline_arm": baseline_arm,
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "config": pin_record(config_path, repo),
        "arms": {
            arm: {
                "result": artifact_record(output_dir / arm / "result.json"),
                "summary": summaries[arm],
            }
            for arm in sorted(arm_results)
        },
        "delta_from_argmax": deltas,
        "candidate_selection": False,
        "training_or_generation_calls": 0,
        "heldout_product_structures_accessed": False,
        "nonclaims": config["nonclaims"],
    }
    if metric_panel is not None:
        result["metric_panel"] = metric_panel
        result["panel_comparison"] = _panel_comparison(
            summaries,
            baseline_arm=baseline_arm,
            panel=metric_panel,
        )
        result["delta_from_baseline"] = result.pop("delta_from_argmax")
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise UgiDevelopmentRealismComparisonError("one or more development assessments failed")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    run_ugi_development_realism_comparison(
        args.config.resolve(),
        args.repo.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "PANEL_CONFIG_SCHEMA",
    "PANEL_RESULT_SCHEMA",
    "RESULT_SCHEMA",
    "UgiDevelopmentRealismComparisonError",
    "run_ugi_development_realism_comparison",
]
