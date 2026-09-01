"""Attribute measured-Ugi role-realism gaps without training or generation.

The frozen development panel reports distributional distances, but an absolute distance does not
say whether generated heads or tails are systematically too large, too small, too compact or too
asymmetric.  This diagnostic reuses the exact same group-balanced measured-train reference and
unambiguous exact-L1 attempt ledgers, then reports raw quantiles and signed standardized shifts for
every frozen role descriptor.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import iter_jsonl, read_json_object, write_json
from forge.model.common_lipid_realism import (
    RealismPolicy,
    build_ugi_development_realism_reference,
    fit_robust_descriptor_scale,
)
from forge.model.ugi_development_realism import (
    ROLE_DESCRIPTOR_NAMES,
    ugi_role_descriptor_vector,
)

CONFIG_SCHEMA = "forge.ugi_role_realism_gap_attribution_config.v1"
RESULT_SCHEMA = "forge.ugi_role_realism_gap_attribution.v1"
ASSESSED_HEADER_SCHEMA = "forge.common_ugi_assessed_attempts.v1"
ASSESSED_ROW_SCHEMA = "forge.common_ugi_assessed_attempt.v1"

ROLE_GROUPS = {
    "amine_head": tuple(range(0, 10)),
    "aldehyde_body_tail": tuple(range(10, 21)),
    "isocyanide_tail": tuple(range(21, 32)),
    "cross_tail": tuple(range(32, 35)),
    "whole_component_system": (35,),
}


class UgiRoleRealismGapAttributionError(ValueError):
    """A pinned input or role-realism attribution contract is malformed."""


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise UgiRoleRealismGapAttributionError("descriptor values are empty or non-finite")
    return {
        "minimum": float(values.min()),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def _load_exact_role_vectors(
    path: Path,
    *,
    expected_method: str,
    expected_seed: int,
    expected_attempts: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    records = list(iter_jsonl(path))
    if not records or not isinstance(records[0], Mapping):
        raise UgiRoleRealismGapAttributionError("assessed attempt ledger is empty")
    header = records.pop(0)
    if (
        set(header) != {"schema_version", "rows"}
        or header.get("schema_version") != ASSESSED_HEADER_SCHEMA
    ):
        raise UgiRoleRealismGapAttributionError("assessed attempt ledger header changed")
    if header.get("rows") != len(records) or len(records) != expected_attempts:
        raise UgiRoleRealismGapAttributionError("assessed attempt denominator changed")

    attempt_vectors: list[np.ndarray] = []
    unique_vectors: dict[str, np.ndarray] = {}
    excluded_ambiguous = 0
    excluded_not_exact = 0
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise UgiRoleRealismGapAttributionError("assessed attempt row is not an object")
        if (
            row.get("schema_version") != ASSESSED_ROW_SCHEMA
            or row.get("attempt_index") != index
            or row.get("method_id") != expected_method
            or row.get("seed") != expected_seed
        ):
            raise UgiRoleRealismGapAttributionError(
                "assessed attempt identity, order or schema changed"
            )
        if row.get("exact_l1_program") is not True:
            excluded_not_exact += 1
            continue
        traces = row.get("exact_l1_traces")
        if not isinstance(traces, list) or not traces:
            raise UgiRoleRealismGapAttributionError("exact-L1 attempt lacks verified traces")
        if len(traces) != 1:
            excluded_ambiguous += 1
            continue
        components = traces[0].get("components_by_role")
        if not isinstance(components, Mapping):
            raise UgiRoleRealismGapAttributionError("exact-L1 trace lacks role components")
        vector = ugi_role_descriptor_vector(
            {str(role): str(smiles) for role, smiles in components.items()}
        )
        attempt_vectors.append(vector)
        canonical = row.get("canonical_smiles")
        if not isinstance(canonical, str) or not canonical:
            raise UgiRoleRealismGapAttributionError("exact-L1 attempt lacks product identity")
        unique_vectors.setdefault(canonical, vector)
    if not attempt_vectors or not unique_vectors:
        raise UgiRoleRealismGapAttributionError(
            "no unambiguous exact-L1 products support gap attribution"
        )
    return (
        np.asarray(attempt_vectors, dtype=np.float64),
        np.asarray(list(unique_vectors.values()), dtype=np.float64),
        {
            "attempts": expected_attempts,
            "unambiguous_exact_l1_attempts": len(attempt_vectors),
            "unique_unambiguous_exact_l1_products": len(unique_vectors),
            "excluded_not_exact_l1": excluded_not_exact,
            "excluded_ambiguous_exact_l1": excluded_ambiguous,
        },
    )


def _descriptor_rows(
    generated_raw: np.ndarray,
    reference_raw: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> list[dict[str, Any]]:
    generated = (generated_raw - center) / scale
    reference = (reference_raw - center) / scale
    grid = np.linspace(0.0, 1.0, 101)
    wasserstein = np.mean(
        np.abs(np.quantile(generated, grid, axis=0) - np.quantile(reference, grid, axis=0)),
        axis=0,
    )
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(ROLE_DESCRIPTOR_NAMES):
        generated_summary = _quantiles(generated_raw[:, index])
        reference_summary = _quantiles(reference_raw[:, index])
        mean_shift = float(generated[:, index].mean() - reference[:, index].mean())
        median_shift = float(np.median(generated[:, index]) - np.median(reference[:, index]))
        rows.append(
            {
                "descriptor": name,
                "role_group": next(
                    group for group, indices in ROLE_GROUPS.items() if index in indices
                ),
                "robust_scaling_center": float(center[index]),
                "robust_scaling_width": float(scale[index]),
                "reference_raw": reference_summary,
                "generated_raw": generated_summary,
                "generated_minus_reference_raw": {
                    key: float(generated_summary[key] - reference_summary[key])
                    for key in ("q10", "median", "q90", "mean")
                },
                "signed_standardized_mean_shift": mean_shift,
                "signed_standardized_median_shift": median_shift,
                "shift_direction": (
                    "higher" if mean_shift > 0.0 else "lower" if mean_shift < 0.0 else "matched"
                ),
                "normalized_wasserstein": float(wasserstein[index]),
            }
        )
    return rows


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ROLE_GROUPS:
        selected = [row for row in rows if row["role_group"] == group]
        wasserstein = np.asarray(
            [float(row["normalized_wasserstein"]) for row in selected], dtype=np.float64
        )
        shifts = np.asarray(
            [float(row["signed_standardized_mean_shift"]) for row in selected],
            dtype=np.float64,
        )
        result[group] = {
            "descriptor_count": len(selected),
            "mean_normalized_wasserstein": float(wasserstein.mean()),
            "mean_absolute_standardized_mean_shift": float(np.abs(shifts).mean()),
            "root_mean_square_standardized_mean_shift": float(np.sqrt(np.mean(shifts**2))),
        }
    return result


def _arm_summary(
    attempt_raw: np.ndarray,
    unique_raw: np.ndarray,
    reference_raw: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_rows = _descriptor_rows(
        attempt_raw,
        reference_raw,
        center=center,
        scale=scale,
    )
    unique_rows = _descriptor_rows(
        unique_raw,
        reference_raw,
        center=center,
        scale=scale,
    )
    return {
        "ledger": dict(ledger),
        "attempt_weighted": {
            "by_descriptor": attempt_rows,
            "by_role_group": _group_summary(attempt_rows),
        },
        "unique_product_weighted": {
            "by_descriptor": unique_rows,
            "by_role_group": _group_summary(unique_rows),
        },
    }


def _comparison(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    baseline_arm: str,
    top_k: int,
) -> dict[str, Any]:
    baseline = arms[baseline_arm]
    rankings: dict[str, Any] = {}
    for weighting in ("attempt_weighted", "unique_product_weighted"):
        baseline_rows = baseline[weighting]["by_descriptor"]
        ranked = sorted(
            baseline_rows,
            key=lambda row: (-float(row["normalized_wasserstein"]), str(row["descriptor"])),
        )
        rankings[weighting] = {
            "top_baseline_mismatches": ranked[:top_k],
            "dominant_descriptor": ranked[0]["descriptor"],
            "dominant_role_group": ranked[0]["role_group"],
        }

    arm_deltas: dict[str, Any] = {}
    for arm_id, arm in sorted(arms.items()):
        if arm_id == baseline_arm:
            continue
        by_weighting: dict[str, Any] = {}
        for weighting in ("attempt_weighted", "unique_product_weighted"):
            baseline_by_name = {
                row["descriptor"]: row for row in baseline[weighting]["by_descriptor"]
            }
            arm_by_name = {row["descriptor"]: row for row in arm[weighting]["by_descriptor"]}
            by_weighting[weighting] = {
                name: {
                    "normalized_wasserstein_delta": float(
                        arm_by_name[name]["normalized_wasserstein"]
                        - baseline_by_name[name]["normalized_wasserstein"]
                    ),
                    "absolute_standardized_mean_shift_delta": float(
                        abs(arm_by_name[name]["signed_standardized_mean_shift"])
                        - abs(baseline_by_name[name]["signed_standardized_mean_shift"])
                    ),
                }
                for name in ROLE_DESCRIPTOR_NAMES
            }
        arm_deltas[arm_id] = by_weighting
    return {
        "baseline_arm": baseline_arm,
        "rankings": rankings,
        "arm_minus_baseline": arm_deltas,
    }


def run_ugi_role_realism_gap_attribution(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the signed train-development role mismatch attribution."""

    config = read_json_object(
        config_path,
        error=UgiRoleRealismGapAttributionError,
        label="Ugi role-realism gap attribution config",
    )
    expected_fields = {
        "schema_version",
        "scientific_question",
        "inputs",
        "arms",
        "baseline_arm",
        "expected_attempts",
        "top_k",
        "nonclaims",
    }
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != expected_fields:
        raise UgiRoleRealismGapAttributionError("gap attribution config fields changed")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {
        "development_realism_config",
        "realism_comparison_result",
    }:
        raise UgiRoleRealismGapAttributionError("gap attribution input pins changed")
    inputs = {label: resolve_pin(pin, repo, label=label) for label, pin in raw_inputs.items()}

    development = read_json_object(
        inputs["development_realism_config"],
        error=UgiRoleRealismGapAttributionError,
        label="development realism config",
    )
    reference_contract = development.get("reference")
    development_inputs = development.get("inputs")
    if (
        development.get("schema_version") != "forge.ugi_development_lipid_realism_config.v1"
        or not isinstance(reference_contract, Mapping)
        or set(reference_contract) != {"kind", "rows_per_group_per_partition", "split_seed"}
        or reference_contract.get("kind") != "source_adjudicated_measured_ugi_train_group_balanced"
        or not isinstance(development_inputs, Mapping)
        or "ugi_assignments" not in development_inputs
    ):
        raise UgiRoleRealismGapAttributionError("development realism contract changed")
    assignments = resolve_pin(development_inputs["ugi_assignments"], repo, label="ugi_assignments")
    policy = RealismPolicy.from_mapping(development["policy"])
    reference = build_ugi_development_realism_reference(
        assignments,
        policy,
        rows_per_group_per_partition=int(reference_contract["rows_per_group_per_partition"]),
        split_seed=int(reference_contract["split_seed"]),
    )
    scaling_raw = np.asarray(
        [ugi_role_descriptor_vector(row.components()) for row in reference.scaling],
        dtype=np.float64,
    )
    evaluation_raw = np.asarray(
        [ugi_role_descriptor_vector(row.components()) for row in reference.evaluation],
        dtype=np.float64,
    )
    scale = fit_robust_descriptor_scale(scaling_raw)

    expected_attempts = config["expected_attempts"]
    top_k = config["top_k"]
    if (
        isinstance(expected_attempts, bool)
        or not isinstance(expected_attempts, int)
        or expected_attempts <= 0
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
        or top_k > len(ROLE_DESCRIPTOR_NAMES)
    ):
        raise UgiRoleRealismGapAttributionError("attempt denominator or top_k is invalid")
    raw_arms = config["arms"]
    if not isinstance(raw_arms, Mapping) or not raw_arms:
        raise UgiRoleRealismGapAttributionError("gap attribution arms are empty")
    baseline_arm = str(config["baseline_arm"])
    if baseline_arm not in raw_arms:
        raise UgiRoleRealismGapAttributionError("baseline arm is absent")

    arms: dict[str, Any] = {}
    arm_input_pins: dict[str, Any] = {}
    for arm_id, arm in sorted(raw_arms.items()):
        if (
            not isinstance(arm_id, str)
            or not isinstance(arm, Mapping)
            or set(arm) != {"assessed_attempts", "method_id", "seed"}
        ):
            raise UgiRoleRealismGapAttributionError("gap attribution arm fields changed")
        ledger_path = resolve_pin(
            arm["assessed_attempts"], repo, label=f"{arm_id}_assessed_attempts"
        )
        attempt_raw, unique_raw, ledger = _load_exact_role_vectors(
            ledger_path,
            expected_method=str(arm["method_id"]),
            expected_seed=int(arm["seed"]),
            expected_attempts=expected_attempts,
        )
        arms[arm_id] = _arm_summary(
            attempt_raw,
            unique_raw,
            evaluation_raw,
            center=scale.center,
            scale=scale.scale,
            ledger=ledger,
        )
        arm_input_pins[arm_id] = pin_record(ledger_path, repo)

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "scientific_question": config["scientific_question"],
        "config": pin_record(config_path, repo),
        "implementation": pin_record(Path(__file__), repo),
        "inputs": {
            **{label: pin_record(path, repo) for label, path in sorted(inputs.items())},
            "ugi_assignments": pin_record(assignments, repo),
            "arm_assessed_attempts": arm_input_pins,
        },
        "reference": {
            "scaling_rows": len(reference.scaling),
            "evaluation_rows": len(reference.evaluation),
            "selection_sha256": reference.realism.audit["selection_sha256"],
            "population": reference.realism.audit["reference_population"],
            "calibration_or_heldout_product_structures_accessed": False,
        },
        "arms": arms,
        "comparison": _comparison(arms, baseline_arm=baseline_arm, top_k=top_k),
        "candidate_selection": False,
        "training_generation_repair_retry_route_or_oracle_calls": 0,
        "heldout_product_structures_accessed": False,
        "nonclaims": config["nonclaims"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise UgiRoleRealismGapAttributionError(f"output already exists: {output_path}")
    write_json(output_path, result)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    run_ugi_role_realism_gap_attribution(
        args.config.resolve(),
        args.repo.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiRoleRealismGapAttributionError",
    "run_ugi_role_realism_gap_attribution",
]
