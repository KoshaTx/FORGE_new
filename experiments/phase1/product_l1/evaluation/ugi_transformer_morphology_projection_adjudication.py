"""Adjudicate the verified paired Transformer morphology projection comparison."""

from __future__ import annotations

import gzip
import io
import json
import math
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_json_object, write_json

CONFIG_SCHEMA = "forge.ugi_transformer_morphology_projection_adjudication_config.v1"
RESULT_SCHEMA = "forge.ugi_transformer_morphology_projection_adjudication.v1"
REDUCED_METHOD = "forge_mixed_transformer_reduced_projection_seed0"
FULL_METHOD = "forge_mixed_transformer_full_role_morphology_seed0"
ATTEMPT_MEMBERS = {
    REDUCED_METHOD: "reduced_projection/assessment/common/assessed_attempts.jsonl.gz",
    FULL_METHOD: "full_role_morphology/assessment/common/assessed_attempts.jsonl.gz",
}


class UgiTransformerMorphologyProjectionAdjudicationError(ValueError):
    """The verified comparison or its adjudication contract changed."""


def _mcnemar_exact_two_sided(full_only: int, reduced_only: int) -> float:
    discordant = full_only + reduced_only
    if discordant == 0:
        return 1.0
    lower = min(full_only, reduced_only)
    numerator = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, float(2.0 * numerator / (1 << discordant)))


def _read_exact_l1_rows(
    archive: tarfile.TarFile,
    *,
    member: str,
    expected_method: str,
    expected_rows: int,
) -> np.ndarray:
    handle = archive.extractfile(member)
    if handle is None:
        raise UgiTransformerMorphologyProjectionAdjudicationError(
            f"comparison archive lacks {member}"
        )
    values: list[int] = []
    with gzip.GzipFile(fileobj=handle, mode="rb") as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8") as text:
            for line_number, line in enumerate(text, start=1):
                row = json.loads(line)
                if line_number == 1 and "rows" in row:
                    if int(row["rows"]) != expected_rows:
                        raise UgiTransformerMorphologyProjectionAdjudicationError(
                            f"{expected_method} header denominator changed"
                        )
                    continue
                index = len(values)
                if (
                    row.get("method_id") != expected_method
                    or int(row.get("attempt_index", -1)) != index
                    or not isinstance(row.get("exact_l1_program"), bool)
                ):
                    raise UgiTransformerMorphologyProjectionAdjudicationError(
                        f"{expected_method} row {index} changed"
                    )
                values.append(int(row["exact_l1_program"]))
    if len(values) != expected_rows:
        raise UgiTransformerMorphologyProjectionAdjudicationError(
            f"{expected_method} attempt denominator changed"
        )
    return np.asarray(values, dtype=np.int8)


def _paired_statistics(
    reduced: np.ndarray,
    full: np.ndarray,
    *,
    seed: int,
    bootstrap_replicates: int,
    confidence: float,
) -> dict[str, Any]:
    if reduced.shape != full.shape or reduced.ndim != 1:
        raise UgiTransformerMorphologyProjectionAdjudicationError(
            "paired exact-L1 vectors have different shapes"
        )
    count = int(reduced.size)
    both = int(((reduced == 1) & (full == 1)).sum())
    full_only = int(((reduced == 0) & (full == 1)).sum())
    reduced_only = int(((reduced == 1) & (full == 0)).sum())
    neither = count - both - full_only - reduced_only
    paired_difference = full.astype(np.int16) - reduced.astype(np.int16)
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        draw = generator.integers(0, count, size=count)
        bootstrap[index] = float(paired_difference[draw].mean())
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return {
        "attempts": count,
        "both_exact_l1": both,
        "full_only_exact_l1": full_only,
        "reduced_only_exact_l1": reduced_only,
        "neither_exact_l1": neither,
        "full_minus_reduced_exact_l1": float(paired_difference.mean()),
        "paired_bootstrap": {
            "seed": seed,
            "replicates": bootstrap_replicates,
            "confidence": confidence,
            "interval": [float(lower), float(upper)],
        },
        "mcnemar_exact_two_sided_p": _mcnemar_exact_two_sided(full_only, reduced_only),
    }


def adjudicate_ugi_transformer_morphology_projection(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Recompute the paired effect from the hash-pinned method-blind attempt ledgers."""

    config = read_json_object(
        config_path,
        error=UgiTransformerMorphologyProjectionAdjudicationError,
        label="Transformer morphology projection adjudication config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise UgiTransformerMorphologyProjectionAdjudicationError(
            "unsupported morphology projection adjudication config"
        )
    inputs = config.get("inputs")
    bootstrap = config.get("paired_bootstrap")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "comparison_details",
        "comparison_result",
    }:
        raise UgiTransformerMorphologyProjectionAdjudicationError(
            "adjudication inputs changed"
        )
    if not isinstance(bootstrap, Mapping):
        raise UgiTransformerMorphologyProjectionAdjudicationError(
            "paired bootstrap contract is missing"
        )
    expected_rows = int(config.get("expected_programs_per_method", 0))
    if (
        expected_rows != 3072
        or int(bootstrap.get("seed", -1)) != 20260827
        or int(bootstrap.get("replicates", 0)) != 20000
        or float(bootstrap.get("confidence", 0.0)) != 0.95
    ):
        raise UgiTransformerMorphologyProjectionAdjudicationError(
            "paired adjudication contract changed"
        )
    comparison_path = resolve_pin(inputs["comparison_result"], repo, label="comparison_result")
    details_path = resolve_pin(inputs["comparison_details"], repo, label="comparison_details")
    comparison = read_json_object(
        comparison_path,
        error=UgiTransformerMorphologyProjectionAdjudicationError,
        label="paired Transformer morphology comparison",
    )
    if (
        comparison.get("schema_version")
        != "forge.ugi_transformer_morphology_projection_comparison.v1"
        or comparison.get("status") != "complete"
        or int(comparison.get("programs_per_method", 0)) != expected_rows
        or not all(comparison.get("gates", {}).values())
    ):
        raise UgiTransformerMorphologyProjectionAdjudicationError(
            "paired comparison is incomplete or failed a gate"
        )
    with tarfile.open(details_path, mode="r") as archive:
        exact = {
            method: _read_exact_l1_rows(
                archive,
                member=member,
                expected_method=method,
                expected_rows=expected_rows,
            )
            for method, member in ATTEMPT_MEMBERS.items()
        }
    paired = _paired_statistics(
        exact[REDUCED_METHOD],
        exact[FULL_METHOD],
        seed=int(bootstrap["seed"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        confidence=float(bootstrap["confidence"]),
    )
    reported_delta = float(
        comparison["full_minus_reduced"]["exact_l1_yield_per_attempt"]
    )
    if not math.isclose(
        float(paired["full_minus_reduced_exact_l1"]),
        reported_delta,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise UgiTransformerMorphologyProjectionAdjudicationError(
            "paired attempt delta does not reproduce the comparison result"
        )
    selected_metrics = (
        "component_novelty_fraction",
        "descriptor_manifold_precision_per_attempt",
        "effective_component_count",
        "exact_l1_yield_per_attempt",
        "local_support_qualified_exact_l1_yield_per_attempt",
        "mean_pairwise_ecfp4_distance",
        "nitrogen_oxygen_bond_product_fraction_per_attempt",
        "oxygen_oxygen_bond_product_fraction_per_attempt",
        "realism_c2st_auc",
        "small_oxygen_ring_product_fraction_per_attempt",
        "unique_open_ended_exact_l1_products_per_attempt",
        "valid_fraction_per_attempt",
    )
    methods = comparison["methods"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "decision": str(comparison["decision"]),
        "paired_exact_l1": paired,
        "selected_metrics": {
            method: {metric: methods[method]["metrics"][metric] for metric in selected_metrics}
            for method in (REDUCED_METHOD, FULL_METHOD)
        },
        "full_minus_reduced": {
            metric: comparison["full_minus_reduced"][metric] for metric in selected_metrics
        },
        "comparison_gates": dict(comparison["gates"]),
        "inputs": {
            "comparison_details": pin_record(details_path, repo),
            "comparison_result": pin_record(comparison_path, repo),
        },
        "candidate_selection": False,
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "interpretation": {
            "primary_exact_l1": "positive_seed0_projection_effect",
            "overall_quality": "mixed_tradeoffs_require_replication",
        },
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_path, result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiTransformerMorphologyProjectionAdjudicationError",
    "adjudicate_ugi_transformer_morphology_projection",
]
