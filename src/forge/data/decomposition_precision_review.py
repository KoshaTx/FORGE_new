"""Score completed blinded M0-05 chemist-review annotations.

This module does not infer family dispositions. It reports precision,
specificity, confidence intervals, and reviewer agreement for human sign-off.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.data.decomposition_precision_audit import BLIND_FIELDS
from forge.data.r1_prime_audit import AuditError, sha256_file

RESULT_SCHEMA_VERSION = "m0_05_human_review_scoring.v1"
BOOLEAN_FIELDS = (
    "plausible_final_assembly",
    "component_identities_and_roles_correct",
    "reactive_atoms_correct",
    "exact_forward_reconstruction",
    "plausible_alternative_decomposition",
)
PASS_FIELDS = (
    "plausible_final_assembly",
    "component_identities_and_roles_correct",
    "reactive_atoms_correct",
    "exact_forward_reconstruction",
)
ALLOWED_BOOLEAN_VALUES = {"yes", "no", "uncertain"}
NEGATIVE_CONTROL_TYPES = {
    "role_swap",
    "cross_record_component_swap",
    "wrong_reaction_family",
    "non_ugi_as_ugi",
    "wrong_attachment_site",
}


def _read_csv(path: Path, description: str) -> list[dict[str, str]]:
    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise AuditError(f"{description} not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AuditError(f"{description} has no CSV header: {path}")
        rows = list(reader)
    return rows


def _read_answer_key(path: Path) -> dict[str, dict[str, str]]:
    try:
        handle = gzip.open(path, "rt", newline="")
    except FileNotFoundError as exc:
        raise AuditError(f"M0-05 answer key not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {
            "review_id",
            "sampling_frame",
            "control_type",
            "expected_reaction_family",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AuditError(
                "M0-05 answer key is missing columns: "
                f"{sorted(required.difference(reader.fieldnames or ()))}"
            )
        rows = list(reader)
    by_id = {row["review_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise AuditError("M0-05 answer key contains duplicate review IDs")
    return by_id


def _normalize_annotation_rows(
    path: Path,
    expected_ids: set[str],
) -> tuple[str, dict[str, dict[str, str]]]:
    rows = _read_csv(path, "reviewer annotation file")
    if not rows:
        raise AuditError(f"reviewer annotation file is empty: {path}")
    missing_columns = set(BLIND_FIELDS).difference(rows[0])
    if missing_columns:
        raise AuditError(f"reviewer annotation file is missing columns: {sorted(missing_columns)}")
    normalized: dict[str, dict[str, str]] = {}
    reviewer_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        review_id = row["review_id"].strip()
        if not review_id or review_id in normalized:
            raise AuditError(f"{path}: row {row_number} has an empty or duplicate review_id")
        reviewer_id = row["reviewer_id"].strip()
        if not reviewer_id:
            raise AuditError(f"{path}: row {row_number} has no reviewer_id")
        reviewer_ids.add(reviewer_id)
        clean = dict(row)
        clean["review_id"] = review_id
        clean["reviewer_id"] = reviewer_id
        for field in BOOLEAN_FIELDS:
            value = row[field].strip().lower()
            if value not in ALLOWED_BOOLEAN_VALUES:
                raise AuditError(
                    f"{path}: row {row_number} field {field} must be one of "
                    f"{sorted(ALLOWED_BOOLEAN_VALUES)}"
                )
            clean[field] = value
        try:
            confidence = int(row["confidence"])
        except ValueError as exc:
            raise AuditError(
                f"{path}: row {row_number} confidence must be an integer from 1 to 5"
            ) from exc
        if confidence not in range(1, 6):
            raise AuditError(f"{path}: row {row_number} confidence must be an integer from 1 to 5")
        clean["confidence"] = str(confidence)
        normalized[review_id] = clean
    if len(reviewer_ids) != 1:
        raise AuditError(f"{path}: every row must contain the same reviewer_id")
    if set(normalized) != expected_ids:
        missing = sorted(expected_ids.difference(normalized))
        extra = sorted(set(normalized).difference(expected_ids))
        raise AuditError(
            f"{path}: review ID set does not match the answer key; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return next(iter(reviewer_ids)), normalized


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if successes < 0 or total < 0 or successes > total:
        raise AuditError("Wilson interval counts must satisfy 0 <= successes <= total")
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "rate": None,
            "lower": None,
            "upper": None,
            "method": "wilson_95",
        }
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "rate": proportion,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "method": "wilson_95",
    }


def _case_accepted(annotation: Mapping[str, str]) -> bool:
    return all(annotation[field] == "yes" for field in PASS_FIELDS)


def _control_detected(
    control_type: str,
    annotation: Mapping[str, str],
) -> bool:
    if control_type == "role_swap":
        return annotation["component_identities_and_roles_correct"] == "no"
    if control_type == "cross_record_component_swap":
        return (
            annotation["component_identities_and_roles_correct"] == "no"
            or annotation["exact_forward_reconstruction"] == "no"
        )
    if control_type in {"wrong_reaction_family", "non_ugi_as_ugi"}:
        return (
            annotation["plausible_final_assembly"] == "no"
            or annotation["exact_forward_reconstruction"] == "no"
        )
    if control_type == "wrong_attachment_site":
        return annotation["reactive_atoms_correct"] == "no"
    raise AuditError(f"unsupported fixed-negative control type: {control_type}")


def _per_reviewer_metrics(
    annotations: Mapping[str, Mapping[str, str]],
    answer_key: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    family_counts: dict[str, list[bool]] = defaultdict(list)
    control_counts: dict[str, list[bool]] = defaultdict(list)
    degenerate_challenges: list[bool] = []
    for review_id, answer in answer_key.items():
        annotation = annotations[review_id]
        control_type = answer["control_type"]
        if not control_type:
            family_counts[answer["expected_reaction_family"]].append(_case_accepted(annotation))
        elif control_type in NEGATIVE_CONTROL_TYPES:
            control_counts[control_type].append(_control_detected(control_type, annotation))
        elif control_type == "degenerate_reconstructing":
            degenerate_challenges.append(not _case_accepted(annotation))
        else:
            raise AuditError(f"answer key contains unknown control type {control_type!r}")
    return {
        "precision_per_family": {
            family: wilson_interval(sum(outcomes), len(outcomes))
            for family, outcomes in sorted(family_counts.items())
        },
        "specificity_per_negative_control_type": {
            control_type: wilson_interval(sum(outcomes), len(outcomes))
            for control_type, outcomes in sorted(control_counts.items())
        },
        "degenerate_reconstructing_challenge_rejection": wilson_interval(
            sum(degenerate_challenges),
            len(degenerate_challenges),
        ),
        "mean_confidence": sum(int(annotation["confidence"]) for annotation in annotations.values())
        / len(annotations),
    }


def _cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    if len(first) != len(second) or not first:
        raise AuditError("Cohen kappa inputs must be nonempty and have equal length")
    observed = sum(left == right for left, right in zip(first, second, strict=True)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    labels = set(first_counts) | set(second_counts)
    expected = sum(
        (first_counts[label] / len(first)) * (second_counts[label] / len(second))
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def _reviewer_agreement(
    first: Mapping[str, Mapping[str, str]],
    second: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    ordered_ids = sorted(first)
    for field in BOOLEAN_FIELDS:
        left = [first[review_id][field] for review_id in ordered_ids]
        right = [second[review_id][field] for review_id in ordered_ids]
        fields[field] = {
            "exact_agreement": sum(
                left_value == right_value
                for left_value, right_value in zip(left, right, strict=True)
            )
            / len(left),
            "cohen_kappa": _cohen_kappa(left, right),
        }
    return {
        "fields": fields,
        "mean_exact_agreement": sum(details["exact_agreement"] for details in fields.values())
        / len(fields),
    }


def _consensus_metrics(
    reviewers: Sequence[Mapping[str, Mapping[str, str]]],
    answer_key: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    family_counts: dict[str, list[bool]] = defaultdict(list)
    control_counts: dict[str, list[bool]] = defaultdict(list)
    degenerate_challenges: list[bool] = []
    for review_id, answer in answer_key.items():
        control_type = answer["control_type"]
        if not control_type:
            family_counts[answer["expected_reaction_family"]].append(
                all(_case_accepted(reviewer[review_id]) for reviewer in reviewers)
            )
        elif control_type in NEGATIVE_CONTROL_TYPES:
            control_counts[control_type].append(
                all(_control_detected(control_type, reviewer[review_id]) for reviewer in reviewers)
            )
        else:
            degenerate_challenges.append(
                all(not _case_accepted(reviewer[review_id]) for reviewer in reviewers)
            )
    return {
        "rule": (
            "conservative_unanimous: a positive passes only if every reviewer marks "
            "all four required fields yes; a negative control is detected only if "
            "every reviewer identifies its declared defect"
        ),
        "precision_per_family": {
            family: wilson_interval(sum(outcomes), len(outcomes))
            for family, outcomes in sorted(family_counts.items())
        },
        "specificity_per_negative_control_type": {
            control_type: wilson_interval(sum(outcomes), len(outcomes))
            for control_type, outcomes in sorted(control_counts.items())
        },
        "degenerate_reconstructing_challenge_rejection": wilson_interval(
            sum(degenerate_challenges),
            len(degenerate_challenges),
        ),
    }


def score_reviews(
    answer_key_path: Path,
    annotation_paths: Sequence[Path],
    output_path: Path,
    generated_utc: str,
) -> dict[str, Any]:
    """Validate and score two independent completed reviewer files."""

    if len(annotation_paths) != 2:
        raise AuditError("M0-05 scoring requires exactly two independent reviewer files")
    if output_path.exists():
        raise AuditError(f"M0-05 review-scoring output already exists: {output_path}")
    answer_key = _read_answer_key(answer_key_path)
    expected_ids = set(answer_key)
    reviewer_ids: list[str] = []
    reviewers: list[dict[str, dict[str, str]]] = []
    for path in annotation_paths:
        reviewer_id, annotations = _normalize_annotation_rows(path, expected_ids)
        reviewer_ids.append(reviewer_id)
        reviewers.append(annotations)
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise AuditError("M0-05 reviewer files must have distinct reviewer IDs")
    per_reviewer = {
        reviewer_id: _per_reviewer_metrics(annotations, answer_key)
        for reviewer_id, annotations in zip(reviewer_ids, reviewers, strict=True)
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-05",
        "generated_utc": generated_utc,
        "status": "human_review_scored_awaiting_family_disposition_signoff",
        "inputs": {
            "answer_key": {
                "path": str(answer_key_path),
                "sha256": sha256_file(answer_key_path),
            },
            "reviewer_annotations": [
                {
                    "reviewer_id": reviewer_id,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for reviewer_id, path in zip(
                    reviewer_ids,
                    annotation_paths,
                    strict=True,
                )
            ],
        },
        "review_rows": len(answer_key),
        "per_reviewer": per_reviewer,
        "consensus": _consensus_metrics(reviewers, answer_key),
        "reviewer_agreement": _reviewer_agreement(reviewers[0], reviewers[1]),
        "family_disposition": {
            family: "human_signoff_required"
            for family in sorted(
                {
                    answer["expected_reaction_family"]
                    for answer in answer_key.values()
                    if not answer["control_type"]
                }
            )
        },
        "claim_boundary": (
            "Scoring does not automatically admit a reaction family to L1 training. "
            "A human chemist must record admit, restrict, or reject after reviewing "
            "family precision, controls, disagreements, and degenerate cases."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        dir=output_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result
