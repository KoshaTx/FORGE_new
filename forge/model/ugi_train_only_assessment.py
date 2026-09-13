"""Exact Ugi assessment against training identities without inspecting held-out structures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import resolve_pin
from forge.core.io import iter_csv
from forge.model.common_ugi_benchmark import (
    UGI_PROGRAM_ID,
    CommonUgiAttempt,
    adjudicate_ugi_attempts,
)
from forge.model.reaction_program_evaluation import (
    ReactionProgramEvaluationError,
    _CanonicalMemo,
    evaluate_reaction_program_samples,
)


class UgiTrainOnlyAssessmentError(ValueError):
    """Training identities violate the pinned, train-only assessment contract."""


@dataclass(frozen=True)
class UgiTrainingIdentities:
    """Constitutional identities and their train-only loading receipt."""

    training_products: frozenset[str]
    training_components: Mapping[str, frozenset[str]]
    provenance: dict[str, Any]


def load_ugi_training_identities(
    assignments_path: Path, adapter: Ugi3AssemblyAdapter, expected_sha256: str
) -> UgiTrainingIdentities:
    """Mask non-training rows before reading any molecular field, then canonicalize training.

    Hashing authenticates the whole assignment file; only TRAIN structures are interpreted. Existing
    family-fold columns must agree with the product training fold. Duplicate source strings and
    stereoisomers contribute a single constitutional identity, following the shared evaluator.
    """

    if adapter.reaction_id != UGI_PROGRAM_ID:
        raise UgiTrainOnlyAssessmentError("training assessment requires the Ugi assembly adapter")
    pin = {"path": assignments_path.name, "sha256": expected_sha256}
    path = resolve_pin(pin, assignments_path.parent, label="Ugi training assignments")
    roles = adapter.roles
    if not roles or len(set(roles)) != len(roles):
        raise UgiTrainOnlyAssessmentError("Ugi adapter roles are empty or duplicated")
    products: set[str] = set()
    components: dict[str, set[str]] = {role: set() for role in roles}
    canonical = _CanonicalMemo()
    rows_seen = train_rows = masked_rows = 0
    family_fold_checks = {role: 0 for role in roles}
    for row_number, row in enumerate(iter_csv(path), start=2):
        rows_seen += 1
        fold = row.get("primary_product_fold")
        if not isinstance(fold, str) or not fold:
            raise UgiTrainOnlyAssessmentError(f"assignment row {row_number} lacks a product fold")
        if fold != "train":
            masked_rows += 1
            continue
        train_rows += 1
        for role in roles:
            field = f"{role}_family_fold"
            if field in row:
                if row[field] != "train":
                    raise UgiTrainOnlyAssessmentError(
                        f"training assignment row {row_number} has inconsistent {field}"
                    )
                family_fold_checks[role] += 1
        for field, destination in [
            ("canonical_product_smiles", products),
            *((f"{role}_smiles", components[role]) for role in roles),
        ]:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise UgiTrainOnlyAssessmentError(
                    f"training assignment row {row_number} lacks {field}"
                )
            try:
                destination.add(canonical(value))
            except ReactionProgramEvaluationError as error:
                raise UgiTrainOnlyAssessmentError(
                    f"training assignment row {row_number} has invalid {field}: {error}"
                ) from error
    if not products or any(not values for values in components.values()):
        raise UgiTrainOnlyAssessmentError("Ugi training identity support is empty")
    # Do not authenticate changed bytes after interpreting a different file version.
    resolve_pin(pin, assignments_path.parent, label="Ugi training assignments after loading")
    return UgiTrainingIdentities(
        training_products=frozenset(products),
        training_components=MappingProxyType(
            {role: frozenset(values) for role, values in components.items()}
        ),
        provenance={
            "schema_version": "forge.ugi_train_only_identities.v1",
            "assignments": {
                "path": str(assignments_path),
                "sha256": expected_sha256,
                "bytes": path.stat().st_size,
            },
            "registry": {
                "path": str(adapter.registry_path),
                "sha256": adapter.registry_sha256,
            },
            "program_id": UGI_PROGRAM_ID,
            "roles": list(roles),
            "rows_seen": rows_seen,
            "train_rows": train_rows,
            "non_train_rows_masked_before_structure_access": masked_rows,
            "unique_training_products": len(products),
            "unique_training_components_by_role": {
                role: len(values) for role, values in components.items()
            },
            "training_family_fold_checks_by_role": family_fold_checks,
            "identity_convention": "connected RDKit canonical SMILES with isomericSmiles=False",
            "heldout_structures_interpreted": False,
        },
    )


def assess_ugi_train_only_attempts(
    attempts: Sequence[CommonUgiAttempt],
    *,
    adapter: Ugi3AssemblyAdapter,
    training_identities: UgiTrainingIdentities,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retain every attempt and report the shared exact-L1, diversity and novelty metrics.

    Component novelty abstains for ambiguous or missing exact decompositions. Its all-attempt
    incidence retains those attempts in the denominator; it is not a novelty estimate for them.
    Native valid-only and unique-decomposition-only rates are retained unchanged alongside it.
    """

    reference = training_identities.provenance
    if (
        adapter.reaction_id != UGI_PROGRAM_ID
        or reference.get("program_id") != UGI_PROGRAM_ID
        or reference.get("registry", {}).get("sha256") != adapter.registry_sha256
        or reference.get("roles") != list(adapter.roles)
        or set(training_identities.training_components) != set(adapter.roles)
        or not training_identities.training_products
        or any(not values for values in training_identities.training_components.values())
    ):
        raise UgiTrainOnlyAssessmentError(
            "training identities do not match the Ugi adapter support"
        )
    rows = adjudicate_ugi_attempts(attempts, adapter=adapter)
    evaluated = evaluate_reaction_program_samples(
        rows,
        training_products={UGI_PROGRAM_ID: training_identities.training_products},
        training_components={UGI_PROGRAM_ID: training_identities.training_components},
    )
    metrics = evaluated["overall"]
    denominator = len(rows)

    def incidence(count: int) -> dict[str, int | float]:
        return {"count": count, "denominator": denominator, "fraction": count / denominator}

    return rows, {
        "schema_version": "forge.ugi_train_only_assessment.v1",
        "method_id": rows[0]["method_id"],
        "seed": rows[0]["seed"],
        "attempts": denominator,
        "training_reference": deepcopy(reference),
        "metrics": metrics,
        "per_program": evaluated["per_program"],
        "all_attempt_metrics": {
            "whole_product_novel_to_train": incidence(metrics["whole_product_novel_to_train"]),
            "decomposed_products_with_any_novel_component": incidence(
                metrics["decomposed_products_with_any_novel_component"]
            ),
            "component_novel_to_train_by_role": {
                role: incidence(metrics["component_metrics_by_role"][role]["novel"])
                for role in adapter.roles
            },
            "component_novelty_evaluable": incidence(metrics["unique_decomposition_rows"]),
            "component_novelty_abstained": incidence(
                denominator - metrics["unique_decomposition_rows"]
            ),
        },
        "coverage_and_precision_reported": evaluated["coverage_and_precision_reported"],
        "reductive_amination_substructure_rate_reported": False,
        "heldout_structures_interpreted": False,
        "candidate_selection": False,
        "nonclaims": [
            "Exact L1 is registry-transform consistency, not synthesis-success probability.",
            "Train-only novelty does not establish held-component or held-product generalization.",
            "Ambiguous and unresolved decompositions abstain from component novelty assessment.",
            "All-attempt novelty incidence retains abstentions without assigning them novelty.",
        ],
    }


__all__ = [
    "UgiTrainOnlyAssessmentError",
    "UgiTrainingIdentities",
    "assess_ugi_train_only_attempts",
    "load_ugi_training_identities",
]
