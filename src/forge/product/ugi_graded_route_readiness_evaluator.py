"""Binary controller bridge for independently graded Ugi route readiness.

The matched SMC runner currently consumes a binary terminal utility.  This
adapter maps exact-complete or family-projected/all-current three-component
products to one and all lower evidence tiers to zero.  It retains the strict
route assessment beside the graded receipt and never represents the utility as
a probability of synthesis success.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.product.ugi_nonzero_guidance_runner import (
    GuidanceAssessmentContext,
    GuidanceRouteEvaluation,
)
from forge.product.ugi_production_zero_guidance_seam_v3 import (
    ProductionGuidanceRouteEvaluatorV3,
)
from forge.product.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.value.ugi_proposal_augmented_route_readiness import (
    EXACT,
    FAMILY_ALL,
    READINESS_UTILITY,
    UNRESOLVED,
    component_readiness_class,
    product_readiness_class,
)

GRADED_ROUTE_READINESS_POLICY = {
    "policy_id": "ugi_graded_route_readiness_binary_all_current_v1",
    "positive_component_tiers": [EXACT, FAMILY_ALL],
    "product_aggregation": "minimum across amine, aldehyde and isocyanide roles",
    "proposal_model_score_used": False,
    "proposal_only_hypotheses_authorized": False,
    "family_projection_promoted_to_exact": False,
    "utility_is_synthesis_success_probability": False,
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


UGI_GRADED_ROUTE_READINESS_POLICY_SHA256 = _sha256_payload(GRADED_ROUTE_READINESS_POLICY)


class UgiGradedRouteReadinessEvaluatorError(RuntimeError):
    """Raised when route-readiness evidence or terminal identity is malformed."""


def load_graded_component_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {(row["role"], row["canonical_smiles"]): row for row in rows}
    if len(output) != len(rows):
        raise UgiGradedRouteReadinessEvaluatorError("graded component identities collide")
    return output


def load_semantic_proposal_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the best frozen proposal receipt per component without its score."""

    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    order = {
        "exact_known_route": 4,
        "known_family_forward_consistent_projection": 3,
        "semantically_equivalent_known_family_projection": 3,
        "new_family_hypothesis_retained": 1,
        "ambiguous": 0,
        "rejected": 0,
    }
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        semantic = row.get("semantic_equivalence")
        if not isinstance(semantic, Mapping):
            raise UgiGradedRouteReadinessEvaluatorError("semantic proposal receipt is missing")
        key = (str(row["role"]), str(row["target_smiles"]))
        candidate = {
            "semantic_resolution_status": str(semantic["semantic_resolution_status"]),
            "graph_consistent_discovery_hypothesis": bool(
                row["graph_consistent_discovery_hypothesis"]
            ),
            "rank": int(row["rank"]),
            "reactants": list(row["reactants"]),
            "proposal_sha256": str(row["proposal_sha256"]),
            "model_score_used": False,
            "may_enter_synthesis_value": False,
        }
        current = output.get(key)
        if current is None or (
            order.get(candidate["semantic_resolution_status"], 0),
            -candidate["rank"],
        ) > (
            order.get(str(current["semantic_resolution_status"]), 0),
            -int(current["rank"]),
        ):
            output[key] = candidate
    return output


@dataclass
class GradedRouteReadinessEvaluator:
    """Wrap the strict planner and expose an evidence-safe readiness utility."""

    strict_evaluator: ProductionGuidanceRouteEvaluatorV3
    graded_components: dict[tuple[str, str], dict[str, str]]
    semantic_proposals: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    support_records: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        terminal: Any,
        context: GuidanceAssessmentContext,
    ) -> GuidanceRouteEvaluation:
        strict = self.strict_evaluator(terminal, context)
        strict_record = self.strict_evaluator.support_records[-1]
        native = native_completion_record_from_locked_terminal(terminal)
        components = native.get("component_smiles_by_role")
        if not isinstance(components, dict) or len(components) != 3:
            raise UgiGradedRouteReadinessEvaluatorError(
                "exact Ugi terminal lacks three reconstructed components"
            )
        component_receipts = []
        classes = []
        for role, smiles in sorted(components.items()):
            evidence = self.graded_components.get((role, smiles))
            evidence_class = (
                "not_in_graded_development_ledger"
                if evidence is None
                else evidence["graded_evidence_class"]
            )
            readiness = component_readiness_class(evidence_class)
            classes.append(readiness)
            component_receipts.append(
                {
                    "role": role,
                    "canonical_smiles": smiles,
                    "graded_evidence_class": evidence_class,
                    "route_readiness_class": readiness,
                    "program_family": None if evidence is None else evidence["program_family"],
                    "projection_leaf_status": (
                        None if evidence is None else evidence["projection_leaf_status"]
                    ),
                    "proposal_model_score_used": False,
                    "proposal_discovery": self.semantic_proposals.get((role, smiles)),
                    "proposal_discovery_sets_readiness": False,
                }
            )
        readiness_class = product_readiness_class(classes)
        utility = 1.0 if readiness_class in {EXACT, FAMILY_ALL} else 0.0
        receipt = {
            "schema_version": "forge.ugi_graded_route_readiness_receipt.v1",
            "terminal_sha256": terminal.terminal_sha256,
            "strict_assessment_receipt_sha256": strict.assessment_receipt_sha256,
            "strict_exact_route_utility": strict.route_completion_utility,
            "route_readiness_class": readiness_class,
            "ordinal_readiness_utility": READINESS_UTILITY.get(
                readiness_class, READINESS_UTILITY[UNRESOLVED]
            ),
            "binary_controller_utility": utility,
            "components": component_receipts,
            "policy": GRADED_ROUTE_READINESS_POLICY,
            "synthesis_success_probability": None,
        }
        receipt_sha256 = _sha256_payload(receipt)
        support_record = {
            **strict_record,
            "ordinal": len(self.support_records),
            "strict_assessment_receipt": strict_record["assessment_receipt"],
            "strict_assessment_receipt_sha256": strict.assessment_receipt_sha256,
            "strict_exact_potential": strict_record["potential"],
            "strict_exact_utility_bridge": strict_record["utility_bridge"],
            "strict_route_dossier_sha256": strict.route_dossier_sha256,
            "potential": {
                "schema_version": "forge.ugi_graded_route_readiness_potential.v1",
                "route_completion_utility": utility,
                "route_readiness_class": readiness_class,
                "true_planner_censor": False,
                "field_name_retained_for_runner_compatibility": True,
            },
            "utility_bridge": {
                "schema_version": "forge.ugi_graded_route_readiness_bridge.v1",
                "route_completion_utility": utility,
                "value_policy_id": UGI_GRADED_ROUTE_READINESS_POLICY_SHA256,
                "support_bonus": utility == 1.0,
                "censored": False,
            },
            "assessment_receipt": receipt,
            "assessment_receipt_sha256": receipt_sha256,
            "graded_route_readiness": receipt,
            "graded_route_readiness_receipt_sha256": receipt_sha256,
            "route_dossier_sha256": receipt_sha256 if utility == 1.0 else None,
            "scalar_value": None,
            "success_probability": None,
        }
        self.support_records.append(support_record)
        return GuidanceRouteEvaluation(
            route_completion_utility=utility,
            value_policy_id=UGI_GRADED_ROUTE_READINESS_POLICY_SHA256,
            usage=strict.usage,
            assessment_receipt_sha256=receipt_sha256,
            route_dossier_sha256=receipt_sha256 if utility == 1.0 else None,
            wall_seconds=strict.wall_seconds,
            gpu_device_seconds=strict.gpu_device_seconds,
        )


__all__ = [
    "GRADED_ROUTE_READINESS_POLICY",
    "GradedRouteReadinessEvaluator",
    "UGI_GRADED_ROUTE_READINESS_POLICY_SHA256",
    "UgiGradedRouteReadinessEvaluatorError",
    "load_graded_component_index",
    "load_semantic_proposal_index",
]
