"""Apply the frozen HeLa applicability policy to the branched exploration pool."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.potency import ugi_production_full_support_rescoring as v2
from forge.potency.ugi_hela_potency_diagnostic import (
    FrozenHeLaOracleWorker,
    HeLaBatchPredictor,
    HeLaPotencyDiagnosticPolicy,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_branch_exploration_applicability_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_branch_exploration_applicability.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_branch_exploration_applicability_ledger.v1"
EXPECTED_IMPLEMENTATION = {"runner", "source", "tests"}


class UgiBranchExplorationApplicabilityError(RuntimeError):
    """Raised when branch applicability is not evaluated under the frozen policy."""


def _prepare_branch_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_draws: int
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Mask chemically inadmissible terminals and retain branch diagnostics."""

    if len(rows) != expected_draws:
        raise UgiBranchExplorationApplicabilityError("branch generation denominator changed")
    prepared: list[dict[str, Any]] = []
    metadata: dict[int, dict[str, Any]] = {}
    for source in rows:
        draw = int(source.get("draw_index", -1))
        if draw in metadata or draw not in range(expected_draws):
            raise UgiBranchExplorationApplicabilityError("branch draw coordinates changed")
        admission = source.get("terminal_chemical_admission")
        if not isinstance(admission, Mapping):
            raise UgiBranchExplorationApplicabilityError("terminal admission is absent")
        admitted = admission.get("admitted")
        exact_l1 = admission.get("exact_l1")
        raw_valid = admission.get("raw_molecule_valid")
        reason = admission.get("reason")
        if (
            not isinstance(admitted, bool)
            or not isinstance(exact_l1, bool)
            or not isinstance(raw_valid, bool)
            or not isinstance(reason, str)
            or not reason
        ):
            raise UgiBranchExplorationApplicabilityError("terminal admission is malformed")
        if admitted and (not exact_l1 or not raw_valid or reason != "admitted"):
            raise UgiBranchExplorationApplicabilityError(
                "admitted terminal has inconsistent evidence"
            )
        copied = dict(source)
        copied["arm_id"] = "branch_exploration"
        if not admitted:
            copied["native_terminal"] = None
        prepared.append(copied)
        descriptors = source.get("tail_component_descriptors")
        if descriptors is not None and not isinstance(descriptors, Mapping):
            raise UgiBranchExplorationApplicabilityError("tail descriptors are malformed")
        metadata[draw] = {
            "terminal_chemical_admitted": admitted,
            "terminal_admission_reason": reason,
            "raw_exact_l1": exact_l1,
            "raw_molecule_valid": raw_valid,
            "scheduled_branch_class": str(source.get("scheduled_branch_class", "")),
            "realized_carbon_branch_class": str(source.get("realized_carbon_branch_class") or ""),
            "exact_refit_corpus_product": bool(source.get("exact_refit_corpus_product")),
            "tail_component_descriptors": descriptors,
        }
    if set(metadata) != set(range(expected_draws)):
        raise UgiBranchExplorationApplicabilityError("branch draw lattice is incomplete")
    return prepared, metadata


def _flatten_metadata(
    classified: list[dict[str, Any]], metadata: Mapping[int, Mapping[str, Any]]
) -> None:
    tail_roles = ("oxoester_aldehyde_body_tail", "isocyanide_tail")
    for row in classified:
        values = metadata[int(row["draw_index"])]
        row.update(
            {
                key: values[key]
                for key in (
                    "terminal_chemical_admitted",
                    "terminal_admission_reason",
                    "raw_exact_l1",
                    "raw_molecule_valid",
                    "scheduled_branch_class",
                    "realized_carbon_branch_class",
                    "exact_refit_corpus_product",
                )
            }
        )
        descriptors = values["tail_component_descriptors"]
        for prefix, role in zip(("aldehyde", "isocyanide"), tail_roles, strict=True):
            role_values = descriptors.get(role, {}) if isinstance(descriptors, Mapping) else {}
            for descriptor in (
                "heavy_atoms",
                "carbon_atoms",
                "carbon_branch_points",
                "adjacent_carbon_branch_edges",
                "carbon_carbon_double_bonds",
                "ester_carbonyls",
            ):
                row[f"{prefix}_{descriptor}"] = role_values.get(descriptor, "")
        if not values["terminal_chemical_admitted"]:
            row["reason"] = f"terminal_chemical_admission:{values['terminal_admission_reason']}"


def _is_realized_branch(row: Mapping[str, Any]) -> bool:
    return bool(row["realized_carbon_branch_class"]) and (
        row["realized_carbon_branch_class"] != "linear_tail_origins"
    )


def _is_long_ester_aldehyde(row: Mapping[str, Any]) -> bool:
    return bool(
        row["terminal_chemical_admitted"]
        and int(row["aldehyde_carbon_branch_points"] or 0) > 0
        and int(row["aldehyde_carbon_atoms"] or 0) >= 14
        and int(row["aldehyde_ester_carbonyls"] or 0) > 0
    )


def _failure_views(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    failures: Counter[str] = Counter()
    for row in rows:
        if not row["terminal_chemical_admitted"] or row["overall_bin"] == "interpolative":
            continue
        for item in str(row["view_bins"]).split(";"):
            if not item:
                continue
            view, value = item.split(":", 1)
            if value != "interpolative":
                failures[f"{view}:{value}"] += 1
    return dict(sorted(failures.items()))


def build_branch_exploration_applicability(
    repo: Path,
    config_path: Path,
    *,
    predictor: HeLaBatchPredictor | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Classify and score the branch pool without changing any frozen policy."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = v2._load(config_path, label="branch applicability config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "frozen_before_branch_applicability_rescoring"
    ):
        raise UgiBranchExplorationApplicabilityError("unsupported branch applicability config")
    paths = {
        label: v2._pin(repo, record, label=label) for label, record in config["inputs"].items()
    }
    raw_implementation = config.get("implementation")
    if (
        not isinstance(raw_implementation, Mapping)
        or set(raw_implementation) != EXPECTED_IMPLEMENTATION
    ):
        raise UgiBranchExplorationApplicabilityError("implementation pins changed")
    implementation_paths = {
        label: v2._pin(repo, record, label=f"implementation.{label}")
        for label, record in raw_implementation.items()
    }
    generation = v2._load(paths["generation_result"], label="branch generation result")
    if generation.get("status") != "complete_full_corpus_branch_exploration_generation":
        raise UgiBranchExplorationApplicabilityError("branch generation status changed")
    if generation.get("artifacts", {}).get("terminal_ledger.jsonl.gz", {}).get(
        "sha256"
    ) != sha256_file(paths["terminal_ledger"]):
        raise UgiBranchExplorationApplicabilityError("generation does not pin its ledger")
    frozen = v2._load(paths["frozen_rescoring_config"], label="frozen rescoring config")
    if (
        frozen.get("schema_version") != "phase1_ugi_production_full_support_rescoring_config.v3"
        or frozen.get("status") != "frozen_before_chemically_admitted_full_support_oracle_rescoring"
    ):
        raise UgiBranchExplorationApplicabilityError("frozen rescoring policy changed")
    policy_path = v2._pin(
        repo, frozen["inputs"]["hela_diagnostic_policy"], label="hela diagnostic policy"
    )
    policy = HeLaPotencyDiagnosticPolicy(repo, policy_path)
    pattern_policies = frozen["pattern_policies"]
    extended = v2._extended_scales(policy, frozen["extended_calibration_scales"])
    scales: dict[str, Any] = {**policy.scales, **extended}

    source_rows = v2._read_jsonl(paths["terminal_ledger"])
    prepared, metadata = _prepare_branch_rows(
        source_rows, expected_draws=int(config["production_contract"]["draws"])
    )
    classified = v2._classify(prepared, policy, pattern_policies)
    _flatten_metadata(classified, metadata)
    unique = v2._unique_scoring_rows(classified)
    if predictor is None:
        with FrozenHeLaOracleWorker(policy) as worker:
            receipts = v2._apply_scores(
                classified,
                unique,
                worker,
                scales,
                int(config["scoring"]["batch_size"]),
            )
            ready_receipt = worker.ready_receipt_sha256
    else:
        receipts = v2._apply_scores(
            classified,
            unique,
            predictor,
            scales,
            int(config["scoring"]["batch_size"]),
        )
        ready_receipt = "test_predictor"

    fields = tuple(key for key in classified[0] if key != "candidate")
    ledger = v2._gzip_csv(classified, fields)
    admitted = [row for row in classified if row["terminal_chemical_admitted"]]
    branched = [row for row in admitted if _is_realized_branch(row)]
    long_ester = [row for row in admitted if _is_long_ester_aldehyde(row)]
    interpolative = [row for row in admitted if row["overall_bin"] == "interpolative"]
    branched_interpolative = [row for row in branched if row["overall_bin"] == "interpolative"]
    branched_scored = [row for row in branched if row["oracle_scored"]]
    long_ester_scored = [row for row in long_ester if row["oracle_scored"]]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_branch_exploration_frozen_policy_applicability_rescoring",
        "scope": dict(config["scope"]),
        "inputs": {
            **{
                label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
                for label, path in sorted(paths.items())
            },
            "hela_diagnostic_policy": {
                "path": str(policy_path.relative_to(repo)),
                "sha256": sha256_file(policy_path),
            },
            "config": {
                "path": str(config_path.relative_to(repo)),
                "sha256": sha256_file(config_path),
            },
        },
        "implementation": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(implementation_paths.items())
        },
        "policy": {
            "frozen_v3_policy_reused_without_changes": True,
            "terminal_chemical_admission_precedes_applicability": True,
            "all_product_and_role_views_must_be_interpolative": True,
            "exact_novelty_is_provenance_not_applicability": True,
            "branching_is_not_a_categorical_veto": True,
            "potency_changes_generation": False,
            "pattern_policies": pattern_policies,
        },
        "summary": {
            "attempts": len(classified),
            "terminal_chemical_admitted": len(admitted),
            "realized_carbon_branched": len(branched),
            "long_ester_branched_aldehydes": len(long_ester),
            "all_view_interpolative": len(interpolative),
            "realized_branched_all_view_interpolative": len(branched_interpolative),
            "realized_branched_oracle_scored": len(branched_scored),
            "realized_branched_calibrated": sum(
                bool(row["calibration_scale"]) for row in branched_scored
            ),
            "realized_branched_conservative_high": sum(
                bool(row["conservative_high_potency"]) for row in branched_scored
            ),
            "long_ester_branched_aldehyde_oracle_scored": len(long_ester_scored),
            "long_ester_branched_aldehyde_conservative_high": sum(
                bool(row["conservative_high_potency"]) for row in long_ester_scored
            ),
            "unique_oracle_calls": len(unique),
            "reason_counts": dict(
                sorted(Counter(str(row["reason"]) for row in classified).items())
            ),
            "branched_pattern_counts": dict(
                sorted(Counter(str(row["pattern_id"]) for row in branched_scored).items())
            ),
            "branched_authority_tiers": dict(
                sorted(Counter(str(row["authority_tier"]) for row in branched_scored).items())
            ),
            "applicability_failure_views": _failure_views(admitted),
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
        "decision": {
            "branched_molecules_may_enter_applicability_domain": True,
            "applicability_boundary_relaxed": False,
            "branch_exploration_potency_claim_authorized_globally": False,
            "next_gate": "route_blinded_diversity_selection_by_authority_tier",
        },
        "nonclaims": [
            "Branching alone does not authorize an oracle score.",
            "A raw oracle score does not establish prospective potency.",
            "Exploratory novelty tiers do not establish broad branched-tail generalization.",
            "This rescore does not change molecular generation or select a synthesis panel.",
        ],
    }
    result = {
        **content,
        "result_sha256": hashlib.sha256(v2._stable_json(content).encode()).hexdigest(),
    }
    return result, ledger


__all__ = [
    "UgiBranchExplorationApplicabilityError",
    "build_branch_exploration_applicability",
]
