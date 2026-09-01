"""Run the frozen method-blind whole-lipid structural-realism assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json, write_jsonl
from forge.model.common_lipid_realism import (
    ATTEMPT_ASSESSMENT_SCHEMA,
    RealismPolicy,
    assess_lipid_realism,
    build_realism_reference,
    build_ugi_development_realism_reference,
    build_ugi_realism_reference,
)
from forge.model.common_ugi_benchmark import adjudicate_ugi_attempts, load_attempt_ledger
from forge.model.ugi_development_realism import assess_ugi_role_realism

CONFIG_SCHEMA = "forge.common_lipid_realism_config.v1"
UGI_MATCHED_CONFIG_SCHEMA = "forge.ugi_matched_lipid_realism_config.v1"
UGI_DEVELOPMENT_CONFIG_SCHEMA = "forge.ugi_development_lipid_realism_config.v1"
RESULT_SCHEMA = "forge.common_lipid_realism_complete_assessment.v1"


class LipidRealismAssessmentError(ValueError):
    """A ledger cannot be assessed under the frozen method-blind realism protocol."""


def run_lipid_realism_assessment(
    config_path: Path,
    repo: Path,
    attempts_path: Path,
    output_dir: Path,
    *,
    method_id: str,
    seed: int,
    expected_attempts: int | None = None,
) -> dict[str, Any]:
    """Assess one method/seed ledger without generation, routes, or oracle calls."""

    config = read_json_object(
        config_path, error=LipidRealismAssessmentError, label="lipid realism config"
    )
    schema = config.get("schema_version")
    expected_fields = {
        "schema_version",
        "scientific_question",
        "inputs",
        "policy",
        "metrics",
        "nonclaims",
    }
    if schema in {UGI_MATCHED_CONFIG_SCHEMA, UGI_DEVELOPMENT_CONFIG_SCHEMA}:
        expected_fields.add("reference")
    if schema == UGI_DEVELOPMENT_CONFIG_SCHEMA:
        expected_fields.add("role_metrics")
    if (
        schema not in {CONFIG_SCHEMA, UGI_MATCHED_CONFIG_SCHEMA, UGI_DEVELOPMENT_CONFIG_SCHEMA}
        or set(config) != expected_fields
    ):
        raise LipidRealismAssessmentError("unsupported or malformed lipid realism config")
    metrics = config["metrics"]
    if (
        not isinstance(metrics, dict)
        or metrics.get("attempt_denominator_includes_invalid_failed_duplicate_and_out_of_support")
        is not True
        or metrics.get("fingerprint_and_descriptor_manifolds_reported_separately") is not True
        or metrics.get("no_result_selected_thresholds") is not True
        or metrics.get("qed_excluded") is not True
    ):
        raise LipidRealismAssessmentError("lipid realism metric guardrails changed")
    nonclaims = config["nonclaims"]
    if (
        not isinstance(nonclaims, list)
        or not nonclaims
        or any(not isinstance(value, str) or not value for value in nonclaims)
    ):
        raise LipidRealismAssessmentError("lipid realism nonclaims must be explicit")
    raw_inputs = config["inputs"]
    if schema == CONFIG_SCHEMA:
        expected_inputs = {"r0_constitutional", "r0_fold_assignments"}
    elif schema == UGI_MATCHED_CONFIG_SCHEMA:
        expected_inputs = {"ugi_assignments"}
    else:
        expected_inputs = {"ugi_assignments", "qualified_ugi_reactions"}
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != expected_inputs:
        raise LipidRealismAssessmentError("lipid realism input pins changed")
    inputs = {key: resolve_pin(value, repo, label=key) for key, value in raw_inputs.items()}
    policy = RealismPolicy.from_mapping(config["policy"])
    attempts = load_attempt_ledger(
        attempts_path,
        expected_method=method_id,
        expected_seed=seed,
        expected_attempts=expected_attempts,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LipidRealismAssessmentError(f"output directory is not empty: {output_dir}")
    development_reference = None
    role_contract = None
    if schema == CONFIG_SCHEMA:
        reference = build_realism_reference(
            inputs["r0_constitutional"], inputs["r0_fold_assignments"], policy
        )
    elif schema == UGI_MATCHED_CONFIG_SCHEMA:
        reference_contract = config["reference"]
        if not isinstance(reference_contract, dict) or set(reference_contract) != {
            "kind",
            "evaluation_fold",
        }:
            raise LipidRealismAssessmentError("Ugi realism reference contract changed")
        if reference_contract["kind"] != "source_adjudicated_measured_ugi":
            raise LipidRealismAssessmentError("unsupported Ugi realism reference kind")
        reference = build_ugi_realism_reference(
            inputs["ugi_assignments"],
            policy,
            evaluation_fold=str(reference_contract["evaluation_fold"]),
        )
    else:
        reference_contract = config["reference"]
        if not isinstance(reference_contract, dict) or set(reference_contract) != {
            "kind",
            "rows_per_group_per_partition",
            "split_seed",
        }:
            raise LipidRealismAssessmentError("Ugi development reference contract changed")
        if reference_contract["kind"] != "source_adjudicated_measured_ugi_train_group_balanced":
            raise LipidRealismAssessmentError("unsupported Ugi development reference kind")
        role_contract = config["role_metrics"]
        if not isinstance(role_contract, dict) or set(role_contract) != {
            "exact_trace_policy",
            "continuous_metrics",
            "sanity_shift_standardized_units",
            "heldout_product_access",
        }:
            raise LipidRealismAssessmentError("Ugi development role metric contract changed")
        if (
            role_contract["exact_trace_policy"] != "unambiguous_verified_exact_l1_only"
            or role_contract["continuous_metrics"]
            != [
                "normalized_wasserstein",
                "energy_distance",
                "rbf_mmd2",
                "nearest_reference_distance",
            ]
            or role_contract["heldout_product_access"] is not False
        ):
            raise LipidRealismAssessmentError("Ugi development role metric guardrails changed")
        development_reference = build_ugi_development_realism_reference(
            inputs["ugi_assignments"],
            policy,
            rows_per_group_per_partition=reference_contract["rows_per_group_per_partition"],
            split_seed=reference_contract["split_seed"],
        )
        reference = development_reference.realism
    rows, assessment = assess_lipid_realism(attempts, reference, policy)

    output_dir.mkdir(parents=True, exist_ok=True)
    assessed_path = output_dir / "assessed_attempts.jsonl.gz"
    write_jsonl(
        assessed_path,
        [
            {"schema_version": ATTEMPT_ASSESSMENT_SCHEMA, "rows": len(rows)},
            *rows,
        ],
    )
    ugi_role_realism = None
    ugi_assessed_path = None
    if development_reference is not None and role_contract is not None:
        adapter = Ugi3AssemblyAdapter.from_registry(inputs["qualified_ugi_reactions"])
        ugi_rows = adjudicate_ugi_attempts(attempts, adapter=adapter)
        ugi_assessed_path = output_dir / "ugi_assessed_attempts.jsonl.gz"
        write_jsonl(
            ugi_assessed_path,
            [
                {"schema_version": "forge.common_ugi_assessed_attempts.v1", "rows": len(ugi_rows)},
                *ugi_rows,
            ],
        )
        ugi_role_realism = assess_ugi_role_realism(
            ugi_rows,
            development_reference,
            policy,
            sanity_shift_standardized_units=float(role_contract["sanity_shift_standardized_units"]),
        )
    gates = {
        "attempt_denominator_preserved": len(attempts) == assessment["attempts"],
        "method_and_seed_preserved": assessment["method_id"] == method_id
        and int(assessment["seed"]) == seed,
        "reference_population_matches_contract": (
            assessment["reference"]["reference_population"].startswith("source-study-held-out")
            if schema == CONFIG_SCHEMA
            else assessment["reference"]["reference_population"].startswith(
                "source-adjudicated measured Ugi"
            )
        ),
        "training_only_descriptor_scaling": (
            assessment["reference"]["scaling_population"] == "R0_train only"
            if schema == CONFIG_SCHEMA
            else (
                assessment["reference"]["scaling_population"]
                == "source-adjudicated measured Ugi train fold only"
                if schema == UGI_MATCHED_CONFIG_SCHEMA
                else assessment["reference"]["scaling_population"].startswith(
                    "source-adjudicated measured Ugi train-fold group-balanced"
                )
            )
        ),
        "reference_selected_independently_of_method": assessment["reference"]["selection"].endswith(
            "independent of method outputs"
        ),
        "coverage_and_precision_reported_separately": assessment[
            "coverage_and_precision_reported_separately"
        ]
        is True,
        "invalid_failed_and_out_of_support_preserved": assessment[
            "attempt_denominator_includes_invalid_failed_and_out_of_support"
        ]
        is True,
        "qed_absent": assessment["qed_reported"] is False,
        "route_or_oracle_calls_during_assessment_zero": assessment["route_or_oracle_calls"] == 0,
        "candidate_selection_absent": assessment["candidate_selection"] is False,
    }
    if schema == UGI_DEVELOPMENT_CONFIG_SCHEMA:
        assert ugi_role_realism is not None
        gates.update(
            {
                "train_development_reference_is_group_balanced": assessment["reference"][
                    "grouping"
                ].endswith("equal selected group mass"),
                "calibration_and_heldout_product_structures_not_accessed": assessment["reference"][
                    "calibration_or_heldout_product_structures_accessed"
                ]
                is False,
                "role_realism_sanity_controls_pass": ugi_role_realism["status"] == "pass",
                "role_realism_uses_no_heldout_products": ugi_role_realism[
                    "heldout_product_structures_accessed"
                ]
                is False,
                "role_realism_route_or_oracle_calls_zero": ugi_role_realism["route_or_oracle_calls"]
                == 0,
                "role_realism_candidate_selection_absent": ugi_role_realism["candidate_selection"]
                is False,
            }
        )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "method_id": method_id,
        "seed": seed,
        "attempts": artifact_record(attempts_path),
        "assessed_attempts": artifact_record(assessed_path),
        "assessment": assessment,
        "ugi_role_realism": ugi_role_realism,
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "gates": gates,
        "candidate_selection": False,
    }
    if ugi_assessed_path is not None:
        result["ugi_assessed_attempts"] = artifact_record(ugi_assessed_path)
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise LipidRealismAssessmentError(f"lipid realism gates failed: {gates}")
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "UGI_DEVELOPMENT_CONFIG_SCHEMA",
    "UGI_MATCHED_CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "LipidRealismAssessmentError",
    "run_lipid_realism_assessment",
]
