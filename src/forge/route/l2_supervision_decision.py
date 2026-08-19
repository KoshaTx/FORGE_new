"""Freeze the M0-09 L2 supervision and architecture decision."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file as _sha256_file

CONFIG_SCHEMA_VERSION = "m0_09_l2_supervision_decision_config.v2"
RESULT_SCHEMA_VERSION = "m0_09_l2_supervision_decision.v2"
RESULT_NAME = "l2_supervision_decision.json"


class L2SupervisionDecisionError(RuntimeError):
    """Raised when the M0-09 decision inputs or policy are invalid."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise L2SupervisionDecisionError(f"cannot load {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise L2SupervisionDecisionError(f"{label} must be a JSON object")
    return value


def _required_mapping(
    value: Any,
    *,
    label: str,
    fields: Sequence[str] = (),
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise L2SupervisionDecisionError(f"{label} must be an object")
    missing = set(fields) - set(value)
    if missing:
        raise L2SupervisionDecisionError(f"{label} is missing fields: {sorted(missing)}")
    return value


def _required_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise L2SupervisionDecisionError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _required_fraction(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise L2SupervisionDecisionError(f"{label} must be a number between zero and one")
    return float(value)


def _nested_integer(
    value: Mapping[str, Any],
    path: Sequence[str],
    *,
    label: str,
) -> int:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise L2SupervisionDecisionError(f"{label} is missing {'.'.join(path)}")
        current = current[key]
    return _required_integer(current, label=f"{label} {'.'.join(path)}")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen M0-09 decision configuration."""

    config = _load_json(path, label="L2 supervision decision config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise L2SupervisionDecisionError(f"config schema must be {CONFIG_SCHEMA_VERSION!r}")
    frozen_utc = config.get("frozen_utc")
    if not isinstance(frozen_utc, str) or not frozen_utc.endswith("Z"):
        raise L2SupervisionDecisionError("frozen_utc must be an explicit UTC timestamp")
    inputs = _required_mapping(
        config.get("inputs"),
        label="inputs",
    )
    if not inputs:
        raise L2SupervisionDecisionError("inputs must not be empty")
    for input_id, raw_spec in inputs.items():
        if not isinstance(input_id, str) or not input_id:
            raise L2SupervisionDecisionError("input identifiers must be nonempty strings")
        spec = _required_mapping(
            raw_spec,
            label=f"input {input_id}",
            fields=("path", "sha256", "schema_version"),
        )
        if (
            not isinstance(spec["path"], str)
            or not spec["path"]
            or Path(spec["path"]).is_absolute()
            or ".." in Path(spec["path"]).parts
        ):
            raise L2SupervisionDecisionError(f"input {input_id} path must be repository-relative")
        if not isinstance(spec["sha256"], str) or len(spec["sha256"]) != 64:
            raise L2SupervisionDecisionError(f"input {input_id} sha256 must contain 64 characters")
        if not isinstance(spec["schema_version"], str) or not spec["schema_version"]:
            raise L2SupervisionDecisionError(f"input {input_id} schema_version must be nonempty")

    gates = _required_mapping(
        config.get("architecture_gates"),
        label="architecture_gates",
        fields=(
            "joint_product_l1",
            "hybrid_l2",
            "monolithic_complete_route_decoder_necessary_conditions",
        ),
    )
    for gate_name, raw_gate in gates.items():
        gate = _required_mapping(raw_gate, label=f"gate {gate_name}")
        if not gate:
            raise L2SupervisionDecisionError(f"gate {gate_name} must not be empty")
        for field, value in gate.items():
            _required_integer(
                value,
                label=f"gate {gate_name}.{field}",
            )

    thresholds = _required_mapping(
        config.get("future_operational_closure_thresholds"),
        label="future_operational_closure_thresholds",
        fields=(
            "weighted_high_priority_motif_route_support_minimum",
            "eligible_candidate_complete_route_fraction_minimum",
            "eligible_candidate_decision_coverage_minimum",
            "eligible_candidate_missing_route_knowledge_fraction_maximum",
            "marginal_weighted_coverage_gain_maximum",
            "consecutive_saturation_rounds_required",
            "prospective_panel_complete_dossier_fraction_required",
            "prospective_panel_forward_consistency_fraction_required",
            "prospective_panel_current_terminal_closure_fraction_required",
            "high_risk_precursor_backup_route_fraction_required",
        ),
    )
    for field, value in thresholds.items():
        if field == "consecutive_saturation_rounds_required":
            _required_integer(value, label=field, minimum=1)
        else:
            _required_fraction(value, label=field)

    boundaries = _required_mapping(
        config.get("claims_boundary"),
        label="claims_boundary",
    )
    if not boundaries or any(value is not True for value in boundaries.values()):
        raise L2SupervisionDecisionError("every claims_boundary safeguard must be true")

    safeguards = _required_mapping(
        config.get("future_training_safeguards"),
        label="future_training_safeguards",
        fields=(
            "broad_corpus_replay_required",
            "masked_l1_loss_for_unannotated_structures",
            "ugi_only_finetuning_without_replay_allowed",
            "raw_route_model_likelihood_allowed_as_synthesis_value",
            "preprospective_synthesis_value_claim",
            "guidance_must_change_transition_probabilities_before_candidate_lock",
            "post_hoc_baseline_uses_same_prior_and_planner",
            "guidance_selection_requires_novelty_diversity_and_coverage_floors",
            "route_closure_strata",
        ),
    )
    required_true = (
        "broad_corpus_replay_required",
        "masked_l1_loss_for_unannotated_structures",
        "guidance_must_change_transition_probabilities_before_candidate_lock",
        "post_hoc_baseline_uses_same_prior_and_planner",
        "guidance_selection_requires_novelty_diversity_and_coverage_floors",
    )
    required_false = (
        "ugi_only_finetuning_without_replay_allowed",
        "raw_route_model_likelihood_allowed_as_synthesis_value",
    )
    if any(safeguards[field] is not True for field in required_true):
        raise L2SupervisionDecisionError("required future training safeguards must be true")
    if any(safeguards[field] is not False for field in required_false):
        raise L2SupervisionDecisionError("prohibited future training behaviors must be false")
    if safeguards["preprospective_synthesis_value_claim"] != (
        "evidence_weighted_route_completion_value_not_success_probability"
    ):
        raise L2SupervisionDecisionError(
            "preprospective synthesis-value claim is not frozen correctly"
        )
    strata = safeguards["route_closure_strata"]
    if not isinstance(strata, list) or strata != [
        "familiar_agile_components",
        "transferred_known_components",
        "genuinely_generated_components",
    ]:
        raise L2SupervisionDecisionError(
            "route_closure_strata must contain the three frozen strata"
        )
    return config


def _load_inputs(
    config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    root = repo_root.resolve()
    payloads: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    inputs = _required_mapping(config["inputs"], label="inputs")
    for input_id, raw_spec in sorted(inputs.items()):
        spec = _required_mapping(raw_spec, label=f"input {input_id}")
        path = (root / str(spec["path"])).resolve()
        if not path.is_relative_to(root):
            raise L2SupervisionDecisionError(f"input {input_id} escapes the repository root")
        observed_hash = _sha256_file(path)
        if observed_hash != spec["sha256"]:
            raise L2SupervisionDecisionError(
                f"input {input_id} hash mismatch: expected "
                f"{spec['sha256']}, observed {observed_hash}"
            )
        payload = _load_json(path, label=f"input {input_id}")
        if payload.get("schema_version") != spec["schema_version"]:
            raise L2SupervisionDecisionError(
                f"input {input_id} schema mismatch: expected "
                f"{spec['schema_version']!r}, observed "
                f"{payload.get('schema_version')!r}"
            )
        payloads[input_id] = payload
        records.append(
            {
                "input_id": input_id,
                "path": str(spec["path"]),
                "bytes": path.stat().st_size,
                "sha256": observed_hash,
                "schema_version": spec["schema_version"],
            }
        )
    return payloads, records


def _review_by_id(
    paper_reviews: Mapping[str, Any],
    review_id: str,
) -> Mapping[str, Any]:
    reviews = paper_reviews.get("reviews")
    if not isinstance(reviews, list):
        raise L2SupervisionDecisionError("paper route review input must contain a reviews array")
    matches = [
        review
        for review in reviews
        if isinstance(review, dict) and review.get("review_id") == review_id
    ]
    if len(matches) != 1:
        raise L2SupervisionDecisionError(
            f"expected one paper review {review_id!r}, observed {len(matches)}"
        )
    return matches[0]


def _minimum_gate(
    observations: Mapping[str, int],
    requirements: Mapping[str, Any],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    checks: dict[str, dict[str, Any]] = {}
    passed = True
    for field, minimum in requirements.items():
        observed = observations[field]
        field_passed = observed >= int(minimum)
        passed = passed and field_passed
        checks[field] = {
            "observed": observed,
            "minimum": minimum,
            "passed": field_passed,
        }
    return passed, checks


def _joint_l1_gate(
    observations: Mapping[str, int],
    requirements: Mapping[str, Any],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    minimum_fields = {
        key: value for key, value in requirements.items() if key.startswith("minimum_")
    }
    maximum_fields = {
        key: value for key, value in requirements.items() if key.startswith("maximum_")
    }
    minimum_observations = {key: observations[key] for key in minimum_fields}
    passed, checks = _minimum_gate(minimum_observations, minimum_fields)
    for field, maximum in maximum_fields.items():
        observed = observations[field]
        field_passed = observed <= int(maximum)
        passed = passed and field_passed
        checks[field] = {
            "observed": observed,
            "maximum": maximum,
            "passed": field_passed,
        }
    return passed, checks


def build_l2_supervision_decision(
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build the frozen supervision sufficiency and stopping decision."""

    config = load_config(config_path)
    inputs, input_records = _load_inputs(config, repo_root)
    root = repo_root.resolve()
    resolved_config = config_path.resolve()
    if not resolved_config.is_relative_to(root):
        raise L2SupervisionDecisionError("decision config must be inside the repository root")
    input_records.insert(
        0,
        {
            "input_id": "decision_config",
            "path": resolved_config.relative_to(root).as_posix(),
            "bytes": resolved_config.stat().st_size,
            "sha256": _sha256_file(resolved_config),
            "schema_version": CONFIG_SCHEMA_VERSION,
        },
    )

    paper_reviews = inputs["paper_route_reviews"]
    precursor = inputs["precursor_capability"]
    aldehyde_head = inputs["ugi3_aldehyde_head_capability"]
    virtual = inputs["agile_virtual_capability"]
    programs = inputs["component_programs"]
    terminal = inputs["terminal_queue"]
    transfer = inputs["hydrophobic_transfer"]
    lx_review = _review_by_id(
        paper_reviews,
        "lx_2024_aldehyde_tail_transfer_review",
    )

    joint_l1_observations = {
        "minimum_measured_products_passing_qualified_site_policy": (
            _nested_integer(
                aldehyde_head,
                (
                    "summary",
                    "agile_measured_products_passing_qualified_site_multiplicity",
                ),
                label="aldehyde/head capability",
            )
        ),
        "minimum_exact_virtual_decompositions": _nested_integer(
            virtual,
            ("summary", "products_with_exact_qualified_ugi_decomposition"),
            label="virtual capability",
        ),
        "maximum_virtual_products_without_exact_decomposition": (
            _nested_integer(
                virtual,
                (
                    "summary",
                    "products_without_exact_qualified_ugi_decomposition",
                ),
                label="virtual capability",
            )
        ),
        "maximum_virtual_products_with_multiple_exact_decompositions": (
            _nested_integer(
                virtual,
                (
                    "summary",
                    "products_with_multiple_exact_decompositions",
                ),
                label="virtual capability",
            )
        ),
    }

    hybrid_l2_observations = {
        "minimum_structure_resolved_route_instances": _nested_integer(
            paper_reviews,
            ("summary", "structure_resolved_l2_route_instances"),
            label="paper route reviews",
        ),
        "minimum_route_families": _nested_integer(
            paper_reviews,
            ("summary", "l2_route_families"),
            label="paper route reviews",
        ),
        "minimum_exact_source_programs": _nested_integer(
            programs,
            ("summary", "components_with_exact_source_program"),
            label="component programs",
        ),
        "minimum_direct_transfer_aldehydes": _nested_integer(
            lx_review,
            ("source_component_scope", "source_reported_aldehyde_tail_count"),
            label="LX_2024 review",
        ),
    }

    monolithic_observations = {
        "minimum_complete_product_routes": _nested_integer(
            virtual,
            ("summary", "products_with_complete_l2_l3_candidate"),
            label="virtual capability",
        ),
        "minimum_reported_negative_outcomes": _nested_integer(
            paper_reviews,
            ("summary", "reported_negative_outcomes"),
            label="paper route reviews",
        ),
        "minimum_head_upstream_routes": (
            _nested_integer(
                paper_reviews,
                (
                    "summary",
                    "upstream_component_routes_by_role",
                    "ugi3_amine_head",
                ),
                label="paper route reviews",
            )
            + _nested_integer(
                paper_reviews,
                (
                    "summary",
                    "upstream_component_routes_by_role",
                    "head",
                ),
                label="paper route reviews",
            )
        ),
    }

    gates = _required_mapping(
        config["architecture_gates"],
        label="architecture_gates",
    )
    joint_l1_passed, joint_l1_checks = _joint_l1_gate(
        joint_l1_observations,
        _required_mapping(gates["joint_product_l1"], label="joint L1 gate"),
    )
    hybrid_l2_passed, hybrid_l2_checks = _minimum_gate(
        hybrid_l2_observations,
        _required_mapping(gates["hybrid_l2"], label="hybrid L2 gate"),
    )
    monolithic_passed, monolithic_checks = _minimum_gate(
        monolithic_observations,
        _required_mapping(
            gates["monolithic_complete_route_decoder_necessary_conditions"],
            label="monolithic L2 necessary-condition gate",
        ),
    )

    evidence = {
        "product_and_l1": {
            "measured_products_passing_qualified_site_policy": (
                joint_l1_observations["minimum_measured_products_passing_qualified_site_policy"]
            ),
            "virtual_products_with_exact_unique_decomposition": (
                joint_l1_observations["minimum_exact_virtual_decompositions"]
            ),
            "unique_virtual_components": _nested_integer(
                virtual,
                ("summary", "unique_components_total"),
                label="virtual capability",
            ),
        },
        "l2_source_supervision": {
            "reviewed_source_packages": _nested_integer(
                paper_reviews,
                ("summary", "source_packages_chemistry_reviewed"),
                label="paper route reviews",
            ),
            "route_instances": _nested_integer(
                paper_reviews,
                ("summary", "l2_route_instances"),
                label="paper route reviews",
            ),
            "structure_resolved_route_instances": hybrid_l2_observations[
                "minimum_structure_resolved_route_instances"
            ],
            "reaction_instances": _nested_integer(
                paper_reviews,
                ("summary", "l2_reaction_instances"),
                label="paper route reviews",
            ),
            "route_families": hybrid_l2_observations["minimum_route_families"],
            "reported_negative_outcomes": monolithic_observations[
                "minimum_reported_negative_outcomes"
            ],
            "head_upstream_routes": monolithic_observations["minimum_head_upstream_routes"],
        },
        "recursive_programs_and_closure": {
            "exact_source_program_components": hybrid_l2_observations[
                "minimum_exact_source_programs"
            ],
            "family_projected_program_components": _nested_integer(
                programs,
                ("summary", "components_with_family_projected_program"),
                label="component programs",
            ),
            "program_families": len(
                _required_mapping(
                    programs["summary"].get("program_families"),
                    label="component program families",
                )
            ),
            "complete_product_routes": monolithic_observations["minimum_complete_product_routes"],
            "current_accepted_procurement_candidates": _nested_integer(
                terminal,
                ("summary", "candidates_with_current_accepted_procurement"),
                label="terminal queue",
            ),
            "unresolved_terminal_candidates": _nested_integer(
                terminal,
                ("summary", "unresolved_terminal_candidates"),
                label="terminal queue",
            ),
        },
        "cross_platform_transfer": {
            "pilot_source_platforms": _nested_integer(
                transfer,
                ("summary", "independent_source_platforms"),
                label="hydrophobic transfer",
            ),
            "pilot_route_complete_components": _nested_integer(
                transfer,
                ("summary", "computationally_route_complete"),
                label="hydrophobic transfer",
            ),
            "lx_2024_direct_aldehyde_transfers": hybrid_l2_observations[
                "minimum_direct_transfer_aldehydes"
            ],
        },
        "native_component_scope": {
            "agile_heads_with_upstream_route": _nested_integer(
                precursor,
                ("summary", "agile_heads_with_upstream_route"),
                label="precursor capability",
            ),
            "exact_agile_isocyanide_route_candidates": _nested_integer(
                precursor,
                (
                    "summary",
                    "exact_agile_isocyanide_upstream_route_candidates",
                ),
                label="precursor capability",
            ),
            "structure_resolved_agile_aldehyde_routes": _nested_integer(
                aldehyde_head,
                ("summary", "structure_resolved_agile_aldehyde_routes"),
                label="aldehyde/head capability",
            ),
        },
    }

    if not joint_l1_passed or not hybrid_l2_passed:
        recommended_architecture = "insufficient_for_current_forge_design"
    else:
        recommended_architecture = "hierarchical_joint_product_and_l1_with_hybrid_recursive_l2"

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-09 L2 supervision sufficiency and stopping decision",
        "frozen_utc": config["frozen_utc"],
        "inputs": input_records,
        "evidence": evidence,
        "gates": {
            "joint_product_l1": {
                "passed": joint_l1_passed,
                "checks": joint_l1_checks,
                "interpretation": (
                    "Dense exact product-component supervision supports "
                    "joint whole-product and L1 assembly learning."
                ),
            },
            "hybrid_l2": {
                "passed": hybrid_l2_passed,
                "checks": hybrid_l2_checks,
                "interpretation": (
                    "Observed route families and exact programs can seed "
                    "template search and learned proposal or ranking."
                ),
            },
            "monolithic_complete_route_decoder_necessary_conditions": {
                "passed": monolithic_passed,
                "checks": monolithic_checks,
                "interpretation": (
                    "Failure means the current corpus does not establish "
                    "viability of a monolithic complete-route decoder. "
                    "Passing would be necessary, not sufficient."
                ),
            },
        },
        "future_operational_closure_thresholds": dict(
            config["future_operational_closure_thresholds"]
        ),
        "future_training_safeguards": dict(config["future_training_safeguards"]),
        "decision": {
            "inventory_is_sufficient_to_choose_architecture": (
                joint_l1_passed and hybrid_l2_passed
            ),
            "recommended_architecture": recommended_architecture,
            "joint_product_l1_supervision_supported": joint_l1_passed,
            "hybrid_recursive_l2_supported": hybrid_l2_passed,
            "monolithic_joint_product_complete_route_decoder_supported": False,
            "monolithic_necessary_conditions_passed": monolithic_passed,
            "l2_model_built": False,
            "current_operational_closure_achieved": False,
            "paper_mining_policy": "targeted_gap_driven_only",
            "next_mining_triggers": [
                "repeated missing-route-knowledge failures among eligible candidates",
                "unresolved head or terminal closure needed by the candidate panel",
                "new isocyanide substrate-scope evidence",
                "negative or failed synthesis evidence needed for calibration",
            ],
            "rationale": [
                "Product and final-assembly supervision is dense and exact.",
                "L2 supervision is chemically useful but positive-heavy.",
                "No complete product route currently closes across every L2 and L3 branch.",
                "No upstream amine-head route is present in the reviewed source set.",
                "No reported negative synthesis outcome is present in the reviewed source set.",
                "Cross-platform aldehyde transfer is positive and no longer just speculative.",
            ],
        },
        "claims_boundary": dict(config["claims_boundary"]),
    }


def write_l2_supervision_decision(
    result: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Write the deterministic M0-09 decision artifact atomically."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=output_dir,
        prefix=f".{RESULT_NAME}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    destination = output_dir / RESULT_NAME
    os.replace(temporary, destination)
    return destination
