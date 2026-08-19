#!/usr/bin/env python3
"""Run the frozen, arm-blind, bounded hybrid Ugi L1/L2/L3 route cascade.

Stages are separate processes because the two learned proposal engines have
mutually incompatible pinned runtimes.  Each stage pins the previous stage's
artifacts by SHA-256, so the cascade is restartable and every step is
independently reproducible:

    exact-evidence   .venv                              frozen recursive evidence
    graph2edits      .venv-graph2edits-py311-arm64      bounded single-step proposals
    aizynthfinder    .venv-aizynthfinder-py311-arm64    residual bounded proposals
    adjudicate       .venv                              independent adjudication
    audit            .venv                              attrition and balance audit

Generation arm, cohort, authority tier and potency never reach route search or
the planner cache key.  They are joined back only in the audit stage.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "src"))

from forge.potency.ugi_semantic_annotations import ROLE_NAMES  # noqa: E402
from forge.product.ugi_bounded_hybrid_route_cascade import (  # noqa: E402
    ADJUDICATION_SCHEMA_VERSION,
    ATTRITION_CSV_FIELDS,
    CACHE_CLONE_ID,
    CANDIDATE_EXACT_LEDGER_SCHEMA_VERSION,
    CANDIDATE_LEDGER_SCHEMA_VERSION,
    COMPLETE,
    COMPONENT_EXACT_LEDGER_SCHEMA_VERSION,
    COMPONENT_LEDGER_SCHEMA_VERSION,
    COMPONENT_STATES,
    EXACT_EVIDENCE_SCHEMA_VERSION,
    LEAF_LEDGER_SCHEMA_VERSION,
    NEVER_EXPANDED,
    NOT_ASSESSED_OUTCOME,
    PREFLIGHT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    UgiBoundedHybridRouteCascadeError,
    assert_expected_population,
    atomic_write,
    attrition_csv_rows,
    blindness_receipt,
    branch_sufficiency_recommendation,
    build_candidate_population,
    canonical_json_bytes,
    collect_route_leaves,
    component_descriptors,
    component_sha256,
    component_target_id,
    csv_gzip_bytes,
    derive_final_component_state,
    jsonl_gzip_bytes,
    load_cascade_contract,
    load_json,
    product_route_state,
    read_csv_gzip,
    read_jsonl_gzip,
    rescoring_index,
    resolve_potency_reporting,
    sha256_file,
    sha256_payload,
    summarize_attrition,
    tail_length_bucket,
    unique_component_targets,
    unresolved_disposition,
    unresolved_leaf_classes,
)

PROPOSAL_ONLY_AUTHORITY = {
    "proposal_only_not_route_evidence": True,
    "may_create_route_evidence": False,
    "may_close_route": False,
    "may_set_final_component_state_complete": False,
    "public_stock_solution_is_current_procurement_evidence": False,
    "model_score_used_as_synthesis_value": False,
    "unsolved_search_means_unsynthesizable": False,
}


def _rdkit_version() -> str:
    from rdkit import rdBase

    return str(rdBase.rdkitVersion)


def _runtime_receipt() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "rdkit_version": _rdkit_version(),
    }


def _pin_artifact(repo: Path, path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}


def _require_stage_artifact(repo: Path, path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise UgiBoundedHybridRouteCascadeError(
            f"missing upstream stage artifact ({label}): {path.relative_to(repo)}"
        )
    return path


def _write_result(repo: Path, path: Path, content: dict[str, Any]) -> dict[str, Any]:
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(path, canonical_json_bytes(result))
    return result


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight(contract: Any) -> dict[str, Any]:
    repo = contract.repo
    records = build_candidate_population(contract)
    population = assert_expected_population(contract, records)
    receipt = blindness_receipt(contract, records)
    output_dir = contract.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    content = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "bounded_hybrid_route_cascade_preflight_complete",
        "config": {
            "path": str(contract.config_path.relative_to(repo)),
            "sha256": contract.config_sha256,
        },
        "assessment_as_of_utc": contract.assessment_as_of_utc,
        "cumulative_source_inputs_sha256": contract.cumulative_source_inputs_sha256,
        "procurement_snapshot": dict(contract.config["procurement_snapshot"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(contract.paths.items())
        },
        "population": population,
        "blindness_receipt": receipt,
        "runtime": _runtime_receipt(),
        "scope": dict(contract.config["scope"]),
        "decision_gates": {
            "g0_preflight_population_and_hashes": "passed",
            "g1_exact_constitutional_l1_all_rows": "deferred_to_exact_evidence_stage",
            "g2_arm_blind_cache_and_search": "receipt_recorded",
            "g3_identical_budget_all_arms": "declared_in_config",
        },
        "nonclaims": list(contract.config["nonclaims"]),
    }
    return _write_result(repo, output_dir / "preflight.json", content)


# ---------------------------------------------------------------------------
# stage: exact evidence
# ---------------------------------------------------------------------------


def _declared_support_violation_detail(product_smiles: str, graph_support: Any) -> dict[str, Any]:
    """Name the exact atom/bond states that fall outside declared support."""

    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(product_smiles)
    if molecule is None:
        return {"parsed": False, "out_of_vocabulary_atom_states": []}
    vocabulary = set(graph_support.atom_vocabulary)
    observed = {
        (
            atom.GetSymbol(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetNumExplicitHs(),
        )
        for atom in molecule.GetAtoms()
    }
    return {
        "parsed": True,
        "declared_atom_state_count": len(vocabulary),
        "out_of_vocabulary_atom_states": [
            {
                "symbol": symbol,
                "formal_charge": charge,
                "aromatic": aromatic,
                "explicit_hydrogens": hydrogens,
            }
            for symbol, charge, aromatic, hydrogens in sorted(
                observed - vocabulary, key=lambda state: (state[0], state[1], state[2], state[3])
            )
        ],
    }


def _not_admitted_candidate_row(
    record: Any,
    *,
    unit_id: str,
    gate: str,
    error: Exception,
    product_smiles: str,
    graph_support: Any,
) -> dict[str, Any]:
    """Retain a candidate that could not enter route assessment, with its reason."""

    causes: list[str] = []
    cause = error.__cause__
    while cause is not None:
        causes.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return {
        "schema_version": CANDIDATE_EXACT_LEDGER_SCHEMA_VERSION,
        "unit_id": unit_id,
        "canonical_product": record.canonical_product,
        "components": dict(record.components),
        "route_assessment_admitted": False,
        "admission_failure": {
            "gate": gate,
            "error": f"{type(error).__name__}: {error}",
            "causes": causes,
            "detail": _declared_support_violation_detail(product_smiles, graph_support),
            "gate_was_not_relaxed": True,
            "candidate_retained_in_denominator": True,
            "interpretation": (
                "The product satisfied terminal chemical admission during generation but "
                "falls outside the declared graph-support contract re-verified before route "
                "assessment. This is a declared-support statement, not a claim of "
                "unsynthesizability."
            ),
        },
        "terminal_sha256": None,
        "generation_trace_sha256": None,
        "exact_l1_reverified": True,
        "l1_reverification": None,
        "l1_reaction_sha256": None,
        "strict_route_complete": False,
        "strict_complete_role_count": 0,
        "roles": [],
        "support_audit": None,
        "assessment_sha256": None,
        "route_inputs_excluded_reporting_fields": True,
    }


def stage_exact_evidence(contract: Any) -> dict[str, Any]:
    from types import MappingProxyType

    from forge.product.ugi_generated_terminal_support import (
        DeclaredGraphSupportContext,
        declared_graph_support_context_sha256,
    )
    from forge.product.ugi_held_component_gate import load_ugi_reaction_contract
    from forge.product.ugi_matched_budget_orchestration import (
        MatchedArm,
        MatchedAssessmentContext,
        MatchedGenerationRequest,
        MatchedScheduleEntry,
        RouteComputeUsage,
    )
    from forge.product.ugi_production_terminal_route_evaluator import (
        build_production_ugi_terminal_aware_planner_factory,
    )
    from forge.product.ugi_restartable_terminal_support_adapter import (
        UgiRestartableTerminalSupportAdapterError,
        adapt_restartable_completion_row_for_route_support,
        canonical_morphology_program_bytes,
        native_completion_record_from_locked_terminal,
    )
    from forge.product.ugi_selected_restartable_generator import _atom_vocabulary_states
    from forge.product.ugi_selected_restartable_generator_v2 import (
        CLOSURE_CHECKPOINT_SHA256,
        COMPONENT_RECOVERY_CONTRACT_SHA256,
        L1_REACTION_SHA256,
    )
    from forge.route.planner_cache import FilePlannerCache
    from forge.route.planner_cache_snapshot import (
        OverlayFilePlannerCache,
        ReadOnlyFilePlannerCache,
        build_file_planner_cache_snapshot_manifest,
    )
    from forge.route.terminal_assessment import (
        QualifiedUgiL1Reverifier,
        assess_locked_ugi_terminal_routes,
        required_three_role_route_reservation,
    )
    from forge.value.ugi_exact_closure_guidance import (
        UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
        exact_closure_potential_from_product_value,
    )

    repo = contract.repo
    output_dir = contract.output_dir
    preflight_path = _require_stage_artifact(repo, output_dir / "preflight.json", label="preflight")
    preflight = load_json(preflight_path, label="preflight result")
    if preflight.get("status") != "bounded_hybrid_route_cascade_preflight_complete":
        raise UgiBoundedHybridRouteCascadeError("preflight stage is not complete")

    records = build_candidate_population(contract)
    assert_expected_population(contract, records)
    receipt = blindness_receipt(contract, records)
    if receipt != preflight["blindness_receipt"]:
        raise UgiBoundedHybridRouteCascadeError(
            "route population changed since preflight; blindness receipt differs"
        )

    refit = load_json(contract.paths["production_refit_result"], label="refit result")
    model_config = refit.get("model")
    if not isinstance(model_config, dict):
        raise UgiBoundedHybridRouteCascadeError("production refit result has no model config")
    generator_checkpoint_sha256 = sha256_file(contract.paths["joint_checkpoint"])
    closure_checkpoint_sha256 = sha256_file(contract.paths["closure_checkpoint"])
    if closure_checkpoint_sha256 != CLOSURE_CHECKPOINT_SHA256:
        raise UgiBoundedHybridRouteCascadeError("frozen closure checkpoint changed")

    graph_support = DeclaredGraphSupportContext(
        generator_checkpoint_sha256=generator_checkpoint_sha256,
        model_config=MappingProxyType(dict(model_config)),
        atom_vocabulary=_atom_vocabulary_states(contract.paths["atom_vocabulary"]),
    )
    reaction = load_ugi_reaction_contract(contract.paths["qualified_reaction_registry"])
    reverifier = QualifiedUgiL1Reverifier(
        reaction_contract=reaction,
        l1_reaction_sha256=L1_REACTION_SHA256,
    )
    factory = build_production_ugi_terminal_aware_planner_factory(
        repo_root=repo,
        assessment_as_of_utc=contract.assessment_as_of_utc,
        expected_cumulative_source_inputs_sha256=contract.cumulative_source_inputs_sha256,
        selected_generator_checkpoint_sha256=generator_checkpoint_sha256,
        graph_support=graph_support,
        l1_reverifier=reverifier,
        candidate_record_resolver=native_completion_record_from_locked_terminal,
    )
    declared_limits = dict(contract.stage("exact_evidence")["budget_limits"])
    if factory.planner_context.budget_limits.to_dict() != declared_limits:
        raise UgiBoundedHybridRouteCascadeError(
            "frozen planner budget limits differ from the declared cascade budget"
        )

    cache_root = output_dir / "planner_cache"
    if cache_root.exists() and any(cache_root.rglob("*")):
        raise UgiBoundedHybridRouteCascadeError("planner cache root must be absent or empty")
    base_cache = FilePlannerCache(cache_root / "base")
    lane_cache = FilePlannerCache(cache_root / "lane")
    base_manifest = build_file_planner_cache_snapshot_manifest(
        base_cache,
        factory.planner_context,
        assessment_at_utc=factory.assessment_as_of_utc,
    )
    readonly_base = ReadOnlyFilePlannerCache(
        base_cache,
        base_manifest,
        factory.planner_context,
        assessment_at_utc=factory.assessment_as_of_utc,
    )
    lane = OverlayFilePlannerCache(readonly_base, lane_cache, clone_id=CACHE_CLONE_ID)
    lane_before = lane.before_manifest

    # The whole 256-product population is locked before any route call.  The
    # manifest digest is computed from the arm-blind route inputs only.
    post_hoc_lock_sha256 = receipt["route_population_sha256"]
    required = required_three_role_route_reservation(factory.planner_context.budget_limits)

    candidate_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        native = record.native_terminal
        generation = record.generation_row
        unit_id = f"bounded-hybrid-route:{sha256_payload(record.canonical_product)[:20]}"
        entry = MatchedScheduleEntry(
            unit_id=unit_id,
            morphology_program=canonical_morphology_program_bytes(native["program"]),
            program_index=int(record.draw_index),
            particle_index=int(generation.get("shard_local_index", 0)),
            checkpoint_index=0,
            generator_checkpoint_sha256=generator_checkpoint_sha256,
            closure_checkpoint_sha256=closure_checkpoint_sha256,
            rollout_index=int(generation.get("shard_index", 0)),
            productive_generation_calls=1,
            route_reservation=RouteComputeUsage(),
        )
        request = MatchedGenerationRequest(
            arm=MatchedArm.POST_HOC,
            entry=entry,
            productive_seed=int(generation["terminal_seed"]),
        )
        try:
            terminal = adapt_restartable_completion_row_for_route_support(
                native,
                generation_request=request,
                l1_reaction=reaction,
                l1_reaction_sha256=L1_REACTION_SHA256,
                component_recovery_contract_sha256=COMPONENT_RECOVERY_CONTRACT_SHA256,
                graph_support=graph_support,
                l1_reverifier=reverifier,
            ).locked_terminal
        except UgiRestartableTerminalSupportAdapterError as error:
            # The declared graph-support contract is never relaxed.  The
            # candidate is retained in the denominator with its exact
            # disposition so no failure disappears silently.
            candidate_rows.append(
                _not_admitted_candidate_row(
                    record,
                    unit_id=unit_id,
                    gate="declared_graph_support_reverification",
                    error=error,
                    product_smiles=record.canonical_product,
                    graph_support=graph_support,
                )
            )
            print(
                f"exact-evidence {index}/{len(records)} NOT-ADMITTED "
                f"({record.arm_id}:{record.draw_index})",
                flush=True,
            )
            continue
        context = MatchedAssessmentContext(
            arm=MatchedArm.POST_HOC,
            route_seed=int(contract.config["seed"]),
            remaining_budget=required,
            unit_reservation=required,
            cache_snapshot_sha256=readonly_base.manifest.snapshot_sha256,
            cache_clone_id=CACHE_CLONE_ID,
            post_hoc_lock_manifest_sha256=post_hoc_lock_sha256,
        )
        planner = factory.build_planner(lane, factory.planner_context, context, terminal)
        assessment_receipt = assess_locked_ugi_terminal_routes(
            terminal,
            l1_reverifier=factory.l1_reverifier,
            planner=planner,
            planner_context=factory.planner_context,
            assessment_context=context,
            assessment_at_utc=factory.assessment_as_of_utc,
        )
        potential = exact_closure_potential_from_product_value(assessment_receipt.product_value)
        receipt_dict = assessment_receipt.to_dict()
        role_records: list[dict[str, Any]] = []
        for role_assessment, role_potential in zip(
            assessment_receipt.role_assessments, potential.roles, strict=True
        ):
            if role_assessment.role != role_potential.role:
                raise UgiBoundedHybridRouteCascadeError("role receipt ordering diverged")
            assessment_dict = role_assessment.assessment.to_dict()
            leaves = collect_route_leaves(assessment_dict["route_tree"])
            role_records.append(
                {
                    "role": role_assessment.role,
                    "canonical_smiles": role_assessment.target.canonical_smiles,
                    "component_sha256": component_sha256(
                        role_assessment.role, role_assessment.target.canonical_smiles
                    ),
                    "assessment_outcome": assessment_dict["outcome"],
                    "strict_complete": bool(role_potential.strict_complete),
                    "guidance_reason": role_potential.reason.value,
                    "cache_key_sha256": role_assessment.cache_key_sha256,
                    "physical_cache_hit": role_assessment.budget_after.physical_cache_hits == 1,
                    "component_value": role_assessment.component_value.to_dict(),
                    "budget_after": role_assessment.budget_after.to_dict(),
                    "route_tree": assessment_dict["route_tree"],
                    "leaves": leaves,
                    "unresolved_leaf_classes": unresolved_leaf_classes(leaves),
                }
            )
        candidate_rows.append(
            {
                "schema_version": CANDIDATE_EXACT_LEDGER_SCHEMA_VERSION,
                "unit_id": unit_id,
                "canonical_product": record.canonical_product,
                "components": dict(record.components),
                "route_assessment_admitted": True,
                "admission_failure": None,
                "terminal_sha256": terminal.terminal_sha256,
                "generation_trace_sha256": terminal.generation_trace_sha256,
                "exact_l1_reverified": True,
                "l1_reverification": receipt_dict["l1_reverification"],
                "l1_reaction_sha256": receipt_dict["payload"]["l1_reaction_sha256"],
                "strict_route_complete": bool(potential.strict_route_complete),
                "strict_complete_role_count": int(potential.strict_complete_role_count),
                "roles": role_records,
                "support_audit": planner.support_audit.to_dict(),
                "assessment_sha256": assessment_receipt.assessment_sha256,
                # Deliberately absent: arm, cohort, authority tier, potency.
                "route_inputs_excluded_reporting_fields": True,
            }
        )
        if index % 32 == 0 or index == len(records):
            print(f"exact-evidence {index}/{len(records)}", flush=True)

    lane_after = lane.current_manifest()
    base_after = build_file_planner_cache_snapshot_manifest(
        base_cache,
        factory.planner_context,
        assessment_at_utc=factory.assessment_as_of_utc,
    )
    if base_after != base_manifest:
        raise UgiBoundedHybridRouteCascadeError("immutable base cache snapshot changed")

    # Deduplicate to unique component targets; an arm-blind representative is
    # chosen by lowest terminal hash so the choice cannot depend on arm.
    by_component: dict[str, dict[str, Any]] = {}
    for candidate in candidate_rows:
        for role_record in candidate["roles"]:
            key = str(role_record["component_sha256"])
            current = by_component.get(key)
            representative = {
                "terminal_sha256": candidate["terminal_sha256"],
                "support_audit": candidate["support_audit"],
                "l1_reaction_sha256": candidate["l1_reaction_sha256"],
            }
            if current is None:
                by_component[key] = {
                    "schema_version": COMPONENT_EXACT_LEDGER_SCHEMA_VERSION,
                    "component_sha256": key,
                    "target_id": component_target_id(
                        role_record["role"], role_record["canonical_smiles"]
                    ),
                    "role": role_record["role"],
                    "canonical_smiles": role_record["canonical_smiles"],
                    "occurrence_count": 1,
                    "assessment_outcome": role_record["assessment_outcome"],
                    "strict_complete": role_record["strict_complete"],
                    "guidance_reason": role_record["guidance_reason"],
                    "cache_key_sha256": role_record["cache_key_sha256"],
                    "component_value": role_record["component_value"],
                    "route_tree": role_record["route_tree"],
                    "leaves": role_record["leaves"],
                    "unresolved_leaf_classes": role_record["unresolved_leaf_classes"],
                    "representative": representative,
                }
                continue
            if (
                current["assessment_outcome"] != role_record["assessment_outcome"]
                or current["strict_complete"] != role_record["strict_complete"]
                or current["cache_key_sha256"] != role_record["cache_key_sha256"]
            ):
                raise UgiBoundedHybridRouteCascadeError(
                    "one canonical component received inconsistent route outcomes"
                )
            current["occurrence_count"] += 1
            if representative["terminal_sha256"] < current["representative"]["terminal_sha256"]:
                current["representative"] = representative

    # Components whose only parent products were not admitted are still carried
    # at full denominator with an explicit unassessed disposition.
    targets = unique_component_targets(records)
    for role, smiles in targets:
        key = component_sha256(role, smiles)
        if key in by_component:
            continue
        parents = [
            candidate["canonical_product"]
            for candidate in candidate_rows
            if candidate["components"][role] == smiles
        ]
        by_component[key] = {
            "schema_version": COMPONENT_EXACT_LEDGER_SCHEMA_VERSION,
            "component_sha256": key,
            "target_id": component_target_id(role, smiles),
            "role": role,
            "canonical_smiles": smiles,
            "occurrence_count": len(parents),
            "assessment_outcome": "not_assessed_no_admitted_parent_product",
            "strict_complete": False,
            "guidance_reason": "parent_product_outside_declared_graph_support",
            "cache_key_sha256": None,
            "component_value": None,
            "route_tree": None,
            "leaves": [],
            "unresolved_leaf_classes": {},
            "representative": None,
            "unassessed_parent_products": sorted(parents),
        }
    component_rows = sorted(
        by_component.values(), key=lambda row: (row["role"], row["canonical_smiles"])
    )
    if len(component_rows) != len(targets):
        raise UgiBoundedHybridRouteCascadeError("component deduplication lost targets")

    candidate_rows.sort(key=lambda row: row["canonical_product"])
    candidate_path = output_dir / "candidate_exact_ledger.jsonl.gz"
    component_path = output_dir / "component_exact_evidence.jsonl.gz"
    cache_path = output_dir / "cache_audit.json"
    atomic_write(candidate_path, jsonl_gzip_bytes(candidate_rows))
    atomic_write(component_path, jsonl_gzip_bytes(component_rows))
    cache_audit = {
        "schema": "forge.ugi_bounded_hybrid_route_cache_audit.v1",
        "single_lane": True,
        "guided_post_hoc_split_inherited": False,
        "clone_id": CACHE_CLONE_ID,
        "context_sha256": factory.planner_context_sha256,
        "base_snapshot_sha256": base_manifest.snapshot_sha256,
        "base_unchanged": base_after == base_manifest,
        "lane_entries_before": lane_before.entry_count,
        "lane_entries_after": lane_after.entry_count,
        "unique_component_targets": len(component_rows),
        # Counted from realized role receipts: a candidate that failed declared
        # graph-support re-verification consumed no planner call.
        "logical_planner_calls": sum(len(candidate["roles"]) for candidate in candidate_rows),
        "physical_cache_hits": sum(
            1
            for candidate in candidate_rows
            for role_record in candidate["roles"]
            if role_record["physical_cache_hit"]
        ),
        "cache_key_fields": [
            "schema_version",
            "target",
            "context",
            "resource_usage_before_call",
        ],
        "cache_key_excludes": [
            "arm_id",
            "cohort",
            "authority_tier",
            "conservative_high_potency",
            "potency_utility",
            "draw_index",
            "canonical_product",
        ],
    }
    atomic_write(cache_path, canonical_json_bytes(cache_audit))

    outcome_counts = Counter(row["assessment_outcome"] for row in component_rows)
    content = {
        "schema_version": EXACT_EVIDENCE_SCHEMA_VERSION,
        "status": "bounded_hybrid_route_cascade_exact_evidence_complete",
        "config": {
            "path": str(contract.config_path.relative_to(repo)),
            "sha256": contract.config_sha256,
        },
        "preflight": _pin_artifact(repo, preflight_path),
        "assessment_as_of_utc": contract.assessment_as_of_utc,
        "planner_context_sha256": factory.planner_context_sha256,
        "route_policy_sha256": UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
        "declared_graph_support_sha256": declared_graph_support_context_sha256(graph_support),
        "generator_checkpoint_sha256": generator_checkpoint_sha256,
        "budget_limits": declared_limits,
        "post_hoc_lock_manifest_sha256": post_hoc_lock_sha256,
        "blindness_receipt": receipt,
        "runtime": _runtime_receipt(),
        "summary": {
            "candidate_rows": len(candidate_rows),
            "route_assessment_admitted_rows": sum(
                1 for row in candidate_rows if row["route_assessment_admitted"]
            ),
            "route_assessment_not_admitted_rows": sum(
                1 for row in candidate_rows if not row["route_assessment_admitted"]
            ),
            "not_admitted_candidates": [
                {
                    "canonical_product": row["canonical_product"],
                    "gate": row["admission_failure"]["gate"],
                    "out_of_vocabulary_atom_states": row["admission_failure"]["detail"][
                        "out_of_vocabulary_atom_states"
                    ],
                }
                for row in candidate_rows
                if not row["route_assessment_admitted"]
            ],
            "exact_l1_reverified_rows": sum(
                1 for row in candidate_rows if row["exact_l1_reverified"]
            ),
            "strict_route_complete_rows": sum(
                1 for row in candidate_rows if row["strict_route_complete"]
            ),
            "unique_component_targets": len(component_rows),
            "component_assessment_outcomes": dict(sorted(outcome_counts.items())),
            "strict_complete_components": sum(
                1 for row in component_rows if row["strict_complete"]
            ),
        },
        "cache": cache_audit,
        "artifacts": {
            "candidate_exact_ledger.jsonl.gz": {
                **_pin_artifact(repo, candidate_path),
                "rows": len(candidate_rows),
                "schema_version": CANDIDATE_EXACT_LEDGER_SCHEMA_VERSION,
            },
            "component_exact_evidence.jsonl.gz": {
                **_pin_artifact(repo, component_path),
                "rows": len(component_rows),
                "schema_version": COMPONENT_EXACT_LEDGER_SCHEMA_VERSION,
            },
            "cache_audit.json": _pin_artifact(repo, cache_path),
        },
        "decision_gates": {
            "g1_exact_constitutional_l1_all_rows": "passed",
            "g2_arm_blind_cache_and_search": "passed",
            "g3_identical_budget_all_arms": "passed",
            "g5_denominator_preservation": "passed",
        },
        "nonclaims": list(contract.config["nonclaims"]),
    }
    return _write_result(repo, output_dir / "exact_evidence_result.json", content)


# ---------------------------------------------------------------------------
# stage: graph2edits
# ---------------------------------------------------------------------------


def _unresolved_component_rows(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return components that are unresolved *and* have an assessable parent.

    A component whose only parent products failed declared graph-support
    re-verification has no qualified route root, so no proposal engine can be
    given a well-formed request for it.  It stays in the denominator and is
    reported as search-censored rather than being quietly proposed on.
    """

    return [
        row
        for row in component_rows
        if not row["strict_complete"]
        and row["assessment_outcome"] != NOT_ASSESSED_OUTCOME
        and row.get("representative") is not None
    ]


def stage_graph2edits(contract: Any) -> dict[str, Any]:
    from forge.route.graph2edits_backend import (
        GRAPH2EDITS_BACKEND_ID,
        GRAPH2EDITS_CHECKPOINT_LICENSE,
        GRAPH2EDITS_IMPLEMENTATION_VERSION,
        GRAPH2EDITS_SOURCE_LOCATOR,
        GRAPH2EDITS_TRAINING_CORPUS_ID,
        Graph2EditsInferencePolicy,
        Graph2EditsLocalArtifacts,
        Graph2EditsProposalBackend,
        build_syntheseus_worker,
    )
    from forge.route.graph2edits_one_gap_diagnostic import _qualification_from_dict
    from forge.route.l2_forward_resolver import load_independent_l2_forward_resolver
    from forge.route.planner import RouteTarget
    from forge.route.proposal_discovery_status import SourceNeutralProposalDiscoveryResolver
    from forge.route.proposal_engine import (
        ProposalBackendManifest,
        ProposalRequest,
        ProposalTargetKind,
        RootQualificationReceipt,
    )
    from forge.route.semantic_family_equivalence import audit_semantic_family_equivalence

    repo = contract.repo
    output_dir = contract.output_dir
    upstream_path = _require_stage_artifact(
        repo, output_dir / "exact_evidence_result.json", label="exact evidence"
    )
    upstream = load_json(upstream_path, label="exact evidence result")
    if upstream.get("status") != "bounded_hybrid_route_cascade_exact_evidence_complete":
        raise UgiBoundedHybridRouteCascadeError("exact-evidence stage is not complete")
    component_path = repo / upstream["artifacts"]["component_exact_evidence.jsonl.gz"]["path"]
    if (
        sha256_file(component_path)
        != (upstream["artifacts"]["component_exact_evidence.jsonl.gz"]["sha256"])
    ):
        raise UgiBoundedHybridRouteCascadeError("component exact-evidence ledger changed")
    component_rows = read_jsonl_gzip(component_path, label="component exact evidence")
    targets = _unresolved_component_rows(component_rows)

    settings = contract.stage("graph2edits")
    policy = Graph2EditsInferencePolicy()
    if policy.max_edit_steps != int(settings["max_edit_steps"]):
        raise UgiBoundedHybridRouteCascadeError(
            "frozen Graph2Edits edit-step budget differs from the declared cascade budget"
        )
    runtime = load_json(
        contract.paths["graph2edits_runtime_qualification"], label="Graph2Edits runtime"
    )
    checkpoint_path = repo / runtime["artifacts"]["checkpoint"]["path"]
    if sha256_file(checkpoint_path) != runtime["artifacts"]["checkpoint"]["sha256"]:
        raise UgiBoundedHybridRouteCascadeError("Graph2Edits checkpoint changed")
    corpus_manifest_path = contract.paths["graph2edits_training_corpus_manifest"]
    manifest = ProposalBackendManifest(
        backend_id=GRAPH2EDITS_BACKEND_ID,
        implementation_version=GRAPH2EDITS_IMPLEMENTATION_VERSION,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_license=GRAPH2EDITS_CHECKPOINT_LICENSE,
        training_corpus_id=GRAPH2EDITS_TRAINING_CORPUS_ID,
        training_corpus_snapshot_sha256=sha256_file(corpus_manifest_path),
        source_locator=GRAPH2EDITS_SOURCE_LOCATOR,
    )
    backend = Graph2EditsProposalBackend(
        manifest=manifest,
        artifacts=Graph2EditsLocalArtifacts(
            checkpoint_path=checkpoint_path,
            training_corpus_manifest_path=corpus_manifest_path,
            corpus_path_root=repo,
        ),
        worker=build_syntheseus_worker(
            model_dir=checkpoint_path.parent,
            device=str(settings["device"]),
            max_edit_steps=policy.max_edit_steps,
        ),
        policy=policy,
    )
    exact_resolver = load_independent_l2_forward_resolver(
        contract.paths["graph2edits_forward_resolver"], repo_root=repo
    )
    resolver = SourceNeutralProposalDiscoveryResolver(exact_resolver=exact_resolver)

    operational_policy_id = "forge.bounded_hybrid_route_cascade_proposal_only.v1"
    operational_policy_sha256 = sha256_payload(
        {
            "policy_id": operational_policy_id,
            "proposal_only": True,
            "operational_screen_run": False,
            "route_closure_authorized": False,
        }
    )
    maximum_proposals = int(settings["maximum_proposals"])
    repeat_count = int(settings["repeat_count"])
    proposal_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for index, row in enumerate(targets, start=1):
        role = str(row["role"])
        smiles = str(row["canonical_smiles"])
        support = row["representative"]["support_audit"]["support"]
        roots = support["root_targets"]
        qualifications = support["root_qualifications"]
        matches = [
            (root, qualification)
            for root, qualification in zip(roots, qualifications, strict=True)
            if str(root.get("role")) == role and str(root.get("canonical_smiles")) == smiles
        ]
        if len(matches) != 1:
            raise UgiBoundedHybridRouteCascadeError(
                f"component does not match exactly one qualified root: {role} {smiles}"
            )
        root_dict, qualification_dict = matches[0]
        route_root = RouteTarget(
            role=role,
            canonical_smiles=smiles,
            product_context_smiles=tuple(root_dict.get("product_context_smiles", ())),
        )
        try:
            request = ProposalRequest(
                route_root=route_root,
                target=route_root,
                depth=0,
                target_kind=ProposalTargetKind.ROOT,
                root_qualification=RootQualificationReceipt(
                    route_root=route_root,
                    qualification=_qualification_from_dict(qualification_dict),
                    terminal_sha256=support["terminal_sha256"],
                    generator_checkpoint_sha256=support["generator_checkpoint_sha256"],
                    l1_reaction_sha256=row["representative"]["l1_reaction_sha256"],
                    qualification_artifact_sha256=row["representative"]["support_audit"][
                        "support_sha256"
                    ],
                ),
                operational_policy_id=operational_policy_id,
                operational_policy_sha256=operational_policy_sha256,
            )
        except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
            target_rows.append(
                {
                    "component_sha256": row["component_sha256"],
                    "role": role,
                    "canonical_smiles": smiles,
                    "attempted": False,
                    "execution_failure": f"{type(error).__name__}: {error}",
                    "proposal_count": 0,
                    "deterministic_across_repeats": None,
                }
            )
            print(f"graph2edits {index}/{len(targets)} unqualified-root {role}", flush=True)
            continue
        try:
            repeats = [
                backend.propose_with_trace(request, maximum_proposals=maximum_proposals)
                for _ in range(repeat_count)
            ]
        except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
            target_rows.append(
                {
                    "component_sha256": row["component_sha256"],
                    "role": role,
                    "canonical_smiles": smiles,
                    "attempted": True,
                    "execution_failure": f"{type(error).__name__}: {error}",
                    "proposal_count": 0,
                    "deterministic_across_repeats": None,
                }
            )
            print(f"graph2edits {index}/{len(targets)} execution-failure {role}", flush=True)
            continue
        identities = [
            tuple(
                (proposal.reactant_smiles, proposal.model_score, proposal.rank)
                for proposal in batch.proposals
            )
            for batch in repeats
        ]
        deterministic = all(identity == identities[0] for identity in identities[1:])
        if not deterministic:
            raise UgiBoundedHybridRouteCascadeError(
                f"Graph2Edits output was not deterministic for {role} {smiles}"
            )
        for proposal in repeats[0].proposals:
            resolution = resolver.resolve(proposal)
            semantic = audit_semantic_family_equivalence(resolution, resolver.transforms)
            proposal_rows.append(
                {
                    "schema_version": "phase1_ugi_bounded_hybrid_route_graph2edits_ledger.v1",
                    "component_sha256": row["component_sha256"],
                    "target_id": row["target_id"],
                    "role": role,
                    "target_smiles": smiles,
                    "occurrence_count": int(row["occurrence_count"]),
                    "rank": proposal.rank,
                    "model_score": proposal.model_score,
                    "reactants": list(proposal.reactant_smiles),
                    "proposal_sha256": proposal.proposal_sha256,
                    "raw_discovery_resolution": resolution.to_dict(),
                    "semantic_equivalence": semantic,
                    "graph_consistent_discovery_hypothesis": semantic[
                        "graph_consistent_discovery_hypothesis"
                    ],
                    "authority": dict(PROPOSAL_ONLY_AUTHORITY),
                }
            )
        target_rows.append(
            {
                "component_sha256": row["component_sha256"],
                "role": role,
                "canonical_smiles": smiles,
                "attempted": True,
                "execution_failure": None,
                "proposal_count": len(repeats[0].proposals),
                "deterministic_across_repeats": True,
            }
        )
        print(
            f"graph2edits {index}/{len(targets)} {role} proposals={len(repeats[0].proposals)}",
            flush=True,
        )

    proposal_rows.sort(
        key=lambda item: (str(item["role"]), str(item["target_smiles"]), int(item["rank"]))
    )
    target_rows.sort(key=lambda item: (str(item["role"]), str(item["canonical_smiles"])))
    ledger_path = output_dir / "graph2edits_proposals.jsonl.gz"
    atomic_write(ledger_path, jsonl_gzip_bytes(proposal_rows))
    content = {
        "schema_version": "phase1_ugi_bounded_hybrid_route_cascade_graph2edits.v1",
        "status": "bounded_hybrid_route_cascade_graph2edits_complete",
        "config": {
            "path": str(contract.config_path.relative_to(repo)),
            "sha256": contract.config_sha256,
        },
        "exact_evidence_result": _pin_artifact(repo, upstream_path),
        "backend_manifest": manifest.to_dict(),
        "settings": {
            "device": str(settings["device"]),
            "maximum_proposals": maximum_proposals,
            "max_edit_steps": policy.max_edit_steps,
            "repeat_count": repeat_count,
        },
        "runtime": _runtime_receipt(),
        "summary": {
            "unresolved_component_targets": len(targets),
            "targets_attempted": sum(1 for row in target_rows if row["attempted"]),
            "targets_with_execution_failure": sum(
                1 for row in target_rows if row["execution_failure"] is not None
            ),
            "targets_with_proposals": sum(1 for row in target_rows if row["proposal_count"] > 0),
            "proposal_count": len(proposal_rows),
            "targets_with_graph_consistent_hypothesis": len(
                {
                    str(row["component_sha256"])
                    for row in proposal_rows
                    if row["graph_consistent_discovery_hypothesis"]
                }
            ),
            "semantic_status_counts": dict(
                sorted(
                    Counter(
                        str(row["semantic_equivalence"]["semantic_resolution_status"])
                        for row in proposal_rows
                    ).items()
                )
            ),
        },
        "targets": target_rows,
        "artifacts": {
            "graph2edits_proposals.jsonl.gz": {
                **_pin_artifact(repo, ledger_path),
                "rows": len(proposal_rows),
            }
        },
        "scientific_authority": dict(PROPOSAL_ONLY_AUTHORITY),
        "nonclaims": list(contract.config["nonclaims"]),
    }
    return _write_result(repo, output_dir / "graph2edits_result.json", content)


# ---------------------------------------------------------------------------
# stage: aizynthfinder
# ---------------------------------------------------------------------------


def stage_aizynthfinder(contract: Any) -> dict[str, Any]:
    import random

    import numpy as np

    repo = contract.repo
    output_dir = contract.output_dir
    exact_path = _require_stage_artifact(
        repo, output_dir / "exact_evidence_result.json", label="exact evidence"
    )
    g2e_path = _require_stage_artifact(
        repo, output_dir / "graph2edits_result.json", label="graph2edits"
    )
    exact = load_json(exact_path, label="exact evidence result")
    g2e = load_json(g2e_path, label="graph2edits result")
    if g2e.get("status") != "bounded_hybrid_route_cascade_graph2edits_complete":
        raise UgiBoundedHybridRouteCascadeError("graph2edits stage is not complete")
    component_path = repo / exact["artifacts"]["component_exact_evidence.jsonl.gz"]["path"]
    component_rows = read_jsonl_gzip(component_path, label="component exact evidence")
    g2e_ledger_path = repo / g2e["artifacts"]["graph2edits_proposals.jsonl.gz"]["path"]
    if sha256_file(g2e_ledger_path) != g2e["artifacts"]["graph2edits_proposals.jsonl.gz"]["sha256"]:
        raise UgiBoundedHybridRouteCascadeError("Graph2Edits proposal ledger changed")
    g2e_rows = read_jsonl_gzip(g2e_ledger_path, label="graph2edits proposals")

    resolved_by_g2e = {
        str(row["component_sha256"])
        for row in g2e_rows
        if row["graph_consistent_discovery_hypothesis"]
    }
    targets = [
        row
        for row in _unresolved_component_rows(component_rows)
        if str(row["component_sha256"]) not in resolved_by_g2e
    ]

    settings = contract.stage("aizynthfinder")
    single_step = settings["single_step"]
    full_search = settings["full_search"]
    seed = int(contract.config["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))

    runtime_manifest = load_json(
        contract.paths["aizynthfinder_runtime_manifest"], label="AiZynthFinder runtime manifest"
    )
    for asset in runtime_manifest["assets"]:
        asset_path = repo / asset["path"]
        if not asset_path.is_file():
            raise UgiBoundedHybridRouteCascadeError(
                f"AiZynthFinder asset is absent: {asset['path']}"
            )
        if asset_path.stat().st_size != int(asset["bytes"]):
            raise UgiBoundedHybridRouteCascadeError(
                f"AiZynthFinder asset byte count changed: {asset['path']}"
            )
    engine_config_path = repo / runtime_manifest["config"]["path"]
    if sha256_file(engine_config_path) != runtime_manifest["config"]["sha256"]:
        raise UgiBoundedHybridRouteCascadeError("AiZynthFinder engine config changed")

    from aizynthfinder.aizynthfinder import AiZynthExpander, AiZynthFinder
    from rdkit import Chem

    def _canonical_reactants(reaction: Any) -> list[str]:
        outcomes = getattr(reaction, "reactants", ())
        if not outcomes:
            return []
        reactants: list[str] = []
        for molecule in outcomes[0]:
            parsed = Chem.MolFromSmiles(molecule.smiles)
            if parsed is None:
                raise UgiBoundedHybridRouteCascadeError(
                    "AiZynthFinder returned an invalid reactant"
                )
            reactants.append(Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=False))
        return sorted(reactants)

    def _proposal_record(group: tuple[Any, ...], *, rank: int) -> dict[str, Any]:
        reaction = group[0]
        metadata = dict(getattr(reaction, "metadata", {}))
        retained = {
            key: metadata.get(key)
            for key in (
                "policy_name",
                "policy_probability",
                "policy_probability_rank",
                "feasibility",
                "template_hash",
                "template_code",
                "library_occurence",
                "classification",
                "mapped_reaction_smiles",
            )
        }
        reactants = _canonical_reactants(reaction)
        return {
            "rank": rank,
            "canonical_reactants": reactants,
            "proposal_sha256": sha256_payload(
                {"rank": rank, "canonical_reactants": reactants, "metadata": retained}
            ),
            "metadata": retained,
            "authority": dict(PROPOSAL_ONLY_AUTHORITY),
        }

    expander = AiZynthExpander(configfile=str(engine_config_path))
    expander.expansion_policy.select(list(single_step["expansion_policies"]))
    expander.filter_policy.select(list(single_step["filter_policies"]))
    finder = AiZynthFinder(configfile=str(engine_config_path))
    finder.expansion_policy.select(list(single_step["expansion_policies"]))
    finder.filter_policy.select(list(single_step["filter_policies"]))
    finder.stock.select(list(full_search["stocks"]))
    finder.config.search.time_limit = int(full_search["time_limit_seconds"])
    finder.config.search.iteration_limit = int(full_search["iteration_limit"])
    finder.config.search.max_transforms = int(full_search["maximum_transforms"])

    hypothesis_rows: list[dict[str, Any]] = []
    for index, row in enumerate(targets, start=1):
        role = str(row["role"])
        smiles = str(row["canonical_smiles"])
        record: dict[str, Any] = {
            "schema_version": "phase1_ugi_bounded_hybrid_route_aizynthfinder_ledger.v1",
            "component_sha256": row["component_sha256"],
            "target_id": row["target_id"],
            "role": role,
            "target_smiles": smiles,
            "occurrence_count": int(row["occurrence_count"]),
            "attempted": True,
            "execution_failure": None,
            "single_step_proposals": [],
            "full_search": None,
            "authority": dict(PROPOSAL_ONLY_AUTHORITY),
        }
        try:
            groups = expander.do_expansion(smiles, return_n=int(single_step["maximum_proposals"]))
            record["single_step_proposals"] = [
                _proposal_record(group, rank=rank)
                for rank, group in enumerate(groups, start=1)
                if group
            ]
        except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
            record["execution_failure"] = f"single_step:{type(error).__name__}: {error}"
        try:
            finder.target_smiles = smiles
            finder.prepare_tree()
            finder.tree_search(show_progress=False)
            finder.build_routes()
            stats = finder.extract_statistics()
            routes = finder.routes.dict_with_extra(include_metadata=True, include_scores=True)
            record["full_search"] = {
                "is_solved_to_public_stock": bool(stats.get("is_solved")),
                "statistics": json.loads(json.dumps(stats, default=str)),
                "top_route_hypotheses": json.loads(
                    json.dumps(
                        routes[: int(full_search["maximum_routes_stored"])],
                        default=str,
                    )
                ),
                "authority": "planner_diagnostic_not_route_evidence",
            }
        except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
            previous = record["execution_failure"]
            failure = f"full_search:{type(error).__name__}: {error}"
            record["execution_failure"] = failure if previous is None else f"{previous}; {failure}"
        hypothesis_rows.append(record)
        solved = bool((record["full_search"] or {}).get("is_solved_to_public_stock"))
        print(
            f"aizynthfinder {index}/{len(targets)} {role} "
            f"proposals={len(record['single_step_proposals'])} solved={solved}",
            flush=True,
        )

    hypothesis_rows.sort(key=lambda item: (str(item["role"]), str(item["target_smiles"])))
    ledger_path = output_dir / "aizynthfinder_hypotheses.jsonl.gz"
    atomic_write(ledger_path, jsonl_gzip_bytes(hypothesis_rows))
    content = {
        "schema_version": "phase1_ugi_bounded_hybrid_route_cascade_aizynthfinder.v1",
        "status": "bounded_hybrid_route_cascade_aizynthfinder_complete",
        "config": {
            "path": str(contract.config_path.relative_to(repo)),
            "sha256": contract.config_sha256,
        },
        "exact_evidence_result": _pin_artifact(repo, exact_path),
        "graph2edits_result": _pin_artifact(repo, g2e_path),
        "runtime_manifest": {
            "path": str(contract.paths["aizynthfinder_runtime_manifest"].relative_to(repo)),
            "sha256": sha256_file(contract.paths["aizynthfinder_runtime_manifest"]),
        },
        "settings": {
            "seed": seed,
            "single_step": json.loads(json.dumps(single_step)),
            "full_search": json.loads(json.dumps(full_search)),
        },
        "runtime": _runtime_receipt(),
        "summary": {
            "residual_targets": len(targets),
            "targets_excluded_because_graph2edits_hypothesis_exists": len(resolved_by_g2e),
            "targets_with_single_step_proposals": sum(
                1 for row in hypothesis_rows if row["single_step_proposals"]
            ),
            "single_step_proposal_count": sum(
                len(row["single_step_proposals"]) for row in hypothesis_rows
            ),
            "full_search_solved_to_public_stock": sum(
                1
                for row in hypothesis_rows
                if (row["full_search"] or {}).get("is_solved_to_public_stock")
            ),
            "execution_failures": sum(
                1 for row in hypothesis_rows if row["execution_failure"] is not None
            ),
        },
        "artifacts": {
            "aizynthfinder_hypotheses.jsonl.gz": {
                **_pin_artifact(repo, ledger_path),
                "rows": len(hypothesis_rows),
            }
        },
        "scientific_authority": dict(PROPOSAL_ONLY_AUTHORITY),
        "nonclaims": list(contract.config["nonclaims"]),
    }
    return _write_result(repo, output_dir / "aizynthfinder_result.json", content)


# ---------------------------------------------------------------------------
# stage: adjudicate
# ---------------------------------------------------------------------------


def stage_adjudicate(contract: Any) -> dict[str, Any]:
    repo = contract.repo
    output_dir = contract.output_dir
    exact_path = _require_stage_artifact(
        repo, output_dir / "exact_evidence_result.json", label="exact evidence"
    )
    g2e_path = _require_stage_artifact(
        repo, output_dir / "graph2edits_result.json", label="graph2edits"
    )
    aizynth_path = _require_stage_artifact(
        repo, output_dir / "aizynthfinder_result.json", label="aizynthfinder"
    )
    exact = load_json(exact_path, label="exact evidence result")
    g2e = load_json(g2e_path, label="graph2edits result")
    aizynth = load_json(aizynth_path, label="aizynthfinder result")

    component_path = repo / exact["artifacts"]["component_exact_evidence.jsonl.gz"]["path"]
    if (
        sha256_file(component_path)
        != (exact["artifacts"]["component_exact_evidence.jsonl.gz"]["sha256"])
    ):
        raise UgiBoundedHybridRouteCascadeError("component exact-evidence ledger changed")
    component_rows = read_jsonl_gzip(component_path, label="component exact evidence")
    g2e_ledger = repo / g2e["artifacts"]["graph2edits_proposals.jsonl.gz"]["path"]
    aizynth_ledger = repo / aizynth["artifacts"]["aizynthfinder_hypotheses.jsonl.gz"]["path"]
    for path, pin, label in (
        (g2e_ledger, g2e["artifacts"]["graph2edits_proposals.jsonl.gz"]["sha256"], "graph2edits"),
        (
            aizynth_ledger,
            aizynth["artifacts"]["aizynthfinder_hypotheses.jsonl.gz"]["sha256"],
            "aizynthfinder",
        ),
    ):
        if sha256_file(path) != pin:
            raise UgiBoundedHybridRouteCascadeError(f"{label} ledger changed")
    g2e_rows = read_jsonl_gzip(g2e_ledger, label="graph2edits proposals")
    aizynth_rows = read_jsonl_gzip(aizynth_ledger, label="aizynthfinder hypotheses")

    g2e_targets = {str(row["component_sha256"]): row for row in g2e["targets"]}
    g2e_by_component: dict[str, list[dict[str, Any]]] = {}
    for row in g2e_rows:
        g2e_by_component.setdefault(str(row["component_sha256"]), []).append(row)
    aizynth_by_component = {str(row["component_sha256"]): row for row in aizynth_rows}

    graded = {
        (str(row["role"]), str(row["canonical_smiles"])): row
        for row in read_csv_gzip(contract.paths["graded_evidence_ledger"])
    }

    component_ledger: list[dict[str, Any]] = []
    leaf_rows: list[dict[str, Any]] = []
    for row in component_rows:
        key = str(row["component_sha256"])
        role = str(row["role"])
        smiles = str(row["canonical_smiles"])
        strict_complete = bool(row["strict_complete"])
        assessment_outcome = str(row["assessment_outcome"])

        g2e_target = g2e_targets.get(key)
        g2e_proposals = g2e_by_component.get(key, [])
        g2e_record = {
            "attempted": bool(g2e_target["attempted"]) if g2e_target else False,
            "execution_failure": (g2e_target or {}).get("execution_failure"),
            "proposal_count": len(g2e_proposals),
            "deterministic_across_repeats": (g2e_target or {}).get("deterministic_across_repeats"),
            "graph_consistent_hypothesis": any(
                proposal["graph_consistent_discovery_hypothesis"] for proposal in g2e_proposals
            ),
            "semantic_status_counts": dict(
                sorted(
                    Counter(
                        str(proposal["semantic_equivalence"]["semantic_resolution_status"])
                        for proposal in g2e_proposals
                    ).items()
                )
            ),
            "forward_resolution_statuses": dict(
                sorted(
                    Counter(
                        str(proposal["raw_discovery_resolution"].get("status"))
                        for proposal in g2e_proposals
                    ).items()
                )
            ),
            "proposals": [
                {
                    "rank": proposal["rank"],
                    "reactants": proposal["reactants"],
                    "proposal_sha256": proposal["proposal_sha256"],
                    "discovery_status": proposal["raw_discovery_resolution"].get("status"),
                    "semantic_resolution_status": proposal["semantic_equivalence"][
                        "semantic_resolution_status"
                    ],
                    "graph_consistent_discovery_hypothesis": proposal[
                        "graph_consistent_discovery_hypothesis"
                    ],
                }
                for proposal in sorted(g2e_proposals, key=lambda item: int(item["rank"]))
            ],
        }

        aizynth_row = aizynth_by_component.get(key)
        aizynth_record = {
            "attempted": bool(aizynth_row["attempted"]) if aizynth_row else False,
            "execution_failure": (aizynth_row or {}).get("execution_failure"),
            "single_step_proposal_count": len((aizynth_row or {}).get("single_step_proposals", [])),
            "full_search_solved_to_public_stock": bool(
                ((aizynth_row or {}).get("full_search") or {}).get("is_solved_to_public_stock")
            ),
            "stored_route_hypotheses": len(
                ((aizynth_row or {}).get("full_search") or {}).get("top_route_hypotheses", [])
            ),
            "single_step_proposals": [
                {
                    "rank": proposal["rank"],
                    "canonical_reactants": proposal["canonical_reactants"],
                    "proposal_sha256": proposal["proposal_sha256"],
                }
                for proposal in (aizynth_row or {}).get("single_step_proposals", [])
            ],
        }

        engines_attempted = [
            record["attempted"] for record in (g2e_record, aizynth_record) if record["attempted"]
        ]
        engines_failed = [
            record
            for record in (g2e_record, aizynth_record)
            if record["attempted"] and record["execution_failure"] is not None
        ]
        proposal_search_censored = bool(engines_attempted) and len(engines_failed) == len(
            engines_attempted
        )

        state, basis = derive_final_component_state(
            strict_complete=strict_complete,
            assessment_outcome=assessment_outcome,
            proposal_search_censored=proposal_search_censored,
        )
        evidence = graded.get((role, smiles))
        graded_record = {
            "graded_evidence_class": (
                str(evidence["graded_evidence_class"])
                if evidence
                else "not_in_graded_development_ledger"
            ),
            "program_family": (evidence or {}).get("program_family") or None,
            "projection_leaf_status": (evidence or {}).get("projection_leaf_status") or None,
            "family_projection_is_exact_route_closure": False,
        }
        leaves = row["leaves"]
        for leaf in leaves:
            leaf_rows.append(
                {
                    "schema_version": LEAF_LEDGER_SCHEMA_VERSION,
                    "component_sha256": key,
                    "parent_role": role,
                    "parent_component_smiles": smiles,
                    "leaf_smiles": leaf["leaf_smiles"],
                    "depth": leaf["depth"],
                    "outcome": leaf["outcome"],
                    "detail": leaf["detail"],
                    "evidence_tiers": leaf["evidence_tiers"],
                    "closed_to_current_terminal_material": leaf["outcome"] == "complete",
                    "unresolved_leaf_class": (
                        None if leaf["outcome"] == "complete" else leaf["outcome"]
                    ),
                    "procurement_snapshot": dict(contract.config["procurement_snapshot"]),
                }
            )
        component_ledger.append(
            {
                "schema_version": COMPONENT_LEDGER_SCHEMA_VERSION,
                "component_sha256": key,
                "target_id": row["target_id"],
                "role": role,
                "canonical_smiles": smiles,
                "occurrence_count": int(row["occurrence_count"]),
                "descriptors": component_descriptors(smiles),
                "exact_evidence": {
                    "assessment_outcome": assessment_outcome,
                    "strict_complete": strict_complete,
                    "unresolved_disposition": unresolved_disposition(
                        row.get("route_tree"), strict_complete=strict_complete
                    ),
                    "root_expanded": bool(
                        isinstance(row.get("route_tree"), dict)
                        and row["route_tree"].get("children")
                    ),
                    "guidance_reason": str(row["guidance_reason"]),
                    "cache_key_sha256": str(row["cache_key_sha256"]),
                    "component_value": row["component_value"],
                    "leaf_count": len(leaves),
                    "current_terminal_leaf_count": sum(
                        1 for leaf in leaves if leaf["outcome"] == "complete"
                    ),
                    "unresolved_leaf_classes": row["unresolved_leaf_classes"],
                },
                "graded_family_projection": graded_record,
                "graph2edits": g2e_record,
                "aizynthfinder": aizynth_record,
                "adjudication": {
                    "independent_forward_verification_required": True,
                    "proposal_search_censored": proposal_search_censored,
                    "proposal_only_engines_cannot_close_route": True,
                    "closure_authority": "independent_exact_recursive_evidence_and_procurement",
                },
                "final_component_state": state,
                "final_component_state_basis": basis,
                "nonclaims": [
                    "A Graph2Edits or AiZynthFinder hypothesis is not route evidence.",
                    "A public-stock planner solution is not current procurement evidence.",
                    "An unresolved component is unresolved under the frozen bounded cascade, "
                    "evidence and 3 August procurement snapshot, not unsynthesizable.",
                ],
            }
        )

    # Gate G4: nothing may reach `complete` without independent exact closure.
    for row in component_ledger:
        if (
            row["final_component_state"] == COMPLETE
            and not row["exact_evidence"]["strict_complete"]
        ):
            raise UgiBoundedHybridRouteCascadeError(
                "a component reached complete without independent exact closure"
            )
    if len(component_ledger) != len(component_rows):
        raise UgiBoundedHybridRouteCascadeError("adjudication changed the component denominator")

    component_ledger.sort(key=lambda item: (str(item["role"]), str(item["canonical_smiles"])))
    leaf_rows.sort(
        key=lambda item: (
            str(item["parent_role"]),
            str(item["parent_component_smiles"]),
            int(item["depth"]),
            str(item["leaf_smiles"]),
        )
    )
    component_ledger_path = output_dir / "component_route_ledger.jsonl.gz"
    leaf_ledger_path = output_dir / "leaf_ledger.jsonl.gz"
    atomic_write(component_ledger_path, jsonl_gzip_bytes(component_ledger))
    atomic_write(leaf_ledger_path, jsonl_gzip_bytes(leaf_rows))

    content = {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "status": "bounded_hybrid_route_cascade_adjudication_complete",
        "config": {
            "path": str(contract.config_path.relative_to(repo)),
            "sha256": contract.config_sha256,
        },
        "exact_evidence_result": _pin_artifact(repo, exact_path),
        "graph2edits_result": _pin_artifact(repo, g2e_path),
        "aizynthfinder_result": _pin_artifact(repo, aizynth_path),
        "graded_evidence_ledger": {
            "path": str(contract.paths["graded_evidence_ledger"].relative_to(repo)),
            "sha256": sha256_file(contract.paths["graded_evidence_ledger"]),
        },
        "runtime": _runtime_receipt(),
        "policy": {
            "component_states": list(COMPONENT_STATES),
            "closure_requires_independent_exact_recursive_evidence": True,
            "proposal_alone_may_close_route": False,
            "public_stock_solution_is_procurement_evidence": False,
            "family_projection_is_exact_route_closure": False,
        },
        "summary": {
            "components": len(component_ledger),
            "states": dict(
                sorted(Counter(row["final_component_state"] for row in component_ledger).items())
            ),
            "states_by_role": {
                role: dict(
                    sorted(
                        Counter(
                            row["final_component_state"]
                            for row in component_ledger
                            if row["role"] == role
                        ).items()
                    )
                )
                for role in ROLE_NAMES
            },
            "graded_evidence_classes": dict(
                sorted(
                    Counter(
                        row["graded_family_projection"]["graded_evidence_class"]
                        for row in component_ledger
                    ).items()
                )
            ),
            "unresolved_disposition": dict(
                sorted(
                    Counter(
                        str(row["exact_evidence"]["unresolved_disposition"])
                        for row in component_ledger
                        if row["exact_evidence"]["unresolved_disposition"] is not None
                    ).items()
                )
            ),
            "leaf_rows": len(leaf_rows),
            "unresolved_leaf_classes": dict(
                sorted(
                    Counter(
                        str(row["unresolved_leaf_class"])
                        for row in leaf_rows
                        if row["unresolved_leaf_class"] is not None
                    ).items()
                )
            ),
        },
        "artifacts": {
            "component_route_ledger.jsonl.gz": {
                **_pin_artifact(repo, component_ledger_path),
                "rows": len(component_ledger),
                "schema_version": COMPONENT_LEDGER_SCHEMA_VERSION,
            },
            "leaf_ledger.jsonl.gz": {
                **_pin_artifact(repo, leaf_ledger_path),
                "rows": len(leaf_rows),
                "schema_version": LEAF_LEDGER_SCHEMA_VERSION,
            },
        },
        "decision_gates": {"g4_proposal_only_cannot_close_route": "passed"},
        "nonclaims": list(contract.config["nonclaims"]),
    }
    return _write_result(repo, output_dir / "adjudication_result.json", content)


# ---------------------------------------------------------------------------
# stage: audit
# ---------------------------------------------------------------------------


def stage_audit(contract: Any) -> dict[str, Any]:
    repo = contract.repo
    output_dir = contract.output_dir
    exact_path = _require_stage_artifact(
        repo, output_dir / "exact_evidence_result.json", label="exact evidence"
    )
    adjudication_path = _require_stage_artifact(
        repo, output_dir / "adjudication_result.json", label="adjudication"
    )
    exact = load_json(exact_path, label="exact evidence result")
    adjudication = load_json(adjudication_path, label="adjudication result")

    candidate_exact_path = repo / exact["artifacts"]["candidate_exact_ledger.jsonl.gz"]["path"]
    if (
        sha256_file(candidate_exact_path)
        != (exact["artifacts"]["candidate_exact_ledger.jsonl.gz"]["sha256"])
    ):
        raise UgiBoundedHybridRouteCascadeError("candidate exact ledger changed")
    candidate_exact_rows = read_jsonl_gzip(candidate_exact_path, label="candidate exact ledger")
    component_ledger_path = (
        repo / adjudication["artifacts"]["component_route_ledger.jsonl.gz"]["path"]
    )
    if (
        sha256_file(component_ledger_path)
        != (adjudication["artifacts"]["component_route_ledger.jsonl.gz"]["sha256"])
    ):
        raise UgiBoundedHybridRouteCascadeError("component route ledger changed")
    component_rows = read_jsonl_gzip(component_ledger_path, label="component route ledger")
    component_by_key = {str(row["component_sha256"]): row for row in component_rows}

    records = build_candidate_population(contract)
    assert_expected_population(contract, records)
    by_product = {record.canonical_product: record for record in records}
    if len(by_product) != len(records):
        raise UgiBoundedHybridRouteCascadeError("shortlist products are not unique")

    reporting_fields = list(contract.config["reporting_only_fields"])
    rescoring = rescoring_index(read_csv_gzip(contract.paths["terminal_rescoring_ledger"]))
    candidate_rows: list[dict[str, Any]] = []
    for exact_row in candidate_exact_rows:
        product = str(exact_row["canonical_product"])
        record = by_product.get(product)
        if record is None:
            raise UgiBoundedHybridRouteCascadeError(
                "route ledger contains a product absent from the shortlist"
            )
        short = record.shortlist_row
        role_states: list[str] = []
        role_records: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        admitted = bool(exact_row["route_assessment_admitted"])
        # A candidate rejected by the declared graph-support gate has no role
        # assessments; its three roles are synthesized from the component
        # ledger so the roll-up and denominators stay complete.
        exact_role_records = exact_row["roles"] or [
            {
                "role": role,
                "canonical_smiles": record.components[role],
                "component_sha256": component_sha256(role, record.components[role]),
            }
            for role in ROLE_NAMES
        ]
        for role_record in exact_role_records:
            key = str(role_record["component_sha256"])
            component = component_by_key.get(key)
            if component is None:
                raise UgiBoundedHybridRouteCascadeError(
                    "candidate references a component absent from the component ledger"
                )
            role_states.append(str(component["final_component_state"]))
            role_records.append(
                {
                    "role": str(role_record["role"]),
                    "canonical_smiles": str(role_record["canonical_smiles"]),
                    "component_sha256": key,
                    "final_component_state": str(component["final_component_state"]),
                    "final_component_state_basis": str(component["final_component_state_basis"]),
                    "exact_assessment_outcome": component["exact_evidence"]["assessment_outcome"],
                    "guidance_reason": component["exact_evidence"]["guidance_reason"],
                    "graded_evidence_class": component["graded_family_projection"][
                        "graded_evidence_class"
                    ],
                    "graph2edits_proposal_count": component["graph2edits"]["proposal_count"],
                    "aizynthfinder_single_step_proposal_count": component["aizynthfinder"][
                        "single_step_proposal_count"
                    ],
                    "aizynthfinder_solved_to_public_stock": component["aizynthfinder"][
                        "full_search_solved_to_public_stock"
                    ],
                    "unresolved_leaf_classes": component["exact_evidence"][
                        "unresolved_leaf_classes"
                    ],
                }
            )
            receipts.append(
                {
                    "role": str(role_record["role"]),
                    "component_sha256": key,
                    "shared_with_candidates": int(component["occurrence_count"]),
                    "cache_key_sha256": component["exact_evidence"]["cache_key_sha256"],
                }
            )
        state = product_route_state(role_states)
        descriptors_by_role = {
            str(role_record["role"]): component_by_key[str(role_record["component_sha256"])][
                "descriptors"
            ]
            for role_record in exact_role_records
        }
        aldehyde = descriptors_by_role["oxoester_aldehyde_body_tail"]
        product_descriptors = {
            "product_unsaturated": any(
                value["unsaturated"] for value in descriptors_by_role.values()
            ),
            "product_branched": any(
                value["carbon_branch_points"] > 0 for value in descriptors_by_role.values()
            ),
            "product_ester_groups": sum(
                value["ester_groups"] for value in descriptors_by_role.values()
            ),
            "product_ether_oxygens": sum(
                value["ether_oxygens"] for value in descriptors_by_role.values()
            ),
            "product_amide_groups": sum(
                value["amide_groups"] for value in descriptors_by_role.values()
            ),
            "aldehyde_carbon_atoms": aldehyde["carbon_atoms"],
            "aldehyde_tail_length_bucket": tail_length_bucket(aldehyde["carbon_atoms"]),
            "by_role": descriptors_by_role,
        }
        candidate_rows.append(
            {
                "schema_version": CANDIDATE_LEDGER_SCHEMA_VERSION,
                "canonical_product": product,
                "components": dict(record.components),
                "route_assessment_admitted": admitted,
                "admission_failure": exact_row["admission_failure"],
                "terminal_sha256": exact_row["terminal_sha256"],
                "exact_l1_reverified": bool(exact_row["exact_l1_reverified"]),
                "l1_reverification": exact_row["l1_reverification"],
                "exact_evidence_complete_roles": int(exact_row["strict_complete_role_count"]),
                "roles": role_records,
                "component_receipt_refs": receipts,
                "product_route_state": state,
                "descriptors": product_descriptors,
                "proposal_coverage": {
                    "graph2edits_any_role": any(
                        item["graph2edits_proposal_count"] > 0 for item in role_records
                    ),
                    "aizynthfinder_incremental_any_role": any(
                        item["aizynthfinder_single_step_proposal_count"] > 0
                        for item in role_records
                    ),
                    "union_any_role": any(
                        item["graph2edits_proposal_count"] > 0
                        or item["aizynthfinder_single_step_proposal_count"] > 0
                        for item in role_records
                    ),
                },
                "unresolved_risks": sorted(
                    {
                        f"{item['role']}:{item['final_component_state']}"
                        for item in role_records
                        if item["final_component_state"] != COMPLETE
                    }
                ),
                "reporting": {name: short.get(name) for name in sorted(reporting_fields)}
                | resolve_potency_reporting(record, rescoring)
                | {
                    "arm_id": record.arm_id,
                    "draw_index": record.draw_index,
                    "program": short.get("program"),
                    "program_sha256": short.get("program_sha256"),
                },
                "route_inputs_excluded_reporting_fields": True,
                "reporting_fields_joined_after_route_computation": True,
            }
        )

    if len(candidate_rows) != len(records):
        raise UgiBoundedHybridRouteCascadeError("audit changed the candidate denominator")
    candidate_rows.sort(key=lambda row: str(row["canonical_product"]))
    summary = summarize_attrition(candidate_rows, component_rows)
    branch = branch_sufficiency_recommendation(candidate_rows)

    candidate_path = output_dir / "candidate_route_ledger.jsonl.gz"
    attrition_path = output_dir / "attrition_by_arm_cohort.csv.gz"
    atomic_write(candidate_path, jsonl_gzip_bytes(candidate_rows))
    atomic_write(
        attrition_path,
        csv_gzip_bytes(ATTRITION_CSV_FIELDS, attrition_csv_rows(summary)),
    )

    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "bounded_hybrid_route_cascade_complete",
        "config": {
            "path": str(contract.config_path.relative_to(repo)),
            "sha256": contract.config_sha256,
        },
        "assessment_as_of_utc": contract.assessment_as_of_utc,
        "procurement_snapshot": dict(contract.config["procurement_snapshot"]),
        "stage_results": {
            "preflight": _pin_artifact(repo, output_dir / "preflight.json"),
            "exact_evidence": _pin_artifact(repo, exact_path),
            "graph2edits": _pin_artifact(repo, output_dir / "graph2edits_result.json"),
            "aizynthfinder": _pin_artifact(repo, output_dir / "aizynthfinder_result.json"),
            "adjudication": _pin_artifact(repo, adjudication_path),
        },
        "blindness_receipt": exact["blindness_receipt"],
        "cache": exact["cache"],
        "runtime": _runtime_receipt(),
        "summary": {
            "candidates": len(candidate_rows),
            "unique_products": len({row["canonical_product"] for row in candidate_rows}),
            "components": len(component_rows),
            "route_assessment_admitted_candidates": sum(
                1 for row in candidate_rows if row["route_assessment_admitted"]
            ),
            "route_assessment_not_admitted_candidates": [
                {
                    "canonical_product": row["canonical_product"],
                    "arm_id": row["reporting"]["arm_id"],
                    "cohort": row["reporting"].get("cohort"),
                    "gate": row["admission_failure"]["gate"],
                    "out_of_vocabulary_atom_states": row["admission_failure"]["detail"][
                        "out_of_vocabulary_atom_states"
                    ],
                }
                for row in candidate_rows
                if not row["route_assessment_admitted"]
            ],
            "product_route_states": dict(
                sorted(Counter(row["product_route_state"] for row in candidate_rows).items())
            ),
            "component_states": adjudication["summary"]["states"],
            "unresolved_disposition": adjudication["summary"]["unresolved_disposition"],
            "attrition": summary,
            "branch_sufficiency": branch,
        },
        "interpretation": {
            "what_route_complete_measures": (
                "All three Ugi roles closed under the frozen exact-evidence source, which is a "
                "hand-curated index of qualified component routes reaching a 60-item current "
                "terminal-material ledger. It is NOT a retrosynthetic search result and NOT a "
                "synthesizability estimate."
            ),
            "why_the_split_matters": (
                "A component absent from the index returns missing_knowledge at depth zero having "
                "attempted no disconnection. Reading the closure rate without the "
                "never-expanded/after-search split would mistake an index-miss rate for a search "
                "outcome."
            ),
            "unresolved_never_expanded": sum(
                1
                for row in component_rows
                if row["exact_evidence"]["unresolved_disposition"] == NEVER_EXPANDED
            ),
            "unresolved_after_search": sum(
                1
                for row in component_rows
                if row["exact_evidence"]["unresolved_disposition"] is not None
                and row["exact_evidence"]["unresolved_disposition"] != NEVER_EXPANDED
            ),
            "comparable_prior_census": (
                "results/phase1/ugi_route_aware_panel_feasibility_v3 reported the same metric on "
                "an older 1,855-row population at approximately 6.5-6.9% unique-product closure; "
                "this run's rate is directly comparable because the metric definition is unchanged."
            ),
            "proposal_engines_closed_zero_routes": True,
            "not_a_synthesizability_claim": True,
        },
        "artifacts": {
            "candidate_route_ledger.jsonl.gz": {
                **_pin_artifact(repo, candidate_path),
                "rows": len(candidate_rows),
                "schema_version": CANDIDATE_LEDGER_SCHEMA_VERSION,
            },
            "attrition_by_arm_cohort.csv.gz": _pin_artifact(repo, attrition_path),
            "component_route_ledger.jsonl.gz": _pin_artifact(repo, component_ledger_path),
            "leaf_ledger.jsonl.gz": _pin_artifact(
                repo, repo / adjudication["artifacts"]["leaf_ledger.jsonl.gz"]["path"]
            ),
        },
        "decision_gates": {
            "g0_preflight_population_and_hashes": "passed",
            "g1_exact_constitutional_l1_all_rows": "passed",
            "g2_arm_blind_cache_and_search": "passed",
            "g3_identical_budget_all_arms": "passed",
            "g4_proposal_only_cannot_close_route": "passed",
            "g5_denominator_preservation": "passed",
            "g6_report_balance_never_rebalance_by_dropping": "reported",
            "g7_branch_sufficiency_recommendation_only": "reported",
        },
        "decision": {
            "panel_locked": False,
            "preregistration_created": False,
            "candidate_selection_performed": False,
            "procurement_refresh_required_before_panel_lock": True,
            "next_gate": "user or PI review of the route-aware decision package",
        },
        "scope": dict(contract.config["scope"]),
        "nonclaims": list(contract.config["nonclaims"]),
    }
    return _write_result(repo, output_dir / "result.json", content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


STAGES = {
    "preflight": stage_preflight,
    "exact-evidence": stage_exact_evidence,
    "graph2edits": stage_graph2edits,
    "aizynthfinder": stage_aizynthfinder,
    "adjudicate": stage_adjudicate,
    "audit": stage_audit,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_ugi_bounded_hybrid_route_cascade_v1.json"),
    )
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    contract = load_cascade_contract(repo, config_path)
    result = STAGES[args.stage](contract)
    print(json.dumps(result.get("summary", result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
