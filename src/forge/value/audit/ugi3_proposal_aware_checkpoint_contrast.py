"""Proposal-aware, family-neutral route contrast on frozen checkpoint completions.

This audit is the missing bridge between proposal-only Graph2Edits discovery
and the earlier exact-ledger checkpoint contrast.  It independently accepts a
route hypothesis only when a registered step reconstructs the target and all
required leaves are current, or when a proposal matches the final step of a
mechanically derived Ugi-component program whose complete step sequence is
forward verified and terminal closed.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.route.evidence.ugi3_stepwise_route_program_adjudication import (
    _execute_program,
    _load_transforms,
)
from forge.route.sources.ugi3_source_neutral_proposal_adjudication import (
    Ugi3ProposalAdjudicationError,
    _content_sha256,
    _gzip_jsonl,
    _read_json,
    _read_jsonl_gzip,
    _sha256_file,
)
from forge.route.sources.ugi3_source_neutral_proposal_adjudication_v2 import (
    _load_current_terminal_keys,
    _matching_forward_precursors,
    _program_from_target,
)
from forge.route.terminals.ugi3_agile_template_saturation_stress import projected_leaf_candidates
from forge.route.terminals.ugi3_precursor_leaf_closure import constitutional_key
from forge.route.terminals.ugi3_virtual_programs import Ugi3VirtualProgramError
from forge.value.audit.ugi3_source_bounded_route_value_contrast import EXACT
from forge.value.audit.ugi3_stepwise_route_value_contrast import FAMILY_ALL

CONFIG_SCHEMA_VERSION = "forge.ugi3_proposal_aware_checkpoint_contrast_config.v1"
RESULT_SCHEMA_VERSION = "forge.ugi3_proposal_aware_checkpoint_contrast.v1"
LEDGER_SCHEMA_VERSION = "forge.ugi3_proposal_aware_checkpoint_component_ledger.v1"


class ProposalAwareContrastError(RuntimeError):
    """Raised when a frozen proposal-aware contrast input is malformed."""


def _safe_program(role: str, target: str) -> tuple[str, list[dict[str, Any]]] | None:
    try:
        return _program_from_target(role=role, target=target)
    except (Ugi3VirtualProgramError, Ugi3ProposalAdjudicationError):
        return None


def _semantic_graph_consistent(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("graph_consistent_discovery_hypothesis")
        or (
            isinstance(row.get("semantic_equivalence"), Mapping)
            and row["semantic_equivalence"].get("graph_consistent_discovery_hypothesis")
        )
    )


def _direct_current_trace(
    proposals: Sequence[Mapping[str, Any]], current_terminal_keys: set[str]
) -> dict[str, Any] | None:
    """Return the first registered forward trace whose complete precursor tuple is current."""

    for proposal in sorted(proposals, key=lambda value: int(value["rank"])):
        if not _semantic_graph_consistent(proposal):
            continue
        resolution = proposal.get("raw_discovery_resolution")
        if not isinstance(resolution, Mapping):
            continue
        for trace in resolution.get("traces", []):
            if not isinstance(trace, Mapping) or not bool(trace.get("target_reconstructed")):
                continue
            reactants = trace.get("role_ordered_reactants")
            if not isinstance(reactants, list) or not reactants:
                continue
            canonical = tuple(constitutional_key(str(value)) for value in reactants)
            if all(value in current_terminal_keys for value in canonical):
                return {
                    "proposal_sha256": proposal["proposal_sha256"],
                    "proposal_rank": int(proposal["rank"]),
                    "reaction_id": str(trace["reaction_id"]),
                    "reactants": list(canonical),
                }
    return None


def _assess_component(
    *,
    role: str,
    target: str,
    proposals: Sequence[Mapping[str, Any]],
    current_terminal_keys: set[str],
    transforms: Mapping[str, Any],
) -> dict[str, Any]:
    coherent = any(_semantic_graph_consistent(row) for row in proposals)
    direct = _direct_current_trace(proposals, current_terminal_keys)
    program_projection = _safe_program(role, target)
    program_family: str | None = None
    receipts: list[dict[str, Any]] = []
    expected_proposal: Mapping[str, Any] | None = None
    expected_precursor: str | None = None
    leaves: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    full_program_forward_verified = False
    if program_projection is not None:
        program_family, steps = program_projection
        expected_precursor = (
            None if not steps else constitutional_key(str(steps[-1]["reactants"][0]))
        )
        if expected_precursor is not None:
            for proposal in sorted(proposals, key=lambda value: int(value["rank"])):
                if not _semantic_graph_consistent(proposal):
                    continue
                if _matching_forward_precursors(
                    proposal,
                    program=program_family,
                    expected_precursor=expected_precursor,
                ):
                    expected_proposal = proposal
                    break
        try:
            _, receipts, full_program_forward_verified = _execute_program(
                role=role,
                target=target,
                transforms=transforms,
            )
        except (Ugi3VirtualProgramError, Ugi3ProposalAdjudicationError):
            receipts = []
            full_program_forward_verified = False
        leaves = tuple(constitutional_key(value) for value in projected_leaf_candidates(steps))
        unresolved = tuple(sorted(value for value in leaves if value not in current_terminal_keys))

    program_closed = bool(
        expected_proposal is not None
        and full_program_forward_verified
        and leaves
        and not unresolved
    )
    terminal_closed = bool(direct is not None or program_closed)
    if direct is not None:
        disposition = "direct_registered_step_terminal_closed"
    elif program_closed:
        disposition = "proposal_matched_stepwise_program_terminal_closed"
    elif coherent and full_program_forward_verified and unresolved:
        disposition = "forward_verified_program_upstream_open"
    elif coherent and program_projection is None:
        disposition = "proposal_coherent_without_adjudicable_program"
    elif coherent:
        disposition = "proposal_coherent_not_terminal_closed"
    else:
        disposition = "no_graph_consistent_proposal"
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "role": role,
        "target_smiles": target,
        "proposal_count": len(proposals),
        "proposal_coherent": coherent,
        "direct_terminal_trace": direct,
        "program_annotation": program_family,
        "program_annotation_is_gate": False,
        "expected_final_precursor": expected_precursor,
        "expected_final_proposal_sha256": (
            None if expected_proposal is None else expected_proposal["proposal_sha256"]
        ),
        "step_receipts": receipts,
        "all_program_steps_forward_verified": full_program_forward_verified,
        "projected_leaves": list(leaves),
        "unresolved_projected_leaves": list(unresolved),
        "all_starting_materials_current": bool(leaves and not unresolved),
        "terminal_closed_route_hypothesis": terminal_closed,
        "route_value_is_success_probability": False,
        "proposal_model_score_used": False,
        "reaction_family_used_as_gate": False,
        "disposition": disposition,
    }


def _product_utility(
    components: Sequence[Mapping[str, Any]], qualified: set[tuple[str, str]]
) -> int:
    if len(components) != 3:
        raise ProposalAwareContrastError("product route utility requires three components")
    return int(
        all(
            str(component.get("graded_evidence_class")) in {EXACT, FAMILY_ALL}
            or (str(component.get("role")), str(component.get("canonical_smiles"))) in qualified
            for component in components
        )
    )


def build_proposal_aware_checkpoint_contrast(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Adjudicate proposal-derived routes and recompute frozen decision contrast."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ProposalAwareContrastError("proposal-aware contrast config schema changed")
    paths: dict[str, Path] = {}
    for label, record in config.get("inputs", {}).items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ProposalAwareContrastError(f"malformed input: {label}")
        path = repo / str(record["path"])
        if _sha256_file(path) != str(record["sha256"]):
            raise ProposalAwareContrastError(f"input hash changed: {label}")
        paths[str(label)] = path
    required = {
        "support_audits",
        "proposal_result",
        "proposal_ledger",
        "precursor_leaf_audit_config",
        "forward_registry",
        "variant_esterification",
        "variant_primary_alcohol_oxidation",
        "variant_amine_formylation",
        "variant_formamide_dehydration",
        "cross_platform_step_evidence",
        "superseded_exact_ledger_contrast",
    }
    if set(paths) != required:
        raise ProposalAwareContrastError("proposal-aware contrast input set changed")
    proposal_result = _read_json(paths["proposal_result"])
    if proposal_result.get("status") != "checkpoint_component_proposal_diagnostic_complete":
        raise ProposalAwareContrastError("checkpoint proposal diagnostic is not complete")

    current_terminal_keys, terminal_receipt = _load_current_terminal_keys(
        repo, paths["precursor_leaf_audit_config"]
    )
    transforms = _load_transforms(paths)
    proposal_rows = _read_jsonl_gzip(paths["proposal_ledger"])
    by_component: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in proposal_rows:
        by_component[(str(row["role"]), str(row["target_smiles"]))].append(row)
    expected_targets = int(proposal_result["summary"]["unique_unresolved_targets"])
    if len(by_component) != expected_targets:
        raise ProposalAwareContrastError("proposal ledger target census changed")

    component_rows = [
        _assess_component(
            role=role,
            target=target,
            proposals=proposals,
            current_terminal_keys=current_terminal_keys,
            transforms=transforms,
        )
        for (role, target), proposals in sorted(by_component.items())
    ]
    qualified = {
        (str(row["role"]), str(row["target_smiles"]))
        for row in component_rows
        if bool(row["terminal_closed_route_hypothesis"])
    }

    support = _read_json(paths["support_audits"])
    population = config["population"]
    checkpoints = {int(value) for value in population["checkpoints"]}
    checkpoint_rows = [
        row
        for row in support.get("records", [])
        if row.get("arm") == population["arm"]
        and row.get("assessment_phase") == population["checkpoint_phase"]
        and int(row.get("checkpoint")) in checkpoints
    ]
    final_rows = [
        row
        for row in support.get("records", [])
        if row.get("arm") == population["arm"]
        and row.get("assessment_phase") == population["terminal_phase"]
    ]
    if not checkpoint_rows or not final_rows:
        raise ProposalAwareContrastError("proposal-aware contrast population is empty")

    transitions: Counter[str] = Counter()
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    checkpoint_ready: Counter[int] = Counter()
    checkpoint_total: Counter[int] = Counter()
    for row in checkpoint_rows:
        receipt = row.get("graded_route_readiness")
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("components"), list):
            raise ProposalAwareContrastError("graded checkpoint receipt is malformed")
        old = int(receipt.get("binary_controller_utility"))
        new = _product_utility(receipt["components"], qualified)
        transitions[f"{old}->{new}"] += 1
        checkpoint = int(row["checkpoint"])
        groups[(checkpoint, int(row["program_index"]))].append(new)
        checkpoint_ready[checkpoint] += new
        checkpoint_total[checkpoint] += 1

    mixed_by_checkpoint: Counter[int] = Counter()
    observed_groups = 0
    mixed_groups = 0
    for (checkpoint, _), values in groups.items():
        if len(values) < 2:
            continue
        observed_groups += 1
        if len(set(values)) > 1:
            mixed_groups += 1
            mixed_by_checkpoint[checkpoint] += 1

    final_products: set[str] = set()
    for row in final_rows:
        receipt = row.get("graded_route_readiness")
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("components"), list):
            raise ProposalAwareContrastError("graded final receipt is malformed")
        if _product_utility(receipt["components"], qualified):
            audit = row.get("support_audit")
            product = (
                audit.get("support", {}).get("product_smiles")
                if isinstance(audit, Mapping)
                else None
            )
            final_products.add(str(product or row["terminal_sha256"]))

    thresholds = config["retry_authorization_gates"]
    gate_results = {
        "qualified_component_count": len(qualified)
        >= int(thresholds["minimum_qualified_components"]),
        "mixed_checkpoint_group_count": mixed_groups
        >= int(thresholds["minimum_mixed_checkpoint_groups"]),
        "mixed_checkpoint_group_fraction": bool(
            observed_groups
            and mixed_groups / observed_groups
            >= float(thresholds["minimum_mixed_checkpoint_group_fraction"])
        ),
        "mixed_group_at_every_checkpoint": all(
            mixed_by_checkpoint[checkpoint] > 0 for checkpoint in checkpoints
        ),
        "unique_route_ready_productive_finals": len(final_products)
        >= int(thresholds["minimum_unique_route_ready_productive_finals"]),
    }
    authorized = all(gate_results.values())
    dispositions = Counter(str(row["disposition"]) for row in component_rows)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "proposal_aware_checkpoint_contrast_complete",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": _sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "terminal_evidence": terminal_receipt,
        "summary": {
            "proposal_evaluated_components": len(component_rows),
            "proposal_coherent_components": sum(
                bool(row["proposal_coherent"]) for row in component_rows
            ),
            "new_terminal_closed_components": len(qualified),
            "component_dispositions": dict(sorted(dispositions.items())),
            "checkpoint_assessments": len(checkpoint_rows),
            "observed_checkpoint_groups_with_at_least_two_assessments": observed_groups,
            "mixed_checkpoint_groups": mixed_groups,
            "mixed_checkpoint_group_fraction": (
                None if not observed_groups else mixed_groups / observed_groups
            ),
            "mixed_groups_by_checkpoint": {
                str(checkpoint): mixed_by_checkpoint[checkpoint]
                for checkpoint in sorted(checkpoints)
            },
            "route_ready_by_checkpoint": {
                str(checkpoint): {
                    "ready": checkpoint_ready[checkpoint],
                    "assessed": checkpoint_total[checkpoint],
                }
                for checkpoint in sorted(checkpoints)
            },
            "old_to_proposal_aware_utility": dict(sorted(transitions.items())),
            "unique_route_ready_productive_finals": len(final_products),
        },
        "retry_authorization": {
            "gate_results": gate_results,
            "all_gates_pass": authorized,
            "one_frozen_matched_synthesis_challenger_authorized": authorized,
            "lambda_tuning_authorized": False,
            "synthesis_guidance_promoted": False,
        },
        "decision": {
            "next_step": (
                "run one frozen matched proposal-aware challenger"
                if authorized
                else "retain complete generation plus proposal-first post-generation routing"
            )
        },
        "scientific_authority": {
            "proposal_model_scores_used": False,
            "reaction_family_used_as_proposal_filter": False,
            "reaction_family_used_as_route_gate": False,
            "proposal_alone_creates_evidence": False,
            "route_hypothesis_is_experimental_success": False,
            "route_value_is_success_probability": False,
            "biological_scores_used": False,
            "read_only_contrast_after_proposal_execution": True,
            "synthesis_guidance_promoted": False,
        },
        "artifacts": {},
    }
    ledger = _gzip_jsonl(component_rows)
    result["artifacts"]["component_ledger.jsonl.gz"] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "rows": len(component_rows),
        "sha256": hashlib.sha256(ledger).hexdigest(),
    }
    result["result_sha256"] = _content_sha256(result)
    return result, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "ProposalAwareContrastError",
    "RESULT_SCHEMA_VERSION",
    "_direct_current_trace",
    "_product_utility",
    "build_proposal_aware_checkpoint_contrast",
]
