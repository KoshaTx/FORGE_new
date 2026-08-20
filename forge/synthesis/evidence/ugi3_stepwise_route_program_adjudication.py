"""Adjudicate Ugi precursor routes as stepwise chemical programs.

This audit keeps whole-program source envelopes as a high-confidence evidence
tier but prevents them from defining chemical possibility.  Every proposed
program is re-executed step by step with independently pinned transforms.  A
route whose exact starting materials are current is a terminal-closed route
hypothesis even when its spacer or chain length lies outside the exact AGILE
series; it is not promoted to an experimentally proven route or success
probability.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.synthesis.engine.qualified_forward import (
    QualifiedForwardReaction,
    load_qualified_forward_reaction,
    unique_forward_products,
)
from forge.synthesis.sources.ugi3_source_neutral_proposal_adjudication import (
    Ugi3ProposalAdjudicationError,
    _content_sha256,
    _gzip_jsonl,
    _read_json,
    _read_jsonl_gzip,
    _sha256_file,
)
from forge.synthesis.terminals.ugi3_precursor_leaf_closure import (
    ALDEHYDE_ROLE,
    ISOCYANIDE_ROLE,
    constitutional_key,
)
from forge.synthesis.terminals.ugi3_virtual_programs import _aldehyde_program, _isocyanide_program

CONFIG_SCHEMA_VERSION = "phase1_ugi3_stepwise_route_program_adjudication_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_stepwise_route_program_adjudication.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_stepwise_route_program_adjudication_ledger.v1"

TRANSFORM_BINDINGS = {
    "esterification": (
        "ugi3_upstream_esterification_exact_source_v1",
        "variant_esterification",
    ),
    "alcohol_to_aldehyde_oxidation": (
        "ugi3_upstream_primary_alcohol_oxidation_exact_source_v1",
        "variant_primary_alcohol_oxidation",
    ),
    "amine_formylation": (
        "ugi3_upstream_amine_formylation_exact_source_v1",
        "variant_amine_formylation",
    ),
    "formamide_dehydration_to_isocyanide": (
        "ugi3_upstream_formamide_dehydration_exact_source_v1",
        "variant_formamide_dehydration",
    ),
}


def _load_transforms(paths: Mapping[str, Path]) -> dict[str, QualifiedForwardReaction]:
    registry = paths["forward_registry"]
    return {
        transformation: load_qualified_forward_reaction(
            registry,
            paths[variant_label],
            reaction_id=reaction_id,
        )
        for transformation, (reaction_id, variant_label) in TRANSFORM_BINDINGS.items()
    }


def _program(role: str, target: str) -> tuple[str, list[dict[str, Any]]]:
    if role == ALDEHYDE_ROLE:
        return _aldehyde_program(target)
    if role == ISOCYANIDE_ROLE:
        return _isocyanide_program(target)
    raise Ugi3ProposalAdjudicationError(f"no stepwise program for role {role!r}")


def _execute_program(
    *,
    role: str,
    target: str,
    transforms: Mapping[str, QualifiedForwardReaction],
) -> tuple[str, list[dict[str, Any]], bool]:
    family, steps = _program(role, target)
    receipts: list[dict[str, Any]] = []
    for step in steps:
        transformation = str(step["transformation"])
        compiled = transforms[transformation]
        products = unique_forward_products(
            compiled,
            [str(value) for value in step["reactants"]],
            max_products=100,
            isomeric_smiles=False,
        )
        expected = constitutional_key(str(step["product"]))
        observed = tuple(constitutional_key(value) for value in products)
        receipts.append(
            {
                "step_index": int(step["step_index"]),
                "transformation": transformation,
                "reaction_id": compiled.reaction_id,
                "reactants": list(step["reactants"]),
                "expected_product": expected,
                "forward_products": list(observed),
                "expected_product_reconstructed": expected in observed,
            }
        )
    return family, receipts, all(bool(row["expected_product_reconstructed"]) for row in receipts)


def build_stepwise_route_program_adjudication(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build a deterministic, step-level gate-sensitivity audit."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3ProposalAdjudicationError("unsupported stepwise audit config")
    paths: dict[str, Path] = {}
    for label, record in config.get("inputs", {}).items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise Ugi3ProposalAdjudicationError(f"malformed input: {label}")
        path = repo / str(record["path"])
        if _sha256_file(path) != str(record["sha256"]):
            raise Ugi3ProposalAdjudicationError(f"input hash changed: {label}")
        paths[str(label)] = path
    required = {
        "proposal_first_result",
        "proposal_first_ledger",
        "route_assessment_ledger",
        "forward_registry",
        "variant_esterification",
        "variant_primary_alcohol_oxidation",
        "variant_amine_formylation",
        "variant_formamide_dehydration",
        "cross_platform_step_evidence",
    }
    if set(paths) != required:
        raise Ugi3ProposalAdjudicationError("stepwise audit input set changed")
    proposal_result = _read_json(paths["proposal_first_result"])
    if proposal_result.get("status") != "proposal_first_source_neutral_adjudication_complete":
        raise Ugi3ProposalAdjudicationError("proposal-first audit is not qualified")
    source_rows = _read_jsonl_gzip(paths["proposal_first_ledger"])
    transforms = _load_transforms(paths)

    output_rows: list[dict[str, Any]] = []
    terminal_closed_components: set[tuple[str, str]] = set()
    dispositions: Counter[str] = Counter()
    evidence_tiers: Counter[str] = Counter()
    step_counts: Counter[str] = Counter()
    for row in source_rows:
        role = str(row["role"])
        target = str(row["target_smiles"])
        coherent = bool(row["checks"]["any_graph_consistent_proposal"])
        family: str | None = row.get("program_family")
        receipts: list[dict[str, Any]] = []
        forward_verified = False
        if coherent and role in {ALDEHYDE_ROLE, ISOCYANIDE_ROLE}:
            family, receipts, forward_verified = _execute_program(
                role=role,
                target=target,
                transforms=transforms,
            )
        for receipt in receipts:
            step_counts[str(receipt["transformation"])] += 1
        all_current = bool(
            row.get("projected_leaves") and not row.get("unresolved_projected_leaves")
        )
        whole_program_source_bounded = bool(row["checks"]["source_series_evidence_supported"])
        if whole_program_source_bounded:
            evidence_tier = "whole_program_source_bounded"
        elif forward_verified:
            evidence_tier = "step_precedent_local_forward_verified"
        else:
            evidence_tier = "proposal_only_or_unmapped"
        evidence_tiers[evidence_tier] += 1
        terminal_closed = bool(forward_verified and all_current)
        if terminal_closed:
            terminal_closed_components.add((role, target))
        if terminal_closed and whole_program_source_bounded:
            disposition = "source_bounded_terminal_closed"
        elif terminal_closed:
            disposition = "stepwise_precedent_terminal_closed"
        elif forward_verified and whole_program_source_bounded:
            disposition = "source_bounded_upstream_open"
        elif forward_verified:
            disposition = "stepwise_precedent_upstream_open"
        else:
            disposition = "no_forward_verified_program"
        dispositions[disposition] += 1
        output_rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "role": role,
                "target_smiles": target,
                "program_annotation": family,
                "program_annotation_is_gate": False,
                "proposal_coherent": coherent,
                "step_receipts": receipts,
                "all_steps_forward_verified": forward_verified,
                "whole_program_source_bounded": whole_program_source_bounded,
                "step_evidence_tier": evidence_tier,
                "projected_leaves": list(row.get("projected_leaves", [])),
                "unresolved_projected_leaves": list(row.get("unresolved_projected_leaves", [])),
                "all_starting_materials_current": all_current,
                "terminal_closed_route_hypothesis": terminal_closed,
                "exact_substrate_experimental_route_proven": bool(
                    row["scope"].get("exact_source_pair")
                ),
                "synthesis_success_probability": None,
                "disposition": disposition,
            }
        )

    route_rows = _read_jsonl_gzip(paths["route_assessment_ledger"])
    arm_exact: dict[str, set[str]] = defaultdict(set)
    arm_stepwise: dict[str, set[str]] = defaultdict(set)
    for row in route_rows:
        components = row.get("canonical_components")
        if not isinstance(components, Mapping) or len(components) != 3:
            raise Ugi3ProposalAdjudicationError("route row component contract changed")
        strict = {
            str(item["role"]): bool(item["strict_complete"])
            for item in row.get("potential", {}).get("roles", [])
        }
        if set(strict) != set(components):
            raise Ugi3ProposalAdjudicationError("strict route role assessment changed")
        product = str(row["canonical_product"])
        arm = str(row["arm_id"])
        if all(strict.values()):
            arm_exact[arm].add(product)
        if all(
            strict[str(role)] or (str(role), str(smiles)) in terminal_closed_components
            for role, smiles in components.items()
        ):
            arm_stepwise[arm].add(product)
    arm_summary = {
        arm: {
            "strict_exact_complete": len(arm_exact[arm]),
            "stepwise_terminal_closed_or_better": len(arm_exact[arm] | arm_stepwise[arm]),
            "incremental_stepwise_terminal_closed": len(arm_stepwise[arm] - arm_exact[arm]),
        }
        for arm in sorted(set(arm_exact) | set(arm_stepwise))
    }

    content: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "stepwise_route_program_adjudication_complete",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": _sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "summary": {
            "targets": len(output_rows),
            "proposal_coherent": sum(row["proposal_coherent"] for row in output_rows),
            "all_steps_forward_verified": sum(
                row["all_steps_forward_verified"] for row in output_rows
            ),
            "terminal_closed_route_hypotheses": len(terminal_closed_components),
            "exact_substrate_experimental_routes": sum(
                row["exact_substrate_experimental_route_proven"] for row in output_rows
            ),
            "evidence_tiers": dict(sorted(evidence_tiers.items())),
            "dispositions": dict(sorted(dispositions.items())),
            "forward_verified_steps": dict(sorted(step_counts.items())),
            "arms": arm_summary,
        },
        "decision": {
            "family_annotation_constrains_chemical_support": False,
            "whole_program_source_envelope_retained_as_confidence_tier": True,
            "stepwise_terminal_closed_route_hypothesis_is_experimental_success": False,
            "proposal_engine_creates_evidence": False,
            "production_evaluator_changed": False,
            "synthesis_tilting_promoted": False,
            "next_gate": (
                "Refresh terminal-material coverage, then test whether graded stepwise "
                "route values create nonuniform contrast on frozen guidance particles."
            ),
        },
        "artifacts": {},
    }
    ledger = _gzip_jsonl(output_rows)
    content["artifacts"]["stepwise_route_ledger.jsonl.gz"] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "rows": len(output_rows),
        "sha256": hashlib.sha256(ledger).hexdigest(),
    }
    content["result_sha256"] = _content_sha256(content)
    return content, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "build_stepwise_route_program_adjudication",
]
