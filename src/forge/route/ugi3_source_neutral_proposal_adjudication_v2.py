"""Proposal-first adjudication of Graph2Edits Ugi component routes.

Version 1 incorrectly required a legacy projected-family label before a
forward-consistent proposal could be assigned to that family.  That made a
missing knowledge-base annotation a circular rejection gate.  This module
instead derives candidate programs from the target role and independently
executed bond changes, then uses reaction families only to retrieve source
scope and upstream-program evidence.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.route.ugi3_agile_template_saturation_stress import (
    projected_leaf_candidates,
)
from forge.route.ugi3_precursor_leaf_closure import (
    ALDEHYDE_ROLE,
    HEAD_ROLE,
    ISOCYANIDE_ROLE,
    _parse_utc,
    constitutional_key,
    current_terminal_evidence,
)
from forge.route.ugi3_precursor_leaf_closure import (
    _validated_inputs as _validated_leaf_inputs,
)
from forge.route.ugi3_source_neutral_proposal_adjudication import (
    FATTY_ALDEHYDE,
    ISOCYANIDE,
    QUALIFIED_FAMILY,
    SUPPORTED_PROGRAMS,
    Ugi3ProposalAdjudicationError,
    _canonical,
    _content_sha256,
    _gzip_jsonl,
    _read_json,
    _read_jsonl_gzip,
    _sha256_file,
    assess_pair_scope,
    build_scope_profiles,
    extract_source_pairs,
)
from forge.route.ugi3_virtual_programs import _aldehyde_program, _isocyanide_program

CONFIG_SCHEMA_VERSION = "phase1_ugi3_source_neutral_proposal_adjudication_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi3_source_neutral_proposal_adjudication.v2"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_source_neutral_proposal_adjudication_ledger.v2"


def _program_from_target(*, role: str, target: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Infer a mechanical program without consulting a legacy family label."""

    if role == ALDEHYDE_ROLE:
        return _aldehyde_program(target)
    if role == ISOCYANIDE_ROLE:
        return _isocyanide_program(target)
    if role == HEAD_ROLE:
        return None
    raise Ugi3ProposalAdjudicationError(f"unsupported component role: {role!r}")


def _matching_forward_precursors(
    raw_row: Mapping[str, Any],
    *,
    program: str,
    expected_precursor: str,
) -> tuple[str, ...]:
    """Return exact target-reconstructing precursors for the expected edit."""

    resolution = raw_row.get("raw_discovery_resolution")
    if not isinstance(resolution, Mapping):
        return ()
    if program in {FATTY_ALDEHYDE, "primary_alcohol_oxidation"}:
        expected_fragment = "primary_alcohol_oxidation"
    elif program == ISOCYANIDE:
        expected_fragment = "formamide_dehydration"
    else:
        return ()
    matched: set[str] = set()
    for trace in resolution.get("traces", []):
        if not isinstance(trace, Mapping):
            continue
        reactants = trace.get("role_ordered_reactants")
        if (
            bool(trace.get("target_reconstructed"))
            and expected_fragment in str(trace.get("reaction_id"))
            and isinstance(reactants, list)
            and len(reactants) == 1
        ):
            precursor = _canonical(str(reactants[0]))
            if precursor == expected_precursor:
                matched.add(precursor)
    return tuple(sorted(matched))


def _semantic_by_sha(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = str(row.get("proposal_sha256"))
        if identity in output:
            raise Ugi3ProposalAdjudicationError("semantic proposal identities collide")
        output[identity] = row
    return output


def _load_current_terminal_keys(
    repo: Path, leaf_config_path: Path
) -> tuple[set[str], dict[str, Any]]:
    leaf_config = _read_json(leaf_config_path)
    leaf_paths = _validated_leaf_inputs(leaf_config, repo)
    assessment_raw = leaf_config.get("assessment_as_of_utc")
    expiry_days = leaf_config.get("procurement_evidence_default_expiry_days")
    if not isinstance(assessment_raw, str) or not isinstance(expiry_days, int):
        raise Ugi3ProposalAdjudicationError("leaf-audit time policy is malformed")
    assessment = _parse_utc(assessment_raw, label="leaf-audit assessment time")
    terminals = current_terminal_evidence(
        leaf_paths,
        assessment_as_of=assessment,
        default_expiry_days=expiry_days,
        repo=repo,
    )
    receipt = {
        "config_path": str(leaf_config_path.relative_to(repo)),
        "config_sha256": _sha256_file(leaf_config_path),
        "assessment_as_of_utc": assessment_raw,
        "current_terminal_count": len(terminals),
        "nested_inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "sha256": _sha256_file(path),
            }
            for label, path in sorted(leaf_paths.items())
        },
    }
    return set(terminals), receipt


def build_source_neutral_adjudication_v2(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build the proposal-first source-neutral adjudication."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3ProposalAdjudicationError("unsupported v2 adjudication config")
    paths: dict[str, Path] = {}
    for label, record in config.get("inputs", {}).items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise Ugi3ProposalAdjudicationError(f"malformed input: {label}")
        path = repo / str(record["path"])
        if _sha256_file(path) != str(record["sha256"]):
            raise Ugi3ProposalAdjudicationError(f"input hash changed: {label}")
        paths[str(label)] = path
    required = {
        "raw_proposal_ledger",
        "semantic_proposal_ledger",
        "route_assessment_ledger",
        "exact_source_routes",
        "precursor_leaf_audit_config",
        "invalidated_v1_result",
        "invalidated_v1_ledger",
    }
    if set(paths) != required:
        raise Ugi3ProposalAdjudicationError("v2 adjudication input set changed")

    source_pairs = extract_source_pairs(_read_json(paths["exact_source_routes"]))
    profiles = build_scope_profiles(
        source_pairs,
        minimum_similarity_floor=float(config["scope_policy"]["minimum_pair_similarity_floor"]),
    )
    current_terminal_keys, terminal_receipt = _load_current_terminal_keys(
        repo, paths["precursor_leaf_audit_config"]
    )
    raw_rows = _read_jsonl_gzip(paths["raw_proposal_ledger"])
    semantic = _semantic_by_sha(_read_jsonl_gzip(paths["semantic_proposal_ledger"]))
    raw_by_component: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        raw_by_component[(str(row["role"]), str(row["target_smiles"]))].append(row)

    adjudications: list[dict[str, Any]] = []
    qualified_components: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key, proposals in sorted(raw_by_component.items()):
        role, target = key
        coherent = [
            proposal
            for proposal in sorted(proposals, key=lambda row: int(row["rank"]))
            if bool(
                semantic.get(str(proposal["proposal_sha256"]), {}).get(
                    "graph_consistent_discovery_hypothesis"
                )
            )
        ]
        program_projection = _program_from_target(role=role, target=target)
        program = None if program_projection is None else program_projection[0]
        steps = [] if program_projection is None else program_projection[1]
        expected_precursor = None if not steps else _canonical(str(steps[-1]["reactants"][0]))
        selected: Mapping[str, Any] | None = None
        matched_precursor: str | None = None
        if program in SUPPORTED_PROGRAMS and expected_precursor is not None:
            for proposal in coherent:
                matched = _matching_forward_precursors(
                    proposal,
                    program=program,
                    expected_precursor=expected_precursor,
                )
                if matched:
                    selected = proposal
                    matched_precursor = matched[0]
                    break

        proposal_coherent = bool(coherent)
        expected_forward = selected is not None
        if expected_forward and program is not None and matched_precursor is not None:
            scope = assess_pair_scope(
                program_family=program,
                precursor=matched_precursor,
                target=target,
                profile=profiles[program],
            )
        else:
            scope = {
                "qualified": False,
                "exact_source_pair": False,
                "target_feature_envelope": False,
                "precursor_feature_envelope": False,
                "nearest_source_pair_similarity": None,
                "minimum_pair_similarity": None,
                "source_pair_count": 0,
                "scope_is_success_probability": False,
            }
        evidence_supported = bool(expected_forward and scope["qualified"])
        leaves = tuple(constitutional_key(value) for value in projected_leaf_candidates(steps))
        unresolved = tuple(sorted(value for value in leaves if value not in current_terminal_keys))
        route_closed = bool(evidence_supported and leaves and not unresolved)
        if route_closed:
            disposition = "fully_route_closed"
            qualified_components.add(key)
        elif evidence_supported:
            disposition = "evidence_supported_upstream_open"
        elif expected_forward:
            disposition = "forward_consistent_outside_source_scope"
        elif proposal_coherent and program is None:
            disposition = "proposal_coherent_no_evidenced_family"
        elif proposal_coherent:
            disposition = "proposal_coherent_wrong_expected_final_precursor"
        else:
            disposition = "no_graph_consistent_proposal"
        counts[disposition] += 1
        role_counts[role][disposition] += 1
        adjudications.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "role": role,
                "target_smiles": target,
                "program_family": program,
                "family_role": "evidence_retrieval_annotation_not_proposal_gate",
                "proposal_sha256": (None if selected is None else selected["proposal_sha256"]),
                "proposal_rank": None if selected is None else int(selected["rank"]),
                "precursor_smiles": matched_precursor,
                "projected_leaves": list(leaves),
                "unresolved_projected_leaves": list(unresolved),
                "checks": {
                    "any_graph_consistent_proposal": proposal_coherent,
                    "independent_expected_step_forward_consistent": expected_forward,
                    "source_bounded_pair_scope": bool(scope["qualified"]),
                    "source_series_evidence_supported": evidence_supported,
                    "all_projected_terminal_leaves_current": route_closed,
                },
                "scope": scope,
                "adjudicated_route_state": (QUALIFIED_FAMILY if route_closed else None),
                "adjudicated_route_value": 0.75 if route_closed else None,
                "route_value_is_success_probability": False,
                "disposition": disposition,
                "proposal_model_score_used": False,
                "proposal_family_label_used": False,
                "legacy_program_family_used": False,
                "proposal_source_created_evidence": False,
                "route_closure_authorized": route_closed,
            }
        )

    route_rows = _read_jsonl_gzip(paths["route_assessment_ledger"])
    arm_exact: dict[str, set[str]] = defaultdict(set)
    arm_qualified: dict[str, set[str]] = defaultdict(set)
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
            strict[str(role)] or (str(role), str(smiles)) in qualified_components
            for role, smiles in components.items()
        ):
            arm_qualified[arm].add(product)
    arm_summary = {
        arm: {
            "unique_products_strict_exact_complete": len(arm_exact[arm]),
            "unique_products_source_bounded_family_or_better": len(
                arm_exact[arm] | arm_qualified[arm]
            ),
            "incremental_unique_products": len(arm_qualified[arm] - arm_exact[arm]),
        }
        for arm in sorted(set(arm_exact) | set(arm_qualified))
    }

    proposal_coherent_count = sum(
        bool(row["checks"]["any_graph_consistent_proposal"]) for row in adjudications
    )
    evidence_supported_count = sum(
        bool(row["checks"]["source_series_evidence_supported"]) for row in adjudications
    )
    route_closed_count = len(qualified_components)
    content: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "proposal_first_source_neutral_adjudication_complete",
        "supersedes": {
            "result": str(paths["invalidated_v1_result"].relative_to(repo)),
            "ledger": str(paths["invalidated_v1_ledger"].relative_to(repo)),
            "reason": (
                "v1 circularly required a legacy program-family annotation before "
                "allowing a forward-consistent proposal to infer that family"
            ),
            "v1_result_must_not_support_scientific_or_production_decisions": True,
        },
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
            "proposal_targets": len(adjudications),
            "proposal_coherent_targets": proposal_coherent_count,
            "proposal_coherent_fraction": proposal_coherent_count / len(adjudications),
            "evidence_supported_targets": evidence_supported_count,
            "evidence_supported_fraction": evidence_supported_count / len(adjudications),
            "fully_route_closed_targets": route_closed_count,
            "fully_route_closed_fraction": route_closed_count / len(adjudications),
            "component_dispositions": dict(sorted(counts.items())),
            "component_dispositions_by_role": {
                role: dict(sorted(values.items())) for role, values in sorted(role_counts.items())
            },
            "arms": arm_summary,
            "balanced_unique_products_source_bounded_family_or_better": min(
                (
                    row["unique_products_source_bounded_family_or_better"]
                    for row in arm_summary.values()
                ),
                default=0,
            ),
        },
        "decision": {
            "proposal_engine_qualified_as_route_evidence": False,
            "reaction_family_is_a_proposal_prerequisite": False,
            "reaction_family_is_an_evidence_retrieval_annotation": True,
            "production_evaluator_changed": False,
            "synthesis_tilting_promoted": False,
            "next_gate": (
                "Measure route-value contrast only from fully route-closed targets; "
                "retain proposal-coherent and evidence-supported states as separate diagnostics."
            ),
        },
        "artifacts": {},
    }
    ledger = _gzip_jsonl(adjudications)
    content["artifacts"]["adjudication_ledger.jsonl.gz"] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "rows": len(adjudications),
        "sha256": hashlib.sha256(ledger).hexdigest(),
    }
    content["result_sha256"] = _content_sha256(content)
    return content, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "build_source_neutral_adjudication_v2",
]
