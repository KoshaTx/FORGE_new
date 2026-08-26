"""Freeze and assess one method-blind component-evidence union across Ugi methods.

The public worklist contains chemistry and opaque component identifiers, but no method membership.
Membership is held in a separate private ledger until evidence adjudication closes.  This prevents
the planner/evidence pass from receiving method identity while ensuring that every method is scored
against exactly the same component index and search disposition.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file, sha256_json
from forge.core.io import iter_jsonl, read_json_object, write_json, write_jsonl
from forge.synthesis.assessment.common_route_evidence import (
    assess_common_route_evidence,
    load_frozen_component_evidence,
)

WORKLIST_SCHEMA = "forge.common_ugi_method_blind_route_worklist.v1"
MEMBERSHIP_SCHEMA = "forge.common_ugi_private_route_membership.v1"
FREEZE_SCHEMA = "forge.common_ugi_method_blind_route_union.v1"
ADJUDICATION_SCHEMA = "forge.common_ugi_method_blind_route_adjudication.v1"
EVIDENCE_CONFIG_SCHEMA = "forge.common_ugi_method_blind_route_evidence_config.v1"
EVIDENCE_RESULT_SCHEMA = "forge.common_ugi_method_blind_route_evidence.v1"


class MethodBlindRouteUnionError(ValueError):
    """The shared route population or evidence ledger is incomplete or contaminated."""


def _load_assessed(path: Path) -> list[dict[str, Any]]:
    records = list(iter_jsonl(path))
    if (
        not records
        or records[0]
        != {"schema_version": "forge.common_ugi_assessed_attempts.v1", "rows": len(records) - 1}
        or not all(isinstance(row, Mapping) for row in records[1:])
    ):
        raise MethodBlindRouteUnionError(f"common assessed-attempt ledger changed: {path}")
    return [dict(row) for row in records[1:]]


def _ledger_identity(rows: Sequence[Mapping[str, Any]], path: Path) -> tuple[str, int]:
    identities = {(row.get("method_id"), row.get("seed")) for row in rows}
    if len(identities) != 1:
        raise MethodBlindRouteUnionError(f"assessed ledger mixes method or seed identities: {path}")
    method, seed = next(iter(identities))
    if (
        not isinstance(method, str)
        or not method
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise MethodBlindRouteUnionError(f"assessed ledger identity is malformed: {path}")
    return method, seed


def _exact_l1_components(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    if row.get("exact_l1_program") is not True or int(row.get("exact_l1_trace_count", 0)) != 1:
        return []
    traces = row.get("exact_l1_traces")
    if not isinstance(traces, list) or len(traces) != 1 or not isinstance(traces[0], Mapping):
        raise MethodBlindRouteUnionError("unique exact-L1 trace payload is malformed")
    by_role = traces[0].get("components_by_role")
    if not isinstance(by_role, Mapping) or not by_role:
        raise MethodBlindRouteUnionError("unique exact-L1 component mapping is malformed")
    components = []
    for role, value in sorted(by_role.items()):
        values = value if isinstance(value, list) else [value]
        if not isinstance(role, str) or not role or not values:
            raise MethodBlindRouteUnionError("unique exact-L1 component role is malformed")
        for smiles in values:
            if not isinstance(smiles, str) or not smiles:
                raise MethodBlindRouteUnionError("unique exact-L1 component SMILES is malformed")
            components.append((role, smiles))
    return components


def _component_id(role: str, smiles: str) -> str:
    return f"cu-{str(sha256_json({'role': role, 'canonical_smiles': smiles}))[:24]}"


def freeze_method_blind_route_union(
    assessed_paths: Sequence[Path], output_dir: Path
) -> dict[str, Any]:
    """Freeze the exact component union and split blind/public from private membership data."""

    if not assessed_paths:
        raise MethodBlindRouteUnionError("route-union freeze requires assessed ledgers")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MethodBlindRouteUnionError(f"route-union output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    membership: dict[tuple[str, str], dict[tuple[str, int], list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sources = []
    seen_identities = set()
    for path in assessed_paths:
        rows = _load_assessed(path)
        method, seed = _ledger_identity(rows, path)
        identity = (method, seed)
        if identity in seen_identities:
            raise MethodBlindRouteUnionError(f"duplicate method/seed assessed ledger: {identity}")
        seen_identities.add(identity)
        sources.append(
            {
                "method_id": method,
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": str(sha256_file(path)),
                "bytes": path.stat().st_size,
            }
        )
        for row in rows:
            index = row.get("attempt_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise MethodBlindRouteUnionError("assessed attempt index is malformed")
            for key in _exact_l1_components(row):
                membership[key][identity].append(index)
    if not membership:
        raise MethodBlindRouteUnionError(
            "route-union population contains no unique exact-L1 components"
        )
    worklist_rows = [
        {"component_id": _component_id(role, smiles), "role": role, "canonical_smiles": smiles}
        for role, smiles in sorted(membership)
    ]
    component_ids = [row["component_id"] for row in worklist_rows]
    if len(component_ids) != len(set(component_ids)):
        raise MethodBlindRouteUnionError("opaque component identifier collision")
    membership_rows = []
    for (role, smiles), identities in sorted(membership.items()):
        for (method, seed), attempt_indices in sorted(identities.items()):
            membership_rows.append(
                {
                    "component_id": _component_id(role, smiles),
                    "method_id": method,
                    "seed": seed,
                    "occurrences": len(attempt_indices),
                    "attempt_indices": sorted(attempt_indices),
                }
            )
    worklist_path = output_dir / "blind_worklist.jsonl.gz"
    private_path = output_dir / "private_membership.jsonl.gz"
    write_jsonl(
        worklist_path,
        [{"schema_version": WORKLIST_SCHEMA, "rows": len(worklist_rows)}, *worklist_rows],
    )
    write_jsonl(
        private_path,
        [{"schema_version": MEMBERSHIP_SCHEMA, "rows": len(membership_rows)}, *membership_rows],
    )
    result = {
        "schema_version": FREEZE_SCHEMA,
        "status": "frozen",
        "components": len(worklist_rows),
        "method_seed_ledgers": len(sources),
        "blind_worklist": artifact_record(worklist_path),
        "private_membership": artifact_record(private_path),
        "assessed_sources": sorted(sources, key=lambda row: (row["method_id"], row["seed"])),
        "blindness_gates": {
            "worklist_contains_method_identity": False,
            "membership_stored_separately": True,
            "population_is_exact_union_not_method_sample": True,
            "component_deduplication_key": "role_and_canonical_constitution",
        },
        "candidate_selection": False,
    }
    write_json(output_dir / "result.json", result)
    return result


def _load_worklist(union_dir: Path, union: Mapping[str, Any]) -> list[dict[str, str]]:
    path = union_dir / "blind_worklist.jsonl.gz"
    record = union.get("blind_worklist")
    if (
        not path.is_file()
        or not isinstance(record, Mapping)
        or record.get("sha256") != str(sha256_file(path))
        or record.get("bytes") != path.stat().st_size
    ):
        raise MethodBlindRouteUnionError("frozen blind worklist changed")
    records = list(iter_jsonl(path))
    if not records or records[0] != {"schema_version": WORKLIST_SCHEMA, "rows": len(records) - 1}:
        raise MethodBlindRouteUnionError("frozen blind worklist header changed")
    rows = []
    for raw in records[1:]:
        if not isinstance(raw, Mapping) or set(raw) != {
            "component_id",
            "role",
            "canonical_smiles",
        }:
            raise MethodBlindRouteUnionError("frozen blind worklist row changed")
        rows.append({key: str(raw[key]) for key in raw})
    return rows


def _load_source_component_dossiers(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise MethodBlindRouteUnionError("source component dossier ledger is malformed") from error
    required = {
        "component_id",
        "role",
        "canonical_smiles",
        "program_status",
        "l2_forward_status",
        "l3_terminal_status",
        "exact_source_computational_component",
        "family_projected_only",
    }
    if not rows or not required.issubset(rows[0]):
        raise MethodBlindRouteUnionError("source component dossier schema changed")
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["role"], row["canonical_smiles"])
        if not all(key) or key in indexed:
            raise MethodBlindRouteUnionError("source component dossier identity is invalid")
        indexed[key] = row
    return indexed


def build_method_blind_route_evidence(
    config_path: Path, repo: Path, output_dir: Path
) -> dict[str, Any]:
    """Disposition the public union using only a pre-existing exact-source evidence ledger.

    The builder never opens the private membership ledger. Components without exact-source L2 and
    frozen L3 support receive an explicit missing-knowledge abstention; family projections and
    proposal-only routes cannot close a dossier.
    """

    config = read_json_object(
        config_path,
        error=MethodBlindRouteUnionError,
        label="method-blind route evidence config",
    )
    expected_config_fields = {
        "schema_version",
        "status",
        "inputs",
        "policy",
        "candidate_selection",
    }
    if (
        config.get("schema_version") != EVIDENCE_CONFIG_SCHEMA
        or set(config) != expected_config_fields
    ):
        raise MethodBlindRouteUnionError("method-blind route evidence config changed")
    if config.get("status") != "frozen_after_union_before_evidence_lookup":
        raise MethodBlindRouteUnionError("method-blind route evidence config is not frozen")
    if config.get("candidate_selection") is not False:
        raise MethodBlindRouteUnionError("route evidence construction cannot select candidates")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {
        "union_result",
        "blind_worklist",
        "source_result",
        "source_component_ledger",
    }:
        raise MethodBlindRouteUnionError("method-blind route evidence inputs changed")
    inputs = {
        name: resolve_pin(pin, repo, label=f"method-blind route evidence {name}")
        for name, pin in raw_inputs.items()
    }
    union_dir = inputs["union_result"].parent
    if inputs["blind_worklist"] != union_dir / "blind_worklist.jsonl.gz":
        raise MethodBlindRouteUnionError("blind worklist is not owned by the frozen union")
    union = read_json_object(
        inputs["union_result"],
        error=MethodBlindRouteUnionError,
        label="method-blind route union result",
    )
    if union.get("schema_version") != FREEZE_SCHEMA or union.get("status") != "frozen":
        raise MethodBlindRouteUnionError("method-blind route union is not frozen")
    worklist = _load_worklist(union_dir, union)
    source_result = read_json_object(
        inputs["source_result"],
        error=MethodBlindRouteUnionError,
        label="exact-source component dossier result",
    )
    if (
        source_result.get("schema_version") != "phase1_ugi3_complete_computational_dossiers.v1"
        or source_result.get("artifacts", {}).get("component_ledger_sha256")
        != str(sha256_file(inputs["source_component_ledger"]))
        or source_result.get("claims_boundary", {}).get(
            "family_projection_is_exact_source_route_evidence"
        )
        is not False
    ):
        raise MethodBlindRouteUnionError("exact-source component dossier result is inadmissible")
    policy = config.get("policy")
    expected_policy = {
        "source_scope": "preexisting_exact_source_ugi_component_dossiers",
        "unknown_disposition": "explicit_missing_knowledge_abstention",
        "family_projection_can_close": False,
        "proposal_only_route_can_close": False,
        "method_identity_available_to_builder": False,
        "complete_requires_verified_upstream_and_terminal_evidence": True,
    }
    if policy != expected_policy:
        raise MethodBlindRouteUnionError("method-blind route evidence policy changed")
    source = _load_source_component_dossiers(inputs["source_component_ledger"])
    evidence_rows: list[dict[str, Any]] = []
    disposition_counts: dict[str, int] = defaultdict(int)
    for item in worklist:
        key = (item["role"], item["canonical_smiles"])
        source_row = source.get(key)
        exact_source = bool(
            source_row and source_row["exact_source_computational_component"] == "true"
        )
        family_projected = bool(source_row and source_row["family_projected_only"] == "true")
        terminal = bool(
            exact_source and source_row and source_row["l3_terminal_status"] == "closed"
        )
        if source_row and source_row["program_status"] == "accepted_procurement_terminal":
            upstream = exact_source and source_row["l2_forward_status"] == "not_applicable"
            forward_consistency = "not_applicable"
        else:
            upstream = bool(
                exact_source
                and source_row
                and source_row["program_status"] == "exact_source_program"
                and source_row["l2_forward_status"] == "all_steps_uniquely_verified"
            )
            forward_consistency = "exact_unique" if upstream else "not_applicable"
        strict = upstream and terminal
        if strict:
            outcome = "complete"
        elif upstream:
            outcome = "terminal_evidence_missing"
        elif family_projected:
            outcome = "abstain_family_projection_not_exact_source"
        elif source_row is not None:
            outcome = "abstain_incomplete_exact_evidence"
        else:
            outcome = "abstain_missing_knowledge"
        disposition_counts[outcome] += 1
        evidence_rows.append(
            {
                "component_id": item["component_id"],
                "role": item["role"],
                "canonical_smiles": item["canonical_smiles"],
                "final_component_state": "complete" if strict else "unresolved",
                "exact_evidence": {
                    "assessment_outcome": outcome,
                    "strict_complete": strict,
                    "verified_upstream": upstream,
                    "terminal_evidence": terminal,
                    "explicit_abstention": not strict,
                    "current_terminal_leaf_count": 1 if terminal else 0,
                    "component_value": {"forward_consistency": forward_consistency},
                },
                "evidence_basis": (
                    "preexisting_exact_source_component_dossier"
                    if source_row is not None
                    else "no_exact_source_record_in_frozen_evidence_library"
                ),
                "source_component_id": source_row["component_id"] if source_row else None,
            }
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MethodBlindRouteUnionError(f"route-evidence output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "component_evidence.jsonl.gz"
    write_jsonl(evidence_path, evidence_rows)
    result = {
        "schema_version": EVIDENCE_RESULT_SCHEMA,
        "status": "complete_with_explicit_abstentions",
        "config": pin_record(config_path, repo),
        "inputs": {name: pin_record(path, repo) for name, path in sorted(inputs.items())},
        "components": len(evidence_rows),
        "dispositions": dict(sorted(disposition_counts.items())),
        "component_evidence": artifact_record(evidence_path),
        "gates": {
            "public_worklist_only": True,
            "private_membership_read": False,
            "every_union_component_dispositioned": len(evidence_rows) == len(worklist),
            "family_projection_cannot_close": True,
            "proposal_only_route_cannot_close": True,
            "unknown_components_explicitly_abstain": True,
        },
        "nonclaims": [
            "An abstention is missing evidence, not proof that a component is unsynthesizable.",
            "A complete computational dossier is not experimental synthesis success.",
            "The frozen source library is exact but not exhaustive over chemical space.",
        ],
        "candidate_selection": False,
    }
    write_json(output_dir / "result.json", result)
    return result


def adjudicate_method_blind_route_union(
    union_dir: Path,
    evidence_path: Path,
    assessed_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Score every method against a complete evidence disposition for the frozen union."""

    union = read_json_object(
        union_dir / "result.json",
        error=MethodBlindRouteUnionError,
        label="method-blind route union result",
    )
    if union.get("schema_version") != FREEZE_SCHEMA or union.get("status") != "frozen":
        raise MethodBlindRouteUnionError("method-blind route union is not frozen")
    worklist = _load_worklist(union_dir, union)
    expected = {(row["role"], row["canonical_smiles"]) for row in worklist}
    evidence = load_frozen_component_evidence(evidence_path)
    if set(evidence) != expected:
        missing = sorted(expected - set(evidence))
        extra = sorted(set(evidence) - expected)
        raise MethodBlindRouteUnionError(
            f"union evidence must disposition every and only frozen component; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    declared_sources = {
        (row["method_id"], int(row["seed"])): (row["sha256"], int(row["bytes"]))
        for row in union["assessed_sources"]
    }
    supplied: dict[tuple[str, int], tuple[Path, list[dict[str, Any]]]] = {}
    for path in assessed_paths:
        rows = _load_assessed(path)
        identity = _ledger_identity(rows, path)
        if identity in supplied:
            raise MethodBlindRouteUnionError(f"duplicate adjudication ledger: {identity}")
        supplied[identity] = (path, rows)
    if set(supplied) != set(declared_sources):
        raise MethodBlindRouteUnionError(
            "adjudication ledgers differ from the frozen union population"
        )
    for identity, (path, _) in supplied.items():
        expected_hash, expected_bytes = declared_sources[identity]
        if str(sha256_file(path)) != expected_hash or path.stat().st_size != expected_bytes:
            raise MethodBlindRouteUnionError(f"frozen assessed ledger changed: {identity}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MethodBlindRouteUnionError(f"route-adjudication output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    scope = {
        "population": "exact_role_constitution_union_from_all_common_methods_and_seeds",
        "method_blind_cross_method_union": True,
        "route_evidence_closure_comparison_authorized": True,
        "synthesis_success_comparison_authorized": False,
    }
    for (method, seed), (path, rows) in sorted(supplied.items()):
        assessed, summary = assess_common_route_evidence(
            rows, component_evidence=evidence, evidence_scope=scope
        )
        ledger_path = output_dir / f"{method}__{seed}.jsonl.gz"
        write_jsonl(
            ledger_path,
            [
                {
                    "schema_version": "forge.common_ugi_union_route_assessed_attempts.v1",
                    "rows": len(assessed),
                },
                *assessed,
            ],
        )
        summaries.append(
            {
                "method_id": method,
                "seed": seed,
                "source_sha256": str(sha256_file(path)),
                "assessed_attempts": artifact_record(ledger_path),
                "route_evidence_assessment": summary,
            }
        )
    result = {
        "schema_version": ADJUDICATION_SCHEMA,
        "status": "pass",
        "union": artifact_record(union_dir / "result.json"),
        "evidence": artifact_record(evidence_path),
        "components": len(expected),
        "method_seed_results": summaries,
        "gates": {
            "every_union_component_dispositioned": True,
            "no_out_of_union_evidence": True,
            "same_evidence_index_for_every_method": True,
            "planner_calls_during_scoring_zero": True,
            "synthesis_success_probability_claim_absent": True,
        },
        "nonclaims": [
            "An unresolved or search-censored component is not proof of unsynthesizability.",
            "Route-evidence closure is not experimental synthesis success.",
            "Exact L1 replay is transform consistency, not a synthesis-success probability.",
        ],
        "candidate_selection": False,
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "MethodBlindRouteUnionError",
    "adjudicate_method_blind_route_union",
    "build_method_blind_route_evidence",
    "freeze_method_blind_route_union",
]
