from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from experiments.phase1.multireaction.method_blind_route_union import (
    MethodBlindRouteUnionError,
    adjudicate_method_blind_route_union,
    build_method_blind_route_evidence,
    freeze_method_blind_route_union,
)
from forge.core.hashing import sha256_file
from forge.core.io import iter_jsonl, write_json, write_jsonl


def _assessed(path: Path, *, method: str, seed: int, components: dict[str, str]) -> None:
    row = {
        "method_id": method,
        "seed": seed,
        "attempt_index": 0,
        "exact_l1_program": True,
        "exact_l1_trace_count": 1,
        "exact_l1_traces": [{"components_by_role": components}],
    }
    write_jsonl(
        path,
        [{"schema_version": "forge.common_ugi_assessed_attempts.v1", "rows": 1}, row],
    )


def _evidence(path: Path, components: set[tuple[str, str]], *, omit: tuple[str, str] | None = None):
    rows = []
    for role, smiles in sorted(components):
        if (role, smiles) == omit:
            continue
        rows.append(
            {
                "role": role,
                "canonical_smiles": smiles,
                "final_component_state": "unresolved",
                "exact_evidence": {
                    "strict_complete": False,
                    "assessment_outcome": "missing_knowledge",
                    "current_terminal_leaf_count": 0,
                    "component_value": {"forward_consistency": "not_applicable"},
                },
            }
        )
    write_jsonl(path, rows)


def test_union_worklist_is_exact_deduplicated_and_method_blind(tmp_path) -> None:
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    shared = "CN"
    _assessed(first, method="one", seed=1, components={"amine_head": shared, "tail": "CCCC"})
    _assessed(second, method="two", seed=1, components={"amine_head": shared, "tail": "CCCCC"})
    union_dir = tmp_path / "union"
    result = freeze_method_blind_route_union([first, second], union_dir)
    public = list(iter_jsonl(union_dir / "blind_worklist.jsonl.gz"))
    private = list(iter_jsonl(union_dir / "private_membership.jsonl.gz"))
    assert result["components"] == 3
    assert all("method_id" not in row for row in public[1:])
    assert {row["method_id"] for row in private[1:]} == {"one", "two"}

    components = {(row["role"], row["canonical_smiles"]) for row in public[1:]}
    evidence = tmp_path / "evidence.jsonl.gz"
    _evidence(evidence, components)
    output = tmp_path / "adjudicated"
    adjudicated = adjudicate_method_blind_route_union(union_dir, evidence, [first, second], output)
    assert adjudicated["status"] == "pass"
    assert adjudicated["gates"]["same_evidence_index_for_every_method"]
    assert all(
        item["route_evidence_assessment"]["evidence_scope"]["method_blind_cross_method_union"]
        for item in adjudicated["method_seed_results"]
    )


def test_union_adjudication_fails_when_one_component_has_no_disposition(tmp_path) -> None:
    assessed = tmp_path / "assessed.jsonl.gz"
    _assessed(assessed, method="one", seed=1, components={"amine_head": "CN", "tail": "CCCC"})
    union_dir = tmp_path / "union"
    freeze_method_blind_route_union([assessed], union_dir)
    public = list(iter_jsonl(union_dir / "blind_worklist.jsonl.gz"))[1:]
    components = {(row["role"], row["canonical_smiles"]) for row in public}
    missing = next(iter(components))
    evidence = tmp_path / "evidence.jsonl.gz"
    _evidence(evidence, components, omit=missing)
    with pytest.raises(MethodBlindRouteUnionError, match="disposition every"):
        adjudicate_method_blind_route_union(
            union_dir, evidence, [assessed], tmp_path / "adjudicated"
        )


def test_blind_evidence_builder_admits_only_exact_source_and_abstains_unknown(
    tmp_path: Path,
) -> None:
    assessed = tmp_path / "assessed.jsonl.gz"
    _assessed(
        assessed,
        method="one",
        seed=1,
        components={"amine_head": "CN", "tail": "CCCC"},
    )
    union_dir = tmp_path / "union"
    freeze_method_blind_route_union([assessed], union_dir)
    source_ledger = tmp_path / "source.csv.gz"
    fieldnames = [
        "component_id",
        "role",
        "canonical_smiles",
        "program_status",
        "l2_forward_status",
        "l3_terminal_status",
        "exact_source_computational_component",
        "family_projected_only",
    ]
    with gzip.open(source_ledger, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "component_id": "source-amine",
                "role": "amine_head",
                "canonical_smiles": "CN",
                "program_status": "accepted_procurement_terminal",
                "l2_forward_status": "not_applicable",
                "l3_terminal_status": "closed",
                "exact_source_computational_component": "true",
                "family_projected_only": "false",
            }
        )
    source_result = tmp_path / "source_result.json"
    write_json(
        source_result,
        {
            "schema_version": "phase1_ugi3_complete_computational_dossiers.v1",
            "artifacts": {"component_ledger_sha256": str(sha256_file(source_ledger))},
            "claims_boundary": {"family_projection_is_exact_source_route_evidence": False},
        },
    )
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "schema_version": "forge.common_ugi_method_blind_route_evidence_config.v1",
            "status": "frozen_after_union_before_evidence_lookup",
            "inputs": {
                "union_result": {
                    "path": "union/result.json",
                    "sha256": str(sha256_file(union_dir / "result.json")),
                },
                "blind_worklist": {
                    "path": "union/blind_worklist.jsonl.gz",
                    "sha256": str(sha256_file(union_dir / "blind_worklist.jsonl.gz")),
                },
                "source_result": {
                    "path": "source_result.json",
                    "sha256": str(sha256_file(source_result)),
                },
                "source_component_ledger": {
                    "path": "source.csv.gz",
                    "sha256": str(sha256_file(source_ledger)),
                },
            },
            "policy": {
                "source_scope": "preexisting_exact_source_ugi_component_dossiers",
                "unknown_disposition": "explicit_missing_knowledge_abstention",
                "family_projection_can_close": False,
                "proposal_only_route_can_close": False,
                "method_identity_available_to_builder": False,
                "complete_requires_verified_upstream_and_terminal_evidence": True,
            },
            "candidate_selection": False,
        },
    )
    evidence_dir = tmp_path / "evidence"
    result = build_method_blind_route_evidence(config, tmp_path, evidence_dir)
    rows = list(iter_jsonl(evidence_dir / "component_evidence.jsonl.gz"))
    assert result["components"] == 2
    assert result["gates"]["private_membership_read"] is False
    assert all("method_id" not in row for row in rows)
    indexed = {(row["role"], row["canonical_smiles"]): row for row in rows}
    assert indexed[("amine_head", "CN")]["final_component_state"] == "complete"
    unknown = indexed[("tail", "CCCC")]
    assert unknown["final_component_state"] == "unresolved"
    assert unknown["exact_evidence"]["explicit_abstention"] is True
