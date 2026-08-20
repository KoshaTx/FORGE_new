"""Audit LNPDB head motifs for bounded transfer into the Ugi program."""

from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.chemistry import (
    SUPPORTED_MULTIPLICITY_SEMANTICS,
    audit_reactive_site_multiplicity,
)
from forge.core.hashing import sha256_file
from forge.core.io import atomic_write as _atomic_write
from forge.core.io import read_json_object
from forge.design.audit.ring_support_audit import _ring_signature

CONFIG_SCHEMA_VERSION = "m0_09_lnpdb_head_transfer_config.v1"
RESULT_SCHEMA_VERSION = "m0_09_lnpdb_head_transfer_result.v1"
RESULT_NAME = "lnpdb_head_transfer.json"
LEDGER_NAME = "lnpdb_head_transfer_ledger.csv.gz"
LEDGER_FIELDS = (
    "component_id",
    "canonical_smiles",
    "parse_status",
    "unique_lipid_count",
    "source_count",
    "source_ids_json",
    "source_pmids_json",
    "ring_count",
    "ring_sizes_json",
    "macrocycle",
    "fused",
    "spiro",
    "bridgehead",
    "aromatic_ring_count",
    "heterocyclic_ring_count",
    "raw_amine_handle_matches",
    "symmetry_distinct_amine_sites",
    "allowed_amine_site_multiplicity_json",
    "passes_qualified_amine_handle_policy",
    "site_resolution_required",
    "within_primary_ugi_ring_topology",
    "exact_agile_head_match",
    "disposition",
    "route_evidence_status",
    "procurement_evidence_status",
    "execution_closure_status",
    "automatic_candidate_lock_admission",
    "transfer_priority_rank",
    "next_action",
)


class LnpdbHeadTransferError(ValueError):
    """Raised when the LNPDB head-transfer census violates its contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(path, error=LnpdbHeadTransferError, label=label)


def _read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise LnpdbHeadTransferError(f"{label} lacks fields: {sorted(missing)}")
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise LnpdbHeadTransferError(f"{label} not found: {path}") from exc


def _canonical_smiles(value: str, label: str) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise LnpdbHeadTransferError(f"{label} has invalid SMILES: {value!r}")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise LnpdbHeadTransferError(f"{label} is disconnected")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _load_amine_role(
    registry: Mapping[str, Any],
    *,
    reaction_id: str,
    role_name: str,
) -> tuple[Chem.Mol, frozenset[int]]:
    reactions = registry.get("reactions")
    if not isinstance(reactions, list):
        raise LnpdbHeadTransferError("qualified reaction registry lacks reactions")
    matches = [
        reaction
        for reaction in reactions
        if isinstance(reaction, Mapping) and reaction.get("reaction_id") == reaction_id
    ]
    if len(matches) != 1:
        raise LnpdbHeadTransferError(
            f"expected one qualified reaction {reaction_id!r}, found {len(matches)}"
        )
    roles = matches[0].get("reactant_roles")
    if not isinstance(roles, list):
        raise LnpdbHeadTransferError("qualified Ugi reaction lacks reactant roles")
    role_matches = [
        role for role in roles if isinstance(role, Mapping) and role.get("name") == role_name
    ]
    if len(role_matches) != 1:
        raise LnpdbHeadTransferError(
            f"expected one qualified role {role_name!r}, found {len(role_matches)}"
        )
    role = role_matches[0]
    query = Chem.MolFromSmarts(str(role.get("required_handle_smarts", "")))
    allowed_raw = role.get("allowed_site_multiplicity")
    if query is None or not isinstance(allowed_raw, list) or not allowed_raw:
        raise LnpdbHeadTransferError("qualified amine handle policy is invalid")
    allowed = frozenset(int(value) for value in allowed_raw)
    return query, allowed


def _within_primary_ring_topology(
    signature: Mapping[str, Any],
    ring_support: Mapping[str, Any],
) -> bool:
    support = ring_support.get("decision", {}).get(
        "primary_ugi_automatic_support",
        {},
    )
    allowed_sizes = set(support.get("single_ring_sizes", []))
    acyclic = signature["ring_count"] == 0 and support.get("acyclic_products") is True
    single_allowed = (
        signature["ring_count"] == 1
        and signature["ring_sizes"][0] in allowed_sizes
        and signature["ring_count"] <= support.get("maximum_rings", 0)
    )
    excluded = (
        (signature["macrocycle"] and not support.get("macrocycle_allowed"))
        or (signature["fused"] and not support.get("fused_allowed"))
        or (signature["spiro"] and not support.get("spiro_allowed"))
        or (signature["bridgehead"] and not support.get("bridgehead_allowed"))
    )
    return bool((acyclic or single_allowed) and not excluded)


def _priority_key(record: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (
        -int(record["unique_lipid_count"]),
        -int(record["source_count"]),
        -int(int(record.get("heterocyclic_ring_count") or 0) > 0),
        str(record["component_id"]),
    )


def _ledger_row(
    row: Mapping[str, str],
    *,
    amine_query: Chem.Mol,
    allowed_sites: frozenset[int],
    multiplicity_semantics: str,
    agile_heads: set[str],
    ring_support: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "component_id": row["component_id"],
        "canonical_smiles": row["canonical_smiles"],
        "parse_status": row["parse_status"],
        "unique_lipid_count": int(row["unique_lipid_count"]),
        "source_count": int(row["source_count"]),
        "source_ids_json": row["source_ids_json"],
        "source_pmids_json": row["source_pmids_json"],
        "route_evidence_status": row["route_evidence_status"],
        "procurement_evidence_status": row["procurement_evidence_status"],
        "execution_closure_status": row["execution_closure_status"],
        "automatic_candidate_lock_admission": False,
        "transfer_priority_rank": "",
    }
    if row["parse_status"] != "parsed" or not row["canonical_smiles"]:
        return {
            **base,
            "ring_count": "",
            "ring_sizes_json": "[]",
            "macrocycle": "",
            "fused": "",
            "spiro": "",
            "bridgehead": "",
            "aromatic_ring_count": "",
            "heterocyclic_ring_count": "",
            "raw_amine_handle_matches": "",
            "symmetry_distinct_amine_sites": "",
            "allowed_amine_site_multiplicity_json": json.dumps(sorted(allowed_sites)),
            "passes_qualified_amine_handle_policy": False,
            "site_resolution_required": "",
            "within_primary_ugi_ring_topology": False,
            "exact_agile_head_match": False,
            "disposition": "invalid_source_head_structure",
            "next_action": "repair_or_exclude_source_structure",
        }
    canonical = _canonical_smiles(
        row["canonical_smiles"],
        f"LNPDB head {row['component_id']}",
    )
    molecule = Chem.MolFromSmiles(canonical)
    signature = _ring_signature(
        molecule,
        int(ring_support["definitions"]["macrocycle"].rsplit(" ", 1)[-1]),
    )
    sites = audit_reactive_site_multiplicity(molecule, amine_query)
    site_count = sites.count(multiplicity_semantics)
    passes_handle = site_count in allowed_sites
    within_topology = _within_primary_ring_topology(signature, ring_support)
    exact_agile = canonical in agile_heads
    if exact_agile:
        disposition = "exact_agile_head"
        next_action = "use_existing_agile_head_evidence_and_closure_policy"
    elif not passes_handle:
        disposition = "without_qualified_ugi_amine"
        next_action = "retain_for_broad_structure_learning_only"
    elif not within_topology:
        disposition = "outside_primary_ring_topology"
        next_action = "abstain_pending_evidence_and_support_amendment"
    else:
        disposition = "lnpdb_ugi_head_transfer_candidate"
        next_action = "resolve_reactive_site_procurement_route_and_forward_qualification"
    return {
        **base,
        "canonical_smiles": canonical,
        "ring_count": signature["ring_count"],
        "ring_sizes_json": json.dumps(signature["ring_sizes"], separators=(",", ":")),
        "macrocycle": signature["macrocycle"],
        "fused": signature["fused"],
        "spiro": signature["spiro"],
        "bridgehead": signature["bridgehead"],
        "aromatic_ring_count": signature["aromatic_ring_count"],
        "heterocyclic_ring_count": signature["heterocyclic_ring_count"],
        "raw_amine_handle_matches": sites.raw_match_count,
        "symmetry_distinct_amine_sites": sites.symmetry_distinct_match_count,
        "allowed_amine_site_multiplicity_json": json.dumps(sorted(allowed_sites)),
        "passes_qualified_amine_handle_policy": passes_handle,
        "site_resolution_required": site_count > 1,
        "within_primary_ugi_ring_topology": within_topology,
        "exact_agile_head_match": exact_agile,
        "disposition": disposition,
        "next_action": next_action,
    }


def _verify_inputs(
    config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    specifications = config.get("inputs")
    expected_names = {
        "component_source_ledger",
        "qualified_reactions",
        "agile_curated",
        "ring_support",
    }
    if not isinstance(specifications, Mapping) or set(specifications) != expected_names:
        raise LnpdbHeadTransferError("head-transfer input contract is incomplete")
    paths: dict[str, Path] = {}
    verified: dict[str, dict[str, Any]] = {}
    for name, specification in sorted(specifications.items()):
        if not isinstance(specification, Mapping):
            raise LnpdbHeadTransferError(f"{name} input specification is invalid")
        path = repo_root / str(specification.get("path", ""))
        observed = sha256_file(path)
        if observed != specification.get("sha256"):
            raise LnpdbHeadTransferError(
                f"{name} hash mismatch: expected {specification.get('sha256')}, observed {observed}"
            )
        paths[name] = path
        verified[name] = {
            "path": str(path.resolve().relative_to(repo_root.resolve())),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return paths, verified


def _assert_expected(
    expected: Mapping[str, Any],
    *,
    component_count: int,
    heads: Sequence[Mapping[str, Any]],
    agile_head_count: int,
) -> None:
    dispositions = Counter(str(row["disposition"]) for row in heads)
    observed = {
        "component_rows": component_count,
        "head_rows": len(heads),
        "parsed_head_rows": sum(row["parse_status"] == "parsed" for row in heads),
        "invalid_head_rows": dispositions["invalid_source_head_structure"],
        "unique_agile_heads": agile_head_count,
        "exact_agile_head_overlap": dispositions["exact_agile_head"],
        "transfer_candidates": dispositions["lnpdb_ugi_head_transfer_candidate"],
        "outside_primary_ring_topology": dispositions["outside_primary_ring_topology"],
        "without_qualified_ugi_amine": dispositions["without_qualified_ugi_amine"],
    }
    if observed != expected:
        raise LnpdbHeadTransferError(
            f"head-transfer census changed: expected {dict(expected)}, observed {observed}"
        )


def build_lnpdb_head_transfer(
    config_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the deterministic LNPDB head-transfer census and candidate queue."""

    config = _load_json(config_path, "head-transfer config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LnpdbHeadTransferError("unsupported head-transfer config schema")
    if config.get("seed") != 20260730:
        raise LnpdbHeadTransferError("head-transfer seed must remain 20260730")
    semantics = config.get("multiplicity_semantics")
    if semantics not in SUPPORTED_MULTIPLICITY_SEMANTICS:
        raise LnpdbHeadTransferError("unsupported amine multiplicity semantics")
    paths, verified = _verify_inputs(config, repo_root)
    components = _read_csv(
        paths["component_source_ledger"],
        {
            "component_id",
            "role",
            "parse_status",
            "canonical_smiles",
            "unique_lipid_count",
            "source_count",
            "source_ids_json",
            "source_pmids_json",
            "route_evidence_status",
            "procurement_evidence_status",
            "execution_closure_status",
        },
        "LNPDB component ledger",
    )
    agile = _read_csv(
        paths["agile_curated"],
        {"A_smiles"},
        "reconciled AGILE",
    )
    registry = _load_json(paths["qualified_reactions"], "qualified reaction registry")
    ring_support = _load_json(paths["ring_support"], "ring-support result")
    if (
        ring_support.get("schema_version") != "m0_06_lipid_ring_support_result.v1"
        or ring_support.get("decision", {}).get("primary_ugi_automatic_ring_gate_supported")
        is not True
    ):
        raise LnpdbHeadTransferError("ring-support artifact is not an accepted gate")
    amine_query, allowed_sites = _load_amine_role(
        registry,
        reaction_id=str(config.get("reaction_id", "")),
        role_name=str(config.get("amine_role", "")),
    )
    agile_heads = {_canonical_smiles(row["A_smiles"], "AGILE amine head") for row in agile}
    head_rows = [row for row in components if row["role"] == "head"]
    ledger = [
        _ledger_row(
            row,
            amine_query=amine_query,
            allowed_sites=allowed_sites,
            multiplicity_semantics=semantics,
            agile_heads=agile_heads,
            ring_support=ring_support,
        )
        for row in head_rows
    ]
    candidates = [
        row for row in ledger if row["disposition"] == "lnpdb_ugi_head_transfer_candidate"
    ]
    for rank, record in enumerate(sorted(candidates, key=_priority_key), start=1):
        record["transfer_priority_rank"] = rank
    expected = config.get("expected")
    if not isinstance(expected, Mapping):
        raise LnpdbHeadTransferError("head-transfer expected counts are absent")
    _assert_expected(
        expected,
        component_count=len(components),
        heads=ledger,
        agile_head_count=len(agile_heads),
    )
    dispositions = Counter(str(row["disposition"]) for row in ledger)
    candidate_rings = Counter(row["ring_sizes_json"] for row in candidates)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed_lnpdb_head_transfer_census",
        "task": config["task"],
        "scope": "M0 candidate census only; no head is admitted by this artifact",
        "inputs": verified,
        "configuration": {
            "path": str(config_path.resolve().relative_to(repo_root.resolve())),
            "sha256": sha256_file(config_path),
            "seed": config["seed"],
        },
        "summary": {
            "component_rows": len(components),
            "head_rows": len(ledger),
            "parsed_head_rows": sum(row["parse_status"] == "parsed" for row in ledger),
            "unique_agile_heads": len(agile_heads),
            "disposition_counts": dict(sorted(dispositions.items())),
            "transfer_candidate_ring_size_counts": dict(sorted(candidate_rings.items())),
            "transfer_candidates_requiring_site_resolution": sum(
                bool(row["site_resolution_required"]) for row in candidates
            ),
            "transfer_candidates_with_route_evidence": sum(
                row["route_evidence_status"] != "not_assessed" for row in candidates
            ),
            "transfer_candidates_with_procurement_evidence": sum(
                row["procurement_evidence_status"] != "not_assessed" for row in candidates
            ),
            "automatically_admitted_candidates": 0,
        },
        "decision": {
            "bounded_head_transfer_queue_supported": True,
            "broad_generator_head_support_expanded_by_training": (
                "LNPDB head structures may inform broad molecular representation"
            ),
            "primary_ugi_admission_rule": (
                "a transfer candidate still requires site resolution, exact forward "
                "qualification, and route or procurement closure"
            ),
            "arbitrary_macrocycle_generation_authorized": False,
            "exact_ring_fragment_vocabulary_required": False,
            "resume_head_route_mining_for_all_candidates": False,
            "mining_priority": (
                "prioritize generator-demanded or biologically informative candidates "
                "from the frozen queue"
            ),
        },
        "scope_policy": config["scope_policy"],
        "limitations": [
            (
                "LNPDB head occurrence is not evidence that the head is compatible "
                "with the AGILE Ugi conditions."
            ),
            (
                "The component ledger contains no assessed procurement or upstream "
                "route evidence for these transferred heads."
            ),
            ("Source-lipid biological outcomes do not transfer to a new Ugi product."),
            ("The current M0-07 oracle authorizes no biological guidance domain."),
        ],
        "runtime": {
            "python": sys.version,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    return result, sorted(
        ledger,
        key=lambda row: (
            {
                "exact_agile_head": 0,
                "lnpdb_ugi_head_transfer_candidate": 1,
                "outside_primary_ring_topology": 2,
                "without_qualified_ugi_amine": 3,
                "invalid_source_head_structure": 4,
            }[str(row["disposition"])],
            _priority_key(row),
        ),
    )


def write_lnpdb_head_transfer(
    result: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    """Write stable JSON and compressed CSV artifacts atomically."""

    _atomic_write(
        output_dir / RESULT_NAME,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    rows: list[dict[str, Any]] = []
    for record in ledger:
        unknown = set(record) - set(LEDGER_FIELDS)
        if unknown:
            raise LnpdbHeadTransferError(
                f"head-transfer ledger contains unknown fields: {sorted(unknown)}"
            )
        rows.append({field: record.get(field, "") for field in LEDGER_FIELDS})
    buffer = tempfile.SpooledTemporaryFile(mode="w+", newline="")
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)
    csv_payload = buffer.read().encode()
    buffer.close()
    compressed = gzip.compress(csv_payload, compresslevel=9, mtime=0)
    _atomic_write(output_dir / LEDGER_NAME, compressed)
