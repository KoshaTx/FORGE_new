"""Build the unresolved terminal-material queue for AGILE virtual routes."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_bytes, sha256_file
from forge.route.ugi3_virtual_programs import (
    DIRECT_ALDEHYDE_PROGRAM,
    ESTER_PROGRAM,
    ISOCYANIDE_PROGRAM,
)

CONFIG_SCHEMA_VERSION = "m0_09_agile_virtual_ugi3_terminal_queue_config.v1"
RESULT_SCHEMA_VERSION = "m0_09_agile_virtual_ugi3_terminal_queue.v1"
AGILE_ROUTES_SCHEMA_VERSION = "m0_09_agile_component_routes.v1"
PROCUREMENT_SCHEMA_VERSION = "m0_09_ugi3_virtual_terminal_procurement.v1"
FIELDS = (
    "terminal_id",
    "canonical_smiles",
    "terminal_class",
    "source_kinds_json",
    "dependent_component_count",
    "dependent_component_ids_json",
    "dependent_roles_json",
    "dependent_program_families_json",
    "dependency_evidence_levels_json",
    "component_product_incidence",
    "source_procurement_evidence_json",
    "current_procurement_evidence_json",
    "current_accepted_procurement",
    "procurement_status",
    "next_action",
)
CARBOXYLIC_ACID_QUERY = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
DIOL_QUERY = Chem.MolFromSmarts("[OX2H1]")


class Ugi3VirtualTerminalQueueError(ValueError):
    """Raised when the terminal queue violates its frozen contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise Ugi3VirtualTerminalQueueError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Ugi3VirtualTerminalQueueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Ugi3VirtualTerminalQueueError(f"{label} must contain a JSON object")
    return value


def _verify_hash(path: Path, expected: Any, *, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise Ugi3VirtualTerminalQueueError(
            f"{label} expected_sha256 must be a 64-character string"
        )
    observed = sha256_file(path)
    if observed != expected:
        raise Ugi3VirtualTerminalQueueError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def _read_gzip_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3VirtualTerminalQueueError(f"{label} has no header")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3VirtualTerminalQueueError(f"{label} could not be read: {exc}") from exc
    return rows


def _canonicalize(smiles: str, *, label: str) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3VirtualTerminalQueueError(f"{label} contains invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    if canonical != smiles:
        raise Ugi3VirtualTerminalQueueError(
            f"{label} is not canonical: {smiles!r} != {canonical!r}"
        )
    return canonical


def _json_list(value: str, *, label: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise Ugi3VirtualTerminalQueueError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise Ugi3VirtualTerminalQueueError(f"{label} must be a list")
    return parsed


def _leaf_class(smiles: str, program_family: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3VirtualTerminalQueueError(f"proposed leaf contains invalid SMILES: {smiles!r}")
    if program_family == ESTER_PROGRAM:
        if CARBOXYLIC_ACID_QUERY is not None and molecule.HasSubstructMatch(CARBOXYLIC_ACID_QUERY):
            return "fatty_acid"
        hydroxyls = (
            molecule.GetSubstructMatches(DIOL_QUERY, uniquify=True)
            if DIOL_QUERY is not None
            else ()
        )
        if len(hydroxyls) == 2:
            return "diol"
        raise Ugi3VirtualTerminalQueueError(
            "esterification leaf is neither a carboxylic acid nor a diol"
        )
    if program_family == DIRECT_ALDEHYDE_PROGRAM:
        return "primary_alcohol"
    if program_family == ISOCYANIDE_PROGRAM:
        return "primary_amine"
    raise Ugi3VirtualTerminalQueueError(f"unsupported program family {program_family!r}")


def _source_procurement_index(
    artifact: Mapping[str, Any],
) -> dict[str, set[str]]:
    if artifact.get("schema_version") != AGILE_ROUTES_SCHEMA_VERSION:
        raise Ugi3VirtualTerminalQueueError("AGILE component routes have an unsupported schema")
    routes = artifact.get("routes")
    if not isinstance(routes, list):
        raise Ugi3VirtualTerminalQueueError("AGILE component routes must contain a routes list")
    evidence: dict[str, set[str]] = defaultdict(set)
    for route in routes:
        if not isinstance(route, dict) or not isinstance(
            route.get("steps"),
            list,
        ):
            raise Ugi3VirtualTerminalQueueError("AGILE component route contains malformed steps")
        for step in route["steps"]:
            if not isinstance(step, dict) or not isinstance(
                step.get("reactants"),
                list,
            ):
                raise Ugi3VirtualTerminalQueueError(
                    "AGILE component route contains malformed reactants"
                )
            for reactant in step["reactants"]:
                if not isinstance(reactant, dict):
                    raise Ugi3VirtualTerminalQueueError("AGILE source reactant is malformed")
                smiles = reactant.get("canonical_smiles")
                status = reactant.get("procurement_evidence_status")
                if isinstance(smiles, str) and isinstance(status, str):
                    canonical = _canonicalize(
                        smiles,
                        label="AGILE source reactant",
                    )
                    evidence[canonical].add(status)
    return evidence


def _current_procurement_index(
    artifact: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if artifact.get("schema_version") != PROCUREMENT_SCHEMA_VERSION:
        raise Ugi3VirtualTerminalQueueError(
            "terminal procurement snapshot has an unsupported schema"
        )
    snapshot = artifact.get("snapshot")
    expected = artifact.get("expected_counts")
    records = artifact.get("records")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("region") != "US"
        or not isinstance(snapshot.get("accessed_utc"), str)
        or not isinstance(expected, dict)
        or not isinstance(records, list)
    ):
        raise Ugi3VirtualTerminalQueueError("terminal procurement snapshot is malformed")
    by_smiles: dict[str, dict[str, Any]] = {}
    verified = 0
    closed_count = 0
    discrepancies = 0
    for record in records:
        if not isinstance(record, dict):
            raise Ugi3VirtualTerminalQueueError("terminal procurement record is malformed")
        smiles = record.get("canonical_smiles")
        identity = record.get("identity")
        vendor = record.get("vendor_evidence")
        if (
            not isinstance(smiles, str)
            or not isinstance(identity, dict)
            or not isinstance(vendor, dict)
            or not isinstance(vendor.get("product_code"), str)
            or not isinstance(vendor.get("url"), str)
            or not isinstance(vendor.get("purity"), str)
            or not isinstance(vendor.get("availability_observation"), str)
        ):
            raise Ugi3VirtualTerminalQueueError(
                "terminal procurement record lacks item-level evidence"
            )
        canonical = _canonicalize(
            smiles,
            label="terminal procurement record",
        )
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None or identity.get("inchi_key") != Chem.MolToInchiKey(molecule):
            discrepancies += 1
            raise Ugi3VirtualTerminalQueueError(
                "terminal procurement identity does not match its structure"
            )
        if canonical in by_smiles:
            raise Ugi3VirtualTerminalQueueError("terminal procurement structures are not unique")
        is_verified = record.get("procurement_status") == "current_item_level_vendor_verified"
        closed = record.get("current_item_level_procurement_closed") is True
        if closed and not is_verified:
            raise Ugi3VirtualTerminalQueueError(
                "closed procurement record lacks current vendor verification"
            )
        verified += int(is_verified)
        closed_count += int(closed)
        by_smiles[canonical] = record
    observed_counts = {
        "records": len(records),
        "current_item_level_vendor_verified": verified,
        "current_item_level_procurement_closed": closed_count,
        "identity_or_form_discrepancies": discrepancies,
    }
    if observed_counts != expected:
        raise Ugi3VirtualTerminalQueueError(
            "terminal procurement counts do not match their frozen expectation"
        )
    return by_smiles


def _stable_json(values: Sequence[str]) -> str:
    return json.dumps(
        sorted(set(values)),
        separators=(",", ":"),
    )


def _validate_frozen_counts(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    prefix: str = "summary",
) -> None:
    if set(observed) != set(expected):
        raise Ugi3VirtualTerminalQueueError(
            f"{prefix} fields mismatch: expected {sorted(expected)}, "
            f"observed {sorted(observed)}"
        )
    for field, expected_value in expected.items():
        observed_value = observed[field]
        label = f"{prefix}.{field}"
        if isinstance(expected_value, dict):
            if not isinstance(observed_value, dict):
                raise Ugi3VirtualTerminalQueueError(f"{label} must be an object")
            _validate_frozen_counts(
                observed_value,
                expected_value,
                prefix=label,
            )
        elif observed_value != expected_value:
            raise Ugi3VirtualTerminalQueueError(
                f"{label} mismatch: expected {expected_value!r}, " f"observed {observed_value!r}"
            )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(buffer.getvalue().encode())
    return output.getvalue()


def build_ugi3_virtual_terminal_queue(
    config_path: Path,
    component_ledger_path: Path,
    component_program_ledger_path: Path,
    agile_component_routes_path: Path,
    terminal_procurement_path: Path,
) -> tuple[dict[str, Any], bytes]:
    """Build the deduplicated unresolved head and route-leaf queue."""

    config = _load_json(config_path, label="terminal-queue config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3VirtualTerminalQueueError(f"config schema must be {CONFIG_SCHEMA_VERSION!r}")
    inputs = {
        "component_ledger": component_ledger_path,
        "component_program_ledger": component_program_ledger_path,
        "agile_component_routes": agile_component_routes_path,
        "terminal_procurement": terminal_procurement_path,
    }
    for name, path in inputs.items():
        _verify_hash(
            path,
            config["inputs"][name]["expected_sha256"],
            label=name.replace("_", " "),
        )
    components = _read_gzip_csv(
        component_ledger_path,
        label="component ledger",
    )
    programs = _read_gzip_csv(
        component_program_ledger_path,
        label="component-program ledger",
    )
    component_by_id = {row["component_id"]: row for row in components}
    if len(component_by_id) != len(components):
        raise Ugi3VirtualTerminalQueueError("component ledger contains duplicate identifiers")
    if {row["component_id"] for row in programs} != set(component_by_id):
        raise Ugi3VirtualTerminalQueueError("component and component-program ledgers do not align")
    source_procurement = _source_procurement_index(
        _load_json(
            agile_component_routes_path,
            label="AGILE component routes",
        )
    )
    procurement = _current_procurement_index(
        _load_json(
            terminal_procurement_path,
            label="terminal procurement snapshot",
        )
    )
    dependencies: dict[str, dict[str, Any]] = {}

    def add_dependency(
        leaf: str,
        *,
        terminal_class: str,
        source_kind: str,
        component: Mapping[str, str],
        program: Mapping[str, str],
    ) -> None:
        canonical = _canonicalize(leaf, label="terminal candidate")
        entry = dependencies.setdefault(
            canonical,
            {
                "terminal_classes": set(),
                "source_kinds": set(),
                "component_ids": set(),
                "roles": set(),
                "program_families": set(),
                "evidence_levels": set(),
                "component_product_incidence": 0,
            },
        )
        if component["component_id"] not in entry["component_ids"]:
            entry["component_product_incidence"] += int(component["product_count"])
        entry["terminal_classes"].add(terminal_class)
        entry["source_kinds"].add(source_kind)
        entry["component_ids"].add(component["component_id"])
        entry["roles"].add(component["role"])
        if program["program_family"]:
            entry["program_families"].add(program["program_family"])
        entry["evidence_levels"].add(program["evidence_level"])

    for program in programs:
        component = component_by_id[program["component_id"]]
        status = program["program_status"]
        leaves = _json_list(
            program["proposed_leaf_candidates_json"],
            label=f"{program['component_id']} proposed leaves",
        )
        if any(not isinstance(leaf, str) for leaf in leaves):
            raise Ugi3VirtualTerminalQueueError(
                f"{program['component_id']} has a non-string proposed leaf"
            )
        if status == "accepted_procurement_terminal":
            continue
        if status == "procurement_or_route_search_required":
            if component["role"] != "amine_head" or leaves != [component["canonical_smiles"]]:
                raise Ugi3VirtualTerminalQueueError(
                    "unresolved head terminal record is inconsistent"
                )
            add_dependency(
                leaves[0],
                terminal_class="unresolved_amine_head",
                source_kind="unresolved_head_candidate",
                component=component,
                program=program,
            )
            continue
        if not program["program_family"] or not leaves:
            raise Ugi3VirtualTerminalQueueError(
                f"{program['component_id']} lacks a structural leaf program"
            )
        for leaf in leaves:
            add_dependency(
                leaf,
                terminal_class=_leaf_class(
                    leaf,
                    program["program_family"],
                ),
                source_kind="proposed_route_leaf",
                component=component,
                program=program,
            )

    rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = defaultdict(int)
    vendor_claim_count = 0
    accepted_procurement_count = 0
    route_leaf_count = 0
    head_count = 0
    for canonical, entry in sorted(dependencies.items()):
        if len(entry["terminal_classes"]) != 1:
            raise Ugi3VirtualTerminalQueueError(f"terminal {canonical!r} has inconsistent classes")
        terminal_class = next(iter(entry["terminal_classes"]))
        class_counts[terminal_class] += 1
        source_kinds = sorted(entry["source_kinds"])
        route_leaf_count += int("proposed_route_leaf" in source_kinds)
        head_count += int("unresolved_head_candidate" in source_kinds)
        procurement_states = sorted(source_procurement.get(canonical, set()))
        vendor_claim_count += int("vendor_claim_only" in procurement_states)
        current_record = procurement.get(canonical)
        current_closed = (
            current_record is not None
            and current_record.get("current_item_level_procurement_closed") is True
        )
        accepted_procurement_count += int(current_closed)
        terminal_key = canonical.encode()
        rows.append(
            {
                "terminal_id": (
                    "virtual-terminal-" f"{hashlib.sha256(terminal_key).hexdigest()[:20]}"
                ),
                "canonical_smiles": canonical,
                "terminal_class": terminal_class,
                "source_kinds_json": _stable_json(source_kinds),
                "dependent_component_count": len(entry["component_ids"]),
                "dependent_component_ids_json": _stable_json(entry["component_ids"]),
                "dependent_roles_json": _stable_json(entry["roles"]),
                "dependent_program_families_json": _stable_json(entry["program_families"]),
                "dependency_evidence_levels_json": _stable_json(entry["evidence_levels"]),
                "component_product_incidence": entry["component_product_incidence"],
                "source_procurement_evidence_json": _stable_json(procurement_states),
                "current_procurement_evidence_json": json.dumps(
                    current_record or {},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "current_accepted_procurement": str(current_closed).lower(),
                "procurement_status": (
                    "current_item_level_vendor_verified" if current_closed else "unresolved"
                ),
                "next_action": (
                    "none"
                    if current_closed
                    else (
                        current_record.get("next_action")
                        if current_record is not None
                        and isinstance(current_record.get("next_action"), str)
                        else (
                            "verify_exact_identity_current_vendor_or_internal_stock;"
                            "if_unavailable_route_recursively"
                        )
                    )
                ),
            }
        )
    summary = {
        "terminal_candidates": len(rows),
        "unresolved_terminal_candidates": (len(rows) - accepted_procurement_count),
        "proposed_route_leaf_candidates": route_leaf_count,
        "unresolved_head_candidates": head_count,
        "terminal_classes": dict(sorted(class_counts.items())),
        "candidates_with_source_vendor_claim_only": vendor_claim_count,
        "candidates_with_current_accepted_procurement": (accepted_procurement_count),
    }
    _validate_frozen_counts(summary, config["expected_counts"])
    ledger = _csv_bytes(rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config["task"],
        "generated_utc": config["generated_utc"],
        "randomness": {"seed": 0, "used": False},
        "inputs": [
            {
                "asset": config_path.name,
                "role": "audit_config",
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            *[
                {
                    "asset": config["inputs"][name]["asset"],
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for name, path in inputs.items()
            ],
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": summary,
        "artifacts": {
            "agile_virtual_ugi3_terminal_queue.csv.gz": {
                "path": ("results/m0_09/" "agile_virtual_ugi3_terminal_queue.csv.gz"),
                "bytes": len(ledger),
                "sha256": sha256_bytes(ledger),
            }
        },
        "claims_boundary": config["claims_boundary"],
        "decision": {
            "batch_terminal_resolution_by_exact_identity": True,
            "prioritize_using_explicit_dependency_counts": True,
            "current_vendor_verification_required": True,
            "model_built": False,
        },
    }
    return result, ledger


def write_ugi3_virtual_terminal_queue(
    result: Mapping[str, Any],
    ledger: bytes,
    output_dir: Path,
) -> None:
    """Atomically replace deterministic terminal-queue artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "agile_virtual_ugi3_terminal_queue.csv.gz": ledger,
        "agile_virtual_ugi3_terminal_queue.json": (
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    temporary_paths: dict[str, Path] = {}
    try:
        for name, payload in payloads.items():
            descriptor, temporary = tempfile.mkstemp(
                dir=output_dir,
                prefix=f".{name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary)
            temporary_paths[name] = temporary_path
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for name in sorted(payloads):
            os.replace(temporary_paths[name], output_dir / name)
    except Exception:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise
