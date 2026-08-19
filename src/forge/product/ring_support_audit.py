"""Audit lipid-native ring topology and freeze a bounded Ugi campaign policy."""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from forge.core.hashing import sha256_file

CONFIG_SCHEMA_VERSION = "m0_06_lipid_ring_support_config.v1"
RESULT_SCHEMA_VERSION = "m0_06_lipid_ring_support_result.v1"


class RingSupportAuditError(ValueError):
    """Raised when ring-support inputs or frozen policies are invalid."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RingSupportAuditError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RingSupportAuditError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RingSupportAuditError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise RingSupportAuditError(f"{label} lacks fields: {sorted(missing)}")
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise RingSupportAuditError(f"{label} not found: {path}") from exc


def _molecule(smiles: str, label: str) -> Chem.Mol:
    if not smiles:
        raise RingSupportAuditError(f"{label} has empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RingSupportAuditError(f"{label} has invalid SMILES: {smiles!r}")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise RingSupportAuditError(f"{label} is disconnected")
    return molecule


def _ring_signature(molecule: Chem.Mol, macrocycle_minimum: int) -> dict[str, Any]:
    rings = [tuple(int(atom) for atom in ring) for ring in Chem.GetSymmSSSR(molecule)]
    sizes = sorted(len(ring) for ring in rings)
    fused = any(
        len(set(left) & set(right)) >= 2
        for index, left in enumerate(rings)
        for right in rings[index + 1 :]
    )
    aromatic_rings = 0
    heterocyclic_rings = 0
    aromatic_heterocycles = 0
    for ring in rings:
        atoms = [molecule.GetAtomWithIdx(index) for index in ring]
        aromatic = all(atom.GetIsAromatic() for atom in atoms)
        heterocyclic = any(atom.GetAtomicNum() != 6 for atom in atoms)
        aromatic_rings += int(aromatic)
        heterocyclic_rings += int(heterocyclic)
        aromatic_heterocycles += int(aromatic and heterocyclic)
    edge_count = molecule.GetNumBonds()
    node_count = molecule.GetNumAtoms()
    return {
        "ring_count": len(rings),
        "ring_sizes": sizes,
        "cycle_rank": edge_count - node_count + 1,
        "maximum_ring_size": max(sizes, default=0),
        "macrocycle": any(size >= macrocycle_minimum for size in sizes),
        "fused": fused,
        "spiro": rdMolDescriptors.CalcNumSpiroAtoms(molecule) > 0,
        "bridgehead": rdMolDescriptors.CalcNumBridgeheadAtoms(molecule) > 0,
        "aromatic_ring_count": aromatic_rings,
        "heterocyclic_ring_count": heterocyclic_rings,
        "aromatic_heterocycle_count": aromatic_heterocycles,
    }


def _signature_key(signature: Mapping[str, Any]) -> str:
    payload = {
        "ring_sizes": signature["ring_sizes"],
        "fused": signature["fused"],
        "spiro": signature["spiro"],
        "bridgehead": signature["bridgehead"],
        "macrocycle": signature["macrocycle"],
        "aromatic_ring_count": signature["aromatic_ring_count"],
        "heterocyclic_ring_count": signature["heterocyclic_ring_count"],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _aggregate_signatures(
    rows: Sequence[Mapping[str, str]],
    *,
    smiles_field: str,
    label: str,
    macrocycle_minimum: int,
    source_field: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ring_sizes: Counter[int] = Counter()
    ring_size_records: Counter[int] = Counter()
    cycle_ranks: Counter[int] = Counter()
    signature_counts: Counter[str] = Counter()
    source_macrocycles: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        molecule = _molecule(str(row[smiles_field]), f"{label} row {index}")
        signature = _ring_signature(molecule, macrocycle_minimum)
        ring_sizes.update(signature["ring_sizes"])
        ring_size_records.update(set(signature["ring_sizes"]))
        cycle_ranks[signature["cycle_rank"]] += 1
        signature_counts[_signature_key(signature)] += 1
        if signature["macrocycle"] and source_field is not None:
            source_macrocycles.update(
                source for source in str(row[source_field]).split("|") if source
            )
        records.append(signature)
    cyclic = sum(record["ring_count"] > 0 for record in records)
    summary = {
        "records": len(records),
        "acyclic_records": len(records) - cyclic,
        "cyclic_records": cyclic,
        "cyclic_fraction": cyclic / len(records),
        "single_ring_records": sum(record["ring_count"] == 1 for record in records),
        "multiple_ring_records": sum(record["ring_count"] > 1 for record in records),
        "ring_size_instance_counts": {
            str(size): count for size, count in sorted(ring_sizes.items())
        },
        "ring_size_record_counts": {
            str(size): count for size, count in sorted(ring_size_records.items())
        },
        "cycle_rank_counts": {str(rank): count for rank, count in sorted(cycle_ranks.items())},
        "maximum_ring_size": max(
            (record["maximum_ring_size"] for record in records),
            default=0,
        ),
        "macrocycle_records": sum(record["macrocycle"] for record in records),
        "fused_ring_records": sum(record["fused"] for record in records),
        "spiro_ring_records": sum(record["spiro"] for record in records),
        "bridgehead_ring_records": sum(record["bridgehead"] for record in records),
        "aromatic_ring_records": sum(record["aromatic_ring_count"] > 0 for record in records),
        "heterocycle_records": sum(record["heterocyclic_ring_count"] > 0 for record in records),
        "aromatic_heterocycle_records": sum(
            record["aromatic_heterocycle_count"] > 0 for record in records
        ),
        "top_ring_signatures": [
            {"signature": json.loads(signature), "records": count}
            for signature, count in signature_counts.most_common(20)
        ],
        "macrocycle_source_counts": dict(sorted(source_macrocycles.items())),
    }
    return summary, records


def _agile_role_audit(
    rows: Sequence[Mapping[str, str]],
    macrocycle_minimum: int,
) -> dict[str, Any]:
    role_fields = {
        "amine_head": "A_smiles",
        "aldehyde_component": "B_smiles",
        "isocyanide_component": "C_smiles",
    }
    role_summaries = {}
    role_records: dict[str, list[dict[str, Any]]] = {}
    for role, field in role_fields.items():
        summary, records = _aggregate_signatures(
            rows,
            smiles_field=field,
            label=f"AGILE {role}",
            macrocycle_minimum=macrocycle_minimum,
        )
        role_summaries[role] = summary
        role_records[role] = records
    product_summary, product_records = _aggregate_signatures(
        rows,
        smiles_field="model_smiles",
        label="AGILE product",
        macrocycle_minimum=macrocycle_minimum,
    )
    localization = Counter()
    product_head_ring_match = 0
    for index, product in enumerate(product_records):
        cyclic_roles = tuple(
            role for role in role_fields if role_records[role][index]["ring_count"] > 0
        )
        localization["acyclic" if not cyclic_roles else "+".join(cyclic_roles)] += 1
        product_head_ring_match += int(
            product["ring_sizes"] == role_records["amine_head"][index]["ring_sizes"]
            and role_records["aldehyde_component"][index]["ring_count"] == 0
            and role_records["isocyanide_component"][index]["ring_count"] == 0
        )
    return {
        "product": product_summary,
        "components": role_summaries,
        "ring_localization_counts": dict(sorted(localization.items())),
        "product_ring_sizes_match_amine_head_rows": product_head_ring_match,
    }


def _validate_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    policy = config.get("ring_policy")
    if not isinstance(policy, dict):
        raise RingSupportAuditError("ring_policy must be an object")
    required = {
        "macrocycle_minimum_ring_size": 9,
        "primary_ugi_automatic_ring_roles": ["amine_head"],
        "primary_ugi_automatic_single_ring_sizes": [5, 6],
        "primary_ugi_automatic_maximum_rings": 1,
        "primary_ugi_automatic_fused_allowed": False,
        "primary_ugi_automatic_spiro_allowed": False,
        "primary_ugi_automatic_bridgehead_allowed": False,
        "primary_ugi_automatic_macrocycle_allowed": False,
        "exact_ring_fragment_vocabulary_required": False,
        "outside_automatic_support_action": (
            "abstain_from_primary_candidate_lock_pending_exact_evidence"
        ),
    }
    if policy != required:
        raise RingSupportAuditError("ring_policy changed from the frozen contract")
    return dict(policy)


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_ring_support_audit(
    config_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run and persist the deterministic ring-support audit."""

    config = _load_json(config_path, "ring-support config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise RingSupportAuditError("unsupported ring-support config schema")
    if config.get("seed") != 20260730:
        raise RingSupportAuditError("ring-support seed must remain 20260730")
    policy = _validate_policy(config)
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "r0",
        "agile_curated",
        "agile_reconciliation",
    }:
        raise RingSupportAuditError("ring-support inputs are incomplete")
    verified = {}
    paths = {}
    for name, specification in sorted(inputs.items()):
        if not isinstance(specification, Mapping):
            raise RingSupportAuditError(f"{name} input specification is invalid")
        path = repo_root / str(specification.get("path", ""))
        observed = sha256_file(path)
        if observed != specification.get("sha256"):
            raise RingSupportAuditError(
                f"{name} hash mismatch: expected {specification.get('sha256')}, observed {observed}"
            )
        paths[name] = path
        verified[name] = {
            "path": _portable(path, repo_root),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    r0 = _read_csv(
        paths["r0"],
        {"r0_structure_id", "canonical_isomeric_smiles", "observed_source_ids"},
        "R0",
    )
    agile = _read_csv(
        paths["agile_curated"],
        {"label", "model_smiles", "A_smiles", "B_smiles", "C_smiles"},
        "reconciled AGILE",
    )
    expected = config.get("expected")
    if not isinstance(expected, Mapping):
        raise RingSupportAuditError("ring-support expected counts are absent")
    if len(r0) != expected.get("r0_rows") or len(agile) != expected.get(
        "agile_single_structure_rows"
    ):
        raise RingSupportAuditError("ring-support input row count changed")
    reconciliation = _load_json(paths["agile_reconciliation"], "AGILE reconciliation")
    if reconciliation.get("summary", {}).get("curated_single_structure_records") != len(agile):
        raise RingSupportAuditError("AGILE reconciliation does not authenticate curated rows")
    macrocycle_minimum = int(policy["macrocycle_minimum_ring_size"])
    r0_summary, _ = _aggregate_signatures(
        r0,
        smiles_field="canonical_isomeric_smiles",
        label="R0",
        macrocycle_minimum=macrocycle_minimum,
        source_field="observed_source_ids",
    )
    agile_audit = _agile_role_audit(agile, macrocycle_minimum)
    automatic_sizes = policy["primary_ugi_automatic_single_ring_sizes"]
    agile_product = agile_audit["product"]
    automatic_gate_supported = (
        set(int(size) for size in agile_product["ring_size_instance_counts"])
        == set(automatic_sizes)
        and agile_product["multiple_ring_records"] == 0
        and agile_product["macrocycle_records"] == 0
        and agile_product["fused_ring_records"] == 0
        and agile_product["spiro_ring_records"] == 0
        and agile_product["bridgehead_ring_records"] == 0
        and agile_audit["components"]["aldehyde_component"]["cyclic_records"] == 0
        and agile_audit["components"]["isocyanide_component"]["cyclic_records"] == 0
        and agile_audit["product_ring_sizes_match_amine_head_rows"] == len(agile)
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed_lipid_native_ring_support_audit",
        "scope": "M0 support policy only; no production product-prior training",
        "inputs": verified,
        "configuration": {
            "path": _portable(config_path, repo_root),
            "sha256": sha256_file(config_path),
            "seed": config["seed"],
        },
        "definitions": {
            "rings": "RDKit symmetrized smallest set of smallest rings",
            "macrocycle": f"ring size at least {macrocycle_minimum}",
            "fused": "two perceived rings share at least two atoms",
            "spiro": "RDKit spiro atom count greater than zero",
            "bridgehead": "RDKit bridgehead atom count greater than zero",
        },
        "broad_r0": r0_summary,
        "reconciled_agile_ugi": agile_audit,
        "decision": {
            "primary_ugi_automatic_ring_gate_supported": automatic_gate_supported,
            "primary_ugi_automatic_support": {
                "acyclic_products": True,
                "ring_roles": policy["primary_ugi_automatic_ring_roles"],
                "single_ring_sizes": automatic_sizes,
                "maximum_rings": policy["primary_ugi_automatic_maximum_rings"],
                "fused_allowed": policy["primary_ugi_automatic_fused_allowed"],
                "spiro_allowed": policy["primary_ugi_automatic_spiro_allowed"],
                "bridgehead_allowed": policy["primary_ugi_automatic_bridgehead_allowed"],
                "macrocycle_allowed": policy["primary_ugi_automatic_macrocycle_allowed"],
            },
            "exact_ring_fragment_vocabulary_required": policy[
                "exact_ring_fragment_vocabulary_required"
            ],
            "outside_automatic_support_action": policy["outside_automatic_support_action"],
            "broad_prior_rule": (
                "retain experimentally observed ring topologies for representation "
                "learning without granting primary Ugi candidate-lock support"
            ),
            "evidence_override_rule": (
                "a nonautomatic ring topology requires exact source or prospective "
                "assembly evidence, complete L2/L3 closure, and an explicit support "
                "amendment before primary candidate lock"
            ),
        },
        "limitations": [
            (
                "The broad R0 census uses the corrected constitutional corpus and "
                "does not represent stereochemical distinctions."
            ),
            (
                "The Ugi campaign gate is authenticated by the corrected 1,100-record "
                "AGILE single-structure set and is unaffected by B4 mixture exclusion."
            ),
            (
                "Ring-topology compatibility does not establish synthesis, biological "
                "activity, oracle applicability, or in-vivo performance."
            ),
        ],
        "runtime": {
            "python": sys.version,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    if not automatic_gate_supported:
        raise RingSupportAuditError(
            "reconciled AGILE ring evidence does not support the frozen automatic gate"
        )
    _atomic_write_json(output_path, result)
    return result
