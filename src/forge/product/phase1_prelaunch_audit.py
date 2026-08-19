"""Audit every auxiliary product through the exact Phase 1 representation."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem

from forge.core.hashing import sha256_file
from forge.product.defog_feasibility import AtomState, FeasibilityError
from forge.product.phase1_flow import _load_json, _read_csv, _size_bucket
from forge.product.sparse_topology_feasibility import (
    SPARSE_BOND_TO_INDEX,
    _kekulized_molecule,
    build_sparse_atom_vocabulary,
)

CONFIG_SCHEMA_VERSION = "phase1_product_prelaunch_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_product_prelaunch_audit.v1"


class Phase1PrelaunchAuditError(ValueError):
    """Raised when the prelaunch audit contract or an input is invalid."""


def _resolve_and_verify(repo: Path, specification: Mapping[str, Any], label: str) -> Path:
    if not {"path", "sha256"}.issubset(specification):
        raise Phase1PrelaunchAuditError(f"{label} is missing path or sha256")
    relative = Path(str(specification["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise Phase1PrelaunchAuditError(f"{label} path must remain repository-relative")
    path = (repo / relative).resolve()
    if not path.is_relative_to(repo.resolve()) or not path.is_file():
        raise Phase1PrelaunchAuditError(f"{label} not found inside repository: {path}")
    observed = sha256_file(path)
    if observed != specification["sha256"]:
        raise Phase1PrelaunchAuditError(
            f"{label} SHA-256 mismatch: expected {specification['sha256']}, observed {observed}"
        )
    return path


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Phase1PrelaunchAuditError("unsupported prelaunch-audit config schema")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"phase1_manifest"}:
        raise Phase1PrelaunchAuditError("prelaunch-audit inputs changed")
    expected = config.get("expected")
    if not isinstance(expected, dict):
        raise Phase1PrelaunchAuditError("prelaunch-audit expectations are missing")
    if (
        int(expected.get("r1_rows", 0)) <= 0
        or int(expected.get("maximum_heavy_atoms", 0)) != 282
        or int(expected.get("maximum_closure_slots", 0)) != 12
        or int(expected.get("required_components_per_product", 0)) != 1
        or float(expected.get("minimum_realism_weight", -1.0)) != 0.0
        or expected.get("required_size_bin_match") is not True
        or expected.get("required_r1_state_subset_of_r0") is not True
    ):
        raise Phase1PrelaunchAuditError("prelaunch-audit scientific contract changed")


def audit_r1_rows(
    rows: Iterable[Mapping[str, str]],
    atom_vocabulary: Sequence[AtomState],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic support census for streamed R1 rows."""

    vocabulary = set(atom_vocabulary)
    observed_states: Counter[tuple[str, int, bool]] = Counter()
    unsupported_states: Counter[tuple[str, int, bool]] = Counter()
    size_bins: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    maximum_heavy_atoms = 0
    maximum_closures = 0
    total_rows = 0

    def record_failure(kind: str, row_number: int, value: Any) -> None:
        failures[kind] += 1
        bucket = examples.setdefault(kind, [])
        if len(bucket) < 10:
            bucket.append({"row_number": row_number, "value": value})

    for row_number, row in enumerate(rows, start=1):
        total_rows += 1
        smiles = row.get("canonical_smiles", "")
        try:
            molecule = _kekulized_molecule(smiles)
        except FeasibilityError:
            record_failure("invalid_or_non_kekulizable_smiles", row_number, smiles)
            continue

        components = len(Chem.GetMolFrags(molecule))
        if components != int(expected["required_components_per_product"]):
            record_failure("component_count_mismatch", row_number, components)
        heavy_atoms = molecule.GetNumHeavyAtoms()
        closures = molecule.GetNumBonds() - heavy_atoms + components
        maximum_heavy_atoms = max(maximum_heavy_atoms, heavy_atoms)
        maximum_closures = max(maximum_closures, closures)
        if heavy_atoms > int(expected["maximum_heavy_atoms"]):
            record_failure("heavy_atom_support_exceeded", row_number, heavy_atoms)
        if closures > int(expected["maximum_closure_slots"]):
            record_failure("closure_support_exceeded", row_number, closures)

        declared_size_bin = row.get("size_bin", "")
        observed_size_bin = _size_bucket(heavy_atoms)
        size_bins[observed_size_bin] += 1
        if declared_size_bin != observed_size_bin:
            record_failure(
                "size_bin_mismatch",
                row_number,
                {"declared": declared_size_bin, "observed": observed_size_bin},
            )

        try:
            realism_weight = float(row.get("realism_weight", "nan"))
        except ValueError:
            realism_weight = math.nan
        if not math.isfinite(realism_weight) or realism_weight < float(
            expected["minimum_realism_weight"]
        ):
            record_failure("invalid_realism_weight", row_number, row.get("realism_weight"))

        for atom in molecule.GetAtoms():
            state = AtomState(atom.GetSymbol(), atom.GetFormalCharge(), False)
            key = (state.symbol, state.formal_charge, state.aromatic)
            observed_states[key] += 1
            if state not in vocabulary:
                unsupported_states[key] += 1
        for bond in molecule.GetBonds():
            if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
                record_failure(
                    "unsupported_bond_type",
                    row_number,
                    str(bond.GetBondType()),
                )
                break

    if total_rows != int(expected["r1_rows"]):
        failures["row_count_mismatch"] += 1
        examples["row_count_mismatch"] = [
            {
                "expected": int(expected["r1_rows"]),
                "observed": total_rows,
            }
        ]
    if unsupported_states:
        failures["unsupported_atom_state"] = sum(unsupported_states.values())
        examples["unsupported_atom_state"] = [
            {"state": list(key), "atom_occurrences": count}
            for key, count in sorted(unsupported_states.items())
        ]

    return {
        "rows": total_rows,
        "maximum_heavy_atoms": maximum_heavy_atoms,
        "maximum_closures": maximum_closures,
        "size_bin_counts": dict(sorted(size_bins.items())),
        "observed_atom_states": [
            {
                "symbol": key[0],
                "formal_charge": key[1],
                "aromatic": key[2],
                "atom_occurrences": count,
            }
            for key, count in sorted(observed_states.items())
        ],
        "unsupported_atom_states": [
            {
                "symbol": key[0],
                "formal_charge": key[1],
                "aromatic": key[2],
                "atom_occurrences": count,
            }
            for key, count in sorted(unsupported_states.items())
        ],
        "failure_counts": dict(sorted(failures.items())),
        "failure_examples": dict(sorted(examples.items())),
        "all_rows_supported": not failures,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def run_prelaunch_audit(
    config_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Audit pinned R0/R1 inputs and persist a deterministic result."""

    repo = repo_root.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, "Phase 1 prelaunch-audit config")
    _validate_config(config)
    manifest_path = _resolve_and_verify(
        repo,
        config["inputs"]["phase1_manifest"],
        "Phase 1 data manifest",
    )
    manifest = _load_json(manifest_path, "Phase 1 data manifest")
    r0_path = _resolve_and_verify(repo, manifest["inputs"]["r0_constitutional"], "R0")
    r1_path = _resolve_and_verify(
        repo,
        manifest["inputs"]["r1_reaction_enumerated"],
        "R1",
    )
    declared_elements = set(manifest["policy"]["representation"]["elements"])
    r0_rows = _read_csv(r0_path)
    vocabulary = build_sparse_atom_vocabulary(r0_rows, declared_elements)
    with r1_path.open(newline="") as handle:
        audit = audit_r1_rows(
            csv.DictReader(handle),
            vocabulary,
            config["expected"],
        )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "pass" if audit["all_rows_supported"] else "fail",
        "task": str(config["task"]),
        "inputs": {
            "config": {
                "path": str(config_path.relative_to(repo)),
                "sha256": sha256_file(config_path),
                "bytes": config_path.stat().st_size,
            },
            "phase1_manifest": {
                **config["inputs"]["phase1_manifest"],
                "bytes": manifest_path.stat().st_size,
            },
            "r0_constitutional": {
                **manifest["inputs"]["r0_constitutional"],
                "bytes": r0_path.stat().st_size,
            },
            "r1_reaction_enumerated": {
                **manifest["inputs"]["r1_reaction_enumerated"],
                "bytes": r1_path.stat().st_size,
            },
        },
        "representation": {
            "declared_elements": sorted(declared_elements),
            "r0_atom_vocabulary": [
                {
                    "symbol": state.symbol,
                    "formal_charge": state.formal_charge,
                    "aromatic": state.aromatic,
                }
                for state in vocabulary
            ],
            "maximum_heavy_atoms": int(config["expected"]["maximum_heavy_atoms"]),
            "maximum_closure_slots": int(config["expected"]["maximum_closure_slots"]),
            "aromaticity_policy": "kekulized_bonds_with_cleared_atom_aromatic_flags",
        },
        "r1_audit": audit,
        "decision": {
            "gpu_training_authorized_by_representation_audit": audit["all_rows_supported"],
            "rows_may_not_be_dropped_to_pass": True,
        },
    }
    _atomic_write_json(output_path.resolve(), result)
    return result
