"""Audit Ugi model support ceilings against exact and transferable components."""

from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.product.defog_feasibility import sha256_file
from forge.product.lipid_support_skeleton import (
    FUNCTIONAL_SUPPORT,
    encode_lipid_support_skeleton,
    skeleton_statistics,
)
from forge.product.ugi_training_cache import load_ugi_training_cache


class UgiSupportBoundsAuditError(RuntimeError):
    """Raised when a support-bound audit input or invariant is invalid."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, value: Any) -> None:
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


def _resolve_inputs(
    config: Mapping[str, Any], repo: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths = {}
    records = {}
    for label, value in config["inputs"].items():
        path = Path(value["path"])
        if not path.is_absolute():
            path = repo / path
        if not path.is_file():
            raise UgiSupportBoundsAuditError(f"missing {label}: {path}")
        observed = sha256_file(path)
        if observed != value["sha256"]:
            raise UgiSupportBoundsAuditError(f"{label} hash changed")
        paths[label] = path
        records[label] = {
            "path": str(path.relative_to(repo)),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return paths, records


def _quantiles(values: Sequence[int | float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "q00": float(array.min()),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "q100": float(array.max()),
    }


def component_capacity_metrics(
    smiles: str,
    *,
    structure_id: str,
    bounds: Mapping[str, int],
) -> dict[str, Any]:
    """Measure conservative support-skeleton capacity for one source component.

    The source component has not necessarily been converted into a Ugi reactant.
    Its complete functional-support skeleton is therefore a conservative size
    proxy: exact Ugi projection may move handle atoms into the fixed product core.
    """

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise UgiSupportBoundsAuditError(f"invalid component SMILES: {structure_id}")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiSupportBoundsAuditError(f"component is disconnected: {structure_id}")
    encoding = encode_lipid_support_skeleton(
        molecule,
        structure_id=structure_id,
        variant=FUNCTIONAL_SUPPORT,
    )
    statistics = skeleton_statistics(encoding)
    retained = set(encoding.retained_atoms)
    adjacency = {index: set() for index in retained}
    for left, right, _ in encoding.skeleton_bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    if not adjacency or not statistics["connected"]:
        raise UgiSupportBoundsAuditError(f"functional support is disconnected: {structure_id}")
    ranks = list(Chem.CanonicalRankAtoms(Chem.MolFromSmiles(encoding.canonical_smiles)))
    root = min(retained, key=lambda index: (ranks[index], index))
    parents = {root: -1}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current], key=lambda index: (ranks[index], index)):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            queue.append(neighbor)
    if len(parents) != len(retained):
        raise UgiSupportBoundsAuditError(f"tree traversal lost atoms: {structure_id}")
    children = Counter(parent for parent in parents.values() if parent >= 0)
    tree_degrees = {index: children[index] + int(parents[index] >= 0) for index in retained}
    junction_budget = sum(max(0, degree - 2) for degree in tree_degrees.values())
    metrics = {
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "support_atoms": int(encoding.retained_count),
        "terminal_decorations": int(encoding.removed_count),
        "maximum_children": max(children.values(), default=0),
        "junction_budget": int(junction_budget),
        "cycle_rank": int(statistics["cycle_rank"]),
    }
    reasons = []
    for metric, bound_name in (
        ("support_atoms", "maximum_component_atoms"),
        ("maximum_children", "maximum_children"),
        ("junction_budget", "maximum_junction_budget"),
        ("cycle_rank", "maximum_cycle_rank"),
    ):
        if metrics[metric] > bounds[bound_name]:
            reasons.append(metric)
    return {
        **metrics,
        "capacity_supported": not reasons,
        "failure_reasons": reasons,
    }


def _exact_component_metrics(
    cache_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, int]], dict[str, Any]]:
    corpus, records_by_fold = load_ugi_training_cache(cache_path)
    components: dict[tuple[str, str], dict[str, int]] = {}
    product_count = 0
    for fold, records in records_by_fold.items():
        assignments = corpus.assignments_by_fold[fold]
        if len(records) != len(assignments):
            raise UgiSupportBoundsAuditError("cache records and assignments disagree")
        for record, assignment in zip(records, assignments, strict=True):
            product_count += 1
            start = 0
            for role_index, role in enumerate(ROLE_NAMES):
                stop = start + int(record.program.node_counts[role_index])
                anchors = int(
                    np.count_nonzero(
                        (record.decoration_anchors >= start) & (record.decoration_anchors < stop)
                    )
                )
                value = {
                    "support_atoms": int(record.program.node_counts[role_index]),
                    "maximum_children": int(record.offspring[start:stop].max(initial=0)),
                    "junction_budget": int(record.program.junction_budgets[role_index]),
                    "cycle_rank": int(record.program.cycle_ranks[role_index]),
                    "attachment_count": int(record.program.attachment_counts[role_index]),
                    "terminal_decorations": anchors,
                }
                key = (role, str(assignment[f"{role}_smiles"]))
                previous = components.setdefault(key, value)
                if previous != value:
                    raise UgiSupportBoundsAuditError(
                        f"component morphology changes across products: {role}: {key[1]}"
                    )
                start = stop
    return components, {"cache_products": product_count, "unique_components": len(components)}


def _coverage_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    weight_field: str | None = None,
) -> dict[str, Any]:
    if not rows:
        return {"records": 0}
    weights = [float(row[weight_field]) if weight_field else 1.0 for row in rows]
    total_weight = sum(weights)
    passing_weight = sum(
        weight for row, weight in zip(rows, weights, strict=True) if row["capacity_supported"]
    )
    failures = Counter(reason for row in rows for reason in row.get("failure_reasons", ()))
    return {
        "records": len(rows),
        "capacity_supported": sum(bool(row["capacity_supported"]) for row in rows),
        "capacity_supported_fraction": sum(bool(row["capacity_supported"]) for row in rows)
        / len(rows),
        "weighted_capacity_supported_fraction": passing_weight / total_weight,
        "failure_reasons": dict(sorted(failures.items())),
        "metrics": {
            metric: _quantiles([int(row[metric]) for row in rows])
            for metric in (
                "support_atoms",
                "maximum_children",
                "junction_budget",
                "cycle_rank",
                "terminal_decorations",
            )
        },
    }


def _product_capacity_summary(
    assignments_path: Path,
    exact_components: Mapping[tuple[str, str], Mapping[str, int]],
    bounds: Mapping[str, int],
) -> dict[str, Any]:
    product_failures = Counter()
    product_count = 0
    failing_products = 0
    product_maxima = Counter()
    unsupported_examples = []
    opener = gzip.open if assignments_path.suffix == ".gz" else open
    with opener(assignments_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            values = [exact_components[(role, row[f"{role}_smiles"])] for role in ROLE_NAMES]
            product_count += 1
            total_atoms = sum(value["support_atoms"] for value in values)
            total_decorations = sum(value["terminal_decorations"] for value in values)
            product_maxima["total_exterior_atoms"] = max(
                product_maxima["total_exterior_atoms"], total_atoms
            )
            product_maxima["terminal_decorations"] = max(
                product_maxima["terminal_decorations"], total_decorations
            )
            reasons = []
            if total_atoms > bounds["maximum_total_atoms"]:
                product_failures["total_exterior_atoms"] += 1
                reasons.append("total_exterior_atoms")
            if total_decorations > bounds["maximum_decorations"]:
                product_failures["terminal_decorations"] += 1
                reasons.append("terminal_decorations")
            if reasons:
                failing_products += 1
            if reasons and len(unsupported_examples) < 25:
                unsupported_examples.append(
                    {
                        "product_id": row["product_id"],
                        "canonical_product_smiles": row["canonical_product_smiles"],
                        "component_smiles": {role: row[f"{role}_smiles"] for role in ROLE_NAMES},
                        "total_exterior_atoms": total_atoms,
                        "terminal_decorations": total_decorations,
                        "failure_reasons": reasons,
                    }
                )
    return {
        "products": product_count,
        "capacity_supported": product_count - failing_products,
        "capacity_supported_fraction": (product_count - failing_products) / product_count,
        "failure_reasons": dict(sorted(product_failures.items())),
        "observed_maxima": dict(product_maxima),
        "unsupported_examples": unsupported_examples,
    }


def audit_ugi_support_bounds(config_path: Path, repo: Path) -> dict[str, Any]:
    """Run the exact Ugi and conservative LNPDB component-capacity audit."""

    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "phase1_ugi_support_bounds_audit_config.v1":
        raise UgiSupportBoundsAuditError("unsupported support-bounds config")
    paths, inputs = _resolve_inputs(config, repo)
    model_config = json.loads(paths["model_config"].read_text())
    bounds = {
        name: int(model_config["model"][name])
        for name in (
            "maximum_component_atoms",
            "maximum_total_atoms",
            "maximum_children",
            "maximum_junction_budget",
            "maximum_cycle_rank",
            "maximum_attachment_count",
            "maximum_decorations",
        )
    }
    exact, cache_summary = _exact_component_metrics(paths["exact_training_cache"])
    registry_rows = _read_csv(paths["component_registry"])
    admitted = [row for row in registry_rows if row["l1_structural_admission"] == "true"]
    exact_rows = []
    for row in admitted:
        key = (row["role"], row["canonical_smiles"])
        if key not in exact:
            raise UgiSupportBoundsAuditError(f"admitted component absent from exact cache: {key}")
        metrics = exact[key]
        reasons = []
        for metric, bound_name in (
            ("support_atoms", "maximum_component_atoms"),
            ("maximum_children", "maximum_children"),
            ("junction_budget", "maximum_junction_budget"),
            ("cycle_rank", "maximum_cycle_rank"),
            ("attachment_count", "maximum_attachment_count"),
        ):
            if metrics[metric] > bounds[bound_name]:
                reasons.append(metric)
        exact_rows.append(
            {
                "component_id": row["component_id"],
                "role": row["role"],
                "canonical_smiles": row["canonical_smiles"],
                **metrics,
                "capacity_supported": not reasons,
                "failure_reasons": reasons,
            }
        )

    widened_products = _product_capacity_summary(paths["widened_assignments"], exact, bounds)
    full_expanded_products = _product_capacity_summary(
        paths["full_expanded_enumeration"], exact, bounds
    )

    hydrophobic_roles = set(config["policy"]["lnpdb_hydrophobic_roles"])
    lnpdb_components: dict[tuple[str, str], dict[str, Any]] = {}
    for lipid in _read_csv(paths["lnpdb_lipid_route_ledger"]):
        seen_in_lipid = set()
        for component in json.loads(lipid["component_evidence_json"] or "[]"):
            role = str(component.get("role") or "")
            smiles = str(component.get("canonical_smiles") or "")
            if (
                role not in hydrophobic_roles
                or component.get("parse_status") != "parsed"
                or not smiles
            ):
                continue
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise UgiSupportBoundsAuditError(
                    f"parsed LNPDB component became invalid: {component.get('component_id')}"
                )
            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
            key = (role, canonical)
            if key in seen_in_lipid:
                continue
            seen_in_lipid.add(key)
            if key not in lnpdb_components:
                lnpdb_components[key] = {
                    "component_id": component.get("component_id"),
                    "role": role,
                    "canonical_smiles": canonical,
                    "lipid_occurrences": 0,
                    **component_capacity_metrics(
                        canonical,
                        structure_id=str(component.get("component_id") or canonical),
                        bounds=bounds,
                    ),
                }
            lnpdb_components[key]["lipid_occurrences"] += 1

    transfer_rows = _read_csv(paths["hydrophobic_motif_transfer"])
    proposed = []
    deferred = []
    aldehyde_role = "oxoester_aldehyde_body_tail"
    for row in transfer_rows:
        smiles = row["proposed_ugi_component_smiles"]
        if not smiles:
            precursor = row["common_precursor_canonical_smiles"]
            deferred.append(
                {
                    "record_id": row["record_id"],
                    "disposition": row["disposition"],
                    "reason": row["defer_reason"],
                    "source_component_capacity": component_capacity_metrics(
                        precursor,
                        structure_id=row["record_id"],
                        bounds=bounds,
                    ),
                }
            )
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise UgiSupportBoundsAuditError(
                f"invalid proposed Ugi component in transfer ledger: {row['record_id']}"
            )
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        key = (aldehyde_role, canonical)
        exact_metrics = exact.get(key)
        proposed.append(
            {
                "record_id": row["record_id"],
                "canonical_smiles": canonical,
                "exact_projected_component": exact_metrics is not None,
                **(
                    {
                        **exact_metrics,
                        "capacity_supported": True,
                        "failure_reasons": [],
                    }
                    if exact_metrics is not None
                    else component_capacity_metrics(
                        canonical,
                        structure_id=row["record_id"],
                        bounds=bounds,
                    )
                ),
            }
        )

    by_role = {}
    for role in ROLE_NAMES:
        values = [row for row in exact_rows if row["role"] == role]
        by_role[role] = _coverage_summary(values)
    lnpdb_rows = list(lnpdb_components.values())
    result = {
        "schema_version": "phase1_ugi_support_bounds_audit_result.v1",
        "status": "pass" if widened_products["capacity_supported_fraction"] == 1.0 else "fail",
        "inputs": inputs,
        "policy": config["policy"],
        "bounds": bounds,
        "exact_ugi": {
            "cache": cache_summary,
            "admitted_components": _coverage_summary(exact_rows),
            "admitted_components_by_role": by_role,
            "widened_products": widened_products,
            "full_expanded_universe": full_expanded_products,
        },
        "lnpdb_hydrophobic_component_proxy": {
            "interpretation": config["policy"]["lnpdb_proxy_interpretation"],
            "all_roles": _coverage_summary(lnpdb_rows, weight_field="lipid_occurrences"),
            "by_role": {
                role: _coverage_summary(
                    [row for row in lnpdb_rows if row["role"] == role],
                    weight_field="lipid_occurrences",
                )
                for role in sorted(hydrophobic_roles)
            },
            "unsupported_examples": sorted(
                [row for row in lnpdb_rows if not row["capacity_supported"]],
                key=lambda row: (-row["lipid_occurrences"], row["component_id"]),
            )[:25],
        },
        "cross_platform_transfer_pilot": {
            "proposed_ugi_components": _coverage_summary(proposed),
            "exact_projected_components": sum(
                bool(row["exact_projected_component"]) for row in proposed
            ),
            "deferred_source_motifs": len(deferred),
            "deferred_for_capacity": sum(
                not row["source_component_capacity"]["capacity_supported"] for row in deferred
            ),
            "deferred_records": deferred,
        },
        "claim_boundary": {
            "supported": "capacity coverage of exact Ugi components and parsed source hydrophobic components",
            "not_supported": "automatic Ugi compatibility, route closure, biological transfer, or exhaustive LNPDB tail coverage",
        },
    }
    output_dir = repo / config["outputs"]["directory"]
    _atomic_json(output_dir / "result.json", result)
    return result
