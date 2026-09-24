"""Corpus-wide audit of the Ugi topology-to-chemistry training boundary."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.archive.phase1.design_audits.canonical_representation_audit import load_atom_vocabulary
from forge.model.defog_feasibility import AtomState, sha256_file
from forge.model.ugi_adapter_features import tensorize_ugi_l1_support_record
from forge.model.ugi_chemistry_interface import (
    WITHHELD_STATE,
    ChemistryTopologyCondition,
    UgiFixedCoreSchema,
    assemble_ugi_chemistry_topology_condition,
    core_schema_from_record,
    materialize_chemistry_target,
    project_chemistry_topology_condition,
    recompute_adapter_distances,
)
from forge.model.ugi_morphology_program import split_ugi_support_morphology


class UgiChemistryInterfaceAuditError(RuntimeError):
    """Raised when the chemistry boundary cannot be audited exactly."""


_IMPLEMENTATION_PATHS = {
    "ugi_chemistry_interface": Path(__file__).with_name("ugi_chemistry_interface.py").resolve(),
    "ugi_chemistry_interface_audit": Path(__file__).resolve(),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise UgiChemistryInterfaceAuditError(f"could not read {path}: {exc}") from exc


def _state_label(state: AtomState) -> str:
    return json.dumps(state.key(), separators=(",", ":"))


def _schema_payload(
    schema: UgiFixedCoreSchema,
    vocabulary: tuple[AtomState, ...],
) -> dict[str, Any]:
    return {
        "atom_state_indices_by_core_position": list(schema.atom_state_by_core_position),
        "atom_states_by_core_position": [
            None if state < 0 else list(vocabulary[state].key())
            for state in schema.atom_state_by_core_position
        ],
        "bond_states_by_core_position_pair": [
            list(row) for row in schema.bond_state_by_core_positions
        ],
    }


def _condition_field_names() -> set[str]:
    return set(ChemistryTopologyCondition.__dataclass_fields__)


def _conditions_equal(
    left: ChemistryTopologyCondition,
    right: ChemistryTopologyCondition,
) -> bool:
    for field in ChemistryTopologyCondition.__dataclass_fields__:
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if isinstance(left_value, np.ndarray):
            if not np.array_equal(left_value, right_value):
                return False
        elif left_value != right_value:
            return False
    return True


def audit_ugi_chemistry_interface(
    products_path: Path,
    atoms_path: Path,
    atom_vocabulary_path: Path,
) -> dict[str, Any]:
    """Project every Ugi record and prove the condition/target separation."""

    for path in (products_path, atoms_path, atom_vocabulary_path):
        if not path.is_file():
            raise UgiChemistryInterfaceAuditError(f"required input is absent: {path}")
    products = _read_csv(products_path)
    atoms_by_product: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(atoms_path):
        atoms_by_product[row["product_id"]].append(row)
    vocabulary = load_atom_vocabulary(atom_vocabulary_path)
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    counts: Counter[str] = Counter()
    decoration_atoms: Counter[str] = Counter()
    decoration_bonds: Counter[str] = Counter()
    decoration_anchor_origins: Counter[str] = Counter()
    maximum_decorations_per_product = 0
    maximum_decorations_per_anchor = 0
    schema: UgiFixedCoreSchema | None = None

    for product in products:
        product_id = product["product_id"]
        record = tensorize_ugi_l1_support_record(
            product,
            atoms_by_product.get(product_id, []),
            atom_to_index,
            preserve_aromaticity=True,
        )
        observed_schema = core_schema_from_record(record)
        if schema is None:
            schema = observed_schema
        counts["core_schema_matches"] += int(observed_schema == schema)
        condition = project_chemistry_topology_condition(record, schema)
        try:
            component_keys = json.loads(product["component_smiles_json"])
        except json.JSONDecodeError as exc:
            raise UgiChemistryInterfaceAuditError(
                f"invalid component provenance for {product_id}"
            ) from exc
        morphology = split_ugi_support_morphology(
            record,
            component_keys,
            product_id=product_id,
        )
        assembled = assemble_ugi_chemistry_topology_condition(
            structure_id=condition.structure_id,
            offspring_by_role={
                component.role: component.offspring for component in morphology.components
            },
            attachment_counts_by_role={
                component.role: component.attachment_count for component in morphology.components
            },
            closure_left_by_role={
                component.role: component.closure_left for component in morphology.components
            },
            closure_right_by_role={
                component.role: component.closure_right for component in morphology.components
            },
            schema=schema,
        )
        distances = recompute_adapter_distances(condition)
        target = materialize_chemistry_target(record, atom_to_index)
        counts["products"] += 1
        counts["conditions_projected"] += 1
        counts["targets_materialized"] += 1
        counts["generated_exteriors_stitch_exactly"] += int(_conditions_equal(condition, assembled))
        counts["five_fixed_core_atoms"] += int(condition.fixed_atom_mask.sum() == 5)
        counts["noncore_atom_states_withheld"] += int(
            np.all(condition.fixed_atom_states[~condition.fixed_atom_mask] == WITHHELD_STATE)
        )
        counts["noncore_parent_bonds_withheld"] += int(
            np.all(
                condition.fixed_parent_bond_states[~condition.fixed_parent_bond_mask]
                == WITHHELD_STATE
            )
        )
        counts["noncore_closure_bonds_withheld"] += int(
            np.all(
                condition.fixed_closure_bond_states[~condition.fixed_closure_bond_mask]
                == WITHHELD_STATE
            )
        )
        counts["core_distances_recomputed_exactly"] += int(
            np.array_equal(distances.distance_to_core, record.support_adapter.distance_to_core)
        )
        counts["own_port_distances_recomputed_exactly"] += int(
            np.array_equal(
                distances.distance_to_own_port,
                record.support_adapter.distance_to_own_port,
            )
        )
        counts["all_port_distances_recomputed_exactly"] += int(
            np.array_equal(
                distances.distances_to_all_ports,
                record.support_adapter.distances_to_all_ports,
            )
        )
        counts["target_atom_shapes_exact"] += int(
            target.atom_states.shape == (condition.node_count,)
        )
        counts["target_parent_bond_shapes_exact"] += int(
            target.parent_bond_states.shape == (condition.node_count,)
        )
        counts["target_closure_bond_shapes_exact"] += int(
            target.closure_bond_states.shape == (condition.closure_count,)
        )
        aromatic_atoms = sum(vocabulary[int(index)].aromatic for index in target.atom_states)
        aromatic_bonds = int(np.count_nonzero(target.parent_bond_states == 3)) + int(
            np.count_nonzero(target.closure_bond_states == 3)
        )
        counts["products_with_aromatic_atoms"] += int(aromatic_atoms > 0)
        counts["aromatic_support_atoms"] += aromatic_atoms
        counts["aromatic_support_bonds"] += aromatic_bonds

        decoration_count = target.decorations.count
        counts["terminal_decorations"] += decoration_count
        counts["removed_atoms_expected"] += record.skeleton.removed_count
        counts["products_with_terminal_decorations"] += int(decoration_count > 0)
        maximum_decorations_per_product = max(maximum_decorations_per_product, decoration_count)
        anchor_counts = Counter(target.decorations.anchor_indices.tolist())
        maximum_decorations_per_anchor = max(
            maximum_decorations_per_anchor,
            max(anchor_counts.values(), default=0),
        )
        for anchor, atom_state, bond_state in zip(
            target.decorations.anchor_indices,
            target.decorations.atom_states,
            target.decorations.bond_states,
            strict=True,
        ):
            decoration_atoms[_state_label(vocabulary[int(atom_state)])] += 1
            decoration_bonds[str(int(bond_state))] += 1
            decoration_anchor_origins[str(int(condition.origin_states[int(anchor)]))] += 1

    if schema is None:
        raise UgiChemistryInterfaceAuditError("Ugi chemistry corpus is empty")
    product_count = len(products)
    exact_per_product = (
        "core_schema_matches",
        "conditions_projected",
        "targets_materialized",
        "generated_exteriors_stitch_exactly",
        "five_fixed_core_atoms",
        "noncore_atom_states_withheld",
        "noncore_parent_bonds_withheld",
        "noncore_closure_bonds_withheld",
        "core_distances_recomputed_exactly",
        "own_port_distances_recomputed_exactly",
        "all_port_distances_recomputed_exactly",
        "target_atom_shapes_exact",
        "target_parent_bond_shapes_exact",
        "target_closure_bond_shapes_exact",
    )
    forbidden_fields = {
        "node_states",
        "parent_bonds",
        "closure_bonds",
        "distance_to_core",
        "distance_to_own_port",
        "distances_to_all_ports",
    }
    condition_fields = _condition_field_names()
    gates = {
        "all_products_audited": counts["products"] == product_count,
        "all_per_product_gates_pass": all(
            counts[key] == product_count for key in exact_per_product
        ),
        "condition_dataclass_excludes_target_channels": not (condition_fields & forbidden_fields),
        "aromaticity_is_explicitly_present": counts["aromatic_support_atoms"] > 0
        and counts["aromatic_support_bonds"] > 0,
        "decorations_are_terminal_single_atom_targets": counts["terminal_decorations"]
        == counts["removed_atoms_expected"],
    }
    return {
        "schema_version": "phase1_ugi_chemistry_interface_audit.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "semantic_products": {"path": str(products_path), "sha256": sha256_file(products_path)},
            "semantic_atoms": {"path": str(atoms_path), "sha256": sha256_file(atoms_path)},
            "atom_vocabulary": {
                "path": str(atom_vocabulary_path),
                "sha256": sha256_file(atom_vocabulary_path),
            },
        },
        "implementation": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(_IMPLEMENTATION_PATHS.items())
        },
        "counts": dict(sorted(counts.items())),
        "maximum_decorations_per_product": maximum_decorations_per_product,
        "maximum_decorations_per_anchor": maximum_decorations_per_anchor,
        "decoration_atom_states": dict(sorted(decoration_atoms.items())),
        "decoration_bond_states": dict(sorted(decoration_bonds.items())),
        "decoration_anchor_origin_states": dict(sorted(decoration_anchor_origins.items())),
        "fixed_core_schema": _schema_payload(schema, vocabulary),
        "condition_fields": sorted(condition_fields),
        "gates": gates,
        "policy": {
            "fixed_chemistry": "adapter_owned_five_atom_ugi_core_only",
            "withheld_chemistry": "all_noncore_atom_and_bond_states",
            "positional_features": "recomputed_from_current_generated_topology",
            "aromaticity": "explicit_atom_and_bond_states",
            "decorations": "terminal_atom_targets_outside_functional_support",
            "component_catalog_ids_in_model_state": False,
        },
    }
