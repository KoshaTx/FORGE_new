"""Permutation-stability audit for the canonical sparse lipid representation."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

from forge.model.defog_feasibility import AtomState, FeasibilityError, sha256_file
from forge.model.vocabulary import AtomVocabularyError
from forge.model.vocabulary import load_atom_vocabulary as _load_atom_vocabulary
from forge.model.v5_sparse_representation import (
    V5SparseGraphRecord,
    tensorize_v5_sparse_molecule,
    v5_constitutional_roundtrip_exact,
    v5_sparse_program_valid,
)

_IMPLEMENTATION_PATHS = {
    "canonical_representation_audit": Path(__file__).resolve(),
    "v5_sparse_representation": Path(__file__).resolve().parents[4]
    / "forge/model/v5_sparse_representation.py",
    "offspring_tree_decoder": Path(__file__).resolve().parents[4]
    / "forge/model/phase1_tree_topology_flow.py",
}


class CanonicalRepresentationAuditError(RuntimeError):
    """Raised when the canonical-representation audit contract is invalid."""


_ENCODED_ARRAYS = (
    "node_states",
    "offspring",
    "parent_bonds",
    "closure_left",
    "closure_right",
    "closure_bonds",
    "region_states",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CanonicalRepresentationAuditError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CanonicalRepresentationAuditError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CanonicalRepresentationAuditError(f"{label} must be a JSON object")
    return payload


def load_atom_vocabulary(path: Path) -> tuple[AtomState, ...]:
    """Load and validate the frozen Phase 1 atom-state vocabulary."""

    try:
        return _load_atom_vocabulary(path)
    except AtomVocabularyError as error:
        raise CanonicalRepresentationAuditError(str(error)) from error


def sparse_record_signature(record: V5SparseGraphRecord) -> str:
    """Hash every encoded array while excluding record identity and source text."""

    digest = hashlib.sha256()
    digest.update(b"tree_traversal")
    digest.update(record.tree_traversal.encode())
    for name in _ENCODED_ARRAYS:
        value = getattr(record, name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
            continue
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _row_seed(seed: int, structure_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{structure_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _random_nonidentity_order(node_count: int, rng: np.random.Generator) -> list[int]:
    order = np.arange(node_count, dtype=np.int64)
    rng.shuffle(order)
    if node_count > 1 and np.array_equal(order, np.arange(node_count)):
        order = np.roll(order, 1)
    return order.tolist()


def _size_bucket(node_count: int) -> str:
    if node_count <= 40:
        return "le40"
    if node_count <= 64:
        return "41_64"
    if node_count <= 96:
        return "65_96"
    if node_count <= 128:
        return "97_128"
    return "gt128"


def _new_group() -> Counter[str]:
    return Counter(records=0, permutation_encodings=0, exact_permutation_encodings=0)


def _summarize_group(counts: Mapping[str, int]) -> dict[str, Any]:
    permutations = int(counts.get("permutation_encodings", 0))
    exact = int(counts.get("exact_permutation_encodings", 0))
    return {
        "records": int(counts.get("records", 0)),
        "permutation_encodings": permutations,
        "exact_permutation_encodings": exact,
        "exact_permutation_fraction": exact / permutations if permutations else None,
    }


def _has_graph_symmetry(molecule: Chem.Mol) -> bool:
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=False))
    return len(set(ranks)) < len(ranks)


def audit_canonical_representation(
    r0_path: Path,
    atom_vocabulary_path: Path,
    config_path: Path,
    *,
    seed: int,
    permutations_per_record: int,
    tree_traversal: str = "breadth_first_tree_preorder",
    maximum_failure_examples: int = 20,
) -> dict[str, Any]:
    """Audit exact reconstruction and serialized invariance over the full R0 corpus."""

    if seed < 0:
        raise CanonicalRepresentationAuditError("seed must be nonnegative")
    if permutations_per_record < 1:
        raise CanonicalRepresentationAuditError("permutations_per_record must be positive")
    if maximum_failure_examples < 1:
        raise CanonicalRepresentationAuditError("maximum_failure_examples must be positive")
    for path, label in (
        (r0_path, "R0"),
        (atom_vocabulary_path, "atom vocabulary"),
        (config_path, "product config"),
    ):
        if not path.is_file():
            raise CanonicalRepresentationAuditError(f"{label} not found: {path}")

    config = _load_json(config_path, "product config")
    model = config.get("model")
    if not isinstance(model, dict):
        raise CanonicalRepresentationAuditError("product config is missing model settings")
    preserve_aromaticity = bool(model.get("preserve_aromaticity", False))
    root_strategy = str(model.get("root_strategy", "canonical"))
    region_scheme = str(model.get("region_scheme", "none"))
    region_classes = int(model["region_classes"])
    maximum_children = int(model["degree_prior_maximum_degree"])
    maximum_heavy_atoms = int(model["maximum_heavy_atoms"])
    maximum_closures = int(model["maximum_closure_slots"])
    vocabulary = load_atom_vocabulary(atom_vocabulary_path)
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}

    started = time.monotonic()
    counts: Counter[str] = Counter()
    size_groups: defaultdict[str, Counter[str]] = defaultdict(_new_group)
    symmetry_groups: defaultdict[str, Counter[str]] = defaultdict(_new_group)
    failure_types: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []

    opener = gzip.open if r0_path.suffix == ".gz" else open
    with opener(r0_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            structure_id = str(row["r0_structure_id"])
            smiles = str(row["canonical_isomeric_smiles"])
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise CanonicalRepresentationAuditError(f"invalid R0 SMILES: {structure_id}")
            node_count = molecule.GetNumHeavyAtoms()
            size_bucket = _size_bucket(node_count)
            symmetry_group = "symmetric" if _has_graph_symmetry(molecule) else "asymmetric"
            size_groups[size_bucket]["records"] += 1
            symmetry_groups[symmetry_group]["records"] += 1
            counts["records"] += 1
            counts["symmetric_records"] += int(symmetry_group == "symmetric")
            counts["maximum_heavy_atoms"] = max(counts["maximum_heavy_atoms"], node_count)

            base = tensorize_v5_sparse_molecule(
                molecule,
                atom_to_index,
                structure_id=structure_id,
                canonical_smiles=smiles,
                preserve_aromaticity=preserve_aromaticity,
                root_strategy=root_strategy,
                region_scheme=region_scheme,
                tree_traversal=tree_traversal,
            )
            counts["maximum_closures"] = max(counts["maximum_closures"], base.closure_count)
            sparse_valid = v5_sparse_program_valid(
                base,
                atom_vocabulary_size=len(vocabulary),
                region_classes=region_classes,
                maximum_children=maximum_children,
            )
            exact_constitution = v5_constitutional_roundtrip_exact(base, vocabulary)
            counts["valid_sparse_programs"] += int(sparse_valid)
            counts["exact_constitutional_roundtrips"] += int(exact_constitution)
            base_signature = sparse_record_signature(base)

            repeated = tensorize_v5_sparse_molecule(
                molecule,
                atom_to_index,
                structure_id=structure_id,
                canonical_smiles=smiles,
                preserve_aromaticity=preserve_aromaticity,
                root_strategy=root_strategy,
                region_scheme=region_scheme,
                tree_traversal=tree_traversal,
            )
            repeated_exact = sparse_record_signature(repeated) == base_signature
            counts["exact_repeated_encodings"] += int(repeated_exact)
            if not sparse_valid:
                failure_types["base_sparse_program"] += 1
            if not exact_constitution:
                failure_types["base_constitutional_roundtrip"] += 1
            if not repeated_exact:
                failure_types["repeated_encoding"] += 1

            rng = np.random.default_rng(_row_seed(seed, structure_id))
            for permutation_index in range(permutations_per_record):
                order = _random_nonidentity_order(molecule.GetNumAtoms(), rng)
                permuted_molecule = Chem.RenumberAtoms(molecule, order)
                permuted = tensorize_v5_sparse_molecule(
                    permuted_molecule,
                    atom_to_index,
                    structure_id=structure_id,
                    canonical_smiles=smiles,
                    preserve_aromaticity=preserve_aromaticity,
                    root_strategy=root_strategy,
                    region_scheme=region_scheme,
                    tree_traversal=tree_traversal,
                )
                signature = sparse_record_signature(permuted)
                exact = signature == base_signature
                graph_equivalent = v5_constitutional_roundtrip_exact(permuted, vocabulary)
                counts["permutation_encodings"] += 1
                counts["exact_permutation_encodings"] += int(exact)
                counts["graph_equivalent_permutation_encodings"] += int(graph_equivalent)
                size_groups[size_bucket]["permutation_encodings"] += 1
                size_groups[size_bucket]["exact_permutation_encodings"] += int(exact)
                symmetry_groups[symmetry_group]["permutation_encodings"] += 1
                symmetry_groups[symmetry_group]["exact_permutation_encodings"] += int(exact)
                if not graph_equivalent:
                    failure_types["permuted_constitutional_roundtrip"] += 1
                if not exact:
                    failure_types["permutation_serialization"] += 1
                    if len(failure_examples) < maximum_failure_examples:
                        failure_examples.append(
                            {
                                "structure_id": structure_id,
                                "canonical_isomeric_smiles": smiles,
                                "permutation_index": permutation_index,
                                "symmetry_group": symmetry_group,
                                "base_signature": base_signature,
                                "permuted_signature": signature,
                                "graph_equivalent": graph_equivalent,
                            }
                        )

    if counts["records"] == 0:
        raise CanonicalRepresentationAuditError("R0 contains no records")
    support_pass = (
        counts["maximum_heavy_atoms"] <= maximum_heavy_atoms
        and counts["maximum_closures"] <= maximum_closures
    )
    acceptance = {
        "every_record_is_a_valid_sparse_program": (
            counts["valid_sparse_programs"] == counts["records"]
        ),
        "every_record_constitutionally_roundtrips": (
            counts["exact_constitutional_roundtrips"] == counts["records"]
        ),
        "repeated_encoding_is_exact": counts["exact_repeated_encodings"] == counts["records"],
        "all_random_atom_permutations_serialize_exactly": (
            counts["exact_permutation_encodings"] == counts["permutation_encodings"]
        ),
        "all_random_atom_permutations_remain_graph_equivalent": (
            counts["graph_equivalent_permutation_encodings"] == counts["permutation_encodings"]
        ),
        "declared_size_and_closure_support_is_preserved": support_pass,
    }
    status = "pass" if all(acceptance.values()) else "fail"
    elapsed = time.monotonic() - started
    return {
        "schema_version": "phase1_v5_canonical_representation_audit.v2",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "r0_constitutional": {
                "path": str(r0_path),
                "sha256": sha256_file(r0_path),
            },
            "atom_vocabulary": {
                "path": str(atom_vocabulary_path),
                "sha256": sha256_file(atom_vocabulary_path),
            },
            "product_config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
        },
        "implementation": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(_IMPLEMENTATION_PATHS.items())
        },
        "policy": {
            "seed": seed,
            "permutations_per_record": permutations_per_record,
            "permutation_rng": "numpy.PCG64 seeded by sha256(seed:structure_id)",
            "preserve_aromaticity": preserve_aromaticity,
            "root_strategy": root_strategy,
            "region_scheme": region_scheme,
            "tree_traversal": tree_traversal,
            "maximum_heavy_atoms": maximum_heavy_atoms,
            "maximum_closures": maximum_closures,
            "maximum_children": maximum_children,
            "region_classes": region_classes,
            "signature_arrays": list(_ENCODED_ARRAYS),
            "stereochemistry": "excluded_from_product_graph_state",
        },
        "environment": {
            "rdkit_version": rdBase.rdkitVersion,
            "numpy_version": np.__version__,
        },
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "size_strata": {key: _summarize_group(value) for key, value in sorted(size_groups.items())},
        "symmetry_strata": {
            key: _summarize_group(value) for key, value in sorted(symmetry_groups.items())
        },
        "failure_counts": {key: int(value) for key, value in sorted(failure_types.items())},
        "failure_examples": failure_examples,
        "acceptance": acceptance,
        "runtime": {"elapsed_seconds": elapsed},
        "decision": (
            "selected_hybrid_canonicalization_passes_full_representation_gate"
            if status == "pass"
            else "representation_requires_revision_before_v5_training"
        ),
    }


def write_audit(path: Path, result: Mapping[str, Any]) -> None:
    """Atomically persist a canonical-representation audit."""

    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
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


def require_pass(result: Mapping[str, Any]) -> None:
    """Fail a CLI invocation after preserving a negative audit artifact."""

    if result.get("status") != "pass":
        raise FeasibilityError(
            "canonical sparse representation is not invariant under the declared gate"
        )
