"""Deterministic, metadata-isolated tensor cache for M0-07 graph oracles.

The cache is a ZIP archive of NumPy ``.npy`` members written with fixed
timestamps and loaded with ``allow_pickle=False``. It contains numeric graph
tensors only. Labels and source identifiers live in a separate gzip CSV ledger
and are never exposed by the tensor-cache loader.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge.bio.oracle_graph import (
    GraphFeatureVocabulary,
    GraphTensor,
    tensorize_smiles,
)
from forge.core.hashing import sha256_file as _sha256_file

CONFIG_SCHEMA_VERSION = "m0_07_oracle_graph_cache_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_graph_cache.v1"
ARCHIVE_FILENAME = "oracle_graph_tensor_cache.npz"
METADATA_FILENAME = "oracle_graph_tensor_cache_metadata.csv.gz"
RESULT_FILENAME = "oracle_graph_tensor_cache_result.json"

COLLECTIONS = (
    "oracle_product",
    "oracle_amine",
    "oracle_aldehyde",
    "oracle_isocyanide",
    "r0_pretraining_product",
    "virtual_applicability_product",
)
ARRAY_FIELDS = (
    "node_features",
    "edge_index",
    "edge_features",
    "reverse_edge_index",
    "graph_ptr",
    "edge_ptr",
)
METADATA_FIELDS = ("collection", "record_index", "record_key")
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class OracleGraphCacheError(ValueError):
    """Raised when the graph tensor cache violates its frozen contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleGraphCacheError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleGraphCacheError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise OracleGraphCacheError(f"{label} must contain a JSON object")
    return value


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _verify_input(
    repo_root: Path,
    specification: Mapping[str, Any],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise OracleGraphCacheError(f"{label} input specification is incomplete")
    path = repo_root / relative
    if not path.is_file():
        raise OracleGraphCacheError(f"{label} not found: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise OracleGraphCacheError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return path, {
        "path": _portable(path, repo_root),
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def _read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise OracleGraphCacheError(f"{label} is missing fields: {sorted(missing)}")
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise OracleGraphCacheError(f"{label} not found: {path}") from exc


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode())
    digest.update(b"\x00")
    digest.update(json.dumps(list(canonical.shape)).encode())
    digest.update(b"\x00")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PackedGraphArrays:
    """One numeric ragged-graph collection."""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    reverse_edge_index: np.ndarray
    graph_ptr: np.ndarray
    edge_ptr: np.ndarray

    @property
    def graph_count(self) -> int:
        return int(self.graph_ptr.size - 1)

    @property
    def atom_count(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def directed_edge_count(self) -> int:
        return int(self.edge_features.shape[0])

    def validate(
        self,
        *,
        atom_feature_dim: int | None = None,
        bond_feature_dim: int | None = None,
        label: str = "graph pack",
    ) -> None:
        arrays = {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
            "reverse_edge_index": self.reverse_edge_index,
            "graph_ptr": self.graph_ptr,
            "edge_ptr": self.edge_ptr,
        }
        if any(array.dtype.hasobject for array in arrays.values()):
            raise OracleGraphCacheError(f"{label} contains an object array")
        if self.node_features.dtype != np.float32 or self.node_features.ndim != 2:
            raise OracleGraphCacheError(f"{label} node features must be float32 rank two")
        if self.edge_index.dtype != np.int64 or self.edge_index.shape != (
            2,
            self.directed_edge_count,
        ):
            raise OracleGraphCacheError(f"{label} edge index has an invalid shape or dtype")
        if self.edge_features.dtype != np.float32 or self.edge_features.ndim != 2:
            raise OracleGraphCacheError(f"{label} edge features must be float32 rank two")
        if self.reverse_edge_index.dtype != np.int64 or self.reverse_edge_index.shape != (
            self.directed_edge_count,
        ):
            raise OracleGraphCacheError(f"{label} reverse-edge index has an invalid shape or dtype")
        for name, pointer, expected_end in (
            ("graph_ptr", self.graph_ptr, self.atom_count),
            ("edge_ptr", self.edge_ptr, self.directed_edge_count),
        ):
            if pointer.dtype != np.int64 or pointer.ndim != 1 or pointer.size < 2:
                raise OracleGraphCacheError(f"{label} {name} has an invalid shape or dtype")
            if pointer[0] != 0 or pointer[-1] != expected_end or np.any(np.diff(pointer) < 0):
                raise OracleGraphCacheError(f"{label} {name} is not a valid monotone pointer")
        if self.graph_ptr.size != self.edge_ptr.size:
            raise OracleGraphCacheError(f"{label} node and edge pointer counts differ")
        if np.any(np.diff(self.graph_ptr) <= 0):
            raise OracleGraphCacheError(f"{label} contains an empty graph")
        if atom_feature_dim is not None and self.node_features.shape[1] != atom_feature_dim:
            raise OracleGraphCacheError(f"{label} atom feature dimension changed")
        if bond_feature_dim is not None and self.edge_features.shape[1] != bond_feature_dim:
            raise OracleGraphCacheError(f"{label} bond feature dimension changed")
        if self.directed_edge_count:
            if (
                np.min(self.edge_index) < 0
                or np.max(self.edge_index) >= self.atom_count
                or np.min(self.reverse_edge_index) < 0
                or np.max(self.reverse_edge_index) >= self.directed_edge_count
            ):
                raise OracleGraphCacheError(f"{label} contains an out-of-bounds edge index")
            if not np.array_equal(
                self.reverse_edge_index[self.reverse_edge_index],
                np.arange(self.directed_edge_count, dtype=np.int64),
            ):
                raise OracleGraphCacheError(f"{label} reverse-edge mapping is not an involution")
        for index in range(self.graph_count):
            node_start, node_end = self.graph_ptr[index : index + 2]
            edge_start, edge_end = self.edge_ptr[index : index + 2]
            local_edges = self.edge_index[:, edge_start:edge_end]
            local_reverse = self.reverse_edge_index[edge_start:edge_end]
            if local_edges.size and (
                np.min(local_edges) < node_start or np.max(local_edges) >= node_end
            ):
                raise OracleGraphCacheError(f"{label} graph {index} crosses a node boundary")
            if local_reverse.size and (
                np.min(local_reverse) < edge_start or np.max(local_reverse) >= edge_end
            ):
                raise OracleGraphCacheError(f"{label} graph {index} crosses an edge boundary")

    def graph(self, index: int) -> GraphTensor:
        """Recover one local graph without exposing metadata."""

        if index < 0 or index >= self.graph_count:
            raise OracleGraphCacheError(f"graph index {index} is out of range")
        node_start, node_end = (int(value) for value in self.graph_ptr[index : index + 2])
        edge_start, edge_end = (int(value) for value in self.edge_ptr[index : index + 2])
        return GraphTensor(
            node_features=torch.from_numpy(self.node_features[node_start:node_end].copy()),
            edge_index=torch.from_numpy(
                (self.edge_index[:, edge_start:edge_end] - node_start).copy()
            ),
            edge_features=torch.from_numpy(self.edge_features[edge_start:edge_end].copy()),
            reverse_edge_index=torch.from_numpy(
                (self.reverse_edge_index[edge_start:edge_end] - edge_start).copy()
            ),
        )


def pack_graphs(graphs: Sequence[GraphTensor], *, label: str) -> PackedGraphArrays:
    """Pack unpadded graphs into deterministic global arrays."""

    if not graphs:
        raise OracleGraphCacheError(f"{label} contains no graphs")
    node_parts: list[np.ndarray] = []
    edge_index_parts: list[np.ndarray] = []
    edge_feature_parts: list[np.ndarray] = []
    reverse_parts: list[np.ndarray] = []
    graph_ptr = [0]
    edge_ptr = [0]
    node_offset = 0
    edge_offset = 0
    for graph in graphs:
        node = (
            graph.node_features.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        )
        edge_index = (
            graph.edge_index.detach().cpu().contiguous().numpy().astype(np.int64, copy=False)
        )
        edge_feature = (
            graph.edge_features.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        )
        reverse = (
            graph.reverse_edge_index.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.int64, copy=False)
        )
        node_parts.append(node)
        edge_index_parts.append(edge_index + node_offset)
        edge_feature_parts.append(edge_feature)
        reverse_parts.append(reverse + edge_offset)
        node_offset += graph.num_nodes
        edge_offset += graph.num_directed_edges
        graph_ptr.append(node_offset)
        edge_ptr.append(edge_offset)
    packed = PackedGraphArrays(
        node_features=np.ascontiguousarray(np.concatenate(node_parts, axis=0), dtype=np.float32),
        edge_index=np.ascontiguousarray(np.concatenate(edge_index_parts, axis=1), dtype=np.int64),
        edge_features=np.ascontiguousarray(
            np.concatenate(edge_feature_parts, axis=0),
            dtype=np.float32,
        ),
        reverse_edge_index=np.ascontiguousarray(
            np.concatenate(reverse_parts, axis=0),
            dtype=np.int64,
        ),
        graph_ptr=np.asarray(graph_ptr, dtype=np.int64),
        edge_ptr=np.asarray(edge_ptr, dtype=np.int64),
    )
    packed.validate(label=label)
    return packed


def _npy_payload(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(
        output,
        np.ascontiguousarray(array),
        allow_pickle=False,
    )
    return output.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name, array in sorted(arrays.items()):
            if array.dtype.hasobject:
                raise OracleGraphCacheError(f"archive array {name} has object dtype")
            member = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_ZIP_TIMESTAMP)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(
                member,
                _npy_payload(array),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def load_graph_cache(path: Path) -> dict[str, PackedGraphArrays]:
    """Load and validate the safe numeric archive with pickle disabled."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            expected = {
                f"{collection}__{field}" for collection in COLLECTIONS for field in ARRAY_FIELDS
            }
            if keys != expected:
                raise OracleGraphCacheError(
                    f"cache array names changed: missing={sorted(expected - keys)}, "
                    f"extra={sorted(keys - expected)}"
                )
            raw = {name: np.ascontiguousarray(archive[name]) for name in sorted(archive.files)}
    except (FileNotFoundError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise OracleGraphCacheError(f"cannot load safe graph cache {path}: {exc}") from exc
    output = {}
    for collection in COLLECTIONS:
        values = {field: raw[f"{collection}__{field}"] for field in ARRAY_FIELDS}
        pack = PackedGraphArrays(**values)
        pack.validate(label=collection)
        output[collection] = pack
    return output


def _metadata_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=METADATA_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(
        rows,
        key=lambda value: (str(value["collection"]), int(value["record_index"])),
    ):
        writer.writerow({field: row[field] for field in METADATA_FIELDS})
    return gzip.compress(text.getvalue().encode(), compresslevel=9, mtime=0)


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _array_sha256(array),
        }
        for name, array in sorted(arrays.items())
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _tensorize_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    smiles_field: str,
    key_field: str,
    vocabulary: GraphFeatureVocabulary,
    label: str,
) -> list[GraphTensor]:
    output = []
    for index, row in enumerate(rows):
        output.append(
            tensorize_smiles(
                row[smiles_field],
                vocabulary,
                label=f"{label} row {index} {row[key_field]}",
            )
        )
    return output


def _validate_upstream(
    config: Mapping[str, Any],
    graph_corpus: Mapping[str, Any],
    graph_profile: Mapping[str, Any],
) -> None:
    if graph_corpus.get("schema_version") != "m0_07_oracle_graph_corpus.v2":
        raise OracleGraphCacheError("unsupported upstream graph-corpus result schema")
    if graph_corpus.get("decision", {}).get("graph_tensorization_authorized") is not True:
        raise OracleGraphCacheError("upstream graph corpus did not authorize tensorization")
    if graph_corpus.get("representation", {}).get("truncation_allowed") is not False:
        raise OracleGraphCacheError("upstream graph corpus allows truncation")
    if graph_corpus.get("pretraining_policy", {}).get("biological_labels_used") is not False:
        raise OracleGraphCacheError("upstream pretraining corpus used biological labels")
    if graph_corpus.get("pretraining_policy", {}).get("virtual_candidates_used") is not False:
        raise OracleGraphCacheError("upstream pretraining corpus used virtual candidates")
    if graph_profile.get("schema_version") != "m0_07_oracle_graph_profile.v1":
        raise OracleGraphCacheError("unsupported upstream graph-profile result schema")
    if graph_profile.get("status") != "passed_train_only_graph_runtime_gate":
        raise OracleGraphCacheError("upstream graph runtime gate did not pass")
    decision = graph_profile.get("decision", {})
    if (
        decision.get("full_supervised_graph_matrix_authorized") is not True
        or decision.get("calibration_or_test_labels_used") is not False
    ):
        raise OracleGraphCacheError("upstream graph profile violates the train-only gate")
    expected = config["expected"]
    tensorization = graph_profile.get("tensorization", {})
    required_profiles = {
        "curated_product_and_roles": (
            expected["curated_records"],
            expected["curated_graphs"],
            expected["curated_atoms"],
            expected["curated_directed_edges"],
            expected["curated_maximum_atoms"],
        ),
        "r0_pretraining_products": (
            expected["r0_pretraining_records"],
            expected["r0_pretraining_records"],
            expected["r0_pretraining_atoms"],
            expected["r0_pretraining_directed_edges"],
            expected["r0_pretraining_maximum_atoms"],
        ),
        "virtual_applicability_products": (
            expected["virtual_records"],
            expected["virtual_records"],
            expected["virtual_atoms"],
            expected["virtual_directed_edges"],
            expected["virtual_maximum_atoms"],
        ),
    }
    for name, expected_values in required_profiles.items():
        observed = tensorization.get(name, {})
        observed_values = (
            observed.get("records"),
            observed.get("graphs"),
            observed.get("atoms"),
            observed.get("directed_edges"),
            observed.get("maximum_atoms"),
        )
        if observed_values != expected_values:
            raise OracleGraphCacheError(
                f"upstream profile {name} changed: {observed_values} versus {expected_values}"
            )


def build_oracle_graph_cache(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build and atomically persist the deterministic graph tensor cache."""

    config = _load_json(config_path, "graph-cache config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleGraphCacheError("unsupported graph-cache config schema")
    if config.get("seed") != 1729:
        raise OracleGraphCacheError("graph-cache seed must remain 1729")
    policy = config.get("policy")
    required_policy = {
        "archive_format": "deterministic_npz_numeric_npy_members",
        "pickle_allowed": False,
        "archive_contains_targets": False,
        "archive_contains_identifiers": False,
        "metadata_stored_separately": True,
        "truncation_allowed": False,
        "source_row_order_preserved": True,
    }
    if policy != required_policy:
        raise OracleGraphCacheError("graph-cache policy changed")
    if tuple(config.get("collections", ())) != COLLECTIONS:
        raise OracleGraphCacheError("graph-cache collection order changed")

    input_specifications = config.get("inputs")
    if not isinstance(input_specifications, dict):
        raise OracleGraphCacheError("graph-cache config lacks inputs")
    expected_inputs = {
        "curated_oracle_data",
        "graph_corpus_result",
        "graph_profile_result",
        "r0_pretraining_ledger",
        "virtual_candidate_library",
    }
    if set(input_specifications) != expected_inputs:
        raise OracleGraphCacheError("graph-cache input set changed")
    paths: dict[str, Path] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for name, specification in sorted(input_specifications.items()):
        if not isinstance(specification, dict):
            raise OracleGraphCacheError(f"input {name} specification is malformed")
        path, record = _verify_input(repo_root, specification, name)
        paths[name] = path
        inputs[name] = record

    graph_corpus = _load_json(paths["graph_corpus_result"], "graph-corpus result")
    graph_profile = _load_json(paths["graph_profile_result"], "graph-profile result")
    _validate_upstream(config, graph_corpus, graph_profile)
    vocabulary = GraphFeatureVocabulary.from_corpus_result(paths["graph_corpus_result"])

    oracle_rows = _read_csv(
        paths["curated_oracle_data"],
        {
            "label",
            "model_smiles",
            "A_smiles",
            "B_smiles",
            "C_smiles",
            "expt_Hela",
            "expt_Raw",
        },
        "curated oracle data",
    )
    r0_rows = _read_csv(
        paths["r0_pretraining_ledger"],
        {
            "graph_id",
            "constitutional_smiles",
            "atom_count",
            "undirected_bond_count",
            "directed_edge_count",
        },
        "R0 pretraining ledger",
    )
    virtual_rows = _read_csv(
        paths["virtual_candidate_library"],
        {"source_row_index", "canonical_isomeric_smiles"},
        "virtual candidate library",
    )
    expected = config["expected"]
    observed_counts = {
        "curated_records": len(oracle_rows),
        "r0_pretraining_records": len(r0_rows),
        "virtual_records": len(virtual_rows),
    }
    expected_counts = {name: int(expected[name]) for name in observed_counts}
    if observed_counts != expected_counts:
        raise OracleGraphCacheError(
            f"graph-cache source counts changed: {observed_counts} versus {expected_counts}"
        )
    labels = [row["label"] for row in oracle_rows]
    r0_keys = [row["graph_id"] for row in r0_rows]
    virtual_keys = [row["source_row_index"] for row in virtual_rows]
    for name, values in (
        ("oracle labels", labels),
        ("R0 graph IDs", r0_keys),
        ("virtual row indices", virtual_keys),
    ):
        if len(values) != len(set(values)):
            raise OracleGraphCacheError(f"{name} are not unique")

    graph_lists = {
        "oracle_product": _tensorize_rows(
            oracle_rows,
            smiles_field="model_smiles",
            key_field="label",
            vocabulary=vocabulary,
            label="oracle product",
        ),
        "oracle_amine": _tensorize_rows(
            oracle_rows,
            smiles_field="A_smiles",
            key_field="label",
            vocabulary=vocabulary,
            label="oracle amine",
        ),
        "oracle_aldehyde": _tensorize_rows(
            oracle_rows,
            smiles_field="B_smiles",
            key_field="label",
            vocabulary=vocabulary,
            label="oracle aldehyde",
        ),
        "oracle_isocyanide": _tensorize_rows(
            oracle_rows,
            smiles_field="C_smiles",
            key_field="label",
            vocabulary=vocabulary,
            label="oracle isocyanide",
        ),
        "r0_pretraining_product": _tensorize_rows(
            r0_rows,
            smiles_field="constitutional_smiles",
            key_field="graph_id",
            vocabulary=vocabulary,
            label="R0 pretraining",
        ),
        "virtual_applicability_product": _tensorize_rows(
            virtual_rows,
            smiles_field="canonical_isomeric_smiles",
            key_field="source_row_index",
            vocabulary=vocabulary,
            label="virtual applicability",
        ),
    }
    packs = {name: pack_graphs(graph_lists[name], label=name) for name in COLLECTIONS}
    for pack in packs.values():
        pack.validate(
            atom_feature_dim=vocabulary.atom_feature_dim,
            bond_feature_dim=vocabulary.bond_feature_dim,
        )

    curated_atoms = sum(packs[name].atom_count for name in COLLECTIONS[:4])
    curated_edges = sum(packs[name].directed_edge_count for name in COLLECTIONS[:4])
    observed_tensor_counts = {
        "curated_graphs": sum(packs[name].graph_count for name in COLLECTIONS[:4]),
        "curated_atoms": curated_atoms,
        "curated_directed_edges": curated_edges,
        "r0_pretraining_atoms": packs["r0_pretraining_product"].atom_count,
        "r0_pretraining_directed_edges": packs["r0_pretraining_product"].directed_edge_count,
        "virtual_atoms": packs["virtual_applicability_product"].atom_count,
        "virtual_directed_edges": packs["virtual_applicability_product"].directed_edge_count,
    }
    expected_tensor_counts = {name: int(expected[name]) for name in observed_tensor_counts}
    if observed_tensor_counts != expected_tensor_counts:
        raise OracleGraphCacheError(
            f"cached tensor counts changed: {observed_tensor_counts} "
            f"versus {expected_tensor_counts}"
        )

    arrays = {
        f"{collection}__{field}": getattr(pack, field)
        for collection, pack in packs.items()
        for field in ARRAY_FIELDS
    }
    metadata_rows = [
        {
            "collection": "oracle",
            "record_index": index,
            "record_key": label,
        }
        for index, label in enumerate(labels)
    ]
    metadata_rows.extend(
        {
            "collection": "r0_pretraining",
            "record_index": index,
            "record_key": key,
        }
        for index, key in enumerate(r0_keys)
    )
    metadata_rows.extend(
        {
            "collection": "virtual_applicability",
            "record_index": index,
            "record_key": key,
        }
        for index, key in enumerate(virtual_keys)
    )
    metadata_payload = _metadata_payload(metadata_rows)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.graph-cache.", dir=output_dir.parent)
    )
    try:
        archive_path = staging / ARCHIVE_FILENAME
        metadata_path = staging / METADATA_FILENAME
        result_path = staging / RESULT_FILENAME
        _write_deterministic_npz(archive_path, arrays)
        metadata_path.write_bytes(metadata_payload)
        roundtrip = load_graph_cache(archive_path)
        for name in COLLECTIONS:
            original = packs[name]
            loaded = roundtrip[name]
            for field in ARRAY_FIELDS:
                if not np.array_equal(getattr(original, field), getattr(loaded, field)):
                    raise OracleGraphCacheError(f"{name}/{field} changed during cache round trip")

        config_record = {
            "path": _portable(config_path, repo_root),
            "sha256": _sha256_file(config_path),
            "bytes": config_path.stat().st_size,
        }
        artifacts = {
            ARCHIVE_FILENAME: _artifact_record(archive_path),
            METADATA_FILENAME: _artifact_record(metadata_path),
        }
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task": "M0-07",
            "status": "graph_tensor_cache_complete",
            "generated_utc": config["generated_utc"],
            "seed": config["seed"],
            "configuration": config_record,
            "inputs": inputs,
            "policy": policy,
            "feature_vocabulary": vocabulary.to_dict(),
            "feature_dimensions": {
                "atom": vocabulary.atom_feature_dim,
                "bond": vocabulary.bond_feature_dim,
            },
            "summary": {
                **observed_counts,
                **observed_tensor_counts,
                "archive_arrays": len(arrays),
                "metadata_rows": len(metadata_rows),
            },
            "collections": {
                name: {
                    "graphs": pack.graph_count,
                    "atoms": pack.atom_count,
                    "directed_edges": pack.directed_edge_count,
                    "maximum_atoms": int(np.max(np.diff(pack.graph_ptr))),
                    "maximum_directed_edges": int(np.max(np.diff(pack.edge_ptr))),
                }
                for name, pack in packs.items()
            },
            "array_manifest": _array_manifest(arrays),
            "artifacts": artifacts,
            "decision": {
                "cache_authorized_for_numeric_graph_inputs": True,
                "cache_authorized_for_targets_or_identifiers": False,
                "roundtrip_exact": True,
                "oracle_model_frozen": False,
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
        }
        result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        result_path.write_bytes(result_payload)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(staging.iterdir()):
            os.replace(path, output_dir / path.name)
    finally:
        try:
            staging.rmdir()
        except FileNotFoundError:
            pass
    return result
