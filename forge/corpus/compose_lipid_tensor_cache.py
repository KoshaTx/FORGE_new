"""Lossless prepared neural tensors for the complete, already-admitted TRAIN cohort."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge.core.hashing import resolve_pin
from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin

SCHEMA = "forge.compose_lipid_tensor_cache.v1"
NODE_FIELDS = frozenset(
    {
        "nodes",
        "parents",
        "parent_bonds",
        "node_mask",
        "child_mask",
        "role_states",
        "core_position_states",
        "component_instance_states",
        "component_position_states",
        "repeat_group_states",
        "role_morphology_states",
        "adapter_mask",
        "fixed_atom_mask",
        "fixed_parent_mask",
        "fixed_parent_bond_mask",
        "atom_variable_mask",
        "parent_variable_mask",
        "parent_bond_variable_mask",
        "repeat_atom_groups",
        "repeat_bond_groups",
    }
)
CLOSURE_FIELDS = frozenset(
    {
        "closure_left",
        "closure_right",
        "closure_bonds",
        "closure_mask",
        "fixed_closure_endpoint_mask",
        "fixed_closure_bond_mask",
        "closure_endpoint_variable_mask",
        "closure_bond_variable_mask",
    }
)
GRAPH_FIELDS = frozenset({"program_states", "program_depths", "family_states"})


def pack_tensors(clean: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Store integer values exactly, retaining closure slots and all conditioning fields."""
    if set(clean) != NODE_FIELDS | CLOSURE_FIELDS | GRAPH_FIELDS:
        raise ValueError("Prepared tensor field contract changed")
    mask = clean["node_mask"].numpy()
    lengths = mask.sum(1)
    if (
        not len(lengths)
        or np.any(lengths < 1)
        or not np.array_equal(mask, np.arange(mask.shape[1])[None, :] < lengths[:, None])
    ):
        raise ValueError("Expected nonempty prefix-contiguous node masks")
    result = {"offsets": np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)}
    for key, tensor in clean.items():
        array = tensor.numpy()
        if key in NODE_FIELDS:
            if array.shape[:2] != mask.shape or np.any(array[~mask] != 0):
                raise ValueError(f"Unexpected node padding: {key}")
            array = array[mask]
        elif array.shape[0] != len(lengths):
            raise ValueError(f"Record dimension changed: {key}")
        if array.dtype == np.int64:
            limits = np.iinfo(np.int32)
            if array.size and (array.min() < limits.min or array.max() > limits.max):
                raise ValueError(f"Lossless integer storage overflow: {key}")
            array = array.astype(np.int32)
        elif array.dtype != np.bool_:
            raise ValueError(f"Unexpected tensor dtype: {key}")
        result[key] = np.ascontiguousarray(array)
    return result


def write_tensor_shard(
    root: Path, output: Path, start: int, clean: Mapping[str, torch.Tensor]
) -> dict:
    """Publish a fully written shard; disjoint record ranges may be built concurrently."""
    arrays = pack_tensors(clean)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"shard-{start:09d}.npz"
    temporary = path.with_suffix(".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
    temporary.rename(path)
    record = dict(
        start=start,
        stop=start + len(clean["nodes"]),
        atoms=int(arrays["offsets"][-1]),
        maximum_atoms=int(np.diff(arrays["offsets"]).max()),
        maximum_closures=int(clean["closure_mask"].sum(1).max()),
        by_family=dict(sorted(Counter(map(int, arrays["family_states"])).items())),
        artifact=pin(root.resolve(), path),
    )
    write_json(path.with_suffix(".json"), record)
    return record


def finish_tensor_cache(
    root: Path,
    output: Path,
    *,
    shards: Sequence[dict],
    records: int,
    inputs: Mapping[str, Any],
    policy: Mapping[str, Any],
    expected_by_family: Mapping[int, int],
    implementation: Mapping[str, str],
) -> Path:
    """Publish completeness only after contiguous coverage and exact family counts pass."""
    ordered = sorted(shards, key=lambda row: row["start"])
    cursor, by_family = 0, Counter()
    for row in ordered:
        if row["start"] != cursor or row["stop"] <= cursor:
            raise ValueError("Tensor cache has missing, duplicate or overlapping records")
        resolve_pin(row["artifact"], root, label="complete tensor shard")
        cursor = row["stop"]
        by_family.update({int(k): v for k, v in row["by_family"].items()})
    if cursor != records or by_family != Counter(expected_by_family):
        raise ValueError("Tensor cache does not cover the complete admitted cohort")
    manifest = output / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    write_json(
        manifest,
        dict(
            schema_version=SCHEMA,
            complete=True,
            records=records,
            atoms=sum(row["atoms"] for row in ordered),
            by_family=dict(sorted(by_family.items())),
            maximum_atoms=max(row["maximum_atoms"] for row in ordered),
            maximum_closures=max(row["maximum_closures"] for row in ordered),
            shards=ordered,
            inputs=dict(inputs),
            policy=dict(policy),
            implementation=dict(implementation),
            training_admission_changed=False,
        ),
    )
    return manifest


class PreparedTensorCache:
    """Read-only full-cohort arrays in RAM; gather keeps duplicate draws and their order."""

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, str],
        *,
        inputs: Mapping[str, Any],
        policy: Mapping[str, Any],
        maximum_bytes: int = 16 * 1024**3,
    ):
        doc = json.loads(resolve_pin(manifest, root, label="prepared tensor manifest").read_text())
        if (
            doc.get("schema_version") != SCHEMA
            or doc.get("complete") is not True
            or doc["inputs"] != dict(inputs)
            or doc["policy"] != dict(policy)
        ):
            raise ValueError("Prepared cache identity, completeness or conditioning policy changed")
        self.metadata, self.arrays = doc, {}
        self.offsets = np.empty(doc["records"] + 1, dtype=np.int64)
        cursor = atoms = allocated = 0
        census = Counter()
        for row in doc["shards"]:
            if row["start"] != cursor or not cursor < row["stop"] <= doc["records"]:
                raise ValueError("Noncontiguous prepared tensor index")
            path = resolve_pin(row["artifact"], root, label="prepared tensors")
            with np.load(path, allow_pickle=False) as shard:
                if set(shard.files) != NODE_FIELDS | CLOSURE_FIELDS | GRAPH_FIELDS | {"offsets"}:
                    raise ValueError("Prepared shard tensor fields changed")
                offsets = shard["offsets"]
                count = row["stop"] - cursor
                if (
                    offsets.dtype != np.int64
                    or offsets.shape != (count + 1,)
                    or offsets[0] != 0
                    or offsets[-1] != row["atoms"]
                    or np.any(np.diff(offsets) < 1)
                ):
                    raise ValueError("Prepared shard offsets changed")
                self.offsets[cursor : row["stop"] + 1] = offsets + atoms
                for key in sorted(NODE_FIELDS | CLOSURE_FIELDS | GRAPH_FIELDS):
                    value = shard[key]
                    length = row["atoms"] if key in NODE_FIELDS else count
                    if (
                        value.dtype not in (np.dtype("int32"), np.dtype("bool"))
                        or value.shape[0] != length
                    ):
                        raise ValueError(f"Prepared shard shape/dtype changed: {key}")
                    if key not in self.arrays:
                        shape = (
                            doc["atoms"] if key in NODE_FIELDS else doc["records"],
                            *value.shape[1:],
                        )
                        allocated += int(np.prod(shape)) * value.dtype.itemsize
                        if allocated > maximum_bytes:
                            raise ValueError("Prepared cache exceeds explicit RAM budget")
                        self.arrays[key] = np.empty(shape, dtype=value.dtype)
                    destination = self.arrays[key]
                    if destination.dtype != value.dtype or destination.shape[1:] != value.shape[1:]:
                        raise ValueError(f"Inconsistent prepared shard schema: {key}")
                    offset = atoms if key in NODE_FIELDS else cursor
                    destination[offset : offset + length] = value
                census.update(map(int, shard["family_states"]))
            cursor = row["stop"]
            atoms += row["atoms"]
        if (
            cursor != doc["records"]
            or atoms != doc["atoms"]
            or census != Counter({int(k): v for k, v in doc["by_family"].items()})
        ):
            raise ValueError("Incomplete prepared corpus")
        for value in (self.offsets, *self.arrays.values()):
            value.flags.writeable = False

    def batch(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        positions = np.asarray(indices)
        if (
            positions.ndim != 1
            or not len(positions)
            or positions.dtype.kind not in "iu"
            or np.any(positions < 0)
            or np.any(positions >= self.metadata["records"])
        ):
            raise IndexError("Invalid prepared record indices")
        lengths = self.offsets[positions + 1] - self.offsets[positions]
        columns = np.arange(int(lengths.max()))[None, :]
        active = columns < lengths[:, None]
        gather = self.offsets[positions, None] + np.minimum(columns, lengths[:, None] - 1)
        result = {}
        for key, array in self.arrays.items():
            value = array[gather] if key in NODE_FIELDS else array[positions]
            if key in NODE_FIELDS:
                value[~active] = 0
            if value.dtype == np.int32:
                value = value.astype(np.int64)
            result[key] = torch.from_numpy(value)
        return result
