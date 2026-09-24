"""Streaming, memory-mapped preparation cache for verified source-aware graphs.

This is a storage optimization of QualifiedProgramCache, not training admission.
The compiler keeps the complete source population unless an explicit diagnostic
selection is supplied. It never creates sampling weights or fold assignments.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.qualified_program_cache import (
    QualifiedProgramCache,
    QualifiedProgramCacheError,
    QualifiedProgramExample,
    _vocabulary,
)
from forge.corpus.synthesis_program_production_cache import _decode_string, _encode_strings
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.sparse_topology_feasibility import SparseGraphRecord
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)
from forge.model.vocabulary import load_atom_vocabulary

_SCHEMA = "forge.mapped_program_preparation.v1"
_NODE_FIELDS = (
    "node_states",
    "parents",
    "parent_bonds",
    "canonical_atom_order",
    "role_states",
    "core_position_states",
    "fixed_atom_mask",
    "fixed_parent_bond_mask",
)
_CLOSURE_FIELDS = ("closure_left", "closure_right", "closure_bonds", "fixed_closure_bond_mask")
_BLOCK_FIELDS = ("block_role_states", "block_starts", "block_stops")
_GROUPS = {
    "node": _NODE_FIELDS,
    "closure": _CLOSURE_FIELDS,
    "block": _BLOCK_FIELDS,
    "record_id": ("record_id_bytes",),
    "smiles": ("smiles_bytes",),
}
_BYTE_FIELDS = {
    "parent_bonds",
    "closure_bonds",
    "fixed_atom_mask",
    "fixed_parent_bond_mask",
    "fixed_closure_bond_mask",
    "record_id_bytes",
    "smiles_bytes",
}
_DTYPES = {
    **{prefix + "_offsets": np.dtype("<i8") for prefix in _GROUPS},
    **{
        name: np.dtype("u1" if name in _BYTE_FIELDS else "<u2")
        for fields in _GROUPS.values()
        for name in fields
    },
    "program_states": np.dtype("<u2"),
    "program_depths": np.dtype("<u2"),
}


def _pin(repo: Path, path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(repo)), "sha256": str(sha256_file(path))}


def _cast(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    dtype = _DTYPES[name]
    if array.ndim != 1 or array.dtype.kind not in "uib":
        raise QualifiedProgramCacheError(f"Nonintegral mapped field: {name}")
    if array.size and (array.min() < 0 or array.max() > np.iinfo(dtype).max):
        raise QualifiedProgramCacheError(f"Mapped field would overflow: {name}")
    return array.astype(dtype, copy=False)


def _pack(examples: Sequence[QualifiedProgramExample]) -> dict[str, np.ndarray]:
    records = [e.record for e in examples]
    arrays = {}
    for prefix, counts in (
        ("node", [r.node_count for r in records]),
        ("closure", [r.graph.closure_count for r in records]),
        ("block", [len(r.component_blocks) for r in records]),
    ):
        arrays[prefix + "_offsets"] = np.cumsum([0, *counts], dtype=np.int64)
    for prefix, values in (
        ("record_id", [r.graph.structure_id for r in records]),
        ("smiles", [r.graph.canonical_smiles for r in records]),
    ):
        arrays[prefix + "_offsets"], arrays[prefix + "_bytes"] = _encode_strings(values)
    for name in (*_NODE_FIELDS, *_CLOSURE_FIELDS):
        values = [getattr(r, name) if hasattr(r, name) else getattr(r.graph, name) for r in records]
        arrays[name] = _cast(np.concatenate(values), name)
    for name, attribute in zip(_BLOCK_FIELDS, ("role_state", "start", "stop"), strict=True):
        arrays[name] = _cast(
            [getattr(b, attribute) for r in records for b in r.component_blocks], name
        )
    for name, attribute in (
        ("program_states", "program_state"),
        ("program_depths", "program_depth"),
    ):
        arrays[name] = _cast([getattr(r, attribute) for r in records], name)
    return arrays


def _header(handle: BinaryIO, dtype: np.dtype, count: int) -> int:
    handle.seek(0)
    np.lib.format.write_array_header_2_0(
        handle, {"descr": dtype.str, "fortran_order": False, "shape": (count,)}
    )
    return handle.tell()


def compile_qualified_program_cache(
    repo: Path,
    output: Path,
    *,
    population: Mapping[str, str],
    verification: Mapping[str, str],
    chunk_size: int = 256,
    target_ids: Sequence[str] | None = None,
) -> Path:
    """Atomically compile pinned source records, using at most one shard and one chunk.

    Explicit target_ids produce a diagnostic selection, never a full-population
    claim. Completed outputs cannot be overwritten. Partial writes are removed on
    failure; no result manifest becomes visible before the entire cache closes.
    """
    repo, output = repo.resolve(), output.resolve()
    output.relative_to(repo)
    if output.exists():
        raise FileExistsError(output)
    if type(chunk_size) is not int or chunk_size < 1:
        raise QualifiedProgramCacheError("chunk_size must be a positive integer")
    if target_ids is not None and not target_ids:
        raise QualifiedProgramCacheError("Diagnostic selection must not be empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        QualifiedProgramCache(
            repo, population=population, verification=verification, maximum_cached_shards=1
        ) as source,
        tempfile.TemporaryDirectory(prefix=".mapped-", dir=output.parent) as temporary,
    ):
        stage = Path(temporary) / "cache"
        stage.mkdir()
        connection = sqlite3.connect(stage / "sources.sqlite")
        connection.execute(
            "CREATE TABLE sources (idx INTEGER PRIMARY KEY, target_id TEXT UNIQUE NOT NULL, "
            "constitution_id TEXT UNIQUE NOT NULL, family TEXT NOT NULL, "
            "component_instances TEXT NOT NULL, introduced_roles TEXT NOT NULL)"
        )
        handles, counts, headers = {}, {}, {}
        families: Counter[str] = Counter()
        totals = dict(
            records=0,
            atoms=0,
            records_above_96_atoms=0,
            maximum_atoms=0,
            maximum_closures=0,
            precursor_instances=0,
            origin_blocks=0,
            introduced_atom_records=0,
        )
        try:
            for name, dtype in _DTYPES.items():
                handles[name] = (stage / (name + ".npy")).open("w+b")
                headers[name] = _header(handles[name], dtype, 2**63 - 1)
                counts[name] = 0
            stream = (
                (source.record(i) for i in target_ids)
                if target_ids is not None
                else source.iter_records()
            )
            while examples := list(islice(stream, chunk_size)):
                arrays = _pack(examples)
                for name, array in arrays.items():
                    if name.endswith("_offsets"):
                        prefix = name.removesuffix("_offsets")
                        field = _GROUPS[prefix][0]
                        # The payload has not been appended yet: offsets are first in _pack.
                        array = array + counts[field]
                        if counts[name]:
                            array = array[1:]
                    handles[name].write(array.tobytes())
                    counts[name] += array.size
                rows = []
                for example in examples:
                    record = example.record
                    if record.role_morphology_states is not None:
                        raise QualifiedProgramCacheError(
                            "Mapped schema has no role morphology field"
                        )
                    rows.append(
                        (
                            totals["records"],
                            record.graph.structure_id,
                            hashlib.sha256(record.graph.canonical_smiles.encode()).hexdigest(),
                            example.family,
                            json.dumps(example.component_instances),
                            json.dumps(example.introduced_roles),
                        )
                    )
                    totals["records"] += 1
                    totals["atoms"] += record.node_count
                    totals["records_above_96_atoms"] += record.node_count > 96
                    totals["maximum_atoms"] = max(totals["maximum_atoms"], record.node_count)
                    totals["maximum_closures"] = max(
                        totals["maximum_closures"], record.graph.closure_count
                    )
                    totals["precursor_instances"] += sum(example.source_quantities.values())
                    totals["origin_blocks"] += len(record.component_blocks)
                    totals["introduced_atom_records"] += bool(example.introduced_roles)
                    families[example.family] += 1
                connection.executemany("INSERT INTO sources VALUES (?,?,?,?,?,?)", rows)
            connection.commit()
            expected = len(target_ids) if target_ids is not None else len(source)
            if totals["records"] != expected or expected < 1:
                raise QualifiedProgramCacheError("Mapped population count differs from source")
            for name, handle in handles.items():
                if _header(handle, _DTYPES[name], counts[name]) != headers[name]:
                    raise QualifiedProgramCacheError("NPY header size changed")
        finally:
            for handle in handles.values():
                handle.close()
            connection.close()
        remaps = source._read(source.population["artifacts"]["vocabulary-remaps.json"])
        document = {
            "schema_version": _SCHEMA,
            "inputs": {
                "population": dict(population),
                "verification": dict(verification),
                "atom_vocabulary": remaps["atom_vocabulary"],
            },
            "implementation": _pin(repo, Path(__file__).resolve()),
            "selection": "complete_qualified_population" if target_ids is None else "diagnostic",
            "program_vocabulary": asdict(source.vocabulary),
            "totals": totals,
            "by_family": dict(sorted(families.items())),
            "arrays": {
                name: {"shape": [counts[name]], "dtype": dtype.str}
                for name, dtype in sorted(_DTYPES.items())
            },
            "artifacts": {
                path.name: {
                    "path": str((output / path.name).relative_to(repo)),
                    "sha256": str(sha256_file(path)),
                }
                for path in sorted(stage.iterdir())
            },
            "training_admitted": False,
            "sampling_weights_fitted": False,
        }
        (stage / "manifest.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        stage.rename(output)
    return output / "manifest.json"


class MappedProgramCache:
    """Authenticated, read-only sparse arrays; returned records own their memory."""

    def __init__(self, repo: Path, *, manifest: Mapping[str, str]) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self._database: sqlite3.Connection | None = None
        self.metadata = json.loads(resolve_pin(manifest, repo, label="mapped manifest").read_text())
        doc = self.metadata
        if (
            doc.get("schema_version") != _SCHEMA
            or doc.get("training_admitted") is not False
            or doc.get("sampling_weights_fitted") is not False
            or set(doc["arrays"]) != set(_DTYPES)
            or set(doc["artifacts"]) != {*(name + ".npy" for name in _DTYPES), "sources.sqlite"}
        ):
            raise QualifiedProgramCacheError(
                "Mapped preparation schema or admission fields changed"
            )
        for name, value in doc["inputs"].items():
            resolve_pin(value, repo, label="mapped input " + name)
        self.vocabulary = _vocabulary(doc["program_vocabulary"])
        atom_path = resolve_pin(
            doc["inputs"]["atom_vocabulary"], repo, label="mapped atom vocabulary"
        )
        self.atom_vocabulary = QualifiedAtomVocabulary(
            load_atom_vocabulary(atom_path),
            neutral_monovalent_extensions=tuple(
                json.loads(atom_path.read_text())["neutral_monovalent_extensions"]
            ),
        )
        try:
            for name, dtype in _DTYPES.items():
                path = resolve_pin(doc["artifacts"][name + ".npy"], repo, label=name)
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if (
                    array.ndim != 1
                    or array.dtype != dtype
                    or doc["arrays"][name] != {"shape": list(array.shape), "dtype": dtype.str}
                ):
                    raise QualifiedProgramCacheError(f"Mapped array shape or dtype changed: {name}")
                self.arrays[name] = array
            # Validate bounded record slices rather than allocating corpus-sized masks.
            if type(len(self)) is not int or len(self) < 1:
                raise QualifiedProgramCacheError("Mapped population must be nonempty")
            for name in ("program_states", "program_depths"):
                if self.arrays[name].shape != (len(self),):
                    raise QualifiedProgramCacheError(f"Malformed program field: {name}")
            for prefix, fields in _GROUPS.items():
                offsets = self.arrays[prefix + "_offsets"]
                if offsets.shape != (len(self) + 1,) or offsets[0] != 0:
                    raise QualifiedProgramCacheError(f"Malformed {prefix} offsets")
                if any(self.arrays[name].size != offsets[-1] for name in fields):
                    raise QualifiedProgramCacheError(f"Malformed {prefix} payload")
            for start in range(0, len(self), 8192):
                stop = min(start + 8192, len(self))
                subset = {
                    name: self.arrays[name][start:stop]
                    for name in ("program_states", "program_depths")
                }
                for prefix, fields in _GROUPS.items():
                    offsets = self.arrays[prefix + "_offsets"][start : stop + 1]
                    subset[prefix + "_offsets"] = offsets - offsets[0]
                    subset.update(
                        {name: self.arrays[name][offsets[0] : offsets[-1]] for name in fields}
                    )
                QualifiedProgramCache._validate_arrays(subset, stop - start)
            for name, maximum in (
                ("node_states", len(self.atom_vocabulary)),
                ("program_states", len(self.vocabulary.program_states)),
                ("role_states", len(self.vocabulary.role_states)),
                ("block_role_states", len(self.vocabulary.role_states)),
                ("core_position_states", len(self.vocabulary.core_position_states)),
                ("program_depths", self.vocabulary.maximum_steps + 1),
            ):
                if self.arrays[name].max() >= maximum:
                    raise QualifiedProgramCacheError(f"Unknown mapped semantic state: {name}")
            path = resolve_pin(doc["artifacts"]["sources.sqlite"], repo, label="mapped sources")
            self._database = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            self._database.execute("PRAGMA query_only=ON")
            self._database.execute("PRAGMA cache_size=-8192")
            size, lo, hi = self._database.execute(
                "SELECT count(*),min(idx),max(idx) FROM sources"
            ).fetchone()
            if (size, lo, hi) != (len(self), 0, len(self) - 1):
                raise QualifiedProgramCacheError("Mapped source index is incomplete")
        except BaseException:
            self.close()
            raise

    def __len__(self) -> int:
        return self.metadata["totals"]["records"]

    def close(self) -> None:
        self.arrays.clear()
        if self._database is not None:
            self._database.close()
            self._database = None

    def __enter__(self) -> MappedProgramCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def training_measure(self, program_mass: Mapping[str, float]) -> np.ndarray:
        raise QualifiedProgramCacheError(
            "Preparation has no admitted source-balanced training measure"
        )

    def record(self, index: int) -> QualifiedProgramExample:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        if self._database is None:
            raise QualifiedProgramCacheError("Mapped cache is closed")
        return self._record(
            self._database.execute("SELECT * FROM sources WHERE idx=?", (index,)).fetchone()
        )

    def record_by_id(self, identity: str) -> QualifiedProgramExample:
        if self._database is None:
            raise QualifiedProgramCacheError("Mapped cache is closed")
        row = self._database.execute(
            "SELECT * FROM sources WHERE target_id=?", (identity,)
        ).fetchone()
        if row is None:
            raise KeyError(identity)
        return self._record(row)

    def records(self, indices: Sequence[int]) -> tuple[QualifiedProgramExample, ...]:
        return tuple(self.record(index) for index in indices)

    def _record(self, row: tuple) -> QualifiedProgramExample:
        index, identity, constitution, family, instances, introduced = row
        arrays = self.arrays
        smiles = _decode_string(arrays["smiles_offsets"], arrays["smiles_bytes"], index)
        if (
            _decode_string(arrays["record_id_offsets"], arrays["record_id_bytes"], index)
            != identity
            or hashlib.sha256(smiles.encode()).hexdigest() != constitution
        ):
            raise QualifiedProgramCacheError("Mapped source and graph identities differ")

        def values(prefix: str, name: str) -> np.ndarray:
            lo, hi = arrays[prefix + "_offsets"][index : index + 2]
            # Owned records need ordinary arrays. memmap.astype retains the subclass,
            # adding mapping dispatch to every later scalar access despite owning a copy.
            return np.array(
                arrays[name][lo:hi],
                dtype=np.bool_ if name.startswith("fixed_") else np.int64,
                copy=True,
                subok=False,
            )

        graph = SparseGraphRecord(
            structure_id=identity,
            canonical_smiles=smiles,
            **{name: values("node", name) for name in ("node_states", "parents", "parent_bonds")},
            **{
                name: values("closure", name)
                for name in ("closure_left", "closure_right", "closure_bonds")
            },
            edges=np.empty((0, 0), dtype=np.int8),
        )
        lo, hi = arrays["block_offsets"][index : index + 2]
        blocks = tuple(
            SynthesisProgramComponentBlock(
                role=self.vocabulary.role_states[int(arrays["block_role_states"][i])],
                role_state=int(arrays["block_role_states"][i]),
                start=int(arrays["block_starts"][i]),
                stop=int(arrays["block_stops"][i]),
            )
            for i in range(lo, hi)
        )
        state = int(arrays["program_states"][index])
        record = SynthesisProgramGraphRecord(
            graph=graph,
            program_id=self.vocabulary.program_states[state],
            program_state=state,
            program_depth=int(arrays["program_depths"][index]),
            component_blocks=blocks,
            **{name: values("node", name) for name in _NODE_FIELDS[3:]},
            fixed_closure_bond_mask=values("closure", "fixed_closure_bond_mask"),
        )
        return QualifiedProgramExample(
            record,
            family,
            tuple(tuple(v) for v in json.loads(instances)),
            tuple(json.loads(introduced)),
        )
