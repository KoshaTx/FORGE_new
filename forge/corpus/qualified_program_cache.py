"""Bounded loading of audited constitutional graphs and their complete source instances."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_supplement import rows
from forge.corpus.synthesis_program_production_cache import _ARRAY_KEYS, _decode_string
from forge.model.introduced_source_coordinates import (
    collate_source_and_introduced_records,
    source_and_introduced_coordinates,
)
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.sparse_topology_feasibility import SparseGraphRecord
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)
from forge.model.vocabulary import load_atom_vocabulary

_SOURCE_ARRAYS = frozenset(
    {
        "source_instance_states",
        "source_position_states",
        "source_repeat_group_states",
        "source_instance_counts",
    }
)
_SCHEMAS = {
    "forge.constitutional_program_preparation_tensors.v1": False,
    "forge.constitutional_source_instance_preparation_tensors.v1": True,
    "forge.constitutional_source_and_introduced_preparation_tensors.v1": True,
}


class QualifiedProgramCacheError(ValueError):
    """An audited preparation record or its source-instance contract changed."""


@dataclass(frozen=True)
class QualifiedProgramExample:
    """A sparse graph and non-neural source provenance; no sampling/admission fields."""

    record: SynthesisProgramGraphRecord
    family: str
    component_instances: tuple[tuple[str, str, int], ...]
    introduced_roles: tuple[str, ...]

    @property
    def source_quantities(self) -> dict[str, int]:
        quantities = {}
        for role, identity, quantity in self.component_instances:
            if role in quantities or not identity or type(quantity) is not int or quantity < 1:
                raise QualifiedProgramCacheError("Incomplete or invalid source precursor tuple")
            quantities[role] = quantity
        return quantities


def collate_qualified_program_examples(
    examples: Sequence[QualifiedProgramExample],
    *,
    maximum_nodes: int | None = None,
    maximum_closures: int,
    conditioning_mode: str = "program",
) -> dict[str, Any]:
    """Carry verified instances into the shared model interface without source IDs."""
    return collate_source_and_introduced_records(
        [example.record for example in examples],
        [example.source_quantities for example in examples],
        introduced_roles=[example.introduced_roles for example in examples],
        maximum_nodes=maximum_nodes,
        maximum_closures=maximum_closures,
        conditioning_mode=conditioning_mode,
    )


def _vocabulary(value: Mapping[str, Any]) -> ReactionProgramVocabulary:
    return ReactionProgramVocabulary(
        program_states=tuple(value["program_states"]),
        role_states=tuple(value["role_states"]),
        core_position_states=tuple(value["core_position_states"]),
        maximum_steps=value["maximum_steps"],
    )


class QualifiedProgramCache:
    """Read immutable preparation shards in a shared vocabulary with a bounded LRU.

    Source graphs, component quantities and introduction contracts remain pinned.
    Returned records own their arrays and keep adjacency sparse. This preparation
    interface cannot supply a training measure or invent missing source weights.
    """

    def __init__(
        self,
        repo: Path,
        *,
        population: Mapping[str, str],
        verification: Mapping[str, str],
        maximum_cached_shards: int = 4,
    ) -> None:
        if type(maximum_cached_shards) is not int or maximum_cached_shards < 1:
            raise QualifiedProgramCacheError("maximum_cached_shards must be a positive integer")
        self.repo = repo
        self.maximum_cached_shards = maximum_cached_shards
        self._cached: OrderedDict[int, tuple[dict, list[dict], dict]] = OrderedDict()
        self.population = self._read(population)
        checked = self._read(verification)
        if (
            self.population.get("schema_version") != "forge.unified_preparation_index.v1"
            or checked.get("schema_version") != "forge.unified_preparation_verification.v1"
            or checked["inputs"]["population"] != population
            or checked["totals"] != self.population["totals"]
            or self.population["training_admitted"] is not False
            or self.population["sampling_weights_fitted"] is not False
            or checked["training_admitted"] is not False
            or any(
                checked[name] is not True
                for name in (
                    "all_exact_eligible_records_present_once",
                    "source_identity_family_size_and_constitution_preserved",
                    "shard_row_indices_complete",
                    "vocabulary_remaps_lossless",
                )
            )
        ):
            raise QualifiedProgramCacheError("Complete verified preparation is required")
        request = self._read(self.population["request"])
        remaps = self._read(self.population["artifacts"]["vocabulary-remaps.json"])
        if remaps["inputs"] != request["inputs"] or remaps["training_admitted"] is not False:
            raise QualifiedProgramCacheError("Vocabulary remaps belong to different source inputs")
        self.vocabulary = _vocabulary(remaps["program_vocabulary"])
        self._stage_requests, self._remaps = {}, {}
        for stage, fields in remaps["source_to_union_state_indices"].items():
            stage_request_pin = request["inputs"][stage]["request"]
            stage_request = self._read(stage_request_pin)
            if stage_request["inputs"]["atom_vocabulary"] != remaps["atom_vocabulary"]:
                raise QualifiedProgramCacheError("Atom vocabulary differs between source stages")
            self._stage_requests[stage] = stage_request_pin
            mapped = {}
            for name in ("program_states", "role_states", "core_position_states"):
                indices = fields[name]
                target = getattr(self.vocabulary, name)
                if (
                    any(type(i) is not int or not 0 <= i < len(target) for i in indices)
                    or len(indices) != len(set(indices))
                    or [target[i] for i in indices] != stage_request["program_vocabulary"][name]
                ):
                    raise QualifiedProgramCacheError("Semantic state remap is not lossless")
                mapped[name] = np.asarray(indices, dtype=np.int64)
            self._remaps[stage] = mapped
        atom_path = resolve_pin(remaps["atom_vocabulary"], repo, label="qualified atom vocabulary")
        atom_document = json.loads(atom_path.read_text())
        self.atom_vocabulary = QualifiedAtomVocabulary(
            load_atom_vocabulary(atom_path),
            neutral_monovalent_extensions=tuple(atom_document["neutral_monovalent_extensions"]),
        )
        path = resolve_pin(
            self.population["artifacts"]["preparation.sqlite"], repo, label="qualified lookup"
        )
        self._database = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        self._database.execute("PRAGMA query_only=ON")
        self._database.execute("PRAGMA cache_size=-8192")

    def _read(self, value: Mapping[str, str]) -> dict:
        return json.loads(
            resolve_pin(value, self.repo, label="qualified program dependency").read_text()
        )

    def close(self) -> None:
        self._cached.clear()
        self._database.close()

    def __enter__(self) -> QualifiedProgramCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self.population["totals"]["records"])

    def training_measure(self, program_mass: Mapping[str, float]) -> np.ndarray:
        raise QualifiedProgramCacheError(
            "Preparation has no admitted source-balanced training measure"
        )

    def _shard(self, identity: int) -> tuple[dict, list[dict], dict]:
        if identity in self._cached:
            self._cached.move_to_end(identity)
            return self._cached[identity]
        found = self._database.execute(
            "SELECT stage,family,receipt,tensor_path,tensor_sha256 FROM shards WHERE id=?",
            (identity,),
        ).fetchone()
        if found is None:
            raise QualifiedProgramCacheError(f"Unknown qualified tensor shard: {identity}")
        stage, family, receipt, path, sha256 = found
        shard = self._read(json.loads(receipt))
        if (
            shard["request"] != self._stage_requests[stage]
            or shard["family"] != family
            or shard["artifact"] != {"path": path, "sha256": sha256}
            or shard["training_admitted"] is not False
        ):
            raise QualifiedProgramCacheError("Tensor shard differs from its qualified lookup")
        tensor_path = resolve_pin(shard["artifact"], self.repo, label="qualified tensor shard")
        with np.load(tensor_path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].tobytes().decode())
            schema = metadata.get("schema_version")
            if schema not in _SCHEMAS:
                raise QualifiedProgramCacheError("Unsupported preparation tensor schema")
            expected = _ARRAY_KEYS - {"fold_states", "source_weights"}
            if _SCHEMAS[schema]:
                expected |= _SOURCE_ARRAYS
            if set(archive.files) != expected:
                raise QualifiedProgramCacheError("Preparation arrays or admission fields changed")
            arrays = {name: archive[name] for name in archive.files}
        count = shard["counts"]["records"]
        if metadata != {
            "schema_version": schema,
            "request": shard["request"],
            "origin_shard": shard["origin_shard"],
            "rows": count,
            "training_admitted": False,
            "training_weights_defined": False,
        }:
            raise QualifiedProgramCacheError("Tensor metadata differs from its source contract")
        self._validate_arrays(arrays, count)
        origin = self._read(shard["origin_shard"])
        source = list(
            rows(resolve_pin(origin["artifact"], self.repo, label="complete source recipes"))
        )
        if len(source) != count or origin["rows"] != count or origin["family"] != family:
            raise QualifiedProgramCacheError("Source recipes and tensor records are misaligned")
        mapping = self._remaps[stage]
        for name, state_field in (
            ("program_states", "program_states"),
            ("role_states", "role_states"),
            ("block_role_states", "role_states"),
            ("core_position_states", "core_position_states"),
        ):
            if arrays[name].max() >= len(mapping[state_field]):
                raise QualifiedProgramCacheError(f"Unknown source semantic state: {name}")
            arrays[name] = mapping[state_field][arrays[name]]
        if "source_repeat_group_states" in arrays:
            values = arrays["source_repeat_group_states"]
            if values.max() >= len(mapping["role_states"]):
                raise QualifiedProgramCacheError("Unknown source repeat role")
            arrays["source_repeat_group_states"] = mapping["role_states"][values]
        self._cached[identity] = arrays, source, shard
        while len(self._cached) > self.maximum_cached_shards:
            self._cached.popitem(last=False)
        return self._cached[identity]

    @staticmethod
    def _validate_arrays(arrays: Mapping[str, np.ndarray], count: int) -> None:
        if type(count) is not int or count < 1:
            raise QualifiedProgramCacheError("A preparation shard must have positive record count")
        for prefix, names in (
            (
                "node",
                (
                    "node_states",
                    "parents",
                    "parent_bonds",
                    "canonical_atom_order",
                    "role_states",
                    "core_position_states",
                    "fixed_atom_mask",
                    "fixed_parent_bond_mask",
                ),
            ),
            (
                "closure",
                ("closure_left", "closure_right", "closure_bonds", "fixed_closure_bond_mask"),
            ),
            ("block", ("block_role_states", "block_starts", "block_stops")),
            ("record_id", ("record_id_bytes",)),
            ("smiles", ("smiles_bytes",)),
        ):
            offsets = arrays[prefix + "_offsets"]
            if (
                offsets.dtype != np.int64
                or offsets.shape != (count + 1,)
                or offsets[0] != 0
                or np.any(np.diff(offsets) < (0 if prefix == "closure" else 1))
            ):
                raise QualifiedProgramCacheError(f"Malformed {prefix} offsets")
            for name in names:
                value = arrays[name]
                if (
                    value.shape != (offsets[-1],)
                    or value.dtype.kind not in "ui"
                    or np.any(value < 0)
                ):
                    raise QualifiedProgramCacheError(f"Malformed packed field: {name}")
        for name in ("program_states", "program_depths"):
            value = arrays[name]
            if value.shape != (count,) or value.dtype.kind not in "ui" or np.any(value < 1):
                raise QualifiedProgramCacheError(f"Malformed program field: {name}")
        if "source_instance_counts" in arrays:
            for name in _SOURCE_ARRAYS:
                value = arrays[name]
                size = count if name == "source_instance_counts" else arrays["node_offsets"][-1]
                if value.shape != (size,) or value.dtype.kind not in "ui" or np.any(value < 0):
                    raise QualifiedProgramCacheError(f"Malformed source instance field: {name}")
        if any(
            np.any(arrays[name] > 1)
            for name in ("fixed_atom_mask", "fixed_parent_bond_mask", "fixed_closure_bond_mask")
        ):
            raise QualifiedProgramCacheError("Fixed masks must be boolean-valued")

    def record(self, target_id: str) -> QualifiedProgramExample:
        value = self._database.execute(
            "SELECT * FROM records WHERE target_id=?", (target_id,)
        ).fetchone()
        if value is None:
            raise KeyError(target_id)
        return self._record(value)

    def records(self, target_ids: Sequence[str]) -> tuple[QualifiedProgramExample, ...]:
        return tuple(self.record(identity) for identity in target_ids)

    def iter_records(self, *, family: str | None = None) -> Iterator[QualifiedProgramExample]:
        if family is not None and family not in self.population["by_family"]:
            raise QualifiedProgramCacheError(f"Unknown qualified family: {family}")
        query = "SELECT * FROM records"
        parameters = ()
        if family is not None:
            query += " WHERE family=?"
            parameters = (family,)
        query += " ORDER BY shard_id,row_index"
        for value in self._database.execute(query, parameters):
            yield self._record(value)

    def _record(self, lookup: tuple) -> QualifiedProgramExample:
        identity, constitution, family, program, shard_id, index, atoms = lookup
        arrays, source, _ = self._shard(shard_id)
        row = source[index]
        replay = row["semantic_replay"]
        annotation = replay["annotations"]
        if (
            row["old_projection"] != "train"
            or row["corrected_projection"] != "train"
            or row["eligible_for_program_preparation"] is not True
            or row["training_admitted"] is not False
            or row["exact_evidence"]["exact_computed_reconstruction"] is not True
            or replay["complete_search"] is not True
            or replay["disposition"] != "unique_forward_atom_coordinates"
            or annotation is None
        ):
            raise QualifiedProgramCacheError("Source record is protected, unresolved or nonexact")
        smiles = _decode_string(arrays["smiles_offsets"], arrays["smiles_bytes"], index)
        record_id = _decode_string(arrays["record_id_offsets"], arrays["record_id_bytes"], index)
        if (
            record_id != identity
            or row["target_id"] != identity
            or row["family"] != family
            or row["program_key"] != program
            or row["constitution_id"] != constitution
            or hashlib.sha256(smiles.encode()).hexdigest() != constitution
            or annotation["canonical_product_smiles"] != smiles
            or replay["constitutional_products"] != [smiles]
        ):
            raise QualifiedProgramCacheError("Source, graph and indexed identity differ")
        node_start, node_stop = arrays["node_offsets"][index : index + 2]
        closure_start, closure_stop = arrays["closure_offsets"][index : index + 2]
        block_start, block_stop = arrays["block_offsets"][index : index + 2]

        def node(name, dtype=np.int64):
            return arrays[name][node_start:node_stop].astype(dtype, copy=True)

        def closure(name, dtype=np.int64):
            return arrays[name][closure_start:closure_stop].astype(dtype, copy=True)

        graph = SparseGraphRecord(
            structure_id=identity,
            canonical_smiles=smiles,
            node_states=node("node_states"),
            parents=node("parents"),
            parent_bonds=node("parent_bonds"),
            closure_left=closure("closure_left"),
            closure_right=closure("closure_right"),
            closure_bonds=closure("closure_bonds"),
            edges=np.empty((0, 0), dtype=np.int8),
        )
        blocks = tuple(
            SynthesisProgramComponentBlock(
                role=self.vocabulary.role_states[int(arrays["block_role_states"][i])],
                role_state=int(arrays["block_role_states"][i]),
                start=int(arrays["block_starts"][i]),
                stop=int(arrays["block_stops"][i]),
            )
            for i in range(block_start, block_stop)
        )
        record = SynthesisProgramGraphRecord(
            graph=graph,
            canonical_atom_order=node("canonical_atom_order"),
            program_id=program,
            program_state=int(arrays["program_states"][index]),
            program_depth=int(arrays["program_depths"][index]),
            role_states=node("role_states"),
            core_position_states=node("core_position_states"),
            component_blocks=blocks,
            fixed_atom_mask=node("fixed_atom_mask", np.bool_),
            fixed_parent_bond_mask=node("fixed_parent_bond_mask", np.bool_),
            fixed_closure_bond_mask=closure("fixed_closure_bond_mask", np.bool_),
        )
        order = record.canonical_atom_order
        if (
            record.node_count != atoms
            or self.vocabulary.program_states[record.program_state] != program
            or record.program_depth != len(replay["states_by_layer"]) - 1
            or [self.vocabulary.role_states[i] for i in record.role_states]
            != [annotation["atom_roles"][i] for i in order]
            or [self.vocabulary.core_position_states[i] for i in record.core_position_states]
            != [
                (
                    "exterior"
                    if annotation["core_positions"][i] == "exterior"
                    else program + ":" + annotation["core_positions"][i]
                )
                for i in order
            ]
        ):
            raise QualifiedProgramCacheError(
                "Saved graph dimensions or semantic coordinates changed"
            )
        example = QualifiedProgramExample(
            record,
            family,
            tuple(tuple(value) for value in row["component_instances"]),
            tuple(row.get("introduced_roles", ())),
        )
        coordinates = source_and_introduced_coordinates(
            record, example.source_quantities, introduced_roles=example.introduced_roles
        )
        if "source_instance_counts" in arrays:
            if arrays["source_instance_counts"][index] != sum(example.source_quantities.values()):
                raise QualifiedProgramCacheError("Saved source precursor count changed")
            for name, values in (
                ("source_instance_states", coordinates.instance_states),
                ("source_position_states", coordinates.position_states),
                ("source_repeat_group_states", coordinates.repeat_group_states),
            ):
                if not np.array_equal(node(name), values):
                    raise QualifiedProgramCacheError(f"Saved source coordinates changed: {name}")
        return example
