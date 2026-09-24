"""Pinned, streaming access and qualification for the twelve-family R1 support corpus.

Every row remains reaction-enumerated support. This module creates no split, admits no held-out
training data, and does not extend the trained Ugi/BL/LX checkpoint's semantic vocabulary.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import platform
import sqlite3
import tempfile
import time
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import rdBase

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
    load_assembly_libraries,
)
from forge.core.hashing import resolve_pin, sha256_file
from forge.model.defog_feasibility import AtomState
from forge.model.sparse_topology_feasibility import (
    SPARSE_BOND_TO_INDEX,
    SparseGraphRecord,
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
    tensorize_sparse_row,
)
from forge.model.vocabulary import load_atom_vocabulary

CONFIG_SCHEMA = "forge.combinatorial_library_qualification_config.v1"
RESULT_SCHEMA = "forge.combinatorial_library_qualification.v1"


@dataclass(frozen=True)
class LibraryRecord:
    """A source row and exact precursor provenance, before any training admission."""

    source_row: int
    reaction_id: str
    product_smiles: str
    reactant_ids: tuple[str, ...]
    components: tuple[tuple[str, str], ...]
    realism_weight: float


def read_library_records(
    path: Path,
    libraries: Mapping[str, RegistryAssemblyAdapter],
    blocks: Mapping[str, Mapping[str, Any]],
) -> Iterator[LibraryRecord]:
    """Stream all rows; fail explicitly on missing provenance, unknown roles, or invalid weights."""

    required = {
        "canonical_smiles",
        "reaction_family",
        "reactant_ids",
        "reactant_roles",
        "realism_weight",
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise LibraryAssemblyError("R1 corpus is missing required provenance fields")
        for number, row in enumerate(reader, start=1):
            family = row["reaction_family"]
            if family not in libraries:
                raise LibraryAssemblyError(f"row {number}: unknown reaction family {family!r}")
            roles = tuple(row["reactant_roles"].split("|"))
            ids = tuple(row["reactant_ids"].split("|"))
            if roles != libraries[family].roles or len(ids) != len(roles):
                raise LibraryAssemblyError(f"row {number}: precursor arity or role order mismatch")
            try:
                weight = float(row["realism_weight"])
            except (TypeError, ValueError) as exc:
                raise LibraryAssemblyError(f"row {number}: invalid realism_weight") from exc
            if not math.isfinite(weight) or weight < 0:
                raise LibraryAssemblyError(
                    f"row {number}: realism_weight must be nonnegative and finite"
                )
            components = []
            for role, block_id in zip(roles, ids, strict=True):
                block = blocks.get(block_id)
                if block is None or role not in block.get("reaction_roles", ()):
                    raise LibraryAssemblyError(
                        f"row {number}: unresolved block or role {block_id}/{role}"
                    )
                components.append((role, block["canonical_smiles"]))
            yield LibraryRecord(
                number, family, row["canonical_smiles"], ids, tuple(components), weight
            )


def tensorize_library_record(
    record: LibraryRecord, atom_vocabulary: Sequence[AtomState]
) -> SparseGraphRecord:
    """Use the existing whole-graph backbone without fabricating precursor-origin annotations."""

    canonical, _ = constitutional_molecule(record.product_smiles)
    graph = tensorize_sparse_row(
        {"r0_structure_id": f"r1:{record.source_row}", "canonical_isomeric_smiles": canonical},
        {state: index for index, state in enumerate(atom_vocabulary)},
        preserve_aromaticity=True,
    )
    if not sparse_roundtrip_exact(graph) or not sparse_constitutional_roundtrip_exact(
        graph, atom_vocabulary
    ):
        raise LibraryAssemblyError(f"row {record.source_row}: whole-graph round trip failed")
    return graph


_LIBRARIES: dict[str, RegistryAssemblyAdapter] = {}
_ATOM_STATES: set[AtomState] = set()
_MAXIMUM_OUTCOMES = 0


def _initialize(
    registries: list[tuple[Path, str]],
    families: list[str],
    vocabulary: tuple[AtomState, ...],
    bound: int,
) -> None:
    global _LIBRARIES, _ATOM_STATES, _MAXIMUM_OUTCOMES
    _LIBRARIES = load_assembly_libraries(registries, expected_families=families)
    _ATOM_STATES = set(vocabulary)
    _MAXIMUM_OUTCOMES = bound


def _inspect(records: list[LibraryRecord]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        adapter = _LIBRARIES[record.reaction_id]
        item: dict[str, Any] = {
            "source_row": record.source_row,
            "reaction_id": record.reaction_id,
            "realism_weight": record.realism_weight,
            "issues": [],
        }
        try:
            canonical, molecule = constitutional_molecule(record.product_smiles)
        except LibraryAssemblyError as exc:
            item["issues"].append(f"invalid_product: {exc}")
            result.append(item)
            continue
        item.update(
            canonical_smiles=canonical,
            heavy_atoms=molecule.GetNumHeavyAtoms(),
            closures=molecule.GetNumBonds() - molecule.GetNumAtoms() + 1,
            elements=sorted({atom.GetSymbol() for atom in molecule.GetAtoms()}),
        )
        missing = {
            AtomState(
                atom.GetSymbol(),
                atom.GetFormalCharge(),
                atom.GetIsAromatic(),
                atom.GetNumExplicitHs(),
            )
            for atom in molecule.GetAtoms()
        } - _ATOM_STATES
        item["missing_atom_states"] = [asdict(state) for state in sorted(missing, key=repr)]
        if missing:
            item["issues"].append("atom_vocabulary")
        if any(bond.GetBondType() not in SPARSE_BOND_TO_INDEX for bond in molecule.GetBonds()):
            item["issues"].append("bond_vocabulary")
        assessments = adapter.assess_roles(dict(record.components))
        item["failed_roles"] = [asdict(value) for value in assessments if not value.qualified]
        if item["failed_roles"]:
            item["issues"].append("registry_role_policy")
        products = adapter.forward_products(
            dict(record.components), maximum_outcomes=_MAXIMUM_OUTCOMES
        )
        item["forward_exact"] = canonical in products.products
        item["saturated"] = products.saturated
        item["distinct_forward_products"] = len(products.products)
        if products.saturated:
            item["issues"].append("forward_enumeration_saturated")
        if not item["forward_exact"]:
            item["issues"].append("exact_forward_reconstruction")
        result.append(item)
    return result


def _chunks(records: Iterator[LibraryRecord], size: int) -> Iterator[list[LibraryRecord]]:
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _batches(
    records: Iterator[LibraryRecord], workers: int, initializer_args: tuple
) -> Iterator[tuple[list[LibraryRecord], list[dict[str, Any]]]]:
    chunks = iter(_chunks(records, 256))
    if workers == 1:
        _initialize(*initializer_args)
        for batch in chunks:
            yield batch, _inspect(batch)
        return
    # A bounded submission window prevents Executor.map from retaining the complete CSV in RAM.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialize, initargs=initializer_args
    ) as pool:
        pending = deque()
        for _ in range(2 * workers):
            batch = next(chunks, None)
            if batch is not None:
                pending.append((batch, pool.submit(_inspect, batch)))
        while pending:
            batch, future = pending.popleft()
            yield batch, future.result()
            following = next(chunks, None)
            if following is not None:
                pending.append((following, pool.submit(_inspect, following)))


def qualify_combinatorial_libraries(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    repo = repo_root.resolve()
    config_path = (repo / config_path).resolve()
    output = (repo / output_dir).resolve()
    if not output.is_relative_to(repo) or not config_path.is_relative_to(repo):
        raise LibraryAssemblyError("config and output must be inside repo_root")
    config = json.loads(config_path.read_text())
    config_pin = {
        "path": str(config_path.relative_to(repo)),
        "sha256": str(sha256_file(config_path)),
    }
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("sampling_weight") != "realism_weight"
    ):
        raise LibraryAssemblyError(
            "unsupported schema or sampling policy; realism_weight is required"
        )
    for name in ("workers", "maximum_outcomes", "representation_samples_per_family"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise LibraryAssemblyError(f"{name} must be a positive integer")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise LibraryAssemblyError("seed must be a nonnegative integer")
    expected = config["expected_rows_by_family"]
    if not expected or any(type(n) is not int or n < 1 for n in expected.values()):
        raise LibraryAssemblyError("expected_rows_by_family must contain positive integer counts")
    inputs = config["inputs"]
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in inputs.items()}
    registries = [(paths[name], inputs[name]["sha256"]) for name in config["registries"]]
    libraries = load_assembly_libraries(registries, expected_families=sorted(expected))
    raw_blocks = json.loads(paths["building_blocks"].read_text())["blocks"]
    blocks = {row["block_id"]: row for row in raw_blocks}
    if len(blocks) != len(raw_blocks):
        raise LibraryAssemblyError("building block identifiers are not unique")
    vocabulary = tuple(load_atom_vocabulary(paths["atom_vocabulary"]))
    if output.exists():
        raise LibraryAssemblyError(f"output already exists; choose a fresh directory: {output}")
    implementation_root = Path(__file__).resolve().parents[2]
    source_names = (
        "forge/corpus/combinatorial_libraries.py",
        "forge/assembly/families.py",
        "forge/assembly/registry.py",
        "forge/assembly/program.py",
        "forge/assembly/api.py",
        "forge/chemistry/reactive_sites.py",
        "forge/core/hashing.py",
        "forge/model/vocabulary.py",
        "forge/model/defog_feasibility.py",
        "forge/model/sparse_topology_feasibility.py",
        "experiments/phase1/multireaction/combinatorial_libraries.py",
    )
    source_pins = {name: str(sha256_file(implementation_root / name)) for name in source_names}
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stats = {family: Counter() for family in libraries}
    issues = {family: Counter() for family in libraries}
    panels: dict[str, list[tuple[int, int, LibraryRecord]]] = {family: [] for family in libraries}
    elements: dict[str, set[str]] = {family: set() for family in libraries}
    with tempfile.TemporaryDirectory(
        prefix=".library-qualification-", dir=output.parent
    ) as temporary:
        work = Path(temporary)
        db = sqlite3.connect(work / "identity.sqlite")
        db.execute(
            "CREATE TABLE products(family TEXT, smiles TEXT, PRIMARY KEY(family, smiles)) WITHOUT ROWID"
        )
        ledger = work / "row_qualification.jsonl.gz"
        records = read_library_records(paths["corpus"], libraries, blocks)
        args = (registries, sorted(libraries), vocabulary, config["maximum_outcomes"])
        with (
            ledger.open("wb") as binary,
            gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed,
            io.TextIOWrapper(compressed) as handle,
        ):
            for batch, inspected in _batches(records, config["workers"], args):
                for record, item in zip(batch, inspected, strict=True):
                    family = record.reaction_id
                    current = stats[family]
                    current["rows"] += 1
                    current["realism_weight_sum"] += record.realism_weight
                    current["positive_weight_rows"] += record.realism_weight > 0
                    current["zero_weight_rows"] += record.realism_weight == 0
                    current["qualified_rows"] += not item["issues"]
                    current["exact_forward_rows"] += item.get("forward_exact", False)
                    current["multiple_forward_product_rows"] += (
                        item.get("distinct_forward_products", 0) > 1
                    )
                    current["saturated_rows"] += item.get("saturated", False)
                    current["maximum_heavy_atoms"] = max(
                        current["maximum_heavy_atoms"], item.get("heavy_atoms", 0)
                    )
                    current["maximum_closures"] = max(
                        current["maximum_closures"], item.get("closures", 0)
                    )
                    issues[family].update(item["issues"])
                    elements[family].update(item.get("elements", ()))
                    if "canonical_smiles" in item:
                        db.execute(
                            "INSERT OR IGNORE INTO products VALUES (?, ?)",
                            (family, item["canonical_smiles"]),
                        )
                    rank = int(
                        hashlib.sha256(
                            f"{config['seed']}|{family}|{record.source_row}".encode()
                        ).hexdigest(),
                        16,
                    )
                    panel = panels[family]
                    heapq.heappush(panel, (-rank, record.source_row, record))
                    if len(panel) > config["representation_samples_per_family"]:
                        heapq.heappop(panel)
                    handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
        db.commit()
        total_rows = sum(s["rows"] for s in stats.values())
        total_weight = sum(s["realism_weight_sum"] for s in stats.values())
        for family, current in stats.items():
            current["unique_constitutional_products"] = db.execute(
                "SELECT COUNT(*) FROM products WHERE family=?", (family,)
            ).fetchone()[0]
            current["raw_row_fraction"] = current["rows"] / total_rows if total_rows else 0
            current["realism_weight_mass"] = (
                current["realism_weight_sum"] / total_weight if total_weight else 0
            )
        unique_products = db.execute("SELECT COUNT(DISTINCT smiles) FROM products").fetchone()[0]
        db.close()
        controls: dict[str, Any] = {}
        for path, _ in registries:
            for raw in json.loads(path.read_text())["reactions"]:
                family = raw["reaction_id"]
                adapter = libraries[family]
                positive = []
                negative = []
                for example in raw["known_positive_examples"]:
                    check = adapter.check_forward(
                        dict(zip(adapter.roles, example["reactants"], strict=True)),
                        example["expected"],
                        maximum_outcomes=config["maximum_outcomes"],
                    )
                    positive.append(check.exact and not check.saturated)
                for example in raw["known_negative_examples"]:
                    products = adapter.forward_products(
                        dict(zip(adapter.roles, example["reactants"], strict=True)),
                        maximum_outcomes=config["maximum_outcomes"],
                    )
                    negative.append(not products.products and not products.saturated)
                controls[family] = {"positive_exact": positive, "negative_rejected": negative}
        representation = {}
        for family, panel in panels.items():
            rows = []
            for _, _, record in sorted(panel, reverse=True):
                try:
                    graph = tensorize_library_record(record, vocabulary)
                    rows.append(
                        {
                            "source_row": record.source_row,
                            "exact": True,
                            "nodes": graph.node_count,
                            "closures": graph.closure_count,
                        }
                    )
                except (ValueError, RuntimeError, KeyError) as exc:
                    rows.append(
                        {"source_row": record.source_row, "exact": False, "error": str(exc)}
                    )
            representation[family] = rows
        gates = {
            "all_declared_rows_scanned": all(stats[f]["rows"] == n for f, n in expected.items()),
            "all_rows_qualified": all(
                s["rows"] > 0 and s["qualified_rows"] == s["rows"] for s in stats.values()
            ),
            "all_families_have_positive_sampling_mass": all(
                s["realism_weight_sum"] > 0 for s in stats.values()
            ),
            "registry_controls_pass": all(
                all(c["positive_exact"])
                and all(c["negative_rejected"])
                and c["positive_exact"]
                and c["negative_rejected"]
                for c in controls.values()
            ),
            "sampled_whole_graph_roundtrips_exact": all(
                rows and all(row["exact"] for row in rows) for rows in representation.values()
            ),
        }
        for name, pin in inputs.items():
            resolve_pin(pin, repo, label=name)
        if sha256_file(config_path) != config_pin["sha256"] or any(
            sha256_file(implementation_root / name) != digest
            for name, digest in source_pins.items()
        ):
            raise LibraryAssemblyError(
                "configuration or implementation changed during qualification"
            )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "pass" if all(gates.values()) else "blocked",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_pin,
            "inputs": inputs,
            "sources": source_pins,
            "environment": {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "workers": config["workers"],
            },
            "seed": config["seed"],
            "duration_seconds": time.monotonic() - start,
            "summary": {
                "rows": total_rows,
                "unique_constitutional_products": unique_products,
                "families": len(libraries),
            },
            "families": {
                f: {**dict(stats[f]), "issues": dict(issues[f]), "elements": sorted(elements[f])}
                for f in libraries
            },
            "registry_controls": controls,
            "representation_panel": representation,
            "gates": gates,
            "artifacts": {
                "row_qualification": {
                    "path": str((output / ledger.name).relative_to(repo)),
                    "sha256": str(sha256_file(ledger)),
                }
            },
            "calls": {"training": 0, "molecular_generation": 0, "remote_compute": 0},
            "nonclaims": [
                "R1 is reaction-enumerated support, not twelve source-executed synthesis libraries or route certification.",
                "Exact forward replay and registry control specificity do not estimate chemical synthesis precision or success probability.",
                "Atom/bond vocabulary is checked on every row; full sparse encode/decode is checked only on the deterministic reported panel.",
                "This qualification assigns no train/test folds and does not authorize feeding all R1 rows to training.",
                "Zero realism weights are retained exactly and confer zero sampling mass; no positive floor is introduced.",
                "Existing trained checkpoint semantics, Ugi/BL/LX repeated programs, evidence gates and heldouts are unchanged.",
                "No improved realism, diversity, novelty, twelve-family learned generation or L2/L3 closure is established.",
            ],
        }
        (work / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        (work / "identity.sqlite").unlink()
        os.rename(work, output)
    return result
