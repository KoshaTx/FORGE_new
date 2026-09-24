"""Exhaustive eligible-product representation audit using unchanged vocabulary and codec."""

import gzip
import json
import multiprocessing
import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import torch

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_full_partition import PartitionPreparationCorpus
from forge.corpus.compose_lipid_representation import _model_smoke, capacity_covered, kekule_states
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows
from forge.model.defog_feasibility import AtomState, FeasibilityError
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.sparse_topology_feasibility import (
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
    tensorize_sparse_row,
)
from results.phase1.compose_lipid_full_replay_v1.replay_michael import digest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORKERS, BATCH_SIZE = 4, 2048
_VOCABULARY, _MAPPING = None, None


def initialize(vocabulary):
    global _VOCABULARY, _MAPPING
    torch.set_num_threads(1)
    _VOCABULARY = vocabulary
    _MAPPING = {state: index for index, state in enumerate(vocabulary)}


def evaluate(items):
    result = []
    for target, family, atoms, smiles in items:
        row = {
            "target_id": target,
            "family": family,
            "heavy_atoms": atoms,
            "training_admitted": False,
        }
        states = kekule_states(smiles)
        counts = Counter(states)
        row["atom_states"] = [
            {**asdict(state), "atoms": n}
            for state, n in sorted(counts.items(), key=lambda x: x[0].key())
        ]
        unknown = set(counts) - set(_MAPPING)
        if unknown:
            row.update(
                pass_checks=False,
                reason="outside_frozen_vocabulary",
                unknown_states=[asdict(state) for state in sorted(unknown, key=AtomState.key)],
            )
        else:
            try:
                graph = tensorize_sparse_row(
                    {"r0_structure_id": target, "canonical_isomeric_smiles": smiles},
                    _MAPPING,
                    preserve_aromaticity=False,
                )
                checks = {
                    "sparse_edge_roundtrip": sparse_roundtrip_exact(graph),
                    "constitutional_roundtrip": sparse_constitutional_roundtrip_exact(
                        graph, _VOCABULARY
                    ),
                    "valence_capacity": capacity_covered(graph, _VOCABULARY),
                    "active_bond_support": all(
                        0 <= int(b) < 3
                        for b in list(graph.parent_bonds[1:]) + list(graph.closure_bonds)
                    ),
                    "source_heavy_atom_count": graph.node_count == atoms,
                }
                row.update(
                    checks=checks, pass_checks=all(checks.values()), closures=graph.closure_count
                )
            except FeasibilityError as exc:
                row.update(pass_checks=False, reason="codec_error", error=str(exc))
        result.append(row)
    return result


def main():
    started = time.monotonic()
    output = HERE / "audit"
    if (output / "result.json").exists():
        raise ValueError("Published representation result is frozen")
    output.mkdir(exist_ok=True)
    partition_path = ROOT / "results/phase1/compose_lipid_full_partition_v1/audit-v2/result.json"
    vocabulary_receipt = ROOT / "results/phase1/compose_lipid_v8_representation_v1/result.json"
    previous = json.loads(vocabulary_receipt.read_text())
    for name, value in previous["implementation"].items():
        resolve_pin(value, ROOT, label=name)
    vocabulary_path = resolve_pin(
        previous["artifacts"]["atom_vocabulary.json"], ROOT, label="existing vocabulary"
    )
    document = json.loads(vocabulary_path.read_text())
    if document["status"] != "pass" or document["preserve_aromaticity"] is not False:
        raise ValueError("Existing constitutional vocabulary is unqualified")
    vocabulary = QualifiedAtomVocabulary(
        tuple(
            AtomState(**{k: v for k, v in row.items() if k not in {"index", "train_atoms"}})
            for row in document["atom_vocabulary"]
        ),
        tuple(document["neutral_monovalent_extensions"]),
    )
    if list(vocabulary.capacities()) != document["heavy_atom_valence_capacities"]:
        raise ValueError("Representation changed chemical valence limits")
    reader = PartitionPreparationCorpus(ROOT, partition_path)
    implementation = {
        **previous["implementation"],
        **{
            n: pin(ROOT, ROOT / n)
            for n in (
                "forge/corpus/compose_lipid_full_partition.py",
                "forge/corpus/compose_lipid_source_view.py",
                "forge/corpus/compose_lipid_supplement.py",
                "forge/core/hashing.py",
                "results/phase1/compose_lipid_full_replay_v1/replay_michael.py",
                str(Path(__file__).resolve().relative_to(ROOT)),
            )
        },
    }
    inputs = {
        "partition_receipt": pin(ROOT, partition_path),
        "partition": reader.result["artifact"],
        "source_corpus": reader.result["inputs"]["corpus"],
        "vocabulary_receipt": pin(ROOT, vocabulary_receipt),
        "vocabulary": pin(ROOT, vocabulary_path),
    }
    request = {
        "schema_version": "forge.compose_lipid_full_representation_request.v1",
        "seed": 0,
        "inputs": inputs,
        "implementation": implementation,
        "runtime": {
            "workers": WORKERS,
            "batch_size": BATCH_SIZE,
            "device": "cpu",
            "precision": "float32",
            "torch_threads_per_worker": 1,
        },
        "policy": {
            "purpose": "eligible_product_representation_diagnostic_before_final_chemistry_admission",
            "vocabulary": "unchanged_previously_qualified_states_and_valence_limits",
            "source_vocabulary_frequencies_used": False,
            "refit_on_final_admitted_train_required": True,
            "record_cap": None,
            "size_filter": None,
            "training_admitted": False,
            "training_calls": 0,
        },
    }
    request_path = output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Cannot resume representation with changed inputs")
    dump(request_path, request)
    request_pin = pin(ROOT, request_path)
    db = sqlite3.connect(reader.partition.as_uri() + "?mode=ro", uri=True)
    db.execute("ATTACH DATABASE ? AS original", (reader.corpus.as_uri() + "?mode=ro",))
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA temp_store=FILE")
    query = "SELECT e.target_id,e.family,e.heavy_atoms,t.constitution FROM eligible e JOIN original.targets t USING(target_id) "
    samples = {}
    for family, counts in reader.result["summary"]["by_family"].items():
        if not counts.get("eligible_for_program_preparation"):
            continue
        for row in db.execute(query + "WHERE e.family=? ORDER BY e.target_id LIMIT 2", (family,)):
            samples[row[0]] = row
        for row in db.execute(
            query + "WHERE e.family=? ORDER BY e.heavy_atoms DESC,e.target_id LIMIT 1", (family,)
        ):
            samples[row[0]] = row
    samples = list(samples.values())
    initialize(vocabulary)
    mark = time.monotonic()
    serial = evaluate(samples)
    serial_seconds = time.monotonic() - mark
    families = defaultdict(Counter)
    state_counts = Counter()
    maximum = {"atoms": 0, "closures": 0}
    maximum_targets = {}
    failures = []
    receipts = []
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(vocabulary,),
    ) as pool:
        mark = time.monotonic()
        parallel = [r for batch in pool.map(evaluate, [[v] for v in samples]) for r in batch]
        if parallel != serial:
            raise ValueError("Parallel representation differs from serial codec")
        dump(
            output / "preflight.json",
            {
                "request": request_pin,
                "rows": len(samples),
                "sample_targets": [r[0] for r in samples],
                "serial_sha256": digest(serial),
                "parallel_sha256": digest(parallel),
                "serial_seconds": serial_seconds,
                "parallel_cold_seconds": time.monotonic() - mark,
                "largest_per_family_included": True,
                "equivalent": True,
                "training_admitted": False,
            },
        )
        print(f"Representation equivalence passed for {len(samples)} examples", flush=True)

        def account(records, receipt):
            receipts.append(receipt)
            for row in records:
                families[row["family"]].update(
                    {
                        "rows": 1,
                        "passed": row["pass_checks"],
                        "above_96_atoms": row["heavy_atoms"] > 96,
                    }
                )
                families[row["family"]].update(k for k, v in row.get("checks", {}).items() if v)
                for state in row["atom_states"]:
                    key = tuple(
                        (k, state[k])
                        for k in ("symbol", "formal_charge", "aromatic", "explicit_hydrogens")
                    )
                    state_counts[key] += state["atoms"]
                if not row["pass_checks"]:
                    failures.append(row)
                for key, value in (
                    ("atoms", row["heavy_atoms"]),
                    ("closures", row.get("closures", 0)),
                ):
                    if value > maximum[key]:
                        maximum[key] = value
                        maximum_targets[key] = row["target_id"]

        def finish(entry):
            number, item_hash, future = entry
            records = future.result()
            destination = output / f"shard-{number:05d}"
            with tempfile.TemporaryDirectory(prefix=".shard-", dir=output) as tmp:
                stage = Path(tmp)
                ledger = stage / "rows.jsonl.gz"
                with (
                    ledger.open("wb") as raw,
                    gzip.GzipFile(
                        fileobj=raw, filename="", mode="wb", mtime=0, compresslevel=1
                    ) as stream,
                ):
                    for row in records:
                        stream.write((compact(row) + "\n").encode())
                artifact = pin(ROOT, ledger)
                artifact["path"] = str((destination / ledger.name).relative_to(ROOT))
                report = {
                    "request_sha256": request_pin["sha256"],
                    "items_sha256": item_hash,
                    "rows": len(records),
                    "artifact": artifact,
                    "training_admitted": False,
                }
                dump(stage / "result.json", report)
                os.rename(stage, destination)
            account(records, pin(ROOT, destination / "result.json"))
            print(
                json.dumps(
                    {
                        "rows": sum(c["rows"] for c in families.values()),
                        "failures": len(failures),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

        pending = deque()
        iterator = iter(db.execute(query + "ORDER BY e.family,e.target_id"))
        number = 0
        while items := list(islice(iterator, BATCH_SIZE)):
            item_hash = digest(items)
            destination = output / f"shard-{number:05d}"
            if destination.exists():
                report = json.loads((destination / "result.json").read_text())
                if (
                    report["request_sha256"] != request_pin["sha256"]
                    or report["items_sha256"] != item_hash
                    or report["rows"] != len(items)
                ):
                    raise ValueError("Representation shard does not match restart population")
                ledger = resolve_pin(
                    report["artifact"], ROOT, label="completed representation shard"
                )
                account(list(rows(ledger)), pin(ROOT, destination / "result.json"))
            else:
                pending.append((number, item_hash, pool.submit(evaluate, items)))
                if len(pending) == WORKERS:
                    finish(pending.popleft())
            number += 1
        while pending:
            finish(pending.popleft())
    for family, counts in reader.result["summary"]["by_family"].items():
        if families[family]["rows"] != counts.get("eligible_for_program_preparation", 0):
            raise ValueError("Representation omitted eligible records")
    model_records = {row[0]: row for row in samples}
    for target in maximum_targets.values():
        row = db.execute(query + "WHERE e.target_id=?", (target,)).fetchone()
        model_records[target] = row
    db.close()
    smoke = None
    if not failures:
        graphs = [
            tensorize_sparse_row(
                {"r0_structure_id": r[0], "canonical_isomeric_smiles": r[3]},
                _MAPPING,
                preserve_aromaticity=False,
            )
            for r in model_records.values()
        ]
        # Preserve the full published source size envelope, including larger protected rows.
        smoke_maximum = {**maximum, "atoms": reader.result["summary"]["maximum_heavy_atoms"]}
        smoke = _model_smoke(graphs, vocabulary, smoke_maximum)
    for name, value in {**inputs, **implementation}.items():
        resolve_pin(value, ROOT, label=name)
    dump(output / "failures.json", failures)
    dump(
        output / "result.json",
        {
            "schema_version": "forge.compose_lipid_full_eligible_representation.v1",
            "request": request_pin,
            "shards": receipts,
            "preflight": pin(ROOT, output / "preflight.json"),
            "by_family": {f: dict(c) for f, c in sorted(families.items())},
            "totals": dict(sum(families.values(), Counter())),
            "maximum_observed": maximum,
            "maximum_targets": maximum_targets,
            "model_smoke": smoke,
            "atom_counts": [
                {**dict(state), "eligible_atoms": n} for state, n in sorted(state_counts.items())
            ],
            "failures": pin(ROOT, output / "failures.json"),
            "failure_count": len(failures),
            "pass": not failures and smoke is not None and smoke["finite"],
            "elapsed_seconds": time.monotonic() - started,
            "training_admitted": False,
            "training_calls": 0,
            "final_admitted_training_vocabulary_refit_required": True,
            "joint_synthesis_program_representation_qualified": False,
        },
    )


if __name__ == "__main__":
    main()
