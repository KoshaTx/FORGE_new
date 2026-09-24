"""Recover bounded R1 programs and construct globally disjoint, provenance-preserving partitions."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter, deque
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from rdkit import rdBase

from forge.assembly.families import LibraryAssemblyError, load_assembly_libraries
from forge.assembly.library_programs import LibraryProgramLimits, recover_library_programs
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.combinatorial_libraries import read_library_records
from forge.corpus.library_splits import (
    FOLDS,
    FrozenIdentityFolds,
    component_partitions,
    constitution_id,
    program_partition,
)

CONFIG_SCHEMA = "forge.combinatorial_program_dataset_config.v1"
RESULT_SCHEMA = "forge.combinatorial_program_dataset.v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _csv(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


@lru_cache(maxsize=32768)
def _identity(smiles: str) -> str:
    return constitution_id(smiles)


def _protections(config: Mapping[str, Any], paths: Mapping[str, Path]) -> FrozenIdentityFolds:
    result = FrozenIdentityFolds()
    for spec in config["frozen_identity_sources"]:
        name = spec["input"]
        lookup = None
        if "identity_input" in spec:
            lookup = {
                row[spec["join_key"]]: row[spec["identity_field"]]
                for row in _csv(paths[spec["identity_input"]])
            }
        for row in _csv(paths[name]):
            raw = (
                lookup[row[spec["join_key"]]] if lookup is not None else row[spec["identity_field"]]
            )
            if spec.get("identity_kind") == "frozen_constitutional_digest":
                if not raw.startswith("mol-"):
                    raise LibraryAssemblyError(f"{name}: invalid frozen identity prefix")
                identity = raw[4:]
            else:
                identity = _identity(raw)
            for column in spec["fold_fields"]:
                if row[column]:
                    result.add(identity, row[column], f"{name}:{column}")
    return result


_ADAPTERS: dict = {}
_SPECS: dict = {}
_LIMITS: dict = {}


def _initialize(registries: list, specs: dict, limits: dict) -> None:
    global _ADAPTERS, _SPECS, _LIMITS
    _ADAPTERS = load_assembly_libraries(registries, expected_families=sorted(specs))
    _SPECS, _LIMITS = specs, limits


def _recover(job: tuple[str, str, str, list[str]]) -> dict[str, Any]:
    group_id, family, components_json, targets = job
    spec = _SPECS[family]
    search = recover_library_programs(
        _ADAPTERS[family],
        dict(json.loads(components_json)),
        targets,
        accumulator_role=spec["accumulator_role"],
        limits=LibraryProgramLimits(maximum_steps=spec["maximum_steps"], **_LIMITS),
        append_only_size_bound=spec["append_only_size_bound"],
    )
    return {"group_id": group_id, "components": json.loads(components_json), **asdict(search)}


def _jobs(db: sqlite3.Connection) -> Iterator[tuple[str, str, str, list[str]]]:
    for group_id, family, components in db.execute(
        "SELECT gid,family,components FROM groups ORDER BY gid"
    ):
        targets = [
            r[0]
            for r in db.execute(
                "SELECT DISTINCT smiles FROM records WHERE gid=? ORDER BY smiles", (group_id,)
            )
        ]
        yield group_id, family, components, targets


def _recover_all(db: sqlite3.Connection, workers: int, args: tuple) -> Iterator[dict[str, Any]]:
    jobs = iter(_jobs(db))
    if workers == 1:
        _initialize(*args)
        yield from map(_recover, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize, initargs=args) as pool:
        pending = deque()
        for _ in range(workers * 2):
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(_recover, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(_recover, job))


@contextmanager
def _gzip_text(path: Path):
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as handle,
    ):
        yield handle


def _build_records(
    db: sqlite3.Connection, paths: Mapping[str, Path], adapters: Mapping, blocks: Mapping
) -> Counter:
    db.executescript("""
        CREATE TABLE groups(gid TEXT PRIMARY KEY, family TEXT, components TEXT);
        CREATE TABLE records(source_row INTEGER PRIMARY KEY, gid TEXT, pid TEXT, smiles TEXT, weight REAL);
        CREATE INDEX records_gid ON records(gid);
        CREATE INDEX records_pid ON records(pid);
        CREATE TABLE programs(gid TEXT, pid TEXT, exact INTEGER, qualified INTEGER, depth INTEGER,
                              multiplicity INTEGER, witness TEXT, exposures TEXT, reason TEXT,
                              PRIMARY KEY(gid,pid));
        CREATE TABLE products(pid TEXT PRIMARY KEY, smiles TEXT, family TEXT, fold TEXT,
                              reasons TEXT, weight REAL, source_rows INTEGER, exact INTEGER,
                              qualified INTEGER, witness_gid TEXT, sampling_weight REAL DEFAULT 0);
        CREATE TABLE exposures(pid TEXT, identity TEXT, fold TEXT);
        CREATE INDEX exposure_identity ON exposures(identity);
    """)
    counts = Counter()
    with gzip.open(paths["single_event_ledger"], "rt") as ledger:
        records = read_library_records(paths["corpus"], adapters, blocks)
        for record, line in zip(records, ledger, strict=True):
            item = json.loads(line)
            if item["source_row"] != record.source_row or item["reaction_id"] != record.reaction_id:
                raise LibraryAssemblyError("single-event ledger/source row identity mismatch")
            canonical = item.get("canonical_smiles")
            if not canonical:
                raise LibraryAssemblyError(
                    "program dataset requires valid connected source products"
                )
            components = _json(record.components)
            gid = _digest(record.reaction_id + "|" + components)
            db.execute(
                "INSERT OR IGNORE INTO groups VALUES (?,?,?)", (gid, record.reaction_id, components)
            )
            db.execute(
                "INSERT INTO records VALUES (?,?,?,?,?)",
                (record.source_row, gid, _digest(canonical), canonical, record.realism_weight),
            )
            counts[record.reaction_id] += 1
    db.commit()
    return counts


def _store_program(db: sqlite3.Connection, group: Mapping[str, Any]) -> None:
    component_ids = [_identity(value) for _, value in group["components"]]
    for target in group["targets"]:
        exact = target["minimum_steps"] is not None
        qualified = (
            target["search_complete_through_minimum_depth"]
            and target["policy_path_count_capped_at_two"] == 1
        )
        witness = target["policy_intermediate_products"] or target["intermediate_products"]
        identities = sorted(set(component_ids + [_identity(value) for value in witness]))
        reason = (
            target["unresolved_reason"]
            if not exact
            else (
                "registry_policy_no_minimal_path"
                if not target["policy_path_count_capped_at_two"]
                else "multiple_policy_valid_minimal_paths" if not qualified else ""
            )
        )
        db.execute(
            "INSERT INTO programs VALUES (?,?,?,?,?,?,?,?,?)",
            (
                group["group_id"],
                _digest(target["product_smiles"]),
                int(exact),
                int(qualified),
                target["minimum_steps"],
                target["minimum_path_count_capped_at_two"],
                _json(witness),
                _json(identities),
                reason,
            ),
        )


def _assign_products(
    db: sqlite3.Connection, partitions: Mapping[str, str], frozen: FrozenIdentityFolds
) -> Counter:
    reasons = Counter()
    for pid, smiles, weight, source_rows in db.execute(
        "SELECT pid,MIN(smiles),AVG(weight),COUNT(*) FROM records GROUP BY pid ORDER BY pid"
    ):
        candidates = db.execute(
            """SELECT DISTINCT g.gid,g.family,g.components,p.exact,p.qualified,p.witness,p.exposures
            FROM records r JOIN groups g ON r.gid=g.gid JOIN programs p ON p.gid=r.gid AND p.pid=r.pid WHERE r.pid=? ORDER BY g.gid""",
            (pid,),
        ).fetchall()
        families, folds, issues, exposures = set(), set(), set(), {pid}
        qualified_gid = ""
        for gid, family, components, _, qualified, witness, program_exposures in candidates:
            families.add(family)
            component_ids = [_identity(value) for _, value in json.loads(components)]
            intermediate_ids = [_identity(value) for value in json.loads(witness)]
            fold, why = program_partition(component_ids, pid, intermediate_ids, partitions, frozen)
            folds.add(fold)
            issues.update(why)
            exposures.update(component_ids)
            exposures.update(json.loads(program_exposures))
            if qualified and not qualified_gid:
                qualified_gid = gid
        if len(families) != 1:
            issues.add("multiple_reaction_family_annotations")
        if len(folds) != 1:
            issues.add("duplicate_product_partition_conflict")
        fold = next(iter(folds)) if len(folds) == 1 and len(families) == 1 else "quarantine"
        reasons.update(issues)
        family = sorted(families)[0] if len(families) == 1 else "ambiguous"
        db.execute(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,0)",
            (
                pid,
                smiles,
                family,
                fold,
                _json(sorted(issues)),
                weight,
                source_rows,
                int(any(row[3] for row in candidates)),
                int(bool(qualified_gid)),
                qualified_gid,
            ),
        )
        db.executemany(
            "INSERT INTO exposures VALUES (?,?,?)",
            ((pid, identity, fold) for identity in sorted(exposures)),
        )
    db.commit()
    # Protect intermediate identities and alternative source decompositions as well as leaves.
    conflicts = {
        row[0]
        for row in db.execute(
            """SELECT DISTINCT pid FROM exposures WHERE fold != 'quarantine' AND identity IN
        (SELECT identity FROM exposures WHERE fold != 'quarantine' GROUP BY identity HAVING COUNT(DISTINCT fold)>1)"""
        )
    }
    for pid in sorted(conflicts):
        old = json.loads(
            db.execute("SELECT reasons FROM products WHERE pid=?", (pid,)).fetchone()[0]
        )
        db.execute(
            "UPDATE products SET fold='quarantine',reasons=? WHERE pid=?",
            (_json(sorted(set(old + ["cross_partition_intermediate_or_product"]))), pid),
        )
    db.execute(
        "UPDATE exposures SET fold='quarantine' WHERE pid IN (SELECT pid FROM products WHERE fold='quarantine')"
    )
    reasons["cross_partition_intermediate_or_product"] = len(conflicts)
    db.commit()
    return reasons


def _write_datasets(db: sqlite3.Connection, work: Path) -> dict[str, Any]:
    family_totals = dict(
        db.execute(
            "SELECT family,SUM(weight) FROM products WHERE fold='train' AND qualified=1 AND weight>0 GROUP BY family"
        )
    )
    for family, total in family_totals.items():
        db.execute(
            "UPDATE products SET sampling_weight=weight/? WHERE family=? AND fold='train' AND qualified=1 AND weight>0",
            (total * len(family_totals), family),
        )
    product_fields = (
        "product_id",
        "canonical_smiles",
        "reaction_family",
        "fold",
        "partition_reasons_json",
        "mean_realism_weight",
        "source_row_count",
        "exact_program_recovered",
        "unique_policy_qualified_program_available",
        "representative_program_group",
        "training_sampling_weight",
    )
    with _gzip_text(work / "products.csv.gz") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(product_fields)
        writer.writerows(db.execute("SELECT * FROM products ORDER BY pid"))
    with _gzip_text(work / "row_partitions.csv.gz") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "source_row",
                "program_group_id",
                "product_id",
                "reaction_family",
                "fold",
                "original_realism_weight",
                "exact_program_recovered",
                "unique_policy_qualified_program",
                "minimum_steps",
                "abstention_reason",
            )
        )
        writer.writerows(
            db.execute(
                """SELECT r.source_row,r.gid,r.pid,g.family,p.fold,r.weight,q.exact,q.qualified,q.depth,q.reason
            FROM records r JOIN groups g ON g.gid=r.gid JOIN programs q ON q.gid=r.gid AND q.pid=r.pid JOIN products p ON p.pid=r.pid ORDER BY r.source_row"""
            )
        )
    return {
        "weight_policy": "mean_source_realism_weight_per_constitution_then_equal_family_mass_within_eligible_train",
        "train_family_masses": {
            f: row[0]
            for f in sorted(family_totals)
            for row in db.execute("SELECT SUM(sampling_weight) FROM products WHERE family=?", (f,))
        },
        "source_rows": db.execute("SELECT COUNT(*) FROM records").fetchone()[0],
        "unique_products": db.execute("SELECT COUNT(*) FROM products").fetchone()[0],
        "fold_counts": dict(db.execute("SELECT fold,COUNT(*) FROM products GROUP BY fold")),
        "families": {
            f: {
                "source_rows": db.execute(
                    "SELECT COUNT(*) FROM records r JOIN groups g ON r.gid=g.gid WHERE g.family=?",
                    (f,),
                ).fetchone()[0],
                "exact_program_rows": db.execute(
                    "SELECT COUNT(*) FROM records r JOIN groups g ON r.gid=g.gid JOIN programs p ON p.gid=r.gid AND p.pid=r.pid WHERE g.family=? AND p.exact=1",
                    (f,),
                ).fetchone()[0],
                "qualified_program_rows": db.execute(
                    "SELECT COUNT(*) FROM records r JOIN groups g ON r.gid=g.gid JOIN programs p ON p.gid=r.gid AND p.pid=r.pid WHERE g.family=? AND p.qualified=1",
                    (f,),
                ).fetchone()[0],
                "eligible_unique_products_by_fold": dict(
                    db.execute(
                        "SELECT fold,COUNT(*) FROM products WHERE family=? AND qualified=1 AND weight>0 AND fold!='quarantine' GROUP BY fold",
                        (f,),
                    )
                ),
            }
            for f, in db.execute("SELECT DISTINCT family FROM groups ORDER BY family")
        },
    }


def build_library_program_dataset(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise LibraryAssemblyError("config/output must be inside repo; output must be fresh")
    config = json.loads(config_path.read_text())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("role_policy") != "strict_registry_every_step"
    ):
        raise LibraryAssemblyError("unsupported program schema or role policy")
    if config.get("weight_policy") != "mean_source_realism_weight_then_equal_family_mass":
        raise LibraryAssemblyError("unsupported duplicate/realism weighting policy")
    if type(config.get("workers")) is not int or config["workers"] < 1:
        raise LibraryAssemblyError("workers must be a positive integer")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    registry_pins = [
        (paths[name], config["inputs"][name]["sha256"]) for name in config["registries"]
    ]
    adapters = load_assembly_libraries(registry_pins, expected_families=sorted(config["programs"]))
    for family, spec in config["programs"].items():
        LibraryProgramLimits(maximum_steps=spec["maximum_steps"], **config["limits"])
        if type(spec.get("append_only_size_bound")) is not bool:
            raise LibraryAssemblyError(f"{family}: append_only_size_bound must be boolean")
        role = spec["accumulator_role"]
        if (
            role is None
            and spec["maximum_steps"] != 1
            or role is not None
            and (len(adapters[family].roles) != 2 or role not in adapters[family].roles)
        ):
            raise LibraryAssemblyError(f"invalid program contract: {family}")
    previous = json.loads(paths["single_event_result"].read_text())
    if (
        previous["artifacts"]["row_qualification"]["sha256"]
        != config["inputs"]["single_event_ledger"]["sha256"]
    ):
        raise LibraryAssemblyError("single-event result/ledger pins disagree")
    for name in ("corpus", "building_blocks"):
        if previous["inputs"][name] != config["inputs"][name]:
            raise LibraryAssemblyError(f"single-event source changed: {name}")
    source_root = Path(__file__).resolve().parents[2]
    source_names = (
        "forge/corpus/library_program_dataset.py",
        "forge/corpus/library_splits.py",
        "forge/corpus/combinatorial_libraries.py",
        "forge/assembly/library_programs.py",
        "forge/assembly/families.py",
        "forge/assembly/registry.py",
        "forge/chemistry/reactive_sites.py",
        "forge/core/hashing.py",
        "experiments/phase1/multireaction/library_program_dataset.py",
    )
    sources = {name: str(sha256_file(source_root / name)) for name in source_names}
    config_hash = str(sha256_file(config_path))
    raw_blocks = json.loads(paths["building_blocks"].read_text())["blocks"]
    blocks = {row["block_id"]: row for row in raw_blocks}
    if len(blocks) != len(raw_blocks):
        raise LibraryAssemblyError("duplicate building block identities")
    start = time.monotonic()
    frozen = _protections(config, paths)
    partitions = component_partitions(
        (b["canonical_smiles"] for b in raw_blocks),
        frozen,
        seed=config["seed"],
        fractions=config["split_fractions"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".library-programs-", dir=output.parent) as temporary:
        work = Path(temporary)
        db = sqlite3.connect(work / "work.sqlite")
        counts = _build_records(db, paths, adapters, blocks)
        if dict(counts) != config["expected_rows_by_family"]:
            raise LibraryAssemblyError("source row population differs from frozen configuration")
        terminations, depth_counts, abstentions = Counter(), Counter(), Counter()
        groups = db.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
        print(f"prepared {sum(counts.values())} rows in {groups} precursor groups", flush=True)
        with _gzip_text(work / "programs.jsonl.gz") as handle:
            for index, group in enumerate(
                _recover_all(
                    db, config["workers"], (registry_pins, config["programs"], config["limits"])
                ),
                1,
            ):
                _store_program(db, group)
                terminations[group["termination"]] += 1
                depth_counts.update(
                    str(t["minimum_steps"])
                    for t in group["targets"]
                    if t["minimum_steps"] is not None
                )
                handle.write(_json(group) + "\n")
                if index % 10000 == 0:
                    db.commit()
                    print(f"recovered {index}/{groups} precursor groups", flush=True)
        db.commit()
        abstentions.update(
            dict(
                db.execute("SELECT reason,COUNT(*) FROM programs WHERE reason!='' GROUP BY reason")
            )
        )
        partition_reasons = _assign_products(db, partitions, frozen)
        summary = _write_datasets(db, work)
        overlaps = db.execute(
            "SELECT COUNT(*) FROM (SELECT identity FROM exposures WHERE fold!='quarantine' GROUP BY identity HAVING COUNT(DISTINCT fold)>1)"
        ).fetchone()[0]
        forbidden_train = sum(
            1
            for identity, in db.execute(
                "SELECT DISTINCT identity FROM exposures WHERE fold='train'"
            )
            if frozen.effective(identity) in ("calibration", "heldout")
        )
        all_qualified = (
            sum(f["qualified_program_rows"] for f in summary["families"].values())
            == summary["source_rows"]
        )
        gates = {
            "all_source_rows_preserved": summary["source_rows"]
            == sum(config["expected_rows_by_family"].values()),
            "no_identity_overlap_across_active_partitions": overlaps == 0,
            "frozen_nontraining_identities_excluded_from_training": forbidden_train == 0,
            "all_families_have_eligible_train_calibration_and_heldout": all(
                set(f["eligible_unique_products_by_fold"]) == set(FOLDS)
                for f in summary["families"].values()
            ),
            "all_source_rows_have_unique_registry_qualified_programs": all_qualified,
        }
        component_rows = [
            {
                "block_id": row["block_id"],
                "constitution_id": _identity(row["canonical_smiles"]),
                "fold": partitions[_identity(row["canonical_smiles"])],
                "historical_folds": sorted(
                    frozen.folds.get(_identity(row["canonical_smiles"]), ())
                ),
                "historical_sources": sorted(
                    frozen.sources.get(_identity(row["canonical_smiles"]), ())
                ),
            }
            for row in sorted(raw_blocks, key=lambda x: x["block_id"])
        ]
        (work / "component_partitions.json").write_text(
            json.dumps(component_rows, indent=2, sort_keys=True) + "\n"
        )
        for name, pin in config["inputs"].items():
            resolve_pin(pin, repo, label=name)
        if str(sha256_file(config_path)) != config_hash or any(
            str(sha256_file(source_root / name)) != digest for name, digest in sources.items()
        ):
            raise LibraryAssemblyError("code or configuration changed during dataset construction")
        db.close()
        (work / "work.sqlite").unlink()
        artifacts = {
            p.name: {
                "path": str((output / p.name).relative_to(repo)),
                "sha256": str(sha256_file(p)),
            }
            for p in sorted(work.iterdir())
        }
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "pass" if all(gates.values()) else "blocked",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": config["seed"],
            "config": {"path": str(config_path.relative_to(repo)), "sha256": config_hash},
            "inputs": config["inputs"],
            "sources": sources,
            "artifacts": artifacts,
            "environment": {"rdkit": rdBase.rdkitVersion, "workers": config["workers"]},
            "duration_seconds": time.monotonic() - start,
            "summary": summary,
            "programs": {
                "groups": groups,
                "termination_counts": dict(terminations),
                "minimal_depth_counts_per_group_target": dict(depth_counts),
                "abstention_counts_per_group_target": dict(abstentions),
            },
            "partitions": {
                "component_identities": len(partitions),
                "historical_identity_constraints": len(frozen.folds),
                "historical_conflicting_fold_identities": sum(
                    len(f) > 1 for f in frozen.folds.values()
                ),
                "quarantine_reason_counts": dict(partition_reasons),
                "cross_partition_identity_overlaps": overlaps,
                "protected_identities_in_train": forbidden_train,
            },
            "gates": gates,
            "training_calls": 0,
            "generator_sampling_calls": 0,
            "nonclaims": [
                "Programs are computed minimal constitutional traces, not source-executed syntheses or recovered experimental step order.",
                "Raw replay outside registry policy is diagnostic only; it confers no supervision admission.",
                "Multiple policy-valid minimal paths abstain from unique ordered-program supervision.",
                "Every source row, failed program and mixed-partition combination remains in the ledgers; sampling weights are zero outside eligible training products.",
                "Historical partitions are unchanged. Only identity/fold metadata is used; no R0 blocks or biological labels are extracted.",
                "New partitions establish exact identity disjointness, not chemical similarity-family separation or an untouched external test set.",
                "Use fresh model training for these partitions; past checkpoint exposure is not erased by making new splits.",
                "Passing individual trace checks does not establish twelve-family model performance, realism or L2/L3 closure.",
            ],
        }
        (work / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(work, output)
    return result
