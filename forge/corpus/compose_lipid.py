"""Streaming, split-preserving integration of a pinned COMPOSE-Lipid release.

The SQLite index retains the entire universe and both balanced-release files. Import is not
training admission: source flags and exact-program qualification remain separate, fail-closed
contracts. Only the protected TRAIN view is parsed into molecular graphs.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.assembly.compose_lipid import ComposeLipidError, program_catalogue, role_metadata
from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.core.hashing import is_sha256, resolve_pin, sha256_file
from forge.corpus.library_splits import FrozenIdentityFolds

CONFIG_SCHEMA = "forge.compose_lipid_import_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_import.v1"
SPLITS = {
    "train": "train",
    "calibration": "calibration",
    "test": "heldout",
    "reference": "reference",
}
DATA_INPUTS = ("universe", "accepted", "assignments")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ComposeLipidError(f"{path.name}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ComposeLipidError(f"{path.name}:{number}: expected an object")
            yield row


def _pin(path: Path, repo: Path) -> dict[str, Any]:
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def _count(db: sqlite3.Connection, query: str, args: tuple = ()) -> int:
    return int(db.execute(query, args).fetchone()[0])


def _groups(db: sqlite3.Connection, query: str) -> dict[str, int]:
    return dict(db.execute(query))


def _target(row: Mapping[str, Any], families: set[str], references: set[str]) -> None:
    required = {
        "target_id",
        "constitution",
        "primary_family",
        "primary_metadata",
        "route_families",
        "heavy_atoms",
        "size_disposition",
        "source_anchor",
        "training_admissible",
    }
    if not required.issubset(row):
        raise ComposeLipidError(f"target missing fields: {sorted(required - set(row))}")
    if not is_sha256(row["target_id"]):
        raise ComposeLipidError("target_id must be a full provider digest")
    if not isinstance(row["constitution"], str) or not row["constitution"]:
        raise ComposeLipidError("constitution must be a nonempty string")
    if row["primary_family"] not in families | references:
        raise ComposeLipidError(f"unknown family {row['primary_family']!r}")
    routes = row["route_families"]
    if (
        not isinstance(routes, list)
        or not routes
        or len(routes) != len(set(routes))
        or set(routes) - (families | references)
        or row["primary_family"] not in routes
    ):
        raise ComposeLipidError("invalid route_families")
    if not isinstance(row["primary_metadata"], dict):
        raise ComposeLipidError("primary_metadata must be an object")
    if type(row["source_anchor"]) is not bool or type(row["training_admissible"]) is not bool:
        raise ComposeLipidError("source/admission flags must be boolean")
    if type(row["heavy_atoms"]) is not int or row["heavy_atoms"] < 1:
        raise ComposeLipidError("heavy_atoms must be positive")
    if row["size_disposition"] not in ("model_supported", "above_80_heavy_atom_hold"):
        raise ComposeLipidError("unknown provider size_disposition")
    if row["primary_family"] in references and not row["source_anchor"]:
        raise ComposeLipidError("reference-only row is not a source anchor")


def _csv(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


def _frozen_guards(specs: list[dict], paths: Mapping[str, Path]) -> FrozenIdentityFolds:
    guards = FrozenIdentityFolds()
    identities: dict[str, str] = {}
    for spec in specs:
        lookup = None
        if "identity_input" in spec:
            lookup = {
                row[spec["join_key"]]: row[spec["identity_field"]]
                for row in _csv(paths[spec["identity_input"]])
            }
        records = (
            json.loads(paths[spec["input"]].read_text())
            if spec.get("format") == "json"
            else _csv(paths[spec["input"]])
        )
        for row in records:
            value = (
                lookup[row[spec["join_key"]]] if lookup is not None else row[spec["identity_field"]]
            )
            if spec.get("identity_kind") == "frozen_constitutional_digest":
                if not value.startswith("mol-"):
                    raise ComposeLipidError("invalid historical constitutional digest")
                identity = value[4:]
            elif spec.get("identity_kind") == "constitutional_digest":
                identity = value
            else:
                if value not in identities:
                    canonical, _ = constitutional_molecule(value)
                    identities[value] = hashlib.sha256(canonical.encode()).hexdigest()
                identity = identities[value]
            for column in spec["fold_fields"]:
                if row[column]:
                    fold = row[column]
                    if fold == "quarantine" and spec.get("protect_quarantine") is True:
                        fold = "heldout"
                    guards.add(identity, fold, f"{spec['input']}:{column}")
    return guards


def _database(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA cache_size=-32768;
        PRAGMA temp_store=FILE;
        CREATE TABLE targets (
            target_id TEXT PRIMARY KEY, constitution TEXT NOT NULL,
            family TEXT NOT NULL, source_anchor INTEGER NOT NULL,
            heavy_atoms INTEGER NOT NULL, size_disposition TEXT NOT NULL,
            provider_admissible INTEGER NOT NULL, payload TEXT NOT NULL
        );
        CREATE TABLE accepted (target_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE assignments (
            target_id TEXT PRIMARY KEY, family TEXT NOT NULL, provider_split TEXT NOT NULL,
            forge_split TEXT NOT NULL, payload TEXT NOT NULL
        );
        CREATE TABLE train_checks (
            target_id TEXT PRIMARY KEY, constitution_id TEXT,
            disposition TEXT NOT NULL, reason TEXT NOT NULL,
            role_metadata TEXT NOT NULL
        );
    """)
    return db


def _load_universe(db: sqlite3.Connection, path: Path, catalogue: dict, references: set) -> None:
    batch = []
    for row in _read(path):
        _target(row, set(catalogue), references)
        batch.append(
            (
                row["target_id"],
                row["constitution"],
                row["primary_family"],
                int(row["source_anchor"]),
                row["heavy_atoms"],
                row["size_disposition"],
                int(row["training_admissible"]),
                _json(row),
            )
        )
        if len(batch) == 2048:
            db.executemany("INSERT INTO targets VALUES (?,?,?,?,?,?,?,?)", batch)
            db.commit()
            batch.clear()
    db.executemany("INSERT INTO targets VALUES (?,?,?,?,?,?,?,?)", batch)
    db.commit()


def _load_balanced(db: sqlite3.Connection, paths: dict, catalogue: dict, references: set) -> None:
    for number, row in enumerate(_read(paths["accepted"]), 1):
        _target(row, set(catalogue), references)
        source = db.execute(
            "SELECT payload FROM targets WHERE target_id=?", (row["target_id"],)
        ).fetchone()
        if source is None:
            raise ComposeLipidError("accepted target is absent from universe")
        universe = json.loads(source[0])
        if any(row.get(k) != value for k, value in universe.items()):
            raise ComposeLipidError(f"accepted/universe content mismatch: {row['target_id']}")
        db.execute("INSERT INTO accepted VALUES (?,?)", (row["target_id"], _json(row)))
        if number % 2048 == 0:
            db.commit()
    for number, row in enumerate(_read(paths["assignments"]), 1):
        required = {
            "target_id",
            "family",
            "split",
            "test_panels",
            "combination_signature",
            "core_scaffold_signature",
            "regional_topology_signature",
            "structural_group_signature",
        }
        if not required.issubset(row) or row["split"] not in SPLITS:
            raise ComposeLipidError("invalid assignment schema or split")
        source = db.execute(
            "SELECT t.family FROM targets t JOIN accepted a USING(target_id) WHERE target_id=?",
            (row["target_id"],),
        ).fetchone()
        if source is None or source[0] != row["family"]:
            raise ComposeLipidError("assignment target/family differs from accepted release")
        if (row["family"] in references) != (row["split"] == "reference"):
            raise ComposeLipidError("reference-only families must retain reference split")
        if not isinstance(row["test_panels"], list) or (
            row["test_panels"] and row["split"] != "test"
        ):
            raise ComposeLipidError("test panels assigned outside test split")
        for field in required:
            if field.endswith("_signature") and not isinstance(row[field], str):
                raise ComposeLipidError(f"invalid {field}")
        db.execute(
            "INSERT INTO assignments VALUES (?,?,?,?,?)",
            (row["target_id"], row["family"], row["split"], SPLITS[row["split"]], _json(row)),
        )
        if number % 2048 == 0:
            db.commit()
    db.commit()
    if _count(
        db,
        "SELECT count(*) FROM accepted a LEFT JOIN assignments s USING(target_id) WHERE s.target_id IS NULL",
    ):
        raise ComposeLipidError("balanced release has targets without assignments")


def _inspect_train(db: sqlite3.Connection, guards: FrozenIdentityFolds, catalogue: dict) -> dict:
    issues = Counter()
    query = """SELECT t.target_id,t.payload FROM targets t JOIN assignments a USING(target_id)
               WHERE a.provider_split='train' ORDER BY t.target_id"""
    for number, (target_id, payload) in enumerate(db.execute(query), 1):
        row = json.loads(payload)
        metadata = role_metadata(row, catalogue[row["primary_family"]])
        identity = None
        reason = ""
        try:
            canonical, molecule = constitutional_molecule(row["constitution"])
            identity = hashlib.sha256(canonical.encode()).hexdigest()
            if molecule.GetNumHeavyAtoms() != row["heavy_atoms"]:
                reason = "source_heavy_atom_count_mismatch"
            elif canonical != row["constitution"]:
                reason = "canonicalization_differs"
            elif guards.effective(identity) in ("calibration", "heldout"):
                reason = "historical_identity_holdout"
        except LibraryAssemblyError:
            reason = "invalid_constitutional_graph"
        disposition = "quarantine" if reason else "train"
        if reason:
            issues[reason] += 1
            db.execute(
                "UPDATE assignments SET forge_split='quarantine' WHERE target_id=?", (target_id,)
            )
        db.execute(
            "INSERT INTO train_checks VALUES (?,?,?,?,?)",
            (target_id, identity, disposition, reason, _json(metadata)),
        )
        if number % 2048 == 0:
            db.commit()
    db.commit()
    # Keep every provider record, but never expose duplicate identities to training. In v8 some
    # source anchors and enumerated rows have different target IDs for identical SMILES.
    db.execute("""UPDATE assignments SET forge_split='quarantine'
                  WHERE forge_split='train' AND target_id IN (
                      SELECT t.target_id FROM targets t JOIN duplicate_constitutions d USING(constitution)
                  )""")
    db.execute(
        """UPDATE train_checks SET disposition='quarantine',reason='duplicate_source_constitution'
                  WHERE disposition='train' AND target_id IN (
                      SELECT target_id FROM assignments WHERE forge_split='quarantine'
                  )"""
    )
    db.commit()
    issues = Counter(
        dict(
            db.execute("SELECT reason,count(*) FROM train_checks WHERE reason!='' GROUP BY reason")
        )
    )
    return dict(sorted(issues.items()))


def _probe_graphs(db: sqlite3.Connection, paths: dict, per_family: int) -> dict:
    from forge.corpus.combinatorial_libraries import LibraryRecord, tensorize_library_record
    from forge.model.vocabulary import load_atom_vocabulary

    vocabulary = load_atom_vocabulary(paths["atom_vocabulary"])
    result = {}
    for (family,) in db.execute(
        "SELECT DISTINCT family FROM assignments WHERE forge_split='train' ORDER BY family"
    ):
        probes = []
        rows = db.execute(
            """SELECT t.target_id,t.constitution FROM targets t JOIN assignments a USING(target_id)
                             WHERE a.forge_split='train' AND t.family=? ORDER BY t.target_id LIMIT ?""",
            (family, per_family),
        )
        for target_id, smiles in rows:
            try:
                graph = tensorize_library_record(
                    LibraryRecord(0, family, smiles, (), (), 0.0), vocabulary
                )
                probes.append(
                    {
                        "target_id": target_id,
                        "status": "exact_sparse_graph_roundtrip",
                        "atoms": graph.node_count,
                        "closures": graph.closure_count,
                    }
                )
            except (ValueError, KeyError, RuntimeError) as exc:
                probes.append(
                    {
                        "target_id": target_id,
                        "status": "unsupported_by_existing_representation",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        result[family] = probes
    return result


def import_compose_lipid(repo_root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Authenticate first, build privately, then publish a complete index and receipt atomically."""
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise ComposeLipidError("config/output must be inside repository; output must be fresh")
    config = json.loads(config_path.read_text())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("admission_policy") != "preserve_source_flags_require_exact_program"
    ):
        raise ComposeLipidError("unsupported import schema or admission policy")
    if config.get("sampling_policy") != "equal_family_mass_after_qualification":
        raise ComposeLipidError("raw family-count sampling is not supported")
    if (
        type(config.get("graph_probes_per_family")) is not int
        or config["graph_probes_per_family"] < 1
    ):
        raise ComposeLipidError("graph_probes_per_family must be positive")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    checksum_lines = paths["checksums"].read_text().splitlines()
    checksums = {line.split()[1]: line.split()[0] for line in checksum_lines}
    for name in DATA_INPUTS:
        if checksums.get(paths[name].name) != config["inputs"][name]["sha256"]:
            raise ComposeLipidError(f"provider checksum disagrees with input pin: {name}")
    registry_ids = {
        r["reaction_id"]
        for name in config["registries"]
        for r in json.loads(paths[name].read_text())["reactions"]
    }
    key = json.loads(paths["decomposition_key"].read_text())
    catalogue = program_catalogue(key, config["program_bindings"], registry_ids)
    references = set(config["reference_families"])
    if references & set(catalogue):
        raise ComposeLipidError("reference groups cannot also be program families")
    expected = config["expected"]
    if set(expected["universe_by_family"]) != set(catalogue) | references:
        raise ComposeLipidError("expected universe does not cover every family and reference group")
    guards = _frozen_guards(config["historical_identity_guards"], paths)
    start = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compose-lipid-", dir=output.parent) as temporary:
        work = Path(temporary)
        db = _database(work / "corpus.sqlite")
        try:
            _load_universe(db, paths["universe"], catalogue, references)
            db.executescript("""
                CREATE INDEX target_constitution ON targets(constitution);
                CREATE TABLE duplicate_constitutions AS
                    SELECT constitution,count(*) AS source_rows FROM targets
                    GROUP BY constitution HAVING count(*)>1;
                CREATE UNIQUE INDEX duplicate_identity ON duplicate_constitutions(constitution);
            """)
            _load_balanced(db, paths, catalogue, references)
            counts = {
                "universe_by_family": _groups(
                    db, "SELECT family,count(*) FROM targets GROUP BY family ORDER BY family"
                ),
                "accepted_by_family": _groups(
                    db,
                    "SELECT t.family,count(*) FROM targets t JOIN accepted a USING(target_id) GROUP BY t.family ORDER BY t.family",
                ),
                "provider_splits": _groups(
                    db,
                    "SELECT provider_split,count(*) FROM assignments GROUP BY provider_split ORDER BY provider_split",
                ),
            }
            if counts != expected:
                raise ComposeLipidError(
                    f"release population differs from pinned expectations: {_json(counts)}"
                )
            train_issues = _inspect_train(db, guards, catalogue)
            graph_probes = _probe_graphs(db, paths, config["graph_probes_per_family"])
            db.executescript("""
                CREATE INDEX targets_family ON targets(family,target_id);
                CREATE INDEX assignments_fold ON assignments(forge_split,family,target_id);
                CREATE INDEX train_identity ON train_checks(constitution_id);
            """)
            summary = {
                **counts,
                "universe_rows": _count(db, "SELECT count(*) FROM targets"),
                "distinct_source_constitutions": _count(
                    db, "SELECT count(DISTINCT constitution) FROM targets"
                ),
                "duplicate_constitution_classes": _count(
                    db, "SELECT count(*) FROM duplicate_constitutions"
                ),
                "rows_in_duplicate_constitution_classes": _count(
                    db, "SELECT coalesce(sum(source_rows),0) FROM duplicate_constitutions"
                ),
                "accepted_rows": _count(db, "SELECT count(*) FROM accepted"),
                "assignment_rows": _count(db, "SELECT count(*) FROM assignments"),
                "program_families": len(catalogue),
                "architecture_subfamilies": sum(
                    len(v["architecture_subfamilies"]) for v in key.values()
                ),
                "source_anchors": _count(db, "SELECT count(*) FROM targets WHERE source_anchor=1"),
                "size_dispositions": _groups(
                    db, "SELECT size_disposition,count(*) FROM targets GROUP BY size_disposition"
                ),
                "source_size_label_disagreements_with_readme": _count(
                    db,
                    "SELECT count(*) FROM targets WHERE "
                    "(heavy_atoms>80 AND size_disposition='model_supported') OR "
                    "(heavy_atoms<=80 AND size_disposition='above_80_heavy_atom_hold')",
                ),
                "above_80_atoms_by_source_count": _count(
                    db, "SELECT count(*) FROM targets WHERE heavy_atoms>80"
                ),
                "provider_training_admissible_rows": _count(
                    db, "SELECT count(*) FROM targets WHERE provider_admissible=1"
                ),
                "forge_splits": _groups(
                    db,
                    "SELECT forge_split,count(*) FROM assignments GROUP BY forge_split ORDER BY forge_split",
                ),
                "historical_guard_identities": len(guards.folds),
                "train_quarantine_reasons": train_issues,
                "unassigned_universe_rows": _count(
                    db,
                    "SELECT count(*) FROM targets t LEFT JOIN assignments a USING(target_id) WHERE a.target_id IS NULL",
                ),
                "exact_program_qualified_training_rows": 0,
            }
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ComposeLipidError("SQLite integrity check failed")
            db.commit()
        except sqlite3.IntegrityError as exc:
            raise ComposeLipidError(f"duplicate identity or malformed release join: {exc}") from exc
        finally:
            db.close()
        (work / "program_catalogue.json").write_text(
            json.dumps(catalogue, indent=2, sort_keys=True) + "\n"
        )
        artifacts = {
            name: {
                "path": (output / name).relative_to(repo).as_posix(),
                "sha256": str(sha256_file(work / name)),
            }
            for name in ("corpus.sqlite", "program_catalogue.json")
        }
        source_root = Path(__file__).resolve().parents[2]
        source_names = (
            "forge/corpus/compose_lipid.py",
            "forge/assembly/compose_lipid.py",
            "forge/corpus/library_splits.py",
            "forge/assembly/families.py",
            "forge/corpus/combinatorial_libraries.py",
            "forge/model/sparse_topology_feasibility.py",
            "forge/model/vocabulary.py",
            "forge/core/hashing.py",
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "imported_training_unqualified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _pin(config_path, repo),
            "inputs": config["inputs"],
            "implementation": {
                name: {"path": name, "sha256": str(sha256_file(source_root / name))}
                for name in source_names
            },
            "artifacts": artifacts,
            "summary": summary,
            "graph_probes": graph_probes,
            "sampling_policy": config["sampling_policy"],
            "admission": {
                "source_flags_preserved": True,
                "exact_program_qualification_required": True,
                "training_available": False,
                "training_calls": 0,
                "generation_calls": 0,
            },
            "validation": {
                "input_hashes": True,
                "provider_checksums": True,
                "source_rows_retained": True,
                "balanced_universe_join": True,
                "assignment_join": True,
                "provider_splits_preserved": True,
                "historical_train_guards_applied": True,
                "sqlite_integrity": True,
                "heldout_graphs_parsed": False,
                "full_upstream_split_leakage_audit_reproduced": False,
            },
            "duration_seconds": time.monotonic() - start,
        }
        (work / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(work, output)
    return result


def verify_compose_lipid(repo_root: Path, result_path: Path) -> dict[str, Any]:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "imported_training_unqualified"
    ):
        raise ComposeLipidError("unsupported import receipt")
    config_path = resolve_pin(result["config"], repo, label="config")
    if json.loads(config_path.read_text())["inputs"] != result["inputs"]:
        raise ComposeLipidError("receipt input substitution")
    for group in ("inputs", "implementation", "artifacts"):
        for name, pin in result[group].items():
            resolve_pin(pin, repo, label=name)
    if result["admission"]["training_available"] is not False:
        raise ComposeLipidError("import receipt cannot grant training admission")
    return result


class ComposeLipidCorpus:
    """Read-only indexed corpus. Explicit splits prevent accidental universe-as-TRAIN use."""

    def __init__(self, repo_root: Path, result_path: Path):
        self.repo = repo_root.resolve()
        self.result = verify_compose_lipid(self.repo, result_path)
        self.path = self.repo / self.result["artifacts"]["corpus.sqlite"]["path"]
        self.catalogue = json.loads(
            (self.repo / self.result["artifacts"]["program_catalogue.json"]["path"]).read_text()
        )

    def iter_records(self, *, split: str, family: str | None = None) -> Iterator[dict[str, Any]]:
        if split not in {*SPLITS.values(), "quarantine", "unassigned"}:
            raise ComposeLipidError("select an explicit FORGE split (test is named heldout)")
        known = set(self.catalogue) | set(self.result["summary"]["universe_by_family"])
        if family is not None and family not in known:
            raise ComposeLipidError(f"unknown family {family!r}")
        query = """SELECT t.payload,a.payload,s.payload,c.role_metadata,c.reason
                   FROM targets t LEFT JOIN accepted a USING(target_id)
                   LEFT JOIN assignments s USING(target_id) LEFT JOIN train_checks c USING(target_id)
                   WHERE coalesce(s.forge_split,'unassigned')=?"""
        args = [split]
        if family is not None:
            query += " AND t.family=?"
            args.append(family)
        query += " ORDER BY t.target_id"
        db = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
        try:
            for payload, accepted, assignment, roles, reason in db.execute(query, args):
                source = json.loads(accepted or payload)
                yield {
                    "source": source,
                    "assignment": json.loads(assignment) if assignment else None,
                    "forge_split": split,
                    "quarantine_reason": reason or None,
                    "program_id": self.catalogue.get(source["primary_family"], {}).get(
                        "program_id"
                    ),
                    "role_metadata": json.loads(roles) if roles else None,
                    "exact_program_qualified": False,
                    "training_sampling_weight": 0.0,
                }
        finally:
            db.close()

    def iter_training_records(self) -> Iterator[dict[str, Any]]:
        raise ComposeLipidError(
            "COMPOSE import is not training admission: exact family-specific program qualification is required; provider flags remain unchanged"
        )
