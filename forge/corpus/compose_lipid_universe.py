"""Account for the complete COMPOSE universe without admitting unqualified training rows.

Product strings stay inside SQLite for positive overlap checks. Neither evaluation graphs nor
unpartitioned graphs are parsed, decomposed, or returned by this metadata-only preparation stage.
An absent exact-string match never proves constitutional or precursor disjointness.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import closing
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import verify_compose_lipid
from forge.corpus.compose_lipid_precursor_audit import verify_precursor_audit

CONFIG_SCHEMA = "forge.compose_lipid_universe_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_universe_readiness.v1"
STATUS = "complete_universe_accounted_training_unqualified"
POLICY = {
    "seed": 0,
    "random_sampling_used": False,
    "record_cap": None,
    "size_filter": None,
    "graph_parsing": False,
    "source_splits_changed": False,
    "unassigned_records": "pending_partition_before_decomposition",
    "overlap_matching": "positive_exact_source_strings_only",
    "nonmatches": "unresolved_never_assumed_disjoint",
    "duplicate_rows": "retain_provenance_pending_constitutional_weight_deduplication",
    "intended_training_measure": "equal_family_mass_after_all_admission_gates",
    "training_calls": 0,
    "training_rows_admitted": 0,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_universe.py",
    "forge/core/hashing.py",
)
FOLDS = {"train", "calibration", "heldout", "quarantine", "reference"}
EXCLUSION_REASONS = {
    "prior_protected_product",
    "historical_protected_precursor",
    "conflicting_scoped_component_label",
}
AUDIT_REASONS = EXCLUSION_REASONS | {"no_exact_program_evidence", "program_not_exact"}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pin(repo: Path, path: Path) -> dict:
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def _read(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ComposeLipidError(f"non-object universe ledger row: {path}")
            yield row


def _load(repo: Path, config_path: Path) -> tuple[dict, Path, Path]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("universe configuration or scientific policy changed")
    if set(config.get("inputs", {})) != {"import_result", "precursor_audit_result"}:
        raise ComposeLipidError("universe input set changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    imported = verify_compose_lipid(repo, paths["import_result"])
    audit = verify_precursor_audit(repo, paths["precursor_audit_result"])
    # A receipt from a different imported corpus must not be joined by coincidental target IDs.
    protection = json.loads(
        resolve_pin(audit["inputs"]["protection_result"], repo, label="protection").read_text()
    )
    if protection["inputs"]["import_result"] != config["inputs"]["import_result"]:
        raise ComposeLipidError("precursor audit belongs to a different import")
    catalogue = json.loads(
        resolve_pin(
            imported["artifacts"]["program_catalogue.json"], repo, label="catalogue"
        ).read_text()
    )
    expected = config.get("expected_family_rows", {})
    reference = config.get("reference_families", [])
    if (
        not isinstance(expected, dict)
        or not expected
        or any(type(n) is not int or n <= 0 for n in expected.values())
        or not isinstance(reference, list)
        or len(set(reference)) != len(reference)
        or set(reference) & set(catalogue)
        or set(expected) != set(reference) | set(catalogue)
    ):
        raise ComposeLipidError("universe family contract must cover the complete import")
    if expected != imported["summary"]["universe_by_family"]:
        raise ComposeLipidError("universe family counts differ from the authenticated import")
    if sum(expected.values()) != imported["summary"]["universe_rows"]:
        raise ComposeLipidError("universe row count differs from the authenticated import")
    return (
        config,
        resolve_pin(imported["artifacts"]["corpus.sqlite"], repo, label="corpus"),
        resolve_pin(audit["artifacts"]["rows.jsonl.gz"], repo, label="precursor rows"),
    )


def _prepare(db: sqlite3.Connection, audit_rows: Iterable[dict]) -> None:
    """Use disk-backed temporary joins; never copy source molecular strings into Python."""
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-8192")
    db.execute(
        "CREATE TEMP TABLE audit(target_id TEXT PRIMARY KEY, family TEXT NOT NULL, "
        "exact INTEGER NOT NULL, clear INTEGER NOT NULL, reasons TEXT NOT NULL, "
        "excluded INTEGER NOT NULL)"
    )
    batch = []
    for row in audit_rows:
        reasons = row.get("known_exclusion_reasons")
        if (
            not isinstance(row.get("target_id"), str)
            or not row["target_id"]
            or not isinstance(row.get("family"), str)
            or type(row.get("exact_program_evidence")) is not bool
            or type(row.get("clear_of_known_exclusions")) is not bool
            or row.get("training_admitted") is not False
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) or reason not in AUDIT_REASONS for reason in reasons)
            or row["clear_of_known_exclusions"] != (not reasons)
            or (row["clear_of_known_exclusions"] and not row["exact_program_evidence"])
        ):
            raise ComposeLipidError("invalid precursor audit disposition")
        batch.append(
            (
                row["target_id"],
                row["family"],
                row["exact_program_evidence"],
                row["clear_of_known_exclusions"],
                _json(reasons),
                bool(set(reasons) & EXCLUSION_REASONS),
            )
        )
        if len(batch) == 2048:
            db.executemany("INSERT INTO audit VALUES (?,?,?,?,?,?)", batch)
            batch.clear()
    db.executemany("INSERT INTO audit VALUES (?,?,?,?,?,?)", batch)
    if db.execute(
        "SELECT count(*) FROM assignments a LEFT JOIN targets t USING(target_id) "
        "WHERE t.target_id IS NULL OR a.family!=t.family OR a.forge_split NOT IN "
        "('train','calibration','heldout','quarantine','reference') OR a.forge_split IS NULL"
    ).fetchone()[0]:
        raise ComposeLipidError("unknown split or inconsistent universe assignment")
    if (
        db.execute(
            "SELECT count(*) FROM audit c LEFT JOIN assignments a USING(target_id) "
            "WHERE a.target_id IS NULL OR a.forge_split!='train' OR c.family!=a.family"
        ).fetchone()[0]
        or db.execute(
            "SELECT count(*) FROM assignments a LEFT JOIN audit c USING(target_id) "
            "WHERE a.forge_split='train' AND c.target_id IS NULL"
        ).fetchone()[0]
    ):
        raise ComposeLipidError("precursor audit must cover exactly the imported TRAIN population")
    db.execute("CREATE TEMP TABLE protected_strings(constitution TEXT PRIMARY KEY)")
    db.execute(
        "INSERT OR IGNORE INTO protected_strings SELECT t.constitution FROM assignments a "
        "JOIN targets t USING(target_id) WHERE a.forge_split!='train'"
    )
    db.execute(
        "INSERT OR IGNORE INTO protected_strings SELECT t.constitution FROM audit c "
        "JOIN targets t USING(target_id) WHERE c.excluded=1"
    )


def universe_rows(db: sqlite3.Connection, audit_rows: Iterable[dict]) -> Iterator[dict]:
    """Stream every source record, including all sizes, references, and unassigned records."""
    _prepare(db, audit_rows)
    cursor = db.execute(
        "SELECT t.target_id,t.family,t.source_anchor,t.heavy_atoms,t.size_disposition,"
        "a.provider_split,a.forge_split,c.exact,c.clear,c.reasons,"
        "p.constitution IS NOT NULL,coalesce(d.source_rows,1) FROM targets t "
        "LEFT JOIN assignments a USING(target_id) LEFT JOIN audit c USING(target_id) "
        "LEFT JOIN protected_strings p ON t.constitution=p.constitution "
        "LEFT JOIN duplicate_constitutions d ON t.constitution=d.constitution "
        "ORDER BY t.family,t.target_id"
    )
    for (
        target,
        family,
        anchor,
        atoms,
        size,
        provider,
        split,
        exact,
        clear,
        reasons,
        match,
        copies,
    ) in cursor:
        known = json.loads(reasons) if reasons is not None else []
        exclusions = sorted(set(known) & EXCLUSION_REASONS)
        if split in FOLDS - {"train"}:
            disposition = f"excluded_import_{split}"
        elif exclusions:
            disposition = "excluded_by_existing_precursor_audit"
        elif match:
            disposition = "excluded_exact_source_alias_of_known_exclusion"
        elif split is None:
            disposition = "pending_partition_before_decomposition"
        else:
            disposition = "pending_global_qualification"
        yield {
            "target_id": target,
            "family": family,
            "source_anchor": bool(anchor),
            "source_declared_heavy_atoms": atoms,
            "source_size_disposition": size,
            "provider_split": provider if provider is not None else "unassigned",
            "forge_split": split if split is not None else "unassigned",
            "source_string_multiplicity": copies,
            "matches_known_excluded_source_string": bool(match),
            "existing_exact_program_evidence": bool(exact),
            "existing_audit_clear_of_known_exclusions": bool(clear),
            "existing_audit_reasons": known,
            "disposition": disposition,
            "training_admitted": False,
        }


class _Counts:
    def __init__(self) -> None:
        self.families: dict[str, Counter] = defaultdict(Counter)
        self.dispositions: dict[str, Counter] = defaultdict(Counter)
        self.sizes: dict[str, Counter] = defaultdict(Counter)
        self.bounds: dict[str, tuple[int, int]] = {}

    def add(self, row: dict) -> None:
        family = row["family"]
        atoms = row["source_declared_heavy_atoms"]
        if type(atoms) is not int or atoms <= 0:
            raise ComposeLipidError(f"invalid source heavy-atom count: {row['target_id']}")
        low, high = self.bounds.get(family, (atoms, atoms))
        self.bounds[family] = min(low, atoms), max(high, atoms)
        self.families[family].update(
            {
                "rows": 1,
                "outside_200k_selection": row["forge_split"] == "unassigned",
                "existing_exact_program_evidence": row["existing_exact_program_evidence"],
                "existing_audit_clear_of_known_exclusions": row[
                    "existing_audit_clear_of_known_exclusions"
                ],
                "rows_in_repeated_source_string_classes": row["source_string_multiplicity"] > 1,
                "source_declared_above_80_atoms": atoms > 80,
                "source_declared_above_96_atoms": atoms > 96,
                "source_anchor_rows": row["source_anchor"],
                "known_excluded_rows": row["disposition"].startswith("excluded_"),
            }
        )
        self.dispositions[family][row["disposition"]] += 1
        self.sizes[family][row["source_size_disposition"]] += 1

    def result(self, config: dict) -> dict:
        if {key: value["rows"] for key, value in self.families.items()} != config[
            "expected_family_rows"
        ]:
            raise ComposeLipidError("universe census lost or changed family rows")
        return {
            "totals": dict(sorted(sum(self.families.values(), Counter()).items())),
            "by_family": {
                family: {
                    **dict(count),
                    "source_declared_atom_range": list(self.bounds[family]),
                    "dispositions": dict(sorted(self.dispositions[family].items())),
                    "source_size_dispositions": dict(sorted(self.sizes[family].items())),
                    "is_program_family": family not in config["reference_families"],
                }
                for family, count in sorted(self.families.items())
            },
            "program_families": len(self.families) - len(config["reference_families"]),
            "source_reference_categories": len(config["reference_families"]),
            "full_universe_component_holdout_complete": False,
            "full_universe_representation_qualified": False,
            "all_program_families_qualified": False,
            "constitutional_deduplication_complete": False,
            "training_rows_admitted": 0,
            "training_ready": False,
        }


def build_universe_readiness(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo) or not config_path.is_relative_to(repo):
        raise ComposeLipidError("universe output must be fresh and inside the repository")
    config, database, audit = _load(repo, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".universe-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "readiness.jsonl.gz"
        counts = _Counts()
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
            with ledger.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
                    for row in universe_rows(db, _read(audit)):
                        counts.add(row)
                        stream.write((_json(row) + "\n").encode())
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": STATUS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _pin(repo, config_path),
            "inputs": config["inputs"],
            "implementation": {name: _pin(repo, repo / name) for name in IMPLEMENTATION},
            "policy": POLICY,
            "summary": counts.result(config),
            "artifacts": {
                ledger.name: {
                    "path": (output / ledger.name).relative_to(repo).as_posix(),
                    "sha256": str(sha256_file(ledger)),
                }
            },
        }
        (stage / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(stage, output)
    return result


def verify_universe_readiness(repo_root: Path, result_path: Path) -> dict:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != STATUS
        or result.get("policy") != POLICY
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        or set(result.get("artifacts", {})) != {"readiness.jsonl.gz"}
    ):
        raise ComposeLipidError("universe receipt scope changed")
    for name, pin in result["implementation"].items():
        if resolve_pin(pin, repo, label=name) != (repo / name).resolve():
            raise ComposeLipidError("universe implementation substitution")
    config, database, audit = _load(repo, resolve_pin(result["config"], repo, label="config"))
    if result["inputs"] != config["inputs"]:
        raise ComposeLipidError("universe input substitution")
    ledger = resolve_pin(result["artifacts"]["readiness.jsonl.gz"], repo, label="readiness")
    counts = _Counts()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        for actual, expected in zip_longest(_read(ledger), universe_rows(db, _read(audit))):
            if _json(actual) != _json(expected):
                raise ComposeLipidError("universe ledger does not reproduce")
            counts.add(expected)
    if _json(result["summary"]) != _json(counts.result(config)):
        raise ComposeLipidError("universe summary does not reproduce")
    return result
