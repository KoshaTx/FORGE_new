"""Full-universe protection accounting, without silently extending either frozen split.

All source rows survive in the ledger. Only the intersection of both existing TRAIN
assignments can reach a molecular consumer. Positive product aliases and exact component
combinations propagate exclusions across the universe; a nonmatch never proves disjointness.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from itertools import zip_longest
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus import compose_lipid_source_view as source_view
from forge.corpus.compose_lipid_protection import KNOWN_FOLDS, PROTECTED_FOLDS
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_full_preparation_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_full_preparation.v1"
POLICY = {
    "seed": 0,
    "record_cap": None,
    "size_filter": None,
    "source_splits_changed": False,
    "unassigned_targets": "pending_partition_before_decomposition",
    "preparation_requires": "old_and_corrected_train_and_no_known_exclusions",
    "component_identity": "global_component_id_with_roles_and_quantities_preserved",
    "combination_protection": "same_family_exact_role_quantity_multiset_of_either_nontrain_split",
    "product_alias_matching": "positive_exact_source_strings_only_nonmatches_unresolved",
    "study_protection": "supplied_selected_studies_and_known_selected_target_pmids",
    "missing_study_metadata": "unresolved_for_reported_source_records",
    "product_molecules_parsed": 0,
    "training_admitted": False,
    "training_calls": 0,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_full_preparation.py",
    *source_view.IMPLEMENTATION,
)


def _load(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Full-preparation configuration or scientific policy changed")
    if set(config.get("inputs", {})) != {"source_view", "family_definitions"}:
        raise ComposeLipidError("Full-preparation inputs changed")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    reader = source_view.SourcePreparationCorpus(repo, paths["source_view"])
    loaded = source_view._load(
        repo, resolve_pin(reader.result["config"], repo, label="source-view config")
    )
    _, inputs, joins, protected, prior, intake = loaded
    protection = json.loads(
        resolve_pin(
            reader.result["inputs"]["product_protection"], repo, label="protection"
        ).read_text()
    )
    previous = resolve_pin(
        protection["inputs"]["prior_construction"], repo, label="historical construction"
    )
    imported = json.loads(
        resolve_pin(protection["inputs"]["import_result"], repo, label="import").read_text()
    )
    if (
        config["inputs"]["family_definitions"]["sha256"]
        != imported["inputs"]["decomposition_key"]["sha256"]
    ):
        raise ComposeLipidError("Full preparation and imported family definitions differ")
    selected = json.loads(inputs["selected_groups"].read_text())
    definitions = json.loads(paths["family_definitions"].read_text())
    if set(definitions) != set(selected["formal_evaluation_families"]):
        raise ComposeLipidError("Formal family coverage differs from the supplied split")
    return config, inputs, joins, protected, prior, intake, previous, selected, reader


def _prepare(db, protected: set[str], prior: dict, previous, selected: dict) -> None:
    """Build small protected-key tables; keep product strings inside the SQL boundary."""
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    db.executescript(
        "CREATE TEMP TABLE blocked_components(id TEXT PRIMARY KEY);"
        "CREATE TEMP TABLE prior_targets(target_id TEXT PRIMARY KEY, identity TEXT, excluded INT);"
        "CREATE TEMP TABLE protected_strings(constitution TEXT PRIMARY KEY);"
        "CREATE TEMP TABLE protected_combinations(family TEXT, instances TEXT, PRIMARY KEY(family,instances));"
        "CREATE TEMP TABLE selected_studies(id TEXT PRIMARY KEY);"
    )
    db.executemany("INSERT INTO blocked_components VALUES (?)", ((v,) for v in sorted(protected)))
    db.executemany(
        "INSERT INTO prior_targets VALUES (?,?,?)",
        (
            (k, v["constitution_id"], not v["eligible_for_program_preparation"])
            for k, v in prior.items()
        ),
    )
    studies = selected["selected_source_studies"]
    if not isinstance(studies, list) or any(
        not isinstance(s, str) or not s.startswith("pmid:") for s in studies
    ):
        raise ComposeLipidError("Invalid selected source-study identities")
    db.executemany("INSERT INTO selected_studies VALUES (?)", ((s,) for s in studies))
    # All earlier protected source products are included, even if absent from the old 200k slice.
    batch = []
    for row in previous:
        fold = row.get("prior_development_split") or "unassigned"
        if fold not in KNOWN_FOLDS:
            raise ComposeLipidError(f"Unknown historical fold: {fold}")
        if fold in PROTECTED_FOLDS:
            if not isinstance(row.get("constitution"), str) or not row["constitution"]:
                raise ComposeLipidError("Missing historical protected source identity")
            batch.append((row["constitution"],))
        if len(batch) >= 2048:
            db.executemany("INSERT OR IGNORE INTO protected_strings VALUES (?)", batch)
            batch.clear()
    db.executemany("INSERT OR IGNORE INTO protected_strings VALUES (?)", batch)
    db.execute(
        "INSERT OR IGNORE INTO protected_strings SELECT t.constitution FROM original.targets t "
        "LEFT JOIN original.assignments o USING(target_id) "
        "LEFT JOIN assignments a USING(target_id) LEFT JOIN prior_targets p USING(target_id) "
        "WHERE (o.forge_split IS NOT NULL AND o.forge_split!='train') "
        "OR (a.split IS NOT NULL AND a.split!='train') OR p.excluded=1"
    )
    db.execute(
        "INSERT OR IGNORE INTO protected_combinations SELECT c.family,c.instances FROM constructions c "
        "LEFT JOIN original.assignments o USING(target_id) LEFT JOIN assignments a USING(target_id) "
        "WHERE (o.forge_split IS NOT NULL AND o.forge_split!='train') "
        "OR (a.split IS NOT NULL AND a.split!='train')"
    )
    db.execute(
        "INSERT OR IGNORE INTO protected_strings SELECT t.constitution FROM constructions c "
        "JOIN original.targets t USING(target_id) WHERE EXISTS "
        "(SELECT 1 FROM json_each(c.instances) j JOIN blocked_components b "
        "ON json_extract(j.value,'$[1]')=b.id)"
    )
    # No payload or molecule is selected. This is metadata from the supplied component manifest.
    db.execute(
        "CREATE TEMP TABLE held_study_targets AS SELECT DISTINCT c.target_id FROM components c, "
        "json_each(c.pmids) p JOIN selected_studies s "
        "ON s.id='pmid:'||replace(CAST(p.value AS TEXT),'pmid:','')"
    )
    db.execute("CREATE UNIQUE INDEX held_study_target_id ON held_study_targets(target_id)")
    db.execute(
        "INSERT OR IGNORE INTO protected_strings SELECT t.constitution FROM held_study_targets h "
        "JOIN original.targets t USING(target_id)"
    )


def classify_full_record(
    *,
    target,
    family,
    old,
    corrected,
    instances,
    blocked,
    prior,
    alias,
    combination,
    held_study,
    source_anchor,
    study_metadata,
    atoms,
    source_line,
    basis,
    historical,
    formal_families,
) -> dict:
    if (old is None) != (corrected is None):
        raise ComposeLipidError("Original and corrected selected target sets differ")
    if type(atoms) is not int or atoms <= 0:
        raise ComposeLipidError("Missing positive source-declared size")
    if old is not None:
        row = source_view.classify(target, family, old, corrected, instances, blocked, prior)
        reasons = row["exclusion_reasons"]
        identity = row["constitution_id"]
        blocked_ids = row["protected_component_ids"]
    else:
        # Validate the identical instance schema without pretending that an assignment exists.
        checked = source_view.classify(
            target, family, "reference", "reference", instances, blocked, None
        )
        blocked_ids = checked["protected_component_ids"]
        identity = None
        reasons = ["known_protected_component"] if blocked_ids else []
    if alias:
        reasons.append("exact_source_alias_of_protected_product")
    if combination:
        reasons.append("exact_component_combination_of_protected_record")
    if held_study:
        reasons.append("selected_protected_source_study")
    pending = []
    if old is None:
        pending.append("unassigned_partition")
    if family not in formal_families:
        reasons.append("reference_category_without_qualified_program")
    if source_anchor and not study_metadata:
        pending.append("source_study_identity_unresolved")
    eligible = not reasons and not pending
    disposition = (
        "protected"
        if reasons
        else ("unresolved_partition_or_study" if pending else "eligible_for_program_preparation")
    )
    return {
        "target_id": target,
        "family": family,
        "old_split": old or "unassigned",
        "corrected_split": corrected or "unassigned",
        "component_instances": instances,
        "protected_component_ids": blocked_ids,
        "constitution_id": identity,
        "source_anchor": bool(source_anchor),
        "source_declared_heavy_atoms": atoms,
        "construction_source_line": source_line,
        "construction_basis": basis,
        "provider_historical_route_claimed": historical,
        "exclusion_reasons": reasons,
        "pending_reasons": pending,
        "disposition": disposition,
        "eligible_for_program_preparation": eligible,
        "source_study_metadata_available": bool(study_metadata),
        "product_identity_disjointness_qualified": False,
        "training_admitted": False,
    }


def full_records(db, protected, prior, previous, selected):
    _prepare(db, protected, prior, previous, selected)
    query = (
        "SELECT t.target_id,t.family,o.forge_split,a.split,c.instances,p.constitution IS NOT NULL,"
        "b.instances IS NOT NULL,h.target_id IS NOT NULL,t.source_anchor,m.pmids,t.heavy_atoms,"
        "c.source_line,c.basis,c.historical_claim,c.family,o.family,a.family "
        "FROM original.targets t LEFT JOIN constructions c USING(target_id) "
        "LEFT JOIN original.assignments o USING(target_id) LEFT JOIN assignments a USING(target_id) "
        "LEFT JOIN protected_strings p ON t.constitution=p.constitution "
        "LEFT JOIN protected_combinations b ON c.family=b.family AND c.instances=b.instances "
        "LEFT JOIN held_study_targets h USING(target_id) LEFT JOIN components m USING(target_id) "
        "ORDER BY t.family,t.target_id"
    )
    for values in db.execute(query):
        (
            target,
            family,
            old,
            corrected,
            component_json,
            alias,
            combination,
            held,
            anchor,
            pmids,
            atoms,
            line,
            basis,
            historical,
            c_family,
            o_family,
            a_family,
        ) = values
        if c_family != family or (old is not None and (o_family != family or a_family != family)):
            raise ComposeLipidError("Full-universe target/construction/assignment join mismatch")
        yield classify_full_record(
            target=target,
            family=family,
            old=old,
            corrected=corrected,
            instances=json.loads(component_json),
            blocked=protected,
            prior=prior.get(target),
            alias=bool(alias),
            combination=bool(combination),
            held_study=bool(held),
            source_anchor=bool(anchor),
            study_metadata=json.loads(pmids) if pmids else [],
            atoms=atoms,
            source_line=line,
            basis=basis,
            historical=json.loads(historical),
            formal_families=set(selected["formal_evaluation_families"]),
        )


class Counts:
    def __init__(self):
        self.families = defaultdict(Counter)
        self.reasons = defaultdict(Counter)
        self.dispositions = defaultdict(Counter)
        self.maximum = defaultdict(int)

    def add(self, row):
        family = row["family"]
        self.families[family].update(
            {
                "rows": 1,
                "eligible_for_program_preparation": row["eligible_for_program_preparation"],
                "unassigned_rows": row["old_split"] == "unassigned",
                "source_declared_above_96_atoms": row["source_declared_heavy_atoms"] > 96,
            }
        )
        self.reasons[family].update(row["exclusion_reasons"] + row["pending_reasons"])
        self.dispositions[family][row["disposition"]] += 1
        self.maximum[family] = max(self.maximum[family], row["source_declared_heavy_atoms"])

    def result(self, expected):
        if {k: v["rows"] for k, v in self.families.items()} != expected:
            raise ComposeLipidError("Full preparation lost or changed source family coverage")
        return {
            "totals": dict(sum(self.families.values(), Counter())),
            "dispositions": dict(sum(self.dispositions.values(), Counter())),
            "reason_counts_overlap": True,
            "by_family": {
                k: {
                    **dict(v),
                    "dispositions": dict(self.dispositions[k]),
                    "reason_counts": dict(self.reasons[k]),
                    "maximum_source_declared_atoms": self.maximum[k],
                }
                for k, v in sorted(self.families.items())
            },
            "full_universe_partition_complete": False,
            "constitutional_disjointness_complete": False,
            "full_size_representation_qualified": False,
            "balanced_training_weights_fitted": False,
            "training_ready": False,
        }


def _records(loaded):
    _, inputs, joins, protected, prior, _, previous, selected, _ = loaded
    with closing(sqlite3.connect(joins.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("ATTACH DATABASE ? AS original", (inputs["corpus"].as_uri() + "?mode=ro",))
        yield from full_records(db, protected, prior, rows(previous), selected)


def build_full_preparation(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Full preparation output must be fresh and inside the repository")
    loaded = _load(repo, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".full-preparation-", dir=output.parent) as temporary:
        stage = Path(temporary)
        counts = Counts()
        ledger = stage / "preparation.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream,
        ):
            for number, row in enumerate(_records(loaded), 1):
                counts.add(row)
                stream.write((compact(row) + "\n").encode())
                if number % 250000 == 0:
                    print(f"Full preparation: {number:,} source records accounted", flush=True)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "full_universe_protected_preparation_only",
            "config": pin(repo, config_path),
            "inputs": loaded[0]["inputs"],
            "implementation": {name: pin(repo, repo / name) for name in IMPLEMENTATION},
            "policy": POLICY,
            "summary": counts.result(loaded[5]["constructions"]["family_counts"]),
            "artifact": {
                "path": str((output / ledger.name).relative_to(repo)),
                "sha256": str(sha256_file(ledger)),
            },
        }
        dump(stage / "result.json", result)
        os.rename(stage, output)
    return result


def verify_full_preparation(repo_root: Path, result_path: Path):
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("policy") != POLICY
        or result.get("status") != "full_universe_protected_preparation_only"
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
    ):
        raise ComposeLipidError("Full preparation receipt scope changed")
    for name, value in result["implementation"].items():
        if resolve_pin(value, repo, label=name) != repo / name:
            raise ComposeLipidError("Full preparation implementation substitution")
    loaded = _load(repo, resolve_pin(result["config"], repo, label="full preparation config"))
    if result["inputs"] != loaded[0]["inputs"]:
        raise ComposeLipidError("Full preparation input substitution")
    ledger = resolve_pin(result["artifact"], repo, label="full preparation ledger")
    counts = Counts()
    eligible = {}
    for actual, expected in zip_longest(rows(ledger), _records(loaded)):
        if actual is None or expected is None or actual != expected:
            raise ComposeLipidError("Full preparation ledger does not reproduce")
        counts.add(actual)
        if actual["eligible_for_program_preparation"]:
            target = actual["target_id"]
            previous = loaded[-1].preparation.get(target)
            if previous is None or any(
                actual[k] != previous[k]
                for k in ("family", "component_instances", "constitution_id")
            ):
                raise ComposeLipidError("Full preparation released an unqualified target")
            if target in eligible:
                raise ComposeLipidError("Duplicate full preparation target")
            eligible[target] = actual
    if result["summary"] != counts.result(loaded[5]["constructions"]["family_counts"]):
        raise ComposeLipidError("Full preparation summary does not reproduce")
    return result, loaded[-1], eligible


class FullPreparationCorpus:
    def __init__(self, repo_root: Path, result_path: Path):
        self.result, self.reader, self.preparation = verify_full_preparation(repo_root, result_path)
        self.precursors = self.reader.precursors

    def iter_preparation_records(self, *, family: str):
        if family not in self.result["summary"]["by_family"]:
            raise ComposeLipidError(f"Unknown full preparation family: {family}")
        with closing(sqlite3.connect(self.reader.corpus.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("CREATE TEMP TABLE eligible(target_id TEXT PRIMARY KEY)")
            db.executemany(
                "INSERT INTO eligible VALUES (?)",
                ((target,) for target, row in self.preparation.items() if row["family"] == family),
            )
            for target, payload in db.execute(
                "SELECT e.target_id,t.payload FROM eligible e JOIN targets t USING(target_id) "
                "ORDER BY e.target_id"
            ):
                yield {"source": json.loads(payload), "preparation": self.preparation[target]}

    def iter_training_records(self):
        raise ComposeLipidError("Full preparation is not a qualified training dataset")
