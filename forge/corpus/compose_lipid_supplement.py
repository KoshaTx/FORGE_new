"""Authenticate supplemental construction joins without admitting training supervision.

Only precursor molecules are parsed. Product graphs are neither interpreted nor decomposed.
Producer replay claims remain source metadata; this intake does not certify reaction chemistry.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path

from rdkit import Chem

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file


def rows(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as stream:
        for number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ComposeLipidError(f"Non-object row {number}: {path}")
            yield value


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def text_field(row: dict, name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ComposeLipidError(f"Missing or invalid {name}")
    return value


def catalogue(records: Iterable[dict]) -> tuple[dict[str, str], dict]:
    structures = {}
    reverse = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    cross_family: dict[str, set[str]] = defaultdict(set)
    for row in records:
        identity = text_field(row, "component_id")
        constitution = text_field(row, "constitution")
        if identity in structures:
            raise ComposeLipidError("Duplicate precursor component_id")
        molecule = Chem.MolFromSmiles(constitution)
        if molecule is None:
            raise ComposeLipidError(f"Invalid precursor structure: {identity}")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        if canonical != constitution:
            raise ComposeLipidError(f"Noncanonical precursor structure: {identity}")
        if canonical in reverse:
            raise ComposeLipidError("One precursor structure has multiple global identities")
        reverse[canonical] = identity
        structures[identity] = canonical
        for context in row.get("contexts", []):
            cross_family[identity].add(text_field(context, "family"))
            text_field(context, "role")
        for alias in row.get("scoped_aliases", []):
            scoped = "|".join(text_field(alias, field) for field in ("family", "field", "value"))
            if alias.get("scoped_alias") != scoped:
                raise ComposeLipidError("Malformed family-scoped precursor alias")
            aliases[scoped].add(identity)
    return structures, {
        "precursor_structures": len(structures),
        "unique_canonical_precursor_structures": len(reverse),
        "cross_family_precursors": sum(len(families) > 1 for families in cross_family.values()),
        "scoped_aliases": len(aliases),
        "ambiguous_scoped_aliases": sum(len(values) > 1 for values in aliases.values()),
        "join_key": "component_id",
    }


def instances(value: object, structures: dict[str, str], *, quantities: bool) -> str:
    """Return a role-aware multiset; quantities must never become duplicated weight."""
    if not isinstance(value, list) or not value:
        raise ComposeLipidError("Missing component_instances")
    counts: Counter[tuple[str, str]] = Counter()
    for item in value:
        if not isinstance(item, dict):
            raise ComposeLipidError("Invalid component instance")
        identity = text_field(item, "component_id")
        role = text_field(item, "role")
        if identity not in structures:
            raise ComposeLipidError(f"Unresolved component_id: {identity}")
        quantity = item.get("quantity") if quantities else 1
        if type(quantity) is not int or quantity < 1:
            raise ComposeLipidError("Component quantity must be a positive integer")
        if "constitution" in item and item["constitution"] != structures[identity]:
            raise ComposeLipidError("Component identity/structure disagreement")
        key = (role, identity)
        if quantities and key in counts:
            raise ComposeLipidError("Repeated collapsed component instance")
        counts[key] += quantity
    return compact([[role, identity, count] for (role, identity), count in sorted(counts.items())])


def construction(row: dict, structures: dict[str, str], line: int) -> tuple:
    basis = text_field(row, "construction_basis")
    if type(row.get("source_anchor")) is not bool:
        raise ComposeLipidError("Invalid construction source_anchor")
    if type(row.get("family_definition_sufficient")) is not bool:
        raise ComposeLipidError("Invalid family_definition_sufficient")
    steps = row.get("reaction_steps_sites")
    if not isinstance(steps, dict):
        raise ComposeLipidError("Missing reaction_steps_sites")
    historical = steps.get("historical_route_claimed")
    if basis == "lnpdb_compatible_decomposition_replay" and historical is not False:
        raise ComposeLipidError(
            "Compatible decomposition must explicitly disclaim historical route"
        )
    if not isinstance(row.get("provenance"), dict) or not row["provenance"]:
        raise ComposeLipidError("Missing construction provenance")
    return (
        text_field(row, "target_id"),
        text_field(row, "primary_family"),
        row["source_anchor"],
        basis,
        instances(row.get("component_instances"), structures, quantities=True),
        compact(historical),
        line,
    )


def prepare_split(
    db: sqlite3.Connection, paths: dict[str, Path], structures: dict[str, str]
) -> dict:
    db.executescript(
        "CREATE TABLE components(target_id TEXT PRIMARY KEY,family TEXT,instances TEXT,ids TEXT,pmids TEXT);"
        "CREATE TABLE assignments(target_id TEXT PRIMARY KEY,family TEXT,split TEXT,panels TEXT,combination TEXT,morphology TEXT);"
        "CREATE TABLE groups(target_id TEXT PRIMARY KEY,family TEXT,instances TEXT,ids TEXT,pmids TEXT);"
    )
    for row in rows(paths["components"]):
        normalized = instances(row.get("component_instances"), structures, quantities=False)
        identities = sorted({entry[1] for entry in json.loads(normalized)})
        if row.get("component_ids") != identities:
            raise ComposeLipidError("Component manifest ID set disagrees with instances")
        db.execute(
            "INSERT INTO components VALUES (?,?,?,?,?)",
            (
                text_field(row, "target_id"),
                text_field(row, "family"),
                normalized,
                compact(identities),
                compact(row.get("source_pmids", [])),
            ),
        )
    split_counts: Counter[str] = Counter()
    panel_counts: Counter[str] = Counter()
    for row in rows(paths["assignments"]):
        split = row.get("split")
        panels = row.get("test_panels")
        if split not in {"train", "calibration", "test", "reference"} or not isinstance(
            panels, list
        ):
            raise ComposeLipidError("Invalid corrected split")
        if bool(panels) != (split == "test") or len(set(panels)) != len(panels):
            raise ComposeLipidError("Incorrect test-panel assignment")
        db.execute(
            "INSERT INTO assignments VALUES (?,?,?,?,?,?)",
            (
                text_field(row, "target_id"),
                text_field(row, "family"),
                split,
                compact(panels),
                text_field(row, "combination_signature"),
                text_field(row, "morphology_group_signature"),
            ),
        )
        split_counts[split] += 1
        panel_counts.update(panels)
    for row in rows(paths["groups"]):
        normalized = instances(row.get("component_instances"), structures, quantities=False)
        identities = sorted({entry[1] for entry in json.loads(normalized)})
        if row.get("component_ids") != identities:
            raise ComposeLipidError("Split group ID set disagrees with instances")
        db.execute(
            "INSERT INTO groups VALUES (?,?,?,?,?)",
            (
                text_field(row, "target_id"),
                text_field(row, "family"),
                normalized,
                compact(identities),
                compact(row.get("source_pmids", [])),
            ),
        )
    for table in ("components", "groups"):
        if db.execute(
            f"SELECT count(*) FROM assignments a FULL OUTER JOIN {table} c USING(target_id) WHERE a.target_id IS NULL OR c.target_id IS NULL OR a.family!=c.family"
        ).fetchone()[0]:
            raise ComposeLipidError("Corrected split membership/family mismatch")
    if db.execute(
        "SELECT count(*) FROM components c JOIN groups g USING(target_id) WHERE c.instances!=g.instances OR c.pmids!=g.pmids"
    ).fetchone()[0]:
        raise ComposeLipidError("Split groups disagree with component manifest")
    declared = json.loads(paths["split_summary"].read_text())
    if (
        dict(split_counts) != declared["split_counts"]
        or dict(panel_counts) != declared["panel_counts"]
    ):
        raise ComposeLipidError("Corrected split summary disagrees with rows")
    selected = json.loads(paths["selected_groups"].read_text())
    held_components = set(selected["selected_components"])
    if not held_components <= structures.keys():
        raise ComposeLipidError("Selected held component absent from catalogue")
    train_components = set()
    train_family_components = set()
    train_combinations = set()
    train_morphologies = set()
    train_studies = set()
    query = "SELECT a.family,a.combination,a.morphology,c.ids,c.pmids FROM assignments a JOIN components c USING(target_id) WHERE split='train'"
    for family, combination_id, morphology, ids, pmids in db.execute(query):
        identities = json.loads(ids)
        train_components.update(identities)
        train_family_components.update((family, identity) for identity in identities)
        train_combinations.add((family, combination_id))
        train_morphologies.add((family, morphology))
        train_studies.update(
            "pmid:" + str(pmid).removeprefix("pmid:") for pmid in json.loads(pmids)
        )
    selected_studies = set(selected["selected_source_studies"])
    failures: Counter[str] = Counter()
    query = "SELECT a.family,a.combination,a.morphology,c.ids,c.pmids,a.panels FROM assignments a JOIN components c USING(target_id) WHERE split='test'"
    for family, combination_id, morphology, ids, pmids, panels in db.execute(query):
        identities = set(json.loads(ids))
        panels = set(json.loads(panels))
        if "unseen_component_structure" in panels and not identities - train_components:
            failures["component_panel_without_unseen_structure"] += 1
        if "unseen_exact_component_combination" in panels:
            if (family, combination_id) in train_combinations:
                failures["combination_seen_in_train"] += 1
            if any((family, identity) not in train_family_components for identity in identities):
                failures["combination_component_unseen_in_family_train"] += 1
        if "unseen_regional_morphology" in panels and (family, morphology) in train_morphologies:
            failures["morphology_seen_in_train"] += 1
        if "source_study_transfer_global" in panels:
            studies = {"pmid:" + str(pmid).removeprefix("pmid:") for pmid in json.loads(pmids)}
            if not studies & selected_studies:
                failures["study_panel_without_held_study"] += 1
    failures["selected_component_leaks"] = len(held_components & train_components)
    failures["selected_study_leaks"] = len(selected_studies & train_studies)
    if any(failures.values()):
        raise ComposeLipidError(f"Corrected split audit failed: {dict(failures)}")
    return {
        "split_counts": dict(split_counts),
        "panel_counts": dict(panel_counts),
        "selected_held_components": len(held_components),
        "selected_held_studies": len(selected_studies),
        "panel_failures": dict(failures),
        "product_molecules_parsed": 0,
        "scope": "200k structural component/study and supplied group-signature semantics; no product morphology recomputation",
    }


def build_intake(repo: Path, config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text())
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "joins.sqlite").exists():
        raise ComposeLipidError("Intake output already published")
    structures, precursor_summary = catalogue(rows(paths["precursors"]))
    declared = json.loads(paths["export_summary"].read_text())
    if len(structures) != declared["precursor_structures"]:
        raise ComposeLipidError("Precursor count differs from export summary")
    with tempfile.TemporaryDirectory(prefix=".intake-", dir=output) as temporary:
        database = Path(temporary) / "joins.sqlite"
        with closing(sqlite3.connect(database, uri=True)) as db:
            db.execute("PRAGMA cache_size=-32768")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("ATTACH DATABASE ? AS original", (paths["corpus"].as_uri() + "?mode=ro",))
            split_summary = prepare_split(db, paths, structures)
            (output / "split_diagnostic.json").write_text(
                json.dumps(
                    {
                        "inputs": config["inputs"],
                        "precursors": precursor_summary,
                        "split": split_summary,
                        "training_admitted": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            db.execute(
                "CREATE TABLE constructions(target_id TEXT PRIMARY KEY,family TEXT,source_anchor INTEGER,basis TEXT,instances TEXT,historical_claim TEXT,source_line INTEGER)"
            )
            family_counts: Counter[str] = Counter()
            bases: Counter[str] = Counter()
            references = quantity_total = 0
            batch = []
            held_ids = set(json.loads(paths["selected_groups"].read_text())["selected_components"])
            held_hits: Counter[str] = Counter()
            for number, row in enumerate(rows(paths["constructions"]), 1):
                normalized = construction(row, structures, number)
                family_counts[normalized[1]] += 1
                bases[normalized[3]] += 1
                values = json.loads(normalized[4])
                references += len(values)
                quantity_total += sum(value[2] for value in values)
                if any(value[1] in held_ids for value in values):
                    held_hits[normalized[1]] += 1
                batch.append(normalized)
                if len(batch) == 2048:
                    db.executemany("INSERT INTO constructions VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
                if number % 250000 == 0:
                    print(f"Checked construction joins: {number:,}", flush=True)
            db.executemany("INSERT INTO constructions VALUES (?,?,?,?,?,?,?)", batch)
            if (
                dict(family_counts) != declared["primary_family_counts"]
                or dict(bases) != declared["construction_basis_counts"]
            ):
                raise ComposeLipidError("Full construction counts differ from export summary")
            checks = {
                "missing_or_extra_targets": "SELECT count(*) FROM constructions c FULL OUTER JOIN original.targets t USING(target_id) WHERE c.target_id IS NULL OR t.target_id IS NULL",
                "target_metadata_mismatches": "SELECT count(*) FROM constructions c JOIN original.targets t USING(target_id) WHERE c.family!=t.family OR c.source_anchor!=t.source_anchor",
                "selected_component_multiset_mismatches": "SELECT count(*) FROM components m LEFT JOIN constructions c USING(target_id) WHERE c.target_id IS NULL OR m.family!=c.family OR m.instances!=c.instances",
                "changed_selected_membership": "SELECT count(*) FROM assignments a FULL OUTER JOIN original.assignments o USING(target_id) WHERE a.target_id IS NULL OR o.target_id IS NULL OR a.family!=o.family",
            }
            checked = {name: db.execute(sql).fetchone()[0] for name, sql in checks.items()}
            if any(checked.values()):
                raise ComposeLipidError(f"Supplement join checks failed: {checked}")
            transitions = [
                dict(old=old, corrected=new, rows=count)
                for old, new, count in db.execute(
                    "SELECT o.forge_split,a.split,count(*) FROM assignments a JOIN original.assignments o USING(target_id) GROUP BY o.forge_split,a.split ORDER BY o.forge_split,a.split"
                )
            ]
            db.commit()
        database.replace(output / "joins.sqlite")
    result = {
        "schema_version": "forge.compose_lipid_supplement_intake.v1",
        "status": "target_component_joins_verified_training_unqualified",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": str(sha256_file(config_path)),
        },
        "inputs": config["inputs"],
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(repo)),
            "sha256": str(sha256_file(Path(__file__))),
        },
        "artifacts": {
            "joins.sqlite": {
                "path": str((output / "joins.sqlite").relative_to(repo)),
                "sha256": str(sha256_file(output / "joins.sqlite")),
            }
        },
        "precursors": precursor_summary,
        "corrected_split": split_summary,
        "constructions": {
            "rows": sum(family_counts.values()),
            "family_counts": dict(family_counts),
            "basis_counts": dict(bases),
            "component_instance_references": references,
            "quantity_expanded_references": quantity_total,
            "join_failures": checked,
        },
        "full_universe_selected_test_component_overlap": {
            "rows": sum(held_hits.values()),
            "by_family": dict(held_hits),
            "purpose": "protection diagnostic; not a final training partition",
        },
        "old_to_corrected_assignment_counts": transitions,
        "policy": {
            "seed": 0,
            "product_molecules_parsed": 0,
            "precursor_graphs_used_only_for_identity_audit": True,
            "bare_reagent_codes_used_as_keys": False,
            "old_splits_modified": False,
            "training_rows_admitted": 0,
            "training_calls": 0,
            "historical_route_claims_inferred": False,
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
