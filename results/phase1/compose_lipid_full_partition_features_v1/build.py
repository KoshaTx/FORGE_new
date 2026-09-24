"""Compute full-corpus constitutional identities and historical split features on CPU.

Four bounded workers parse every graph without size filtering. The output is an
audited identity/partition index, not a training dataset or a source of components.
"""

import hashlib
import json
import multiprocessing
import os
import runpy
import sqlite3
import tempfile
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path

from rdkit import rdBase

from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import _frozen_guards
from forge.corpus.compose_lipid_historical_morphology import HistoricalMorphology
from forge.corpus.compose_lipid_protection import KNOWN_FOLDS, PROTECTED_FOLDS
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORKERS = 4
BATCH_SIZE = 512


def initialize(policy_path, expected_digest, selected, exceptions, driver, loader):
    global HISTORY, SELECTED, EXCEPTIONS, VIRTUAL
    HISTORY = HistoricalMorphology.from_registry(
        ROOT, Path(policy_path), expected_sha256=expected_digest
    )
    SELECTED = {
        k: set(tuple(v) if isinstance(v, list) else v for v in values)
        for k, values in selected.items()
    }
    EXCEPTIONS = exceptions
    namespace = {}
    runpy.run_path(loader)["functions"](Path(driver), {"instance", "virtual_instances"}, namespace)
    VIRTUAL = namespace["virtual_instances"]


def batch_features(batch):
    output = []
    for (
        target,
        family,
        payload,
        instances_json,
        pmids_json,
        expected_assignment,
        imported_identity,
    ) in batch:
        source = json.loads(payload)
        metadata = source["primary_metadata"]
        instances = json.loads(instances_json)
        policy = HISTORY.corrected.policy
        context = (
            dict(policy["source_anchor_context"])
            if source["source_anchor"]
            else {
                k: v
                for k, v in sorted(metadata.items())
                if k in policy["metadata_fields"] or k.endswith(policy["metadata_suffix"])
            }
        )
        description = {
            "regional_profile": HISTORY.corrected.profile(source["constitution"]),
            "morphology_context": context,
        }
        legacy = EXCEPTIONS.get(target)
        if legacy is None:
            if source["source_anchor"]:
                raise ValueError("Unresolved original source precursor identity")
            legacy = VIRTUAL(family, metadata)
        morphology = HISTORY.describe(
            family=family,
            metadata=metadata,
            source_anchor=source["source_anchor"],
            instances=instances,
            corrected_description=description,
        )
        # The label is extracted from the authenticated original source function.
        combination = HISTORY.corrected.digest(
            [
                SELECTED["combination_label"].copy().pop(),
                family,
                sorted((p["role"], p["precursor_id"]) for p in legacy),
            ]
        )
        structural = morphology["structural_group_signature"]
        studies = {"pmid:" + str(p).removeprefix("pmid:") for p in json.loads(pmids_json or "[]")}
        tests = {
            "unseen_precursor_identity": bool(
                {p["precursor_id"] for p in legacy} & SELECTED["selected_precursors"]
            ),
            "source_study_transfer": any(
                (family, p) in SELECTED["selected_source_studies"] for p in studies
            ),
            "unseen_regional_topology": (family, structural)
            in SELECTED["selected_structural_groups"],
            "unseen_exact_combination": (family, combination)
            in SELECTED["selected_combination_groups"],
        }
        panels = sorted(k for k, hit in tests.items() if hit)
        old_split = (
            "reference"
            if family not in SELECTED["formal_evaluation_families"]
            else (
                "test"
                if panels
                else (
                    "calibration"
                    if (family, structural) in SELECTED["selected_calibration_structural_groups"]
                    or (family, combination) in SELECTED["selected_calibration_combination_groups"]
                    else "train"
                )
            )
        )
        checked = expected_assignment is not None
        if checked:
            expected = json.loads(expected_assignment)
            if (
                any(expected[k] != value for k, value in morphology.items())
                or expected["combination_signature"] != combination
                or expected["split"] != old_split
                or expected["test_panels"] != (panels if old_split == "test" else [])
            ):
                raise ValueError(
                    "Full historical projection changed an original key, assignment or panel"
                )
        canonical, identity, actual_atoms, identity_error = None, None, None, ""
        try:
            canonical, molecule = constitutional_molecule(source["constitution"])
            identity = hashlib.sha256(canonical.encode()).hexdigest()
            actual_atoms = molecule.GetNumHeavyAtoms()
            if actual_atoms != source["heavy_atoms"]:
                identity_error = "source_heavy_atom_count_mismatch"
        except LibraryAssemblyError as exc:
            identity_error = str(exc)
        if imported_identity is not None and identity != imported_identity:
            raise ValueError("Full constitutional identity disagrees with authenticated import")
        output.append(
            (
                target,
                family,
                identity,
                canonical if canonical != source["constitution"] else None,
                actual_atoms,
                identity_error,
                old_split,
                compact(panels if old_split == "test" else []),
                structural,
                combination,
                int(checked),
                int(imported_identity is not None),
                int(source["source_anchor"] and not studies),
            )
        )
    return output


def batches(query):
    while values := query.fetchmany(BATCH_SIZE):
        yield values


def main():
    start = time.monotonic()
    output = HERE / "audit"
    if output.exists():
        raise ValueError("Full partition feature audit already exists")
    recovery_path = (
        ROOT
        / "results/phase1/compose_lipid_split_source_recovery_v2/historical-frozen-group-recovery.json"
    )
    recovery = json.loads(recovery_path.read_text())
    if (
        recovery["pass"] is not True
        or recovery["assignment_failures"]
        or recovery["panel_failures"]
    ):
        raise ValueError("Historical frozen group recovery did not pass")
    for key, value in recovery["inputs"].items():
        resolve_pin(value, ROOT, label=key)
    groups_path = resolve_pin(recovery["artifact"], ROOT, label="recovered immutable groups")
    groups = json.loads(groups_path.read_text())
    selected = {
        k: groups[k]
        for k in (
            "formal_evaluation_families",
            "selected_precursors",
            "selected_source_studies",
            "selected_structural_groups",
            "selected_combination_groups",
            "selected_calibration_structural_groups",
            "selected_calibration_combination_groups",
        )
    }
    corpus = ROOT / recovery["inputs"]["corpus"]["path"]
    joins = ROOT / recovery["inputs"]["joins"]["path"]
    history_path = ROOT / recovery["inputs"]["historical_policy"]["path"]
    history = json.loads(history_path.read_text())
    driver = resolve_pin(
        history["source_assets"]["scripts/build_post_instruction_generator_splits_v8.py"]["file"],
        ROOT,
        label="source precursor identity function",
    )
    module = ROOT / recovery["inputs"]["source_algorithm"]["path"]
    loader = ROOT / recovery["inputs"]["loader"]["path"]
    import ast

    function = next(
        n
        for n in ast.parse(module.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "combination_signature"
    )
    label = next(
        n.args[0].elts[0].value
        for n in ast.walk(function)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "digest"
    )
    selected["combination_label"] = [label]
    with closing(sqlite3.connect(corpus.as_uri() + "?mode=ro", uri=True)) as db:
        exceptional_ids = {
            r[0]
            for r in db.execute(
                "SELECT target_id FROM targets WHERE source_anchor=1 OR json_extract(payload,'$.primary_metadata.design_lane')='separate_muscle_named_context'"
            )
        }
    records_path = ROOT / recovery["inputs"]["records"]["path"]
    exceptions = {
        r["target_id"]: r["precursor_instances"]
        for r in rows(records_path)
        if r["target_id"] in exceptional_ids
    }
    if set(exceptions) != exceptional_ids:
        raise ValueError("Original source or named-context precursor identities are unresolved")
    imported_path = ROOT / "results/phase1/compose_lipid_v8_import_v1/result.json"
    imported = json.loads(imported_path.read_text())
    import_config_path = resolve_pin(imported["config"], ROOT, label="original import config")
    import_config = json.loads(import_config_path.read_text())
    guard_names = {
        spec[key]
        for spec in import_config["historical_identity_guards"]
        for key in ("input", "identity_input")
        if key in spec
    }
    guard_paths = {k: resolve_pin(imported["inputs"][k], ROOT, label=k) for k in guard_names}
    guards = _frozen_guards(import_config["historical_identity_guards"], guard_paths)
    protection_path = ROOT / "results/phase1/compose_lipid_v8_product_protection_v1/result.json"
    protection = json.loads(protection_path.read_text())
    if (
        resolve_pin(protection["inputs"]["import_result"], ROOT, label="protected original import")
        != imported_path
        or imported["artifacts"]["corpus.sqlite"] != recovery["inputs"]["corpus"]
    ):
        raise ValueError("Historical guards and frozen split corpus refer to different imports")
    prior_path = resolve_pin(
        protection["inputs"]["prior_construction"],
        ROOT,
        label="historical source product protection",
    )
    inputs = {
        "historical_group_recovery": pin(ROOT, recovery_path),
        "selected_historical_groups": pin(ROOT, groups_path),
        "corpus": recovery["inputs"]["corpus"],
        "joins": recovery["inputs"]["joins"],
        "historical_policy": pin(ROOT, history_path),
        "source_driver": pin(ROOT, driver),
        "source_module": pin(ROOT, module),
        "source_loader": pin(ROOT, loader),
        "original_records": pin(ROOT, records_path),
        "import_result": pin(ROOT, imported_path),
        "import_config": imported["config"],
        "product_protection": pin(ROOT, protection_path),
        "prior_construction": protection["inputs"]["prior_construction"],
        **{k: imported["inputs"][k] for k in guard_names},
    }
    implementation = {
        name: pin(ROOT, ROOT / name)
        for name in (
            "forge/assembly/families.py",
            "forge/corpus/compose_lipid.py",
            "forge/corpus/library_splits.py",
            "forge/corpus/compose_lipid_partition_signatures.py",
            "forge/corpus/compose_lipid_historical_morphology.py",
            "forge/corpus/compose_lipid_protection.py",
            "forge/corpus/compose_lipid_source_view.py",
            "forge/corpus/compose_lipid_supplement.py",
            "forge/core/hashing.py",
            str(Path(__file__).resolve().relative_to(ROOT)),
        )
    }
    counts, families, old_splits, identity_errors = Counter(), Counter(), Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix=".features-", dir=HERE) as temporary:
        stage = Path(temporary)
        database = stage / "partition_features.sqlite"
        with (
            closing(sqlite3.connect(database)) as out,
            closing(sqlite3.connect(joins.as_uri() + "?mode=ro", uri=True)) as source,
        ):
            out.executescript(
                "PRAGMA temp_store=FILE; PRAGMA cache_size=-32768; CREATE TABLE features(target_id TEXT PRIMARY KEY,family TEXT,constitution_id TEXT,canonical_if_changed TEXT,heavy_atoms INT,identity_error TEXT,old_projection TEXT,old_panels TEXT,old_morphology TEXT,old_combination TEXT,selected_row_verified INT,import_identity_verified INT,source_study_unresolved INT); CREATE TABLE protected_identities(identity TEXT PRIMARY KEY,basis TEXT); CREATE TABLE progress(rows INT);"
            )
            out.executemany(
                "INSERT INTO protected_identities VALUES (?,?)",
                (
                    (identity, "frozen_identity_guard")
                    for identity in sorted(guards.folds)
                    if guards.effective(identity) in {"calibration", "heldout"}
                ),
            )
            prior_failures = []
            for row in rows(prior_path):
                fold = row.get("prior_development_split") or "unassigned"
                if fold not in KNOWN_FOLDS:
                    raise ValueError("Unknown historical source fold")
                if fold in PROTECTED_FOLDS:
                    try:
                        canonical, _ = constitutional_molecule(row["constitution"])
                        identity = hashlib.sha256(canonical.encode()).hexdigest()
                        out.execute(
                            "INSERT OR IGNORE INTO protected_identities VALUES (?,?)",
                            (identity, "historical_source_product"),
                        )
                        counts["historical_protected_rows_canonicalized"] += 1
                    except LibraryAssemblyError as exc:
                        prior_failures.append({"target_id": row["target_id"], "reason": str(exc)})
            source.execute("PRAGMA temp_store=FILE")
            source.execute("PRAGMA cache_size=-32768")
            source.execute("ATTACH DATABASE ? AS original", (corpus.as_uri() + "?mode=ro",))
            query = source.execute(
                "SELECT t.target_id,t.family,t.payload,c.instances,m.pmids,a.payload,i.constitution_id FROM original.targets t JOIN constructions c USING(target_id) LEFT JOIN components m USING(target_id) LEFT JOIN original.assignments a USING(target_id) LEFT JOIN original.train_checks i USING(target_id) ORDER BY t.target_id"
            )
            with ProcessPoolExecutor(
                max_workers=WORKERS,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=(
                    str(history_path),
                    sha256_file(history_path),
                    selected,
                    exceptions,
                    str(driver),
                    str(loader),
                ),
            ) as pool:
                pending = deque()
                source_batches = iter(batches(query))
                for _ in range(WORKERS * 2):
                    batch = next(source_batches, None)
                    if batch is not None:
                        pending.append(pool.submit(batch_features, batch))
                next_log = 100000
                while pending:
                    values = pending.popleft().result()
                    out.executemany(
                        "INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", values
                    )
                    for row in values:
                        counts["rows"] += 1
                        families[row[1]] += 1
                        counts["identity_qualified_rows"] += not bool(row[5])
                        counts["canonical_strings_changed"] += row[3] is not None
                        counts["selected_rows_reproduced"] += row[10]
                        counts["imported_identities_reproduced"] += row[11]
                        counts["source_study_unresolved"] += row[12]
                        counts["above_96_atoms"] += row[4] is not None and row[4] > 96
                        old_splits[row[6]] += 1
                        if row[5]:
                            identity_errors[row[5]] += 1
                    if counts["rows"] >= next_log:
                        out.execute("DELETE FROM progress")
                        out.execute("INSERT INTO progress VALUES (?)", (counts["rows"],))
                        out.commit()
                        print(
                            json.dumps(
                                {
                                    "rows": counts["rows"],
                                    "elapsed_seconds": time.monotonic() - start,
                                    "identity_errors": dict(identity_errors),
                                }
                            ),
                            flush=True,
                        )
                        next_log += 100000
                    batch = next(source_batches, None)
                    if batch is not None:
                        pending.append(pool.submit(batch_features, batch))
            if (
                dict(families) != import_config["expected"]["universe_by_family"]
                or counts["selected_rows_reproduced"] != recovery["source_rows_reproduced"]
            ):
                raise ValueError("Full partition features lost source rows or selected assignments")
            out.execute("CREATE INDEX constitutional_identity ON features(constitution_id)")
            out.execute("CREATE INDEX historical_projection ON features(old_projection)")
            out.execute("DELETE FROM progress")
            out.execute("INSERT INTO progress VALUES (?)", (counts["rows"],))
            out.commit()
            maximum = out.execute("SELECT MAX(heavy_atoms) FROM features").fetchone()[0]
            unique = out.execute("SELECT COUNT(DISTINCT constitution_id) FROM features").fetchone()[
                0
            ]
            guard_hits = out.execute(
                "SELECT COUNT(*) FROM features f JOIN protected_identities p ON f.constitution_id=p.identity"
            ).fetchone()[0]
        for k, value in {**inputs, **implementation}.items():
            resolve_pin(value, ROOT, label=k)
        dump(
            stage / "result.json",
            {
                "schema_version": "forge.compose_lipid_full_partition_features.v1",
                "seed": 0,
                "inputs": inputs,
                "implementation": implementation,
                "artifact": {
                    "path": str((output / database.name).relative_to(ROOT)),
                    "sha256": sha256_file(database),
                },
                "counts": dict(counts),
                "families": dict(families),
                "historical_projection_counts": dict(old_splits),
                "identity_errors": dict(identity_errors),
                "historical_identity_errors": prior_failures,
                "maximum_heavy_atoms": maximum,
                "unique_constitutional_identities": unique,
                "historical_protected_identity_matches": guard_hits,
                "runtime": {
                    "workers": WORKERS,
                    "batch_size": BATCH_SIZE,
                    "maximum_pending_batches": WORKERS * 2,
                    "device": "cpu",
                    "rdkit": rdBase.rdkitVersion,
                    "elapsed_seconds": time.monotonic() - start,
                },
                "policy": {
                    "record_cap": None,
                    "size_filter": None,
                    "source_assignments_changed": False,
                    "full_universe_group_selection_performed": False,
                    "reaction_decomposition_performed": False,
                    "training_calls": 0,
                    "training_admitted": False,
                    "final_partition_qualified": False,
                },
            },
        )
        os.rename(stage, output)


if __name__ == "__main__":
    main()
