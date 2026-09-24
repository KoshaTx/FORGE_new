"""Project frozen component exclusions onto source constructions; never admit training rows."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forge.assembly.families import constitutional_molecule  # noqa: E402
from forge.core.hashing import resolve_pin, sha256_file  # noqa: E402
from forge.corpus.compose_lipid import _frozen_guards  # noqa: E402
from forge.corpus.compose_lipid_supplement import rows  # noqa: E402


def pin(path):
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def build_policy():
    imported_path = ROOT / "results/phase1/compose_lipid_v8_import_v1/result.json"
    imported = json.loads(imported_path.read_text())
    config_path = resolve_pin(imported["config"], ROOT, label="frozen import configuration")
    config = json.loads(config_path.read_text())
    specs = config["historical_identity_guards"]
    needed = {spec["input"] for spec in specs} | {
        spec["identity_input"] for spec in specs if "identity_input" in spec
    }
    paths = {name: resolve_pin(imported["inputs"][name], ROOT, label=name) for name in needed}
    guards = _frozen_guards(specs, paths)
    checked_path = OUT / "precursor_split_check.json"
    checked = json.loads(checked_path.read_text())
    precursors = resolve_pin(checked["inputs"]["precursors"], ROOT, label="precursor catalogue")
    selected_path = resolve_pin(
        checked["inputs"]["selected_groups"], ROOT, label="corrected selected groups"
    )
    selected = set(json.loads(selected_path.read_text())["selected_components"])
    blocked = []
    totals = Counter()
    seen = set()
    for row in rows(precursors):
        identity = row["component_id"]
        if identity in seen:
            raise ValueError("Duplicate precursor identity")
        seen.add(identity)
        canonical, _ = constitutional_molecule(row["constitution"])
        historical = guards.effective(hashlib.sha256(canonical.encode()).hexdigest())
        protected = historical in {"calibration", "heldout"}
        selected_test = identity in selected
        totals["catalogue_components"] += 1
        totals["historically_protected_components"] += protected
        totals["corrected_test_components"] += selected_test
        if protected or selected_test:
            blocked.append(
                {
                    "component_id": identity,
                    "historical_fold": historical,
                    "historically_protected": protected,
                    "corrected_test_component": selected_test,
                }
            )
    if not selected <= seen:
        raise ValueError("Held component absent from catalogue")
    totals["union_protected_components"] = len(blocked)
    result = {
        "schema_version": "forge.compose_lipid_supplement_component_exclusion_policy.v1",
        "inputs": {
            "import_result": pin(imported_path),
            "import_config": pin(config_path),
            "precursors": pin(precursors),
            "selected_groups": pin(selected_path),
            **{name: pin(path) for name, path in sorted(paths.items())},
        },
        "implementation": {
            "script": pin(Path(__file__)),
            "frozen_guards": pin(ROOT / "forge/corpus/compose_lipid.py"),
            "constitutional_identity": pin(ROOT / "forge/assembly/families.py"),
            "fold_policy": pin(ROOT / "forge/corpus/library_splits.py"),
        },
        "totals": dict(totals),
        "blocked_components": blocked,
        "seed": 0,
        "training_admitted": False,
        "scope": "Union of frozen historical calibration/heldout molecular identities and corrected v8.1 selected test component identities; no new support fitting or decomposition.",
    }
    (OUT / "component_exclusion_policy.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["totals"], indent=2), flush=True)
    return result


def project(policy):
    intake_path = OUT / "result.json"
    intake = json.loads(intake_path.read_text())
    source = resolve_pin(
        intake["artifacts"]["joins.sqlite"], ROOT, label="verified construction joins"
    )
    final = OUT / "component_exclusions.sqlite"
    if final.exists():
        raise ValueError("Component exclusion ledger already published")
    with tempfile.TemporaryDirectory(prefix=".intake-protection-", dir=OUT) as temporary:
        database = Path(temporary) / "exclusions.sqlite"
        with closing(sqlite3.connect(database, uri=True)) as db:
            db.execute("ATTACH DATABASE ? AS source", (source.as_uri() + "?mode=ro",))
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA cache_size=-16384")
            db.execute(
                "CREATE TABLE protected_components(component_id TEXT PRIMARY KEY,historical INTEGER,corrected_test INTEGER)"
            )
            db.executemany(
                "INSERT INTO protected_components VALUES (?,?,?)",
                [
                    (
                        row["component_id"],
                        row["historically_protected"],
                        row["corrected_test_component"],
                    )
                    for row in policy["blocked_components"]
                ],
            )
            db.execute(
                "CREATE TABLE excluded_targets(target_id TEXT PRIMARY KEY,family TEXT,historical INTEGER,corrected_test INTEGER)"
            )
            db.execute(
                "INSERT INTO excluded_targets SELECT c.target_id,c.family,max(p.historical),max(p.corrected_test) FROM source.constructions c JOIN json_each(c.instances) j JOIN protected_components p ON p.component_id=json_extract(j.value,'$[1]') GROUP BY c.target_id,c.family"
            )
            by_family = [
                dict(
                    family=family,
                    excluded_targets=count,
                    historical_targets=historical,
                    corrected_test_component_targets=corrected,
                )
                for family, count, historical, corrected in db.execute(
                    "SELECT family,count(*),sum(historical),sum(corrected_test) FROM excluded_targets GROUP BY family ORDER BY family"
                )
            ]
            total, historical, corrected = db.execute(
                "SELECT count(*),sum(historical),sum(corrected_test) FROM excluded_targets"
            ).fetchone()
            if corrected != intake["full_universe_selected_test_component_overlap"]["rows"]:
                raise ValueError("Independent full-universe held-component count disagrees")
            db.commit()
        database.replace(final)
    result = {
        "schema_version": "forge.compose_lipid_supplement_component_exclusions.v1",
        "inputs": {
            "intake": pin(intake_path),
            "constructions": pin(source),
            "policy": pin(OUT / "component_exclusion_policy.json"),
        },
        "implementation": pin(Path(__file__)),
        "artifact": pin(final),
        "totals": {
            "union_excluded_targets": total,
            "historically_protected_component_targets": historical,
            "corrected_test_component_targets": corrected,
        },
        "by_family": by_family,
        "seed": 0,
        "training_admitted": False,
        "scope": "Known component exclusions across the full universe. Product, study, combination/morphology, source qualification and representation gates remain separate; nonmatches are not admitted.",
    }
    (OUT / "component_exclusions.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["totals"], indent=2), flush=True)


if __name__ == "__main__":
    policy = build_policy()
    if (OUT / "result.json").exists():
        project(policy)
