"""Recover original selected groups by replaying pinned pure source functions on 200k rows.

Only reviewed function definitions are evaluated. The source build driver is never
invoked, upstream files remain byte-identical, and no full-universe group selection occurs.
"""

import ast
import base64
import gzip
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_partition_signatures import SourceSplitSignatures
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("Expected pinned source functions are missing")
    # These unchanged nodes contain the reviewed pure split helpers, not the build driver.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def main():
    start = time.monotonic()
    output = HERE / "historical-selected-groups-v2"
    if output.exists():
        raise ValueError("Historical group recovery output already exists")
    replay_path = HERE / "historical-morphology-replay.json"
    replay = json.loads(replay_path.read_text())
    if not replay["pass"] or replay["failures"]:
        raise ValueError("Independent historical morphology replay must pass")
    for k, v in replay["implementation"].items():
        resolve_pin(v, ROOT, label=k)
    paths = {k: resolve_pin(v, ROOT, label=k) for k, v in replay["inputs"].items()}
    historical = json.loads(paths["historical_policy"].read_text())
    sources = {
        k: resolve_pin(v["file"], ROOT, label=k) for k, v in historical["source_assets"].items()
    }
    tree_path = resolve_pin(historical["source_tree"], ROOT, label="source tree")
    tree = json.loads(tree_path.read_text())
    config_source = "configs/corpus/post_instruction_generator_splits_v8.json"
    config_path = HERE / "upstream" / config_source
    raw = config_path.read_bytes()
    blob = next(n for n in tree["tree"] if n["path"] == config_source)
    response_path = HERE / "old-split-config-blob.json"
    response = json.loads(response_path.read_text())
    git_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if (
        git_hash != blob["sha"]
        or len(raw) != blob["size"]
        or response["sha"] != git_hash
        or base64.b64decode(response["content"]) != raw
    ):
        raise ValueError("Original split configuration authentication failed")
    config = json.loads(raw)
    if any(
        config[k] for k in ("biology_used", "beae_outcomes_used", "gpu_used", "training_admissible")
    ):
        raise ValueError("Original split configuration permits an out-of-scope action")
    signatures = SourceSplitSignatures.from_registry(
        ROOT, paths["registry"], expected_sha256=sha256_file(paths["registry"])
    )
    namespace = {
        "digest": signatures.digest,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "math": math,
    }
    driver = sources["scripts/build_post_instruction_generator_splits_v8.py"]
    module = sources["src/compose_lipid/data/generator_splits.py"]
    functions(driver, {"instance", "virtual_instances", "source_instances"}, namespace)
    names = {n.name for n in ast.parse(module.read_text()).body if isinstance(n, ast.FunctionDef)}
    functions(module, names, namespace)
    intake = json.loads(paths["intake"].read_text())
    construction_path = resolve_pin(
        intake["inputs"]["constructions"], ROOT, label="full source constructions"
    )
    source_inputs = {}
    with closing(sqlite3.connect(paths["corpus"].as_uri() + "?mode=ro", uri=True)) as db:
        anchor_ids = {
            r[0]
            for r in db.execute(
                "SELECT t.target_id FROM targets t JOIN assignments a USING(target_id) WHERE t.source_anchor=1"
            )
        }
    for source in rows(construction_path):
        target = source["target_id"]
        if target not in anchor_ids:
            continue
        aliases = source["declared_precursor_identifiers"]
        if not source["source_anchor"] or any(a["field"] != "source_precursor_id" for a in aliases):
            raise ValueError("Reported precursor alias contract differs")
        source_inputs[target] = namespace["source_instances"](
            [{"roles": a["roles"], "precursor_id": a["value"]} for a in aliases]
        )
        if len(source_inputs) == len(anchor_ids):
            break
    if set(source_inputs) != anchor_ids:
        raise ValueError("Original source anchor precursor IDs are missing")
    bundle = ROOT / "data/source_cache/compose_lipid_supplement_2026-09-19"
    checksum_path = bundle / "SHA256SUMS"
    muscle_rel = "original_generator_tasks/family_05_aldehyde_ugi3_enumeration_v8/muscle_bounded_contexts.jsonl.gz"
    digest = next(
        line.split()[0]
        for line in checksum_path.read_text().splitlines()
        if line.split(maxsplit=1)[1].lstrip(" *") == muscle_rel
    )
    muscle_path = resolve_pin(
        {"path": str((bundle / muscle_rel).relative_to(ROOT)), "sha256": digest},
        ROOT,
        label="original MUSCLE contexts",
    )
    muscle_label = next(
        n.args[0].elts[0].value
        for n in ast.walk(ast.parse(driver.read_text()))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "digest"
        and isinstance(n.args[0], ast.List)
        and isinstance(n.args[0].elts[0], ast.Constant)
        and "complete_component" in str(n.args[0].elts[0].value)
    )
    muscle = {
        row["graph"]["identity_id"]: [
            {
                "role": p["role"],
                "precursor_id": signatures.digest([muscle_label, p["role"], p["smiles"]]),
            }
            for p in row["precursors"]
        ]
        for row in rows(muscle_path)
    }
    records, expected, source_groups = [], {}, {}
    precursor_failures, examples, used_muscle = 0, [], 0
    with closing(sqlite3.connect(paths["joins"].as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("ATTACH DATABASE ? AS original", (paths["corpus"].as_uri() + "?mode=ro",))
        for target, family, raw, assignment, pmids in db.execute(
            "SELECT t.target_id,t.family,t.payload,a.payload,m.pmids FROM original.assignments a "
            "JOIN original.targets t USING(target_id) JOIN components m USING(target_id) ORDER BY t.target_id"
        ):
            source, frozen = json.loads(raw), json.loads(assignment)
            if source["source_anchor"]:
                instances = source_inputs[target]
                source_groups[target] = sorted(
                    {"pmid:" + str(p).removeprefix("pmid:") for p in json.loads(pmids)}
                )
            elif target in muscle:
                instances = muscle[target]
                used_muscle += 1
            else:
                instances = namespace["virtual_instances"](family, source["primary_metadata"])
            row = {
                "target_id": target,
                "family": family,
                "precursor_ids": sorted({i["precursor_id"] for i in instances}),
                "precursor_instances": instances,
                "core_scaffold_signature": frozen["core_scaffold_signature"],
                "regional_topology_signature": frozen["regional_topology_signature"],
            }
            if namespace["combination_signature"](row) != frozen["combination_signature"]:
                precursor_failures += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "target_id": target,
                            "family": family,
                            "source_anchor": source["source_anchor"],
                        }
                    )
            records.append(row)
            expected[target] = frozen
    if len(records) != config["expected_targets"] or used_muscle != 12:
        raise ValueError("Original selected population or MUSCLE override count differs")
    pins = {
        **replay["inputs"],
        "morphology_replay": pin(ROOT, replay_path),
        "original_config": pin(ROOT, config_path),
        "original_config_response": pin(ROOT, response_path),
        "source_constructions": intake["inputs"]["constructions"],
        "muscle_contexts": pin(ROOT, muscle_path),
        "source_checksums": pin(ROOT, checksum_path),
    }
    common = {
        "schema_version": "forge.compose_lipid_historical_selected_recovery.v1",
        "seed": config["seed"],
        "implementation": pin(ROOT, Path(__file__).resolve()),
        "inputs": pins,
        "rows": len(records),
        "source_anchors": len(source_inputs),
        "muscle_overrides": used_muscle,
        "precursor_combination_failures": precursor_failures,
        "examples": examples,
        "upstream_pure_functions_executed": True,
        "upstream_build_driver_executed": False,
        "upstream_source_modified": False,
        "full_universe_group_selection_performed": False,
        "training_admitted": False,
        "source_assignments_changed": False,
    }
    if precursor_failures:
        dump(HERE / "historical-selected-recovery-negative-v2.json", {**common, "pass": False})
        raise ValueError("Original precursor identities do not reproduce old combination keys")
    recovered = namespace["make_generator_splits"](
        records,
        source_groups,
        **{
            k: config[k]
            for k in ("seed", "test_fraction", "calibration_fraction", "minimum_training_class")
        },
    )
    assignments = recovered.pop("assignments")
    failures = [row["target_id"] for row in assignments if row != expected[row["target_id"]]]
    field_failures = Counter()
    failed_rows = []
    for row in assignments:
        frozen = expected[row["target_id"]]
        differences = [key for key in row if row[key] != frozen[key]]
        field_failures.update(differences)
        if differences:
            failed_rows.append({"expected": frozen, "replayed": row, "changed_fields": differences})
    common.update(assignment_failures=len(failures), failed_fields=dict(field_failures))
    for k, v in pins.items():
        resolve_pin(v, ROOT, label=k)
    with tempfile.TemporaryDirectory(prefix=".legacy-groups-", dir=HERE) as temporary:
        stage = Path(temporary)
        dump(stage / "selected_groups.json", recovered)
        dump(stage / "assignment-differences.json", failed_rows)
        with (stage / "records.jsonl.gz").open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
                for row in records:
                    stream.write((compact(row) + "\n").encode())
        artifacts = {}
        for name in ("selected_groups.json", "records.jsonl.gz", "assignment-differences.json"):
            artifacts[name] = {
                "path": str((output / name).relative_to(ROOT)),
                "sha256": sha256_file(stage / name),
            }
        dump(
            stage / "result.json",
            {
                **common,
                "pass": not failures,
                "assignment_failures": len(failures),
                "artifacts": artifacts,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        os.rename(stage, output)


if __name__ == "__main__":
    main()
