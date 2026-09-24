"""List missing first-failure inputs and their existing frozen pins without reading holdouts."""

import gzip
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from forge.core.hashing import sha256_file  # noqa: E402


def pin(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def records(value):
    if isinstance(value, dict):
        for path_key, hash_key in (("path", "sha256"), ("asset", "expected_sha256")):
            if isinstance(value.get(path_key), str) and isinstance(value.get(hash_key), str):
                yield value[path_key], value[hash_key]
        for child in value.values():
            yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)


def main():
    xml = ROOT / "results/phase1/compose_lipid_v8_staar_validation_v1/full-tests.xml.gz"
    missing = defaultdict(list)
    failures = {}
    for case in ET.fromstring(gzip.decompress(xml.read_bytes())).iter("testcase"):
        for kind in ("failure", "error"):
            failure = case.find(kind)
            if failure is None:
                continue
            name = case.get("classname") + "::" + case.get("name")
            message = failure.get("message", "")
            paths = sorted(set(re.findall(re.escape(str(ROOT)) + r"/[^\s'\"<>\)]+", message)))
            absent = []
            for raw in paths:
                path = Path(raw)
                if not path.exists():
                    relative = path.relative_to(ROOT).as_posix()
                    missing[relative].append(name)
                    absent.append(relative)
            failures[name] = {
                "kind": kind,
                "message": message,
                "missing_paths_in_first_error": absent,
            }
    pins = defaultdict(list)
    inputs = {
        str(xml.relative_to(ROOT)): pin(xml),
        str(Path(__file__).relative_to(ROOT)): pin(Path(__file__)),
    }
    # Only configurations are inspected. No molecular holdout, checkpoint or private truth is read.
    for path in sorted((ROOT / "configs").rglob("*.json")):
        for asset, digest in records(json.loads(path.read_text())):
            if asset in missing:
                inputs[path.relative_to(ROOT).as_posix()] = pin(path)
                pins[asset].append(
                    {"configuration": path.relative_to(ROOT).as_posix(), "expected_sha256": digest}
                )
    candidates = [
        Path("/Users/rahulmaganti/Kosha/forge"),
        Path("/Users/rahulmaganti/Kosha/results"),
    ]
    source_paths = {
        asset
        for asset in missing
        if asset.startswith(("src/", "scripts/")) and asset.endswith(".py")
    }
    result = {
        "schema_version": "forge.compose_lipid_test_input_inventory.v1",
        "basis": "first failing preconditions in completed JUnit; not an exhaustive dependency graph or causal diagnosis",
        "random_sampling_used": False,
        "seed": 0,
        "sealed_holdouts_read": False,
        "inputs": inputs,
        "missing_inputs": [
            {
                "path": asset,
                "affected_first_failure_ids": sorted(set(names)),
                "known_configuration_pins": pins[asset],
                "same_relative_path_local_recovery_candidates": [
                    {"path": str(root / asset), "exists": (root / asset).exists()}
                    for root in candidates
                ],
            }
            for asset, names in sorted(missing.items())
        ],
        "failure_preconditions": failures,
        "counts": {
            "failing_or_error_cases": len(failures),
            "distinct_missing_paths_reported": len(missing),
            "missing_legacy_source_paths": len(source_paths),
            "missing_other_artifact_paths": len(set(missing) - source_paths),
            "cases_with_explicit_missing_path": sum(
                bool(row["missing_paths_in_first_error"]) for row in failures.values()
            ),
        },
        "recovery_rule": "Restore exact pinned bytes only. Source relocations require reviewed path/digest bindings; no substitutions, changed pins, skipped tests or regenerated paid jobs.",
    }
    (OUTPUT / "missing-inputs.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["counts"])


if __name__ == "__main__":
    main()
