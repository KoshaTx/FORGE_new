"""Preserve the final bounded acquisition state, source pins and validation limits."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from forge.core.hashing import pin_record
from forge.core.io import write_json

repo = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def check(pin):
    path = Path(pin["path"])
    path = path if path.is_absolute() else repo / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
        raise ValueError(f"File changed: {path}")


def tests(path):
    root = ET.parse(path).getroot()
    counts = Counter()
    failures = set()
    for case in root.iter("testcase"):
        counts["tests"] += 1
        kind = next(
            (k for k in ("failure", "error", "skipped") if case.find(k) is not None), "passed"
        )
        counts[kind] += 1
        if kind in ("failure", "error"):
            failures.add((case.attrib["classname"], case.attrib["name"], kind))
    return dict(counts), failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    destination = out / "closeout.json"
    if args.verify:
        result = read(destination)
        for pin in result["files"]:
            check(pin)
        print("Final evidence, qualification and validation pins verified")
        return
    if destination.exists():
        raise ValueError("Preserve the existing closeout")
    baseline_path = repo / "results/phase1/ugi_guidance_artifact_recovery_v2/full-tests.xml"
    baseline, old_failures = tests(baseline_path)
    current, failures = tests(out / "full-tests.xml")
    focused, focused_failures = tests(out / "focused-tests.xml")
    qualification = read(out / "cyclohexylamine_terminal_qualification.json")
    for item in qualification["inputs"].values():
        check(item)
    check(qualification["artifact"])
    if not all(qualification["checks"].values()):
        raise ValueError("Terminal qualification check failed")
    stock = read(out / "supplier_stock_summary.json")
    for item in stock["inputs"].values():
        check(item)
    acquisition = read(out / "result.json")
    paths = [p for p in out.rglob("*") if p.is_file() and p != destination]
    paths += [
        baseline_path,
        repo / "provenance/recovery/ugi_guidance_evidence_acquisition_v3.json",
        repo / "pyproject.toml",
        repo / "uv.lock",
        repo / "Makefile",
    ]
    paths += [repo / item["archive"]["path"] for item in acquisition["archived_sources"]]
    result = {
        "schema_version": "forge.guidance_evidence_acquisition_closeout.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial_evidence_qualified_guidance_blocked",
        "files": [pin_record(p, repo) for p in sorted(set(paths))],
        "historical_recovery": {
            k: acquisition["historical_inventory"][k] for k in ("recovered", "unavailable")
        },
        "fresh_evidence": {
            "terminal_identities_verified": len(acquisition["identity_checks"]),
            "qualified_terminals": qualification["qualified_terminal_count"],
            "qualified_routes": 0,
            "additional_positive_supplier_observations": stock["positive_stock_responses"],
            "additional_observation_qualified": False,
            "stock_lookup_count": stock["lookup_count"],
            "backorder_or_estimate_responses": stock["backorder_or_estimate_responses"],
            "reference_zip_not_user_address": "10001",
        },
        "remaining_work": [
            "Authenticate original missing source records/code to replay the historical source, or finish a separately versioned source qualification; do not change old pins.",
            "Qualify the exact protected-propargyl-alcohol observation from the public supplier response.",
            "Acquire qualifying availability for heptadecanal, bromotridecane and stearolic acid; checked current listings return backorders or supplier estimates.",
            "Complete primary-article visual verification of the C18 oxidation procedure; the accessible new transcription omits scheme images.",
            "Rebuild and separately qualify the new cumulative source, exact-route outputs and synthesis-value controls before passive candidate route contrast.",
        ],
        "validation": {
            "vendor_assets_verified": 30,
            "focused": focused,
            "focused_failures": sorted(focused_failures),
            "full": current,
            "prior_full": baseline,
            "added_failures": sorted(failures - old_failures),
            "removed_failures": sorted(old_failures - failures),
            "terminal_qualification_checks": qualification["checks"],
            "global_phase1_definition_of_done_met": False,
        },
        "policy": {
            "guidance_run_ready": False,
            "historical_pins_changed": False,
            "gates_relaxed": False,
            "generation_calls": 0,
            "training_calls": 0,
            "candidate_selection": False,
            "purchases_or_supplier_contacts": 0,
            "paid_remote_jobs": 0,
        },
    }
    write_json(destination, result)
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
