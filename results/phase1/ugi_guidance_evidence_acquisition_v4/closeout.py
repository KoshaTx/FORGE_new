"""Collect a reproducible, bounded acquisition report without refreshing old evidence."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from forge.core.hashing import pin_record
from forge.core.io import write_json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = OUT.parent / "ugi_guidance_evidence_acquisition_v3"
ARCHIVE = ROOT / "provenance/recovery/ugi_guidance_evidence_acquisition_v4.json"
DESTINATION = OUT / "result.json"


def read(path):
    return json.loads(path.read_text())


def authenticate(pin):
    path = ROOT / pin["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
        raise ValueError(f"Input changed: {path}")


def build(assessment_time):
    qualification = read(OUT / "terminal_qualification.json")
    previous = read(PREVIOUS / "closeout.json")
    cyclo = read(PREVIOUS / "cyclohexylamine_terminal_qualification.json")
    observations = qualification["payload"]["observations"]
    checks = qualification["payload"]["checks"]
    if not all(v is True for group in checks.values() for v in group.values()):
        raise ValueError("Additional terminal check failed")
    if not all(v is True for v in cyclo["checks"].values()):
        raise ValueError("Previous cyclohexylamine check failed")
    for item in [*observations, cyclo["snapshot"]]:
        if not (
            datetime.fromisoformat(item["accessed_utc"])
            <= datetime.fromisoformat(assessment_time)
            < datetime.fromisoformat(item["expires_utc"])
        ):
            raise ValueError("Observation was not current at the report assessment time")
    if len({item["cas"] for item in observations}) != len(observations):
        raise ValueError("Duplicate terminal observation")
    get_receipts = sorted(OUT.glob("batch_*/receipt.json"))
    gets = [row for path in get_receipts for row in read(path)["records"]]
    posts = read(OUT / "stock_responses/receipt.json")["records"]
    counts = Counter()
    for case in ET.parse(OUT / "focused-tests.xml").getroot().iter("testcase"):
        counts["tests"] += 1
        kind = next(
            (k for k in ("failure", "error", "skipped") if case.find(k) is not None),
            "passed",
        )
        counts[kind] += 1
    verify_lines = (OUT / "verify.log").read_text().splitlines()
    vendor_count = sum(line.startswith("  ok ") for line in verify_lines)
    if f"all {vendor_count} present vendored assets verified" not in verify_lines:
        raise ValueError("Vendor verification did not complete")
    archive = read(ARCHIVE)
    paths = {
        path
        for path in OUT.rglob("*")
        if path.is_file() and path != DESTINATION and "__pycache__" not in path.parts
    }
    paths.update(
        {
            ARCHIVE,
            PREVIOUS / "closeout.json",
            PREVIOUS / "cyclohexylamine_terminal_qualification.json",
            PREVIOUS / "route_source_recheck.json",
            ROOT / "results/phase1/ugi_guidance_artifact_recovery_v2/result.json",
            ROOT / "results/phase1/ugi_guidance_artifact_recovery_v2/remaining_inputs.json",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            ROOT / "Makefile",
            ROOT / "tests/test_pinned_sources_are_tracked.py",
            ROOT / "tests/test_current_sampler_preflight.py",
        }
    )
    source_pins = [*qualification["inputs"].values(), *cyclo["inputs"].values()]
    source_pins.append(cyclo["artifact"])
    for pin in source_pins:
        authenticate(pin)
        paths.add(ROOT / pin["path"])
    paths.update(ROOT / item["archive"]["path"] for item in archive["archived_sources"])
    return {
        "schema_version": "forge.guidance_evidence_acquisition_closeout.v2",
        "created_at_utc": assessment_time,
        "status": "four_target_terminal_observations_qualified_guidance_blocked",
        "files": [pin_record(path, ROOT) for path in sorted(paths)],
        "historical_recovery": {
            **previous["historical_recovery"],
            "new_exact_historical_recoveries_in_this_pass": 0,
            "inventory_basis": "Prior authenticated v2 inventory; no old paths restored this pass",
        },
        "acquisition": {
            "public_get_requests": len(gets),
            "public_guest_availability_post_requests": len(posts),
            "http_status_counts": dict(Counter(str(row["http_status"]) for row in gets + posts)),
            "archive_asset_count": len(archive["archived_sources"]),
            "archive_compressed_bytes": sum(
                row["archive"]["bytes"] for row in archive["archived_sources"]
            ),
            "random_sampling_used": False,
        },
        "fresh_evidence": {
            "identified_target_materials": previous["fresh_evidence"][
                "terminal_identities_verified"
            ],
            "additional_qualified_observations": len(observations),
            "total_qualified_target_material_observations": len(observations)
            + cyclo["qualified_terminal_count"],
            "additional_observations": observations,
            "previous_cyclohexylamine_snapshot": cyclo["snapshot"],
            "qualification_scope": "Exact terminal observations only; no cumulative source admission",
            "complete_routes_reassessed": 0,
            "main_article_visually_authenticated": False,
            "new_primary_route_lead": read(OUT / "review.json")["new_primary_route_lead"],
        },
        "remaining_work": [
            "Acquire qualifying current US availability evidence for exact stearolic acid; checked sources do not establish it.",
            "Authenticate the original Oppolzer main article's relevant procedure and schemes; the SI and transcription do not complete that check.",
            "Build and qualify a separately versioned cumulative source, exact-route outputs, synthesis-value controls and guidance preflight before any nonzero guidance.",
            "Exact historical replay alternatively requires the original unavailable sources/code; fresh observations cannot replace those bytes.",
        ],
        "validation": {
            "vendor_assets_verified": vendor_count,
            "focused_current_run": dict(counts),
            "terminal_checks": checks,
            "terminal_check_count": sum(len(group) for group in checks.values()),
            "prior_v3_full_suite_not_rerun_this_pass": previous["validation"]["full"],
            "prior_frozen_source_failure": read(PREVIOUS / "route_source_recheck.json")["errors"],
            "global_phase1_definition_of_done_met": False,
            "verification_semantics": "Verifies bytes and the recorded assessment; does not renew availability",
        },
        "policy": {
            "guidance_run_ready": False,
            "historical_pins_changed": False,
            "gates_relaxed": False,
            "generation_calls": 0,
            "training_calls": 0,
            "guidance_calls": 0,
            "paid_remote_jobs": 0,
            "supplier_messages": 0,
            "purchases": 0,
            "candidate_selection": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        existing = read(DESTINATION)
        for pin in existing["files"]:
            authenticate(pin)
        if build(existing["created_at_utc"]) != existing:
            raise ValueError("Acquisition summary changed on replay")
        print("Verified source pins and reproduced the recorded acquisition summary")
        return
    if DESTINATION.exists():
        raise ValueError("Preserve the existing acquisition result")
    result = build(datetime.now(timezone.utc).isoformat())
    write_json(DESTINATION, result)
    print(json.dumps({k: result[k] for k in ("status", "acquisition", "validation")}, indent=2))


if __name__ == "__main__":
    main()
