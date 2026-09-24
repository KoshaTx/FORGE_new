"""Record the bounded source-artifact search without substituting missing source data.

Network responses are retained inputs. Replaying this script rechecks those responses and
current local paths; it does not perform a fresh network search or qualify training data.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forge.core.hashing import sha256_file  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def main():
    prior = ROOT / "results/phase1/compose_lipid_v8_source_recovery_v1"
    request = read(prior / "full-universe-source-request.json")
    branches = read(OUTPUT / "branches.json")
    artifacts = read(OUTPUT / "actions-artifacts.json")
    releases = read(OUTPUT / "releases.json")
    issues = read(OUTPUT / "issues.json")
    response = (OUTPUT / "drive-response.html").read_text()
    match = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'((?:\\.|[^'\\])*)';", response)
    if match is None:
        raise ValueError("shared Drive response lacks a readable folder inventory")
    payload = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), match[1])
    entries = json.loads(payload)[0]
    files = [
        {
            "id": row[0],
            "name": row[2],
            "mime_type": row[3],
            "size_bytes": row[13],
            "modified_ms": row[10],
        }
        for row in entries
    ]
    old_files = read(prior / "drive-listing.json")["files"]
    same_listing = sorted(files, key=lambda x: x["id"]) == sorted(old_files, key=lambda x: x["id"])
    source_commit = next(
        row["commit"]["sha"]
        for row in branches
        if row["name"] == "dataset/source-corrected-corpus-v3"
    )
    # A changed inventory is a reason to inspect the new evidence, never to emit a stale absence.
    if (
        not same_listing
        or source_commit != request["upstream_commit"]
        or artifacts["total_count"] != 0
        or releases
        or issues
    ):
        raise ValueError("source inventory changed; inspect new artifacts before recording absence")
    local_repo = Path("/Users/rahulmaganti/Kosha/compose_lipid")
    local = [
        {
            "path": str(local_repo / item["directory"] / name),
            "exists": (local_repo / item["directory"] / name).is_file(),
        }
        for item in request["missing_selection_artifacts"]
        for name in item["files"]
    ]
    if any(row["exists"] for row in local):
        raise ValueError("requested local artifact appeared; authenticate it before proceeding")
    inputs = [
        prior / "full-universe-source-request.json",
        prior / "drive-listing.json",
        OUTPUT / "branches.json",
        OUTPUT / "actions-artifacts.json",
        OUTPUT / "releases.json",
        OUTPUT / "issues.json",
        OUTPUT / "drive-response.html",
        Path(__file__),
    ]
    result = {
        "schema_version": "forge.compose_lipid_source_availability.v1",
        "status": "blocked_missing_original_component_and_construction_artifacts",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 0,
        "random_sampling_used": False,
        "source_repository": "KoshaTx/compose_lipid",
        "source_branch_commit": source_commit,
        "source_branch_unchanged": True,
        "workflow_artifacts": artifacts["total_count"],
        "releases": len(releases),
        "issues_or_pull_requests": len(issues),
        "shared_drive_url": read(prior / "drive-listing.json")["url"],
        "shared_drive_files": files,
        "shared_drive_listing_unchanged": same_listing,
        "requested_local_files": local,
        "search_limits": [
            "Only accessible source-repository and shared-folder locations were searched.",
            "Downloads could not be searched: operating-system access was denied.",
            "Absence here does not establish absence from the source author's workstation.",
        ],
        "input_location_requested_from_user": True,
        "substitute_data_created": False,
        "source_splits_changed": False,
        "training_calls": 0,
        "training_rows_admitted": 0,
        "training_ready": False,
        "production_code_changed": False,
        "tests_rerun": False,
        "last_completed_validation": pin(
            ROOT / "results/phase1/compose_lipid_v8_universe_validation_v1/result.json"
        ),
        "inputs": {str(path.relative_to(ROOT)): pin(path) for path in inputs},
    }
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
