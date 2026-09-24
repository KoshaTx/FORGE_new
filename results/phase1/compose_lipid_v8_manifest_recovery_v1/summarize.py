"""Summarize exact historical-manifest recovery without claiming training readiness."""

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
BASELINE = "results/phase1/compose_lipid_v8_reductive_validation_v1/result.json"


def pin(path):
    path = (ROOT / path).resolve()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def report(path):
    raw = path.read_bytes()
    tree = ET.fromstring(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    cases = list(tree.iter("testcase"))
    counts = {
        "tests": len(cases),
        "failures": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped_or_xfail": sum(c.find("skipped") is not None for c in cases),
    }
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped_or_xfail"]
    )
    return {
        "counts": counts,
        **{
            name: {
                f"{c.get('classname')}::{c.get('name')}": c.find(kind).get("message")
                for c in cases
                if c.find(kind) is not None
            }
            for name, kind in (("failures", "failure"), ("errors", "error"))
        },
    }


def main():
    previous = json.loads((ROOT / BASELINE).read_text())
    execution = json.loads((OUTPUT / "execution.json").read_text())
    replay = json.loads((OUTPUT / "replay.json").read_text())
    snapshot = json.loads((OUTPUT / "source-snapshot.json").read_text())
    assert all(pin(path)["sha256"] == sha for path, sha in snapshot.items())
    assert all(replay["checks"].values())
    for label, item in execution.items():
        if label != "full-tests":
            assert item["exit_code"] == 0
    assert "full-tests" in execution
    focused, full = report(OUTPUT / "focused-tests.xml"), report(OUTPUT / "full-tests.xml.gz")
    assert focused["counts"]["failures"] == focused["counts"]["errors"] == 0
    old_bad = set(previous["full"]["failures"]) | set(previous["full"]["errors"])
    bad = set(full["failures"]) | set(full["errors"])
    inputs = {BASELINE, Path(__file__).relative_to(ROOT).as_posix()}
    inputs.update(snapshot)
    inputs.update(replay["inputs"])
    artifacts = {
        p.name: pin(p)
        for p in sorted(OUTPUT.iterdir())
        if p.is_file()
        and p.name not in {"result.json", "summarize.py", "full-tests.log", "full-tests.xml"}
    }
    result = {
        "schema_version": "forge.compose_lipid_manifest_recovery_validation.v1",
        "status": "exact_development_manifest_recovered_global_tests_still_fail",
        "training_ready": False,
        "phase1_definition_of_done_met": False,
        "training_calls": 0,
        "proposal_calls": 0,
        "inputs": {path: pin(path) for path in sorted(inputs)},
        "artifacts": artifacts,
        "execution": execution,
        "replay_comparison": replay["checks"],
        "snapshot_unchanged": True,
        "focused": focused,
        "full": full,
        "failure_comparison": {
            "baseline": pin(BASELINE),
            "new_failure_or_error_ids": sorted(bad - old_bad),
            "resolved_failure_or_error_ids": sorted(old_bad - bad),
        },
        "all_family_scientific_status": pin(BASELINE),
        "vendor_assets_verified": sum(
            line.strip().startswith("ok ")
            for line in (OUTPUT / "vendor-verify.log").read_text().splitlines()
        ),
        "notes": [
            "Only historical implementation source may resolve through reviewed moves or an exact archive path/digest binding.",
            "Data pins, forbidden-input checks, frozen benchmark outputs and original configurations remain unchanged.",
            "Archived bytes are evidence only. The current executor and source-authentication modules have separate recorded digests.",
            "Every frozen scientific payload and every original result field reproduces exactly; fresh results add explicit execution provenance.",
            "This repairs manifest reconstruction and setup tests. It does not execute any proposal, benchmark, guidance or training job.",
            "The full suite remains failing and the 23-family training-readiness goal remains active.",
        ],
    }
    assert result["vendor_assets_verified"] == 30
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        focused["counts"], full["counts"], result["failure_comparison"], pin(OUTPUT / "result.json")
    )


if __name__ == "__main__":
    main()
