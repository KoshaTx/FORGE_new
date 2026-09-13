"""Preserve completed validation logs and their exact JUnit outcome counts."""

import argparse
import gzip
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import pin_record, sha256_file
from forge.core.io import atomic_write, write_json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_directory", type=Path)
    args = parser.parse_args()
    output = OUT / "validation.json"
    if output.exists():
        raise ValueError("Preserve existing validation")
    reports = {}
    archives = []
    for name in ("full-tests", "focused-tests"):
        path = args.log_directory / f"{name}.xml"
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        counts = {
            key: sum(int(s.attrib[key]) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        counts["passed"] = counts["tests"] - sum(
            counts[k] for k in ("failures", "errors", "skipped")
        )
        reports[name] = {
            "counts": counts,
            "duration_seconds": sum(float(s.attrib["time"]) for s in suites),
        }
    for name in (
        "full-tests.xml",
        "full-tests.log",
        "focused-tests.xml",
        "focused-tests.log",
        "verify.log",
    ):
        source = args.log_directory / name
        raw = source.read_bytes()
        packed = gzip.compress(raw, mtime=0)
        destination = OUT / "validation_logs" / f"{name}.gz"
        if destination.exists() and destination.read_bytes() != packed:
            raise ValueError(f"Preserve changed log archive: {destination}")
        if not destination.exists():
            atomic_write(destination, packed)
        archives.append(
            {
                "original": {"filename": name, "sha256": sha256_file(source), "bytes": len(raw)},
                "archive": pin_record(destination, ROOT),
            }
        )
    result = json.loads((OUT / "result.json").read_text())
    write_json(
        output,
        {
            "schema_version": "forge.guidance_acquisition_validation.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": [
                pin_record(p, ROOT)
                for p in (
                    OUT / "result.json",
                    OUT / "qualify.py",
                    OUT / "test_qualify.py",
                    Path(__file__),
                    OUT / "route_source_check.json",
                    ROOT / "Makefile",
                    ROOT / "pyproject.toml",
                    ROOT / "uv.lock",
                )
            ],
            "log_archives": archives,
            "test_reports": reports,
            "vendor_verification_log_contains_success": "all 30 present vendored assets verified"
            in (args.log_directory / "verify.log").read_text(),
            "acquisition_result_status": result["status"],
            "global_phase1_definition_of_done_met": not (
                reports["full-tests"]["counts"]["failures"]
                or reports["full-tests"]["counts"]["errors"]
            ),
            "notes": [
                "New acquisition tests live under results/ and are explicitly included in the focused command, not the repository-default testpaths.",
                "Skipped counts include the JUnit representation of xfail. No failing repository checks were changed or skipped in this pass.",
                "Acquisition replay and touched-source Black/Ruff passed separately; exact command list is in validation_commands.json.",
                "Archive extension changes storage metadata only; original result and extension receipt are retained.",
            ],
        },
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
