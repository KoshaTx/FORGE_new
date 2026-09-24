"""Persist validation commands, source pins and comparison with the preceding test receipt."""

import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/bin/python"


def pin(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def snapshot():
    paths = []
    for folder in ("forge", "cli", "experiments", "tools", "tests", "configs"):
        paths.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".toml"}
        )
    paths.extend(ROOT / name for name in ("pyproject.toml", "uv.lock", "Makefile"))
    return {str(p.relative_to(ROOT)): pin(p)["sha256"] for p in sorted(paths)}


def junit(path):
    cases = list(ET.parse(path).iter("testcase"))
    issues = {"failures": {}, "errors": {}}
    skipped = 0
    for case in cases:
        name = case.get("classname", "") + "::" + case.get("name", "")
        for tag, group in (("failure", "failures"), ("error", "errors")):
            node = case.find(tag)
            if node is not None:
                issues[group][name] = node.get("message", "")
                break
        else:
            skipped += case.find("skipped") is not None
    counts = {k: len(v) for k, v in issues.items()}
    counts.update(
        total=len(cases),
        skipped_or_xfailed=skipped,
        passed=len(cases) - skipped - sum(counts.values()),
    )
    return counts, issues


def main():
    before = snapshot()
    write("source-snapshot.json", before)
    env = dict(os.environ)
    env["PATH"] = str(PYTHON.parent) + os.pathsep + env["PATH"]
    previous = ROOT / "results/phase1/compose_lipid_training_goal_validation_v1"
    commands = json.loads((previous / "checks.json").read_text())
    focused = commands["focused-tests"]["command"]
    focused = [arg for arg in focused if not arg.startswith("--junitxml=")]
    focused += [
        "tests/test_compose_lipid_miao_cyclic.py",
        "tests/test_compose_lipid_staged_replay.py",
        "tests/test_compose_lipid_staged_readiness.py",
    ]
    focused += ["--junitxml=" + str(OUT / "focused-tests.xml")]
    commands = {
        "focused-tests": focused,
        "vendor-verify": ["make", "verify"],
        "full-tests": ["make", "test", "UV_RUN="],
    }
    checks = {}
    for name, command in commands.items():
        started = time.monotonic()
        local_env = dict(env)
        if name == "full-tests":
            local_env["PYTEST_ADDOPTS"] = "--tb=short --junitxml=" + str(OUT / "full-tests.xml")
        print("Starting " + name, flush=True)
        with (OUT / (name + ".log")).open("w") as log:
            code = subprocess.run(
                command, cwd=ROOT, env=local_env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        checks[name] = {
            "command": command,
            "exit_code": code,
            "duration_seconds": time.monotonic() - started,
        }
        write("checks.json", checks)
        print(f"Finished {name}: {code}", flush=True)
    after = snapshot()
    changes = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    write(
        "snapshot-comparison.json",
        {"source_snapshot_unchanged": not changes, "changed_paths": changes},
    )
    full, issues = junit(OUT / "full-tests.xml")
    focused_counts, _ = junit(OUT / "focused-tests.xml")
    _, previous_issues = junit(previous / "full-tests.xml")
    report = {
        "schema_version": "forge.compose_lipid_all_family_validation.v1",
        "seed": 0,
        "implementation": pin(Path(__file__)),
        "inputs": {
            p.name: pin(p)
            for p in [
                OUT / "checks.json",
                OUT / "source-snapshot.json",
                OUT / "snapshot-comparison.json",
                OUT / "focused-tests.xml",
                OUT / "full-tests.xml",
            ]
        },
        "previous_full_tests": pin(previous / "full-tests.xml"),
        "focused": focused_counts,
        "full": full,
        "new_failures": sorted(set(issues["failures"]) - set(previous_issues["failures"])),
        "new_errors": sorted(set(issues["errors"]) - set(previous_issues["errors"])),
        "resolved_failures": sorted(set(previous_issues["failures"]) - set(issues["failures"])),
        "resolved_errors": sorted(set(previous_issues["errors"]) - set(issues["errors"])),
        "source_snapshot_unchanged": not changes,
        "vendor_verify_exit_code": checks["vendor-verify"]["exit_code"],
        "phase1_definition_of_done_met": all(c["exit_code"] == 0 for c in checks.values())
        and not changes,
        "training_admitted": False,
    }
    write(
        "test-failures.json",
        {"inputs": {"full-tests.xml": pin(OUT / "full-tests.xml")}, "issues": issues},
    )
    write("validation_report.json", report)


if __name__ == "__main__":
    main()
