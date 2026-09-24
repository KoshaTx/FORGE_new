"""Validate current COMPOSE training code without historical experiment reproduction."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_training_data import (
    TRAINING_TEST_FILES,
    TRAINING_VALIDATION_SCHEMA,
)


def validate_training(repo: Path, output: Path) -> dict:
    """Write a scoped engineering receipt; this never admits a dataset or starts training."""
    repo, output = repo.resolve(), output.resolve()
    output.relative_to(repo)
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        p
        for directory in ("forge", "experiments/_runtime", "cli", "tools/forge_data")
        for p in (repo / directory).rglob("*.py")
    }
    sources |= {
        repo / name
        for name in TRAINING_TEST_FILES
        + (
            "experiments/phase1/multireaction/compose_lipid_run.py",
            "experiments/phase1/multireaction/compose_lipid_training.py",
            "experiments/phase1/multireaction/compose_lipid_gpu_preflight.py",
            "experiments/phase1/multireaction/compose_lipid_package.py",
            "experiments/phase1/multireaction/compose_lipid_validation.py",
            "tests/test_source_instance_coordinates.py",
            "tests/conftest.py",
            "tests/unreproducible_pins.py",
            "pyproject.toml",
            "uv.lock",
            "data/vendor/MANIFEST.json",
            "results/phase1/compose_lipid_unified_preparation_v1/result.json",
            "results/phase1/compose_lipid_unified_preparation_v1/verification.json",
        )
    }
    snapshot = {p.relative_to(repo).as_posix(): str(sha256_file(p)) for p in sorted(sources)}
    dump(output / "source-snapshot.json", snapshot)
    env = {**os.environ, "PYTHONPATH": str(repo) + os.pathsep + str(repo / "tools")}
    # A filtered or plugin-injected pytest invocation cannot masquerade as this fixed suite.
    env.pop("PYTEST_ADDOPTS", None)
    commands = {
        "vendor": [sys.executable, "-m", "cli", "data", "verify"],
        "tests": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *TRAINING_TEST_FILES,
            "--junitxml=" + str(output / "tests.xml"),
        ],
    }
    statuses, elapsed = {}, {}
    for name, command in commands.items():
        start = time.monotonic()
        with (output / (name + ".log")).open("w") as log:
            statuses[name] = subprocess.run(
                command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            ).returncode
        elapsed[name] = time.monotonic() - start
    suites = list(ET.parse(output / "tests.xml").getroot().iter("testsuite"))
    counts = {
        key: sum(int(s.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    counts["passed"] = counts.pop("tests") - sum(
        counts[k] for k in ("failures", "errors", "skipped")
    )
    classes = {case.get("classname") for s in suites for case in s.iter("testcase")}
    complete = all(name[:-3].replace("/", ".") in classes for name in TRAINING_TEST_FILES)
    unchanged = all(str(sha256_file(repo / name)) == digest for name, digest in snapshot.items())
    passed = (
        all(code == 0 for code in statuses.values())
        and unchanged
        and complete
        and counts["passed"] > 0
        and all(counts[k] == 0 for k in ("failures", "errors", "skipped"))
    )
    result = {
        "schema_version": TRAINING_VALIDATION_SCHEMA,
        "seed": 0,
        "training_checks_passed": passed,
        "test_files": list(TRAINING_TEST_FILES),
        "all_test_files_executed": complete,
        "tests": counts,
        "vendor_verify_exit_code": statuses["vendor"],
        "pytest_exit_code": statuses["tests"],
        "elapsed_seconds": elapsed,
        "source_snapshot_unchanged": unchanged,
        "inputs": {
            name: pin(repo, output / name)
            for name in ("source-snapshot.json", "tests.xml", "tests.log", "vendor.log")
        },
        "commands": commands,
        "scope": "current training code, current prepared-cache controls and synthetic optimizer/restart fixtures",
        "historical_experiment_reproduction_required": False,
        "full_suite_rerun": False,
        "training_admitted": False,
        "real_corpus_optimizer_steps": 0,
        "gpu_execution": False,
        "paid_compute_calls": 0,
    }
    dump(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    result = validate_training(repo, args.output)
    print(f"Current training checks: {result['tests']}; passed={result['training_checks_passed']}")
    if not result["training_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
