"""Retain verification commands, return codes and input snapshots for the precursor audit."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from forge.core.hashing import sha256_file  # noqa: E402


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    production = [
        "forge/assembly/condensation_event.py",
        "forge/corpus/compose_lipid_condensation_event.py",
        "experiments/phase1/multireaction/compose_lipid_condensation_event.py",
        "tests/test_condensation_event.py",
        "tests/test_compose_lipid_condensation_event.py",
        "results/phase1/compose_lipid_v8_ugi4_source_v1/build_registry.py",
        "results/phase1/compose_lipid_v8_ugi4_source_v1/check_prior_dictionary.py",
    ]
    previous = json.loads(
        (ROOT / "results/phase1/compose_lipid_v8_precursor_validation_v1/checks.json").read_text()
    )
    focused = [x for x in previous["focused-tests"]["command"] if x.startswith("tests/")]
    focused.extend(production[3:5])
    snapshot_paths = {
        *(
            p.relative_to(ROOT).as_posix()
            for base in ("forge", "experiments", "tests", "tools", "cli", "paper")
            for p in (ROOT / base).rglob("*.py")
        ),
        "Makefile",
        "pyproject.toml",
        *production,
        "configs/multireaction/compose_lipid_v8_precursor_audit_v2.json",
        "configs/multireaction/compose_lipid_v8_ugi4_program_v1.json",
        "data/vendor/qualified_ugi4_source_program_v1.json",
    }
    snapshot = {path: str(sha256_file(ROOT / path)) for path in sorted(snapshot_paths)}
    write("source-snapshot.json", snapshot)
    env = dict(os.environ)
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env["PATH"]
    # Existing environment only: tests must not need a network install or an external cache write.
    commands = {
        "black": [sys.executable, "-m", "black", "--check", *production],
        "ruff": [sys.executable, "-m", "ruff", "check", *production],
        "focused-tests": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *focused,
            "--tb=short",
            f"--junitxml={OUTPUT / 'focused-tests.xml'}",
        ],
        "vendor-verify": ["make", "verify"],
        "program-verification": [
            sys.executable,
            "-m",
            "experiments.phase1.multireaction.compose_lipid_condensation_event",
            "verify",
            "--result",
            "results/phase1/compose_lipid_v8_ugi4_program_v1/result.json",
        ],
        "audit-verification": [
            sys.executable,
            "-m",
            "experiments.phase1.multireaction.compose_lipid_precursor_audit",
            "--verify",
            "results/phase1/compose_lipid_v8_precursor_audit_v2/result.json",
        ],
        "full-tests": ["make", "test", "UV_RUN="],
    }
    checks = {}
    for label, command in commands.items():
        print(f"Starting {label}", flush=True)
        test_env = dict(env)
        if label == "full-tests":
            test_env["PYTEST_ADDOPTS"] = f"--tb=short --junitxml={OUTPUT / 'full-tests.xml'}"
        started = time.monotonic()
        with (OUTPUT / f"{label}.log").open("w") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=test_env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        checks[label] = {
            "command": command,
            "exit_code": completed.returncode,
            "duration_seconds": time.monotonic() - started,
        }
        write("checks.json", checks)
        print(f"Finished {label}: {completed.returncode}", flush=True)
        if completed.returncode and label != "full-tests":
            raise SystemExit(f"Required check failed: {label}")
    write(
        "snapshot-comparison.json",
        {path: str(sha256_file(ROOT / path)) == digest for path, digest in snapshot.items()},
    )


if __name__ == "__main__":
    main()
