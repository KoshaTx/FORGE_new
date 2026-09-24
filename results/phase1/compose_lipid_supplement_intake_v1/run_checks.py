"""Retain deterministic validation commands, source pins and failure logs."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forge.core.hashing import sha256_file  # noqa: E402


def main():
    prior = json.loads(
        (ROOT / "results/phase1/compose_lipid_v8_universe_validation_v1/checks.json").read_text()
    )
    focused = [item for item in prior["focused-tests"]["command"] if item.startswith("tests/")]
    focused.append("tests/test_compose_lipid_supplement.py")
    snapshot = {
        str(path.relative_to(ROOT)): str(sha256_file(path))
        for base in ("forge", "experiments", "tests", "tools", "cli", "paper")
        for path in sorted((ROOT / base).rglob("*.py"))
    }
    (OUT / "source-snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    env = dict(os.environ)
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env["PATH"]
    commands = {
        "focused-tests": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *focused,
            "--tb=short",
            f"--junitxml={OUT / 'focused-tests.xml'}",
        ],
        "vendor-verify": ["make", "verify"],
        "full-tests": ["make", "test", "UV_RUN="],
    }
    checks = {}
    for label, command in commands.items():
        test_env = dict(env)
        if label == "full-tests":
            test_env["PYTEST_ADDOPTS"] = f"--tb=short --junitxml={OUT / 'full-tests.xml'}"
        print(f"Starting {label}", flush=True)
        started = time.monotonic()
        with (OUT / f"{label}.log").open("w") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=test_env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        checks[label] = {
            "command": command,
            "exit_code": completed.returncode,
            "duration_seconds": time.monotonic() - started,
        }
        (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(f"Finished {label}: {completed.returncode}", flush=True)
        if completed.returncode and label != "full-tests":
            raise SystemExit(f"Required check failed: {label}")
    unchanged = {name: str(sha256_file(ROOT / name)) == digest for name, digest in snapshot.items()}
    (OUT / "snapshot-comparison.json").write_text(
        json.dumps(unchanged, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
