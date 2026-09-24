"""Run the checksum command and full join audit once the resumable download completes."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    artifact = (
        ROOT / "data/source_cache/compose_lipid_supplement_2026-09-19/construction_records.jsonl.gz"
    )
    while not artifact.exists():
        time.sleep(10)
    checks = {}
    commands = {
        "package-check": [sys.executable, str(OUT / "check_package.py")],
        "full-join": [
            sys.executable,
            "-m",
            "experiments.phase1.multireaction.compose_lipid_supplement",
            "--config",
            "configs/multireaction/compose_lipid_supplement_intake_v1.json",
            "--output",
            str(OUT),
        ],
    }
    for name, command in commands.items():
        print(f"Starting {name}", flush=True)
        start = time.monotonic()
        with (OUT / f"{name}.log").open("w") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        checks[name] = {
            "command": command,
            "exit_code": completed.returncode,
            "duration_seconds": time.monotonic() - start,
        }
        (OUT / "intake_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(f"Finished {name}: {completed.returncode}", flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
