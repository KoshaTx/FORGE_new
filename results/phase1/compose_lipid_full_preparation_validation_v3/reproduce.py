"""Require identical consolidated reports and readiness bytes on an independent rerun."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from forge.core.hashing import sha256_file  # noqa: E402

FILES = ["reconstruction_report.json", "readiness.jsonl.gz", "constitutional_deduplication.json"]
before = {name: str(sha256_file(OUT / name)) for name in FILES}
command = [str(ROOT / ".venv/bin/python"), str(OUT / "report.py")]
started = time.monotonic()
with (OUT / "reproduce.log").open("w") as log:
    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
after = {name: str(sha256_file(OUT / name)) for name in FILES}
result = {
    "schema_version": "forge.compose_lipid_full_preparation_reproduction.v1", "seed": 0,
    "command": command, "exit_code": completed.returncode,
    "duration_seconds": time.monotonic() - started,
    "implementation": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": str(sha256_file(Path(__file__)))},
    "inputs": {"report": {"path": str((OUT / "report.py").relative_to(ROOT)), "sha256": str(sha256_file(OUT / "report.py"))}},
    "before_sha256": before, "after_sha256": after,
    "byte_identical": before == after, "training_calls": 0,
}
(OUT / "reproduction.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
if completed.returncode != 0 or before != after:
    raise SystemExit("Consolidated report or readiness reproduction failed")
print("Consolidated report and full readiness ledger reproduce byte-for-byte")
